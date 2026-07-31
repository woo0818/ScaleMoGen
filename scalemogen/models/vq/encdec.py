"""ScaleMoGen VQ encoder and decoder blocks.

Code provenance: adapted from the SALAD skeleton VAE encoder/decoder.
Source repository: https://github.com/seokhyeonhong/salad
"""

import torch
import torch.nn as nn

from scalemogen.models.skeleton.conv import ResSTConv, get_activation
from scalemogen.models.skeleton.pool import STPool, STUnpool
from utils.paramUtil import kit_adj_list, t2m_adj_list, snapmogen_adj_list
from utils.skeleton import adj_list_to_edges


class MotionEncoder(nn.Module):
    def __init__(self, cfg):
        super(MotionEncoder, self).__init__()

        self.data_name = cfg.data.name
        self.pose_dim = cfg.data.dim_pose
        self.joints_num = cfg.data.joint_num #(self.pose_dim + 1) // 12
        self.latent_dim = cfg.model.latent_dim
        self.contact_joints = cfg.data.contact_joints

        self.layers = nn.ModuleList()
        if self.data_name == 'snapmogen':
            for i in range(self.joints_num):
                if i == 0:
                    input_dim = 16
                elif i in self.contact_joints:
                    input_dim = 13
                else:
                    input_dim = 12
                self.layers.append(nn.Sequential(
                    nn.Linear(input_dim, self.latent_dim),
                    get_activation(cfg.model.activation),
                    nn.Linear(self.latent_dim, self.latent_dim),
                ))
        else:
            for i in range(self.joints_num):
                if i == 0:
                    input_dim = 7
                elif i in self.contact_joints:
                    input_dim = 13
                else:
                    input_dim = 12
                self.layers.append(nn.Sequential(
                    nn.Linear(input_dim, self.latent_dim),
                    get_activation(cfg.model.activation),
                    nn.Linear(self.latent_dim, self.latent_dim),
                ))

    def forward(self, x):
        """
        x: [bs, nframes, pose_dim]
        
        nfeats = 12J + 1
            - root_rot_velocity (B, seq_len, 1)
            - root_linear_velocity (B, seq_len, 2)
            - root_y (B, seq_len, 1)
            - rot_data (B, seq_len, (joint_num - 1)*6)
            - ric_data (B, seq_len, (joint_num - 1)*3)
            - local_velocity (B, seq_len, joint_num*3)
            - foot contact (B, seq_len, 4)
        """
        B, T, D = x.size()

        if self.data_name == 'snapmogen':
            # split
            root, rot, ric, vel, contact = torch.split(x, [4, 6 * (self.joints_num), 3 * (self.joints_num), 3 * self.joints_num, 4], dim=-1)
            rot = rot.reshape(B, T, self.joints_num, 6)
            ric = ric.reshape(B, T, self.joints_num, 3)
            vel = vel.reshape(B, T, self.joints_num, 3)

            # joint-wise input
            joints = [] # [B, T, 7]]
            for i in range(0, self.joints_num):
                if i==0:
                    joints.append(torch.cat([root, rot[:, :, i], ric[:, :, i], vel[:, :, i]], dim=-1))
                else:
                    joints.append(torch.cat([rot[:, :, i], ric[:, :, i], vel[:, :, i]], dim=-1))
        else:
            root, ric, rot, vel, contact = torch.split(x, [4, 3 * (self.joints_num - 1), 6 * (self.joints_num - 1), 3 * self.joints_num, 4], dim=-1)
            ric = ric.reshape(B, T, self.joints_num - 1, 3)
            rot = rot.reshape(B, T, self.joints_num - 1, 6)
            vel = vel.reshape(B, T, self.joints_num, 3)

            # joint-wise input
            joints = [torch.cat([root, vel[:, :, 0]], dim=-1)] # [B, T, 7]]
            for i in range(1, self.joints_num):
                joints.append(torch.cat([ric[:, :, i - 1], rot[:, :, i - 1], vel[:, :, i]], dim=-1))
        
        
        for cidx, jidx in enumerate(self.contact_joints):
            joints[jidx] = torch.cat([joints[jidx], contact[:, :, cidx, None]], dim=-1)
        
        # encode
        out = []
        for i in range(self.joints_num):
            out.append(self.layers[i](joints[i]))
        out = torch.stack(out, dim=2)

        return out


class MotionDecoder(nn.Module):
    def __init__(self, cfg):
        super(MotionDecoder, self).__init__()
        
        self.data_name = cfg.data.name
        self.pose_dim = cfg.data.dim_pose
        self.joints_num = cfg.data.joint_num #(self.pose_dim + 1) // 12
        self.latent_dim = cfg.model.latent_dim
        self.contact_joints = cfg.data.contact_joints

        # network components
        self.layers = nn.ModuleList()
        if self.data_name == 'snapmogen':
            for i in range(self.joints_num):
                if i == 0:
                    output_dim = 16
                elif i in self.contact_joints:
                    output_dim = 13
                else:
                    output_dim = 12
                self.layers.append(nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim),
                    get_activation(cfg.model.activation),
                    nn.Linear(self.latent_dim, output_dim),
                ))
        else:
            for i in range(self.joints_num):
                if i == 0:
                    output_dim = 7
                elif i in self.contact_joints:
                    output_dim = 13
                else:
                    output_dim = 12
                self.layers.append(nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim),
                    get_activation(cfg.model.activation),
                    nn.Linear(self.latent_dim, output_dim),
                ))

    def forward(self, x):
        """
        x: [bs, nframes, joints_num, latent_dim]
        """
        B, T, J, D = x.size()
        
        out = []
        for i in range(self.joints_num):
            out.append(self.layers[i](x[:, :, i]))
        
        if self.data_name == 'snapmogen':
            root = out[0][..., :4]
            ric_list, rot_list, vel_list = [], [], []
            for i in range(0, self.joints_num):
                if i==0:
                    rot = out[i][:, :, 4:10]
                    ric = out[i][:, :, 10:13]
                    vel = out[i][:, :, 13:16]
                else:
                    rot = out[i][:, :, :6]
                    ric = out[i][:, :, 6:9]
                    vel = out[i][:, :, 9:12]

                rot_list.append(rot)
                ric_list.append(ric)
                vel_list.append(vel)

            contact = [out[i][:, :, -1] for i in self.contact_joints]

            rot = torch.stack(rot_list, dim=2).reshape(B, T, J * 6)
            ric = torch.stack(ric_list, dim=2).reshape(B, T, J * 3)
            vel = torch.stack(vel_list, dim=2).reshape(B, T, J * 3)

            contact = torch.stack(contact, dim=2).reshape(B, T, len(self.contact_joints))

            motion = torch.cat([
                root, # root
                rot, # rot
                ric, # ric
                vel, # vel
                contact, # contact
            ], dim=-1)

        else:
            root = out[0]
            ric_list, rot_list, vel_list = [], [], []
            for i in range(1, self.joints_num):
                ric = out[i][:, :, :3]
                rot = out[i][:, :, 3:9]
                vel = out[i][:, :, 9:12]

                ric_list.append(ric)
                rot_list.append(rot)
                vel_list.append(vel)

            contact = [out[i][:, :, -1] for i in self.contact_joints]

            ric = torch.stack(ric_list, dim=2).reshape(B, T, (J - 1) * 3)
            rot = torch.stack(rot_list, dim=2).reshape(B, T, (J - 1) * 6)
            vel = torch.stack(vel_list, dim=2).reshape(B, T, (J - 1) * 3)
            contact = torch.stack(contact, dim=2).reshape(B, T, len(self.contact_joints))

            motion = torch.cat([
                root[..., :4], # root
                ric, # ric
                rot, # rot
                torch.cat([root[..., 4:], vel], dim=-1), # vel
                contact, # contact
            ], dim=-1)

        return motion


class STConvEncoder(nn.Module):
    def __init__(self, cfg):
        super(STConvEncoder, self).__init__()

        # adjacency list
        self.adj_list = {
            "t2m": t2m_adj_list,
            "kit": kit_adj_list,
            "snapmogen": snapmogen_adj_list,
            "humanml3d": t2m_adj_list,
        }[cfg.data.name]

        # topology
        self.edge_list = [adj_list_to_edges(self.adj_list)]
        self.mapping_list = []

        # network
        self.layers = nn.ModuleList()
        for i in range(cfg.model.n_layers):
            layers = []
            for _ in range(cfg.model.n_extra_layers):
                layers.append(ResSTConv(
                    self.edge_list[-1],
                    cfg.model.latent_dim,
                    cfg.model.kernel_size,
                    activation=cfg.model.activation,
                    norm=cfg.model.norm,
                    dropout=cfg.model.dropout
                ))
            layers.append(ResSTConv(
                self.edge_list[-1],
                cfg.model.latent_dim,
                cfg.model.kernel_size,
                activation=cfg.model.activation,
                norm=cfg.model.norm,
                dropout=cfg.model.dropout
            ))

            pool = STPool(cfg.data.name, i)
            layers.append(pool)
            self.layers.append(nn.Sequential(*layers))

            self.edge_list.append(pool.new_edges)
            self.mapping_list.append(pool.skeleton_mapping)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class STConvDecoder(nn.Module):
    def __init__(self, cfg, encoder: STConvEncoder):
        super(STConvDecoder, self).__init__()

        # network modules
        self.layers = nn.ModuleList()

        # build network
        mapping_list = encoder.mapping_list.copy()
        edge_list = encoder.edge_list.copy()

        for i in range(cfg.model.n_layers):
            layers = []

            # unpooling
            layers.append(STUnpool(skeleton_mapping=mapping_list.pop()))

            # conv
            edges = edge_list.pop()
            for _ in range(cfg.model.n_extra_layers):
                layers.append(ResSTConv(
                    edge_list[-1],
                    cfg.model.latent_dim,
                    cfg.model.kernel_size,
                    activation=cfg.model.activation,
                    norm=cfg.model.norm,
                    dropout=cfg.model.dropout
                ))
            layers.append(ResSTConv(
                edge_list[-1],
                cfg.model.latent_dim,
                cfg.model.kernel_size,
                activation=cfg.model.activation,
                norm=cfg.model.norm,
                dropout=cfg.model.dropout
            ))

            self.layers.append(nn.Sequential(*layers))

    def forward(self, x):
        """
        x: [B, T, J_in, D]
        out: [B, T, J_out, D]
        """
        for layer in self.layers:
            x = layer(x)
        return x
