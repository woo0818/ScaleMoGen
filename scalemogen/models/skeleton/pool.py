"""ScaleMoGen skeleton pooling layers.

Code provenance: adapted from the SALAD skeleton pooling layers.
Source repository: https://github.com/seokhyeonhong/salad
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.skeleton import *

class STPool(nn.Module):
    """
    Skeleto-Temporal Pooling.
    """
    def __init__(
        self,
        dataset="t2m",
        depth=0,
    ):
        if not dataset in ["t2m", "kit", "snapmogen", "humanml3d"]:
            raise ValueError("dataset should be 't2m' or 'kit' or 'snapmogen")
        
        super(STPool, self).__init__()

        skeleton_pool, skeleton_pool_vec, self.skeleton_mapping, self.new_edges, pool_mask = self._get_skeleton_pooling(dataset, depth)
        self.skeleton_pool_vec = nn.Parameter(skeleton_pool_vec, requires_grad=True)
        self.register_buffer("skeleton_pool", skeleton_pool, persistent=False)
        self.register_buffer("pool_mask", pool_mask, persistent=False)
        
        self.temporal_pool = nn.AvgPool1d(kernel_size=2, stride=2)
    
    def _get_skeleton_pooling(self, dataset, depth):
        if depth == 0:
            if dataset == "snapmogen":
                weight = torch.zeros(12, 24)
                mapping = [
                    [(0, 1, 15, 16, 20), 0],# root
                    [(0, 15, 16, 17), 1],   # left hip
                    [(17, 18, 19), 2],      # left leg
                    [(0, 15, 20, 21), 3],   # right hip
                    [(21, 22, 23), 4],      # right leg
                    [(0, 15, 1, 2, 3), 5],  # spine
                    [(3, 2, 4, 7, 11), 6],  # chest
                    [(3, 7, 8), 7],         # left shoulder
                    [(8, 9, 10), 8],        # left arm
                    [(3, 11, 12), 9],       # right shoulder
                    [(12, 13, 14), 10],     # right arm
                    [(3, 4, 5, 6), 11],     # head
                ]
            elif dataset in ("t2m", "humanml3d"):
                weight = torch.zeros(12, 22)
                mapping = [
                    [(0, 1, 2, 3), 0],       # root
                    [(0, 1, 4), 1],          # left hip
                    [(4, 7, 10), 2],         # left leg
                    [(0, 2, 5), 3],          # right hip
                    [(5, 8, 11), 4],         # right leg
                    [(0, 3, 6, 9), 5],       # spine
                    [(9, 6, 12, 13, 14), 6], # chest
                    [(9, 13, 16), 7],        # left shoulder
                    [(16, 18, 20), 8],       # left arm
                    [(9, 14, 17), 9],        # right shoulder
                    [(17, 19, 21), 10],      # right arm
                    [(9, 12, 15), 11],       # head
                ]
            else:
                weight = torch.zeros(12, 21)
                mapping = [
                    [(0, 1, 11, 16), 0],   # root
                    [(0, 16, 17, 18), 1],  # left hip
                    [(17, 18, 19, 20), 2], # left leg
                    [(0, 11, 12, 13), 3],  # right hip
                    [(12, 13, 14, 15), 4], # right leg
                    [(0, 1, 2, 3), 5],     # spine
                    [(2, 3, 5, 8, 4), 6],  # chest
                    [(3, 8, 9), 7],        # left shoulder
                    [(8, 9, 10), 8],       # left arm
                    [(3, 5, 6), 9],        # right shoulder
                    [(5, 6, 7), 10],       # right arm
                    [(3, 4), 11],          # head
                ]

            new_edges = adj_list_to_edges([
                [1, 3, 5],
                [0, 2],
                [1],
                [0, 4],
                [3],
                [0, 6],
                [5, 7, 9, 11],
                [6, 8],
                [7],
                [6, 10],
                [9],
                [6],
            ])
                
        elif depth == 1:
            weight = torch.zeros(7, 12)
            mapping = [
                [(0, 1, 3, 5), 0], # root
                [(0, 1, 2),    1], # left lower
                [(0, 3, 4),    2], # right lower
                [(0, 5, 6),    3], # spine
                [(6, 7, 8),    4], # left upper
                [(6, 9, 10),   5], # right upper
                [(6, 11),      6], # head
            ]
            new_edges = adj_list_to_edges([
                [1, 2, 3],
                [0],
                [0],
                [0, 4, 5, 6],
                [3],
                [3],
                [3],
            ])

        else:
            weight = torch.ones(7, 7)
            mapping = [
                [(0, 1, 2, 3), 0], # root
                [(0, 1),       1], # left lower
                [(0, 2),       2], # right lower
                [(0, 3),       3], # spine
                [(3, 4),       4], # left upper
                [(3, 5),       5], # right upper
                [(3, 6),       6], # head
            ]
            new_edges = adj_list_to_edges([
                [1, 2, 3],
                [0],
                [0],
                [0, 4, 5, 6],
                [3],
                [3],
                [3],
            ])
        
        for joints, idx in mapping:
            weight[idx, joints] = 1
        weight = weight / weight.sum(dim=1, keepdim=True)
        mask = weight != 0

        weight_vec = weight[mask]
        
        return weight, weight_vec, mapping, new_edges, mask

    def forward(self, x):
        """
        x: [B, T, J, D]
        out: [B, T // 2, J_out, D]
        """
        B, T, J_in, D = x.size()

        # skeleton pooling
        skeleton_pool = self.skeleton_pool.clone()
        skeleton_pool[~self.pool_mask] = 0
        skeleton_pool[self.pool_mask] = self.skeleton_pool_vec
        # self.skeleton_pool = F.softmax(self.skeleton_pool.masked_fill(~self.pool_mask, float('-inf')), dim=1)
        masked_vec = skeleton_pool * self.pool_mask.float()
        max_vec = torch.max(masked_vec, dim=1, keepdim=True)[0]
        exps = torch.exp(masked_vec - max_vec)
        masked_exps = exps * self.pool_mask.float()
        masked_sums = masked_exps.sum(dim=1, keepdim=True)
        zeros = (masked_sums == 0)
        masked_sums += zeros.float()
        norm_skeleton_pool = masked_exps / (masked_sums + 1e-6)
        
        out = torch.matmul(norm_skeleton_pool, x) # [B, T, J_out, D]
        J_out = out.size(2)

        # temporal pooling
        out = out.permute(0, 2, 3, 1).reshape(B * J_out, D, T)
        out = self.temporal_pool(out)
        out = out.reshape(B, J_out, D, -1).permute(0, 3, 1, 2) # [B, T // 2, J_out, D]

        return out
    
class STUnpool(nn.Module):
    """
    Skeleton-Temporal Unpooling.
    """
    def __init__(
        self,
        skeleton_mapping,
    ):
        super(STUnpool, self).__init__()
        # self.skeleton_unpool, self.skeleton_unpool_vec, self.unpool_mask = self._get_skeleton_unpool(skeleton_mapping)
        # self.skeleton_unpool = self.skeleton_unpool.cuda()
        # self.unpool_mask = self.unpool_mask.cuda()
        self.skeleton_unpool = nn.Parameter(self._get_skeleton_unpool(skeleton_mapping), requires_grad=True) # [J_out, J_in]
        # self.skeleton_unpool_vec = nn.Parameter(self.skeleton_unpool_vec, requires_grad=True)
        
        self.temporal_unpool = nn.Upsample(scale_factor=2, mode="linear")
        self.unpool_act = nn.ReLU()
        
    def _get_skeleton_unpool(self, skeleton_mapping):
        max_idx = -1
        for joints, idx in skeleton_mapping:
            max_idx = max(max_idx, *joints)

        weight = torch.zeros(max_idx + 1, len(skeleton_mapping))
        
        for joints, idx in skeleton_mapping:
            weight[joints, idx] = 1

        return weight# , weight_vec, mask
    
    def forward(self, x):
        """
        x: [B, T, J_in, D]
        out: [B, T * upsample_rate, J_in, D]
        """

        B, T, J_in, D = x.size()

        # skeleton unpooling
        skeleton_unpool = self.unpool_act(self.skeleton_unpool)
        out = torch.matmul(skeleton_unpool, x) # [B, T, J_out, D]
        J_out = out.size(2)

        # temporal unpooling
        out = out.permute(0, 2, 3, 1).reshape(B * J_out, D, T)
        out = self.temporal_unpool(out.float())
        out = out.reshape(B, J_out, D, -1).permute(0, 3, 1, 2) # [B, T * upsample_rate, J_out, D]

        return out
