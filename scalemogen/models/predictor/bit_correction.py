"""Bitwise self-correction utilities for ScaleMoGen predictor training.

Code provenance: adapted from the Infinity bitwise token training pipeline.
Source repository: https://github.com/FoundationVision/Infinity
"""

import os.path as osp

import torch
import torch.nn.functional as F
import numpy as np


def labels2image(vae, all_indices, label_type='int_label', scale_schedule=None):
    """Decode bit labels into a debug visualization image."""
    summed_codes, recons_imgs = vae.decode_from_indices(all_indices, scale_schedule, label_type)
    recons_img = recons_imgs[0]
    recons_img = (recons_img + 1) / 2
    recons_img = recons_img.permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)[:,:,::-1]
    return recons_img


def features2image(vae, raw_features):
    """Decode VQ features into a debug visualization image."""
    recons_imgs = vae.decode(raw_features.squeeze(-3))
    recons_img = recons_imgs[0]
    recons_img = (recons_img + 1) / 2
    recons_img = recons_img.permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)[:,:,::-1]
    return recons_img

class ScaleMoGenBitCorrection(object):
    """Builds noisy bitwise token inputs and ground-truth labels for training."""

    def __init__(self, vae, args):
        self.noise_apply_layers = args.noise_apply_layers
        self.noise_apply_requant = args.noise_apply_requant
        self.noise_apply_strength = args.noise_apply_strength
        self.apply_spatial_patchify = args.apply_spatial_patchify
        self.vae = vae
        self.debug_bsc = args.debug_bsc

    def flip_requant(self, vae_scale_schedule, inp_B3HW, raw_features, all_mask, device):
        with torch.amp.autocast('cuda', enabled = False):
            B, C, T, J = raw_features.shape
            if raw_features.dim() == 4:
                codes_out = raw_features.unsqueeze(2)
            else:
                codes_out = raw_features
            cum_var_input = 0
            gt_all_bit_indices = []
            pred_all_bit_indices = []
            x_BLC_wo_prefix = []

            codes_out = codes_out.transpose(1, -1)
            codes_out = self.vae.quantizer2d.project_in(codes_out)
            codes_out = codes_out.transpose(1, -1)

            C = codes_out.shape[1]
            
            for si, (t_scale, spatial_groups) in enumerate(vae_scale_schedule):
                residual = codes_out - cum_var_input
                if si != len(vae_scale_schedule)-1:
                    if t_scale != 1:
                        residual = F.interpolate(residual, size=(1, int(T//t_scale), residual.shape[-1]), mode=self.vae.quantizer2d.z_interplote_down)  # [N, C, 1, T', J]
                    else:
                        residual = residual
                    
                    spatial_grouped = []
                    for group in spatial_groups:
                        group_tensor = residual[..., group] if isinstance(group, list) else residual[..., [group]]
                        group_avg = group_tensor.mean(dim=-1, keepdim=True)
                        spatial_grouped.append(group_avg)

                    residual = torch.cat(spatial_grouped, dim=-1)
                    #interpolate_residual = F.interpolate(residual, size=(pt, ph, pw), mode=self.z_interplote_down)
                else:
                    residual = residual

                #print(residual.shape, all_mask[si].shape)
                quantized, _, bit_indices, loss = self.vae.quantizer2d.lfq(residual, mask=all_mask[si].unsqueeze(-1)) # quantized shape: [B, d_vae, 1, h, w], bit_indices shape: [B,1,h,w,d_vae]
                
                gt_all_bit_indices.append(bit_indices)
                if si < self.noise_apply_layers:
                    noise_apply_strength = np.random.randint(0, 100 * self.noise_apply_strength+1) * 0.01
                    mask = torch.rand(*bit_indices.shape).to(device) < noise_apply_strength
                    pred_bit_indices = bit_indices.clone()
                    pred_bit_indices[mask] = 1 - pred_bit_indices[mask]
                    pred_all_bit_indices.append(pred_bit_indices)
                    if self.noise_apply_requant:
                        quantized = self.vae.quantizer2d.lfq.indices_to_codes(pred_bit_indices.float(), label_type = 'bit_label')
                else:
                    pred_all_bit_indices.append(bit_indices)
                
                up_quantized = torch.zeros((B, C, 1, T, J), device=device, dtype=raw_features.dtype)
                if si != len(vae_scale_schedule)-1:
                    for group_idx, group in enumerate(spatial_groups):
                        joint_indices = group if isinstance(group, list) else [group]
                        num_joints = len(joint_indices)
                        
                        up_group_feat = F.interpolate(quantized[..., group_idx:group_idx+1], size=(1, T, num_joints), mode=self.vae.quantizer2d.z_interplote_up).contiguous()  # → [N, C, T, num_joints]
                        up_quantized[..., joint_indices] = up_group_feat
                else:
                    up_quantized = quantized

                #cum_var_input = cum_var_input + F.interpolate(quantized, size=vae_scale_schedule[-1], mode=self.vae.quantizer2d.z_interplote_up).contiguous()
                cum_var_input = cum_var_input + up_quantized

                if si < len(vae_scale_schedule)-1:
                    next_t_scale, next_spatial_groups = vae_scale_schedule[si+1]
                    if (si+1) != len(vae_scale_schedule)-1:
                        if next_t_scale != 1:
                            this_scale_input = F.interpolate(cum_var_input, size=(1, int(T//next_t_scale), cum_var_input.shape[-1]), mode=self.vae.quantizer2d.z_interplote_up).contiguous()  # [N, C, 1, T', J]
                        else:
                            this_scale_input = cum_var_input

                        spatial_grouped = []
                        for group in next_spatial_groups:
                            group_tensor = this_scale_input[..., group] if isinstance(group, list) else this_scale_input[..., [group]]
                            group_avg = group_tensor.mean(dim=-1, keepdim=True)
                            spatial_grouped.append(group_avg)

                        this_scale_input = torch.cat(spatial_grouped, dim=-1)

                    else:
                        this_scale_input = cum_var_input

                    #this_scale_input = F.interpolate(cum_var_input, size=vae_scale_schedule[si+1], mode=self.vae.quantizer2d.z_interplote_up).contiguous()
                    if self.apply_spatial_patchify:
                        # (B,d,1,H,W) -> (B,d,H,W) -> (B,4d,H/2,W/2)
                        this_scale_input = torch.nn.functional.pixel_unshuffle(this_scale_input.squeeze(-3), 2)
                    x_BLC_wo_prefix.append(this_scale_input.reshape(*this_scale_input.shape[:2], -1).permute(0,2,1)) # (B,H/2*W/2,4C) or (B,H*W,C)


            if self.apply_spatial_patchify:
                gt_ms_idx_Bl = []
                for item in gt_all_bit_indices:
                    # item shape: (B,1,H,W,d)
                    item = item.squeeze(1).permute(0,3,1,2) # (B,d,H,W)
                    # (B,d,H,W) -> (B,4d,H/2,W/2)
                    item = torch.nn.functional.pixel_unshuffle(item, 2)
                    # (B,4d,H/2,W/2) -> (B,H/2,W/2,4d) -> (B,H/2*w/2,4d)
                    item = item.permute(0,2,3,1).reshape(B, -1, 4*self.vae.codebook_dim)
                    gt_ms_idx_Bl.append(item)
            else:
                gt_ms_idx_Bl = [item.reshape(B, -1, self.vae.quantizer2d.codebook_dim) for item in gt_all_bit_indices]
            x_BLC_wo_prefix = torch.cat(x_BLC_wo_prefix, 1)

            if self.debug_bsc:
                self.visualize(vae_scale_schedule, inp_B3HW, gt_all_bit_indices, pred_all_bit_indices)
            
        return x_BLC_wo_prefix, gt_ms_idx_Bl
    
    def visualize(self, vae_scale_schedule, inp_B3HW, gt_all_bit_indices, pred_all_bit_indices):
        import cv2

        gt_img = (inp_B3HW.squeeze(-2) + 1) / 2 * 255
        gt_img = gt_img[0].permute(0,1).cpu().numpy().astype(np.uint8)[:,:]
        recons_img_2 = labels2image(self.vae, gt_all_bit_indices, label_type='bit_label', scale_schedule=vae_scale_schedule)
        recons_img_3 = labels2image(self.vae, pred_all_bit_indices, label_type='bit_label', scale_schedule=vae_scale_schedule)
        cat_image = np.concatenate([gt_img, recons_img_2, recons_img_3], axis=1)
        save_path = osp.abspath('non_teacher_force.jpg')
        cv2.imwrite(save_path, cat_image)
        print(f'Save to {save_path}')
        print(cat_image.shape)
