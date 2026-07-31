"""ScaleMoGen multi-scale BSQ quantizer.

Code provenance: adapted from LFQ/BSQ-style quantization code for coarse-to-fine
motion tokens.
Source repository: https://github.com/FoundationVision/Infinity
"""

import random
from math import ceil
from functools import partial
from itertools import zip_longest
from random import randrange
import random
from einops import rearrange, repeat

import numpy as np

import random
from math import log2, ceil
from functools import partial, lru_cache as cache
from collections import namedtuple
from contextlib import nullcontext

import torch.distributed as dist
from torch.distributed import nn as dist_nn

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module
from torch.amp import autocast

from einops import rearrange, reduce, pack, unpack

from einx import get_at
from copy import deepcopy
# constants
Return = namedtuple('Return', ['quantized', 'indices', 'bit_indices', 'entropy_aux_loss'])
LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy', 'batch_entropy', 'commitment'])

# helper functions
@cache
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def maybe_distributed_mean(t):
    if not is_distributed():
        return t

    dist_nn.all_reduce(t)
    t = t / dist.get_world_size()
    return t

def exists(v):
    return v is not None

def identity(t):
    return t

def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def l2norm(t):
    return F.normalize(t, dim = -1)

# entropy
def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def length_to_mask(length, max_len=None, device: torch.device = None) -> torch.Tensor:
    if device is None:
        device = length.device

    if isinstance(length, list):
        length = torch.tensor(length)

    if max_len is None:
        max_len = max(length)
    
    length = length.to(device)
    # max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ).to(device) < length.unsqueeze(1)
    return mask

def mean_flat(tensor: torch.Tensor, mask=None):
    """
    Take the mean over all non-batch dimensions.
    If mask is provided, compute the masked mean.
    """
    if mask is None:
        return tensor.mean(dim=list(range(1, tensor.ndim)))
    else:
        assert tensor.shape[:3] == mask.shape[:3], \
            f"mask shape {mask.shape} not compatible with tensor shape {tensor.shape}"

        # Broadcast mask to match tensor shape
        mask = mask.expand_as(tensor)  # [B, T, 1, D] 
        denom = mask.sum()
        loss = (tensor * mask).sum() / denom
        return loss
    
def gumbel_sample(
    logits,
    temperature = 1.,
    stochastic = False,
    dim = -1,
    training = True
):

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim = dim)

    return ind

# main class

class ScaleMoGenBSQ(Module):
    """Quantizes motion features over coarse-to-fine temporal and joint scales."""

    def __init__(
        self,
        *,
        dim,
        codebook_dim,
        temp_scales = None,
        soft_clamp_input_value = None,
        aux_loss = False, # intermediate auxiliary loss
        use_decay_factor=False,
        use_stochastic_depth=False,
        drop_rate=0.,
        schedule_mode="original", # ["original", "dynamic", "dense"]
        keep_first_quant=False,
        keep_last_quant=False,
        remove_residual_detach=False,
        random_flip = False,
        flip_prob = 0.5,
        flip_mode = "stochastic", # "stochastic", "deterministic"
        max_flip_lvl = 1,
        random_flip_1lvl = False, # random flip one level each time
        flip_lvl_idx = None,
        drop_when_test=False,
        drop_lvl_idx=None,
        drop_lvl_num=0,
        random_short_schedule = False, # randomly use short schedule (schedule for images of 256x256)
        short_schedule_prob = 0.5,
        disable_flip_prob = 0.0, # disable random flip in this image
        uniform_short_schedule = False,
        ctf_mapping_id = 0,
        **kwargs
    ):
        super().__init__()
        # codebook_dim = dim

        requires_projection = codebook_dim != dim
        self.codebook_dim = codebook_dim
        self.dim = dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection
        self.layernorm = nn.Identity()
        self.use_stochastic_depth = use_stochastic_depth
        self.drop_rate = drop_rate
        self.remove_residual_detach = remove_residual_detach
        self.random_flip = random_flip
        self.flip_prob = flip_prob
        self.flip_mode = flip_mode
        self.max_flip_lvl = max_flip_lvl
        self.random_flip_1lvl = random_flip_1lvl
        self.flip_lvl_idx = flip_lvl_idx
        assert (random_flip and random_flip_1lvl) == False
        self.disable_flip_prob = disable_flip_prob

        self.drop_when_test = drop_when_test
        self.drop_lvl_idx = drop_lvl_idx
        self.drop_lvl_num = drop_lvl_num
        if self.drop_when_test:
            assert drop_lvl_idx is not None
            assert drop_lvl_num > 0
        self.random_short_schedule = random_short_schedule
        self.short_schedule_prob = short_schedule_prob
        self.full2short = {7:7, 10:7, 13:7, 16:16, 20:16, 24:16}
        self.full2short_f8 = {20:20, 24:20, 28:20}
        self.uniform_short_schedule = uniform_short_schedule
        assert not (self.random_short_schedule and self.uniform_short_schedule)

        self.lfq = BSQ(
            dim = codebook_dim,
            codebook_dim=codebook_dim,
            codebook_scale = 1,
            soft_clamp_input_value = soft_clamp_input_value,
            **kwargs
        )

        self.z_interplote_up = 'trilinear'
        self.z_interplote_down = 'area'
        
        self.use_decay_factor = use_decay_factor
        self.schedule_mode = schedule_mode
        self.keep_first_quant = keep_first_quant
        self.keep_last_quant = keep_last_quant
        if self.use_stochastic_depth and self.drop_rate > 0:
            assert self.keep_first_quant or self.keep_last_quant
        
        self.ctf_mapping_id = ctf_mapping_id

        # [80, 40, 20, 10, 5, 2, 1]
        self.temp_scales = temp_scales #[8, 4, 2, 1]


        if self.ctf_mapping_id == 0:  # inner-to-outer, tempFirst
            self.spatial_chain = [
                [[0, 1, 2, 3, 4, 5, 6]],
                [[0, 3], [1, 2, 4, 5, 6]],
                [[0, 3], [1, 2], [4, 5, 6]],
                [[0], [3], [1, 2], [4, 5], [6]],
                [[0], [1], [2], [3], [4], [5], [6]]
            ]
            self.ctf_mapping = [
                (0, 0),  
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (6, 0),
                (6, 1),
                (6, 2),
                (6, 3),
                (6, 4),
            ]
            
        elif self.ctf_mapping_id == 1:  # lower & upper, alter
            '''
            self.spatial_chain = [
                [[0, 1, 2, 3, 4, 5, 6]],              # Coarse: Whole body
                [[0, 1, 2], [3, 4, 5, 6]],           # Mid-level: Lower & Upper
                [[0], [1], [2], [3], [4], [5], [6]]   # Fine: individual joints
            ]
            '''
            self.spatial_chain = [
                [[0, 1, 2, 3, 4, 5, 6]],
                [[0, 1, 2], [3, 4, 5, 6]],
                [[0], [1, 2], [3, 6], [4, 5]],
                [[0], [1], [2], [3], [4], [5], [6]]
            ]
            '''
            self.ctf_mapping = [
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (6, 0),
                (6, 1),
                (6, 2),
                (6, 3)
            ]
            '''

            self.ctf_mapping = [
                (0, 0),
                (1, 0),
                (2, 0),
                (2, 1),
                (3, 1),
                (4, 1),
                (4, 2),
                (5, 2),
                (6, 2),
                (6, 3)
            ]

        elif self.ctf_mapping_id == 2:
            self.spatial_chain = [
                [[0, 1, 2, 3, 4, 5, 6]],              # Coarse: Whole body
            ]
            self.ctf_mapping = [
                (0, 0),  
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (6, 0),
            ]

        elif self.ctf_mapping_id == 3:
            self.spatial_chain = [
                [[0], [1], [2], [3], [4], [5], [6]]   # Fine: individual joints
            ]
            self.ctf_mapping = [
                (0, 0),  
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (6, 0),
            ]
        
        elif self.ctf_mapping_id == 4:  # lower & upper, tempFirst
            self.spatial_chain = [
                [[0, 1, 2, 3, 4, 5, 6]],
                [[0, 1, 2], [3, 4, 5, 6]],
                [[0], [1, 2], [3, 6], [4, 5]],
                [[0], [1], [2], [3], [4], [5], [6]]
            ]
            
            self.ctf_mapping = [
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (6, 0),
                (6, 1),
                (6, 2),
                (6, 3)
            ]
            
        
        elif self.ctf_mapping_id == 5:  # inner to outer increase scale
            self.spatial_chain = [
                [[0, 1, 2, 3, 4, 5, 6]],
                [[0, 3], [1, 2, 4, 5, 6]],
                [[0, 3], [1, 2], [4, 5, 6]],
                [[0], [3], [1, 2], [4, 5], [6]],
                [[0], [1], [2], [3], [4], [5], [6]]
            ]
            '''
            self.ctf_mapping = [
                (0, 0),  
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (6, 0),
                (6, 1),
                (6, 2),
                (6, 3),
                (6, 4),
            ]
            '''
            self.ctf_mapping = [
                (0, 0),  
                (1, 0),
                (1, 1),
                (2, 1),
                (3, 1),
                (3, 2),
                (4, 2),
                (5, 2),
                (5, 3),
                (6, 3),
                (6, 4),
            ]

        self.coarse_to_fine_chain = [
            (self.temp_scales[t_idx], self.spatial_chain[s_idx])
            for t_idx, s_idx in self.ctf_mapping
        ]

        self.len_scale_factor = 4 #TODO

    @property
    def codebooks(self):
        return self.lfq.codebook

    def get_codes_from_indices(self, indices_list):
        all_codes = []
        for indices in indices_list:
            codes = self.lfq.indices_to_codes(indices)
            all_codes.append(codes)
        _, _, T, H, W = all_codes[-1].size()
        summed_codes = 0
        for code in all_codes:
            summed_codes += F.interpolate(code, size=(T, H, W), mode=self.z_interplote_up)
        return summed_codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        return self.project_out(codes_summed)

    def flip_quant(self, x):
        if self.flip_mode == 'stochastic':
            flip_mask = torch.rand_like(x) < self.flip_prob
        else:
            raise NotImplementedError
        x = x.clone()
        x[flip_mask] = -x[flip_mask]
        return x

    def forward(
        self,
        x,
        temperature=0.5, 
        m_lens = None,
        mask = None,
        return_all_codes = False,
    ):
        if x.ndim == 4:
            x = x.unsqueeze(2)
        N, C, _, T, J = x.size() 

        if m_lens is not None:
            full_scale_mask = length_to_mask(torch.ceil(m_lens / self.len_scale_factor).long(), T).unsqueeze(-1)  
        else:
            full_scale_mask = torch.ones((x.shape[0], T, 1), device=x.device).bool()
        
        if self.schedule_mode.startswith("same"): # NOT USED
            scale_num = int(self.schedule_mode[len("same"):])
            #assert T == 1
            scale_schedule = [(1, T, J)] * scale_num
        else:
            scale_schedule = self.coarse_to_fine_chain #get_latent2scale_schedule(T, H, W, mode=self.schedule_mode)
            scale_num = len(scale_schedule)
        
        if self.uniform_short_schedule:
            scale_num_short = self.full2short_f8[scale_num] if self.schedule_mode == "dense_f8" else self.full2short[scale_num]
            scale_num = random.randint(scale_num_short, scale_num)
            scale_schedule = scale_schedule[:scale_num]
        elif self.random_short_schedule and random.random() < self.short_schedule_prob:
            if self.schedule_mode == "dense_f8":
                scale_num = self.full2short_f8[scale_num]
            else:
                scale_num = self.full2short[scale_num]
            scale_schedule = scale_schedule[:scale_num]
       
        # x = self.project_in(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous() # (b, c, t, h, w) => (b, t, h, w, c)
        x = self.project_in(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous() # (b, t, h, w, c) => (b, c, t, h, w) 
        x = self.layernorm(x)

        quantized_out = 0.
        residual = x.clone()

        all_losses = []
        all_indices = []
        all_bit_indices = []
        all_mask = []
        
        # go through the layers
        out_fact = init_out_fact = 1.0
        # residual_list = []
        # interpolate_residual_list = []
        # quantized_list = []
        if self.drop_when_test:
            drop_lvl_start = self.drop_lvl_idx
            drop_lvl_end = self.drop_lvl_idx + self.drop_lvl_num
        disable_flip = True if random.random() < self.disable_flip_prob else False # disable random flip in this image
        with autocast('cuda', enabled = False):
            #for si, (pt, ph, pw) in enumerate(scale_schedule):
            for si, (t_scale, spatial_groups) in enumerate(scale_schedule):
                residual = residual * full_scale_mask.unsqueeze(1).unsqueeze(1)
                
                out_fact = max(0.1, out_fact) if self.use_decay_factor else init_out_fact
                #if (pt, ph, pw) != (T, H, W):
                if si != len(scale_schedule)-1:
                    if t_scale != 1:
                        interpolate_residual = F.interpolate(residual, size=(1, int(T//t_scale), residual.shape[-1]), mode=self.z_interplote_down)  # [N, C, 1, T', J]
                    else:
                        interpolate_residual = residual

                    spatial_grouped = []
                    for group in spatial_groups:
                        group_tensor = interpolate_residual[..., group] if isinstance(group, list) else interpolate_residual[..., [group]]
                        group_avg = group_tensor.mean(dim=-1, keepdim=True)
                        spatial_grouped.append(group_avg)

                    interpolate_residual = torch.cat(spatial_grouped, dim=-1)
                    #interpolate_residual = F.interpolate(residual, size=(pt, ph, pw), mode=self.z_interplote_down)
                else:
                    interpolate_residual = residual

                if m_lens is not None: #TODO
                    mask = length_to_mask(torch.ceil(m_lens / (t_scale * self.len_scale_factor)).long(), interpolate_residual.shape[-2]) # (n t)
                    # mask = mask & keep_mask[:, None]
                    mask = mask.unsqueeze(-1).expand(-1, -1, interpolate_residual.shape[-1])
                    all_mask.append(rearrange(mask, 'n t j -> n (t j)').unsqueeze(-1))

                if self.training and self.use_stochastic_depth and random.random() < self.drop_rate:
                    if (si == 0 and self.keep_first_quant) or (si == scale_num - 1 and self.keep_last_quant):
                        quantized, indices, bit_indices, loss = self.lfq(interpolate_residual, mask = all_mask[-1])
                        if self.random_flip and si < self.max_flip_lvl and (not disable_flip):
                            quantized = self.flip_quant(quantized)
                        quantized = quantized * out_fact
                        all_indices.append(indices)
                        all_losses.append(loss)
                        all_bit_indices.append(bit_indices)
                    else:
                        quantized = torch.zeros_like(interpolate_residual)
                elif self.drop_when_test and drop_lvl_start <= si < drop_lvl_end:
                    continue                     
                else:
                    # residual_norm = torch.norm(interpolate_residual.detach(), dim=1) # (b, t, h, w)
                    # print(si, residual_norm.min(), residual_norm.max(), residual_norm.mean())
                    quantized, indices, bit_indices, loss = self.lfq(interpolate_residual, mask = all_mask[-1])
                    quantized[~mask.unsqueeze(1).unsqueeze(1).expand_as(quantized)] = 0.
                    if self.random_flip and si < self.max_flip_lvl and (not disable_flip):
                        quantized = self.flip_quant(quantized)
                    if self.random_flip_1lvl and si == self.flip_lvl_idx and (not disable_flip):
                        quantized = self.flip_quant(quantized)
                    quantized = quantized * out_fact
                    all_indices.append(indices)
                    all_losses.append(loss)
                    all_bit_indices.append(bit_indices)
                
                # quantized_list.append(torch.norm(quantized.detach(), dim=1).mean())
                #if (pt, ph, pw) != (T, H, W):
                    #quantized = F.interpolate(quantized, size=(T, H, W), mode=self.z_interplote_up).contiguous()

                up_quantized = torch.zeros((N, self.codebook_dim, 1, T, J), device=x.device, dtype=x.dtype)
                if si != len(scale_schedule)-1:
                    for group_idx, group in enumerate(spatial_groups):
                        joint_indices = group if isinstance(group, list) else [group]
                        num_joints = len(joint_indices)
                        
                        up_group_feat = F.interpolate(quantized[..., group_idx:group_idx+1], size=(1, T, num_joints), mode=self.z_interplote_up)  # → [N, C, T, num_joints]
                        up_quantized[..., joint_indices] = up_group_feat.type(up_quantized.dtype)
                else:
                    up_quantized = quantized

                
                if self.remove_residual_detach:
                    residual = residual - up_quantized
                else:
                    residual = residual - up_quantized.detach()
                quantized_out = quantized_out + up_quantized

                if self.use_decay_factor:
                    out_fact -= 0.1
        # print("residual_list:", residual_list)
        # print("interpolate_residual_list:", interpolate_residual_list)
        # print("quantized_list:", quantized_list)
        # import ipdb; ipdb.set_trace()
        # project out, if needed
        quantized_out = quantized_out.permute(0, 2, 3, 4, 1).contiguous() # (b, c, t, h, w) => (b, t, h, w, c)
        quantized_out = self.project_out(quantized_out)
        quantized_out = quantized_out.permute(0, 4, 1, 2, 3).contiguous() # (b, t, h, w, c) => (b, c, t, h, w)

        # image
        if quantized_out.size(2) == 1:
            quantized_out = quantized_out.squeeze(2)

        # stack all losses and indices

        all_losses = torch.stack(all_losses, dim = -1)

        ret = (quantized_out, all_indices, all_bit_indices, all_losses)

        if not return_all_codes:
            return ret

        # whether to return all codes from all codebooks across layers
        all_codes = self.get_codes_from_indices(all_indices)

        # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)

        return (*ret, all_codes)


class BSQ(Module):
    def __init__(
        self,
        *,
        dim = None,
        codebook_dim=None,
        entropy_loss_weight = 0.1,
        commitment_loss_weight = 0.25,
        diversity_gamma = 1.,
        straight_through_activation = nn.Identity(),
        num_codebooks = 1,
        keep_num_codebooks_dim = None,
        codebook_scale = 1.,                        # for residual LFQ, codebook scaled down by 2x at each layer
        frac_per_sample_entropy = 1.,               # make less than 1. to only use a random fraction of the probs for per sample entropy
        has_projections = None,
        projection_has_bias = True,
        soft_clamp_input_value = None,
        cosine_sim_project_in = False,
        cosine_sim_project_in_scale = None,
        channel_first = None,
        experimental_softplus_entropy_loss = False,
        entropy_loss_offset = 5.,                   # how much to shift the loss before softplus
        spherical = True,                          # from https://arxiv.org/abs/2406.07548
        force_quantization_f32 = True,               # will force the quantization step to be full precision
        inv_temperature = 100.0,
        gamma0=1.0, gamma=1.0, zeta=1.0,
        new_quant = False, # new quant function，
        use_out_phi = False, # use output phi network
        use_out_phi_res = False, # residual out phi
    ):
        super().__init__()

        # some assert validations
        assert exists(dim) , 'dim must be specified for BSQ'

        # codebook_dim = dim
        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)
        self.codebook_dims = codebook_dims

        has_projections = default(has_projections, dim != codebook_dims)

        if cosine_sim_project_in:
            cosine_sim_project_in = default(cosine_sim_project_in_scale, codebook_scale)
            project_in_klass = partial(CosineSimLinear, scale = cosine_sim_project_in)
        else:
            project_in_klass = partial(nn.Linear, bias = projection_has_bias)

        self.project_in = project_in_klass(dim, codebook_dims) if has_projections else nn.Identity() # nn.Identity()
        self.project_out = nn.Linear(codebook_dims, dim, bias = projection_has_bias) if has_projections else nn.Identity() # nn.Identity()
        self.has_projections = has_projections

        self.out_phi = nn.Linear(codebook_dims, codebook_dims) if use_out_phi else nn.Identity()
        self.use_out_phi_res = use_out_phi_res
        if self.use_out_phi_res:
            self.out_phi_scale = nn.Parameter(torch.zeros(codebook_dims), requires_grad=True) # init as zero

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # channel first

        self.channel_first = channel_first

        # straight through activation

        self.activation = straight_through_activation

        # For BSQ (binary spherical quantization)
        if not spherical:
            raise ValueError("For BSQ, spherical must be True.")
        self.persample_entropy_compute = 'analytical'
        self.inv_temperature = inv_temperature
        self.gamma0 = gamma0  # loss weight for entropy penalty
        self.gamma = gamma  # loss weight for entropy penalty
        self.zeta = zeta    # loss weight for entire entropy penalty
        self.new_quant = new_quant

        # entropy aux loss related weights

        assert 0 < frac_per_sample_entropy <= 1.
        self.frac_per_sample_entropy = frac_per_sample_entropy

        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

        # codebook scale

        self.codebook_scale = codebook_scale

        # commitment loss

        self.commitment_loss_weight = commitment_loss_weight

        # whether to soft clamp the input value from -value to value

        self.soft_clamp_input_value = soft_clamp_input_value
        assert not exists(soft_clamp_input_value) or soft_clamp_input_value >= codebook_scale

        # whether to make the entropy loss positive through a softplus (experimental, please report if this worked or not in discussions)

        self.entropy_loss_offset = entropy_loss_offset
        self.experimental_softplus_entropy_loss = experimental_softplus_entropy_loss

        # for no auxiliary loss, during inference

        self.register_buffer('mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # whether to force quantization step to be f32

        self.force_quantization_f32 = force_quantization_f32

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    # @property
    # def dtype(self):
    #     return self.codebook.dtype

    def indices_to_codes(
        self,
        indices,
        label_type = 'int_label',
        project_out = True
    ):
        assert label_type in ['int_label', 'bit_label']
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        should_transpose = default(self.channel_first, is_img_or_video)

        if not self.keep_num_codebooks_dim:
            if label_type == 'int_label':
                indices = rearrange(indices, '... -> ... 1')
            else:
                indices = indices.unsqueeze(-2)

        # indices to codes, which are bits of either -1 or 1

        if label_type == 'int_label':
            assert indices[..., None].int().min() > 0
            bits = ((indices[..., None].int() & self.mask) != 0).float() # .to(self.dtype)
        else:
            bits = indices

        codes = self.bits_to_codes(bits)

        codes = l2norm(codes) # must normalize when using BSQ

        codes = rearrange(codes, '... c d -> ... (c d)')

        # whether to project codes out to original dimensions
        # if the input feature dimensions were not log2(codebook size)

        if project_out:
            codes = self.project_out(codes)

        # rearrange codes back to original shape

        if should_transpose:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def quantize(self, z):
        assert z.shape[-1] == self.codebook_dims, f"Expected {self.codebook_dims} dimensions, got {z.shape[-1]}"

        zhat = torch.where(z > 0, 
                           torch.tensor(1, dtype=z.dtype, device=z.device), 
                           torch.tensor(-1, dtype=z.dtype, device=z.device))
        return z + (zhat - z).detach()

    def quantize_new(self, z):
        assert z.shape[-1] == self.codebook_dims, f"Expected {self.codebook_dims} dimensions, got {z.shape[-1]}"

        zhat = torch.where(z > 0, 
                           torch.tensor(1, dtype=z.dtype, device=z.device), 
                           torch.tensor(-1, dtype=z.dtype, device=z.device))

        q_scale = 1. / (self.codebook_dims ** 0.5)
        zhat = q_scale * zhat # on unit sphere

        return z + (zhat - z).detach()

    def soft_entropy_loss(self, z):
        if self.persample_entropy_compute == 'analytical':
            # if self.l2_norm:
            p = torch.sigmoid(-4 * z / (self.codebook_dims ** 0.5) * self.inv_temperature)
            # else:
            #     p = torch.sigmoid(-4 * z * self.inv_temperature)
            prob = torch.stack([p, 1-p], dim=-1) # (b, h, w, 18, 2)
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean() # (b,h,w,18)->(b,h,w)->scalar
        else:
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()

        # macro average of the probability of each subgroup
        avg_prob = reduce(prob, '... g d ->g d', 'mean') # (18, 2)
        codebook_entropy = self.get_entropy(avg_prob, dim=-1, normalize=False)

        # the approximation of the entropy is the sum of the entropy of each subgroup
        return per_sample_entropy, codebook_entropy.sum(), avg_prob

    def get_entropy(self, count, dim=-1, eps=1e-4, normalize=True):
        if normalize: # False
            probs = (count + eps) / (count + eps).sum(dim=dim, keepdim =True)
        else: # True
            probs = count
        H = -(probs * torch.log(probs + 1e-8)).sum(dim=dim)
        return H

    def forward(
        self,
        x,
        return_loss_breakdown = False,
        mask = None,
        entropy_weight=0.1
    ):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        is_img_or_video = x.ndim >= 4
        should_transpose = default(self.channel_first, is_img_or_video)

        # standardize image or video into (batch, seq, dimension)
        x_copy = deepcopy(x.detach())
        if should_transpose:
            x = rearrange(x, 'b d ... -> b ... d')
            x, ps = pack_one(x, 'b * d') # x.shape [b, hwt, c]

        assert x.shape[-1] == self.dim, f'expected dimension of {self.dim} but received {x.shape[-1]}'

        x = self.project_in(x)

        # split out number of codebooks

        x = rearrange(x, 'b n (c d) -> b n c d', c = self.num_codebooks)
        
        x = l2norm(x)

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32

        quantization_context = partial(autocast, 'cuda', enabled = False) if force_f32 else nullcontext

        indices = None
        with quantization_context():

            if force_f32:
                orig_dtype = x.dtype
                x = x.float()
            
            # use straight-through gradients (optionally with custom activation fn) if training
            if self.new_quant:
                quantized = self.quantize_new(x)
            else:
                quantized = self.quantize(x)
                q_scale = 1. / (self.codebook_dims ** 0.5)
                quantized = q_scale * quantized # on unit sphere

            # calculate indices
            # bit_indices = (quantized > 0).int()

            # entropy aux loss
            if self.training:
                if exists(mask):
                    persample_entropy, cb_entropy, avg_prob = self.soft_entropy_loss(x[mask]) # compute entropy
                else:
                    persample_entropy, cb_entropy, avg_prob = self.soft_entropy_loss(x) # compute entropy
                entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy
            else:
                # if not training, just return dummy 0
                entropy_penalty = persample_entropy = cb_entropy = self.zero

            # commit loss
            if self.training and self.commitment_loss_weight > 0.:
                if exists(mask):
                    commit_loss = mean_flat((x-quantized.detach()).pow(2), mask=mask.unsqueeze(-1))
                    #commit_loss = commit_loss[mask]
                else:
                    commit_loss = F.mse_loss(x, quantized.detach(), reduction = 'none')

                commit_loss = commit_loss.mean()
            else:
                commit_loss = self.zero

            # calculate indices
            if exists(mask):
                quantized[~mask] = 0.
            bit_indices = (quantized > 0).int()

            # input back to original dtype if needed
            if force_f32:
                x = x.type(orig_dtype)

        # merge back codebook dim
        x = quantized # rename quantized to x for output
        
        if self.use_out_phi_res:
            x = x + self.out_phi_scale * self.out_phi(x) # apply out_phi on quant output as residual
        else:
            x = self.out_phi(x) # apply out_phi on quant output
        
        x = rearrange(x, 'b n c d -> b n (c d)')

        # project out to feature dimension if needed

        x = self.project_out(x)

        # reconstitute image or video dimensions

        if should_transpose:
            x = unpack_one(x, ps, 'b * d')
            x = rearrange(x, 'b ... d -> b d ...')

            bit_indices = unpack_one(bit_indices, ps, 'b * c d')

        # whether to remove single codebook dim

        if not self.keep_num_codebooks_dim:
            bit_indices = rearrange(bit_indices, '... 1 d -> ... d')

        # complete aux loss

        aux_loss = commit_loss * self.commitment_loss_weight + (self.zeta * entropy_penalty / self.inv_temperature) * entropy_weight
        # returns

        ret = Return(x, indices, bit_indices, aux_loss)

        if not return_loss_breakdown:
            return ret

        return ret, LossBreakdown(persample_entropy, cb_entropy, commit_loss)
