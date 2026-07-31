"""ScaleMoGen autoregressive motion-token predictor.

Code provenance: adapted from the Infinity autoregressive transformer for
bitwise motion tokens.
Source repository: https://github.com/FoundationVision/Infinity
"""

import math
import random
import time
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model
import numpy as np
from einops import rearrange

import scalemogen.utils.dist as dist
from scalemogen.utils.dist import for_visualize
from scalemogen.models.predictor.blocks import AdaLNBeforeHead, CrossAttnBlock, SelfAttnBlock, CrossAttention, FastRMSNorm, precompute_rope2d_freqs_grid

from scalemogen.utils import misc


def length_to_mask(length, max_len, device: torch.device = None) -> torch.Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length)
    
    length = length.to(device)
    # max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ).to(device) < length.unsqueeze(1)
    return mask


class MultiInpIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class TextAttentivePool(nn.Module):
    def __init__(self, Ct5: int, D: int):
        super().__init__()
        self.Ct5, self.D = Ct5, D
        if D > 4096:
            self.head_dim = 64 
        else:
            self.head_dim = 128

        self.num_heads = Ct5 // self.head_dim
        self.ca = CrossAttention(for_attn_pool=True, embed_dim=self.D, kv_dim=Ct5, num_heads=self.num_heads)
    def forward(self, ca_kv):
        return self.ca(None, ca_kv).squeeze(1)

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).reshape(-1, 1, 6, C)   # B16C


class ScaleMoGenBlockGroup(nn.Module):
    def __init__(self, ls, num_blocks_in_a_chunk, index):
        super().__init__()
        self.module = nn.ModuleList()
        for i in range(index, index+num_blocks_in_a_chunk):
            self.module.append(ls[i])

    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, scale_schedule=None, checkpointing_full_block=False, rope2d_freqs_grid=None):
        h = x
        for m in self.module:
            if checkpointing_full_block:
                h = torch.utils.checkpoint.checkpoint(m, h, cond_BD, ca_kv, attn_bias_or_two_vector, scale_schedule, rope2d_freqs_grid, use_reentrant=False)
            else:
                h = m(h, cond_BD, ca_kv, attn_bias_or_two_vector, scale_schedule, rope2d_freqs_grid)
        return h

class ScaleMoGenTransformer(nn.Module):
    """Autoregressive transformer for coarse-to-fine bitwise token prediction."""

    def __init__(
        self, vae_local,
        text_channels=0, text_maxlen=0,     # text-cond generation
        selecting_idx=None,                 # class-cond generation
        embed_dim=1024, depth=16, num_heads=16, mlp_ratio=4.,   # model's architecture
        drop_rate=0., drop_path_rate=0.,    # drop out and drop path
        norm_eps=1e-6, rms_norm=False,      # norm layer
        shared_aln=False, head_aln=True,    # adaptive norm
        cond_drop_rate=0.1,                 # for classifier-free guidance
        rand_uncond=False,
        cross_attn_layer_scale=-1., nm0=False, tau=1, cos_attn=True, swiglu=False,
        raw_scale_schedule=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        head_depth=1,
        top_p=0.0, top_k=0.0,
        block_chunks=1,
        checkpointing=None,
        pad_to_multiplier=0,
        batch_size=2,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        rope2d_each_sa_layer=0,
        rope2d_normalized_by_hw=0,
        pn=None,
        video_frames=1,
        always_training_scales=20,
        apply_spatial_patchify = 0,
        inference_mode=False,
        other_args: Dict[str, Any] = {}, # Added for PE config
    ):
        
        # set hyperparameters
        self.C = embed_dim
        self.inference_mode = inference_mode

        # --- "Fill-in-the-Blank" Training Parameters ---
        self.fill_blank_prob = other_args.get('fill_blank_prob', 0.5)
        self.mask_ratio = other_args.get('mask_ratio', 0.3)
        
        self.apply_spatial_patchify = apply_spatial_patchify
        if self.apply_spatial_patchify:
            self.d_vae = vae_local.quantizer2d.lfq.codebook_dim * 4
        else:
            self.d_vae = vae_local.quantizer2d.lfq.codebook_dim
        self.use_bit_label = use_bit_label
        self.codebook_dim = self.d_vae
        self.V = (self.codebook_dim * 2) if self.use_bit_label else vae_local.vocab_size
        self.bit_mask = vae_local.quantizer2d.lfq.mask if self.use_bit_label else None
        self.Ct5 = text_channels
        self.depth = depth
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.mlp_ratio = mlp_ratio
        self.cond_drop_rate = cond_drop_rate
        norm_eps = float(norm_eps)
        self.norm_eps = norm_eps
        self.prog_si = -1
        self.pn = pn
        self.video_frames = video_frames
        self.always_training_scales = always_training_scales
        self.coarse_to_fine_chain = vae_local.quantizer2d.coarse_to_fine_chain
        self.len_scale_factor = 4
        self.max_motion_length = vae_local.cfg.data.max_motion_length
        self.T = self.max_motion_length // 4
        self.motion_scale_schedule = [(1, self.T//t_s, len(s_groups)) for (t_s, s_groups) in self.coarse_to_fine_chain]

        assert add_lvl_embeding_only_first_block in [0,1]
        self.add_lvl_embeding_only_first_block = add_lvl_embeding_only_first_block
        assert rope2d_each_sa_layer in [0,1]
        self.rope2d_each_sa_layer = rope2d_each_sa_layer
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
        print(
            f"[ScaleMoGen][Predictor] codebook_dim={self.codebook_dim} "
            f"bit_label={self.use_bit_label} lvl_embed_first_block={self.add_lvl_embeding_only_first_block} "
            f"rope_each_layer={rope2d_each_sa_layer} rope_normalized={self.rope2d_normalized_by_hw}",
            flush=True,
        )
        
        super().__init__()

        
        # --- Start of Configurable Hybrid APE Implementation ---
        pe_config = other_args.get('positional_encoding', {})
        self.pe_strategy = pe_config.get('strategy', 'rpe_only')
        # self.pe_strategy = 'both'
        self.pe_crossover_stage = pe_config.get('crossover_stage', -1)
        # self.pe_crossover_stage = 3

        self.use_ape = self.pe_strategy in ['ape_only', 'both']
        self.use_rpe = self.pe_strategy in ['rpe_only', 'both']

        if self.use_ape:
            print("[ScaleMoGen][Predictor] positional_encoding=ape", flush=True)
            # 1. Learned PE for Joint dimension
            max_joints = pe_config.get('max_joints', 30) # A safe upper limit for joint indices/groups
            self.joint_pos_embed = nn.Embedding(max_joints, self.C)

            # 2. Sinusoidal PE for Time dimension
            max_time_len = pe_config.get('max_time_len', 2048)
            time_pe = torch.zeros(max_time_len, self.C)
            position = torch.arange(0, max_time_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.C, 2).float() * (-math.log(10000.0) / self.C))
            time_pe[:, 0::2] = torch.sin(position * div_term)
            time_pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('time_pos_embed', time_pe.unsqueeze(0))

            # 3. Pre-calculate 2D coordinates for APE lookup
            time_coords, joint_coords = self._build_positional_coordinates()
            self.register_buffer('time_coords', time_coords.unsqueeze(0))
            self.register_buffer('joint_coords', joint_coords.unsqueeze(0))
        
        if not self.use_rpe:
            print("[ScaleMoGen][Predictor] positional_encoding=rpe_disabled", flush=True)
        # --- End of Configurable Hybrid APE Implementation ---

        head_up_method = ''
        word_patch_size = 1 if head_up_method in {'', 'no'} else 2
        if word_patch_size > 1:
            assert all(raw_pn % word_patch_size == 0 for raw_pn in raw_scale_schedule), f'raw_scale_schedule={raw_scale_schedule}, not compatible with word_patch_size={word_patch_size}'
        
        self.checkpointing = checkpointing
        self.pad_to_multiplier = max(1, pad_to_multiplier)
        
        self.raw_scale_schedule = raw_scale_schedule    # 'raw' means before any patchifying
        self.first_l = 1
        # solve top-p top-k sampling hyperparameters
        self.top_p, self.top_k = max(min(top_p, 1), 0), (round(top_k * self.V) if 0 < top_k < 1 else round(top_k))
        if self.top_p < 1e-5: self.top_p = 0
        if self.top_k >= self.V or self.top_k <= 0: self.top_k = 0

        self.rng = torch.Generator(device=dist.get_device())
        self.maybe_record_function = nullcontext
        self.text_maxlen = text_maxlen
        self.t2i = text_channels != 0
        
        # [inp & position embedding]
         #init_std = math.sqrt(1 / self.C / 3)
        init_std = math.sqrt(1 / self.C / (3 * 3)) #reduce init_std less than 0.02 value to match gpt-style initialization.
        self.norm0_cond = nn.Identity()
        if self.t2i:
            self.selecting_idx = None
            self.num_classes = 0
            self.D = self.C
            
            cfg_uncond = torch.empty(self.text_maxlen, self.Ct5)
            rng = torch.Generator(device='cpu')
            rng.manual_seed(0)
            # torch.nn.init.trunc_normal_(cfg_uncond, std=1.2) #, generator=rng)
            torch.nn.init.trunc_normal_(cfg_uncond, std=0.02) #, generator=rng) #reduce init_std 
            cfg_uncond /= self.Ct5 ** 0.5
            if rand_uncond:
                self.register_buffer('cfg_uncond', cfg_uncond)
            else:
                self.cfg_uncond = nn.Parameter(cfg_uncond)
            
            self.text_norm = FastRMSNorm(self.Ct5, elementwise_affine=True, eps=norm_eps)
            self.text_proj_for_sos = TextAttentivePool(self.Ct5, self.D)
            self.text_proj_for_ca = nn.Sequential(
                nn.Linear(self.Ct5, self.D),
                nn.GELU(approximate='tanh'),
                nn.Linear(self.D, self.D),
            )
        else:   # class-label cond
            if selecting_idx is None:
                num_classes = 1000
                print(
                    f"[ScaleMoGen][Predictor] warning=missing_selecting_idx default=1/{num_classes} device={dist.get_device()}",
                    flush=True,
                )
                selecting_idx = torch.full((1, num_classes), fill_value=1/num_classes, dtype=torch.float32, device=dist.get_device())
            self.selecting_idx = selecting_idx
            self.num_classes = selecting_idx.shape[-1]
            self.D = self.C
            self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
            nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        if self.rope2d_each_sa_layer:
            rope2d_freqs_grid = precompute_rope2d_freqs_grid(dim=self.C//self.num_heads, scale_schedule=self.motion_scale_schedule, pad_to_multiplier=self.pad_to_multiplier, rope2d_normalized_by_hw=self.rope2d_normalized_by_hw)
            self.rope2d_freqs_grid = rope2d_freqs_grid
        else:
            raise ValueError(f'self.rope2d_each_sa_layer={self.rope2d_each_sa_layer} not implemented')
        self.lvl_embed = nn.Embedding(15, self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # [input layers] input norm && input embedding
        norm_layer = partial(FastRMSNorm if rms_norm else nn.LayerNorm, eps=norm_eps)
        self.norm0_ve = norm_layer(self.d_vae) if nm0 else nn.Identity()
        self.word_embed = nn.Linear(self.d_vae, self.C)
        
        # [shared adaptive layernorm mapping network]
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()

        # [backbone and head]
        self.batch_size = batch_size

        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # dpr means drop path rate (linearly increasing)
        self.unregistered_blocks = []
        for block_idx in range(depth):
            block = (CrossAttnBlock if self.t2i else SelfAttnBlock)(
                embed_dim=self.C, kv_dim=self.D, cross_attn_layer_scale=cross_attn_layer_scale, cond_dim=self.D, act=True, shared_aln=shared_aln, norm_layer=norm_layer,
                num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[block_idx], tau=tau, cos_attn=cos_attn,
                swiglu=swiglu,
                checkpointing_sa_only=self.checkpointing == 'self-attn',
                pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            )
            self.unregistered_blocks.append(block)
        
        # [head]
        V = self.V
        if head_aln:
            self.head_nm = AdaLNBeforeHead(self.C, self.D, act=True, norm_layer=norm_layer)
            self.head = nn.Linear(self.C, V) if head_depth == 1 else nn.Sequential(nn.Linear(self.C, self.C, bias=True), nn.GELU(approximate='tanh'), nn.Linear(self.C, V))
        
            if head_depth == 1:
                nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
                if self.head.bias is not None:
                    nn.init.zeros_(self.head.bias)
            else:
                nn.init.normal_(self.head[0].weight, mean=0.0, std=0.02)
                nn.init.zeros_(self.head[0].bias)
                nn.init.normal_(self.head[2].weight, mean=0.0, std=0.02)
                nn.init.zeros_(self.head[2].bias)
        else:
            self.head_nm = MultiInpIdentity()
            self.head = nn.Sequential(norm_layer(self.C), nn.Linear(self.C, V)) if head_depth == 1 else nn.Sequential(norm_layer(self.C), nn.Linear(self.C, self.C, bias=True), nn.GELU(approximate='tanh'), nn.Linear(self.C, V))
        
        self.num_block_chunks = block_chunks or 1
        self.num_blocks_in_a_chunk = depth // block_chunks
        print(
            f"[ScaleMoGen][Predictor] blocks_per_chunk={self.num_blocks_in_a_chunk} "
            f"depth={depth} block_chunks={block_chunks}",
            flush=True,
        )
        assert self.num_blocks_in_a_chunk * block_chunks == depth
        if self.num_block_chunks == 1:
            self.blocks = nn.ModuleList(self.unregistered_blocks)
        else:
            self.block_chunks = nn.ModuleList()
            for i in range(self.num_block_chunks):
                self.block_chunks.append(ScaleMoGenBlockGroup(self.unregistered_blocks, self.num_blocks_in_a_chunk, i*self.num_blocks_in_a_chunk))
        print(
            f"[ScaleMoGen][Predictor] attention=standard_pytorch_sdpa embed_dim={embed_dim} "
            f"heads={num_heads} depth={depth} mlp_ratio={mlp_ratio} swiglu={swiglu} "
            f"drop={drop_rate} drop_path={drop_path_rate:g}",
            flush=True,
        )

    def get_logits(self, h: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        """
        :param h: hidden_state, shaped (B or batch_size, L or seq_len, C or hidden_dim)
        :param cond_BD: shaped (B or batch_size, D or cond_dim)
        :param tau: temperature
        :return: logits, shaped (B or batch_size, V or vocabulary_size)
        """
        with torch.amp.autocast('cuda', enabled=False):
            return self.head(self.head_nm(h.float(), cond_BD.float()))

    def add_lvl_embeding(self, feature, scale_ind, scale_schedule, need_to_pad=0):
        bs, seq_len, c = feature.shape
        patch_t, patch_h, patch_w = scale_schedule[scale_ind]
        t_mul_h_mul_w = patch_t * patch_h * patch_w
        
        assert t_mul_h_mul_w + need_to_pad == seq_len
        # print(self.lvl_embed(scale_ind*torch.ones((bs,t_mul_h_mul_w),dtype=torch.int).to(feature.device)))
        feature[:, :t_mul_h_mul_w] += self.lvl_embed(scale_ind*torch.ones((bs,t_mul_h_mul_w),dtype=torch.int).to(feature.device))
        return feature
    
    def add_lvl_embeding_for_x_BLC(self, x_BLC, scale_schedule, need_to_pad=0):
        ptr = 0
        x_BLC_list = []
        for scale_ind, patch_t_h_w in enumerate(scale_schedule):
            scale_seq_len = np.array(patch_t_h_w).prod()
            x_BLC_this_scale = x_BLC[:,ptr:ptr+scale_seq_len] # shape: [bs, patch_h*patch_w, c]
            ptr += scale_seq_len
            x_BLC_this_scale = self.add_lvl_embeding(x_BLC_this_scale, scale_ind, scale_schedule)
            x_BLC_list.append(x_BLC_this_scale)
        assert x_BLC.shape[1] == (ptr + need_to_pad), f'{x_BLC.shape[1]} != {ptr} + {need_to_pad}'
        x_BLC_list.append(x_BLC[:,ptr:])
        x_BLC = torch.cat(x_BLC_list, dim=1)
        return x_BLC

    def forward(self, label_B_or_BLT: Union[torch.LongTensor, Tuple[torch.FloatTensor, torch.IntTensor, int]], x_BLC_wo_prefix: torch.Tensor, scale_schedule: List[Tuple[int]],
        cfg_infer=False, motion_padding_mask=None, cond_embs=None, cond_padding_mask=None, g_it=None,
        resample=False, **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:  # returns logits_BLV
        """
        label_B_or_BLT: label_B or (kv_compact, cu_seqlens_k, max_seqlen_k)
        motion_padding_mask: (B, t_seqlen, 1), all pad positions are TRUE else FALSE
        :return: logits BLV, V is vocab_size
        """
        if cfg_infer:
            return self.autoregressive_infer_cfg(label_B_or_BLT=label_B_or_BLT, scale_schedule=scale_schedule, motion_padding_mask=motion_padding_mask, **kwargs)
        
        x_BLC_wo_prefix = x_BLC_wo_prefix.float()       # input should be float32
        B = x_BLC_wo_prefix.shape[0]

        # Randomly alternate between APE and RPE during training, as per the FlowMDM paper.
        use_ape_this_step = self.use_ape
        use_rpe_this_step = self.use_rpe
        if self.use_ape and self.use_rpe and self.training:
            if random.random() > 0.5:
                use_rpe_this_step = False
            else:
                use_ape_this_step = False

        #l_end = x_BLC_wo_prefix.shape[1] + scale_schedule[0][0] * scale_schedule[0][1]

        # [1. get input sequence x_BLC]
        with torch.amp.autocast('cuda', enabled=False):
            kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
            # drop cond
            total = 0
            for le in lens:
                if random.random() < self.cond_drop_rate:
                    kv_compact[total:total+le] = self.cfg_uncond[:le]
                total += le
            must_on_graph = self.cfg_uncond[0, 0] * 0
            kv_compact = self.text_norm(kv_compact).contiguous()
            sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)).float().contiguous()    # cond_BD should be float32
            kv_compact = self.text_proj_for_ca(kv_compact).contiguous()
            kv_compact[0, 0] += must_on_graph
            ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
            
            cond_BD_or_gss = self.shared_ada_lin(cond_BD).contiguous()  # gss: gamma, scale, shift; cond_BD_or_gss should be float32
            
            sos = sos.unsqueeze(1).expand(B, 1, -1) + self.pos_start.expand(B, 1, -1)  

            if scale_schedule[0][0] * scale_schedule[0][1] * scale_schedule[0][2] > 1:
                sos_pad = torch.zeros(B, scale_schedule[0][0] * scale_schedule[0][1] * scale_schedule[0][2]- 1, sos.shape[-1], device=sos.device, dtype=sos.dtype)
                sos = torch.cat([sos, sos_pad], dim=1)

            x_BLC = torch.cat((sos, self.word_embed(self.norm0_ve(x_BLC_wo_prefix))), dim=1)

            # --- Apply APE if enabled for this step ---
            if use_ape_this_step:
                l_ape = x_BLC.shape[1]
                time_indices = self.time_coords[:, :l_ape]
                joint_indices = self.joint_coords[:, :l_ape]
                
                time_pos_embed = self.time_pos_embed[:, time_indices.squeeze(0), :]
                joint_pos_embed = self.joint_pos_embed(joint_indices)

                x_BLC = x_BLC + time_pos_embed + joint_pos_embed
            # --- End APE Application ---

            # [1.1. pad the seqlen dim]
            l_end = x_BLC.shape[1]
            need_to_pad = (l_end + self.pad_to_multiplier - 1) // self.pad_to_multiplier * self.pad_to_multiplier - l_end # 0
            
            d: torch.Tensor = torch.cat([torch.full((pn[0]*pn[1]*pn[2],), i) for i, pn in enumerate(scale_schedule)]).view(1, l_end, 1)
            dT = d.transpose(1, 2)    # dT: 11L

            # mask_for_resample = torch.rand_like(dT, dtype=float) < self.mask_ratio
            # dT[mask_for_resample] += 1


            attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, l_end, l_end)
            
            attn_bias = attn_bias_for_masking[:, :, :l_end, :l_end].contiguous()   # attn_bias: 11LL
            if need_to_pad:
                attn_bias = F.pad(attn_bias, (0, need_to_pad, 0, need_to_pad), value=-torch.inf)
                attn_bias[0, 0, l_end:, 0] = 0
                x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
            attn_bias_or_two_vector = attn_bias.type_as(x_BLC).to(x_BLC.device)

            #SOS_PAD 
            # attn_bias_or_two_vector[:,:,:(sos_pad.shape[1]+1),1:(sos_pad.shape[1]+1)] = - torch.inf
            if scale_schedule[0][0] * scale_schedule[0][1] * scale_schedule[0][2] > 1:
                attn_bias_or_two_vector[:,:,:,1:(sos_pad.shape[1]+1)] = - torch.inf

        # scale_schedule[0] = (1, 1)
        if motion_padding_mask is not None:
            attn_bias_or_two_vector = attn_bias_or_two_vector.repeat(B, 1, 1, 1)
            motion_mask = motion_padding_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]
            motion_mask = motion_mask.repeat(1, 1, attn_bias_or_two_vector.size(2), 1)  # [B, 1, L, S]
            attn_bias_or_two_vector[motion_mask] = -torch.inf

        # [2. block loop]
        SelfAttnBlock.forward, CrossAttnBlock.forward
        checkpointing_full_block = self.checkpointing == 'full-block' and self.training
        checkpointing_full_block = False # TODO check

        x_BLC_start = x_BLC.clone()

        if self.num_block_chunks == 1:
            for i, b in enumerate(self.blocks):
                if self.add_lvl_embeding_only_first_block and i == 0:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                if not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                
                rope_grid = self.rope2d_freqs_grid if use_rpe_this_step else None
                if checkpointing_full_block:
                    x_BLC = torch.utils.checkpoint.checkpoint(b, x_BLC, cond_BD_or_gss, ca_kv, attn_bias_or_two_vector, scale_schedule, rope_grid, use_reentrant=False)
                else:
                    x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_grid)
                
        else:
            for i, chunk in enumerate(self.block_chunks): # this path
                if self.add_lvl_embeding_only_first_block and i == 0:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                if not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                
                rope_grid = self.rope2d_freqs_grid if use_rpe_this_step else None
                x_BLC = chunk(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, scale_schedule=scale_schedule, checkpointing_full_block=checkpointing_full_block, rope2d_freqs_grid=rope_grid)
        
        logits_1 = self.get_logits(x_BLC[:, :l_end], cond_BD)
        
        if not resample:
            return logits_1
        else:
            dT_resample = dT.clone()
            mask_for_resample = torch.rand_like(dT_resample, dtype=float) < self.mask_ratio
            dT_resample[mask_for_resample] += 1
            attn_bias_for_resampling = torch.where(d >= dT_resample, 0., -torch.inf).reshape(1, 1, l_end, l_end)
            attn_bias_for_resampling = attn_bias_for_resampling[:, :, :l_end, :l_end].contiguous()

            # x_BLC_start[:, ~mask_for_resample[0, 0]] = x_BLC[:, ~mask_for_resample[0, 0]].clone()

            if need_to_pad:
                attn_bias_for_resampling = F.pad(attn_bias_for_resampling, (0, need_to_pad, 0, need_to_pad), value=-torch.inf)
                attn_bias_for_resampling[0, 0, l_end:, 0] = 0

            attn_bias_for_resampling = attn_bias_for_resampling.type_as(x_BLC_start).to(x_BLC_start.device)

            #SOS_PAD 
            # attn_bias_or_two_vector[:,:,:(sos_pad.shape[1]+1),1:(sos_pad.shape[1]+1)] = - torch.inf
            if scale_schedule[0][0] * scale_schedule[0][1] * scale_schedule[0][2] > 1:
                attn_bias_for_resampling[:,:,:,1:(sos_pad.shape[1]+1)] = - torch.inf
            
            if motion_padding_mask is not None:
                attn_bias_for_resampling = attn_bias_for_resampling.repeat(B, 1, 1, 1)
                motion_mask = motion_padding_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]
                motion_mask = motion_mask.repeat(1, 1, attn_bias_for_resampling.size(2), 1)  # [B, 1, L, S]
                attn_bias_for_resampling[motion_mask] = -torch.inf
            
            if self.num_block_chunks == 1:
                for i, b in enumerate(self.blocks):
                    if self.add_lvl_embeding_only_first_block and i == 0:
                        x_BLC_resample = self.add_lvl_embeding_for_x_BLC(x_BLC_start, scale_schedule, need_to_pad)
                    if not self.add_lvl_embeding_only_first_block:
                        x_BLC_resample = self.add_lvl_embeding_for_x_BLC(x_BLC_start, scale_schedule, need_to_pad)
                    
                    rope_grid = self.rope2d_freqs_grid if self.use_rpe else None
                    if checkpointing_full_block:
                        x_BLC_resample = torch.utils.checkpoint.checkpoint(b, x_BLC_resample, cond_BD_or_gss, ca_kv, attn_bias_or_two_vector, scale_schedule, rope_grid, use_reentrant=False)
                    else:
                        x_BLC_resample = b(x=x_BLC_resample, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_grid)
            else:
                for i, chunk in enumerate(self.block_chunks): # this path
                    if self.add_lvl_embeding_only_first_block and i == 0:
                        x_BLC_resample = self.add_lvl_embeding_for_x_BLC(x_BLC_start, scale_schedule, need_to_pad)
                    if not self.add_lvl_embeding_only_first_block:
                        x_BLC_resample = self.add_lvl_embeding_for_x_BLC(x_BLC_start, scale_schedule, need_to_pad)
                    
                    rope_grid = self.rope2d_freqs_grid if self.use_rpe else None
                    x_BLC_resample = chunk(x=x_BLC_resample, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, scale_schedule=scale_schedule, checkpointing_full_block=checkpointing_full_block, rope2d_freqs_grid=rope_grid)

            x_BLC_resample[:, ~mask_for_resample[0, 0]] = x_BLC[:, ~mask_for_resample[0, 0]]
            
            logits_2 = self.get_logits(x_BLC_resample[:, :l_end], cond_BD)

            return logits_1, logits_2, mask_for_resample[0, 0]

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        vae=None,
        coarse_to_fine_chain=None,
        all_mask=None,
        motion_padding_mask=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
        g_seed=None, cfg_list=[], tau_list=[], cfg_sc=3, top_k=0, top_p=0.0,
        returns_vemb=0, ratio_Bl1=None, gumbel=0, norm_cfg=False,
        cfg_exp_k: float=0.0, cfg_insertion_layer=[-5],
        resampling_steps=0, confidence_threshold=0.5, # New parameters for resampling
        vae_type=0, softmax_merge_topk=-1, ret_img=False,
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        inference_mode=False,
        save_img_path=None,
        sampling_per_bits=1,
        input_motions=None,
        m_lengths=None,

        sampling_func=None,
        return_idx_prob=False,
        sampling_device="cuda",
        sampling_method="multinomial",
    ):   # returns List[idx_Bl]
        sample_on_cpu = str(sampling_device).lower() == "cpu"
        if g_seed is None:
            rng = None
        elif sample_on_cpu:
            rng = torch.Generator(device="cpu")
            rng.manual_seed(int(g_seed))
        else:
            self.rng.manual_seed(int(g_seed))
            rng = self.rng
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)

        all_mask = []
        non_pad_mask = []
        for _, (t_scale, spatial_groups) in enumerate(coarse_to_fine_chain):
            ds_mlens = torch.ceil(m_lengths / (t_scale * self.len_scale_factor)).long()
            ds_non_pad_mask = length_to_mask(ds_mlens, self.T // t_scale, device=m_lengths.device)

            ds_non_pad_mask = ds_non_pad_mask.unsqueeze(-1).expand(-1, -1, len(spatial_groups))
            all_mask.append(ds_non_pad_mask)
            non_pad_mask.append(rearrange(ds_non_pad_mask, 'n t j -> n (t j)'))
        
        motion_padding_mask = ~(torch.cat(non_pad_mask, dim=1))

        if input_motions is not None:
            recon_motions, _ = vae.forward(input_motions.float(), m_lengths)
            x_quantized, all_indices, all_bit_indices, all_loss = vae.encode(input_motions.float(), m_lengths)
        else:
            recon_motions = None

        # scale_schedule is used by the predictor; vae_scale_schedule is used by the VQ model when spatial patchify is enabled.
        # we need to convert scale_schedule to vae_scale_schedule by multiply 2 to h and w
        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
        
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        if any(np.array(cfg_list) != 1):
            bs = 2*B
            if not negative_label_B_or_BLT:
                kv_compact_un = kv_compact.clone()
                total = 0
                for le in lens:
                    kv_compact_un[total:total+le] = (self.cfg_uncond)[:le]
                    total += le
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1]), dim=0)
            else:
                kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k_un[1:]+cu_seqlens_k[-1]), dim=0)
                max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
        else:
            bs = B

        kv_compact = self.text_norm(kv_compact)
        sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)) # sos shape: [2, 4096]
        kv_compact = self.text_proj_for_ca(kv_compact) # kv_compact shape: [304, 4096]
        ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
        last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + self.pos_start.expand(bs, 1, -1)

        if scale_schedule[0][0] * scale_schedule[0][1] * scale_schedule[0][2] > 1:
            last_stage_pad = torch.zeros(bs, scale_schedule[0][0] * scale_schedule[0][1] * scale_schedule[0][2]- 1, last_stage.shape[-1], device=last_stage.device, dtype=last_stage.dtype)
            last_stage = torch.cat([last_stage, last_stage_pad], dim=1)

        with torch.amp.autocast('cuda', enabled=False):
            cond_BD_or_gss = self.shared_ada_lin(cond_BD.float()).float().contiguous()
        accu_BChw, cur_L, ret = None, 0, []  # current length, list of reconstructed images
        idx_Bl_list, idx_Bld_list = [], []

        if inference_mode:
            for b in self.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(True)
        
        abs_cfg_insertion_layers = []
        add_cfg_on_logits, add_cfg_on_probs = False, False
        leng = len(self.unregistered_blocks)
        for item in cfg_insertion_layer:
            if item == 0: # add cfg on logits
                add_cfg_on_logits = True
            elif item == 1: # add cfg on probs
                add_cfg_on_probs = True # todo in the future, we may want to add cfg on logits and probs
            elif item < 0: # determine to add cfg at item-th layer's output
                assert leng+item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={self.num_block_chunks}'
                abs_cfg_insertion_layers.append(leng+item)
            else:
                raise ValueError(f'cfg_insertion_layer: {item} is not valid')
        
        num_stages_minus_1 = len(scale_schedule)-1
        summed_codes = 0

        l_end = sum(pn[0] * pn[1] * pn[2] for pn in scale_schedule)
        d: torch.Tensor = torch.cat([torch.full((pn[0]*pn[1]*pn[2],), i) for i, pn in enumerate(scale_schedule)]).view(1, l_end, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, l_end, l_end)
        attn_bias = attn_bias_for_masking[:, :, :l_end, :l_end].contiguous()   # attn_bias: 11LL
        need_to_pad = (l_end + self.pad_to_multiplier - 1) // self.pad_to_multiplier * self.pad_to_multiplier - l_end # 0
            
        if need_to_pad:
            attn_bias = F.pad(attn_bias, (0, need_to_pad, 0, need_to_pad), value=-torch.inf)
            attn_bias[0, 0, l_end:, 0] = 0
            x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
        attn_bias_or_two_vector = attn_bias.type_as(last_stage).to(last_stage.device)
        #attn_bias_or_two_vector[:,:,:(last_stage_pad.shape[1]+1),1:(last_stage_pad.shape[1]+1)] = - torch.inf

        if scale_schedule[0][0] * scale_schedule[0][1] * scale_schedule[0][2] > 1:
            attn_bias_or_two_vector[:,:,:,1:(last_stage_pad.shape[1]+1)] = - torch.inf

        if motion_padding_mask is not None:
            attn_bias_or_two_vector = attn_bias_or_two_vector.repeat(B, 1, 1, 1)
            motion_mask = motion_padding_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]
            motion_mask = motion_mask.repeat(1, 1, attn_bias_or_two_vector.size(2), 1)  # [B, 1, L, S]
            attn_bias_or_two_vector[motion_mask] = -torch.inf
            

        idx_BlV_cache_list, probs_cache_list = [], []


        for si, pn in enumerate(scale_schedule):   # si: i-th segment
            cfg = cfg_list[si]
            if si >= trunk_scale:
                break
            last_L = cur_L
            cur_L += np.array(pn).prod()

            need_to_pad = 0

            ratio = si / num_stages_minus_1
            cfg = cfg * ratio

            # --- Start of Dynamic PE Scheduling for Inference ---
            use_ape_for_this_stage = self.use_ape and (si < self.pe_crossover_stage)
            use_rpe_for_this_stage = self.use_rpe and (si >= self.pe_crossover_stage)

            h = last_stage # h has batch size B or 2*B if CFG is active

            if use_ape_for_this_stage:
                time_indices = self.time_coords[:, last_L:cur_L]
                joint_indices = self.joint_coords[:, last_L:cur_L]
                
                time_pos_embed = self.time_pos_embed[:, time_indices.squeeze(0), :]
                joint_pos_embed = self.joint_pos_embed(joint_indices)
                
                # Repeat for CFG if active
                if h.shape[0] != time_pos_embed.shape[0]:
                    time_pos_embed = time_pos_embed.repeat(h.shape[0], 1, 1)
                    joint_pos_embed = joint_pos_embed.repeat(h.shape[0], 1, 1)

                h = h + time_pos_embed + joint_pos_embed

            rope_grid = self.rope2d_freqs_grid if use_rpe_for_this_stage else None
            # --- End of Dynamic PE Scheduling for Inference ---

            # Run transformer blocks with the causal attention mask
            if self.num_block_chunks == 1:
                for block_idx, b in enumerate(self.blocks):
                    if self.add_lvl_embeding_only_first_block and block_idx == 0:
                        h = self.add_lvl_embeding(h, si, scale_schedule, need_to_pad=need_to_pad)
                    if not self.add_lvl_embeding_only_first_block:
                        h = self.add_lvl_embeding(h, si, scale_schedule, need_to_pad=need_to_pad)
                    h = b(x=h, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector[:,:,last_L:cur_L, :cur_L], scale_schedule=scale_schedule, rope2d_freqs_grid=rope_grid, scale_ind=si)
            else: # num_block_chunks > 1
                for block_idx, chunk in enumerate(self.block_chunks):
                    if self.add_lvl_embeding_only_first_block and block_idx == 0:
                        h = self.add_lvl_embeding(h, si, scale_schedule, need_to_pad=need_to_pad)
                    if not self.add_lvl_embeding_only_first_block:
                        h = self.add_lvl_embeding(h, si, scale_schedule, need_to_pad=need_to_pad)
                    for m in chunk.module:
                        h = m(x=h, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector[:,:,last_L:cur_L, :cur_L], scale_schedule=scale_schedule, rope2d_freqs_grid=rope_grid, scale_ind=si)

            # Apply CFG to initial logits
            if (cfg != 0) and add_cfg_on_logits:
                logits_BlV_raw = self.get_logits(h, cond_BD).mul(1/tau_list[si])
                logits_BlV = (1 + cfg) * logits_BlV_raw[:B] - cfg * logits_BlV_raw[B:]
            else:
                logits_BlV = self.get_logits(h[:B], cond_BD[:B]).mul(1/tau_list[si])

            # Part 2: Initial Sampling & Confidence Calculation (Minimum Bit Confidence)
            # probs_BlV = logits_BlV.softmax(dim=-1)
            
            if self.use_bit_label:
                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]

                logits_BlV_reshaped = logits_BlV.reshape(tmp_bs, -1, 2)
                
                probs_BlV_reshaped = logits_BlV_reshaped.softmax(dim=-1)
                
                # Sample initial bits
                current_idx_Bld, probs = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                    logits_BlV_reshaped,
                    rng=rng,
                    top_k=top_k,
                    top_p=top_p,
                    num_samples=1,
                    sampling_val=sampling_func(logits_BlV_reshaped, si, rng) if sampling_func else None,
                    sample_on_cpu=sample_on_cpu,
                    sampling_method=sampling_method,
                )
                # Gather probabilities of chosen bits
                bit_confidence_scores = torch.gather(probs_BlV_reshaped, -1, current_idx_Bld).view(tmp_bs, tmp_seq_len, -1)
                # The confidence of a token is the confidence of its least-confident bit
                
                token_confidence = bit_confidence_scores.mean(dim=-1)
                current_idx_Bld = current_idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)

                if return_idx_prob:
                    idx_BlV_cache_list.append(current_idx_Bld)
                    probs_cache_list.append(probs)
            else:
                current_idx_Bl, probs = sample_with_top_k_top_p_also_inplace_modifying_logits_(
                    logits_BlV,
                    rng=rng,
                    top_k=top_k,
                    top_p=top_p,
                    num_samples=1,
                    sample_on_cpu=sample_on_cpu,
                    sampling_method=sampling_method,
                )
                current_idx_Bl = current_idx_Bl[:, :, 0]
                probs_BlV = probs[:, :, 0, :]
                # For non-bitwise, the confidence is just the probability of the chosen token.
                token_confidence = torch.gather(probs_BlV, -1, current_idx_Bl.unsqueeze(-1)).squeeze(-1)
                if return_idx_prob:
                    idx_BlV_cache_list.append(current_idx_Bl)
                    probs_cache_list.append(probs)

            # Part 3: Iterative Resampling Loop
            if resampling_steps > 0:
                # Create a non-causal (bidirectional) mask for resampling, respecting only the padding
                resampling_attn_bias = torch.zeros_like(attn_bias_or_two_vector)
                if motion_padding_mask is not None:
                    motion_mask_resample = motion_padding_mask.unsqueeze(1).unsqueeze(1).repeat(1, 1, resampling_attn_bias.size(2), 1)
                    resampling_attn_bias[motion_mask_resample] = -torch.inf
                
            for i in range(resampling_steps):
                if token_confidence.shape[1] < 10:
                    break
                current_stage_padding_mask = motion_padding_mask[:, last_L:cur_L]
                non_padding_mask = ~current_stage_padding_mask

                num_non_padded = non_padding_mask.sum()
                if num_non_padded < 10:
                    break
                # --- FIX: Reset KV cache to the state before this stage's initial pass ---
                for b in self.unregistered_blocks:
                    sa = b.sa if isinstance(b, CrossAttnBlock) else b.attn
                    if sa.caching and sa.cached_k is not None and sa.cached_k.shape[-2] > last_L:
                        sa.cached_k = sa.cached_k[..., :last_L, :]
                        sa.cached_v = sa.cached_v[..., :last_L, :]
                # --- End of FIX ---
                
                # --- NEW: Relative Thresholding using Quantiles ---
                dynamic_threshold = torch.quantile(token_confidence[non_padding_mask], 0.3)
                raw_low_confidence_mask = token_confidence < dynamic_threshold
                final_low_confidence_mask = raw_low_confidence_mask & non_padding_mask

                if not final_low_confidence_mask.any():
                    break

                # --- BAMM-style resampling: Modify Attention Mask --- 
                iter_attn_bias = resampling_attn_bias.clone()
                # Make low-confidence tokens invisible to other tokens by masking their columns.
                # Note: The slicing `last_L:cur_L` correctly maps the local `final_low_confidence_mask` to the global mask positions.
                iter_attn_bias[:, :, :, last_L:cur_L][:, :, :, final_low_confidence_mask.squeeze(0)] = -torch.inf

                # The input is the result of the initial pass, `h`.
                # resampling_input = h.clone()

                # --- FIX: Embed the sampled tokens and mix them into the input ---
                if self.use_bit_label:
                    # current_idx_Bld: [B, L, D_bits]
                    # Reshape to [B, H, W, D_bits] for VAE
                    curr_idx_spatial = current_idx_Bld.reshape(B, pn[1], pn[2], -1)
                    # Get codes: returns [B, C_vae, H, W] (based on usage in one_step_fuse/interpolate)
                    codes_spatial = vae.quantizer2d.lfq.indices_to_codes(curr_idx_spatial.float(), label_type='bit_label')
                    # Rearrange to [B, L, C_vae]
                    codes = rearrange(codes_spatial, 'b c h w -> b (h w) c')
                else:
                    codes = self.quant_only_used_in_inference[0].embedding(current_idx_Bl)

                # Project to Model Dimension
                h_sampled = self.word_embed(self.norm0_ve(codes))

                # Mix: Start with original input (context from previous scale)
                resampling_input = last_stage.clone()
                
                # Identify High Confidence positions (Keep Sampled)
                keep_mask = ~final_low_confidence_mask # [B, L]
                keep_mask_expanded = keep_mask.unsqueeze(-1).expand_as(resampling_input)
                
                # If CFG is active (h has 2*B), we typically only resample the conditional part or both?
                # current_idx_Bld is usually just [B, ...].
                # If logits were [2B, ...], sample usually returns [B, ...] (unconditional dropped or handled).
                # In this code, sample is called on logits_BlV (or reshaped) which is processed.
                # Let's assume current_idx_Bld corresponds to the first B elements (conditional/unconditional depending on logic).
                # But resampling_input might be 2*B.
                
                # Safe assignment: Only update the first B elements (assuming that's what we sampled)
                # or if we sampled for 2B, update all.
                # sample_with_top_k... returns [B, ...].
                # autoregressive_infer_cfg logic: logits_BlV is derived from h.
                # if CFG, h is 2*B. logits_BlV is calculated. 
                # Then: logits_BlV = (1+cfg)*cond - cfg*uncond OR just cond.
                # The logits used for sampling are size [B, ...].
                # So current_idx_Bld is [B, ...].
                
                # So we update resampling_input[:B]
                resampling_input[keep_mask_expanded] += h_sampled.expand_as(resampling_input)[keep_mask_expanded]
                
                # If we need to update the negative/unconditional path (second B), we usually don't have sampled tokens for it 
                # unless we sampled them separately. 
                # For CFG resampling, usually we just keep the unconditional input as 'last_stage' (prior).
                
                # Apply APE if needed
                if use_ape_for_this_stage:
                    # time_pos_embed/joint_pos_embed were already repeated for CFG if needed
                    resampling_input = resampling_input + time_pos_embed + joint_pos_embed

                # Rerun the model with the unmodified input but with the new dynamic attention mask
                resampling_h = resampling_input
                if self.num_block_chunks == 1:
                    for block_idx, b in enumerate(self.blocks):
                        '''
                        if self.add_lvl_embeding_only_first_block and block_idx == 0:
                            resampling_h = self.add_lvl_embeding(resampling_h, si, scale_schedule, need_to_pad=need_to_pad)
                        if not self.add_lvl_embeding_only_first_block:
                            resampling_h = self.add_lvl_embeding(resampling_h, si, scale_schedule, need_to_pad=need_to_pad)
                        '''
                        resampling_h = b(x=resampling_h, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=iter_attn_bias[:,:,last_L:cur_L, :cur_L], scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid, scale_ind=si)
                else: # num_block_chunks > 1
                    for block_idx, chunk in enumerate(self.block_chunks):
                        '''
                        if self.add_lvl_embeding_only_first_block and block_idx == 0:
                            resampling_h = self.add_lvl_embeding(resampling_h, si, scale_schedule, need_to_pad=need_to_pad)
                        if not self.add_lvl_embeding_only_first_block:
                            resampling_h = self.add_lvl_embeding(resampling_h, si, scale_schedule, need_to_pad=need_to_pad)
                        '''
                        for m in chunk.module:
                            resampling_h = m(x=resampling_h, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=iter_attn_bias[:,:,last_L:cur_L, :cur_L], scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid, scale_ind=si)
                
                # Get new logits and apply CFG
                new_logits_BlV_raw = self.get_logits(resampling_h, cond_BD).mul(1 / tau_list[si])
                if (cfg != 0) and add_cfg_on_logits:
                    new_logits_BlV = (1 + cfg) * new_logits_BlV_raw[:B] - cfg * new_logits_BlV_raw[B:]
                else:
                    new_logits_BlV = new_logits_BlV_raw[:B] # Only conditional path if CFG is off

                # Sample *only* for the low-confidence positions
                if self.use_bit_label:
                    new_logits_BlV_reshaped = new_logits_BlV.reshape(tmp_bs, -1, 2)
                    new_probs_reshaped = new_logits_BlV_reshaped.softmax(dim=-1)
                    flat_mask = final_low_confidence_mask.view(-1).unsqueeze(-1).expand(-1, 32).flatten()
                    if flat_mask.any():
                        resample_probs = new_probs_reshaped.view(-1, 2)[flat_mask]
                        if sample_on_cpu:
                            resampled_indices = torch.multinomial(
                                resample_probs.float().cpu(),
                                num_samples=1,
                                generator=rng,
                            ).to(current_idx_Bld.device)
                        else:
                            resampled_indices = torch.multinomial(resample_probs, num_samples=1, generator=rng)
                        current_idx_Bld.view(-1, 1)[flat_mask] = resampled_indices
                    
                    # Update confidence scores for the next iteration
                    bit_confidence_scores = torch.gather(new_probs_reshaped, -1, current_idx_Bld.view(tmp_bs, -1, 1)).view(tmp_bs, tmp_seq_len, -1)
                    new_token_confidence = bit_confidence_scores.mean(dim=-1)
                    token_confidence[final_low_confidence_mask] = new_token_confidence[final_low_confidence_mask]
                else:
                    if final_low_confidence_mask.any():
                        resample_probs = new_probs_BlV[final_low_confidence_mask]
                        if sample_on_cpu:
                            resampled_indices = torch.multinomial(
                                resample_probs.float().cpu(),
                                num_samples=1,
                                generator=rng,
                            ).to(current_idx_Bl.device).squeeze(-1)
                        else:
                            resampled_indices = torch.multinomial(
                                resample_probs,
                                num_samples=1,
                                generator=rng,
                            ).squeeze(-1)
                        current_idx_Bl[final_low_confidence_mask] = resampled_indices

                    # Update confidence scores for the next iteration
                    new_confidence_scores = torch.gather(new_probs_BlV, -1, current_idx_Bl.unsqueeze(-1)).squeeze(-1)
                    token_confidence[final_low_confidence_mask] = new_confidence_scores[final_low_confidence_mask]

            # Final assignment after loop
            if self.use_bit_label:
                idx_Bld = current_idx_Bld
            else:
                idx_Bl = current_idx_Bl
            # --- End of Final Corrected Resampling Logic with CFG ---
            if vae_type != 0:
                assert returns_vemb
                if si < gt_leak:
                    idx_Bld = gt_ls_Bl[si]
                else:
                    assert pn[0] == 1
                    idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1) # shape: [B, h, w, d] or [B, h, w, 4d]
                    if self.apply_spatial_patchify: # unpatchify operation
                        idx_Bld = idx_Bld.permute(0,3,1,2) # [B, 4d, h, w]
                        idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2) # [B, d, 2h, 2w]
                        idx_Bld = idx_Bld.permute(0,2,3,1) # [B, 2h, 2w, d]
                    idx_Bld = idx_Bld.unsqueeze(1) # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]

                if all_mask is not None:
                    mask_expanded = all_mask[si].unsqueeze(1).unsqueeze(-1)  # shape: [1, 1, t, 2, 1]
                    idx_Bld = idx_Bld * mask_expanded

                idx_Bld_list.append(idx_Bld)
                codes = vae.quantizer2d.lfq.indices_to_codes(idx_Bld.float(), label_type='bit_label') # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                
                if all_mask is not None:
                    mask_expanded = all_mask[si].unsqueeze(0).unsqueeze(0)
                    codes = codes * mask_expanded 
                
                _, C, _, _, _ = codes.shape
                _, T, J = scale_schedule[-1]

                up_quantized = torch.zeros((B, C, 1, T, J), device= codes.device, dtype= codes.dtype)
                if si != len(vae_scale_schedule)-1:
                    spatial_groups = coarse_to_fine_chain[si][-1]

                    for group_idx, group in enumerate(spatial_groups):
                        joint_indices = group if isinstance(group, list) else [group]
                        num_joints = len(joint_indices)
                        
                        up_group_feat = F.interpolate(codes[..., group_idx:group_idx+1], size=(1, T, num_joints), mode=vae.quantizer2d.z_interplote_up).contiguous()  # → [N, C, T, num_joints]
                        up_quantized[..., joint_indices] = up_group_feat
                    summed_codes += up_quantized
                    
                    next_t_scale, next_spatial_groups = coarse_to_fine_chain[si+1]
                    if next_t_scale != 1:
                        last_stage = F.interpolate(summed_codes, size=(1, int(T//next_t_scale), summed_codes.shape[-1]), mode=vae.quantizer2d.z_interplote_up).contiguous()  # [N, C, 1, T', J]    
                    else:
                        last_stage = summed_codes
                    
                    spatial_grouped = []
                    for group in next_spatial_groups:
                        group_tensor = last_stage[..., group] if isinstance(group, list) else last_stage[..., [group]]
                        group_avg = group_tensor.mean(dim=-1, keepdim=True)
                        spatial_grouped.append(group_avg)
                    last_stage = torch.cat(spatial_grouped, dim=-1)
                    
                    last_stage = last_stage.squeeze(-3)
                    if self.apply_spatial_patchify: # patchify operation
                        last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2) # [B, 4d, h, w]
                    last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] or [B, 4d, h*w]
                    last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d] or [B, h*w, 4d]

                else:
                    summed_codes += codes

            
                
                #if si != num_stages_minus_1:
                #    summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
                #    last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_up) # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                #    last_stage = last_stage.squeeze(-3) # [B, d, h, w] or [B, d, 2h, 2w]
                #    if self.apply_spatial_patchify: # patchify operation
                #        last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2) # [B, 4d, h, w]
                #    last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] or [B, 4d, h*w]
                #    last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d] or [B, h*w, 4d]
                #else:
                #    summed_codes += codes
            else:
                if si < gt_leak:
                    idx_Bl = gt_ls_Bl[si]
                h_BChw = self.quant_only_used_in_inference[0].embedding(idx_Bl).float()   # BlC

                # h_BChw = h_BChw.float().transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][0], scale_schedule[si][1])
                h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][0], scale_schedule[si][1], scale_schedule[si][2])
                ret.append(h_BChw if returns_vemb != 0 else idx_Bl)
                idx_Bl_list.append(idx_Bl)
                if si != num_stages_minus_1:
                    accu_BChw, last_stage = self.quant_only_used_in_inference[0].one_step_fuse(si, num_stages_minus_1+1, accu_BChw, h_BChw, scale_schedule)
            
            if si != num_stages_minus_1:
                last_stage = self.word_embed(self.norm0_ve(last_stage))
                last_stage = last_stage.repeat(bs//B, 1, 1)

        if inference_mode:
            for b in self.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(False)

        if not ret_img:
            return ret, idx_Bl_list, []

        summed_codes = summed_codes.transpose(1, -1)
        summed_codes = vae.quantizer2d.project_out(summed_codes.float())
        summed_codes = summed_codes.transpose(1, -1)

        if vae_type != 0:
            summed_codes = rearrange(summed_codes.squeeze(-3), 'b d t j -> b t j d')
            img = vae.decode(summed_codes.float(), m_lengths=m_lengths)
        else:
            img = vae.viz_from_ms_h_BChw(ret, scale_schedule=scale_schedule, same_shape=True, last_one=True)
        
        if not return_idx_prob:
            return ret, idx_Bl_list, img, recon_motions
        else:
            return ret, idx_Bl_list, img, recon_motions, idx_BlV_cache_list, probs_cache_list
    
    @for_visualize
    def vis_key_params(self, ep):
        return
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=False, assign=False):
        '''
        if 'gpt_fsdp' in state_dict['trainer'].keys():
            state_dict = state_dict['trainer']['gpt_fsdp'] #TODO check 
        else:
            state_dict = state_dict['trainer']['gpt_wo_ddp'] #TODO check 
        '''
        for k in state_dict:
            if 'cfg_uncond' in k:
                old, new = state_dict[k], self.cfg_uncond.data
                min_tlen = min(old.shape[0], new.shape[0])
                if min_tlen == old.shape[0]:
                    state_dict[k] = torch.cat((old.to(device=new.device, dtype=new.dtype), new[min_tlen:]))
                else:
                    state_dict[k] = old[:min_tlen]
        
        for buf_name in (
            'lvl_1L',
            'attn_bias_for_masking',
            'scalemogen_visible_kvlen',
            'scalemogen_invisible_qlen',
        ):
            state_dict.pop(buf_name, None)
            if hasattr(self, buf_name):
                state_dict[buf_name] = getattr(self, buf_name)

        # Check for missing and unexpected keys
        model_keys = set(self.state_dict().keys())
        ckpt_keys = set(state_dict.keys())

        missing_keys = model_keys - ckpt_keys
        unexpected_keys = ckpt_keys - model_keys

        

        if missing_keys:
            print(f"[ScaleMoGen][Predictor] warning=missing_state_keys keys={sorted(missing_keys)}", flush=True)
        if unexpected_keys:
            print(f"[ScaleMoGen][Predictor] warning=unexpected_state_keys keys={sorted(unexpected_keys)}", flush=True)

        return super().load_state_dict(state_dict=state_dict, strict=strict) #, assign=assign)
    
    def special_init(
        self,
        aln_init: float,
        aln_gamma_init: float,
        scale_head: float,
        scale_proj: int,
    ):
        # init head's norm
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(aln_init)    # there's no gamma for head
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        # init head's proj
        if scale_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(scale_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(scale_head)
                self.head[-1].bias.data.zero_()
        
        depth = len(self.unregistered_blocks)
        for block_idx, sab in enumerate(self.unregistered_blocks):
            sab: Union[SelfAttnBlock, CrossAttnBlock]
            # init proj
            scale = 1 / math.sqrt(2*depth if scale_proj == 1 else 2*(1 + block_idx))
            if scale_proj == 1:
                if self.t2i:
                    sab.sa.proj.weight.data.mul_(scale)
                    sab.ca.proj.weight.data.mul_(scale)
                else:
                    sab.attn.proj.weight.data.mul_(scale)
                sab.ffn.fc2.weight.data.mul_(scale)
            # if sab.using_swiglu:
            #     nn.init.ones_(sab.ffn.fcg.bias)
            #     nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            
            # init ada_lin
            if hasattr(sab, 'ada_lin'):
                lin = sab.ada_lin[-1]
                lin.weight.data[:2*self.C].mul_(aln_gamma_init)     # init gamma
                lin.weight.data[2*self.C:].mul_(aln_init)           # init scale and shift
                if hasattr(lin, 'bias') and lin.bias is not None:
                    lin.bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, :2, :].mul_(aln_gamma_init)  # init gamma
                sab.ada_gss.data[:, :, 2:, :].mul_(aln_init)        # init scale and shift
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate}'

    def _build_positional_coordinates(self):
        all_time_indices = []
        all_joint_indices = []

        for i, (t, h, w) in enumerate(self.motion_scale_schedule):
            # Assumption: `h` is the temporal dimension, `w` is the joint dimension for this level.
            # `t` is always 1 in this project's configuration.
            num_tokens_in_level = h * w
            
            # Time coordinates: Calculate absolute time steps
            # `t_scale` is the downsampling factor for this level.
            t_scale = self.coarse_to_fine_chain[i][0]
            time_indices_level = torch.arange(h, dtype=torch.long) * t_scale
            time_indices_level = time_indices_level.repeat_interleave(w) # Shape: [h*w]
            
            # Joint coordinates: Assume sequential IDs for the `w` groups
            joint_indices_level = torch.arange(w, dtype=torch.long)
            joint_indices_level = joint_indices_level.repeat(h) # Shape: [h*w]
            
            all_time_indices.append(time_indices_level)
            all_joint_indices.append(joint_indices_level)
        
        time_coords = torch.cat(all_time_indices)
        joint_coords = torch.cat(all_joint_indices)
        return time_coords, joint_coords
    
    def get_layer_id_and_scale_exp(self, para_name: str):
        raise NotImplementedError


def sample_with_top_k_top_p_also_inplace_modifying_logits_(
    logits_BlV: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    rng=None,
    num_samples=1,
    sampling_val=None,
    sample_on_cpu=False,
    sampling_method="multinomial",
) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)

    probs = logits_BlV.softmax(dim=-1).view(-1, V)

    if sampling_val is None:
        method = str(sampling_method).lower()
        if method in {"argmax", "greedy"}:
            idx = probs.argmax(dim=-1, keepdim=True).expand(-1, num_samples)
        elif method == "multinomial":
            sampling_probs = probs.float().cpu() if sample_on_cpu else probs
            idx = torch.multinomial(
                sampling_probs,
                num_samples=num_samples,
                replacement=replacement,
                generator=rng,
            ).to(logits_BlV.device)
        else:
            raise ValueError(f"Unsupported sampling_method={sampling_method!r}")
        idx = idx.view(B, l, num_samples)
        probs = rearrange(probs, "(b l s) d -> b l s d", b=B, l=l, s=1)
    else:
        idx, probs = sampling_val

    return idx, probs

def sampling_with_top_k_top_p_also_inplace_modifying_probs_(probs_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = probs_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = probs_BlV < probs_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        probs_BlV.masked_fill_(idx_to_remove, 0)
    if top_p > 0:
        sorted_probs, sorted_idx = probs_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_probs.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        probs_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), 0)
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    probs_BlV = probs_BlV / probs_BlV.sum(-1, keepdims=True)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(probs_BlV.view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)


def get_params_num(d, w, mlp):
    m = round(mlp * w / 256) * 256
    s = d * (w**2 * 8 + w*m * 2)    # sa+ca, mlp
    s += w**2 * 6       # saln
    s += 4096 * w       # pred
    s += 32 * w         # we
    
    Ct5 = 4096
    s += Ct5*w * 4      # T5 attn pool
    s += Ct5*w + w*w    # T5 mlp
    return f'{s/1e9:.2f}B'


TIMM_KEYS = {'img_size', 'pretrained', 'pretrained_cfg', 'pretrained_cfg_overlay', 'global_pool', 'cache_dir'}


def _filtered_model_kwargs(kwargs):
    """Drop timm wrapper kwargs before constructing the predictor."""
    return {k: v for k, v in kwargs.items() if k not in TIMM_KEYS}


def _build_scalemogen_transformer(depth, embed_dim, num_heads, drop_path_rate, block_chunks=None, **kwargs):
    """Build a ScaleMoGen transformer variant for timm registration."""
    block_chunks = kwargs.pop("block_chunks", block_chunks if block_chunks is not None else 1)
    return ScaleMoGenTransformer(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=4,
        block_chunks=block_chunks,
        drop_path_rate=drop_path_rate,
        **_filtered_model_kwargs(kwargs),
    )


@register_model
def scalemogen_2b(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, **kwargs):
    return _build_scalemogen_transformer(depth, embed_dim, num_heads, drop_path_rate, **kwargs)


@register_model
def scalemogen_20b(depth=58, embed_dim=4608, num_heads=4608//128, drop_path_rate=0.25, **kwargs):
    return _build_scalemogen_transformer(depth, embed_dim, num_heads, drop_path_rate, **kwargs)


@register_model
def scalemogen_layer12(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, **kwargs):
    return _build_scalemogen_transformer(depth, embed_dim, num_heads, drop_path_rate, block_chunks=4, **kwargs)


@register_model
def scalemogen_layer16(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, **kwargs):
    return _build_scalemogen_transformer(depth, embed_dim, num_heads, drop_path_rate, **kwargs)


@register_model
def scalemogen_layer24(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, **kwargs):
    return _build_scalemogen_transformer(depth, embed_dim, num_heads, drop_path_rate, **kwargs)


@register_model
def scalemogen_layer32(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, **kwargs):
    return _build_scalemogen_transformer(depth, embed_dim, num_heads, drop_path_rate, **kwargs)


@register_model
def scalemogen_layer40(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, **kwargs):
    return _build_scalemogen_transformer(depth, embed_dim, num_heads, drop_path_rate, **kwargs)


@register_model
def scalemogen_layer48(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, **kwargs):
    return _build_scalemogen_transformer(depth, embed_dim, num_heads, drop_path_rate, **kwargs)
