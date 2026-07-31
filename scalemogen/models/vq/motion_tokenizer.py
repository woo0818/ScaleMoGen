"""ScaleMoGen BSQ motion tokenizer.

Code provenance: adapted from the SALAD skeleton VAE implementation for motion
tokenization, with the active ScaleMoGen BSQ quantizer retained.
Source repository: https://github.com/seokhyeonhong/salad
"""

import torch
import torch.nn as nn
from einops import rearrange

from scalemogen.models.vq.bsq_quantizer import ScaleMoGenBSQ
from scalemogen.models.vq.encdec import MotionDecoder, MotionEncoder, STConvDecoder, STConvEncoder


def length_to_mask(length, max_len, device: torch.device = None) -> torch.Tensor:
    """Create a boolean sequence mask from motion lengths."""
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length)

    length = length.to(device)
    mask = torch.arange(max_len, device=device).expand(len(length), max_len).to(device)
    return mask < length.unsqueeze(1)


class ScaleMoGenVQ(nn.Module):
    """Motion tokenizer using skeleton encoders and multi-scale BSQ quantization."""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.lfq_weight = 4.0
        self.len_scale_factor = 4

        self.motion_enc = MotionEncoder(cfg)
        self.motion_dec = MotionDecoder(cfg)
        self.conv_enc = STConvEncoder(cfg)
        self.conv_dec = STConvDecoder(cfg, self.conv_enc)

        self.quantizer2d = ScaleMoGenBSQ(
            dim=cfg.model.latent_dim,
            codebook_dim=cfg.quantizer.bsq_dim,
            temp_scales=cfg.quantizer.temp_scales,
            ctf_mapping_id=cfg.quantizer.ctf_mapping_id,
            entropy_loss_weight=cfg.quantizer.entropy_loss_weight,
            diversity_gamma=cfg.quantizer.diversity_gamma,
            commitment_loss_weight=cfg.quantizer.commitment_loss_weight,
            new_quant=cfg.quantizer.new_quant,
            use_decay_factor=cfg.quantizer.use_decay_factor,
            use_stochastic_depth=cfg.quantizer.use_stochastic_depth,
            drop_rate=cfg.quantizer.drop_rate,
            schedule_mode=cfg.quantizer.schedule_mode,
            keep_first_quant=cfg.quantizer.keep_first_quant,
            keep_last_quant=cfg.quantizer.keep_last_quant,
            remove_residual_detach=cfg.quantizer.remove_residual_detach,
            use_out_phi=cfg.quantizer.use_out_phi,
            use_out_phi_res=cfg.quantizer.use_out_phi_res,
            random_flip=cfg.quantizer.random_flip,
            flip_prob=cfg.quantizer.flip_prob,
            flip_mode=cfg.quantizer.flip_mode,
            max_flip_lvl=cfg.quantizer.max_flip_lvl,
            random_flip_1lvl=cfg.quantizer.random_flip_1lvl,
            flip_lvl_idx=cfg.quantizer.flip_lvl_idx,
            drop_when_test=cfg.quantizer.drop_when_test,
            drop_lvl_idx=cfg.quantizer.drop_lvl_idx,
            drop_lvl_num=cfg.quantizer.drop_lvl_num,
            random_short_schedule=cfg.quantizer.random_short_schedule,
            short_schedule_prob=cfg.quantizer.short_schedule_prob,
            disable_flip_prob=cfg.quantizer.disable_flip_prob,
            zeta=cfg.quantizer.zeta,
            gamma=cfg.quantizer.gamma,
            uniform_short_schedule=cfg.quantizer.uniform_short_schedule,
        )

    def freeze(self):
        """Switch to eval mode and freeze all parameters."""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode_for_raw_features(self, x):
        """Encode motion features before BSQ quantization."""
        x = x.detach().float()
        x_encode = self.motion_enc(x)
        x_encode = self.conv_enc(x_encode)
        return rearrange(x_encode, "b t j d -> b d t j")

    def encode(self, x, m_lengths=None):
        """Encode motions into BSQ quantized features and token indices."""
        x_encode = self.encode_for_raw_features(x).float()
        x_quantized, all_indices, all_bit_indices, all_loss = self.quantizer2d(
            x_encode,
            temperature=0.5,
            m_lens=m_lengths,
        )
        x_quantized = rearrange(x_quantized, "b d t j -> b t j d")
        return x_quantized, all_indices, all_bit_indices, all_loss

    def decode(self, x, m_lengths=None):
        """Decode quantized features back into motion features."""
        x_decoded = x.clone()
        if m_lengths is not None:
            mask = length_to_mask(
                m_lengths // self.len_scale_factor,
                x_decoded.shape[1],
                device=x_decoded.device,
            )
            x_decoded = x_decoded * mask.unsqueeze(-1).unsqueeze(-1)

        x_decoded = self.conv_dec(x_decoded)
        return self.motion_dec(x_decoded)

    def forward(self, x, m_lengths=None):
        """Reconstruct motions and return the BSQ training losses."""
        x_quantized, _, _, all_loss = self.encode(x, m_lengths)
        commit_loss = torch.mean(all_loss) * self.lfq_weight
        loss_dict = {
            "loss_commit": commit_loss,
            "perpexity": commit_loss,
        }
        return self.decode(x_quantized, m_lengths), loss_dict
