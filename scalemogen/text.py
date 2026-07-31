"""Text encoder loading helpers for ScaleMoGen.

This module loads the T5 text encoder used by the ScaleMoGen predictor.
"""

import torch
from transformers import AutoTokenizer, T5EncoderModel

from scalemogen.checkpoint import predictor_settings


def resolve_torch_dtype(dtype_name):
    """Resolve a config dtype name into a torch dtype."""
    name = str(dtype_name or "fp32").lower()
    if name in {"fp32", "float32", "full", "none"}:
        return torch.float32
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported text encoder dtype: {dtype_name!r}")


def load_text_encoder(train_cfg, device, dtype_name=None, encoder_device=None):
    """Load the tokenizer and T5 encoder used by the predictor checkpoint."""
    predictor_cfg = predictor_settings(train_cfg)
    tokenizer = AutoTokenizer.from_pretrained(predictor_cfg.t5_path, revision=None, legacy=True)
    tokenizer.model_max_length = predictor_cfg.tlen

    dtype = resolve_torch_dtype(dtype_name or getattr(predictor_cfg, "text_encoder_dtype", "fp32"))
    target_device = torch.device(encoder_device or device)
    encoder = T5EncoderModel.from_pretrained(predictor_cfg.t5_path, torch_dtype=dtype)
    encoder.to(target_device).eval()
    encoder.requires_grad_(False)
    for param in encoder.parameters():
        param.requires_grad_(False)
    return tokenizer, encoder
