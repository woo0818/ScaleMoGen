"""Runtime controls for reproducible ScaleMoGen evaluation."""

import torch


def configure_reproducible_eval_runtime():
    """Disable GPU fast paths that can change fp32 eval numerics across devices."""
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
