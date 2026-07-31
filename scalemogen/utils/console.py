"""Console formatting helpers for ScaleMoGen training scripts."""

import datetime
import math
import time
from collections import OrderedDict


METRIC_ALIASES = {
    "loss": "loss",
    "loss_recon": "rec",
    "loss_rec": "rec",
    "loss_vel": "vel",
    "loss_global": "root",
    "loss_pos": "root",
    "loss_fk": "fk",
    "loss_commit": "commit",
    "perpexity": "ppl",
    "perplexity": "ppl",
    "loss_end_vel": "end_vel",
    "loss_end_pos": "end_pos",
    "Lm": "loss",
    "Lt": "loss_tail",
    "Accm": "acc_bit",
    "Acct": "acc_tail",
    "Acc_bit_mean": "acc_bit",
    "Acc_token_mean": "acc_token",
    "grad_norm_t": "grad",
    "tnm": "grad",
    "tlr": "lr",
    "lr": "lr",
}


def as_float(value):
    """Return a scalar float from Python or tensor-like values."""
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def format_seconds(seconds):
    """Format seconds as H:MM:SS."""
    if seconds is None or not math.isfinite(seconds):
        return "?"
    return str(datetime.timedelta(seconds=max(0, int(seconds))))


def format_metrics(metrics, precision=4):
    """Format metric key-value pairs with stable ScaleMoGen names."""
    parts = []
    for key, value in metrics.items():
        if value is None:
            continue
        name = METRIC_ALIASES.get(key, key)
        value = as_float(value)
        if name in {"acc_bit", "acc_token", "acc_tail"}:
            parts.append(f"{name}={value:.2f}")
        elif name == "lr":
            parts.append(f"{name}={value:.2e}")
        elif abs(value) >= 100:
            parts.append(f"{name}={value:.2f}")
        else:
            parts.append(f"{name}={value:.{precision}f}")
    return " ".join(parts)


def print_scalemogen_progress(
    component,
    phase,
    metrics=None,
    epoch=None,
    max_epoch=None,
    step=None,
    total_steps=None,
    inner_iter=None,
    elapsed=None,
    eta=None,
    extra=None,
):
    """Print a compact ScaleMoGen training or validation line."""
    fields = [f"[ScaleMoGen][{component}][{phase}]"]
    if epoch is not None:
        if max_epoch is not None:
            fields.append(f"ep={int(epoch):04d}/{int(max_epoch):04d}")
        else:
            fields.append(f"ep={int(epoch):04d}")
    if step is not None:
        if total_steps is not None:
            fields.append(f"it={int(step):07d}/{int(total_steps):07d}")
        else:
            fields.append(f"it={int(step):07d}")
    if inner_iter is not None:
        fields.append(f"inner={int(inner_iter):04d}")
    if elapsed is not None:
        fields.append(f"elapsed={format_seconds(elapsed)}")
    if eta is not None:
        fields.append(f"eta={format_seconds(eta)}")
    if extra:
        fields.append(str(extra))
    if metrics:
        if not isinstance(metrics, OrderedDict):
            metrics = OrderedDict(metrics)
        fields.append(format_metrics(metrics))
    print(" ".join(fields), flush=True)


def elapsed_and_eta(start_time, step, total_steps):
    """Return elapsed and ETA seconds for progress reporting."""
    elapsed = time.time() - start_time
    if not step or not total_steps:
        return elapsed, None
    progress = min(max(step / total_steps, 1e-8), 1.0)
    eta = elapsed / progress - elapsed
    return elapsed, eta
