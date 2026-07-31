"""ScaleMoGen generation helpers."""


def _nested_getattr(obj, path):
    """Return a nested attribute or None when any segment is missing."""
    cur = obj
    for name in path.split("."):
        if not hasattr(cur, name):
            return None
        cur = getattr(cur, name)
    return cur


def coarse_to_fine_chain_from_vq(vq_model):
    """Return the coarse-to-fine chain stored by a loaded VQ tokenizer."""
    for path in ("quantizer2d.coarse_to_fine_chain", "quantizer.coarse_to_fine_chain", "coarse_to_fine_chain"):
        chain = _nested_getattr(vq_model, path)
        if chain is not None:
            return chain
    raise AttributeError("ScaleMoGen VQ does not expose a coarse_to_fine_chain")


def scale_schedule_from_vq(vq_model, max_motion_length=None):
    """Build the predictor scale schedule from a loaded VQ tokenizer."""
    if max_motion_length is None:
        max_motion_length = _nested_getattr(vq_model, "cfg.data.max_motion_length")
    if max_motion_length is None:
        raise AttributeError("max_motion_length is missing; pass it explicitly or keep vq_model.cfg.data.max_motion_length")

    len_scale_factor = (
        _nested_getattr(vq_model, "quantizer2d.len_scale_factor")
        or _nested_getattr(vq_model, "quantizer.len_scale_factor")
        or getattr(vq_model, "len_scale_factor", None)
    )
    if len_scale_factor is None:
        raise AttributeError("ScaleMoGen VQ does not expose len_scale_factor")

    token_length = int(max_motion_length) // int(len_scale_factor)
    schedule = []
    for item in coarse_to_fine_chain_from_vq(vq_model):
        if len(item) == 3:
            schedule.append(tuple(item))
            continue
        temporal_scale, spatial_groups = item
        schedule.append((1, token_length // int(temporal_scale), len(spatial_groups)))
    return schedule
