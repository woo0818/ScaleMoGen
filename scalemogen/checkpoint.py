"""ScaleMoGen checkpoint and model loading utilities.

This module loads the ScaleMoGen VQ tokenizer and autoregressive predictor.
"""

import os
from glob import glob
from os.path import join as pjoin

import torch
from timm.models import create_model

from config.load_config import load_config
from scalemogen.config import apply_predictor_defaults, apply_vq_defaults
from scalemogen.models.vq import ScaleMoGenVQ

# Import predictor module so timm model names are registered before create_model.
import scalemogen.models.predictor.motion_transformer  # noqa: F401

VQ_CONFIG_NAMES = {
    "humanml3d": [
        "train_vq_hml.yaml",
        "train_vq.yaml",
    ],
    "default": [
        "train_vq.yaml",
        "train_vq_hml.yaml",
    ],
}

PREDICTOR_CONFIG_NAMES = {
    "humanml3d": [
        "train_scalemogen_predictor_hml.yaml",
        "train_scalemogen_predictor.yaml",
    ],
    "default": [
        "train_scalemogen_predictor.yaml",
        "train_scalemogen_predictor_hml.yaml",
    ],
}


def first_existing(paths):
    """Return the first existing path from a list of candidate paths."""
    for path in paths:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError("None of these paths exist: " + ", ".join(paths))


def unique_paths(paths):
    """Return paths in order while removing duplicates."""
    seen = set()
    unique = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _checkpoint_state_dict(ckpt, candidates):
    """Extract a model state dict from supported checkpoint layouts."""
    for key in candidates:
        if key in ckpt:
            return ckpt[key]
    return ckpt


def _predictor_state_dict(ckpt):
    """Extract the ScaleMoGen predictor state dict from a checkpoint."""
    if "trainer" in ckpt:
        return ckpt["trainer"]["gpt_wo_ddp"]
    return _checkpoint_state_dict(ckpt, ("gpt_wo_ddp", "model"))


def load_state_dict_checked(module, state_dict, label):
    """Load a state dict and raise on missing or unexpected keys."""
    load_result = module.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"{label} checkpoint mismatch: missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    return load_result


def predictor_settings(cfg):
    """Return the predictor section from a ScaleMoGen config."""
    apply_predictor_defaults(cfg)
    return cfg.predictor


def predictor_model_name(cfg):
    """Return the registered ScaleMoGen predictor model name."""
    name = predictor_settings(cfg).model
    if name.rsplit("c", maxsplit=1)[-1].isdecimal():
        name, _ = name.rsplit("c", maxsplit=1)
    return name


def is_scalemogen_vq_config(vq_cfg):
    """Return whether a VQ config belongs to the ScaleMoGen tokenizer family."""
    return "scalemogen" in vq_cfg.exp.name


def build_scalemogen_vq(vq_cfg):
    """Construct the ScaleMoGen VQ tokenizer from a training config."""
    apply_vq_defaults(vq_cfg)
    if not is_scalemogen_vq_config(vq_cfg):
        raise NotImplementedError("ScaleMoGen currently supports BSQ VQ checkpoints")
    return ScaleMoGenVQ(vq_cfg)


def vq_config_path(cfg):
    """Resolve the saved VQ training config for the current dataset."""
    names = VQ_CONFIG_NAMES["humanml3d" if cfg.data.name == "humanml3d" else "default"]
    config_dir = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, "vq", cfg.vq_name)
    candidates = [pjoin(config_dir, name) for name in names]
    candidates.extend(sorted(glob(pjoin(config_dir, "train_vq*.yaml"))))
    return first_existing(unique_paths(candidates))


def predictor_train_config_path(cfg):
    """Resolve the saved predictor training config for the current dataset."""
    names = PREDICTOR_CONFIG_NAMES["humanml3d" if cfg.data.name == "humanml3d" else "default"]
    return first_existing(
        [pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, "predictor", cfg.exp.name, name) for name in names]
    )


def load_scalemogen_vq(cfg, device):
    """Load the ScaleMoGen VQ tokenizer and its saved config."""
    cfg_path = vq_config_path(cfg)
    vq_cfg = load_config(cfg_path)
    model = build_scalemogen_vq(vq_cfg)
    ckpt_path = pjoin(vq_cfg.exp.root_ckpt_dir, cfg.data.name, "vq", cfg.vq_name, "model", cfg.vq_ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(_checkpoint_state_dict(ckpt, ("model", "vq_model")))
    model.to(device).eval()
    return model, vq_cfg, cfg_path, ckpt_path


def predictor_kwargs(train_cfg, vq_model, batch_size):
    """Build constructor kwargs for the ScaleMoGen predictor."""
    predictor_cfg = predictor_settings(train_cfg)
    return dict(
        pretrained=False,
        global_pool="",
        text_channels=predictor_cfg.Ct5,
        text_maxlen=predictor_cfg.tlen,
        norm_eps=predictor_cfg.norm_eps,
        rms_norm=predictor_cfg.rms,
        shared_aln=predictor_cfg.saln,
        head_aln=predictor_cfg.haln,
        cond_drop_rate=predictor_cfg.cfg,
        rand_uncond=predictor_cfg.rand_uncond,
        drop_rate=predictor_cfg.drop,
        cross_attn_layer_scale=predictor_cfg.ca_gamma,
        nm0=predictor_cfg.nm0,
        tau=predictor_cfg.tau,
        cos_attn=predictor_cfg.cos,
        swiglu=predictor_cfg.swi,
        raw_scale_schedule=predictor_cfg.scale_schedule,
        head_depth=predictor_cfg.dec,
        top_p=predictor_cfg.tp,
        top_k=predictor_cfg.tk,
        checkpointing=predictor_cfg.enable_checkpointing,
        pad_to_multiplier=predictor_cfg.pad_to_multiplier,
        batch_size=batch_size,
        add_lvl_embeding_only_first_block=predictor_cfg.add_lvl_embeding_only_first_block,
        use_bit_label=predictor_cfg.use_bit_label,
        rope2d_each_sa_layer=predictor_cfg.rope2d_each_sa_layer,
        rope2d_normalized_by_hw=predictor_cfg.rope2d_normalized_by_hw,
        pn=predictor_cfg.pn,
        always_training_scales=predictor_cfg.always_training_scales,
        apply_spatial_patchify=predictor_cfg.apply_spatial_patchify,
        vae_local=vq_model,
    )


def disable_builtin_initializers():
    """Disable default Linear and LayerNorm parameter initialization."""
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def create_scalemogen_predictor(train_cfg, vq_model, batch_size):
    """Create a ScaleMoGen predictor from a training config and tokenizer."""
    return create_model(
        predictor_model_name(train_cfg),
        **predictor_kwargs(train_cfg, vq_model, batch_size),
    )


def load_scalemogen_predictor(cfg, vq_model, device, batch_size=None, disable_init=False):
    """Load the ScaleMoGen autoregressive predictor and its saved config."""
    train_cfg_path = predictor_train_config_path(cfg)
    train_cfg = load_config(train_cfg_path)
    if disable_init:
        disable_builtin_initializers()
    model = create_scalemogen_predictor(train_cfg, vq_model, batch_size or cfg.eval.batch_size)

    ckpt_path = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, "predictor", cfg.exp.name, "model", cfg.eval.model_ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    load_state_dict_checked(model, _predictor_state_dict(ckpt), "ScaleMoGen predictor")
    model.to(device).eval()
    model.rng = torch.Generator(device=device)
    return model, train_cfg, train_cfg_path, ckpt_path


def config_linkage_rows(cfg, vq_cfg, train_cfg):
    """Return common config fields used to verify checkpoint provenance."""
    return [
        ("data.name", cfg.data.name, train_cfg.data.name),
        ("vq_name", cfg.vq_name, getattr(train_cfg, "vq_name", None)),
        ("vq_ckpt", cfg.vq_ckpt, getattr(train_cfg, "vq_ckpt", None)),
        ("joint_num", cfg.data.joint_num, train_cfg.data.joint_num),
        ("dim_pose", cfg.data.dim_pose, train_cfg.data.dim_pose),
        ("max_motion_length", cfg.data.max_motion_length, train_cfg.data.max_motion_length),
        ("vq.bsq_dim", vq_cfg.quantizer.bsq_dim, vq_cfg.quantizer.bsq_dim),
        ("vq.temp_scales", vq_cfg.quantizer.temp_scales, vq_cfg.quantizer.temp_scales),
        ("vq.ctf_mapping_id", vq_cfg.quantizer.ctf_mapping_id, vq_cfg.quantizer.ctf_mapping_id),
    ]
