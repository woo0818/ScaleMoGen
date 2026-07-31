"""Configuration defaults for active ScaleMoGen training and evaluation."""

from copy import deepcopy

from config.load_config import AttributeDict


PREDICTOR_DEFAULTS = {
    # Model architecture.
    "model": "scalemogen_layer12",
    "mask": False,
    "Ct5": 2048,
    "tlen": 512,
    "norm_eps": 1e-6,
    "rms": False,
    "saln": 1,
    "haln": True,
    "cfg": 0.1,
    "rand_uncond": False,
    "drop": 0.0,
    "ca_gamma": -1,
    "nm0": False,
    "tau": 1,
    "cos": 1,
    "swi": False,
    "scale_schedule": None,
    "dec": 1,
    "tp": 0.0,
    "tk": 0.0,
    "enable_checkpointing": "full-block",
    "pad_to_multiplier": 1,
    "add_lvl_embeding_only_first_block": 1,
    "use_bit_label": 1,
    "rope2d_each_sa_layer": 1,
    "rope2d_normalized_by_hw": 2,
    "pn": "0.06M",
    "always_training_scales": 100,
    "apply_spatial_patchify": 0,
    # Training loop.
    "ep": 100,
    "lbs": 16,
    "workers": 8,
    "ac": 1,
    "opt": "adamw",
    "ada": "0.9_0.97",
    "sche": "lin0",
    "fp16": 0,
    "use_bf16_eval": False,
    "tfast": 0,
    "tclip": 5,
    "stable": False,
    "bitloss_type": "mean",
    "use_fsdp_model_ema": 0,
    "zero": 0,
    "enable_hybrid_shard": False,
    "dbg": False,
    "nowd": 1,
    "online_t5": 1,
    "t5_path": "google/flan-t5-xl",
    "text_encoder_dtype": "fp32",
    "text_encoder_device": None,
    # Initialization.
    "tini": -1,
    "aln": 1e-3,
    "alng": 5e-6,
    "hd0": 0.02,
    "diva": 1,
    # Optimizer and scheduler.
    "gpt_training": True,
    "gblr": 1e-4,
    "dblr": None,
    "tblr": 6e-3,
    "gwd": 0.005,
    "dwd": 0.0005,
    "twd": 0.005,
    "gwde": 0,
    "dwde": 0,
    "twde": 0,
    "wp": 1e-8,
    "wp0": 0.005,
    "wpe": 1,
    "oeps": 0.0,
    "cdec": False,
    # Losses and trainer options.
    "vae_type": 32,
    "reweight_loss_by_scale": 1,
    "tema": 0,
    "ls": 0.0,
    "lz": 0.0,
    "eq": 0,
    "xen": False,
    "resos": None,
    # Checkpointing and logging.
    "auto_resume": False,
    "save_model_iters_freq": 1000,
    "viz_every_n_steps": 5000,
    "prof_freq": 50,
    "log_freq": 50,
    "log_every_iter": False,
    "diffs": "",
}


VQ_DEFAULTS = {
    "quantizer": {
        "mu": 0.99,
        "temp_scales": [80, 40, 20, 10, 5, 2, 1],
        "start_drop": 1,
        "quantize_dropout_prob": 0.5,
        "bsq_dim": 24,
        "masking": False,
        "ctf_mapping_id": 3,
        "entropy_loss_weight": 0.1,
        "diversity_gamma": 1.0,
        "commitment_loss_weight": 0.25,
        "new_quant": True,
        "use_decay_factor": False,
        "use_stochastic_depth": True,
        "drop_rate": 0.5,
        "schedule_mode": "dense",
        "keep_first_quant": False,
        "keep_last_quant": True,
        "remove_residual_detach": True,
        "use_out_phi": False,
        "use_out_phi_res": False,
        "random_flip": False,
        "flip_prob": 0.5,
        "flip_mode": "stochastic",
        "max_flip_lvl": 1,
        "random_flip_1lvl": False,
        "flip_lvl_idx": 0,
        "drop_when_test": False,
        "drop_lvl_idx": None,
        "drop_lvl_num": 0,
        "random_short_schedule": False,
        "short_schedule_prob": 0.5,
        "disable_flip_prob": 0.0,
        "zeta": 1.0,
        "gamma": 1.0,
        "uniform_short_schedule": False,
    },
    "model": {
        "latent_dim": 384,
        "kernel_size": 3,
        "n_layers": 2,
        "n_extra_layers": 1,
        "norm": "none",
        "activation": "gelu",
        "dropout": 0.1,
    },
    "training": {
        "batch_size": 32,
        "max_epoch": 500,
        "weight_decay": 0.0,
        "lr": 2.0e-4,
        "milestones": [150_000, 250_000],
        "gamma": 0.3,
        "warm_up_iter": 2000,
        "log_every": 100,
        "save_latest": 500,
        "eval_every_e": 1,
        "recons_loss": "l1_smooth",
        "ema": True,
        "lambda_commit": 0.02,
        "lambda_global": 0.5,
        "lambda_vel": 0.5,
        "lambda_fk": 0.0,
        "lambda_perplexity": 0.02,
        "lambda_recon": 1.0,
        "lambda_pos": 0.5,
        "lambda_kl": 0.02,
    },
}


def _to_attribute_dict(value):
    """Convert nested dictionaries into AttributeDict values."""
    if isinstance(value, AttributeDict):
        return value
    if isinstance(value, dict):
        return AttributeDict({k: _to_attribute_dict(v) for k, v in value.items()})
    return value


def _merge_missing(target, defaults):
    """Fill missing keys in an AttributeDict without overriding explicit config values."""
    for key, default_value in defaults.items():
        if key not in target:
            target[key] = _to_attribute_dict(deepcopy(default_value))
        elif isinstance(target[key], dict) and isinstance(default_value, dict):
            _merge_missing(target[key], default_value)


def apply_predictor_defaults(cfg):
    """Attach active ScaleMoGen predictor defaults to a loaded config."""
    if "predictor" in cfg:
        _merge_missing(cfg.predictor, PREDICTOR_DEFAULTS)
    return cfg


def apply_vq_defaults(cfg):
    """Attach active ScaleMoGen VQ defaults to a loaded config."""
    _merge_missing(cfg, VQ_DEFAULTS)
    return cfg
