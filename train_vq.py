"""ScaleMoGen VQ tokenizer training entrypoint."""

import argparse
import os
import shutil
from glob import glob
from os.path import join as pjoin

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from torch.utils.data import DataLoader

from config.load_config import load_config
from scalemogen.checkpoint import build_scalemogen_vq
from scalemogen.config import apply_vq_defaults
from scalemogen.dataset_dispatch import build_vq_train_data
from scalemogen.training import ScaleMoGenVQTrainer
from utils.fixseeds import fixseed


def parse_args():
    """Parse VQ training arguments."""
    parser = argparse.ArgumentParser(description="Train the ScaleMoGen VQ tokenizer.")
    parser.add_argument("--config", default="config/train_vq.yaml")
    return parser.parse_args()


def load_vq_model(vq_cfg, device):
    """Build the ScaleMoGen VQ model from config."""
    vq_model = build_scalemogen_vq(vq_cfg)
    vq_model.to(device)
    return vq_model, vq_cfg


def main():
    """Train a ScaleMoGen VQ tokenizer for the dataset specified in config."""
    args = parse_args()
    config_name = os.path.basename(args.config)
    cfg = apply_vq_defaults(load_config(args.config))
    cfg.exp.checkpoint_dir = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, "vq", cfg.vq_name)

    if cfg.exp.is_continue:
        saved_config_path = None
        candidate_paths = [
            pjoin(cfg.exp.checkpoint_dir, name)
            for name in (config_name, "train_vq.yaml", "train_vq_hml.yaml")
        ]
        candidate_paths.extend(sorted(glob(pjoin(cfg.exp.checkpoint_dir, "train_vq*.yaml"))))
        for candidate_path in candidate_paths:
            if os.path.exists(candidate_path):
                saved_config_path = candidate_path
                break
        if saved_config_path is None:
            raise FileNotFoundError(f"No saved VQ config found in {cfg.exp.checkpoint_dir}")
        n_cfg = apply_vq_defaults(load_config(saved_config_path))
        n_cfg.exp.is_continue = True
        n_cfg.exp.device = cfg.exp.device
        n_cfg.exp.checkpoint_dir = cfg.exp.checkpoint_dir
        cfg = n_cfg
    else:
        os.makedirs(cfg.exp.checkpoint_dir, exist_ok=True)
        shutil.copy(args.config, cfg.exp.checkpoint_dir)

    fixseed(cfg.exp.seed)
    if cfg.exp.device != "cpu":
        torch.cuda.set_device(cfg.exp.device)
    torch.autograd.set_detect_anomaly(True)
    device = torch.device(cfg.exp.device)

    cfg.exp.model_dir = pjoin(cfg.exp.checkpoint_dir, "model")
    cfg.exp.eval_dir = pjoin(cfg.exp.checkpoint_dir, "animation")
    cfg.exp.log_dir = pjoin(cfg.exp.root_log_dir, cfg.data.name, "vq", cfg.exp.name)
    os.makedirs(cfg.exp.model_dir, exist_ok=True)
    os.makedirs(cfg.exp.eval_dir, exist_ok=True)
    os.makedirs(cfg.exp.log_dir, exist_ok=True)

    vq_model, vq_cfg = load_vq_model(cfg, device=device)
    cfg.vq = vq_cfg.quantizer

    params = sum(param.numel() for param in vq_model.parameters())
    print(
        f"[ScaleMoGen][VQ] dataset={cfg.data.name} exp={cfg.exp.name} "
        f"params={params / 1_000_000:.2f}M device={device}",
        flush=True,
    )

    data_bundle = build_vq_train_data(cfg, device)
    train_loader = DataLoader(
        data_bundle.train_dataset,
        batch_size=cfg.training.batch_size,
        drop_last=True,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        data_bundle.val_dataset,
        batch_size=cfg.training.batch_size,
        drop_last=True,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
    )

    trainer = ScaleMoGenVQTrainer(cfg, vq_model, device=device)
    trainer.train(
        train_loader,
        val_loader,
        data_bundle.eval_loader,
        data_bundle.eval_wrapper,
        data_bundle.plot_t2m,
        data_bundle.forward_kinematic_func,
    )


if __name__ == "__main__":
    main()
