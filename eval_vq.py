"""ScaleMoGen VQ reconstruction evaluation entrypoint."""

import argparse
import os
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.skeleton import Skeleton
from config.load_config import load_config
from dataset.dataset import TextMotionDataset
from model.evaluator.evaluator_wrapper import EvaluatorWrapper
from model.evaluator.hml.dataset_motion_loader import get_dataset_motion_loader
from model.evaluator.hml.t2m_eval_wrapper import EvaluatorModelWrapper
from scalemogen.checkpoint import load_scalemogen_vq
from scalemogen.dataset_dispatch import snapmogen_evaluator_config_path
from scalemogen.runtime import configure_reproducible_eval_runtime
from utils import bvh_io
from utils.eval_t2m import evaluation_vqvae, evaluation_vqvae_hml
from utils.fixseeds import fixseed
from utils.get_opt import get_opt
from utils.motion_process_bvh import recover_pos_from_rot


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ScaleMoGen VQ reconstruction quality.")
    parser.add_argument("--config", default="config/eval_vq.yaml")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--vq_ckpt", default=None, help="Override cfg.vq_ckpt.")
    parser.add_argument("--vq_name", default=None, help="Override cfg.vq_name.")
    return parser.parse_args()


def _apply_overrides(cfg, args):
    """Apply CLI overrides that are useful for checkpoint diagnosis."""
    if args.vq_ckpt:
        cfg.vq_ckpt = args.vq_ckpt
    if args.vq_name:
        cfg.vq_name = args.vq_name
    return cfg


def _snapmogen_loader(cfg, split):
    """Create the deterministic SnapMoGen text-motion loader used for VQ eval."""
    cfg.data.deterministic_eval = cfg.eval.get("deterministic_data", True)
    cfg.data.feat_dir = pjoin(cfg.data.root_dir, "renamed_feats")
    meta_dir = pjoin(cfg.data.root_dir, "meta_data")
    split_dir = pjoin(cfg.data.root_dir, "data_split_info1")
    all_caption_path = pjoin(cfg.data.root_dir, "all_caption_clean.json")

    mean = np.load(pjoin(meta_dir, "mean.npy"))
    std = np.load(pjoin(meta_dir, "std.npy"))
    mid_split_file = pjoin(split_dir, f"{split}_fnames.txt")
    cid_split_file = pjoin(split_dir, f"{split}_ids.txt")
    dataset = TextMotionDataset(cfg, mean, std, mid_split_file, cid_split_file, all_caption_path)

    generator = torch.Generator()
    generator.manual_seed(int(cfg.exp.seed))
    loader = DataLoader(
        dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=bool(cfg.eval.get("shuffle", False)),
        num_workers=int(cfg.eval.get("num_workers", 0)),
        pin_memory=True,
        generator=generator,
    )
    return loader, dataset


def _eval_snapmogen(cfg, vq_model, device, split):
    """Run SnapMoGen VQ reconstruction metrics."""
    loader, dataset = _snapmogen_loader(cfg, split)
    template_anim = bvh_io.load(pjoin(cfg.data.root_dir, "renamed_bvhs", "m_ep2_00086.bvh"))
    skeleton = Skeleton(template_anim.offsets, template_anim.parents, device=device)

    def forward_kinematic_func(data):
        if data.ndim == 2:
            data = data.unsqueeze(0)
        motions = dataset.inv_transform(data)
        return recover_pos_from_rot(motions, joints_num=cfg.data.joint_num, skeleton=skeleton)

    eval_cfg = load_config(snapmogen_evaluator_config_path())
    eval_wrapper = EvaluatorWrapper(eval_cfg, device=device)

    print(
        f"[ScaleMoGen][VQ Eval] data_root={cfg.data.root_dir} realpath={os.path.realpath(cfg.data.root_dir)}",
        flush=True,
    )
    print(
        f"[ScaleMoGen][VQ Eval] split={split} deterministic_data={dataset.deterministic} "
        f"shuffle={bool(cfg.eval.get('shuffle', False))} num_workers={int(cfg.eval.get('num_workers', 0))}",
        flush=True,
    )

    fid, div, top1, top2, top3, matching, mpjpe = evaluation_vqvae(
        out_dir=cfg.exp.eval_dir,
        val_loader=loader,
        net=vq_model,
        writer=None,
        ep=0,
        best_fid=float("inf"),
        best_div=float("inf"),
        best_top1=0,
        best_top2=0,
        best_top3=0,
        best_matching=0,
        best_mpjpe=float("inf"),
        nfeats=cfg.data.dim_pose,
        eval_wrapper=eval_wrapper,
        device=device,
        fk_func=forward_kinematic_func,
        save_ckpt=False,
        draw=False,
        save_anim=False,
        plot_eval=None,
    )
    print(
        f"[ScaleMoGen][VQ Eval] summary FID={fid:.4f} matching={matching:.4f} "
        f"R@1={top1:.4f} R@2={top2:.4f} R@3={top3:.4f} MPJPE={mpjpe:.4f}",
        flush=True,
    )


def _eval_humanml3d(cfg, vq_model, device, split):
    """Run HumanML3D VQ reconstruction metrics."""
    dataset_opt_path = "checkpoint_dir/humanml3d/Comp_v6_KLD005/opt.txt"
    wrapper_opt = get_opt(dataset_opt_path, device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    loader, dataset = get_dataset_motion_loader(
        dataset_opt_path,
        cfg.eval.batch_size,
        split,
        device=device,
        deterministic=bool(cfg.eval.get("deterministic_data", True)),
        shuffle=bool(cfg.eval.get("shuffle", False)),
        num_workers=int(cfg.eval.get("num_workers", 0)),
        seed=cfg.exp.seed,
    )
    print(
        f"[ScaleMoGen][VQ Eval] data_root={cfg.data.root_dir} realpath={os.path.realpath(cfg.data.root_dir)}",
        flush=True,
    )
    print(
        f"[ScaleMoGen][VQ Eval] split={split} deterministic_data={dataset.deterministic} "
        f"shuffle={bool(cfg.eval.get('shuffle', False))} num_workers={int(cfg.eval.get('num_workers', 0))}",
        flush=True,
    )

    fid, div, top1, top2, top3, matching, _ = evaluation_vqvae_hml(
        out_dir=cfg.exp.eval_dir,
        val_loader=loader,
        net=vq_model,
        writer=None,
        ep=0,
        best_fid=float("inf"),
        best_div=float("inf"),
        best_top1=0,
        best_top2=0,
        best_top3=0,
        best_matching=float("inf"),
        eval_wrapper=eval_wrapper,
        save=False,
        draw=False,
    )
    print(
        f"[ScaleMoGen][VQ Eval] summary FID={fid:.4f} matching={matching:.4f} "
        f"R@1={top1:.4f} R@2={top2:.4f} R@3={top3:.4f}",
        flush=True,
    )


def main():
    args = parse_args()
    cfg = _apply_overrides(load_config(args.config), args)
    fixseed(cfg.exp.seed)
    configure_reproducible_eval_runtime()

    device = torch.device(cfg.exp.device)
    if device.type == "cuda":
        torch.cuda.set_device(cfg.exp.device)

    cfg.exp.eval_dir = pjoin(
        cfg.exp.root_ckpt_dir,
        cfg.data.name,
        "vq",
        cfg.vq_name,
        "evaluation",
        cfg.eval.get("eval_name", "vq_reconstruction"),
    )
    os.makedirs(cfg.exp.eval_dir, exist_ok=True)

    print("[ScaleMoGen][VQ Eval] loading VQ...", flush=True)
    vq_model, vq_cfg, vq_cfg_path, vq_ckpt_path = load_scalemogen_vq(cfg, device)
    print(f"[ScaleMoGen][VQ Eval] loaded_vq_config={vq_cfg_path}", flush=True)
    print(f"[ScaleMoGen][VQ Eval] loaded_vq_ckpt={vq_ckpt_path}", flush=True)
    print(
        f"[ScaleMoGen][VQ Eval] dataset={cfg.data.name} vq_name={cfg.vq_name} "
        f"vq_ckpt={cfg.vq_ckpt} device={device}",
        flush=True,
    )

    if cfg.data.name == "humanml3d":
        _eval_humanml3d(cfg, vq_model, device, args.split)
    else:
        _eval_snapmogen(cfg, vq_model, device, args.split)


if __name__ == "__main__":
    main()
