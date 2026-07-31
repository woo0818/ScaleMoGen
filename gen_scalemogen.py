"""ScaleMoGen text-to-motion generation entrypoint."""

import argparse
import os
import random
import warnings
from os.path import join as pjoin

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch

from config.load_config import load_config
from scalemogen.checkpoint import load_scalemogen_predictor, load_scalemogen_vq
from scalemogen.dataset_dispatch import (
    build_generation_renderer,
    hstack_videos,
    is_humanml3d,
    parse_generation_entries,
)
from scalemogen.generation import coarse_to_fine_chain_from_vq, scale_schedule_from_vq
from scalemogen.text import load_text_encoder
from tools.run_scalemogen import gen_one_motion
from utils.fixseeds import fixseed


def parse_args():
    """Parse generation arguments."""
    parser = argparse.ArgumentParser(description="Generate motions with ScaleMoGen.")
    parser.add_argument("--config", type=str, default="config/eval_scalemogen.yaml")
    parser.add_argument(
        "--mode",
        type=str,
        default="casual",
        choices=["casual", "test"],
        help="SnapMoGen only: casual prompts or full test split. HumanML3D always uses the test split.",
    )
    parser.add_argument("--text_file", type=str, default="./text_descriptions_casual.txt")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--max_scale",
        type=int,
        nargs="+",
        default=None,
        help="Scale levels to visualize. Full scale is appended automatically.",
    )
    return parser.parse_args()


def _generation_dir(cfg, mode):
    """Return the output directory for generated motions."""
    suffix = "gen_test" if is_humanml3d(cfg) or mode == "test" else "gen_casual"
    return pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, "predictor", cfg.exp.name, suffix)


def _scale_levels(args, scale_schedule):
    """Return requested partial scale levels plus the full scale."""
    total_scales = len(scale_schedule)
    if args.max_scale is None:
        return None
    levels = sorted(set(args.max_scale))
    if total_scales not in levels:
        levels.append(total_scales)
    print(f"Multi-scale comparison mode: scales {levels} (total available: {total_scales})")
    return levels


def _generate_one(
    cfg,
    device,
    predictor_model,
    vq_model,
    text_tokenizer,
    text_encoder,
    entry,
    seed,
    coarse_to_fine_chain,
    scale_schedule,
    trunk_scale=None,
):
    """Run one ScaleMoGen generation call."""
    return gen_one_motion(
        predictor_model,
        vq_model,
        text_tokenizer,
        text_encoder,
        entry.prompt,
        all_mask=None,
        motion_padding_mask=None,
        g_seed=seed,
        gt_leak=0,
        gt_ls_Bl=None,
        cfg_list=cfg.eval.cfg,
        tau_list=cfg.eval.tau,
        coarse_to_fine_chain=coarse_to_fine_chain,
        scale_schedule=scale_schedule,
        cfg_insertion_layer=[cfg.eval.cfg_insertion_layer],
        sampling_per_bits=cfg.eval.sampling_per_bits,
        enable_positive_prompt=cfg.eval.enable_positive_prompt,
        input_motions=None,
        m_lengths=torch.tensor([entry.motion_length], device=device),
        trunk_scale=trunk_scale,
        use_bf16=cfg.eval.get("use_bf16", False),
        sampling_device=cfg.eval.get("sampling_device", "cuda"),
        sampling_method=cfg.eval.get("sampling_method", "multinomial"),
    )


def main():
    """Generate motions for the dataset specified in config."""
    args = parse_args()
    cfg = load_config(args.config)
    fixseed(cfg.exp.seed)
    device = torch.device(cfg.exp.device)
    if device.type == "cuda":
        torch.cuda.set_device(cfg.exp.device)

    gen_dir = _generation_dir(cfg, args.mode)
    os.makedirs(gen_dir, exist_ok=True)
    renderer = build_generation_renderer(cfg, device)
    renderer.prepare_dirs(gen_dir)

    entries = parse_generation_entries(cfg, args.mode, args.text_file)
    if args.max_samples is not None:
        entries = entries[: args.max_samples]
    print(f"[ScaleMoGen][Gen] dataset={cfg.data.name} samples={len(entries)} out_dir={gen_dir}")

    print("Loading VQ model...")
    vq_model, _, _, vq_ckpt_path = load_scalemogen_vq(cfg, device)
    print(f"VQ model loaded from {vq_ckpt_path}")

    print("Loading ScaleMoGen predictor...")
    predictor_model, train_cfg, _, predictor_ckpt_path = load_scalemogen_predictor(
        cfg, vq_model, device, disable_init=True
    )
    print(f"ScaleMoGen predictor loaded from {predictor_ckpt_path}")

    print("Loading text encoder...")
    text_tokenizer, text_encoder = load_text_encoder(
        train_cfg,
        device,
        dtype_name=cfg.eval.get("text_encoder_dtype", "fp32"),
        encoder_device=cfg.eval.get("text_encoder_device", cfg.exp.device),
    )

    coarse_to_fine_chain = coarse_to_fine_chain_from_vq(vq_model)
    scale_schedule = scale_schedule_from_vq(vq_model, cfg.data.max_motion_length)
    scale_levels = _scale_levels(args, scale_schedule)
    total_scales = len(scale_schedule)

    print(
        f"[ScaleMoGen][Gen] cfg={cfg.eval.cfg} tau={cfg.eval.tau} "
        f"use_bf16={cfg.eval.get('use_bf16', False)} "
        f"sampling_device={cfg.eval.get('sampling_device', 'cuda')} "
        f"sampling_method={cfg.eval.get('sampling_method', 'multinomial')}"
    )

    log_path = pjoin(gen_dir, "text_descriptions.txt")
    with open(log_path, "w") as log_file:
        for entry_idx, entry in enumerate(entries):
            print(f"\n[{entry_idx + 1}/{len(entries)}] Generating: {entry.prompt[:80]}...")
            print(f"  Motion length: {entry.motion_length} frames")
            seed = random.randint(0, 10000)

            if scale_levels is not None:
                mp4_paths = []
                for scale in scale_levels:
                    trunk_scale = scale if scale < total_scales else 1000
                    label = f"Scale {scale}/{total_scales}"
                    generated_motion, _ = _generate_one(
                        cfg,
                        device,
                        predictor_model,
                        vq_model,
                        text_tokenizer,
                        text_encoder,
                        entry,
                        seed,
                        coarse_to_fine_chain,
                        scale_schedule,
                        trunk_scale=trunk_scale,
                    )
                    pred_motions = generated_motion.unsqueeze(0).float()
                    actual_len = min(entry.motion_length, pred_motions.shape[1])
                    suffix = f"scale{scale if scale < total_scales else total_scales}"
                    mp4_paths.append(
                        renderer.render(
                            gen_dir,
                            entry,
                            pred_motions,
                            actual_len,
                            label,
                            suffix,
                            save_full=scale >= total_scales,
                            include_gt=args.mode == "test",
                        )
                    )
                concat_path = pjoin(gen_dir, "mp4", f"{entry.safe_key}_concat.mp4")
                hstack_videos(mp4_paths, concat_path)
                print(f"  Saved concat MP4: {concat_path}")
            else:
                generated_motion, _ = _generate_one(
                    cfg,
                    device,
                    predictor_model,
                    vq_model,
                    text_tokenizer,
                    text_encoder,
                    entry,
                    seed,
                    coarse_to_fine_chain,
                    scale_schedule,
                )
                pred_motions = generated_motion.unsqueeze(0).float()
                actual_len = min(entry.motion_length, pred_motions.shape[1])
                mp4_path = renderer.render(
                    gen_dir,
                    entry,
                    pred_motions,
                    actual_len,
                    entry.prompt,
                    "gen",
                    save_full=True,
                    include_gt=args.mode == "test",
                )
                print(f"  Saved MP4: {mp4_path}")

            log_file.write(f"{entry.key} # {entry.prompt} # {entry.motion_length}\n")
            print(f"  Done ({entry_idx + 1}/{len(entries)})")

    print(f"\nGeneration complete. {len(entries)} motions saved to {gen_dir}")


if __name__ == "__main__":
    main()
