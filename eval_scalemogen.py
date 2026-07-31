"""ScaleMoGen text-to-motion evaluation entrypoint."""

import argparse
import os
from os.path import join as pjoin

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from config.load_config import load_config
from scalemogen.checkpoint import load_scalemogen_predictor, load_scalemogen_vq
from scalemogen.dataset_dispatch import build_eval_data
from scalemogen.runtime import configure_reproducible_eval_runtime
from scalemogen.text import load_text_encoder
from utils.fixseeds import fixseed


def parse_args():
    """Parse evaluation arguments."""
    parser = argparse.ArgumentParser(description="Evaluate ScaleMoGen.")
    parser.add_argument("--config", default="config/eval_scalemogen.yaml")
    return parser.parse_args()


def main():
    """Run text-to-motion evaluation for the dataset specified in config."""
    args = parse_args()
    cfg = load_config(args.config)
    fixseed(cfg.exp.seed)
    configure_reproducible_eval_runtime()

    device = torch.device(cfg.exp.device)
    if device.type == "cuda":
        torch.cuda.set_device(cfg.exp.device)

    eval_name = cfg.eval.get("eval_name", "scalemogen_eval")
    cfg.exp.eval_dir = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, "predictor", cfg.exp.name, "evaluation", eval_name)
    os.makedirs(cfg.exp.eval_dir, exist_ok=True)

    print("Loading VQ model...")
    vq_model, _, _, vq_ckpt_path = load_scalemogen_vq(cfg, device)
    print(f"VQ model loaded from {vq_ckpt_path}")

    print("Loading ScaleMoGen predictor...")
    predictor_model, train_cfg, _, predictor_ckpt_path = load_scalemogen_predictor(
        cfg, vq_model, device, batch_size=cfg.eval.batch_size
    )
    print(f"ScaleMoGen predictor loaded from {predictor_ckpt_path}")

    text_tokenizer, text_encoder = load_text_encoder(
        train_cfg,
        device,
        dtype_name=cfg.eval.get("text_encoder_dtype", "fp32"),
        encoder_device=cfg.eval.get("text_encoder_device", cfg.exp.device),
    )
    print(
        f"[ScaleMoGen][Eval] text_encoder_dtype={cfg.eval.get('text_encoder_dtype', 'fp32')} "
        f"text_encoder_device={next(text_encoder.parameters()).device}",
        flush=True,
    )

    print("Loading dataset and evaluator...")
    data_bundle = build_eval_data(cfg, device)
    print(
        f"[ScaleMoGen][Eval] dataset={cfg.data.name} data_root={cfg.data.root_dir} "
        f"realpath={os.path.realpath(cfg.data.root_dir)}",
        flush=True,
    )
    print(
        f"[ScaleMoGen][Eval] deterministic_data={cfg.eval.get('deterministic_data', True)} "
        f"shuffle={cfg.eval.get('shuffle', False)} num_workers={cfg.eval.get('num_workers', 0)}",
        flush=True,
    )

    repeat_times = data_bundle.repeat_times
    use_bf16 = bool(cfg.eval.get("use_bf16", False))
    seed_mode = cfg.eval.get("seed_mode", "deterministic")
    base_seed = cfg.eval.get("base_seed", cfg.exp.seed)
    sampling_device = cfg.eval.get("sampling_device", "cuda")
    sampling_method = cfg.eval.get("sampling_method", "multinomial")
    print(
        f"[ScaleMoGen][Eval] use_bf16={use_bf16} seed_mode={seed_mode} base_seed={base_seed} "
        f"sampling_device={sampling_device} sampling_method={sampling_method}",
        flush=True,
    )

    metrics = {"fid": [], "diversity": [], "top1": [], "top2": [], "top3": [], "matching": [], "multimodality": []}
    for repeat_id in range(repeat_times):
        fid, diversity, r_precision, matching_score, multimodality = data_bundle.evaluation_func(
            out_dir=cfg.exp.eval_dir,
            val_loader=data_bundle.eval_loader,
            predictor=predictor_model,
            vq_model=vq_model,
            text_tokenizer=text_tokenizer,
            text_encoder=text_encoder,
            eval_wrapper=data_bundle.eval_wrapper,
            device=device,
            repeat_id=repeat_id,
            cfg_val=cfg.eval.cfg,
            tau_val=cfg.eval.tau,
            sampling_per_bits=cfg.eval.sampling_per_bits,
            cfg_insertion_layer=cfg.eval.cfg_insertion_layer,
            enable_positive_prompt=cfg.eval.enable_positive_prompt,
            cal_mm=cfg.eval.cal_mm,
            use_bf16=use_bf16,
            seed_mode=seed_mode,
            base_seed=base_seed,
            sampling_device=sampling_device,
            sampling_method=sampling_method,
        )
        metrics["fid"].append(fid)
        metrics["diversity"].append(diversity)
        metrics["top1"].append(r_precision[0])
        metrics["top2"].append(r_precision[1])
        metrics["top3"].append(r_precision[2])
        metrics["matching"].append(matching_score)
        metrics["multimodality"].append(multimodality)

    print("\n" + "=" * 20 + " Quantitative Evaluation Results " + "=" * 20)
    print(f"Dataset: {cfg.data.name}")
    print(f"Evaluation Model: {cfg.eval.model_ckpt}")
    for key, values in metrics.items():
        if not values:
            continue
        mean = np.mean(values)
        std = np.std(values)
        if data_bundle.report_confidence:
            conf_interval = 1.96 * std / np.sqrt(len(values))
            print(f"  {key.upper():<15}: Mean={mean:.4f}, Std={std:.4f}, Conf={conf_interval:.4f}")
        else:
            print(f"  {key.upper():<15}: Mean={mean:.4f}, Std={std:.4f}")
    print("=" * 67)


if __name__ == "__main__":
    main()
