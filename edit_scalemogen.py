"""ScaleMoGen SnapMoGen text-guided motion editing entrypoint.

Idea provenance: inspired by the AREdit paper. The editing utilities are
implemented for ScaleMoGen rather than copied from an official public code
release.
"""

import argparse
import os
import random
import subprocess
import warnings
from os.path import join as pjoin

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
from einops import rearrange, repeat

from common.animation import Animation
from common.skeleton import Skeleton
from config.load_config import load_config
from rest_pose_retarget import RestPoseRetargeter
from scalemogen.checkpoint import load_scalemogen_predictor, load_scalemogen_vq
from scalemogen.generation import coarse_to_fine_chain_from_vq, scale_schedule_from_vq
from scalemogen.text import load_text_encoder
from tools.run_scalemogen import gen_one_motion
from utils import bvh_io
from utils.edit_utils import sampling_func_wrapper
from utils.fixseeds import fixseed
from utils.motion_process_bvh import recover_bvh_from_rot
from utils.paramUtil import kinematic_chain
from utils.utils import plot_3d_motion

warnings.filterwarnings("ignore")


EDIT_PRESETS = {
    "arms_up": {
        "motion_len": 240,
        "prompt": (
            "The person walks forward at a relaxed pace with an upright posture. "
            "Their arms swing naturally at their sides in opposition to the legs. "
            "The steps are even and steady, with a smooth heel-to-toe motion. "
            "The head faces forward, and the torso remains stable throughout the walk."
        ),
        "target_prompt": (
            "The person walks forward at a relaxed pace with an upright posture. "
            "Their arms lift upward away from the sides, raising the hands to about "
            "shoulder height or higher. The steps are even and steady, with a smooth "
            "heel-to-toe motion. The head faces forward, and the torso remains stable "
            "throughout the walk."
        ),
        "tau": 0.1,
        "gamma": 3,
        "edit_temp": None,
        "edit_joints": [4, 5],
    },
    "legs_march": {
        "motion_len": 240,
        "prompt": (
            "The person walks forward at a steady pace with an upright posture. "
            "Their arms swing gently at their sides. The steps are even and steady, "
            "with a smooth heel-to-toe motion. The head faces forward and the torso "
            "remains calm throughout the walk."
        ),
        "target_prompt": (
            "The person marches forward at a steady pace with an upright posture. "
            "Their arms swing gently at their sides. With each step, the knee lifts "
            "high toward the chest before the foot plants firmly on the ground. The "
            "head faces forward and the torso remains calm throughout the walk."
        ),
        "tau": 0.1,
        "gamma": 3,
        "edit_temp": None,
        "edit_joints": [1, 2],
    },
    "early_motion": {
        "motion_len": 240,
        "prompt": (
            "The person stays still for a moment in place with arms relaxed at the "
            "sides, then gradually transitions into walking forward at a relaxed pace "
            "with arms swinging naturally. The steps are smooth and even, with a "
            "heel-to-toe motion throughout."
        ),
        "target_prompt": (
            "The person jogs lightly in place with knees bouncing and arms bent at the "
            "elbows, then gradually transitions into walking forward at a relaxed pace "
            "with arms swinging naturally. The steps are smooth and even, with a "
            "heel-to-toe motion throughout."
        ),
        "tau": 0.1,
        "gamma": 4,
        "edit_temp": [0, 80],
        "edit_joints": None,
    },
    "mid_motion": {
        "motion_len": 296,
        "prompt": (
            "The person walks forward steadily in a relaxed pace, pauses in the middle, "
            "stands still briefly, then resumes walking forward at the same steady pace."
        ),
        "target_prompt": (
            "The person walks forward steadily in a relaxed pace, then bends the knees "
            "and lowers into a squat position midway, rises back upright, and resumes "
            "walking forward at the same steady pace."
        ),
        "tau": 0.1,
        "gamma": 3,
        "edit_temp": [80, 160],
        "edit_joints": None,
    },
    "late_motion": {
        "motion_len": 240,
        "prompt": (
            "The person walks forward steadily with natural arm swing and upright "
            "posture. The gait is even and rhythmic throughout."
        ),
        "target_prompt": (
            "The person walks forward steadily with natural arm swing and upright "
            "posture. Toward the end, they slow to a stop and gradually lower "
            "themselves to sit on the ground, bending the knees and lowering the hips "
            "until fully seated."
        ),
        "tau": 0.1,
        "gamma": 3,
        "edit_temp": [160, 240],
        "edit_joints": None,
    },
    "dance_exaggerated": {
        "motion_len": 240,
        "prompt": (
            "The person dances in a hip-hop beat. The hips swing from side to side, "
            "both arms sweep up and out in arcs, and the whole body bounces."
        ),
        "target_prompt": (
            "The person dances in a hip-hop beat with exaggerated, theatrical "
            "movements. The hips swing widely from side to side, both arms sweep up "
            "and out in large expressive arcs, and the whole body bounces with energy. "
            "Each movement is dramatically amplified and full of flair."
        ),
        "tau": 0.1,
        "gamma": 3,
        "edit_temp": None,
        "edit_joints": None,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Edit a SnapMoGen sample with ScaleMoGen.")
    parser.add_argument("--config", default="config/eval_scalemogen.yaml")
    parser.add_argument("--preset", default="dance_exaggerated", choices=sorted(EDIT_PRESETS))
    parser.add_argument("--edit_name", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--target_prompt", default=None)
    parser.add_argument("--motion_len", type=int, default=None)
    parser.add_argument("--edit_temp", type=int, nargs=2, default=None)
    parser.add_argument("--edit_joints", type=int, nargs="+", default=None)
    parser.add_argument("--all_joints", action="store_true")
    parser.add_argument("--full_time", action="store_true")
    parser.add_argument("--tau", type=float, default=None, help="Cache replacement threshold for editing.")
    parser.add_argument("--gamma", type=int, default=None, help="Number of early scales sampled freely.")
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--gen_tau", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--target_seed", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--skip_mp4", action="store_true")
    parser.add_argument("--skip_bvh", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")
    return parser.parse_args()


def resolve_edit_request(args, cfg):
    """Merge a named preset with command-line overrides."""
    preset = dict(EDIT_PRESETS[args.preset])
    edit_name = args.edit_name or args.preset
    prompt = args.prompt or preset["prompt"]
    target_prompt = args.target_prompt or preset["target_prompt"]
    motion_len = args.motion_len or preset["motion_len"]
    motion_len = min(motion_len, int(cfg.data.max_motion_length))
    motion_len = max(int(cfg.data.unit_length), (motion_len // int(cfg.data.unit_length)) * int(cfg.data.unit_length))

    edit_temp = None if args.full_time else (args.edit_temp if args.edit_temp is not None else preset["edit_temp"])
    edit_joints = None if args.all_joints else (args.edit_joints if args.edit_joints is not None else preset["edit_joints"])
    tau = args.tau if args.tau is not None else preset["tau"]
    gamma = args.gamma if args.gamma is not None else preset["gamma"]
    cfg_value = args.cfg if args.cfg is not None else cfg.eval.cfg
    gen_tau = args.gen_tau if args.gen_tau is not None else cfg.eval.tau
    return edit_name, prompt, target_prompt, motion_len, edit_temp, edit_joints, tau, gamma, cfg_value, gen_tau


def ffmpeg_hstack(left_path, right_path, output_path):
    """Create a side-by-side comparison video when ffmpeg is available."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        left_path,
        "-i",
        right_path,
        "-filter_complex",
        "hstack",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        output_path,
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except FileNotFoundError:
        print("[ScaleMoGen][Edit] ffmpeg not found; skipped comparison video", flush=True)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    fixseed(cfg.exp.seed)

    device = torch.device(cfg.exp.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    edit_name, prompt, target_prompt, motion_len, edit_temp, edit_joints, edit_tau, gamma, cfg_value, gen_tau = (
        resolve_edit_request(args, cfg)
    )
    edit_dir = args.output_dir or pjoin(
        cfg.exp.root_ckpt_dir,
        cfg.data.name,
        "predictor",
        cfg.exp.name,
        "edit",
        edit_name,
    )
    bvh_dir = pjoin(edit_dir, "bvh")
    os.makedirs(edit_dir, exist_ok=True)
    os.makedirs(bvh_dir, exist_ok=True)

    print(f"[ScaleMoGen][Edit] preset={args.preset} output={edit_dir}", flush=True)
    print(f"[ScaleMoGen][Edit] source={prompt}", flush=True)
    print(f"[ScaleMoGen][Edit] target={target_prompt}", flush=True)
    print(
        f"[ScaleMoGen][Edit] length={motion_len} edit_temp={edit_temp} edit_joints={edit_joints} "
        f"tau={edit_tau} gamma={gamma}",
        flush=True,
    )

    print("[ScaleMoGen][Edit] loading_vq", flush=True)
    vq_model, _vq_cfg, _, vq_ckpt_path = load_scalemogen_vq(cfg, device)
    print(f"[ScaleMoGen][Edit] loaded_vq={vq_ckpt_path}", flush=True)

    print("[ScaleMoGen][Edit] loading_predictor", flush=True)
    predictor_model, train_cfg, _, predictor_ckpt_path = load_scalemogen_predictor(
        cfg,
        vq_model,
        device,
        batch_size=1,
        disable_init=True,
    )
    print(f"[ScaleMoGen][Edit] loaded_predictor={predictor_ckpt_path}", flush=True)

    text_tokenizer, text_encoder = load_text_encoder(
        train_cfg,
        device,
        dtype_name=cfg.eval.get("text_encoder_dtype", "fp32"),
        encoder_device=cfg.eval.get("text_encoder_device", cfg.exp.device),
    )
    print(
        f"[ScaleMoGen][Edit] loading_text_encoder dtype={cfg.eval.get('text_encoder_dtype', 'fp32')} "
        f"device={next(text_encoder.parameters()).device}",
        flush=True,
    )

    coarse_to_fine_chain = coarse_to_fine_chain_from_vq(vq_model)
    scale_schedule = scale_schedule_from_vq(vq_model, cfg.data.max_motion_length)
    len_scale_factor = getattr(vq_model.quantizer2d, "len_scale_factor", cfg.data.unit_length // 2)
    use_bf16 = bool(args.use_bf16 or cfg.eval.get("use_bf16", False))

    cfg.data.feat_dir = pjoin(cfg.data.root_dir, "renamed_feats")
    meta_dir = pjoin(cfg.data.root_dir, "meta_data")
    mean = np.load(pjoin(meta_dir, "mean.npy"))
    std = np.load(pjoin(meta_dir, "std.npy"))
    mean_tensor = torch.tensor(mean, dtype=torch.float32, device=device)
    std_tensor = torch.tensor(std, dtype=torch.float32, device=device)

    template_anim = bvh_io.load(pjoin(cfg.data.root_dir, "renamed_bvhs", "m_ep2_00086.bvh"))
    skeleton = Skeleton(template_anim.offsets, template_anim.parents, device=device)
    retargeter = RestPoseRetargeter()

    def forward_kinematic_func(data):
        """Recover global joint positions and local rotations from normalized motion features."""
        if data.ndim == 2:
            data = data.unsqueeze(0)
        motions = data.to(device) * std_tensor + mean_tensor
        batch_size, _, _ = motions.shape
        global_quats, local_quats, root_pos = recover_bvh_from_rot(
            motions,
            cfg.data.joint_num,
            skeleton,
            keep_shape=False,
        )
        _, global_pos = skeleton.fk_local_quat(local_quats, root_pos)
        global_pos = rearrange(global_pos, "(b l) j d -> b l j d", b=batch_size)
        local_quats = rearrange(local_quats, "(b l) j d -> b l j d", b=batch_size)
        root_pos = rearrange(root_pos, "(b l) d -> b l d", b=batch_size)
        return global_pos, local_quats, root_pos

    def save_bvh(motion, length, path):
        """Save a generated normalized motion as BVH."""
        _, local_quats, root_pos = forward_kinematic_func(motion)
        anim = Animation(
            local_quats[0, :length].detach().cpu().numpy(),
            repeat(root_pos[0, :length].detach().cpu().numpy(), "i j -> i k j", k=len(template_anim)),
            template_anim.orients,
            template_anim.offsets,
            template_anim.parents,
            template_anim.names,
            template_anim.frametime,
        )
        bvh_io.save(
            path,
            retargeter.rest_pose_retarget(anim),
            names=anim.names,
            frametime=anim.frametime,
            order="xyz",
            quater=True,
        )
        print(f"[ScaleMoGen][Edit] saved_bvh={path}", flush=True)

    def save_mp4(motion, length, title, path):
        """Save a generated normalized motion as MP4."""
        global_pos, _, _ = forward_kinematic_func(motion)
        plot_3d_motion(
            path,
            kinematic_chain,
            global_pos[0, :length].detach().cpu().numpy(),
            title=title,
            fps=cfg.data.fps,
            radius=100,
        )
        print(f"[ScaleMoGen][Edit] saved_mp4={path}", flush=True)

    source_seed = args.seed if args.seed is not None else random.randint(0, 10000)
    target_seed = args.target_seed if args.target_seed is not None else source_seed
    length_tensor = torch.tensor([motion_len], dtype=torch.long, device=device)

    with torch.no_grad():
        source_motion, _, idx_cache, prob_cache = gen_one_motion(
            predictor_model,
            vq_model,
            text_tokenizer,
            text_encoder,
            prompt,
            g_seed=source_seed,
            cfg_list=cfg_value,
            tau_list=gen_tau,
            scale_schedule=scale_schedule,
            coarse_to_fine_chain=coarse_to_fine_chain,
            sampling_per_bits=cfg.eval.sampling_per_bits,
            cfg_insertion_layer=[cfg.eval.cfg_insertion_layer],
            enable_positive_prompt=cfg.eval.enable_positive_prompt,
            m_lengths=length_tensor,
            top_p=args.top_p,
            top_k=args.top_k,
            resampling_steps=0,
            sampling_func=None,
            return_idx_prob=True,
            use_bf16=use_bf16,
        )

        edit_sampling_func = sampling_func_wrapper(
            idx_cache,
            prob_cache,
            tau=edit_tau,
            gamma=gamma,
            edit_temp=edit_temp,
            edit_joints=edit_joints,
            coarse_to_fine_chain=coarse_to_fine_chain,
            motion_scale_schedule=scale_schedule,
            len_scale_factor=len_scale_factor,
        )
        edited_motion, *_ = gen_one_motion(
            predictor_model,
            vq_model,
            text_tokenizer,
            text_encoder,
            target_prompt,
            g_seed=target_seed,
            cfg_list=cfg_value,
            tau_list=gen_tau,
            scale_schedule=scale_schedule,
            coarse_to_fine_chain=coarse_to_fine_chain,
            sampling_per_bits=cfg.eval.sampling_per_bits,
            cfg_insertion_layer=[cfg.eval.cfg_insertion_layer],
            enable_positive_prompt=cfg.eval.enable_positive_prompt,
            m_lengths=length_tensor,
            top_p=args.top_p,
            top_k=args.top_k,
            resampling_steps=0,
            sampling_func=edit_sampling_func,
            return_idx_prob=True,
            use_bf16=use_bf16,
        )

    np.savez(
        pjoin(edit_dir, "source_motion.npz"),
        motion=source_motion.float().detach().cpu().numpy(),
        prompt=prompt,
        seed=source_seed,
        motion_len=motion_len,
    )
    np.savez(
        pjoin(edit_dir, "edited_motion.npz"),
        motion=edited_motion.float().detach().cpu().numpy(),
        prompt=target_prompt,
        seed=target_seed,
        motion_len=motion_len,
    )

    if not args.skip_bvh:
        save_bvh(source_motion.float(), motion_len, pjoin(bvh_dir, "source_motion.bvh"))
        save_bvh(edited_motion.float(), motion_len, pjoin(bvh_dir, "edited_motion.bvh"))

    if not args.skip_mp4:
        source_mp4 = pjoin(edit_dir, "source_motion.mp4")
        edited_mp4 = pjoin(edit_dir, "edited_motion.mp4")
        save_mp4(source_motion.float(), motion_len, "Source Motion", source_mp4)
        save_mp4(edited_motion.float(), motion_len, "Edited Motion", edited_mp4)
        ffmpeg_hstack(source_mp4, edited_mp4, pjoin(edit_dir, "comparison_motion.mp4"))

    print(f"[ScaleMoGen][Edit] done={edit_dir}", flush=True)


if __name__ == "__main__":
    main()
