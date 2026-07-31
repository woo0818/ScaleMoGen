"""Dataset-specific ScaleMoGen loading and rendering helpers.

This module keeps dataset differences out of the training, evaluation, and
generation entrypoints. SnapMoGen uses its official data conventions, and
HumanML3D follows the common T2M evaluation conventions.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from os.path import join as pjoin

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
from einops import rearrange, repeat
from torch.utils.data import DataLoader

from common.animation import Animation
from common.skeleton import Skeleton
from config.load_config import load_config
from dataset.dataset import TextMotionDataset as SnapMoGenTextMotionDataset
from dataset.humanml3d_dataset import TextMotionDataset as HumanML3DTextMotionDataset
from model.evaluator.evaluator_wrapper import EvaluatorWrapper
from model.evaluator.hml.dataset_motion_loader import get_dataset_motion_loader
from model.evaluator.hml.t2m_eval_wrapper import EvaluatorModelWrapper
from rest_pose_retarget import RestPoseRetargeter
from utils import bvh_io
from utils.eval_t2m import evaluation_scalemogen, evaluation_scalemogen_hml
from utils.foot_ik import foot_lock
from utils.get_opt import get_opt
from utils.motion_process_bvh import recover_bvh_from_rot, recover_from_ric, recover_pos_from_rot
from utils.paramUtil import kinematic_chain, t2m_kinematic_chain
from utils.utils import plot_3d_motion


SNAPMOGEN_EVALUATOR_CONFIGS = (
    "checkpoint_dir/snapmogen/evaluator/eval_klde-5_late-5_nlayer6_norm/evaluator.yaml",
)
HUMANML3D_EVALUATOR_OPT = "checkpoint_dir/humanml3d/Comp_v6_KLD005/opt.txt"


@dataclass
class DatasetBundle:
    """Container returned by dataset-specific builder functions."""

    train_dataset: object = None
    val_dataset: object = None
    eval_dataset: object = None
    eval_loader: object = None
    eval_wrapper: object = None
    plot_t2m: object = None
    forward_kinematic_func: object = None
    evaluation_func: object = None
    eval_tau: float = 1.3
    repeat_times: int = 3
    report_confidence: bool = False


@dataclass
class GenerationEntry:
    """A single text-to-motion generation request."""

    key: str
    prompt: str
    motion_length: int
    mid: str = None
    start: int = None
    end: int = None

    @property
    def safe_key(self):
        """Return a filesystem-safe key for generated artifacts."""
        return str(self.key).replace("#", "_").replace("/", "_")


def dataset_name(cfg):
    """Return the normalized dataset name from a ScaleMoGen config."""
    return str(cfg.data.name).lower()


def is_humanml3d(cfg):
    """Return whether the config targets HumanML3D."""
    return dataset_name(cfg) in {"humanml3d", "t2m"}


def is_snapmogen(cfg):
    """Return whether the config targets SnapMoGen."""
    return dataset_name(cfg) == "snapmogen"


def snapmogen_evaluator_config_path():
    """Return the SnapMoGen evaluator config path."""
    for path in SNAPMOGEN_EVALUATOR_CONFIGS:
        if os.path.exists(path):
            return path
    return SNAPMOGEN_EVALUATOR_CONFIGS[0]


def _require_supported_dataset(cfg):
    """Raise for datasets that do not have an active ScaleMoGen dispatch path."""
    if not (is_snapmogen(cfg) or is_humanml3d(cfg)):
        raise ValueError(f"Unsupported ScaleMoGen dataset: {cfg.data.name}")


def _as_int(value):
    """Convert Python or scalar tensor lengths into an int."""
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    return int(value)


def _snapmogen_paths(cfg):
    """Return standard SnapMoGen feature, split, and caption paths."""
    cfg.data.feat_dir = pjoin(cfg.data.root_dir, "renamed_feats")
    meta_dir = pjoin(cfg.data.root_dir, "meta_data")
    split_dir = pjoin(cfg.data.root_dir, "data_split_info1")
    return {
        "mean": pjoin(meta_dir, "mean.npy"),
        "std": pjoin(meta_dir, "std.npy"),
        "captions": pjoin(cfg.data.root_dir, "all_caption_clean.json"),
        "train_mid": pjoin(split_dir, "train_fnames.txt"),
        "train_ids": pjoin(split_dir, "train_ids.txt"),
        "val_mid": pjoin(split_dir, "val_fnames.txt"),
        "val_ids": pjoin(split_dir, "val_ids.txt"),
        "test_mid": pjoin(split_dir, "test_fnames.txt"),
        "test_ids": pjoin(split_dir, "test_ids.txt"),
        "template_bvh": pjoin(cfg.data.root_dir, "renamed_bvhs", "m_ep2_00086.bvh"),
    }


def _humanml3d_paths(cfg):
    """Return standard HumanML3D feature and split paths."""
    cfg.data.feat_dir = pjoin(cfg.data.root_dir, "new_joint_vecs")
    return {
        "train_ids": pjoin(cfg.data.root_dir, "train.txt"),
        "val_ids": pjoin(cfg.data.root_dir, "val.txt"),
        "test_ids": pjoin(cfg.data.root_dir, "test.txt"),
        "texts": pjoin(cfg.data.root_dir, "texts"),
        "features": pjoin(cfg.data.root_dir, "new_joint_vecs"),
    }


def _snapmogen_mean_std(cfg):
    """Load SnapMoGen normalization statistics."""
    paths = _snapmogen_paths(cfg)
    return np.load(paths["mean"]), np.load(paths["std"])


def _humanml3d_opt_and_stats(device):
    """Load HumanML3D evaluator options and normalization statistics."""
    wrapper_opt = get_opt(HUMANML3D_EVALUATOR_OPT, device)
    mean = np.load(pjoin(wrapper_opt.meta_dir, "mean.npy"))
    std = np.load(pjoin(wrapper_opt.meta_dir, "std.npy"))
    return wrapper_opt, mean, std


def _humanml3d_inv_transform(data, mean, std):
    """Inverse-normalize HumanML3D motion features."""
    if isinstance(data, np.ndarray):
        return data * std[: data.shape[-1]] + mean[: data.shape[-1]]
    if isinstance(data, torch.Tensor):
        dim = data.shape[-1]
        mean_t = torch.from_numpy(mean[:dim]).float().to(data.device)
        std_t = torch.from_numpy(std[:dim]).float().to(data.device)
        return data * std_t + mean_t
    raise TypeError("Expected np.ndarray or torch.Tensor")


def _snapmogen_plotter(cfg, dataset, device):
    """Build SnapMoGen FK and plotting functions."""
    paths = _snapmogen_paths(cfg)
    template_anim = bvh_io.load(paths["template_bvh"])
    skeleton = Skeleton(template_anim.offsets, template_anim.parents, device=device)

    def forward_kinematic_func(data):
        motions = dataset.inv_transform(data)
        return recover_pos_from_rot(motions, joints_num=cfg.data.joint_num, skeleton=skeleton)

    def plot_t2m(data, save_dir, captions=None, m_lengths=None, save_path=None):
        global_pos = forward_kinematic_func(data).detach().cpu().numpy()
        for i in range(len(global_pos)):
            path = save_path if save_path is not None else pjoin(save_dir, "%02d.mp4" % i)
            length = _as_int(m_lengths[i]) if m_lengths is not None else len(global_pos[i])
            title = captions[i] if captions is not None else "None"
            plot_3d_motion(path, kinematic_chain, global_pos[i, :length], title=title, fps=30, radius=100)

    return plot_t2m, forward_kinematic_func


def _humanml3d_plotter(mean, std):
    """Build HumanML3D plotting function."""

    def plot_t2m(data, save_dir, captions=None, m_lengths=None, save_path=None):
        if isinstance(data, torch.Tensor):
            data_np = data.float().cpu().detach().numpy()
        else:
            data_np = np.asarray(data, dtype=np.float32)
        data_np = _humanml3d_inv_transform(data_np, mean, std)
        for i in range(len(data_np)):
            path = save_path if save_path is not None else pjoin(save_dir, "%02d.mp4" % i)
            length = _as_int(m_lengths[i]) if m_lengths is not None else len(data_np[i])
            title = captions[i] if captions is not None else "None"
            joint = recover_from_ric(torch.from_numpy(data_np[i, :length]).float(), 22).numpy()
            plot_3d_motion(path, t2m_kinematic_chain, joint, title=title, fps=20, radius=2)

    return plot_t2m


def build_predictor_train_data(cfg, device):
    """Build predictor train/eval datasets and evaluator from cfg.data.name."""
    _require_supported_dataset(cfg)

    if is_humanml3d(cfg):
        paths = _humanml3d_paths(cfg)
        wrapper_opt, mean, std = _humanml3d_opt_and_stats(device)
        train_dataset = HumanML3DTextMotionDataset(cfg, mean, std, paths["train_ids"])
        val_dataset = HumanML3DTextMotionDataset(cfg, mean, std, paths["val_ids"])
        eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
        eval_loader, eval_dataset = get_dataset_motion_loader(
            HUMANML3D_EVALUATOR_OPT,
            32,
            "test",
            device=device,
            deterministic=True,
            shuffle=False,
            num_workers=0,
            seed=cfg.exp.seed,
        )
        return DatasetBundle(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            eval_dataset=eval_dataset,
            eval_loader=eval_loader,
            eval_wrapper=eval_wrapper,
            plot_t2m=_humanml3d_plotter(mean, std),
            forward_kinematic_func=None,
            evaluation_func=evaluation_scalemogen_hml,
            eval_tau=1.0,
        )

    paths = _snapmogen_paths(cfg)
    mean, std = _snapmogen_mean_std(cfg)
    train_dataset = SnapMoGenTextMotionDataset(
        cfg, mean, std, paths["train_mid"], paths["train_ids"], paths["captions"]
    )
    val_dataset = SnapMoGenTextMotionDataset(cfg, mean, std, paths["val_mid"], paths["val_ids"], paths["captions"])
    eval_dataset = SnapMoGenTextMotionDataset(cfg, mean, std, paths["val_mid"], paths["val_ids"], paths["captions"])
    eval_dataset.deterministic = True
    plot_t2m, fk_func = _snapmogen_plotter(cfg, train_dataset, device)
    eval_cfg = load_config(snapmogen_evaluator_config_path())
    eval_wrapper = EvaluatorWrapper(eval_cfg, device=device)
    eval_generator = torch.Generator()
    eval_generator.manual_seed(int(cfg.exp.seed))
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_cfg.matching_pool_size,
        drop_last=True,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        generator=eval_generator,
    )
    return DatasetBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_dataset=eval_dataset,
        eval_loader=eval_loader,
        eval_wrapper=eval_wrapper,
        plot_t2m=plot_t2m,
        forward_kinematic_func=fk_func,
        evaluation_func=evaluation_scalemogen,
        eval_tau=1.3,
    )


def build_vq_train_data(cfg, device):
    """Build VQ train/eval datasets and evaluator from cfg.data.name."""
    _require_supported_dataset(cfg)

    if is_humanml3d(cfg):
        paths = _humanml3d_paths(cfg)
        wrapper_opt, mean, std = _humanml3d_opt_and_stats(device)
        train_dataset = HumanML3DTextMotionDataset(cfg, mean, std, paths["train_ids"])
        val_dataset = HumanML3DTextMotionDataset(cfg, mean, std, paths["val_ids"])
        eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
        eval_loader, eval_dataset = get_dataset_motion_loader(HUMANML3D_EVALUATOR_OPT, 32, "test", device=device)
        return DatasetBundle(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            eval_dataset=eval_dataset,
            eval_loader=eval_loader,
            eval_wrapper=eval_wrapper,
            plot_t2m=_humanml3d_plotter(mean, std),
            forward_kinematic_func=None,
        )

    paths = _snapmogen_paths(cfg)
    mean, std = _snapmogen_mean_std(cfg)
    train_dataset = SnapMoGenTextMotionDataset(
        cfg, mean, std, paths["train_mid"], paths["train_ids"], paths["captions"]
    )
    val_dataset = SnapMoGenTextMotionDataset(cfg, mean, std, paths["val_mid"], paths["val_ids"], paths["captions"])
    eval_dataset = SnapMoGenTextMotionDataset(cfg, mean, std, paths["val_mid"], paths["val_ids"], paths["captions"])
    plot_t2m, fk_func = _snapmogen_plotter(cfg, train_dataset, device)
    eval_cfg = load_config(snapmogen_evaluator_config_path())
    eval_wrapper = EvaluatorWrapper(eval_cfg, device=device)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_cfg.matching_pool_size,
        drop_last=True,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
    )
    return DatasetBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_dataset=eval_dataset,
        eval_loader=eval_loader,
        eval_wrapper=eval_wrapper,
        plot_t2m=plot_t2m,
        forward_kinematic_func=fk_func,
    )


def build_eval_data(cfg, device):
    """Build text-to-motion evaluation loader and evaluator from cfg.data.name."""
    _require_supported_dataset(cfg)
    cfg.data.deterministic_eval = cfg.eval.get("deterministic_data", True)

    if is_humanml3d(cfg):
        wrapper_opt, mean, std = _humanml3d_opt_and_stats(device)
        eval_loader, eval_dataset = get_dataset_motion_loader(
            HUMANML3D_EVALUATOR_OPT,
            cfg.eval.batch_size,
            "test",
            device=device,
            deterministic=bool(cfg.eval.get("deterministic_data", True)),
            shuffle=bool(cfg.eval.get("shuffle", False)),
            num_workers=int(cfg.eval.get("num_workers", 0)),
            seed=cfg.exp.seed,
        )
        return DatasetBundle(
            eval_dataset=eval_dataset,
            eval_loader=eval_loader,
            eval_wrapper=EvaluatorModelWrapper(wrapper_opt),
            plot_t2m=_humanml3d_plotter(mean, std),
            evaluation_func=evaluation_scalemogen_hml,
            repeat_times=cfg.eval.get("repeat_times", 20),
            report_confidence=True,
        )

    paths = _snapmogen_paths(cfg)
    mean, std = _snapmogen_mean_std(cfg)
    eval_dataset = SnapMoGenTextMotionDataset(cfg, mean, std, paths["test_mid"], paths["test_ids"], paths["captions"])
    eval_shuffle = bool(cfg.eval.get("shuffle", False))
    eval_workers = int(cfg.eval.get("num_workers", 0))
    eval_generator = torch.Generator()
    eval_generator.manual_seed(int(cfg.exp.seed))
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=eval_shuffle,
        num_workers=eval_workers,
        pin_memory=True,
        generator=eval_generator,
    )
    plot_t2m, _ = _snapmogen_plotter(cfg, eval_dataset, device)
    eval_wrapper_cfg = load_config(snapmogen_evaluator_config_path())
    return DatasetBundle(
        eval_dataset=eval_dataset,
        eval_loader=eval_loader,
        eval_wrapper=EvaluatorWrapper(eval_wrapper_cfg, device=device),
        plot_t2m=plot_t2m,
        evaluation_func=evaluation_scalemogen,
        repeat_times=cfg.eval.get("repeat_times", 3),
        report_confidence=False,
    )


def parse_generation_entries(cfg, mode, text_file):
    """Parse generation prompts according to cfg.data.name."""
    _require_supported_dataset(cfg)
    if is_humanml3d(cfg):
        return _parse_humanml3d_test_entries(cfg)
    if mode == "casual":
        return _parse_snapmogen_casual_entries(text_file)
    return _parse_snapmogen_test_entries(cfg)


def _parse_snapmogen_casual_entries(text_file):
    """Parse SnapMoGen casual generation prompts."""
    entries = []
    with open(text_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            colon_idx = line.index(":")
            idx = line[:colon_idx].strip()
            parts = line[colon_idx + 1 :].strip().split("#")
            long_text = parts[1].strip()
            motion_length = int(parts[2].strip())
            entries.append(GenerationEntry(key=str(idx), prompt=long_text, motion_length=motion_length))
    return entries


def _parse_snapmogen_test_entries(cfg):
    """Parse SnapMoGen test split generation prompts."""
    paths = _snapmogen_paths(cfg)
    with open(paths["captions"], "r") as f:
        all_captions = json.load(f)
    entries = []
    with open(paths["test_ids"], "r") as f:
        for line in f:
            cid = line.strip()
            if not cid:
                continue
            mid, start, end = cid.split("#")
            motion_length = int(end) - int(start)
            if motion_length < cfg.data.min_motion_length:
                continue
            motion_length = min(motion_length, cfg.data.max_motion_length)
            motion_length = (motion_length // cfg.data.unit_length) * cfg.data.unit_length
            captions = all_captions.get(cid, {}).get("gpt", [])
            if not captions:
                continue
            entries.append(
                GenerationEntry(
                    key=cid,
                    prompt=captions[0],
                    motion_length=motion_length,
                    mid=mid,
                    start=int(start),
                    end=int(end),
                )
            )
    return entries


def _parse_humanml3d_test_entries(cfg):
    """Parse HumanML3D test split generation prompts."""
    paths = _humanml3d_paths(cfg)
    entries = []
    with open(paths["test_ids"], "r") as f:
        for line in f:
            mid = line.strip()
            if not mid:
                continue
            feat_path = pjoin(paths["features"], f"{mid}.npy")
            text_path = pjoin(paths["texts"], f"{mid}.txt")
            if not os.path.exists(feat_path) or not os.path.exists(text_path):
                continue
            motion_length = np.load(feat_path).shape[0]
            if motion_length < cfg.data.min_motion_length:
                continue
            motion_length = min(motion_length, cfg.data.max_motion_length)
            motion_length = (motion_length // cfg.data.unit_length) * cfg.data.unit_length
            with open(text_path, "r") as text_file:
                first_line = text_file.readline().strip()
            if not first_line:
                continue
            entries.append(GenerationEntry(key=mid, prompt=first_line.split("#")[0].strip(), motion_length=motion_length, mid=mid))
    return entries


class SnapMoGenGenerationRenderer:
    """Save SnapMoGen generated motions as MP4 and BVH files."""

    def __init__(self, cfg, device):
        paths = _snapmogen_paths(cfg)
        self.cfg = cfg
        self.device = device
        self.mean, self.std = _snapmogen_mean_std(cfg)
        self.mean_tensor = torch.tensor(self.mean, dtype=torch.float32, device=device)
        self.std_tensor = torch.tensor(self.std, dtype=torch.float32, device=device)
        self.template_anim = bvh_io.load(paths["template_bvh"])
        self.skeleton = Skeleton(self.template_anim.offsets, self.template_anim.parents, device=device)
        self.retargeter = RestPoseRetargeter()

    def prepare_dirs(self, gen_dir):
        """Create SnapMoGen generation output directories."""
        os.makedirs(pjoin(gen_dir, "bvh"), exist_ok=True)
        os.makedirs(pjoin(gen_dir, "mp4"), exist_ok=True)

    def forward_kinematic(self, data):
        """Convert normalized features to global joint positions and BVH channels."""
        motions = data * self.std_tensor + self.mean_tensor
        batch, _, _ = data.shape
        _, local_quats, root_pos = recover_bvh_from_rot(motions, self.cfg.data.joint_num, self.skeleton, keep_shape=False)
        _, global_pos = self.skeleton.fk_local_quat(local_quats, root_pos)
        global_pos = rearrange(global_pos, "(b l) j d -> b l j d", b=batch)
        local_quats = rearrange(local_quats, "(b l) j d -> b l j d", b=batch)
        root_pos = rearrange(root_pos, "(b l) d -> b l d", b=batch)
        return global_pos, local_quats, root_pos

    def render(self, gen_dir, entry, pred_motions, actual_len, title, suffix, save_full=False, include_gt=False):
        """Render one SnapMoGen prediction and optionally save full BVH outputs."""
        global_pos, local_quat, root_pos = self.forward_kinematic(pred_motions)
        mp4_path = pjoin(gen_dir, "mp4", f"{entry.safe_key}_{suffix}.mp4")
        plot_3d_motion(mp4_path, kinematic_chain, global_pos[0, :actual_len].detach().cpu().numpy(), title=title, fps=30, radius=100)

        if save_full:
            gen_anim = Animation(
                local_quat[0, :actual_len].detach().cpu().numpy(),
                repeat(root_pos[0, :actual_len].detach().cpu().numpy(), "i j -> i k j", k=len(self.template_anim)),
                self.template_anim.orients,
                self.template_anim.offsets,
                self.template_anim.parents,
                self.template_anim.names,
                self.template_anim.frametime,
            )
            gen_bvh_path = pjoin(gen_dir, "bvh", f"{entry.safe_key}_gen.bvh")
            bvh_io.save(
                gen_bvh_path,
                self.retargeter.rest_pose_retarget(gen_anim),
                names=gen_anim.names,
                frametime=gen_anim.frametime,
                order="xyz",
                quater=True,
            )
            gen_anim_ik, _, _ = foot_lock(gen_anim)
            gen_bvh_ik_path = pjoin(gen_dir, "bvh", f"{entry.safe_key}_gen_ik.bvh")
            bvh_io.save(
                gen_bvh_ik_path,
                self.retargeter.rest_pose_retarget(gen_anim_ik),
                names=gen_anim_ik.names,
                frametime=gen_anim_ik.frametime,
                order="xyz",
                quater=True,
            )
            if include_gt and entry.mid is not None:
                self._save_gt_bvh(gen_dir, entry)
        return mp4_path

    def _save_gt_bvh(self, gen_dir, entry):
        """Save the ground-truth SnapMoGen BVH for a test entry."""
        gt_raw = np.load(pjoin(self.cfg.data.feat_dir, f"{entry.mid}.npy"))[entry.start : entry.end]
        gt_raw_norm = (gt_raw - self.mean) / self.std
        gt_tensor = torch.tensor(gt_raw_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        gt_len = min(entry.end - entry.start, gt_tensor.shape[1])
        _, gt_local_quat, gt_root_pos = self.forward_kinematic(gt_tensor)
        gt_anim = Animation(
            gt_local_quat[0, :gt_len].detach().cpu().numpy(),
            repeat(gt_root_pos[0, :gt_len].detach().cpu().numpy(), "i j -> i k j", k=len(self.template_anim)),
            self.template_anim.orients,
            self.template_anim.offsets,
            self.template_anim.parents,
            self.template_anim.names,
            self.template_anim.frametime,
        )
        gt_bvh_path = pjoin(gen_dir, "bvh", f"{entry.safe_key}_gt.bvh")
        bvh_io.save(
            gt_bvh_path,
            self.retargeter.rest_pose_retarget(gt_anim),
            names=gt_anim.names,
            frametime=gt_anim.frametime,
            order="xyz",
            quater=True,
        )


class HumanML3DGenerationRenderer:
    """Save HumanML3D generated motions as MP4 and NPY files."""

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        _, self.mean, self.std = _humanml3d_opt_and_stats(device)

    def prepare_dirs(self, gen_dir):
        """Create HumanML3D generation output directories."""
        os.makedirs(pjoin(gen_dir, "mp4"), exist_ok=True)
        os.makedirs(pjoin(gen_dir, "npy"), exist_ok=True)

    def render(self, gen_dir, entry, pred_motions, actual_len, title, suffix, save_full=False, include_gt=False):
        """Render one HumanML3D prediction and optionally save full NPY output."""
        pred_np = _humanml3d_inv_transform(pred_motions[0, :actual_len].cpu().detach().numpy(), self.mean, self.std)
        joint_pos = recover_from_ric(torch.from_numpy(pred_np).float(), self.cfg.data.joint_num).numpy()
        mp4_path = pjoin(gen_dir, "mp4", f"{entry.safe_key}_{suffix}.mp4")
        plot_3d_motion(mp4_path, t2m_kinematic_chain, joint_pos, title=title, fps=20, radius=2)
        if save_full:
            np.save(pjoin(gen_dir, "npy", f"{entry.safe_key}.npy"), pred_motions[0, :actual_len].cpu().numpy())
        return mp4_path


def build_generation_renderer(cfg, device):
    """Return the renderer for cfg.data.name."""
    _require_supported_dataset(cfg)
    return HumanML3DGenerationRenderer(cfg, device) if is_humanml3d(cfg) else SnapMoGenGenerationRenderer(cfg, device)


def hstack_videos(input_paths, output_path):
    """Horizontally concatenate MP4 videos with ffmpeg."""
    cmd = ["ffmpeg", "-y"]
    for input_path in input_paths:
        cmd.extend(["-i", input_path])
    cmd.extend(["-filter_complex", f"hstack=inputs={len(input_paths)}", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", output_path])
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
