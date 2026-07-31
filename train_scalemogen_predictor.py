"""ScaleMoGen predictor training entrypoint."""

import os
import argparse
from os.path import join as pjoin
import gc
import math
import random
import time
import glob
import traceback
from collections import deque
from contextlib import nullcontext
from functools import partial
from typing import List, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from scalemogen.checkpoint import (
    create_scalemogen_predictor,
    disable_builtin_initializers,
    load_scalemogen_vq,
    predictor_model_name,
)
from scalemogen.config import apply_predictor_defaults
from scalemogen.dataset_dispatch import build_predictor_train_data
from config.load_config import load_config

import numpy as np
from utils.fixseeds import *

import shutil

import scalemogen.utils.dist as dist
from scalemogen.utils import misc
from scalemogen.utils.console import print_scalemogen_progress
from scalemogen.utils.save_and_load import CKPTSaver, auto_resume
from scalemogen.text import load_text_encoder

from transformers import T5EncoderModel, T5TokenizerFast

import torch._dynamo
torch._dynamo.config.suppress_errors = True

class BestCheckpointTracker:
    def __init__(self, mode='min', metric_name='fid', save_dir='.'):
        self.mode = mode
        self.metric_name = metric_name
        self.save_dir = save_dir
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_ckpt_path = None

    def update(self, score, current_ckpt_path, epoch, step):
        is_better = (score < self.best_score) if self.mode == 'min' else (score > self.best_score)
        
        if is_better:
            prev_best_path = self.best_ckpt_path
            self.best_score = score
            
            # Format: best_fid_ep_100_it_5000.pth
            new_filename = f'best_{self.metric_name}_ep_{epoch}_it_{step}.pth'
            new_path = os.path.join(self.save_dir, new_filename)
            self.best_ckpt_path = new_path
            
            print(
                f"[ScaleMoGen][Predictor] new_best_{self.metric_name}={score:.4f} file={new_filename}",
                flush=True,
            )
            
            try:
                shutil.copyfile(current_ckpt_path, new_path)
            except Exception as e:
                print(f"[ScaleMoGen][Predictor] failed_to_copy_best={new_path}: {e}", flush=True)
                return

            # Remove old best to keep only one
            if prev_best_path and prev_best_path != new_path and os.path.exists(prev_best_path):
                try:
                    os.remove(prev_best_path)
                    print(
                        f"[ScaleMoGen][Predictor] removed_previous_best={os.path.basename(prev_best_path)}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[ScaleMoGen][Predictor] failed_to_remove_previous_best={prev_best_path}: {e}", flush=True)


def encode_text_batch(text_tokenizer, text_encoder, captions, target_device):
    """Encode a caption batch into compact T5 features on the predictor device."""
    encoder_device = next(text_encoder.parameters()).device
    tokens = text_tokenizer(
        text=captions,
        max_length=text_tokenizer.model_max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    input_ids = tokens.input_ids.to(encoder_device, non_blocking=True)
    mask = tokens.attention_mask.to(encoder_device, non_blocking=True)

    with torch.inference_mode():
        text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()

    lens: List[int] = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
    Ltext = max(lens)

    kv_compact = []
    for len_i, feat_i in zip(lens, text_features.unbind(0)):
        kv_compact.append(feat_i[:len_i])
    kv_compact = torch.cat(kv_compact, dim=0)
    return (
        kv_compact.to(target_device, non_blocking=True),
        lens,
        cu_seqlens_k.to(target_device, non_blocking=True),
        Ltext,
    )


def compile_model(m, fast):
    if fast == 0:
        return m
    return torch.compile(m, mode={
        1: 'reduce-overhead',
        2: 'max-autotune',
        3: 'default',
    }[fast]) if hasattr(torch, 'compile') else m



def load_vq_model(cfg, device):
    vq_model, vq_cfg, _, ckpt_path = load_scalemogen_vq(cfg, device)
    print(f"[ScaleMoGen][Predictor] loaded_vq={vq_cfg.exp.name} ckpt={ckpt_path}", flush=True)
    return vq_model, vq_cfg



def main():
    # torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser(description="Train the ScaleMoGen predictor.")
    parser.add_argument("--config", default="config/train_scalemogen_predictor.yaml")
    args, _ = parser.parse_known_args()
    config_name = os.path.basename(args.config)

    cfg = apply_predictor_defaults(load_config(args.config))

    exp_name = cfg.exp.name

    cfg.exp.checkpoint_dir = pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'predictor', exp_name)

    #dist.init_distributed_mode(local_out_path=args.local_out_path, fork=False, timeout_minutes=day3 if int(os.environ.get('LONG_DBG', '0') or '0') > 0 else 30)
    #local_out_path = os.path.join(os.path.dirname(__file__), 'local_output') 
    
    local_out_path = cfg.exp.checkpoint_dir
    if not os.path.exists(local_out_path):
        os.makedirs(local_out_path)

    dist.init_distributed_mode(local_out_path=local_out_path, fork=False, timeout_minutes=30000)

    #cfg.predictor.local_out_path = os.path.join(os.path.dirname(__file__), 'local_output')
    cfg.predictor.local_out_path = cfg.exp.checkpoint_dir

    if cfg.exp.is_continue:
        saved_config_path = None
        for candidate in (config_name, "train_scalemogen_predictor.yaml", "train_scalemogen_predictor_hml.yaml"):
            candidate_path = pjoin(cfg.predictor.local_out_path, candidate)
            if os.path.exists(candidate_path):
                saved_config_path = candidate_path
                break
        if saved_config_path is None:
            raise FileNotFoundError(f"No saved predictor config found in {cfg.predictor.local_out_path}")
        n_cfg = apply_predictor_defaults(load_config(saved_config_path))
        n_cfg.exp.is_continue = True
        n_cfg.exp.device = cfg.exp.device
        n_cfg.exp.checkpoint_dir = cfg.predictor.local_out_path
        n_cfg.predictor = cfg.predictor
        cfg = n_cfg
        cfg.predictor.auto_resume = True
        # print(cfg)
    else:
        os.makedirs(cfg.predictor.local_out_path, exist_ok=True)
        shutil.copy(args.config, cfg.predictor.local_out_path)

    cfg.predictor.log_txt_path = os.path.join(cfg.predictor.local_out_path, 'log.txt')
    cfg.predictor.model_dir = pjoin(cfg.predictor.local_out_path, 'model')
    cfg.predictor.eval_dir = pjoin(cfg.predictor.local_out_path, 'animation')
    cfg.predictor.log_dir = pjoin(cfg.exp.root_log_dir, cfg.data.name, 'scalemogen', exp_name)
    cfg.predictor.exp_name = exp_name

    os.makedirs(cfg.predictor.model_dir, exist_ok=True)
    os.makedirs(cfg.predictor.eval_dir, exist_ok=True)
    os.makedirs(cfg.predictor.log_dir, exist_ok=True)

    fixseed(cfg.exp.seed)


    #if cfg.exp.device != 'cpu':
    #    torch.cuda.set_device(cfg.exp.device)
    #device = torch.device(cfg.exp.device)

    local_rank = int(dist.get_local_rank())
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")


    # torch.autograd.set_detect_anomaly(True)

    vq_model, vq_cfg = load_vq_model(cfg, device=device)
    cfg.vq = vq_cfg.quantizer
    data_bundle = build_predictor_train_data(cfg, device)
    train_dataset = data_bundle.train_dataset
    val_dataset = data_bundle.val_dataset
    eval_loader = data_bundle.eval_loader
    eval_wrapper = data_bundle.eval_wrapper
    plot_t2m = data_bundle.plot_t2m
    forward_kinematic_func = data_bundle.forward_kinematic_func
    evaluation_func = data_bundle.evaluation_func
    cfg.predictor.eval_tau = data_bundle.eval_tau
    cfg.predictor.eval_sampling_device = getattr(cfg.predictor, "eval_sampling_device", "cuda")
    cfg.predictor.eval_sampling_method = getattr(cfg.predictor, "eval_sampling_method", "multinomial")

    print(f"[ScaleMoGen][Predictor] world_size={dist.get_world_size()} local_rank={local_rank}", flush=True)

    if cfg.predictor.lbs:
        bs_per_gpu = cfg.predictor.lbs / cfg.predictor.ac
    else:
        bs_per_gpu = cfg.predictor.bs / cfg.predictor.ac / dist.get_world_size()
    bs_per_gpu = round(bs_per_gpu)
    cfg.predictor.batch_size = bs_per_gpu
    cfg.predictor.bs = cfg.predictor.glb_batch_size = cfg.predictor.batch_size * dist.get_world_size()
    cfg.predictor.workers = min(cfg.predictor.workers, bs_per_gpu)

    cfg.predictor.alng = float(cfg.predictor.alng)
    if cfg.predictor.alng < 0:
        cfg.predictor.alng = cfg.predictor.aln
    

    cfg.predictor.r_accu = 1 / cfg.predictor.ac   # gradient accumulation
    #cfg.predictor.data_load_reso = None
    #cfg.predictor.rand |= args.seed is None
    cfg.predictor.sche = 'lin0'
    if cfg.predictor.wp == 0:
        cfg.predictor.wp = cfg.predictor.ep * 1/100

    cfg.predictor.ada = cfg.predictor.ada or ('0.9_0.96' if cfg.predictor.gpt_training else '0.5_0.9')
    #cfg.predictor.dada = cfg.predictor.dada or cfg.predictor.ada
    cfg.predictor.opt = cfg.predictor.opt.lower().strip()
    
    

    #cfg.predictor.dblr = cfg.predictor.dblr or cfg.predictor.gblr
    cfg.predictor.dblr = cfg.predictor.gblr

    
    cfg.predictor.gblr = float(cfg.predictor.gblr)
    cfg.predictor.dblr = float(cfg.predictor.dblr)
    cfg.predictor.tblr = float(cfg.predictor.tblr)

    cfg.predictor.glr = cfg.predictor.ac * cfg.predictor.gblr * cfg.predictor.glb_batch_size / 256
    cfg.predictor.dlr = cfg.predictor.ac * cfg.predictor.dblr * cfg.predictor.glb_batch_size / 256
    cfg.predictor.tlr = cfg.predictor.ac * cfg.predictor.tblr * cfg.predictor.glb_batch_size / 256

    cfg.predictor.glr = cfg.predictor.glr / 64.
    cfg.predictor.dlr = cfg.predictor.dlr / 64.
    cfg.predictor.tlr = cfg.predictor.tlr / 64.

    cfg.predictor.gwde = cfg.predictor.gwde or cfg.predictor.gwd
    cfg.predictor.dwde = cfg.predictor.dwde or cfg.predictor.dwd
    cfg.predictor.twde = cfg.predictor.twde or cfg.predictor.twd

    cfg.predictor.cdec = True   # clip_decay_ratio = (0.3 ** (20 * progress) + 0.2) if args.cdec else 1



    train_loader = DataLoader(train_dataset, batch_size=cfg.predictor.batch_size, drop_last=True, num_workers=cfg.predictor.workers,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.predictor.batch_size, drop_last=True, num_workers=cfg.predictor.workers,
                              shuffle=True, pin_memory=True)
    vbs = round(cfg.predictor.batch_size * 1.5)
    print(
            f"[ScaleMoGen][Predictor] dataset={cfg.data.name} batch_size={cfg.predictor.batch_size} val_batch_size={vbs}",
        flush=True,
    )
    ld_val = math.ceil(50000 / vbs)
    iters_train = len(train_loader)

    saver = CKPTSaver(dist.is_master(), eval_milestone=None, max_to_keep=5)


    check_vq_performance = False
    if check_vq_performance:
        assert "scalemogen" in vq_cfg.exp.name
        vq_cfg.exp.is_train = False
        from scalemogen.training import ScaleMoGenVQTrainer
        vq_trainer = ScaleMoGenVQTrainer(vq_cfg, vq_model, device=device)
        vq_trainer.eval(train_loader, val_loader, eval_loader, eval_wrapper, plot_t2m, forward_kinematic_func)


    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from scalemogen.models.predictor.motion_transformer import ScaleMoGenBlockGroup, ScaleMoGenTransformer
    from scalemogen.models.predictor.ema import get_ema_model
    from scalemogen.models.predictor.init_param import init_weights
    from scalemogen.utils.amp_opt import AmpOptimizer
    from scalemogen.utils.lr_control import filter_params
    
    disable_builtin_initializers()

    model_str = predictor_model_name(cfg)
    print(f"[ScaleMoGen][Predictor] model={model_str}", flush=True)
    gpt_wo_ddp: ScaleMoGenTransformer = create_scalemogen_predictor(cfg, vq_model, cfg.predictor.batch_size)

    
    if cfg.predictor.use_fsdp_model_ema:
        gpt_wo_ddp_ema = get_ema_model(gpt_wo_ddp)
    else:
        gpt_wo_ddp_ema = None
    gpt_wo_ddp = gpt_wo_ddp.to(device)

    if cfg.predictor.tini < 0:
        cfg.predictor.tini = math.sqrt(1 / gpt_wo_ddp.C / 9)
    init_weights(gpt_wo_ddp, other_std=cfg.predictor.tini)

    gpt_wo_ddp.special_init(aln_init=float(cfg.predictor.aln), aln_gamma_init=float(cfg.predictor.alng), scale_head=cfg.predictor.hd0, scale_proj=cfg.predictor.diva)

    ndim_dict = {name: para.ndim for name, para in gpt_wo_ddp.named_parameters() if para.requires_grad}
    
    gpt_uncompiled = gpt_wo_ddp
    gpt_wo_ddp = compile_model(gpt_wo_ddp, cfg.predictor.tfast)

    gpt_ddp_ema = None

    if cfg.predictor.zero:
        from torch.distributed.fsdp import ShardingStrategy
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy
        

        # use mix prec: https://github.com/pytorch/pytorch/issues/76607
        if gpt_wo_ddp.num_block_chunks == 1:  # no chunks
            auto_wrap_policy = ModuleWrapPolicy([type(gpt_wo_ddp.unregistered_blocks[0]), ])
        else:
            auto_wrap_policy = ModuleWrapPolicy([ScaleMoGenBlockGroup, ])
        
        if cfg.predictor.enable_hybrid_shard:
            from torch.distributed.device_mesh import init_device_mesh
            sharding_strategy = ShardingStrategy.HYBRID_SHARD if cfg.predictor.zero == 3 else ShardingStrategy._HYBRID_SHARD_ZERO2
            world_size = dist.get_world_size()
            assert world_size % cfg.predictor.inner_shard_degree == 0
            assert cfg.predictor.inner_shard_degree > 1 and cfg.predictor.inner_shard_degree < world_size
            device_mesh = init_device_mesh('cuda', (world_size // cfg.predictor.inner_shard_degree, cfg.predictor.inner_shard_degree))
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD if cfg.predictor.zero == 3 else ShardingStrategy.SHARD_GRAD_OP
            device_mesh = None
        print(
            f"[ScaleMoGen][Predictor] fsdp_init zero={cfg.predictor.zero} "
            f"sharding={sharding_strategy} auto_wrap={auto_wrap_policy}",
            flush=True,
        )
        
        gpt_ddp: FSDP = FSDP(
            gpt_wo_ddp, 
            device_id=dist.get_local_rank(),
            #device_id=device,
            sharding_strategy=sharding_strategy, 
            mixed_precision=None,
            auto_wrap_policy=auto_wrap_policy, 
            #use_orig_params=True, 
            #sync_module_states=True, 
            use_orig_params=False,
            sync_module_states=False,
            limit_all_gathers=True,
            #device_mesh=device_mesh,
        ).to(device)
        
        if cfg.predictor.use_fsdp_model_ema:
            gpt_wo_ddp_ema = gpt_wo_ddp_ema.to(device)
            gpt_ddp_ema: FSDP = FSDP(
                gpt_wo_ddp_ema, 
                #device_id=dist.get_local_rank(),
                device_id=device,
                sharding_strategy=sharding_strategy, 
                mixed_precision=None,
                auto_wrap_policy=auto_wrap_policy, 
                use_orig_params=getattr(cfg.predictor, "fsdp_orig", False),
                sync_module_states=True, 
                limit_all_gathers=True,
            )

    else:
        ddp_class = DDP if dist.initialized() else misc.NullDDP
        gpt_ddp: DDP = ddp_class(gpt_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=cfg.predictor.dbg, broadcast_buffers=False)
    torch.cuda.synchronize()


    # =============== build optimizer ===============
    nowd_keys = set()
    if cfg.predictor.nowd >= 1:
        nowd_keys |= {
            'cls_token', 'start_token', 'task_token', 'cfg_uncond',
            'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
            'gamma', 'beta',
            'ada_gss', 'moe_bias',
            'scale_mul',
            'text_proj_for_sos.ca.mat_q',
        }
        nowd_keys |= {
            'cls_token', 'start_token', 'task_token', 'cfg_uncond',
            'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
            'gamma', 'beta',
            'ada_gss', 'moe_bias',
            'scale_mul',
            'text_proj_for_sos.ca.mat_q',
        }
    if cfg.predictor.nowd >= 2:
        nowd_keys |= {'class_emb', 'embedding'}
    
    names, paras, para_groups = filter_params(gpt_ddp if cfg.predictor.zero else gpt_wo_ddp, ndim_dict, nowd_keys=nowd_keys)
    del ndim_dict
    if '_' in cfg.predictor.ada:
        beta0, beta1 = map(float, cfg.predictor.ada.split('_'))
    else:
        beta0, beta1 = float(cfg.predictor.ada), -1

    
    cfg.predictor.device = device
    
    opt_clz = {
        'sgd':   partial(torch.optim.SGD, momentum=beta0, nesterov=True),
        'adam':  partial(torch.optim.AdamW, betas=(beta0, beta1)),
        'adamw': partial(torch.optim.AdamW, betas=(beta0, beta1)),
    }[cfg.predictor.opt]
    opt_kw = dict(lr=cfg.predictor.tlr, weight_decay=0)
    if cfg.predictor.oeps: opt_kw['eps'] = cfg.predictor.oeps
    print(f"[ScaleMoGen][Predictor] optimizer={opt_clz} options={opt_kw}", flush=True)
    gpt_optim = AmpOptimizer('gpt', cfg.predictor.fp16, opt_clz(params=para_groups, **opt_kw), gpt_ddp if cfg.predictor.zero else gpt_wo_ddp, cfg.predictor.r_accu, cfg.predictor.tclip, cfg.predictor.zero)
    del names, paras, para_groups


    if cfg.predictor.online_t5:
        text_tokenizer, text_encoder = load_text_encoder(
            cfg,
            device,
            dtype_name=cfg.predictor.get("text_encoder_dtype", "fp32"),
            encoder_device=cfg.predictor.get("text_encoder_device", cfg.exp.device),
        )
        print(
            f"[ScaleMoGen][Predictor] loading_t5={cfg.predictor.t5_path} "
            f"dtype={cfg.predictor.get('text_encoder_dtype', 'fp32')} "
            f"device={next(text_encoder.parameters()).device}",
            flush=True,
        )
    else:
        text_tokenizer = text_encoder = None

    from scalemogen.training import ScaleMoGenPredictorTrainer

    cfg.predictor.viz_every_n_steps = getattr(cfg.predictor, 'viz_every_n_steps', 5000)

    trainer = ScaleMoGenPredictorTrainer(
        is_visualizer=dist.is_visualizer(), device=device, raw_scale_schedule=cfg.predictor.scale_schedule, resos=cfg.predictor.resos,
        vae_local=vq_model, gpt_wo_ddp=gpt_wo_ddp, gpt=gpt_ddp, ema_ratio=cfg.predictor.tema, max_it=iters_train * cfg.predictor.ep,
        gpt_opt=gpt_optim, label_smooth=cfg.predictor.ls, z_loss_ratio=cfg.predictor.lz, eq_loss=cfg.predictor.eq, xen=cfg.predictor.xen,
        dbg_unused=cfg.predictor.dbg, zero=cfg.predictor.zero, vae_type=cfg.predictor.vae_type,
        reweight_loss_by_scale=cfg.predictor.reweight_loss_by_scale, gpt_wo_ddp_ema=gpt_wo_ddp_ema, 
        gpt_ema=gpt_ddp_ema, use_fsdp_model_ema=cfg.predictor.use_fsdp_model_ema, is_train=True, log_dir=cfg.predictor.log_dir, other_args=cfg.predictor,
        plot_t2m_func=plot_t2m, train_loader=train_loader, val_loader=val_loader,
    )


    # auto resume from broken experiment
    auto_resume_info, start_ep, start_it, acc_str, eval_milestone, trainer_state, args_state = auto_resume(cfg.predictor, 'ar-ckpt*.pth')
    print(
        f"[ScaleMoGen][Predictor] global_batch_size={cfg.predictor.glb_batch_size} "
        f"local_batch_size={cfg.predictor.batch_size}",
        flush=True,
    )
    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True) # don't load vae again
        for state in trainer.gpt_opt.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    
    start_it = start_it % iters_train
    print(f"[ScaleMoGen][Predictor] resume_ep={start_ep} start_it={start_it} iters_per_epoch={iters_train}", flush=True)
    
    del vq_model, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim
    dist.barrier()

    gc.collect(), torch.cuda.empty_cache()

    #world_size = int(os.environ["WORLD_SIZE"])
    enable_timeline_sdk = False
    start_time, min_L_mean, min_L_tail, max_acc_mean, max_acc_tail = time.time(), 999., 999., -1., -1.
    last_val_loss_mean, best_val_loss_mean, last_val_acc_mean, best_val_acc_mean = 999., 999., 0., 0.
    last_val_loss_tail, best_val_loss_tail, last_val_acc_tail, best_val_acc_tail = 999., 999., 0., 0.
    seg5 = np.linspace(1, cfg.predictor.ep, 5+1, dtype=int).tolist()
    logging_params_milestone: List[int] = np.linspace(1, cfg.predictor.ep, 10+1, dtype=int).tolist()
    milestone_ep_feishu_log = set(seg5[:])
    vis_milestone_ep = set(seg5[:]) | set(x for x in (2, 4, 8, 16) if x <= cfg.predictor.ep)
    for x in [6, 12, 3, 24, 18, 48, 72, 96]:
        if len(vis_milestone_ep) < 10 and x <= cfg.predictor.ep:
            vis_milestone_ep.add(x)
    
    PARA_EMB, PARA_ALN, PARA_OT = 0, 0, 0
    for n, p in trainer.gpt_wo_ddp.named_parameters():
        if not p.requires_grad: continue
        if any(k in n for k in ('class_emb', 'pos_1LC', 'lvl_embed')):
            PARA_EMB += p.numel()
        elif any(k in n for k in ('ada_lin',)):
            PARA_ALN += p.numel()
        else:
            PARA_OT += p.numel()
    PARA_ALL = PARA_EMB + PARA_ALN + PARA_OT

    #trainer.gpt_opt.log_param(ep=-1)
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    ep_lg = max(1, cfg.predictor.ep // 10) if cfg.predictor.ep <= 100 else max(1, cfg.predictor.ep // 20)
    
    # Initialize Best Checkpoint trackers
    best_fid_tracker = BestCheckpointTracker(mode='min', metric_name='fid', save_dir=cfg.predictor.model_dir)
    best_match_tracker = BestCheckpointTracker(mode='max', metric_name='alignment', save_dir=cfg.predictor.model_dir)

    # ============================================= epoch loop begins =============================================
    L_mean, L_tail = -1, -1
    epochs_loss_nan = 0
    # build wandb logger
    #if dist.is_master():
    #    wandb_utils.wandb.init(project=cfg.predictor.project_name, name=cfg.predictor.exp_name, config={})

    for ep in range(start_ep, cfg.predictor.ep):
        if ep % ep_lg == 0 or ep == start_ep:
            print(
                f"[ScaleMoGen][Predictor] ep={ep:04d} resumed_from_ep={start_ep} "
                f"resumed_from_it={start_it} status={acc_str}",
                flush=True,
            )
        # set epoch for dataloader
        #if cfg.predictor.use_streaming_dataset:
        #    train_loader.dataset.set_epoch(ep)

        # [train one epoch]
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep=ep,
            is_first_ep=ep == start_ep,
            start_it=start_it if ep == start_ep else 0,
            me=None,
            saver=saver,
            args=cfg.predictor,
            ld_or_itrt=iter(train_loader),
            iters_train=iters_train,
            text_tokenizer=text_tokenizer, text_encoder=text_encoder,
            trainer=trainer,
            logging_params_milestone=logging_params_milestone,
            enable_timeline_sdk=enable_timeline_sdk,
            eval_loader=eval_loader,
            eval_wrapper=eval_wrapper,
            evaluation_func=evaluation_func,
            best_fid_tracker=best_fid_tracker,
            best_match_tracker=best_match_tracker,
        )

        # [update the best loss or acc]
        L_mean, L_tail, acc_mean, acc_tail, grad_norm = stats['Lm'], stats['Lt'], stats['Accm'], stats['Acct'], stats['tnm']
        min_L_mean, max_acc_mean, max_acc_tail = min(min_L_mean, L_mean), max(max_acc_mean, acc_mean), max(max_acc_tail, acc_tail)
        if L_tail != -1:
            min_L_tail = min(min_L_tail, L_tail)

        # Run validation and log metrics
        if dist.is_master():
            val_stats = run_validation_epoch(ep, trainer, val_loader, text_tokenizer, text_encoder, cfg.predictor)
            
            # Log validation metrics to TensorBoard/wandb
            if hasattr(trainer, 'logger') and trainer.logger is not None:
                val_lm = val_stats.get('Lm_val', -1)
                val_accm = val_stats.get('Accm_val', -1)
                val_acct = val_stats.get('Acct_val', -1)
                if val_lm != -1:
                    trainer.logger.add_scalar('val/L_mean', val_lm, ep)
                if val_accm != -1:
                    trainer.logger.add_scalar('val/Acc_token_mean', val_accm, ep)
                if val_acct != -1:
                    trainer.logger.add_scalar('val/Acc_bit_mean', val_acct, ep)

            # Log to console
            print_scalemogen_progress(
                "Predictor",
                "val",
                epoch=ep,
                metrics={
                    "Lm": val_stats.get("Lm_val", -1),
                    "Accm": val_stats.get("Accm_val", -1),
                    "Acct": val_stats.get("Acct_val", -1),
                },
            )
            
            last_val_loss_mean = val_stats.get('Lm_val', 999)
            
            # --- Start of modification: Save best checkpoint ---
            if last_val_loss_mean < best_val_loss_mean:
                best_val_loss_mean = last_val_loss_mean
                print(f"[ScaleMoGen][Predictor] new_best_val_loss={best_val_loss_mean:.4f}; saving best checkpoint", flush=True)
                try:
                    # Find the latest checkpoint saved by the saver
                    list_of_files = glob.glob(os.path.join(cfg.predictor.model_dir, 'ar-ckpt-*.pth'))
                    if list_of_files:
                        latest_file = max(list_of_files, key=os.path.getctime)
                        best_ckpt_path = os.path.join(cfg.predictor.model_dir, 'ckpt-best.pth')
                        shutil.copyfile(latest_file, best_ckpt_path)
                        print(f"[ScaleMoGen][Predictor] saved_best={best_ckpt_path}", flush=True)
                except Exception as e:
                    print(f"[ScaleMoGen][Predictor] failed_to_save_best={e}", flush=True)
            # --- End of modification ---

            last_val_acc_mean = val_stats.get('Accm_val', 0)
            best_val_acc_mean = max(best_val_acc_mean, last_val_acc_mean)


def run_validation_epoch(ep, trainer, val_loader, text_tokenizer, text_encoder, args):
    trainer.gpt.eval()
    me = misc.MetricLogger()
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
    header = f'[ScaleMoGen][Predictor][val] ep={ep:04d}'

    original_backward_clip_step = trainer.gpt_opt.backward_clip_step
    trainer.gpt_opt.backward_clip_step = lambda *args, **kwargs: (torch.tensor(0.0), 0.0)

    for it, data in me.log_every(0, len(val_loader), iter(val_loader), args.log_freq, args.log_every_iter, header):
        g_it = ep * len(val_loader) + it
        
        # Data preparation
        captions, inp, m_lens = data
        text_cond_tuple = encode_text_batch(text_tokenizer, text_encoder, captions, args.device)
        inp = inp.to(args.device, non_blocking=True)

        trainer.train_step(
            ep=ep, it=it, g_it=g_it, stepping=False, clip_decay_ratio=1.0,
            metric_lg=me, 
            logging_params=False, 
            inp_B3HW=inp, 
            text_cond_tuple=text_cond_tuple,
            m_lens=m_lens,
            args=args,
            is_validation=True,
        )
    
    trainer.gpt_opt.backward_clip_step = original_backward_clip_step
    
    me.synchronize_between_processes()
    trainer.gpt.train()
    val_stats = {f'{k}_val': meter.global_avg for k, meter in me.meters.items()}
    return val_stats

g_speed_ls = deque(maxlen=128)

def train_one_ep(
    ep: int, is_first_ep: bool, start_it: int, me: misc.MetricLogger,
    saver: CKPTSaver, args, ld_or_itrt, iters_train: int, 
    text_tokenizer: T5TokenizerFast, text_encoder: T5EncoderModel, trainer, logging_params_milestone, enable_timeline_sdk: bool,
    eval_loader=None, eval_wrapper=None, evaluation_func=None, best_fid_tracker=None, best_match_tracker=None,
):
    # IMPORTANT: import heavy packages after the Dataloader object creation/iteration to avoid OOM
    from scalemogen.training import ScaleMoGenPredictorTrainer
    from scalemogen.utils.lr_control import lr_wd_annealing
    trainer: ScaleMoGenPredictorTrainer
    
    step_cnt = 0
    header = f'[ScaleMoGen][Predictor][train] ep={ep:04d}/{args.ep:04d}'
    
    with misc.Low_GPU_usage(files=[args.log_txt_path], sleep_secs=20, verbose=True) as telling_dont_kill:
        last_touch = time.time()
        g_it, max_it = ep * iters_train, args.ep * iters_train
        
        maybe_record_function = nullcontext
        trainer.gpt_wo_ddp.maybe_record_function = maybe_record_function
        
        last_t_perf = time.time()
        speed_ls: deque = g_speed_ls
        FREQ = min(args.prof_freq, iters_train//2-1)
        NVIDIA_IT_PLUS_1 = set(FREQ*i for i in (1, 2, 3, 4, 6, 8))
        ranges = set([2 ** i for i in range(20)])
        if ep <= 1: ranges |= {1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40}
        PRINTABLE_IT_PLUS_1 = set(FREQ*i for i in ranges)

        me = misc.MetricLogger()
        [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{value:.2g}')) for x in ['tlr']]
        [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['tnm']]
        [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
        [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
        # ============================================= iteration loop begins =============================================
        for it, data in me.log_every(start_it, iters_train, ld_or_itrt, args.log_freq, args.log_every_iter, header):
            g_it = ep * iters_train + it

            # calling inc_step to sync the global_step
            if enable_timeline_sdk:
                ndtimeline.inc_step()

            if (it+1) % FREQ == 0:
                speed_ls.append((time.time() - last_t_perf) / FREQ)
                last_t_perf = time.time()

            if (g_it+1) % args.save_model_iters_freq == 0:
                with misc.Low_GPU_usage(files=[args.log_txt_path], sleep_secs=3, verbose=True):
                    saver.sav(
                        args=args,
                        g_it=g_it + 1,
                        next_ep=ep,
                        next_it=it + 1,
                        trainer=trainer,
                        acc_str='[todo]',
                        eval_milestone=None,
                    )
                
                if dist.is_master() and eval_loader is not None and evaluation_func is not None and best_fid_tracker is not None:
                    print(f"[ScaleMoGen][Predictor][eval] it={g_it+1:07d} start", flush=True)
                    try:
                        # Find the latest checkpoint saved
                        list_of_files = glob.glob(os.path.join(args.model_dir, 'ar-ckpt-*.pth'))
                        ckpt_path = max(list_of_files, key=os.path.getctime) if list_of_files else None

                        if ckpt_path:
                            # Use default eval params
                            fid, diversity, r_precision, matching_score, multimodality = evaluation_func(
                                out_dir=args.eval_dir,
                                val_loader=eval_loader,
                                predictor=trainer.gpt_wo_ddp,
                                vq_model=trainer.vae_local,
                                text_tokenizer=text_tokenizer,
                                text_encoder=text_encoder,
                                eval_wrapper=eval_wrapper,
                                device=args.device,
                                repeat_id=0,
                                cfg_val=7.0,
                                tau_val=getattr(args, "eval_tau", 1.3),
                                sampling_per_bits=1,
                                cfg_insertion_layer=0,
                                enable_positive_prompt=0,
                                cal_mm=False,
                                use_bf16=getattr(args, "use_bf16_eval", False),
                                seed_mode="deterministic",
                                base_seed=getattr(args, "seed", 0),
                                sampling_device=getattr(args, "eval_sampling_device", "cuda"),
                                sampling_method=getattr(args, "eval_sampling_method", "multinomial"),
                            )
                            print_scalemogen_progress(
                                "Predictor",
                                "eval",
                                step=g_it + 1,
                                metrics={"FID": fid, "matching": matching_score},
                            )
                            best_fid_tracker.update(fid, ckpt_path, ep, it+1)
                            best_match_tracker.update(matching_score, ckpt_path, ep, it+1)
                            
                            # Log to TensorBoard
                            if hasattr(trainer, 'logger') and trainer.logger is not None:
                                trainer.logger.add_scalar('eval/FID', fid, g_it+1)
                                trainer.logger.add_scalar('eval/Alignment', matching_score, g_it+1)
                    except Exception as e:
                        print(f"[ScaleMoGen][Predictor][eval] failed={e}", flush=True)
                        traceback.print_exc()
            
            with maybe_record_function('before_train'):
                # [get data]
                captions, inp, m_lens = data
                text_cond_tuple = encode_text_batch(text_tokenizer, text_encoder, captions, args.device)
                inp = inp.to(args.device, non_blocking=True)
                if it > start_it + 10:
                    telling_dont_kill.early_stop()
                
                # [logging]
                args.cur_it = f'{it+1}/{iters_train}'
                args.last_wei_g = me.meters['tnm'].median
                if dist.is_local_master() and (it >= start_it + 10) and (time.time() - last_touch > 90):
                    _, args.remain_time, args.finish_time = me.iter_time.time_preds(max_it - g_it + (args.ep - ep) * 15)      # +15: other cost
                    #args.dump_log()
                    last_touch = time.time()
                
                # [schedule learning rate]
                wp_it = args.wp * iters_train
                min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(args.sche, trainer.gpt_opt.optimizer, args.tlr, args.twd, args.twde, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe)
                
                # [get scheduled hyperparameters]
                progress = g_it / (max_it - 1)
                clip_decay_ratio = (0.3 ** (20 * progress) + 0.2) if args.cdec else 1
                
                stepping = (g_it + 1) % args.ac == 0
                step_cnt += int(stepping)
            
            with maybe_record_function('in_training'):
                grad_norm_t, scale_log2_t = trainer.train_step(
                    ep=ep, it=it, g_it=g_it, stepping=stepping, clip_decay_ratio=clip_decay_ratio,
                    metric_lg=me, 
                    logging_params=stepping and step_cnt == 1 and (ep < 4 or ep in logging_params_milestone), 
                    inp_B3HW=inp, 
                    text_cond_tuple=text_cond_tuple,
                    m_lens=m_lens,
                    args=args,
                )

            if dist.is_master() and (g_it + 1) % args.viz_every_n_steps == 0:
                print(f"[ScaleMoGen][Predictor][viz] it={g_it+1:07d} start", flush=True)
                trainer.visualize_motion(ep, g_it + 1, text_tokenizer, text_encoder, tag='train')
                trainer.visualize_motion(ep, g_it + 1, text_tokenizer, text_encoder, tag='val')
            
            with maybe_record_function('after_train'):
                me.update(tlr=max_tlr)
    # ============================================= iteration loop ends =============================================
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)  # +15: other cost



if __name__ == '__main__':
    try:
        main()
    except Exception as _e:
        time.sleep(dist.get_rank() * 1 + random.random() * 0.5)
        try:
            # noinspection PyArgumentList
            print(f"[ScaleMoGen][Predictor][rank={dist.get_rank():02d}] error_type={type(_e).__name__}", flush=True)
        except:
            try: print(f"[ScaleMoGen][Predictor][rank={dist.get_rank():02d}] error_type={type(_e).__name__}", flush=True)
            except: pass
        if dist.is_master():
            print(f"[ScaleMoGen][Predictor][error]\n{_e}", flush=True)
            traceback.print_exc()
        raise _e
    #finally:
    #    misc.os_system(f'rm -rf {wait1}')
    #    dist.finalize()
    #    if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(sys.stderr, dist.BackupStreamToFile):
    #        sys.stdout.close(), sys.stderr.close()
