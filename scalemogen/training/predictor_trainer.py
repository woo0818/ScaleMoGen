"""ScaleMoGen predictor trainer.

Code provenance: adapted from the Infinity autoregressive training pipeline.
Source repository: https://github.com/FoundationVision/Infinity
"""

import gc
import random
import time
from functools import partial
from pprint import pformat
from typing import List, Optional, Tuple, Union
import os.path as osp

import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as tdist
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
import cv2
from einops import rearrange, repeat

import scalemogen.utils.dist as dist
from scalemogen.models.predictor.motion_transformer import ScaleMoGenTransformer
from scalemogen.models.predictor.ema import update_ema
from scalemogen.models.predictor.bit_correction import ScaleMoGenBitCorrection
from scalemogen.generation import coarse_to_fine_chain_from_vq, scale_schedule_from_vq
# Legacy arg and wandb helpers are intentionally not imported here.
from scalemogen.utils import misc# , wandb_utils
from scalemogen.utils.amp_opt import AmpOptimizer
from scalemogen.utils.dynamic_resolution import dynamic_resolution_h_w

from os.path import join as pjoin
import os
from tools.run_scalemogen import gen_one_motion

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
fulloptstate_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

class ScaleMoGenPredictorTrainer(object):
    def __init__(
        self, is_visualizer: bool, device, raw_scale_schedule: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local, gpt_wo_ddp: ScaleMoGenTransformer, gpt: DDP, ema_ratio: float, max_it: int,
        gpt_opt: AmpOptimizer, label_smooth: float, z_loss_ratio: float, eq_loss: int, xen: bool,
        dbg_unused=False,zero=0, vae_type=True, reweight_loss_by_scale=False,
        gpt_wo_ddp_ema=None, gpt_ema=None, use_fsdp_model_ema=False, is_train=False, log_dir=None, other_args=None,
        plot_t2m_func=None, train_loader=None, val_loader=None,
    ):
        super().__init__()
        self.dbg_unused = dbg_unused
        self.device = device
        
        self.zero = zero
        self.vae_type = vae_type
        
        self.gpt: Union[DDP, FSDP, nn.Module]
        self.gpt, self.vae_local, self.quantize_local = gpt, vae_local, vae_local.quantizer2d
        self.gpt_opt: AmpOptimizer = gpt_opt
        self.gpt_wo_ddp: Union[ScaleMoGenTransformer, torch._dynamo.eval_frame.OptimizedModule] = gpt_wo_ddp  # after torch.compile
        self.gpt_wo_ddp_ema = gpt_wo_ddp_ema
        self.gpt_ema = gpt_ema
        self.coarse_to_fine_chain = self.quantize_local.coarse_to_fine_chain
        self.max_motion_length = self.vae_local.cfg.data.max_motion_length
        self.len_scale_factor = self.quantize_local.len_scale_factor
        self.bitwise_self_correction = ScaleMoGenBitCorrection(self.vae_local, other_args)
        self.use_fsdp_model_ema = use_fsdp_model_ema
        self.batch_size, self.seq_len = 0, 0
        self.seq_len_each = []
        self.reweight_loss_by_scale = reweight_loss_by_scale
        print(f"[ScaleMoGen][Predictor] reweight_loss_by_scale={self.reweight_loss_by_scale}", flush=True)
        
        self.using_ema = ema_ratio != 0 and self.zero == 0
        self.ema_ratio = abs(ema_ratio)
        self.ema_cpu = ema_ratio < 0
        self.is_visualizer = is_visualizer
        
        gpt_uncompiled = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
        del gpt_uncompiled.rng
        gpt_uncompiled.rng = torch.Generator(device=device)
        del gpt_uncompiled
        
        self.cached_state_not_ema = None
        if self.using_ema:
            self.pi_para_copy_for_parallel_ema = []
            all_tot = tot = 0
            for pi, para in enumerate(self.gpt_opt.paras):          # only learnable parameters need ema update
                if pi % dist.get_world_size() == dist.get_rank():   # model-parallel-style split
                    p_ema = para.data.cpu() if self.ema_cpu else para.data.clone()
                    self.pi_para_copy_for_parallel_ema.append((pi, p_ema))
                    tot += p_ema.numel()
                all_tot += para.numel()
            t = torch.zeros(dist.get_world_size())
            t[dist.get_rank()] = float(tot)
            dist.allreduce(t)
            t = [round(x) for x in t.tolist()]
            print(f'[ema tot #para] min={min(t)/1e6:.2f}, max={max(t)/1e6:.2f}, sum={sum(t)/1e6:.2f}, error={sum(t)-all_tot}')
            # lvl_1L, attn_bias_for_masking, zero_k_bias are never changed
            # check we only have these buffers so that we can skip buffer copy in ema update (only perform param update)
            assert all(any(s in name for s in ('lvl_1L', 'attn_bias_for_masking', 'zero_k_bias')) for name, _ in self.gpt_wo_ddp.named_buffers())
        else:
            self.pi_para_copy_for_parallel_ema = None
        
        self.label_smooth = label_smooth
        self.z_loss_ratio = z_loss_ratio
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='none')
        self.eq_loss = eq_loss
        
        if self.eq_loss:
            self.loss_eq_weight = torch.empty(1, self.raw_L, device=device)
            cur = 0
            for raw_pn in raw_scale_schedule:
                l = raw_pn*raw_pn
                self.loss_eq_weight[0, cur:cur+l] = 1./((raw_pn*raw_pn) if self.eq_loss == 2 else raw_pn)
                cur += l
            self.loss_eq_weight /= self.loss_eq_weight.sum()
        else:
            self.loss_eq_weight = 1.
        
        self.cmap_sim: ListedColormap = sns.color_palette('viridis', as_cmap=True)
        
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
        self.generator = np.random.default_rng(0)

        if is_train:
            self.logger = SummaryWriter(log_dir)

        self.other_args = other_args
        self.plot_t2m_func = plot_t2m_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.viz_batch_train = None
        self.viz_batch_val = None
    
    def lengths_to_mask(self, lengths, max_len):
        # max_len = max(lengths)
        mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        return mask #(b, len)

    def train_step(
        self, ep: int, it: int, g_it: int, stepping: bool, clip_decay_ratio: float, metric_lg: misc.MetricLogger, logging_params: bool,
        inp_B3HW: FTen, text_cond_tuple: Union[ITen, FTen], m_lens: ITen,
        args, is_validation=False,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        
        B = inp_B3HW.shape[0]  # if isinstance(inp_B3HW, torch.Tensor) else inp_B3HW[0].shape[0]
        T = inp_B3HW.shape[1] // self.len_scale_factor
        #V = self.vae_local.vocab_size TODO
        device = inp_B3HW.device


        h_div_w = inp_B3HW.shape[-2] / inp_B3HW.shape[-1]
        h_div_w_templates = np.array(list(dynamic_resolution_h_w.keys()))
        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w-h_div_w_templates))]
        
        #scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
        #scale_schedule = [(min(t, T//4+1), h, w) for (t,h,w) in scale_schedule]

        scale_schedule = [(1, T//t_s, len(s_groups)) for (t_s, s_groups) in self.coarse_to_fine_chain]

        non_pad_mask = []
        all_mask = []

        for _, (t_scale, spatial_groups) in enumerate(self.coarse_to_fine_chain):
            ds_mlens = torch.ceil(m_lens / (t_scale * self.len_scale_factor)).long()
            ds_non_pad_mask = self.lengths_to_mask(ds_mlens, T // t_scale)

            
            ds_non_pad_mask = ds_non_pad_mask.unsqueeze(-1).expand(-1, -1, len(spatial_groups))
            #ds_non_pad_mask = ds_non_pad_mask.repeat(1, len(spatial_groups))
            all_mask.append(rearrange(ds_non_pad_mask, 'n t j -> n (t j)'))
            non_pad_mask.append(rearrange(ds_non_pad_mask, 'n t j -> n (t j)'))
            
            #ids.append(ele)
            #time_to_arrival_pe.append(self.get_pe_from_mlens(ds_mlens, T))

        non_pad_mask = torch.cat(non_pad_mask, dim=1)
        
        # [forward]
        with self.gpt_opt.amp_ctx:
            with torch.amp.autocast('cuda', enabled=False):
                with torch.no_grad():
                    #if args.apply_spatial_patchify:
                    #    vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
                    #else:
                    #    vae_scale_schedule = scale_schedule
                    vae_scale_schedule = self.coarse_to_fine_chain
                    
                    raw_features = self.vae_local.encode_for_raw_features(inp_B3HW)
            
            x_BLC_wo_prefix, gt_ms_idx_Bl = self.bitwise_self_correction.flip_requant(vae_scale_schedule, inp_B3HW, raw_features, all_mask, device)
            # x_BLC_wo_prefix: torch.Size([bs, 2*2+3*3+...+64*64, d or 4d])

            # truncate scales
            training_scales = args.always_training_scales
            training_seq_len = np.array(scale_schedule)[:training_scales].prod(axis=1).sum()
            # x_BLC_wo_prefix = x_BLC_wo_prefix[:, :(training_seq_len-np.array(scale_schedule[0]).prod()), :] # TODO check for motion data
            x_BLC_wo_prefix = x_BLC_wo_prefix[:, :training_seq_len-1, :] # TODO check for motion data
            
            self.gpt_wo_ddp.forward  
            logits_BLV = self.gpt(text_cond_tuple, x_BLC_wo_prefix, scale_schedule=scale_schedule[:training_scales], motion_padding_mask=~non_pad_mask, g_it=g_it, resample=False) # [bs, 1*1+...+64*64, vocab_size or log2(vocab_size)*2]
            self.batch_size, self.seq_len = logits_BLV.shape[:2]

            self.seq_len_each = [idx_Bl.shape[1] for idx_Bl in gt_ms_idx_Bl]
            
            gt_BL = torch.cat(gt_ms_idx_Bl, dim=1)[:,:training_seq_len].contiguous().type(torch.long) # [bs, 1*1+...+64*64, 16] or [bs, 1*1+...+64*64]
            if args.use_bit_label:
                tmp_bs, tmp_seq_len, tmp_channel = logits_BLV.shape
                loss = self.train_loss(logits_BLV.reshape(tmp_bs, tmp_seq_len, -1, 2).permute(0,3,1,2), gt_BL)
                # loss_resample = self.train_loss(logits_BLV_resample.reshape(tmp_bs, tmp_seq_len, -1, 2).permute(0,3,1,2), gt_BL)
                # loss[:, mask_for_resample] += loss_resample[:, mask_for_resample]
                
                if args.bitloss_type == 'mean':
                    loss = loss.mean(dim=-1)
                elif args.bitloss_type == 'sum':
                    loss = loss.sum(dim=-1)
                else:
                    raise NotImplementedError(f'{args.bitloss_type=}')
            else:
                loss = self.train_loss(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).reshape(B, -1)

            loss = (loss * non_pad_mask.to(device))

            if self.reweight_loss_by_scale:
                lw = []
                last_scale_area = np.sqrt(np.prod(scale_schedule[-1]))
                for (pt, ph, pw) in scale_schedule[:training_scales]:
                    this_scale_area = np.sqrt(pt * ph * pw)
                    lw.extend([last_scale_area / this_scale_area for _ in range(pt * ph * pw)])
                lw = torch.tensor(lw, device=loss.device)[None, ...]
                lw = lw / lw.sum()
            else:
                lw = 1. / self.seq_len
            loss = loss.mul(lw).sum(dim=-1).mean()

        # [backward]
        grad_norm_t, scale_log2_t = self.gpt_opt.backward_clip_step(ep=ep, it=it, g_it=g_it, stepping=stepping, logging_params=logging_params, loss=loss, clip_decay_ratio=clip_decay_ratio, stable=args.stable)
        
        # update ema
        if args.use_fsdp_model_ema:
            update_ema(self.gpt_ema, self.gpt)

        # [zero_grad]
        if stepping:
            if self.using_ema: self.ema_update(g_it)
            if self.dbg_unused:
                ls = []
                for n, p in self.gpt_wo_ddp.named_parameters():
                    if p.grad is None:
                        ls.append(n)
                if len(ls):
                    raise AttributeError(f'unused param: {ls}')
        
            self.gpt_opt.optimizer.zero_grad(set_to_none=True)
        
        
        # [metric logging]
        if metric_lg.log_every_iter or it == 0 or it in metric_lg.log_iters:
            B, seq_len = logits_BLV.shape[:2]
            if args.use_bit_label:
                res_loss = self.train_loss(logits_BLV.reshape(B, seq_len, -1, 2).permute(0,3,1,2), gt_BL).mean(dim=-1)#.mean(0)
                res_loss = res_loss * non_pad_mask.to(device)
                bitwise_acc = (logits_BLV.reshape(B, seq_len, -1, 2).argmax(dim=-1) == gt_BL).float() # shape: [bs, seq_len, codebook_dim]
            else:
                res_loss = self.train_loss(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).reshape(B, -1).mean(0)
                pred_BL = logits_BLV.argmax(dim=-1)
                mask = self.vae_local.quantizer2d.lfq.mask
                pred_bits = ((pred_BL[..., None].int() & mask) != 0)
                gt_bits = ((gt_BL[..., None].int() & mask) != 0)
                bitwise_acc = (pred_bits == gt_bits).float() # shape: [bs, seq_len, codebook_dim]
            res_bit_acc = bitwise_acc.mean(-1)
            res_token_acc = (bitwise_acc.sum(-1) == self.vae_local.quantizer2d.lfq.codebook_dim).float()
            
            #loss_token_mean, acc_bit_mean, acc_token_mean = res_loss.mean().item(), res_bit_acc.mean().item() * 100., res_token_acc.mean().item() * 100.

            loss_token_mean = (res_loss * non_pad_mask.to(device)).sum() / non_pad_mask.to(device).sum()
            acc_bit_mean = (res_bit_acc * non_pad_mask.to(device)).sum() / non_pad_mask.to(device).sum()
            acc_token_mean = (res_token_acc * non_pad_mask.to(device)).sum() / non_pad_mask.to(device).sum()

            loss_token_mean = loss_token_mean.item()
            acc_bit_mean = acc_bit_mean.item() * 100.
            acc_token_mean = acc_token_mean.item() * 100.

            ptr = 0
            L_list, acc_bit_list, acc_token_list = [], [], []
            for scale_ind in range(min(training_scales, len(scale_schedule))):
                start, end = ptr, ptr + np.array(scale_schedule[scale_ind]).prod()
                L_list.append((res_loss[:,start:end] * non_pad_mask[:,start:end].to(device)).sum()/non_pad_mask[:,start:end].to(device).sum())
                acc_bit_list.append((res_bit_acc[:,start:end] * non_pad_mask[:,start:end].to(device)).sum()/non_pad_mask[:,start:end].to(device).sum())
                acc_token_list.append((res_token_acc[:,start:end] * non_pad_mask[:,start:end].to(device)).sum()/non_pad_mask[:,start:end].to(device).sum())
                ptr = end
            
            metrics = torch.tensor(L_list + acc_bit_list + acc_token_list +[grad_norm_t.item(), loss_token_mean, acc_bit_mean, acc_token_mean], device=loss.device)
            
            if torch.cuda.device_count() > 1: 
                tdist.all_reduce(metrics, op=tdist.ReduceOp.SUM)
            metrics = metrics.cpu().data.numpy() / dist.get_world_size()
            leng = len(L_list)
            L_list, acc_bit_list, acc_token_list, grad_norm_t, loss_token_mean, acc_bit_mean, acc_token_mean = metrics[:leng], \
                metrics[leng:2*leng], metrics[2*leng:3*leng], metrics[-4], metrics[-3], metrics[-2], metrics[-1]
            Lmean = loss_token_mean
            Ltail = L_list[-1]
            acc_mean = acc_bit_mean if args.use_bit_label else acc_token_mean
            acc_tail = acc_bit_list[-1] if args.use_bit_label else acc_token_list[-1]
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm_t)    # todo: Accm, Acct
            wandb_log_dict = {"Overall/L_mean": Lmean, 'Overall/Acc_bit_mean': acc_bit_mean, 'Overall/Acc_token_mean': acc_token_mean, 'Overall/grad_norm_t': grad_norm_t}
            for si, (loss_si, acc_bit_si, acc_token_si) in enumerate(zip(L_list, acc_bit_list, acc_token_list)):
                wandb_log_dict[f'Detail/L_s{si+1:02d}'] = loss_si
                wandb_log_dict[f'Detail/Acc_bit_s{si+1:02d}'] = acc_bit_si
                wandb_log_dict[f'Detail/Acc_token_s{si+1:02d}'] = acc_token_si
                # print(wandb_log_dict)
                # wandb_utils.log(wandb_log_dict, step=g_it)
                
            if not is_validation:
                # log to self.logger
                for key, value in wandb_log_dict.items():
                    self.logger.add_scalar(key, value, g_it)


        return grad_norm_t, scale_log2_t
    
    def __repr__(self):
        return (
            f'\n'
            f'[ScaleMoGenTrainer.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[ScaleMoGenTrainer.structure]: {super(ScaleMoGenPredictorTrainer, self).__repr__().replace(ScaleMoGenPredictorTrainer.__name__, "")}'
        )
    
    def ema_load(self):
        self.cached_state_not_ema = {k: v.cpu() for k, v in self.gpt_wo_ddp.state_dict().items()}
        for pi, p_ema in self.pi_para_copy_for_parallel_ema:
            self.gpt_opt.paras[pi].data.copy_(p_ema)
        for pi, para in enumerate(self.gpt_opt.paras):
            dist.broadcast(para, src_rank=pi % dist.get_world_size())
    
    def ema_recover(self):
        self.gpt_wo_ddp.load_state_dict(self.cached_state_not_ema)
        del self.cached_state_not_ema
        self.cached_state_not_ema = None
    
    # p_ema = p_ema*0.9 + p*0.1 <==> p_ema.lerp_(p, 0.1)
    # p_ema.mul_(self.ema_ratio).add_(p.mul(self.ema_ratio_1))
    # @profile(precision=4, stream=open('ema_update.log', 'w+'))
    def ema_update(self, g_it): # todo: 将来再用离线ema
        # if self.using_ema and (g_it + 1) in self.ema_upd_it:
        stt = time.time()
        for pi, p_ema in self.pi_para_copy_for_parallel_ema:
            p = self.gpt_opt.paras[pi]
            p_ema.data.mul_(self.ema_ratio).add_(p.data.to(p_ema.device), alpha=1-self.ema_ratio)
        # ii = self.ema_upd_it.index(g_it + 1)
        ii = g_it
        if ii < 3:
            print(f'[ema upd {self.ema_ratio}, cpu={self.ema_cpu}, @ g_it={g_it}] cost: {time.time()-stt:.2f}s')
    
    def get_config(self):
        return {
            'dynamic_resolution_h_w': dynamic_resolution_h_w,
            'label_smooth': self.label_smooth, 'eq_loss': self.eq_loss,
            'ema_ratio':    self.ema_ratio,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
    def state_dict(self):
        m = self.vae_local
        if hasattr(m, '_orig_mod'):
            m = m._orig_mod
        state = {'config': self.get_config(), 'vae_local': m.state_dict()}
        
        if self.zero:   # TODO: fixme
            state['gpt_fsdp'] = None
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                state['gpt_fsdp'] = self.gpt.state_dict()
                if self.use_fsdp_model_ema:
                    state['gpt_ema_fsdp'] = self.gpt_ema.state_dict()

                #for rank in range(dist.get_world_size()):
                #    print(f"[Rank {dist.get_rank()}] param group lengths: {[len(pg['params']) for pg in self.gpt_opt.optimizer.param_groups]}")

                #if dist.get_rank() == 0:
                #    gpt_fsdp_opt = FSDP.optim_state_dict(
                #        model=self.gpt,
                #        optim=self.gpt_opt.optimizer
                #    )
                #else:
                #    gpt_fsdp_opt = None
                gpt_fsdp_opt = None

                state['gpt_fsdp_opt'] = gpt_fsdp_opt
            if self.gpt_opt.scaler is not None:
                state['gpt_opt_scaler'] = self.gpt_opt.scaler.state_dict()
        
        else:
            if self.using_ema:  # TODO: fixme
                self.ema_load()
                state['gpt_ema_for_vis'] = {k: v.cpu() for k, v in self.gpt_wo_ddp.state_dict().items()}
                self.ema_recover()
            
            for k in ('gpt_wo_ddp', 'gpt_opt'):
                m = getattr(self, k)
                if m is not None:
                    if hasattr(m, '_orig_mod'):
                        m = m._orig_mod
                    state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        if self.zero:
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                self.gpt.load_state_dict(state['gpt_fsdp'])
                if self.use_fsdp_model_ema:
                    self.gpt_ema.load_state_dict(state['gpt_ema_fsdp'])
                one_group_opt_state = state['gpt_fsdp_opt']
                """
                AdamW state['gpt_fsdp_opt']:
                {
                    'state': { <para_name>: {'exp_avg': <unsharded_tensor>, 'exp_avg_sq': <unsharded_tensor>, 'step': <int>} },
                    'param_groups': [{...}]
                }
                one_group_opt_state['param_groups'] = self.gpt_opt.optimizer.state_dict()['param_groups']
                """
                optim_state_dict = FSDP.optim_state_dict_to_load(model=self.gpt, optim=self.gpt_opt.optimizer, optim_state_dict=one_group_opt_state)
                self.gpt_opt.optimizer.load_state_dict(optim_state_dict)

            if self.gpt_opt.scaler is not None:
                try: self.gpt_opt.scaler.load_state_dict(state['gpt_opt_scaler'])
                except Exception as e: print(f'[fp16 load_state_dict err] {e}')
        else:
            for k in ('gpt_wo_ddp', 'gpt_opt'):
                if skip_vae and 'vae' in k: continue
                m = getattr(self, k)
                if m is not None:
                    if hasattr(m, '_orig_mod'):
                        m = m._orig_mod
                    ret = m.load_state_dict(state[k], strict=strict)
                    if ret is not None:
                        missing, unexpected = ret
                        print(f'[ScaleMoGenTrainer.load_state_dict] {k} missing:  {missing}')
                        print(f'[ScaleMoGenTrainer.load_state_dict] {k} unexpected:  {unexpected}')
            
            if self.using_ema:
                if 'gpt_ema_for_vis' in state:
                    for pi, para in self.pi_para_copy_for_parallel_ema:
                        para.copy_(state['gpt_ema_for_vis'][self.gpt_opt.names[pi]])
                    print(f'[ScaleMoGenTrainer.load_state_dict] gpt_ema_for_vis: load succeed')
                else:
                    print(f'[ScaleMoGenTrainer.load_state_dict] gpt_ema_for_vis: key NOT FOUND in state!!')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[ScaleMoGenPredictor.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)

    @torch.no_grad()
    def visualize_motion(self, ep, g_it, text_tokenizer, text_encoder, tag='train', NUM_SAMPLES_TO_VIZ=8):
        self.gpt_wo_ddp.eval() # Set to evaluation mode

        if tag == 'train':
            if self.viz_batch_train is None:
                # Get a fixed batch from the train loader for consistent visualization
                self.viz_batch_train = next(iter(self.train_loader))
            batch = self.viz_batch_train
        else: # val
            if self.viz_batch_val is None:
                self.viz_batch_val = next(iter(self.val_loader))
            batch = self.viz_batch_val

        save_dir = pjoin(self.other_args.eval_dir, f'ep{ep:04d}_it{g_it:06d}_{tag}')
        os.makedirs(save_dir, exist_ok=True)

        all_captions, all_input_motions, all_m_lengths = batch
        
        # Loop through the first N samples
        for i in range(min(NUM_SAMPLES_TO_VIZ, len(all_captions))):
            
            captions = [all_captions[i]]
            input_motions = all_input_motions[[i]].to(self.device)
            m_lengths = all_m_lengths[[i]].to(self.device)
            prompt = captions[0]

            print(
                f"[ScaleMoGen][Predictor][viz] sample={i:02d} tag={tag} prompt={prompt}",
                flush=True,
            )
            
            coarse_to_fine_chain = coarse_to_fine_chain_from_vq(self.vae_local)
            scale_schedule = scale_schedule_from_vq(self.vae_local, self.max_motion_length)

            generated_motion, rec_motions = gen_one_motion(
                predictor=self.gpt_wo_ddp,
                vae=self.vae_local,
                text_tokenizer=text_tokenizer,
                text_encoder=text_encoder,
                prompt=prompt,
                all_mask=None,
                motion_padding_mask=None,
                g_seed=random.randint(0, 10000),
                gt_leak=0,
                gt_ls_Bl=None,
                cfg_list=5,
                tau_list=0.5,
                coarse_to_fine_chain=coarse_to_fine_chain,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[0],
                sampling_per_bits=1,
                enable_positive_prompt=0,
                input_motions=input_motions,
                m_lengths=m_lengths,
                use_bf16=getattr(self.other_args, "use_bf16_eval", False),
            )

            # Save ground truth
            self.plot_t2m_func(
                data=input_motions.float(),
                save_dir=save_dir,
                captions=[f"GT: {prompt}"],
                m_lengths=m_lengths,
                save_path=pjoin(save_dir, f"{i:02d}_gt.mp4") # Add index to filename
            )

            # Save reconstruction
            self.plot_t2m_func(
                data=rec_motions.unsqueeze(0).float(),
                save_dir=save_dir,
                captions=[f"Recon: {prompt}"],
                m_lengths=m_lengths,
                save_path=pjoin(save_dir, f"{i:02d}_recon.mp4") # Add index to filename
            )

            # Save generated
            self.plot_t2m_func(
                data=generated_motion.unsqueeze(0).float(),
                save_dir=save_dir,
                captions=[f"Gen: {prompt}"],
                m_lengths=m_lengths,
                save_path=pjoin(save_dir, f"{i:02d}_gen.mp4") # Add index to filename
            )

        print(
            f"[ScaleMoGen][Predictor][viz] saved={save_dir} samples={min(NUM_SAMPLES_TO_VIZ, len(all_captions))}",
            flush=True,
        )

        self.gpt_wo_ddp.train() # Set back to training mode
