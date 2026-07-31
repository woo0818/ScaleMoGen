"""Checkpoint saving and resume helpers for ScaleMoGen predictor training.

Code provenance: adapted from the Infinity checkpoint utilities.
Source repository: https://github.com/FoundationVision/Infinity
"""

import gc
import os
import re
import time
from typing import List, Optional, Tuple

import torch

import glob
#from scalemogen.utils import arg_util
import scalemogen.utils.dist as dist


def glob_with_epoch_iter(pattern, recursive=False): 
    def extract_ep_iter(filename):
        match = re.search(r'ep(\d+)-iter(\d+)', filename)
        if match:
            ep = int(match.group(1))
            iter_idx = int(match.group(2))
            return ep, iter_idx
        return 0, 0
    return sorted(glob.glob(pattern, recursive=recursive), key=lambda x: extract_ep_iter(os.path.basename(x)), reverse=True)


def glob_with_global_step(pattern, recursive=False): 
    def extract_ep_iter(filename):
        match = re.search(r'global_step_(\d+)', filename)
        if match:
            iter_idx = int(match.group(1))
            return iter_idx
        return 0
    return sorted(glob.glob(pattern, recursive=recursive), key=lambda x: extract_ep_iter(os.path.basename(x)), reverse=True)
        

from collections import deque
        

class CKPTSaver(object):
    def __init__(self, is_master: bool, eval_milestone: List[Tuple[float, float]], max_to_keep: int = 5):
        self.is_master = is_master
        self.acc_str, self.eval_milestone = '[no acc str]', eval_milestone
        if self.is_master:
            self.ckpt_queue = deque(maxlen=max_to_keep)
    
    def sav(
        self, args, g_it: int, next_ep: int, next_it: int, trainer,
        acc_str: Optional[str] = None, eval_milestone: Optional[List[Tuple[float, float]]] = None,
    ):
        if acc_str is not None: self.acc_str = acc_str
        if eval_milestone is not None: self.eval_milestone = eval_milestone
        
        fname = f'ar-ckpt-giter{g_it//1000:03d}K-ep{next_ep}-iter{next_it}.pth'
        local_out_ckpt = os.path.join(args.model_dir, fname)
        
        # NOTE: all rank should call this state_dict(), not master only!
        trainer_state = trainer.state_dict()
        
        if self.is_master:
            torch.save({
                #'args':         args.state_dict(),
                'gpt_training': args.gpt_training,
                'arch':         args.model if args.gpt_training else args.vv,
                'epoch':        next_ep,
                'iter':         next_it,
                'trainer':      trainer_state,
                'acc_str':      self.acc_str,
                'milestones':   self.eval_milestone,
            }, local_out_ckpt)

            if self.ckpt_queue.maxlen is not None and len(self.ckpt_queue) == self.ckpt_queue.maxlen:
                oldest_ckpt = self.ckpt_queue[0]
                try:
                    os.remove(oldest_ckpt)
                    print(f'[CKPTSaver] Removed old checkpoint: {os.path.basename(oldest_ckpt)}')
                except OSError as e:
                    print(f'[CKPTSaver] Error removing old checkpoint {oldest_ckpt}: {e}')
            self.ckpt_queue.append(local_out_ckpt)
            
            print(f'[ScaleMoGen][Checkpoint] saved={local_out_ckpt}', flush=True)
        
        del trainer_state
        time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
        dist.barrier()
        

def auto_resume(args, pattern='ckpt*.pth') -> Tuple[List[str], int, int, str, List[Tuple[float, float]], dict, dict]:
    info = []
    resume = ''
    if args.auto_resume:
        all_ckpt = glob_with_epoch_iter(os.path.join(args.model_dir, pattern))
        if len(all_ckpt) == 0:
            info.append(f'[auto_resume] no ckpt found @ {pattern}')
            info.append(f'[auto_resume quit]')
        else:
            resume = all_ckpt[0]
            info.append(f'[auto_resume] auto load from @ {resume} ...')
    else:
        info.append(f'[auto_resume] disabled')
        info.append(f'[auto_resume quit]')
    
    if len(resume) == 0:
        return info, 0, 0, '[no acc str]', [], {}, {}

    print(f'auto resume from {resume}')

    try:
        ckpt = torch.load(resume, map_location='cpu')
    except Exception as e:
        info.append(f'[auto_resume] failed, {e} @ {resume}')
        if len(all_ckpt) < 2:
            return info, 0, 0, '[no acc str]', [], {}, {}
        try: # another chance to load from bytenas
            ckpt = torch.load(all_ckpt[1], map_location='cpu')
        except Exception as e:
            info.append(f'[auto_resume] failed, {e} @ {all_ckpt[1]}')
            return info, 0, 0, '[no acc str]', [], {}, {}
    
    dist.barrier()
    ep, it = ckpt['epoch'], ckpt['iter']
    eval_milestone = ckpt.get('milestones', [])
    info.append(f'[auto_resume success] resume from ep{ep}, it{it},    eval_milestone: {eval_milestone}')
    return info, ep, it, ckpt.get('acc_str', '[no acc str]'), eval_milestone, ckpt['trainer'], ckpt['arch']
