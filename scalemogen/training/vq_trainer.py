"""ScaleMoGen VQ tokenizer trainer.

Code provenance: adapted from the SALAD VQ training pipeline.
Source repository: https://github.com/seokhyeonhong/salad
"""

import torch
from os.path import join as pjoin

import os
import time

from copy import deepcopy

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict, defaultdict

from utils.eval_t2m import evaluation_vqvae, evaluation_vqvae_hml #, evaluation_vae, test_vae
from utils.utils import print_current_loss
from scalemogen.utils.console import print_scalemogen_progress

from trainers.base_trainer import BaseTrainer

def def_value():
    return 0.0

def length_to_mask(length, max_len, device: torch.device = None) -> torch.Tensor:
    if device is None:
        device = length.device

    if isinstance(length, list):
        length = torch.tensor(length)
    
    length = length.to(device)
    # max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ).to(device) < length.unsqueeze(1)
    return mask


def mean_flat(tensor: torch.Tensor, mask=None):
    """
    Take the mean over all non-batch dimensions.
    """
    if mask is None:
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    else:
        # mask = mask.unsqueeze(2)           # [B, T] -> [T, B, 1]
        assert tensor.dim() == 3
        denom = mask.sum() * tensor.shape[-1]
        loss = (tensor * mask).sum() / denom
        return loss
    

class ScaleMoGenVQTrainer(BaseTrainer):
    def __init__(self, cfg, vq_model, device):
        self.cfg = cfg
        self.data_name = cfg.data.name
        self.vq_model = vq_model
        self.device = device

        self.is_train = cfg.exp.is_train
        self.logger = None
        if cfg.exp.is_train:
            self.logger = SummaryWriter(cfg.exp.log_dir)
            if cfg.training.recons_loss == 'l1':
                self.recon_criterion = torch.nn.L1Loss()
            elif cfg.training.recons_loss == "l1_smooth":
                self.recon_criterion = torch.nn.SmoothL1Loss()
        
        if self.data_name == 'snapmogen':
            self.end_joints = torch.tensor([10, 14, 18, 19, 22, 23], device=self.device)
        else:
            self.end_joints = torch.tensor([7, 8, 10, 11, 15, 20, 21], device=self.device)
        
        if cfg.training.ema:
            self.ema_model = deepcopy(vq_model).to(device)
            self.ema_model.eval()
            self.requires_grad(self.ema_model, False)

        self.cfg.training.lambda_commit = 1.0

    def forward(self, batch_data):
        captions, motions, m_lens = batch_data
        motions = motions.detach().to(self.device).float()
        m_lens = m_lens.detach().to(self.device).long()

        if self.data_name == 'snapmogen':
            root, rot, ric, vel, contact = torch.split(motions, [4, 6 * (self.cfg.data.joint_num), 3 * (self.cfg.data.joint_num), 3 * self.cfg.data.joint_num, 4], dim=-1)

            pred_motion, loss_dict = self.vq_model.forward(motions, m_lens)
            pred_root, pred_rot, pred_ric, pred_vel, pred_contact = torch.split(pred_motion, [4, 6 * (self.cfg.data.joint_num), 3 * self.cfg.data.joint_num, 3 * self.cfg.data.joint_num, 4], dim=-1)
        else:
            root, ric, rot, vel, contact = torch.split(motions, [4, 3 * (self.cfg.data.joint_num - 1), 6 * (self.cfg.data.joint_num - 1), 3 * self.cfg.data.joint_num, 4], dim=-1)

            pred_motion, loss_dict = self.vq_model.forward(motions, m_lens)
            pred_root, pred_ric, pred_rot, pred_vel, pred_contact = torch.split(pred_motion, [4, 3 * (self.cfg.data.joint_num - 1), 6 * (self.cfg.data.joint_num - 1), 3 * self.cfg.data.joint_num, 4], dim=-1)


        self.motions = motions
        self.pred_motion = pred_motion

        mask = length_to_mask(m_lens, max_len=motions.shape[1])

        # loss
        loss_rec = mean_flat(
            F.smooth_l1_loss(self.pred_motion, self.motions, reduction='none'),
            mask=mask.unsqueeze(-1)
        )

        B, T, _ = motions.shape

        if self.cfg.training.lambda_fk == 0: # currently this
            loss_fk = torch.tensor(0.).to(self.device).float()
        else:
            loss_fk = mean_flat(
                F.smooth_l1_loss(fk_func(self.motions).view(B, T, -1), 
                                 fk_func(self.pred_motion).view(B, T, -1), 
                                 reduction='none'),
                mask=mask.unsqueeze(-1))
        
        loss_global = mean_flat(
            F.smooth_l1_loss(self.pred_motion[..., :4], self.motions[..., :4], reduction='none'),
            mask=mask.unsqueeze(-1)
        )

        loss_vel = mean_flat(
            F.smooth_l1_loss(pred_vel, vel, reduction='none'),
            mask=mask.unsqueeze(-1)
        )

        if self.data_name == 'snapmogen':
            end_pos_gt = ric.view(ric.shape[0], ric.shape[1], self.cfg.data.joint_num, 3)[:, :, self.end_joints].view(B, T, -1)
            end_pos_recon = pred_ric.view(ric.shape[0], ric.shape[1], self.cfg.data.joint_num, 3)[:, :, self.end_joints].view(B, T, -1)

            end_vel_gt = vel.view(vel.shape[0], vel.shape[1], self.cfg.data.joint_num, 3)[:, :, self.end_joints].view(B, T, -1)
            end_vel_recon = pred_vel.view(vel.shape[0], vel.shape[1], self.cfg.data.joint_num, 3)[:, :, self.end_joints].view(B, T, -1)
        else:
            end_pos_gt = ric.view(ric.shape[0], ric.shape[1], self.cfg.data.joint_num - 1, 3)[:, :, self.end_joints - 1].view(B, T, -1)
            end_pos_recon = pred_ric.view(ric.shape[0], ric.shape[1], self.cfg.data.joint_num - 1, 3)[:, :, self.end_joints - 1].view(B, T, -1)

            end_vel_gt = vel.view(vel.shape[0], vel.shape[1], self.cfg.data.joint_num, 3)[:, :, self.end_joints].view(B, T, -1)
            end_vel_recon = pred_vel.view(vel.shape[0], vel.shape[1], self.cfg.data.joint_num, 3)[:, :, self.end_joints].view(B, T, -1)

        loss_end_vel = mean_flat(
            F.smooth_l1_loss(
                end_vel_recon,
                end_vel_gt,
                reduction='none'
            ),
            mask=mask.unsqueeze(-1)
        )
        loss_end_pos = mean_flat(
            F.smooth_l1_loss(
                end_pos_recon,
                end_pos_gt,
                reduction='none'
            ),
            mask=mask.unsqueeze(-1)
        )
        loss_commit = loss_dict["loss_commit"]
        loss = loss_rec + \
            self.cfg.training.lambda_global * loss_global + \
            self.cfg.training.lambda_fk * loss_fk + \
            self.cfg.training.lambda_commit * loss_commit + \
            self.cfg.training.lambda_vel * loss_vel
        
        loss_dict["loss_recon"] = loss_rec
        loss_dict["loss_vel"] = loss_vel
        loss_dict["loss_global"] = loss_global
        loss_dict["loss_end_vel"] = loss_end_vel
        loss_dict["loss_end_pos"] = loss_end_pos

        return loss, loss_dict


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vq_model.param_groups:
            param_group["lr"] = current_lr


    def save(self, file_name, epoch, total_iter):
        state = {
            "vq_model": self.vq_model.state_dict(),
            "opt_vq_model": self.opt_vq_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "total_iter": total_iter,
        }

        if self.cfg.training.ema:
            state["ema_model"] = self.ema_model.state_dict()
        torch.save(state, file_name)


    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)#, weights_only=True)
        self.vq_model.load_state_dict(checkpoint['vq_model'])
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        try:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        except:
            pass

        if self.cfg.training.ema:
            self.ema_model.load_state_dict(checkpoint['ema_model'])
        return checkpoint['epoch'], checkpoint['total_iter']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval=None, fk_func=None):
        self.vq_model.to(self.device)

        self.opt_vq_model = optim.AdamW(self.vq_model.parameters(), lr=self.cfg.training.lr, betas=(0.9, 0.99), weight_decay=self.cfg.training.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_vq_model, 
                                                              milestones=self.cfg.training.milestones, 
                                                              gamma=self.cfg.training.gamma)


        epoch = 0
        it = 0

        #if self.cfg.training.ema:
        #    self.update_ema(self.ema_model, self.vq_model, decay=0)

        if self.cfg.exp.is_continue:
            model_dir = pjoin(self.cfg.exp.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print(f"[ScaleMoGen][VQ] resumed ep={epoch:04d} it={it:07d}", flush=True)

        start_time = time.time()
        total_iters = self.cfg.training.max_epoch * len(train_loader)
        print(
            f"[ScaleMoGen][VQ] dataset={self.cfg.data.name} "
            f"epochs={self.cfg.training.max_epoch} total_iters={total_iters} "
            f"train_batches={len(train_loader)} val_batches={len(eval_val_loader)}",
            flush=True,
        )
        # val_loss = 0
        # min_val_loss = np.inf
        # min_val_epoch = epoch
        # current_lr = self.cfg.training.lr
        def def_value():
            return 0.0
        logs = defaultdict(def_value, OrderedDict())

        # sys.exit()
        if self.cfg.data.name == 'snapmogen':
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe = evaluation_vqvae(
                self.cfg.exp.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, best_fid=1000,
                best_div=100, best_top1=0,
                best_top2=0, best_top3=0, best_matching=0, best_mpjpe=100, nfeats=self.cfg.data.dim_pose,
                eval_wrapper=eval_wrapper, device=self.device, fk_func=fk_func, save_ckpt=True, save_anim=False, plot_eval=plot_eval)
        else:
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, _ = evaluation_vqvae_hml(
                self.cfg.exp.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, best_fid=1000,
                best_div=100, best_top1=0,
                best_top2=0, best_top3=0, best_matching=100,
                eval_wrapper=eval_wrapper, save=False)

        while epoch < self.cfg.training.max_epoch:
            self.vq_model.train()
            for i, batch_data in enumerate(train_loader):

                it += 1
                if it < self.cfg.training.warm_up_iter:
                    current_lr = self.update_lr_warm_up(it, self.cfg.training.warm_up_iter, self.cfg.training.lr)
                loss, loss_dict = self.forward(batch_data)
                self.opt_vq_model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vq_model.parameters(), max_norm=0.5) # from MogenTS
                self.opt_vq_model.step()

                if it >= self.cfg.training.warm_up_iter:
                    self.scheduler.step()
                
                for tag, value in loss_dict.items():
                    logs[tag] += value.item()
                logs['lr'] += self.opt_vq_model.param_groups[0]['lr']

                if it % self.cfg.training.log_every == 0:
                    mean_loss = OrderedDict()
                    # self.logger.add_scalar('val_loss', val_loss, it)
                    # self.l
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.cfg.training.log_every, it)
                        mean_loss[tag] = value / self.cfg.training.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.cfg.training.save_latest == 0:
                    self.save(pjoin(self.cfg.exp.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.cfg.exp.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            # if epoch % self.cfg.save_every_e == 0:
            #     self.save(pjoin(self.cfg.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print(f"[ScaleMoGen][VQ][val] ep={epoch:04d} start", flush=True)
            self.vq_model.eval()
            val_loss_rec = []
            val_loss_vel = []
            val_loss_global = []
            val_loss_end_pos = []
            val_loss_end_vel = []
            val_loss_commit_b = []
            val_loss_commit_t = []
            val_loss = []
            val_perpexity_b = []
            val_perpexity_t = []
            val_perpexity = []
            val_loss_commit = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, loss_dict = self.forward(batch_data)
                    # val_loss_rec += self.l1_criterion(self.recon_motions, self.motions).item()
                    # val_loss_emb += self.embedding_loss.item()
                    val_loss.append(loss.item())
                    val_loss_rec.append(loss_dict["loss_recon"].item())
                    val_loss_vel.append(loss_dict["loss_vel"].item())
                    val_loss_global.append(loss_dict["loss_global"].item())

                    val_loss_end_vel.append(loss_dict["loss_end_vel"].item())
                    val_loss_end_pos.append(loss_dict["loss_end_pos"].item())
                    val_loss_commit.append(loss_dict["loss_commit"].item())
                    val_perpexity.append(loss_dict["perpexity"].item())

            # val_loss = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss = val_loss / (len(val_dataloader) + 1)
            # val_loss_rec = val_loss_rec / (len(val_dataloader) + 1)
            # val_loss_emb = val_loss_emb / (len(val_dataloader) + 1)
            self.logger.add_scalar('Val/loss', sum(val_loss) / len(val_loss), epoch)
            self.logger.add_scalar('Val/loss_rec', sum(val_loss_rec) / len(val_loss_rec), epoch)
            self.logger.add_scalar('Val/loss_vel', sum(val_loss_vel) / len(val_loss_vel), epoch)
            self.logger.add_scalar('Val/loss_end_vel', sum(val_loss_end_vel) / len(val_loss_end_vel), epoch)
            self.logger.add_scalar('Val/loss_end_pos', sum(val_loss_end_pos) / len(val_loss_end_pos), epoch)
            self.logger.add_scalar('Val/loss_pos', sum(val_loss_global) / len(val_loss_global), epoch)
            self.logger.add_scalar('Val/loss_perplexity', sum(val_perpexity) / len(val_loss_rec), epoch)
            self.logger.add_scalar('Val/loss_commit', sum(val_loss_commit) / len(val_loss), epoch)
            #self.logger.add_scalar('Val/loss_commit_b', sum(val_loss_commit_b) / len(val_loss), epoch)
            #self.logger.add_scalar('Val/loss_perplexity_b', sum(val_perpexity_b) / len(val_loss_rec), epoch)
            #self.logger.add_scalar('Val/loss_commit_t', sum(val_loss_commit_t) / len(val_loss), epoch)
            #self.logger.add_scalar('Val/loss_perplexity_t', sum(val_perpexity_t) / len(val_loss_rec), epoch)

            val_metrics = OrderedDict(
                loss=sum(val_loss) / len(val_loss),
                loss_rec=sum(val_loss_rec) / len(val_loss_rec),
                loss_vel=sum(val_loss_vel) / len(val_loss_vel),
                loss_pos=sum(val_loss_global) / len(val_loss_global),
                loss_commit=sum(val_loss_commit) / len(val_loss_commit),
                perpexity=sum(val_perpexity) / len(val_perpexity),
            )
            print_scalemogen_progress("VQ", "val", metrics=val_metrics, epoch=epoch)

            # if sum(val_loss) / len(val_loss) < min_val_loss:
            #     min_val_loss = sum(val_loss) / len(val_loss)
            # # if sum(val_loss_vel) / len(val_loss_vel) < min_val_loss:
            # #     min_val_loss = sum(val_loss_vel) / len(val_loss_vel)
            #     min_val_epoch = epoch
            #     self.save(pjoin(self.cfg.model_dir, 'finest.tar'), epoch, it)
            #     print('Best Validation Model So Far!~')

            if self.cfg.data.name == 'snapmogen':
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe = evaluation_vqvae(
                    self.cfg.exp.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, best_fid=best_fid,
                    best_div=best_div, best_top1=best_top1,
                    best_top2=best_top2, best_top3=best_top3, best_matching=best_matching, best_mpjpe=best_mpjpe, nfeats=self.cfg.data.dim_pose,
                    eval_wrapper=eval_wrapper, device=self.device, fk_func=fk_func, save_ckpt=True, save_anim=False, plot_eval=plot_eval)
            else:
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching, _ = evaluation_vqvae_hml(
                    self.cfg.exp.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, best_fid=best_fid,
                    best_div=best_div, best_top1=best_top1,
                    best_top2=best_top2, best_top3=best_top3, best_matching=best_matching,
                    eval_wrapper=eval_wrapper, save=True)


            if epoch % self.cfg.training.eval_every_e == 0:
                captions = batch_data[0]
                data = torch.cat([self.motions[:4], self.pred_motion[:4]], dim=0)
                lengths = batch_data[2]
                # np.save(pjoin(self.cfg.eval_dir, 'E%04d.npy' % (epoch)), data)
                save_dir = pjoin(self.cfg.exp.eval_dir, 'E%04d' % (epoch))
                os.makedirs(save_dir, exist_ok=True)
                plot_eval(data, save_dir, captions[:4] * 2, torch.cat([lengths[:4], lengths[:4]], dim=0))

    def eval(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval=None, fk_func=None):
        self.vq_model.to(self.device)
        
        def def_value():
            return 0.0
        logs = defaultdict(def_value, OrderedDict())

        # sys.exit()
        if self.cfg.data.name == 'snapmogen':
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe = evaluation_vqvae(
                self.cfg.vq_name, eval_val_loader, self.vq_model, None, ep=0, best_fid=1000,
                best_div=100, best_top1=0,
                best_top2=0, best_top3=0, best_matching=0, best_mpjpe=100, nfeats=self.cfg.data.dim_pose,
                eval_wrapper=eval_wrapper, device=self.device, fk_func=fk_func, save_ckpt=False, draw=False, plot_eval=plot_eval)
        else:
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, _ = evaluation_vqvae_hml(
                self.cfg.exp.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, best_fid=1000,
                best_div=100, best_top1=0,
                best_top2=0, best_top3=0, best_matching=100,
                eval_wrapper=eval_wrapper, save=False)
SKELTokenizerTrainer = ScaleMoGenVQTrainer
