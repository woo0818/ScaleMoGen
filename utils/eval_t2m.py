"""Evaluation metrics and loops for ScaleMoGen motion generation.

Code provenance: adapted from the SnapMoGen evaluation pipeline.
Source repository: https://github.com/snap-research/SnapMoGen
"""

import hashlib
import os

# import clip
import numpy as np
import torch
# from scipy import linalg
from utils.metrics import *
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
from scalemogen.generation import coarse_to_fine_chain_from_vq, scale_schedule_from_vq
from tools.run_scalemogen import gen_one_motion
# import visualization.plot_3d_global as plot_3d
# from utils.motion_process import recover_from_ric
#
#
# def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
#     xyz = xyz[:1]
#     bs, seq = xyz.shape[:2]
#     xyz = xyz.reshape(bs, seq, -1, 3)
#     plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(), title_batch, outname)
#     plot_xyz = np.transpose(plot_xyz, (0, 1, 4, 2, 3))
#     writer.add_video(tag, plot_xyz, nb_iter, fps=20)


def length_to_mask(length, max_len, device: torch.device = None) -> torch.Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length)
    
    length = length.to(device)
    # max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ).to(device) < length.unsqueeze(1)
    return mask


def _cpu_metric_tensor(tensor):
    """Detach an evaluation tensor before storing it for final metric aggregation."""
    return tensor.detach().cpu()


def _cat_metric_tensors(tensors):
    """Concatenate stored CPU metric tensors and return a NumPy array."""
    return torch.cat(tensors, dim=0).numpy()


def reconstruct_vqvae_motion(net, motion, m_lengths=None):
    """Reconstruct motion from a VQ tokenizer across supported encode layouts."""
    lengths = m_lengths.clone() if m_lengths is not None else None
    if lengths is not None:
        encoded = net.encode(motion, lengths)
    else:
        encoded = net.encode(motion)

    if isinstance(encoded, tuple):
        if len(encoded) >= 4:
            latent = encoded[0]
        elif len(encoded) >= 2:
            latent = encoded[1]
        else:
            latent = encoded[0]
    else:
        latent = encoded

    try:
        rec_motion = net.decode(latent, lengths.clone()) if lengths is not None else net.decode(latent)
    except TypeError:
        rec_motion = net.decode(latent)

    if lengths is not None:
        mask = torch.arange(motion.shape[1], device=motion.device).unsqueeze(0).expand(motion.shape[0], -1)
        mask = mask >= lengths.unsqueeze(1)
        rec_motion = rec_motion.masked_fill(mask.unsqueeze(-1), 0)
    return rec_motion


def _generation_seed(seed_mode, base_seed, repeat_id, sample_idx, text, draw_idx):
    """Return the optional per-sample generation seed used during evaluation."""
    mode = (seed_mode or "random").lower()
    if mode in {"random", "none"}:
        return None

    base_seed = int(base_seed)
    repeat_id = int(repeat_id)
    sample_idx = int(sample_idx)
    draw_idx = int(draw_idx)

    if mode in {"fixed", "constant"}:
        return base_seed
    if mode in {"sequential", "seq"}:
        draw_offset = 0 if draw_idx < 0 else draw_idx + 1
        return (base_seed + repeat_id * 1_000_000 + sample_idx * 1_000 + draw_offset) % (2**31 - 1)
    if mode in {"deterministic", "stable", "text"}:
        payload = f"{base_seed}|{repeat_id}|{sample_idx}|{draw_idx}|{text}".encode("utf-8")
        return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**31 - 1)

    raise ValueError(f"Unsupported seed_mode={seed_mode!r}")


@torch.no_grad()
def evaluation_evaluator(out_dir, eval_val_loader, writer, ep, best_top1, best_top2, best_top3, 
                         best_matching, eval_model, device, save_ckpt=True, draw=True):
    # eval_model.eval()

    def save(file_path, ep):
        state = {
            "latent_enc": eval_model.latent_enc.state_dict(),
            "text_enc": eval_model.text_enc.state_dict(),
            "ep": ep,
        }

        if "motion_enc" in eval_model.state_dict():
            state["motion_enc"] = eval_model.motion_enc.state_dict()
        
        # if "text_enc" in eval_model.state_dict():
        #     state["text_enc"] = eval_model.text_enc.state_dict(),


        torch.save(state, file_path)

    motion_annotation_list = []

    R_precision_real = 0

    nb_sample = 0
    matching_score_real = 0
    for batch in eval_val_loader:
        # print(len(batch))
        texts, motions, m_lengths = batch

        motions = motions[..., :148]
        motions = motions.to(device).float().detach()
        m_lengths = m_lengths.to(device).long().detach()

        et, _ = eval_model.encode_text(texts, sample_mean=True)
        fid_em, em, _ = eval_model.encode_motion(motions, m_lengths, sample_mean=True)

        bs, _ = motions.shape[0], motions.shape[1]


        motion_annotation_list.append(_cpu_metric_tensor(fid_em))

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match

        nb_sample += bs

    motion_annotation_np = _cat_metric_tensors(motion_annotation_list)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample

    matching_score_real = matching_score_real / nb_sample

    msg = "--> \t Eva. Ep %d:, Diversity Real. %.4f, R_precision_real. (%.4f, %.4f, %.4f), matching_score_real. %.4f"%\
          (ep, diversity_real, R_precision_real[0],R_precision_real[1], R_precision_real[2], matching_score_real )
    # logger.info(msg)
    print(msg)

    if draw:
        writer.add_scalar('Eval/Diversity', diversity_real, ep)
        writer.add_scalar('Eval/top1', R_precision_real[0], ep)
        writer.add_scalar('Eval/top2', R_precision_real[1], ep)
        writer.add_scalar('Eval/top3', R_precision_real[2], ep)
        writer.add_scalar('Eval/matching_score', matching_score_real, ep)


    # msg = "--> --> \t Diversity %.5f !!!"%(diversity_real)
    # print(msg)
        # if save:
        #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision_real[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision_real[0])
        if draw: print(msg)
        best_top1 = R_precision_real[0]
        if save_ckpt:
            save(os.path.join(out_dir, 'net_best_top1.tar'), ep)
        # if save:
        #     torch.save({'vq_model': net.state_dict(), 'ep':ep}, os.path.join(out_dir, 'net_best_top1.tar'))

    if R_precision_real[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision_real[1])
        if draw: print(msg)
        best_top2 = R_precision_real[1]

    if R_precision_real[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision_real[2])
        if draw: print(msg)
        best_top3 = R_precision_real[2]

    if matching_score_real > best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_real)
        if draw: print(msg)
        best_matching = matching_score_real
        if save_ckpt:
            # save(os.path.join(out_dir, 'net_best_mm.tar'),
            #      ep
            #      )
            save(os.path.join(out_dir, 'net_best_mm.tar'), ep)
    # eval_model.train()

    return diversity_real, best_top1, best_top2, best_top3, best_matching


@torch.no_grad()
def evaluation_vqvae(out_dir, val_loader, net, writer, ep, best_fid, best_div, best_top1,
                     best_top2, best_top3, best_matching, best_mpjpe, nfeats,
                     eval_wrapper, device, fk_func, save_ckpt=True, draw=True, save_anim=True, plot_eval=None):
    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0

    mpjpe_error_sum = 0
    frame_count_sum = 0

    jitter_sum = 0
    frame_count_jitter_sum = 0

    net.eval()
    for batch in val_loader:
        texts, motions, m_lengths = batch

        # motions = motions[..., :148]
        motions = motions.to(device).float().detach()
        m_lengths = m_lengths.to(device).long().detach()

        et, _ = eval_wrapper.encode_text(texts, sample_mean=True)
        fid_em, em, _ = eval_wrapper.encode_motion(motions[..., :148], m_lengths, sample_mean=True)
        bs, _ = motions.shape[0], motions.shape[1]

        if 'salad' in out_dir:
            pred_pose_eval, loss_dict = net(motions[...,:nfeats], m_lengths)
            mask = torch.arange(motions.shape[1], device=device).unsqueeze(0).expand(motions.shape[0], -1) >= m_lengths.unsqueeze(1)
            rec_motions = pred_pose_eval.masked_fill(mask.unsqueeze(-1), 0)
        else:
            rec_motions = reconstruct_vqvae_motion(net, motions[..., :nfeats], m_lengths)
        
        fid_em_pred, em_pred, _ = eval_wrapper.encode_motion(rec_motions[..., :148], m_lengths, sample_mean=True)
        
        batch_mpjpe_error, batch_frame_count = calculate_mpjpe(
            fk_func(rec_motions), 
            fk_func(motions),
            mask=length_to_mask(m_lengths, motions.shape[1]),
            only_local=False
            )
        
        batch_jitter, batch_frame_count_jitter = calculate_jitter(
            fk_func(rec_motions),
            mask=length_to_mask(m_lengths, motions.shape[1]),
            only_endjoint=True
        )
        
        mpjpe_error_sum += batch_mpjpe_error
        frame_count_sum += batch_frame_count

        jitter_sum += batch_jitter
        frame_count_jitter_sum += batch_frame_count_jitter

        motion_pred_list.append(_cpu_metric_tensor(fid_em_pred))
        motion_annotation_list.append(_cpu_metric_tensor(fid_em))

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True,  is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = _cat_metric_tensors(motion_annotation_list)
    motion_pred_np = _cat_metric_tensors(motion_pred_list)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    mpjpe_error = mpjpe_error_sum / frame_count_sum

    jitter = jitter_sum / frame_count_jitter_sum

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Ep %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_score_real. %.4f, matching_score_pred. %.4f, mpjpe. %.4f, jitter. %.4f"%\
          (ep, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred, mpjpe_error, jitter )
    # logger.info(msg)
    print(msg)

    if draw:
        writer.add_scalar('Eval/FID', fid, ep)
        writer.add_scalar('Eval/Diversity', diversity, ep)
        writer.add_scalar('Eval/top1', R_precision[0], ep)
        writer.add_scalar('Eval/top2', R_precision[1], ep)
        writer.add_scalar('Eval/top3', R_precision[2], ep)
        writer.add_scalar('Eval/matching_score', matching_score_pred, ep)
        writer.add_scalar('Eval/mpjpe', mpjpe_error, ep)
        writer.add_scalar('Eval/jitter', jitter, ep)

    draw = True
    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw: print(msg)
        best_fid = fid
        if save_ckpt:
            torch.save({'model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!!"%(best_div, diversity)
        if draw: print(msg)
        best_div = diversity
        # if save:
        #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision[0])
        if draw: print(msg)
        best_top1 = R_precision[0]
        # if save_ckpt:
        #     torch.save({'vq_model': net.state_dict(), 'ep':ep}, os.path.join(out_dir, 'net_best_top1.tar'))

    if R_precision[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision[1])
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision[2])
        if draw: print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred > best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_pred)
        if draw: print(msg)
        best_matching = matching_score_pred
        if save_ckpt:
            torch.save({'model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_mm.tar'))

    if mpjpe_error < best_mpjpe:
        msg = f"--> --> \t mpjpe Improved from %.5f to %.5f !!!" % (best_mpjpe, mpjpe_error)
        if draw: print(msg)
        best_mpjpe = mpjpe_error
        if save_ckpt:
            torch.save({'model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_mpjpe.tar'))

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = rec_motions[rand_idx]
        captions = [texts[k] for k in rand_idx]
        lengths = m_lengths[rand_idx]
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        print(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(out_dir, 'animation', 'E%04d.npy' % (ep)), data.detach().cpu().numpy())
        # print(lengths)
        plot_eval(data, save_dir, captions, lengths)

    # if save:
    #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    # net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe


@torch.no_grad()
def evaluation_vqvae_hml(out_dir, val_loader, net, writer, ep, best_fid, best_div, best_top1,
                     best_top2, best_top3, best_matching, eval_wrapper, save=True, draw=True):
    net.eval()
    device = next(net.parameters()).device

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.to(device)
        m_length = m_length.to(device)
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]
        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        # pred_pose_eval, loss_commit, perplexity = net(motion)
        if 'salad' in out_dir:
            pred_pose_eval, loss_dict = net(motion, m_length)
            mask = torch.arange(motion.shape[1], device=device).unsqueeze(0).expand(motion.shape[0], -1) >= m_length.unsqueeze(1)
            pred_pose_eval = pred_pose_eval.masked_fill(mask.unsqueeze(-1), 0)
        else:
            pred_pose_eval = reconstruct_vqvae_motion(net, motion, m_length)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        motion_pred_list.append(_cpu_metric_tensor(em_pred))
        motion_annotation_list.append(_cpu_metric_tensor(em))

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = _cat_metric_tensors(motion_annotation_list)
    motion_pred_np = _cat_metric_tensors(motion_pred_list)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Ep %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_score_real. %.4f, matching_score_pred. %.4f"%\
          (ep, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred )
    # logger.info(msg)
    print(msg)

    if draw:
        writer.add_scalar('Eval/FID', fid, ep)
        writer.add_scalar('Eval/Diversity', diversity, ep)
        writer.add_scalar('Eval/top1', R_precision[0], ep)
        writer.add_scalar('Eval/top2', R_precision[1], ep)
        writer.add_scalar('Eval/top3', R_precision[2], ep)
        writer.add_scalar('Eval/matching_score', matching_score_pred, ep)

    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw: print(msg)
        best_fid = fid
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!!"%(best_div, diversity)
        if draw: print(msg)
        best_div = diversity
        # if save:
        #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision[0])
        if draw: print(msg)
        best_top1 = R_precision[0]
        # if save:
        #     torch.save({'vq_model': net.state_dict(), 'ep':ep}, os.path.join(out_dir, 'net_best_top1.tar'))

    if R_precision[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision[1])
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision[2])
        if draw: print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_pred)
        if draw: print(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_mm.tar'))

    # if save:
    #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer



@torch.no_grad()
def evaluation_scalemogen(
    out_dir, val_loader, predictor, vq_model, text_tokenizer, text_encoder,
    eval_wrapper, device, repeat_id,
    # Generation hyperparams
    cfg_val, tau_val, sampling_per_bits, cfg_insertion_layer, enable_positive_prompt,
    cal_mm=True, plot_fn=None, gt_leak=0, use_bf16=False, seed_mode='deterministic', base_seed=0,
    sampling_device='cuda', sampling_method='multinomial',
):
    predictor.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0
    nb_sample = 0
    
    if not cal_mm:
        num_mm_batch = 0
    else:
        num_mm_batch = 1

    coarse_to_fine_chain = coarse_to_fine_chain_from_vq(vq_model)
    scale_schedule = scale_schedule_from_vq(vq_model)

    for i, batch in enumerate(tqdm(val_loader, desc=f"Repeat {repeat_id}")):
        texts, motions, m_lengths = batch
        motions = motions.to(device).float()
        m_lengths = m_lengths.to(device).long()

        et, _ = eval_wrapper.encode_text(texts, sample_mean=True)
        fid_em, em, _ = eval_wrapper.encode_motion(motions[..., :148], m_lengths, sample_mean=True)
        bs = motions.shape[0]

        gt_ls_Bl = None
        if gt_leak > 0:
            _, _, all_bit_indices, _ = vq_model.encode(motions, m_lengths)
            gt_ls_Bl = all_bit_indices

        # For multimodality on the first batch
        if i < num_mm_batch:
            motion_multimodality_batch = []
            for k in range(bs):
                text_k = texts[k]
                motion_k_multimodality = []
                
                curr_gt_ls_Bl = None
                if gt_ls_Bl is not None:
                    curr_gt_ls_Bl = [t[k:k+1] for t in gt_ls_Bl]

                for draw_idx in range(30):
                    generated_motion, _ = gen_one_motion(
                        predictor, vq_model, text_tokenizer, text_encoder,
                        prompt=text_k,
                        g_seed=_generation_seed(seed_mode, base_seed, repeat_id, nb_sample + k, text_k, draw_idx),
                        cfg_list=cfg_val,
                        tau_list=tau_val,
                        scale_schedule=scale_schedule,
                        coarse_to_fine_chain=coarse_to_fine_chain,
                        sampling_per_bits=sampling_per_bits,
                        cfg_insertion_layer=[cfg_insertion_layer],
                        enable_positive_prompt=enable_positive_prompt,
                        m_lengths=m_lengths[k][None],
                        gt_leak=gt_leak,
                        gt_ls_Bl=curr_gt_ls_Bl,
                        use_bf16=use_bf16,
                        sampling_device=sampling_device,
                        sampling_method=sampling_method,
                    )
                    fid_em_pred_mm, _, _ = eval_wrapper.encode_motion(generated_motion.unsqueeze(0).float()[..., :148], m_lengths[k][None], sample_mean=True)
                    motion_k_multimodality.append(_cpu_metric_tensor(fid_em_pred_mm))
                
                motion_multimodality_batch.append(torch.cat(motion_k_multimodality, dim=0).unsqueeze(0))
            motion_multimodality.append(_cpu_metric_tensor(torch.cat(motion_multimodality_batch, dim=0)))

        # For FID, R-precision, etc. (1 generation per text)
        batch_pred_motions = []
        batch_pred_lengths = []
        for k in range(bs):
            text_k = texts[k]
            # text_k = "The person bends at the hips and knees to reach toward the ground. One hand extends forward to grasp an imaginary object. They straighten their legs and torso smoothly, lifting the object to waist height while maintaining balance and a neutral posture."
            # target_prompt = "The person squats deeply with bent knees and a forward-leaning torso to pick up a heavy imaginary object. Both hands grasp it firmly. The lift is slow and effortful, with visible strain in the arms and back. The shoulders tense, and the motion ends with a slight pause to regain balance."
            curr_gt_ls_Bl = None
            if gt_ls_Bl is not None:
                curr_gt_ls_Bl = [t[k:k+1] for t in gt_ls_Bl]

            generated_motion, _ = gen_one_motion(
                predictor, vq_model, text_tokenizer, text_encoder,
                prompt=text_k,
                g_seed=_generation_seed(seed_mode, base_seed, repeat_id, nb_sample + k, text_k, -1),
                cfg_list=cfg_val,
                tau_list=tau_val,
                scale_schedule=scale_schedule,
                coarse_to_fine_chain=coarse_to_fine_chain,
                sampling_per_bits=sampling_per_bits,
                cfg_insertion_layer=[cfg_insertion_layer],
                enable_positive_prompt=enable_positive_prompt,
                m_lengths=m_lengths[k][None],
                top_p=0.0,
                top_k=0,
                gt_leak=gt_leak,
                gt_ls_Bl=curr_gt_ls_Bl,
                resampling_steps=0,
                use_bf16=use_bf16,
                sampling_device=sampling_device,
                sampling_method=sampling_method,
            )
            batch_pred_motions.append(_cpu_metric_tensor(generated_motion))
            batch_pred_lengths.append(generated_motion.shape[0])
            
            if plot_fn is not None:
                save_path = os.path.join(out_dir, 'animation', f'{nb_sample+k:04d}.mp4')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plot_fn(generated_motion[None], out_dir, [text_k], [m_lengths[k]], save_path=save_path)
                # plot_fn(generated_motion[None], out_dir, [text_k], [320], save_path=save_path)
        
        max_len_pred = max(batch_pred_lengths) if batch_pred_lengths else 0
        
        # Handle case where batch_pred_motions is empty or first motion is empty
        if not batch_pred_motions or batch_pred_motions[0].shape[-1] == 0:
            continue
            
        padded_motions = torch.zeros(bs, max_len_pred, batch_pred_motions[0].shape[-1], device=device)
        for k, motion in enumerate(batch_pred_motions):
            length_k = int(m_lengths[k].item())
            # padded_motions[k, :motion.shape[0]] = motion
            padded_motions[k, :length_k] = motion[:length_k].to(device, non_blocking=True)
        
        pred_motions = padded_motions
        fid_em_pred, em_pred, _ = eval_wrapper.encode_motion(pred_motions[..., :148], m_lengths, sample_mean=True)

        motion_annotation_list.append(_cpu_metric_tensor(fid_em))
        motion_pred_list.append(_cpu_metric_tensor(fid_em_pred))

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True, is_cosine_sim=True)
        temp_match = cosine_similarity_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        
        temp_R_pred = calculate_R_precision(et.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True, is_cosine_sim=True)
        temp_match_pred = cosine_similarity_matrix(et.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R_pred
        matching_score_pred += temp_match_pred

        nb_sample += bs
        

    motion_annotation_np = _cat_metric_tensors(motion_annotation_list)
    motion_pred_np = _cat_metric_tensors(motion_pred_list)
    
    if cal_mm and motion_multimodality:
        motion_multimodality = _cat_metric_tensors(motion_multimodality)
        multimodality = calculate_multimodality(motion_multimodality, 10)
    else:
        multimodality = 0
        
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}, multimodality. {multimodality:.4f}"
    print(msg)
    
    return fid, diversity, R_precision, matching_score_pred, multimodality

@torch.no_grad()
def evaluation_scalemogen_hml(
    out_dir, val_loader, predictor, vq_model, text_tokenizer, text_encoder,
    eval_wrapper, device, repeat_id,
    # Generation hyperparams
    cfg_val, tau_val, sampling_per_bits, cfg_insertion_layer, enable_positive_prompt,
    cal_mm=True, plot_fn=None, gt_leak=0, use_bf16=False, seed_mode='deterministic', base_seed=0,
    sampling_device='cuda', sampling_method='multinomial',
):
    predictor.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0
    nb_sample = 0
    
    if not cal_mm:
        num_mm_batch = 0
    else:
        num_mm_batch = 1

    coarse_to_fine_chain = coarse_to_fine_chain_from_vq(vq_model)
    scale_schedule = scale_schedule_from_vq(vq_model)

    for i, batch in enumerate(tqdm(val_loader, desc=f"Repeat {repeat_id}")):
        word_embeddings, pos_one_hots, texts, sent_lens, motions, m_lengths, _ = batch
        
        motions = motions.to(device).float()
        m_lengths = m_lengths.to(device).long()
        word_embeddings = word_embeddings.to(device)
        pos_one_hots = pos_one_hots.to(device)
        sent_lens = sent_lens.to(device)

        # Get GT embeddings using HML wrapper interface
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_lens, motions, m_lengths)
        
        bs = motions.shape[0]

        gt_ls_Bl = None
        if gt_leak > 0:
            _, _, all_bit_indices, _ = vq_model.encode(motions, m_lengths)
            gt_ls_Bl = all_bit_indices

        # For multimodality on the first batch
        if i < num_mm_batch:
            motion_multimodality_batch = []
            for k in range(bs):
                text_k = texts[k]
                motion_k_multimodality = []
                
                curr_gt_ls_Bl = None
                if gt_ls_Bl is not None:
                    curr_gt_ls_Bl = [t[k:k+1] for t in gt_ls_Bl]

                for draw_idx in range(30):
                    generated_motion, _ = gen_one_motion(
                        predictor, vq_model, text_tokenizer, text_encoder,
                        prompt=text_k,
                        g_seed=_generation_seed(seed_mode, base_seed, repeat_id, nb_sample + k, text_k, draw_idx),
                        cfg_list=cfg_val,
                        tau_list=tau_val,
                        scale_schedule=scale_schedule,
                        coarse_to_fine_chain=coarse_to_fine_chain,
                        sampling_per_bits=sampling_per_bits,
                        cfg_insertion_layer=[cfg_insertion_layer],
                        enable_positive_prompt=enable_positive_prompt,
                        m_lengths=m_lengths[k][None],
                        gt_leak=gt_leak,
                        gt_ls_Bl=curr_gt_ls_Bl,
                        use_bf16=use_bf16,
                        sampling_device=sampling_device,
                        sampling_method=sampling_method,
                    )
                    # Encode single generated motion
                    # get_motion_embeddings expects (1, seq, dim) and (1,) len
                    fid_em_pred_mm = eval_wrapper.get_motion_embeddings(
                        generated_motion.unsqueeze(0).float()[..., :263], 
                        m_lengths[k][None]
                    )
                    motion_k_multimodality.append(_cpu_metric_tensor(fid_em_pred_mm))
                
                motion_multimodality_batch.append(torch.cat(motion_k_multimodality, dim=0).unsqueeze(0))
            motion_multimodality.append(_cpu_metric_tensor(torch.cat(motion_multimodality_batch, dim=0)))

        # For FID, R-precision, etc. (1 generation per text)
        batch_pred_motions = []
        batch_pred_lengths = []
        for k in range(bs):
            text_k = texts[k]
            curr_gt_ls_Bl = None
            if gt_ls_Bl is not None:
                curr_gt_ls_Bl = [t[k:k+1] for t in gt_ls_Bl]

            generated_motion, _ = gen_one_motion(
                predictor, vq_model, text_tokenizer, text_encoder,
                prompt=text_k,
                g_seed=_generation_seed(seed_mode, base_seed, repeat_id, nb_sample + k, text_k, -1),
                cfg_list=cfg_val,
                tau_list=tau_val,
                scale_schedule=scale_schedule,
                coarse_to_fine_chain=coarse_to_fine_chain,
                sampling_per_bits=sampling_per_bits,
                cfg_insertion_layer=[cfg_insertion_layer],
                enable_positive_prompt=enable_positive_prompt,
                m_lengths=m_lengths[k][None],
                top_p=0.0,
                top_k=0,
                gt_leak=gt_leak,
                gt_ls_Bl=curr_gt_ls_Bl,
                resampling_steps=0,
                use_bf16=use_bf16,
                sampling_device=sampling_device,
                sampling_method=sampling_method,
            )
            batch_pred_motions.append(_cpu_metric_tensor(generated_motion))
            batch_pred_lengths.append(generated_motion.shape[0])
            
            if plot_fn is not None:
                save_path = os.path.join(out_dir, 'animation', f'{nb_sample+k:04d}.mp4')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plot_fn(generated_motion[None], out_dir, [text_k], [m_lengths[k]], save_path=save_path)
        
        max_len_pred = max(batch_pred_lengths) if batch_pred_lengths else 0
        
        if not batch_pred_motions or batch_pred_motions[0].shape[-1] == 0:
            continue
            
        padded_motions = torch.zeros(bs, max_len_pred, batch_pred_motions[0].shape[-1], device=device)
        for k, motion in enumerate(batch_pred_motions):
            length_k = int(m_lengths[k].item())
            padded_motions[k, :length_k] = motion[:length_k].to(device, non_blocking=True)
        
        pred_motions = padded_motions
        
        # Get Pred embeddings using HML wrapper interface
        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_lens, pred_motions[..., :263], m_lengths)

        motion_annotation_list.append(_cpu_metric_tensor(em))
        motion_pred_list.append(_cpu_metric_tensor(em_pred))

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        
        temp_R_pred = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match_pred = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R_pred
        matching_score_pred += temp_match_pred

        nb_sample += bs
        

    motion_annotation_np = _cat_metric_tensors(motion_annotation_list)
    motion_pred_np = _cat_metric_tensors(motion_pred_list)
    
    if cal_mm and motion_multimodality:
        motion_multimodality = _cat_metric_tensors(motion_multimodality)
        multimodality = calculate_multimodality(motion_multimodality, 10)
    else:
        multimodality = 0
        
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}, multimodality. {multimodality:.4f}"
    print(msg)
    
    return fid, diversity, R_precision, matching_score_pred, multimodality
