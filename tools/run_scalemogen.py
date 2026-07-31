"""ScaleMoGen text-conditioned motion generation helpers."""

import os
from typing import List

import torch
import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt=False):
    """Encode a text prompt into the compact T5 conditioning tuple."""
    captions = [prompt]
    tokens = text_tokenizer(
        text=captions,
        max_length=text_tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    device = next(text_encoder.parameters()).device
    input_ids = tokens.input_ids.to(device, non_blocking=True)
    mask = tokens.attention_mask.to(device, non_blocking=True)
    with torch.inference_mode():
        text_features = text_encoder(input_ids=input_ids, attention_mask=mask)["last_hidden_state"].float()

    lens: List[int] = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
    max_text_len = max(lens)
    kv_compact = torch.cat([feat_i[:len_i] for len_i, feat_i in zip(lens, text_features.unbind(0))], dim=0)
    return kv_compact, lens, cu_seqlens_k, max_text_len


def move_text_condition(text_condition, device):
    """Move the tensor parts of a compact text condition to a target device."""
    kv_compact, lens, cu_seqlens_k, max_text_len = text_condition
    return kv_compact.to(device), lens, cu_seqlens_k.to(device), max_text_len


def gen_one_motion(
    predictor,
    vae,
    text_tokenizer,
    text_encoder,
    prompt,
    all_mask=None,
    motion_padding_mask=None,
    cfg_list=None,
    tau_list=None,
    negative_prompt="",
    coarse_to_fine_chain=None,
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_type=32,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
    input_motions=None,
    m_lengths=None,
    resampling_steps=0,
    sampling_func=None,
    return_idx_prob=False,
    trunk_scale=1000,
    use_bf16=False,
    sampling_device="cuda",
    sampling_method="multinomial",
):
    """Generate one motion sample with the ScaleMoGen autoregressive predictor."""
    cfg_list = [] if cfg_list is None else cfg_list
    tau_list = [] if tau_list is None else tau_list
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)

    text_cond_tuple = encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt)
    negative_label = encode_prompt(text_tokenizer, text_encoder, negative_prompt) if negative_prompt else None
    predictor_device = next(predictor.parameters()).device
    text_cond_tuple = move_text_condition(text_cond_tuple, predictor_device)
    if negative_label is not None:
        negative_label = move_text_condition(negative_label, predictor_device)

    with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16, cache_enabled=False):
        outputs = predictor.autoregressive_infer_cfg(
            vae=vae,
            coarse_to_fine_chain=coarse_to_fine_chain,
            scale_schedule=scale_schedule,
            all_mask=all_mask,
            motion_padding_mask=motion_padding_mask,
            label_B_or_BLT=text_cond_tuple,
            g_seed=g_seed,
            B=1,
            negative_label_B_or_BLT=negative_label,
            force_gt_Bhw=None,
            cfg_sc=cfg_sc,
            cfg_list=cfg_list,
            tau_list=tau_list,
            top_k=top_k,
            top_p=top_p,
            returns_vemb=1,
            ratio_Bl1=None,
            gumbel=gumbel,
            norm_cfg=False,
            cfg_exp_k=cfg_exp_k,
            cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type,
            softmax_merge_topk=softmax_merge_topk,
            ret_img=True,
            trunk_scale=trunk_scale,
            gt_leak=gt_leak,
            gt_ls_Bl=gt_ls_Bl,
            inference_mode=True,
            sampling_per_bits=sampling_per_bits,
            input_motions=input_motions,
            m_lengths=m_lengths,
            resampling_steps=resampling_steps,
            sampling_func=sampling_func,
            return_idx_prob=return_idx_prob,
            sampling_device=sampling_device,
            sampling_method=sampling_method,
        )

    generated_motion = outputs[2][0]
    reconstructed_motion = outputs[3][0] if outputs[3] is not None else None
    if return_idx_prob:
        return generated_motion, reconstructed_motion, outputs[4], outputs[5]
    return generated_motion, reconstructed_motion
