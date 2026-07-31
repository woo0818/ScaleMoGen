"""Token-level motion editing helpers for ScaleMoGen.

Idea provenance: inspired by the AREdit paper. This implementation was written
for ScaleMoGen rather than copied from an official public code release.
"""

import torch
from einops import rearrange

def sampling_func_wrapper(
    idx_BlV_cache,
    prob_cache,
    tau=0.3,
    gamma=3,
    coarse_to_fine_chain=None,
    motion_scale_schedule=None,
    edit_temp=None,
    edit_joints=None,
    len_scale_factor=4,
    edit_threshold=None,
):
    """Build a token sampling function for prompt-guided motion editing."""

    def sampling_func(logits_BlV, si, rng=None):
        if idx_BlV_cache is None:
            return None, None
        
        B, l, V = logits_BlV.shape
        probs = logits_BlV.softmax(dim=-1).view(-1, V)

        num_samples = 1

        max_edit_scale = edit_threshold if edit_threshold is not None else len(coarse_to_fine_chain)
        edit_flag = si < max_edit_scale

        t_scale, spatial_groups = coarse_to_fine_chain[si]
        _, T, J = motion_scale_schedule[si]


        i_mask = rearrange(idx_BlV_cache[si], "B (T J) V -> B T J V", T=T, J=J)

        edit_TJ = torch.zeros_like(i_mask, dtype=torch.bool)  # (B,T,J,V)

        if edit_temp is not None:
            edit_t0, edit_t1 = edit_temp

            t0_bin = int(edit_t0 / len_scale_factor) // int(t_scale)
            t1_bin = int(edit_t1 / len_scale_factor) // int(t_scale)

            t0_bin = max(0, min(T, t0_bin))
            t1_bin = max(0, min(T, t1_bin))
            if t1_bin > t0_bin:
                edit_TJ[:, t0_bin:t1_bin, :, :] = True

        if edit_joints is not None:
            if isinstance(edit_joints, int):
                edit_joints_list = [edit_joints]
            else:
                edit_joints_list = list(edit_joints)
            edit_joints_set = set(edit_joints_list)

            for g_idx, g in enumerate(spatial_groups):
                group_indices = g if isinstance(g, (list, tuple)) else [g]
                if set(group_indices).issubset(edit_joints_set):
                    if edit_temp is not None:
                        # not necessary for this mode
                        edit_TJ[:, t0_bin:t1_bin, g_idx, :] = True 
                    else:
                        edit_TJ[:, :, g_idx, :] = True 
        
        edit_TJ = rearrange(edit_TJ, "B T J V -> (B T J V) 1")

        if edit_flag:
            i_cache = idx_BlV_cache[si] if idx_BlV_cache else None
            p_cache = prob_cache[si] if prob_cache else None

            idx_Bl_cache = rearrange(i_cache, "B l V->(B l V) 1")
            probs_cache = rearrange(p_cache, "B l V d->(B l V) d")

            conf_to_change = (-torch.gather(probs, 1, idx_Bl_cache) + torch.gather(probs_cache, 1, idx_Bl_cache)).clip(0)
            probs = torch.scatter(probs, 1, idx_Bl_cache, torch.gather(probs, 1, idx_Bl_cache))

            if si < gamma:
                if edit_TJ.sum() > 0:
                    _, idx = torch.topk(probs, k=num_samples, dim=-1)
                    idx = torch.multinomial(
                        probs,
                        num_samples=num_samples,
                        replacement=True,
                        generator=rng,
                    )
                    idx = idx * edit_TJ + idx_Bl_cache * (~edit_TJ)
                    
                else:
                    idx = idx_Bl_cache
                    
            else:
                _, idx = torch.topk(probs, k=num_samples, dim=-1)
                idx = torch.multinomial(
                    probs,
                    num_samples=num_samples,
                    replacement=True,
                    generator=rng,
                )

                mask = conf_to_change > tau

                if edit_TJ.sum() > 0:
                    mask = mask + edit_TJ

                idx = idx * mask + idx_Bl_cache * (~mask)

        else:
            idx = torch.multinomial(
                probs,
                num_samples=num_samples,
                replacement=True,
                generator=rng,
            )

        idx = idx.view(B, l)
        probs = rearrange(probs, "(b l s) d -> b l s d", b=B, l=l, s=1)
        
        return idx[:, :, None], probs
    
    return sampling_func
