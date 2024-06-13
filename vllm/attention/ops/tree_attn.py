# The kernels in this file are adapted from LightLLM's context_attention_fwd:
# https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py

import torch
import triton
import triton.language as tl
import pdb
from typing import List, Optional, Tuple

if triton.__version__ >= "2.1.0":
    @triton.jit
    def _fwd_kernel(
        Q,
        K_cache,
        V_cache,
        B_Loc,
        sm_scale,
        B_Ctxlen,
        prompt_lens,
        block_size,
        x,
        Out,
        stride_b_loc_b,
        stride_b_loc_s,
        stride_qbs,
        stride_qh,
        stride_qd,
        stride_obs,
        stride_oh,
        stride_od,
        stride_k_cache_bs,
        stride_k_cache_h,
        stride_k_cache_d,
        stride_k_cache_bl,
        stride_k_cache_x,
        stride_v_cache_bs,
        stride_v_cache_h,
        stride_v_cache_d,
        stride_v_cache_bl,
        num_queries_per_kv: int,
        tree_width: int,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        cur_batch = tl.program_id(0)
        cur_head = tl.program_id(1)
        start_m = tl.program_id(2)

        cur_kv_head = cur_head // num_queries_per_kv

        cur_batch_ctx_len = tl.load(B_Ctxlen + cur_batch)
        cur_batch_in_all_start_index = cur_batch * tree_width
        cur_batch_prompt_len = tl.load(prompt_lens + cur_batch)

        block_start_loc = BLOCK_M * start_m

        # initialize offsets
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_q = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs +
            cur_head * stride_qh + offs_d[None, :] * stride_qd)

        q = tl.load(Q + off_q, mask=offs_m[:, None] < tree_width, other=0.0)

        # # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            bn = tl.load(B_Loc + cur_batch * stride_b_loc_b +
                         ((start_n + offs_n) // block_size) * stride_b_loc_s,
                         mask=(start_n + offs_n) < cur_batch_ctx_len,
                         other=0)
            off_k = (bn[None, :] * stride_k_cache_bs +
                     cur_kv_head * stride_k_cache_h +
                     (offs_d[:, None] // x) * stride_k_cache_d +
                     ((start_n + offs_n[None, :]) % block_size) *
                     stride_k_cache_bl +
                     (offs_d[:, None] % x) * stride_k_cache_x)
            off_v = (
                bn[:, None] * stride_v_cache_bs +
                cur_kv_head * stride_v_cache_h +
                offs_d[None, :] * stride_v_cache_d +
                (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl)
            k = tl.load(K_cache + off_k,
                        mask=(start_n + offs_n[None, :]) < cur_batch_ctx_len,
                        other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)

            cur_step = start_n + offs_n[None, :]  # [1, BlockN]
            is_prompt = cur_step < cur_batch_prompt_len  # [1, BlockN]
            tree_mask = (
                cur_step - cur_batch_prompt_len - offs_m[:, None]
            ) % tree_width == 0  # [1, BlockN] - [BlockM, 1] = [BlockM, BlockN]

            tree_mask = is_prompt or tree_mask
            mask = tree_mask and (cur_step < cur_batch_ctx_len)

            qk = tl.where(mask, qk, -3.4028234663852886e+38)

            qk *= sm_scale

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)
            p = tl.exp(qk - m_ij[:, None])

            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            beta = tl.exp(m_ij - m_i_new)
            l_i_new = alpha * l_i + beta * l_ij
            # -- update output accumulator --
            # scale p
            p_scale = beta / l_i_new
            p = p * p_scale[:, None]
            # scale acc
            acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc

            cur_step = start_n + offs_n[:, None]  # (BlockN, 1)

            v = tl.load(V_cache + off_v,
                        mask=(start_n + offs_n[:, None]) < cur_batch_ctx_len,
                        other=0.0)

            p = p.to(v.dtype)
            acc += tl.dot(p, v)
            # # update m_i and l_is
            l_i = l_i_new
            m_i = m_i_new

        # initialize pointers to output
        off_o = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs +
            cur_head * stride_oh + offs_d[None, :] * stride_od)

        out_ptrs = Out + off_o
        tl.store(out_ptrs, acc, mask=offs_m[:, None] < tree_width)
        return

    @torch.inference_mode()
    def tree_attention_fwd(q,
                           o,
                           k_cache,
                           v_cache,
                           block_table,
                           context_len,
                           prompt_len,
                           tree_width,
                           alibi_slopes=None):

        cap = torch.cuda.get_device_capability()
        BLOCK_N = 128 if cap[0] >= 8 else 64
        BLOCK_M = triton.cdiv(tree_width, 16) * 16
        # shape constraints
        Lq = q.shape[-1]
        Lk = k_cache.shape[-1] * k_cache.shape[-3]
        assert Lq == Lk
        assert Lk in {16, 32, 64, 128}

        sm_scale = 1.0 / (Lq**0.5)
        batch, head = context_len.shape[0], q.shape[1]
        num_queries_per_kv = q.shape[1] // k_cache.shape[1]

        grid = (batch, head, triton.cdiv(tree_width, BLOCK_M))  # batch, head,

        num_warps = 8

        _fwd_kernel[grid](
            q,
            k_cache,
            v_cache,
            block_table,
            sm_scale,
            context_len,
            prompt_len,
            v_cache.shape[3],
            8,
            o,
            block_table.stride(0),
            block_table.stride(1),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            k_cache.stride(
                4),  #[num_blocks, num_kv_heads, head_size/x, block_size, x]
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(
                3),  #[num_blocks, num_kv_heads, head_size, block_size]
            num_queries_per_kv=num_queries_per_kv,
            tree_width=tree_width,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=Lk,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=1,
        )
        return

    def create_tree_attention_mask(context_len, prompt_len, max_context_len, tree_width,
                                num_kv_head, dtype):
        prompt_mask = torch.zeros((num_kv_head, tree_width, prompt_len),
                                dtype=dtype)
        none_mask_value = torch.arange(context_len - prompt_len).repeat(
            tree_width, 1) - torch.arange(tree_width)[:, None]
        none_mask_value = none_mask_value % tree_width
        none_mask_value = none_mask_value == 0

        min_value = torch.finfo(dtype).min

        generate_mask = torch.full(none_mask_value.shape, min_value, dtype=dtype)
        generate_mask[none_mask_value] = 0
        generate_mask = generate_mask.unsqueeze(0).repeat(num_kv_head, 1, 1)
        
        pad_mask = torch.zeros((num_kv_head, tree_width, max_context_len - context_len),
                                dtype=dtype)
        
        return torch.concat([pad_mask, prompt_mask, generate_mask], dim=2)
        # return torch.concat([prompt_mask, generate_mask, pad_mask], dim=2)

    def ref_masked_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask.float()

        attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
        out = torch.einsum("hqk,khd->qhd", attn_weights, value)
        return out

    def ref_query_cached_kv_attention(
        output: torch.Tensor,
        query: torch.Tensor,
        num_queries_per_kv: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
        masks: torch.Tensor
    ) -> None:
        num_query_heads = query.shape[1]
        num_kv_heads = value_cache.shape[1]
        head_size = value_cache.shape[2]
        block_size = value_cache.shape[3]

        block_tables = block_tables.cpu().tolist()
        seq_lens = seq_lens.cpu().tolist()
        num_seqs = len(seq_lens)

        query = query.reshape(num_seqs, -1, num_query_heads, head_size)
        output = output.reshape(query.shape)

        for i in range(num_seqs):
            q = query[i]
            block_table = block_tables[i]
            seq_len = int(seq_lens[i])

            keys = []
            values = []
            for j in range(seq_len):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size

                k = key_cache[block_number, :, :, block_offset, :]
                k = k.reshape(num_kv_heads, head_size)
                keys.append(k)

                v = value_cache[block_number, :, :, block_offset]
                values.append(v)
            keys = torch.stack(keys, dim=0)
            values = torch.stack(values, dim=0)
            if num_queries_per_kv > 1:
                # Handle MQA and GQA
                keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
                values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

            mask = masks[i,:,:,-seq_len:]
            # print(f"{mask.shape=}")
            alibi_bias = None
            if alibi_slopes is not None:
                # Create the ALiBi bias used in the paged attention kernel.
                position_ids = torch.arange(seq_len).int()
                alibi_bias = (position_ids - seq_len + 1).float()
                alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(
                    1, 1, -1)
                mask += alibi_bias
            out = ref_masked_attention(q, keys, values, scale, mask)
            out = out.view(-1, num_query_heads, head_size)
            output[i].copy_(out, non_blocking=True)
        output.reshape(-1, num_kv_heads, head_size)

    