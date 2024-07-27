from ast import Return
import torch
import triton
import triton.language as tl
from typing import List, Optional, Tuple


def create_tree_attention_mask(context_len, prompt_len, max_context_len,
                               tree_width, num_kv_head, dtype):
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

    pad_mask = torch.zeros(
        (num_kv_head, tree_width, max_context_len - context_len), dtype=dtype)

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


def ref_query_cached_kv_attention(output: torch.Tensor, query: torch.Tensor,
                                  num_queries_per_kv: int,
                                  key_cache: torch.Tensor,
                                  value_cache: torch.Tensor,
                                  block_tables: torch.Tensor,
                                  seq_lens: torch.Tensor, scale: float,
                                  alibi_slopes: Optional[torch.Tensor],
                                  masks: torch.Tensor) -> None:
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

        mask = masks[i, :, :, -seq_len:]
        # mask = masks[i]
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


@triton.jit(do_not_specialize=[
    "max_input_len", "stride_am_ql", "stride_am_kl", "stride_btbs", "q_len"
])
def _fwd_kernel_tree_attn(
    Q,
    K,
    V,
    Out,
    B_Seqlen,
    Block_Tables,
    Attn_Mask,
    sm_scale,
    block_size,
    max_input_len,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_kbt,
    stride_kx,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_vbt,
    stride_obs,
    stride_oh,
    stride_od,
    stride_btbs,
    stride_btd,
    stride_am_bs,
    stride_am_ql,
    stride_am_kl,
    q_len,
    kv_group_num,
    x,
    MASK_INPUT,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)  # cur_batch_req_idx
    cur_head = tl.program_id(1)
    start_m = cur_batch * q_len

    cur_kv_head = cur_head // kv_group_num
    cur_batch_seq_len = q_len  # 新加的q/kv长度
    cur_kv_len = tl.load(B_Seqlen + cur_batch)
    prompt_cache_len = cur_kv_len - q_len  # 原始kv的长度

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = tl.arange(0, BLOCK_M)

    off_q = (start_m +
             offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[
                 None, :] * stride_qd
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_num = tl.cdiv(cur_kv_len, BLOCK_N)  # 向上取整
    for start_n in range(0, block_num * BLOCK_N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        block_offset = (start_n + offs_n) % block_size
        block_table_idx = ((start_n + offs_n) // block_size)
        block_number = tl.load(Block_Tables + cur_batch * stride_btbs +
                               block_table_idx * stride_btd,
                               mask=(start_n + offs_n) < cur_kv_len,
                               other=0)
        # [d, n] stride_kbs, stride_kh, stride_kd, stride_kbt, stride_kx
        off_k = (block_number[None, :] * stride_kbs + cur_kv_head * stride_kh +
                 (offs_d[:, None] // x) * stride_kd +
                 block_offset[None, :] * stride_kbt +
                 (offs_d[:, None] % x) * stride_kx)
        # stride_vbs, stride_vh, stride_vd, stride_vbt,
        off_v = (block_number[:, None] * stride_vbs + cur_kv_head * stride_vh +
                 offs_d[None, :] * stride_vd +
                 block_offset[:, None] * stride_vbt)

        # -- compute qk ----
        k = tl.load(K + off_k,
                    mask=(start_n + offs_n[None, :]) < cur_kv_len,
                    other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        if MASK_INPUT:
            off_mask = cur_batch * stride_am_bs + offs_m[:,
                                                         None] * stride_am_ql + (
                                                             start_n + offs_n
                                                         )[None, :] * stride_am_kl
            mask = tl.load(Attn_Mask + off_mask,
                           mask=((start_n + offs_n[None, :]) < cur_kv_len) &
                           (offs_m[:, None] < cur_batch_seq_len),
                           other=-10000.0)
            qk = tl.where(mask == 0, qk, float("-100000000.0"))
        else:
            qk = tl.where(
                offs_m[:, None] + prompt_cache_len >=
                start_n + offs_n[None, :], qk, float("-100000000.0"))

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
        acc_scale = tl.where(offs_m + prompt_cache_len >= start_n, acc_scale,
                             1.0)
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(V + off_v,
                    mask=(start_n + offs_n[:, None]) < cur_kv_len,
                    other=0.0)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # initialize pointers to output
    off_o = ((start_m + offs_m[:, None]) * stride_obs + cur_head * stride_oh +
             offs_d[None, :] * stride_od)
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
    return


@torch.inference_mode()
def tree_attention_fwd(
    o,
    q,
    k,
    v,
    num_kv_heads,
    sm_scale,  # scale
    block_tables,
    b_seq_len,  # seq_lens,
    block_size,
    max_input_len,  # max_seq_len,
    attention_mask,
):
    """
        q: [batch_size * q_len, num_q_heads, head_dim]
            - 假设每个 batch 中所有查询的长度相同(例如 q_len = 26).
        k: [k_len, num_kv_heads, head_dim // x, block_size, x]
            - 跟v的区别是用x 切分head_dim.
        v: [k_len, num_kv_heads, head_dim, block_size]
        block_tables: [batch_size, block_num]
        b_seq_len: [batch_size]
            - 每个请求的kv长度
        attention_mask: [batch_size, max_len_q, max_len_k]
            - 0表示不需要计算注意力, 1表示需要计算, 默认所有head对应的mask相同.
    """
    MASK_INPUT = True
    if attention_mask is None:
        MASK_INPUT = False
        stride_am_bs, stride_am_ql, stride_am_kl = 0, 0, 0
    else:
        assert attention_mask.dim() == 3
        stride_am_bs, stride_am_ql, stride_am_kl = attention_mask.stride(
            0), attention_mask.stride(1), attention_mask.stride(2),

    # 大于等于q的长度的最小2的幂次方
    BLOCK_M = 32
    assert len(q.shape) == 3 and len(k.shape) == 5 and len(v.shape) == 4
    batch, num_q_heads, head_dim = b_seq_len.shape[0], q.shape[1], q.shape[2]
    kv_group_num = num_q_heads // num_kv_heads
    grid = (batch, num_q_heads)
    q_len = q.shape[0] // batch
    assert q_len <= BLOCK_M, "q_len should be less than BLOCK_M"
    x = k.shape[4]  # k划分的维度
    BLOCK_N = 128
    cached_bin = _fwd_kernel_tree_attn[grid](q,
                                             k,
                                             v,
                                             o,
                                             b_seq_len,
                                             block_tables,
                                             attention_mask,
                                             sm_scale,
                                             block_size,
                                             max_input_len,
                                             q.stride(0),
                                             q.stride(1),
                                             q.stride(2),
                                             k.stride(0),
                                             k.stride(1),
                                             k.stride(2),
                                             k.stride(3),
                                             k.stride(4),
                                             v.stride(0),
                                             v.stride(1),
                                             v.stride(2),
                                             v.stride(3),
                                             o.stride(0),
                                             o.stride(1),
                                             o.stride(2),
                                             block_tables.stride(0),
                                             block_tables.stride(1),
                                             stride_am_bs,
                                             stride_am_ql,
                                             stride_am_kl,
                                             q_len=q_len,
                                             kv_group_num=kv_group_num,
                                             x=x,
                                             MASK_INPUT=MASK_INPUT,
                                             BLOCK_M=BLOCK_M,
                                             BLOCK_DMODEL=head_dim,
                                             BLOCK_N=BLOCK_N,
                                             num_stages=1,
                                             num_warps=8)
    return cached_bin