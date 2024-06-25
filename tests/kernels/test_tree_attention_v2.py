import random
from typing import List, Optional, Tuple

import pytest
import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

from vllm import _custom_ops as ops
from vllm.attention.ops.tree_attn import tree_attention_fwd
from vllm.utils import get_max_shared_memory_bytes, is_hip

from .allclose_default import get_default_atol, get_default_rtol

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
# MAX_SEQ_LEN = get_max_shared_memory_bytes() // FLOAT32_BYTES - 512
MAX_SEQ_LEN = 5
# There may not be enough gpu memory due to large NUM_BLOCKS.
# Reduce NUM_BLOCKS when it happens.
NUM_BLOCKS = 4321  # Arbitrary values for testing
PARTITION_SIZE = 512
# flshattF and tritonflashattF supported: {torch.float16, torch.bfloat16}
# DTYPES = [torch.half, torch.bfloat16, torch.float
#           ] if not is_hip() else [torch.half, torch.bfloat16]
DTYPES = [torch.half]
# NUM_GEN_SEQS = [1, 9, 35]  # Arbitrary values for testing
NUM_GEN_SEQS = [2]  # Arbitrary values for testing
NUM_PREFILL_SEQS = [3]  # Arbitrary values for testing
# NUM_HEADS = [(40, 40), (64, 8)]  # Arbitrary values for testing
NUM_HEADS = [(4, 4)]  # Arbitrary values for testing

# FlashAttention forward only supports head dimension at most 128
# https://github.com/ROCmSoftwarePlatform/flash-attention/blob/3d2b6f5d037782cc2c906909a46fb7e2e1b48b25/csrc/flash_attn_rocm/flash_api.cpp#L62
# HEAD_SIZES = [64, 80, 96, 112, 128, 192, 256
#               ] if not is_hip() else [64, 80, 96, 112, 128]
HEAD_SIZES = [128]

# BLOCK_SIZES = [16, 32]
BLOCK_SIZES = [16]
# USE_ALIBI = [False, True]
USE_ALIBI = [False]
# KV_CACHE_DTYPE = ["auto", "fp8"]
KV_CACHE_DTYPE = ["auto"]
SEEDS = [0]
# CUDA_DEVICES = [
#     f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
# ]

CUDA_DEVICES = ["cuda:0"]

# QUERY_LEN =  [1, 7, 31]
QUERY_LEN = [3]


def create_tree_attention_mask(context_len, prompt_len, tree_width,
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
    return torch.concat([prompt_mask, generate_mask], dim=2)


def create_tree_attention_mask_v2(context_len, q_len, num_kv_head, dtype):
    mask = torch.zeros((num_kv_head, q_len, context_len), dtype=dtype)

    min_value = torch.finfo(dtype).min

    for s in range(q_len):
        num_masked = torch.randint(1, context_len, (1, )).item()
        mask_indices = torch.randperm(context_len)[:num_masked]
        for b in range(num_kv_head):
            mask[b, s, mask_indices] = min_value

    return mask


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

        mask = masks[i]
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


@pytest.mark.parametrize("num_seqs", NUM_GEN_SEQS)
@pytest.mark.parametrize("q_len", [1])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("use_alibi", USE_ALIBI)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_paged_attention(
    kv_cache_factory,
    num_seqs: int,
    q_len: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    seed: int,
    device: str,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(num_seqs * q_len,
                        num_query_heads,
                        head_size,
                        dtype=dtype)
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=torch.float)

    seq_lens = [random.randint(q_len, MAX_SEQ_LEN) for _ in range(num_seqs)]
    seq_lens[-1] = MAX_SEQ_LEN
    max_seq_len = max(seq_lens)
    prompt_lens = [x - q_len for x in seq_lens]
    seq_lens = torch.tensor(seq_lens, dtype=torch.int)

    # Create the block tables.
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(NUM_BLOCKS, block_size, 1,
                                                num_kv_heads, head_size,
                                                kv_cache_dtype, dtype, seed,
                                                device)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Using default kv_scale
    kv_scale = 1.0

    # Test for origin paged attention kernel. fit for qlen == 1 and causal mask
    assert (q_len == 1)

    #  The flattened mask tensor, shape: (sum(q_len[i] * k_len[i] for i in range(batch_size)).
    #  using triu attention mask (fill -inf for masking and 0 for others) is equivalent to setting causal=True.
    custom_masks = []
    for _ in range(num_seqs):
        custom_mask = create_tree_attention_mask(seq_lens[_],
                                                 prompt_lens[_],
                                                 q_len,
                                                 num_query_heads,
                                                 dtype=torch.float)
        custom_masks.append(custom_mask)
    # custom_masks = torch.stack(custom_masks, dim=0)

    output = torch.empty_like(query)
    ops.paged_attention_v1(
        output,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens,
        block_size,
        max_seq_len,
        alibi_slopes,
        kv_cache_dtype,
        kv_scale,
    )

    ref_output = torch.empty_like(query)
    ref_query_cached_kv_attention(ref_output, query, num_queries_per_kv,
                                  key_cache, value_cache, block_tables,
                                  seq_lens, scale, alibi_slopes, custom_masks)

    # NOTE(woosuk): Due to the kernel-level differences in the two
    # implementations, there is a small numerical difference in the two
    # outputs. Thus, we use a relaxed tolerance for the test.
    atol = get_default_atol(output) if is_hip() else 1e-3
    rtol = get_default_rtol(output) if is_hip() else 1e-5

    # NOTE(zhaoyang): FP8 KV Cache will introduce quantization error,
    # so we use a relaxed tolerance for the test.
    atol, rtol = 1e-3, 1e-5
    if kv_cache_dtype == "fp8":
        atol, rtol = 1e-2, 1e-5
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol)


# @pytest.mark.parametrize("num_seqs", NUM_GEN_SEQS)
# @pytest.mark.parametrize("q_len", QUERY_LEN)
# @pytest.mark.parametrize("num_heads", NUM_HEADS)
# @pytest.mark.parametrize("head_size", HEAD_SIZES)
# @pytest.mark.parametrize("use_alibi", USE_ALIBI)
# @pytest.mark.parametrize("block_size", BLOCK_SIZES)
# @pytest.mark.parametrize("dtype", DTYPES)
# @pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
# @pytest.mark.parametrize("seed", SEEDS)
# @pytest.mark.parametrize("device", CUDA_DEVICES)
# def test_tree_attention_v1(
#     kv_cache_factory,
#     num_seqs: int,
#     q_len: int,
#     num_heads: Tuple[int, int],
#     head_size: int,
#     use_alibi: bool,
#     block_size: int,
#     dtype: torch.dtype,
#     kv_cache_dtype: str,
#     seed: int,
#     device: str,
# ) -> None:
#     random.seed(seed)
#     torch.random.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#     torch.set_default_device(device)
#     scale = float(1.0 / (head_size**0.5))
#     num_query_heads, num_kv_heads = num_heads
#     query = torch.empty(num_seqs * q_len, num_query_heads, head_size, dtype=dtype)
#     query.uniform_(-scale, scale)

#     assert num_query_heads % num_kv_heads == 0
#     num_queries_per_kv = num_query_heads // num_kv_heads
#     alibi_slopes = None
#     if use_alibi:
#         alibi_slopes = torch.randn(num_query_heads, dtype=torch.float)

#     seq_lens = [random.randint(q_len, MAX_SEQ_LEN) for _ in range(num_seqs)]
#     seq_lens[-1] = MAX_SEQ_LEN
#     max_seq_len = max(seq_lens)
#     prompt_lens = [x - q_len for x in seq_lens]
#     seq_lens = torch.tensor(seq_lens, dtype=torch.int)
#     prompt_lens = torch.tensor(prompt_lens, dtype=torch.int)

#     # Create the block tables.
#     max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
#     block_tables = []
#     for _ in range(num_seqs):
#         block_table = [
#             random.randint(0, NUM_BLOCKS - 1)
#             for _ in range(max_num_blocks_per_seq)
#         ]
#         block_tables.append(block_table)
#     block_tables = torch.tensor(block_tables, dtype=torch.int)

#     # Create the KV caches.
#     key_caches, value_caches = kv_cache_factory(NUM_BLOCKS, block_size, 1,
#                                                 num_kv_heads, head_size,
#                                                 kv_cache_dtype, dtype, seed,
#                                                 device)
#     key_cache, value_cache = key_caches[0], value_caches[0]

#     # Using default kv_scale
#     kv_scale = 1.0

#     # Test for simple tree_attention_fwd

#     #  The flattened mask tensor, shape: (sum(q_len[i] * k_len[i] for i in range(batch_size)).
#     #  using triu attention mask (fill -inf for masking and 0 for others) is equivalent to setting causal=True.
#     custom_masks = []
#     for _ in range(num_seqs):
#         custom_mask = create_tree_attention_mask(
#             seq_lens[_],
#             prompt_lens[_],
#             q_len,
#             num_query_heads,
#             dtype=torch.float
#         )
#         custom_masks.append(custom_mask)
#     # custom_masks = torch.stack(custom_masks, dim=0)

#     output = torch.empty_like(query)
#     tree_attention_fwd(query, output, key_cache, value_cache, block_tables,
#                        seq_lens, prompt_lens, q_len, alibi_slopes)

#     ref_output = torch.empty_like(query)
#     ref_query_cached_kv_attention(
#         ref_output,
#         query,
#         num_queries_per_kv,
#         key_cache,
#         value_cache,
#         block_tables,
#         seq_lens,
#         scale,
#         alibi_slopes,
#         custom_masks
#     )

#     # NOTE(woosuk): Due to the kernel-level differences in the two
#     # implementations, there is a small numerical difference in the two
#     # outputs. Thus, we use a relaxed tolerance for the test.
#     atol = get_default_atol(output) if is_hip() else 1e-3
#     rtol = get_default_rtol(output) if is_hip() else 1e-5

#     # NOTE(zhaoyang): FP8 KV Cache will introduce quantization error,
#     # so we use a relaxed tolerance for the test.
#     atol, rtol = 1e-3, 1e-5
#     if kv_cache_dtype == "fp8":
#         atol, rtol = 1e-2, 1e-5
#     assert torch.allclose(output, ref_output, atol=atol, rtol=rtol)


@pytest.mark.parametrize("num_seqs", NUM_GEN_SEQS)
@pytest.mark.parametrize("q_len", QUERY_LEN)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("use_alibi", USE_ALIBI)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_tree_attention_v2(
    kv_cache_factory,
    num_seqs: int,
    q_len: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    seed: int,
    device: str,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(num_seqs * q_len,
                        num_query_heads,
                        head_size,
                        dtype=dtype)
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=torch.float)

    seq_lens = [random.randint(q_len, MAX_SEQ_LEN) for _ in range(num_seqs)]
    seq_lens[-1] = MAX_SEQ_LEN
    max_seq_len = max(seq_lens)
    prompt_lens = [x - q_len for x in seq_lens]
    seq_lens = torch.tensor(seq_lens, dtype=torch.int)
    prompt_lens = torch.tensor(prompt_lens, dtype=torch.int)

    # Create the block tables.
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(NUM_BLOCKS, block_size, 1,
                                                num_kv_heads, head_size,
                                                kv_cache_dtype, dtype, seed,
                                                device)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Using default kv_scale
    kv_scale = 1.0

    # Test for common tree_attention_fwd
    custom_masks = []
    flattened_mask_tensor = []
    for _ in range(num_seqs):
        # [num_query_heads, q_len, seq_len]
        custom_mask = create_tree_attention_mask_v2(seq_lens[_],
                                                    q_len,
                                                    num_query_heads,
                                                    dtype=torch.float)
        custom_masks.append(custom_mask)
        flattened_mask_tensor.append(custom_mask[0].view(-1))

    #  The flattened mask tensor, shape: (sum(q_len * seq_lens[i] for i in range(batch_size)).
    flattened_mask_tensor = torch.cat(flattened_mask_tensor, dim=0)
    output = torch.empty_like(query)
    # tree_attention_fwd_v2(query, output, key_cache, value_cache, block_tables,
    #    seq_lens, prompt_lens, flattened_mask_tensor, alibi_slopes)

    ref_output = torch.empty_like(query)
    ref_query_cached_kv_attention(ref_output, query, num_queries_per_kv,
                                  key_cache, value_cache, block_tables,
                                  seq_lens, scale, alibi_slopes, custom_masks)
    # print(f"{ref_output=}")
    # # NOTE(woosuk): Due to the kernel-level differences in the two
    # # implementations, there is a small numerical difference in the two
    # # outputs. Thus, we use a relaxed tolerance for the test.
    # atol = get_default_atol(output) if is_hip() else 1e-3
    # rtol = get_default_rtol(output) if is_hip() else 1e-5

    # # NOTE(zhaoyang): FP8 KV Cache will introduce quantization error,
    # # so we use a relaxed tolerance for the test.
    # atol, rtol = 1e-3, 1e-5
    # if kv_cache_dtype == "fp8":
    #     atol, rtol = 1e-2, 1e-5
    # assert torch.allclose(output, ref_output, atol=atol, rtol=rtol)
