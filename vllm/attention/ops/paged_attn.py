from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import time
from vllm import _custom_ops as ops
from vllm.attention.ops.prefix_prefill import context_attention_fwd
from vllm.attention.ops.tree_attn import ref_query_cached_kv_attention
from vllm.attention.ops.tree_attention import tree_attention_fwd
import pdb
# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512


@dataclass
class PagedAttentionMetadata:
    """Metadata for PagedAttention."""
    # (batch_size,). The length of sequences (entire tokens seen so far) per
    # sequence.
    seq_lens_tensor: Optional[torch.Tensor]
    # Maximum sequence length in the batch. 0 if it is prefill-only batch.
    max_decode_seq_len: int
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]


class PagedAttention:
    _function_cache = {}

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [64, 80, 96, 112, 128, 192, 256]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size * num_kv_heads * head_size)

    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = 16 // kv_cache.element_size()
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x,
                                   -1, x)
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)
        return key_cache, value_cache

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        kv_scale: float,
    ) -> None:
        ops.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping.flatten(),
            kv_cache_dtype,
            kv_scale,
        )

    @staticmethod
    def forward_decode(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        context_lens: torch.Tensor,
        max_seq_len: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
        kv_scale: float,
        attn_masks: Optional[torch.Tensor],
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
    ) -> torch.Tensor:
        if blocksparse_vert_stride is not None and blocksparse_vert_stride > 1:
            # use blocksparse paged attention
            block_size = value_cache.size(-1)
            assert (blocksparse_block_size > 0 and
                    blocksparse_block_size % block_size == 0), \
                (f"{blocksparse_block_size=} needs to be a multiple of"
                 f"{block_size=} used in block_tables.")

        output = torch.empty_like(query)
        block_size = value_cache.shape[3]
        num_seqs, num_heads, head_size = query.shape
        max_num_partitions = ((max_seq_len + _PARTITION_SIZE - 1) //
                              _PARTITION_SIZE)
        # NOTE(woosuk): We use a simple heuristic to decide whether to use
        # PagedAttention V1 or V2. If the number of partitions is 1, we use
        # V1 to avoid the overhead of reduction. Also, if the number of
        # sequences or heads is large, we use V1 since there is enough work
        # to parallelize.
        # TODO(woosuk): Tune this heuristic.
        # For context len > 8192, use V2 kernel to avoid shared memory shortage.
        use_v1 = (max_seq_len <= 8192
                  and (max_num_partitions == 1 or num_seqs * num_heads > 512))

        # Hard Codes for dispatch tree attention
        if attn_masks is not None:
            # attn_masks.shape [1, 1, 4, 18]  query[4, 32, 128]
            # st = time.perf_counter()
            # if (query.shape[0] == 26):
            # print(f"{query.shape}")
            # if query.shape[0] not in PagedAttention._function_cache:
            #     PagedAttention._function_cache[query.shape[0]] = tree_attention_fwd(output, query, key_cache, value_cache,
            #                     num_kv_heads, scale, block_tables, seq_lens,
            #                     block_size, max_seq_len,
            #                     attn_masks[:, 0, :, :])
            # else:
            #     print(f"{PagedAttention._function_cache[query.shape[0]]}")
            #     (PagedAttention._function_cache[query.shape[0]])(output, query, key_cache, value_cache,
            #                     num_kv_heads, scale, block_tables, seq_lens,
            #                     block_size, max_seq_len,
            #                     attn_masks[:, 0, :, :])
            # elif (query.shape[0] == 4):
            tree_attention_fwd(output, query, key_cache, value_cache,
                                num_kv_heads, scale, block_tables, seq_lens,
                                block_size, max_seq_len,
                                attn_masks[:, 0, :, :])
            # elif (query.shape[0] == 2):
            #     tree_attention_fwd3(output, query, key_cache, value_cache,
            #                         num_kv_heads, scale, block_tables, seq_lens,
            #                         block_size, max_seq_len,
            #                         attn_masks[:, 0, :, :])
                # ref_query_cached_kv_attention(output, query,
                #                               num_heads // num_kv_heads,
                #                               key_cache, value_cache,
                #                               block_tables, seq_lens, scale,
                #                               alibi_slopes, attn_masks)
            # ref_out = torch.empty_like(output)
            # ref_query_cached_kv_attention(output, query, num_heads // num_kv_heads, key_cache, value_cache, block_tables, seq_lens, scale, alibi_slopes, attn_masks)
            # print(f"{(time.perf_counter() - st)=}")
            # print(torch.max(torch.abs(output - ref_out)))

        # else:
        #     custom_masks = []
        #     for _ in range(num_seqs):
        #         custom_mask = create_tree_attention_mask(
        #             seq_lens[_],
        #             context_lens[_],
        #             max(seq_lens),
        #             1,
        #             num_heads,
        #             dtype=torch.float
        #         ).to(query.device)
        #         custom_masks.append(custom_mask)
        #     custom_masks = torch.stack(custom_masks, dim=0)
        #     ref_query_cached_kv_attention(
        #         output,
        #         query,
        #         num_heads // num_kv_heads,
        #         key_cache,
        #         value_cache,
        #         block_tables,
        #         seq_lens,
        #         scale,
        #         alibi_slopes,
        #         custom_masks
        #     )
        elif use_v1:
            # Run PagedAttention V1.
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
                tp_rank,
                blocksparse_local_blocks,
                blocksparse_vert_stride,
                blocksparse_block_size,
                blocksparse_head_sliding_step,
            )
        else:
            # Run PagedAttention V2.
            assert _PARTITION_SIZE % block_size == 0
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions, head_size),
                dtype=output.dtype,
                device=output.device,
            )
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)
            ops.paged_attention_v2(
                output,
                exp_sums,
                max_logits,
                tmp_output,
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
                tp_rank,
                blocksparse_local_blocks,
                blocksparse_vert_stride,
                blocksparse_block_size,
                blocksparse_head_sliding_step,
            )
        return output

    @staticmethod
    def forward_prefix(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens_tensor: torch.Tensor,
        context_lens: torch.Tensor,
        max_query_len: int,
        alibi_slopes: Optional[torch.Tensor],
        sliding_window: Optional[int],
    ) -> torch.Tensor:
        output = torch.empty_like(query)
        context_attention_fwd(
            query,
            key,
            value,
            output,
            key_cache,
            value_cache,
            block_tables,
            # query_start_loc is (batch_size + 1,)
            query_start_loc[:-1],
            seq_lens_tensor,
            context_lens,
            max_query_len,
            alibi_slopes,
            sliding_window,
        )
        return output

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)

        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        ops.copy_blocks(key_caches, value_caches, src_to_dists)
