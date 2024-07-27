from typing import List, Optional

import torch

from vllm import _custom_ops as ops

try:
    from vllm.attention.backends.flash_attn import FlashAttentionMetadata
except ModuleNotFoundError:
    # vllm_flash_attn is not installed, use the identical ROCm FA metadata
    from vllm.attention.backends.rocm_flash_attn import (
        ROCmFlashAttentionMetadata as FlashAttentionMetadata)

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, MultiModalConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig, SpeculativeConfig)
from vllm.logger import init_logger
from vllm.sequence import (ExecuteModelRequest, IntermediateTensors,
                           SamplerOutput)
from vllm.worker.model_runner import (ModelInputForGPUWithSamplingMetadata,
                                      ModelRunner)

logger = init_logger(__name__)

# A flag to enable debug prints for the updated input tensors
# before each step.
debug_advance_input = False
# A flag to allow GPU advance step for draft model runner.
# Set to False for debugging.
allow_gpu_advance_step = False

class TP1DraftModelRunner(ModelRunner):
    """Specialized model runner for speculative decoding draft model.
    Since the draft model always execute k forward passes consecutively to
    generate k speculative tokens in a single speculative decoding step,
    we could get rid of most CPU-GPU synchronization and data transfer
    overheads by keeping model input and output tensors on GPU all the time.

    TODOs:
    1. Currently supports only flash-attn, add support for other attn_backends.
    2. Support TP > 1 (this requires some designs because we do not expect
       any broadcasting inside execute_model).
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        multimodal_config: Optional[MultiModalConfig] = None,
        speculative_config: Optional[SpeculativeConfig] = None,
        prompt_adapter_config: Optional[PromptAdapterConfig] = None,
        return_hidden_states: bool = False,
    ):
        # if return_hidden_states:
        #     raise ValueError(
        #         "return_hidden_states is not supported for 
        #           TP1DraftModelRunner."
        #     )

        super().__init__(
            model_config=model_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            cache_config=cache_config,
            load_config=load_config,
            lora_config=lora_config,
            kv_cache_dtype=kv_cache_dtype,
            is_driver_worker=is_driver_worker,
            multimodal_config=multimodal_config,
            speculative_config=speculative_config,
            prompt_adapter_config=prompt_adapter_config,
            return_hidden_states=return_hidden_states,
        )

    def _update_flash_attn_metadata(self, attn_metadata, num_seqs,
                                    num_queries):
        assert isinstance(attn_metadata, FlashAttentionMetadata)

        if num_seqs != num_queries:
            assert num_seqs > num_queries
            assert attn_metadata.use_cuda_graph

        assert attn_metadata.num_prefills == 0
        assert attn_metadata.num_prefill_tokens == 0
        # assert attn_metadata.num_decode_tokens == num_seqs
        assert attn_metadata.slot_mapping.shape == (num_seqs, )

        assert len(attn_metadata.seq_lens) == num_seqs
        assert attn_metadata.seq_lens_tensor.shape == (num_seqs, )
        assert attn_metadata.max_query_len == 1
        assert attn_metadata.max_prefill_seq_len == 0
        assert attn_metadata.max_decode_seq_len == max(attn_metadata.seq_lens)

        assert attn_metadata.query_start_loc.shape == (num_queries + 1, )
        assert attn_metadata.seq_start_loc.shape == (num_seqs + 1, )

        assert attn_metadata.context_lens_tensor.shape == (num_queries, )

        assert attn_metadata.block_tables.shape[0] == num_seqs

        # Update query lengths. Note that we update only queries and not seqs,
        # since tensors may be padded due to captured cuda graph batch size
        for i in range(num_queries):
            attn_metadata.seq_lens[i] += 1
        attn_metadata.max_decode_seq_len = max(attn_metadata.seq_lens)

    def _update_sampling_metadata(self, sampling_metadata, num_seqs,
                                  num_queries):

        assert sampling_metadata.num_prompts == 0
        assert len(sampling_metadata.seq_groups) == num_queries
        assert sampling_metadata.selected_token_indices.shape == (
            num_queries, )
        # assert sampling_metadata.categorized_sample_indices == TODO: Add if needed # noqa: E501

        # Verify that all sequences are decodes
        for i in range(num_queries):
            seq_group = sampling_metadata.seq_groups[i]

            assert seq_group.is_prompt is False  # No prompt
            assert seq_group.prompt_logprob_indices == []  # No prompt
            assert seq_group.sample_indices == [i]  # Simple
            assert seq_group.seq_len is None  # Decode
            assert seq_group.query_len is None  # Decode

    def _gpu_advance_step(
            self, model_input: ModelInputForGPUWithSamplingMetadata,
            last_output: SamplerOutput
    ) -> ModelInputForGPUWithSamplingMetadata:
        # Currently, we expect "decode mode" only
        assert not model_input.is_prompt

        # Get num_seqs
        num_seqs = len(model_input.seq_lens)
        num_queries = len(model_input.query_lens)

        # Get output tokens GPU tensor
        sampled_token_ids = last_output.sampled_token_ids
        assert sampled_token_ids is not None

        # Update attn_metadata
        attn_metadata = model_input.attn_metadata
        assert isinstance(attn_metadata, FlashAttentionMetadata)
        self._update_flash_attn_metadata(attn_metadata, num_seqs, num_queries)

        # Update GPU tensors
        ops.advance_step(num_seqs=num_seqs,
                         num_queries=num_queries,
                         block_size=self.cache_config.block_size,
                         input_tokens=model_input.input_tokens,
                         sampled_token_ids=sampled_token_ids,
                         input_positions=model_input.input_positions,
                         seq_lens=attn_metadata.seq_lens_tensor,
                         slot_mapping=attn_metadata.slot_mapping,
                         block_tables=attn_metadata.block_tables)

        # Update sampling_metadata
        sampling_metadata = model_input.sampling_metadata
        self._update_sampling_metadata(sampling_metadata, num_seqs,
                                       num_queries)

        # Create new input
        new_model_input = self._model_input_cls(
            input_tokens=model_input.input_tokens,
            input_positions=model_input.input_positions,
            attn_metadata=attn_metadata,
            seq_lens=attn_metadata.seq_lens,
            query_lens=model_input.query_lens,
            lora_mapping=model_input.lora_mapping,
            lora_requests=model_input.lora_requests,
            spec_input_hidden_states=last_output.hidden_states,
            multi_modal_kwargs=model_input.multi_modal_kwargs,
            sampling_metadata=model_input.sampling_metadata,
            is_prompt=False,
        )

        # Ensure we skip CPU samples
        assert new_model_input.sampling_metadata.skip_sampler_cpu_output is True
        # We can reuse sampling tensors since every decode iteration is the same
        new_model_input.sampling_metadata.reuse_sampling_tensors = True

        if debug_advance_input:
            logger.debug("NEW INPUT: ")
            logger.debug("  input_tokens = %s", new_model_input.input_tokens)
            logger.debug("  input_positions = %s",
                         new_model_input.input_positions)
            logger.debug("  seq_lens = %d", new_model_input.seq_lens)
            logger.debug("  query_lens = %d", new_model_input.query_lens)
            logger.debug("  attn_metadata:")
            logger.debug("    seq_lens_tensor: %s",
                         attn_metadata.seq_lens_tensor)
            logger.debug("    slot_mapping: %s", attn_metadata.slot_mapping)
            logger.debug("    block_tables: %s", attn_metadata.block_tables)

        return new_model_input

    def supports_gpu_multi_step(self, execute_model_req: ExecuteModelRequest):
        """Determines if draft_model_runner GPU multi-step can be used.
        Currently required conditions are:
            1. Only decodes 
            2. Only flash-attn
            3. No LORA
            4. No prompt_adapter_config
        """
        if not allow_gpu_advance_step:
            return False

        # We allow multi-step GPU only in decode mode
        for seq_group in execute_model_req.seq_group_metadata_list:
            if seq_group.is_prompt:
                return False

        # TODO: Add support for other attn backends
        if self.attn_backend.get_name() != "flash-attn":
            return False

        # TODO: Add support for LORA
        if self.lora_config:
            return False

        # TODO: Add soft-tuning prompt adapter support
        if self.prompt_adapter_config:
            return False

        return True

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        """Executes num_steps forward passes with advacement of input tensors 
        on the GPU. Look at supports_gpu_multi_step(..) for pre-conditions.

        Optimizations used:
            1. Input tensors are updated on the GPU directly
            2. Skips GPU=>CPU serialization of sampler outputs (we don't need 
                them since we do batch expansion later that uses GPU outputs)
            3. Reuses sampling tensors (since we run only decodes and they have
                a repeating sampling logic)
        """

        if self.model.config.model_type in ["eagle"] and num_steps > 1:
            return self.execute_eagle_draft_model(
                model_input, kv_caches, intermediate_tensors, num_steps)

        # When num_steps == 1, we execute the fallback here for the GPU
        # advance_step, which runs prepare_inputs on CPU and for each spec
        # iteration invokes this function only once
        # (Look at multi-step-worker code)
        is_fallback = num_steps == 1
        if not is_fallback:
            # Since we do not broadcast data inside execute_model anymore,
            # we need to figure out the best way to support TP > 1 in this
            # case, because we will at least need to broadcast the sampled
            # tokens to all workers.
            if not self.is_driver_worker:
                raise ValueError("TP1DraftModelRunner only supports TP=1.")

            # Sanity
            if self.lora_config is not None:
                raise ValueError("TP1DraftModelRunner has no support for LORA")
            if self.prompt_adapter_config is not None:
                raise ValueError("TP1DraftModelRunner has no support for "
                                 "prompt_adapter_config")
            if model_input.multi_modal_kwargs:
                raise ValueError(
                    "TP1DraftModelRunner has no support for multi_modal_kwargs"
                )
        else:
            if self.lora_config:
                assert model_input.lora_requests is not None
                assert model_input.lora_mapping is not None
                self.set_active_loras(model_input.lora_requests,
                                      model_input.lora_mapping)

            if self.prompt_adapter_config:
                assert model_input.prompt_adapter_requests is not None
                assert model_input.prompt_adapter_mapping is not None
                self.set_active_prompt_adapters(
                    model_input.prompt_adapter_requests,
                    model_input.prompt_adapter_mapping)

        # Detect exec mode
        assert model_input.attn_metadata is not None
        use_cuda_graph = False
        if model_input.attn_metadata.num_prefills > 0:
            # In this case, execute_model(..) was called directly
            if num_steps > 1:
                raise ValueError(
                    "execute_model(..) of draft_model_runner can be called "
                    "directly only with a single-step prefill")
        else:
            # We can skip CPU samples for spec token generation.
            # (We do allow CPU samples for num_steps == 1 to support the
            # fallback case, where supports_gpu_multi_step(..) does not pass)
            model_input.sampling_metadata.skip_sampler_cpu_output = (
                not is_fallback)

            # Attn attr defines if we use cuda graphs
            use_cuda_graph = model_input.attn_metadata.use_cuda_graph

        # Get model
        if use_cuda_graph:
            graph_batch_size = model_input.input_tokens.shape[0]
            model_executable = (self.graph_runners[model_input.virtual_engine]
                                [graph_batch_size])
        else:
            model_executable = self.model

        outputs: List[SamplerOutput] = []
        for step in range(num_steps):
            multi_modal_kwargs = model_input.multi_modal_kwargs or {}

            spec_model_kwargs = {
                "spec_input_hidden_states":
                model_input.spec_input_hidden_states,
            } if model_input.spec_input_hidden_states is not None else {}

            # Run model
            hidden_states = model_executable(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                kv_caches=kv_caches,
                attn_metadata=model_input.attn_metadata,
                intermediate_tensors=intermediate_tensors,
                **multi_modal_kwargs,
                **spec_model_kwargs,
            )

            # Compute the logits.
            logits = self.model.compute_logits(hidden_states,
                                               model_input.sampling_metadata)

            # Sample the next token.
            outputs.append(
                self.model.sample(
                    logits=logits,
                    sampling_metadata=model_input.sampling_metadata,
                ))

            if self.return_hidden_states:
                # we only need to pass hidden states of most recent token
                assert model_input.sampling_metadata is not None
                indices = model_input.sampling_metadata.selected_token_indices
                if model_input.is_prompt:
                    hidden_states = hidden_states.index_select(0, indices)
                elif use_cuda_graph:
                    hidden_states = hidden_states[:len(indices)]
                else:
                    hidden_states = hidden_states

                outputs[-1].hidden_states = hidden_states

            # Prepare inputs for the next step
            if step != num_steps - 1:
                model_input = self._gpu_advance_step(model_input, outputs[-1])

        return outputs


    @torch.inference_mode()
    def execute_eagle_draft_model(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:  # Detect exec mode
        assert model_input.attn_metadata is not None
        use_cuda_graph = False
        is_fallback = num_steps == 1
        if model_input.attn_metadata.num_prefills > 0:
            if num_steps > 1:
                raise ValueError(
                    "execute_model(..) of draft_model_runner can be called "
                    "directly only with a single-step prefill"
                )
        else:
            model_input.sampling_metadata.skip_sampler_cpu_output = not is_fallback
            use_cuda_graph = model_input.attn_metadata.use_cuda_graph

        # Get model
        if use_cuda_graph:
            graph_batch_size = model_input.input_tokens.shape[0]
            model_executable = self.graph_runners[model_input.virtual_engine][
                graph_batch_size
            ]
        else:
            model_executable = self.model

        batch_size = len(model_input.seq_lens)
        token_id_list: List[Optional[torch.Tensor]] = []
        token_prob_list: List[Optional[torch.Tensor]] = []
        parents_token_id_list: List[Optional[torch.Tensor]] = []
        position_ids_init = torch.zeros(
            self.model.config.topk,
            device=self.device,
            dtype=torch.long,
        )

        # init tree mask
        tree_mask_init = torch.eye(
            self.model.config.topk, device=self.device
        )[None, None]
        tree_mask: Optional[torch.Tensor] = None

        # init topk_cs_index from topk ([batch_size, topk])
        topk_cs_index = (
            torch.arange(self.model.config.topk, device=self.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        multi_modal_kwargs = model_input.multi_modal_kwargs or {}

        spec_model_kwargs = (
            {
                "spec_input_hidden_states": model_input.spec_input_hidden_states,
            }
            if model_input.spec_input_hidden_states is not None
            else {}
        )

        # Run model
        hidden_states = model_executable(
            input_ids=model_input.input_tokens,
            positions=model_input.input_positions,
            kv_caches=kv_caches,
            attn_metadata=model_input.attn_metadata,
            intermediate_tensors=intermediate_tensors,
            **multi_modal_kwargs,
            **spec_model_kwargs,
        )

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states, model_input.sampling_metadata)

        # Compute the log probabilities.
        logprobs = torch.log_softmax(logits, dim=-1)
        top = torch.topk(logprobs, self.model.config.topk, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p
        # Records infos for the root node
        token_prob_list.append(scores.unsqueeze(1))
        token_id_list.append(topk_index.unsqueeze(1))
        parents_token_id_list.append(
            torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        )

        for depth_idx in range(self.model.config.depth):
            # Preapre inputs for step
            if depth_idx == 0:
                q_len_per_batch = 1
                spec_model_kwargs["spec_input_hidden_states"] = (
                    hidden_states[None]
                    .unsqueeze(2)
                    .repeat(1, 1, self.model.config.topk, 1)
                    .view(1, -1, hidden_states.shape[-1])
                    .squeeze(0)
                )

                # NOTE: Star codes
                tree_mask = tree_mask_init.repeat(batch_size, 1, 1, 1)

                next_input_tokens = topk_index.view(-1)
            else:
                q_len_per_batch = self.model.config.topk
                out_ids = topk_cs_index // self.model.config.topk
                spec_model_kwargs["spec_input_hidden_states"] = torch.gather(
                    hidden_states.view(batch_size, self.model.config.topk, -1),
                    1,
                    out_ids.unsqueeze(-1).expand(-1, -1, self.model.config.hidden_size),
                ).view(-1, self.model.config.hidden_size)

                # NOTE: Star codes
                tree_mask = torch.stack(
                    [
                        torch.cat(
                            (tree_mask[i, :, out_ids[i]], tree_mask_init[0]), dim=2
                        )
                        for i in range(batch_size)
                    ]
                )

                next_input_tokens = torch.gather(
                    topk_index.view(batch_size, -1), 1, topk_cs_index
                ).view(-1)

            # FIXME(chenzhengda) Create a new model_input or modify inplace ?
            next_input_positions = (
                model_input.input_positions.unsqueeze(1)
                + position_ids_init.expand(model_input.input_positions.shape[0], -1)
                + depth_idx
                + 1
            ).view(-1)

            # Update attn_metadata
            attn_metadata = model_input.attn_metadata
            next_slot_mapping_list = []
            for batch_idx in range(batch_size):
                context_len = (
                    model_input.attn_metadata.context_lens_tensor[batch_idx]
                    + q_len_per_batch
                )
                for token_idx in range(
                    context_len, context_len + self.model.config.topk
                ):
                    block_number = model_input.attn_metadata.block_tables[batch_idx][
                        token_idx // self.cache_config.block_size
                    ]
                    block_offset = token_idx % self.cache_config.block_size
                    slot = block_number * self.cache_config.block_size + block_offset
                    next_slot_mapping_list.append(slot)
            next_slot_mapping_tensor = torch.tensor(
                next_slot_mapping_list,
                dtype=torch.long,
                device=self.device,
            )
            attn_metadata.slot_mapping = next_slot_mapping_tensor

            attn_metadata.num_decode_tokens = len(next_input_tokens)
            attn_metadata.seq_lens = [
                x + self.model.config.topk for x in attn_metadata.seq_lens
            ]
            attn_metadata.seq_lens_tensor += self.model.config.topk
            attn_metadata.max_decode_seq_len = torch.max(
                attn_metadata.seq_lens_tensor
            ).item()
            attn_metadata.context_lens_tensor += q_len_per_batch

            mask = torch.zeros(
                (
                    len(attn_metadata.seq_lens_tensor),
                    max(attn_metadata.seq_lens_tensor),
                ),
                device=self.device,
                dtype=torch.int32,
            )
            for i, length in enumerate(attn_metadata.seq_lens_tensor):
                mask[i, -length:] = 1
                # triton kernel needs right padding for custom mask
                # mask[i, :length] = 1

            attention_mask = _prepare_decoder_attention_mask(
                mask,
                (batch_size, self.model.config.topk),
                next_input_tokens,
                max(attn_metadata.context_lens_tensor),
                tree_mask,
            )
            attn_metadata.decode_metadata.custom_masks = attention_mask

            # Update sampling_metadata
            model_input.sampling_metadata.selected_token_indices = torch.arange(
                batch_size * self.model.config.topk,
                dtype=torch.long,
                device=self.device,
            )

            # Create new input
            new_model_input = self._model_input_cls(
                input_tokens=next_input_tokens,
                input_positions=next_input_positions,
                attn_metadata=attn_metadata,
                seq_lens=attn_metadata.seq_lens,
                query_lens=model_input.query_lens,
                lora_mapping=model_input.lora_mapping,
                lora_requests=model_input.lora_requests,
                spec_input_hidden_states=spec_model_kwargs["spec_input_hidden_states"],
                multi_modal_kwargs=model_input.multi_modal_kwargs,
                sampling_metadata=model_input.sampling_metadata,
                is_prompt=False,
            )

            # Run model
            hidden_states = model_executable(
                input_ids=new_model_input.input_tokens,
                positions=new_model_input.input_positions,
                kv_caches=kv_caches,
                attn_metadata=new_model_input.attn_metadata,
                intermediate_tensors=intermediate_tensors,
                **multi_modal_kwargs,
                **spec_model_kwargs,
            )

            # Compute the logits.
            logits = self.model.compute_logits(
                hidden_states, model_input.sampling_metadata
            )

            # Compute the log probabilities.
            logprobs = torch.log_softmax(logits, dim=-1)
            logprobs = logprobs.view(batch_size, self.model.config.topk, -1)

            # Update Parents infos
            bias1 = self.model.config.topk if depth_idx > 0 else 0
            bias2 = max(0, depth_idx - 1)
            bias = 1 + self.model.config.topk**2 * bias2 + bias1
            parents = topk_cs_index + bias
            parents_token_id_list.append(parents)

            # Update Tree infos
            top = torch.topk(logprobs, self.model.config.topk, dim=-1)
            topk_index, topk_p = top.indices, top.values
            cu_scores = topk_p + scores.unsqueeze(-1)
            topk_cs = torch.topk(
                cu_scores.view(batch_size, -1), self.model.config.topk, dim=-1
            )
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p
            token_id_list.append(topk_index)
            token_prob_list.append(cu_scores)

        token_id_list_tensor = torch.cat(token_id_list, dim=1).view(batch_size, -1)
        token_prob_list_tensor = torch.cat(token_prob_list, dim=1).view(batch_size, -1)
        top_scores = torch.topk(
            token_prob_list_tensor, self.model.config.total_tokens, dim=-1
        )
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = torch.stack(
            [token_id_list_tensor[i][index] for i, index in enumerate(top_scores_index)]
        )
        draft_tokens = torch.cat(
            (model_input.input_tokens.view(batch_size, 1), draft_tokens), dim=-1
        )
        # draft_tokens = draft_tokens.squeeze(1)
        draft_parents = torch.cat(parents_token_id_list, dim=-1)
        draft_parents = torch.stack(
            [
                draft_parents[i][index // self.model.config.topk].long()
                for i, index in enumerate(top_scores_index)
            ]
        )

        mask_index = torch.searchsorted(
            top_scores_index, draft_parents - 1, right=False
        )
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        tree_mask = (
            torch.eye(self.model.config.total_tokens + 1, device=self.device)
            .bool()
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        tree_mask[:, :, 0] = True
        for i in range(batch_size):
            for j in range(self.model.config.total_tokens):
                tree_mask[i][j + 1] = (
                    tree_mask[i][j + 1]
                    .clone()
                    .add_(tree_mask[i][mask_index_list[i][j]])
                )
        tree_position_ids = torch.sum(tree_mask, dim=-1) - 1
        tree_mask = tree_mask.float()[:, None, None]
        # draft_tokens = draft_tokens[:,None]

        noleaf_indexs = [torch.unique(mask_idx).tolist() for mask_idx in mask_index]
        noleaf_nums = [len(index) - 1 for index in noleaf_indexs]
        leaf_nums = [
            (self.model.config.total_tokens - noleaf_num) for noleaf_num in noleaf_nums
        ]

        # Notes : padding to [total_tokens, total_tokens]
        retrieve_indices_list = [
            (
                torch.zeros(
                    self.model.config.total_tokens + 1,
                    self.model.config.total_tokens + 1,
                    device=self.device,
                    dtype=torch.long,
                )
                - 1
            )
            for i in range(len(leaf_nums))
        ]
        for i in range(batch_size):
            rid = 0
            for j in range(self.model.config.total_tokens + 1):
                if j not in noleaf_indexs[i]:
                    cid = j
                    depth = tree_position_ids[i][j].item()
                    for k in reversed(range(depth + 1)):
                        retrieve_indices_list[i][rid][k] = cid
                        cid = mask_index_list[i][cid - 1]
                    rid += 1

        retrieve_indices_tensor = torch.stack(retrieve_indices_list)

        padding = (torch.zeros(batch_size, 1, dtype=torch.long) - 1).to(
            draft_tokens.device
        )
        paded_draft_tokens = torch.cat((draft_tokens, padding), dim=1)

        # draft_tokens’s shape: [batch_size, total_tokens]
        # tree_position_ids‘s shape: [batch_size, total_tokens]
        # tree_mask‘s shape: [batch_size, 1, 1, total_tokens, total_tokens]
        # retrieve_indices_tensor‘s shape: [batch_size, num_proposal_seqs_per_batch, num_proposal_lens_per_batch]
        outputs: List[SamplerOutput] = []

        for idx in range(batch_size):
            candidates = paded_draft_tokens[
                idx, retrieve_indices_tensor[idx]
            ]  # [max_sample_num, max_sample_len]
            outputs.append(
                SamplerOutput(
                    outputs=None,
                    tree_token_ids=draft_tokens,  # [batch, max_sample_len]
                    tree_positions=tree_position_ids,  # [batch, max_sample_len]
                    tree_attention_masks=tree_mask,  # [batch, max_sample_len]
                    tree_retrieve_indices=retrieve_indices_tensor,  # [max_sample_num, max_sample_len]
                    sampled_token_ids=candidates,  # [max_sample_num, max_sample_len]
                    logprobs=torch.zeros(
                        (
                            self.model.config.total_tokens + 1,
                            self.model.config.total_tokens + 1,
                            self.vocab_size,
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    sampled_token_probs=torch.zeros(
                        (
                            self.model.config.total_tokens + 1,
                            self.model.config.total_tokens + 1,
                            self.vocab_size,
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                )
            )

        return outputs


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


# [3, 11]  (3, 2) 9 [3, 1, 2, 2]
def _prepare_decoder_attention_mask(
    attention_mask, input_shape, inputs_embeds, past_key_values_length, tree_mask
):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            # inputs_embeds.dtype,
            torch.float32,  # [MODIFIED] force to cast to float32
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(
            attention_mask, torch.float32, tgt_len=input_shape[-1]
        ).to(inputs_embeds.device)
        # [3, 1, 2, 11]
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )

    # [MODIFIED] add tree mask
    if tree_mask is not None:
        tree_len = tree_mask.size(-1)
        bs = combined_attention_mask.size(0)
        # combined_attention_mask[:, :, -tree_len:, -tree_len:][tree_mask.repeat(bs, 1, 1, 1) == 0] = torch.finfo(torch.float32).min
        combined_attention_mask[:, :, -tree_len:, -tree_len:][tree_mask == 0] = (
            torch.finfo(torch.float32).min
        )

    return combined_attention_mask
