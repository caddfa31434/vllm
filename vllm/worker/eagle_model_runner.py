from typing import Dict, List, Optional, Set, Tuple

import torch

from vllm.attention import AttentionMetadata
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         VisionLanguageConfig)
from vllm.distributed import broadcast_tensor_dict
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.pooling_params import PoolingParams
from vllm.sequence import (ExtraTensorData, MultiModalData, SamplerOutput,
                           SequenceData, SequenceGroupMetadata)
from vllm.worker.model_runner import ModelRunner, _move_extra_tensor_data_to_seq_outputs

logger = init_logger(__name__)


class EagleSpeculativeModelRunner(ModelRunner):

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
        vision_language_config: Optional[VisionLanguageConfig] = None,
    ):
        super().__init__(model_config,
                         parallel_config,
                         scheduler_config,
                         device_config,
                         cache_config,
                         load_config,
                         lora_config=lora_config,
                         kv_cache_dtype=kv_cache_dtype,
                         is_driver_worker=is_driver_worker,
                         vision_language_config=vision_language_config)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[torch.Tensor],
        extra_inputs: Optional[ExtraTensorData] = None,
        extra_outputs: Optional[Set[str]] = None,
    ) -> Optional[SamplerOutput]:
        (input_tokens, input_positions, attn_metadata, sampling_metadata,
         lora_requests, lora_mapping, multi_modal_input,
         extra_inputs) = self.prepare_input_tensors(seq_group_metadata_list,
                                                    extra_inputs)

        if self.lora_config:
            self.set_active_loras(lora_requests, lora_mapping)

        # Currently cuda graph is only supported by the decode phase.
        prefill_meta = attn_metadata.prefill_metadata
        decode_meta = attn_metadata.decode_metadata
        if prefill_meta is None and decode_meta.use_cuda_graph:
            graph_batch_size = input_tokens.shape[0]
            model_executable = self.graph_runners[graph_batch_size]
        else:
            model_executable = self.model
        execute_model_kwargs = {
            "input_ids": input_tokens,
            "positions": input_positions,
            "kv_caches": kv_caches,
            "attn_metadata": attn_metadata,
        }
        if self.vision_language_config:
            execute_model_kwargs.update({"image_input": multi_modal_input})

        if extra_inputs:
            execute_model_kwargs.update(extra_inputs.asdict())

        execute_model_kwargs = {
            k: v
            for k, v in execute_model_kwargs.items() if k in self.model_inputs
        }

        extra_tensor_data = ExtraTensorData()

        hidden_states = model_executable(**execute_model_kwargs)

        if extra_outputs and "hidden_states" in extra_outputs:
            extra_tensor_data["hidden_states"] = hidden_states

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states, sampling_metadata)

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return None

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        if extra_outputs:
            sampled_extra_tensor_data = extra_tensor_data.index_select(
                0, sampling_metadata.selected_token_indices)

            if prefill_meta is not None:
                for k in extra_tensor_data:
                    extra_tensor_data[k] = extra_tensor_data[k].roll(shifts=1,
                                                                     dims=0)
                    extra_tensor_data[k].masked_fill_(
                        (input_positions == 0).unsqueeze(-1), 0)

                if output is not None:
                    _move_extra_tensor_data_to_seq_outputs(
                        output, sampled_extra_tensor_data, sampling_metadata)

                    output.extra_tensor_data = extra_tensor_data
            else:
                if output is not None:
                    output.extra_tensor_data = sampled_extra_tensor_data

        return output
