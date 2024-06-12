import copy
from math import e
import weakref
from typing import List, Tuple

import torch

from vllm.sequence import (ExecuteModelRequest, SamplerOutput,
                           SequenceGroupMetadata)
from vllm.spec_decode.interfaces import SpeculativeProposals
from vllm.spec_decode.top1_proposer import TopKProposer
from vllm.spec_decode.multi_step_worker import MultiStepWorker


class EagleWorker(MultiStepWorker):
    """The EagleWorker is equivalent to a Worker except that it allows
    multiple forward passes in a single call, assuming the scheduler has
    allocated enough space to store the additional KV. This reduces overhead
    by invoking the scheduler less.

    The EagleWorker does not support cache swap operations, or beam search.
    Cache swap operations do not require large modifications. On the other hand,
    beam search requires memory allocations during sequence forks and thus
    requires more thought for EagleWorker support.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Lazy initialization list.
        self._proposer: TopKProposer

    @torch.inference_mode()
    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_num: int,
        sample_len: int,
    ) -> Tuple[List[List[SamplerOutput]], bool]:
        """Run the model forward pass sample_len times. Returns the list of
        sampler output, one per model forward pass, along with indicator of
        whether torch tensor in sampler output need to be transposed in latter
        sampler_output_to_torch logic.

        For multi step worker, this indicator shall be True.
        """
        self._raise_if_unsupported(execute_model_req)

        # Assert enough KV space for sample_len tokens per sequence.
        self._assert_enough_kv_space(execute_model_req.seq_group_metadata_list,
                                     sample_num * sample_len)

        # Run model sample_num * sample_len times.
        model_outputs_topK = []

        for sample_idx in range(sample_num):
            # if sample_idx == sample_num - 1:
            if 1:
                # Shallow copy input data so modifications (such as appending tokens)
                # do not cause side-effects.
                copied_seq_group_metadata_list = self._shallow_copy_inputs(
                    execute_model_req.seq_group_metadata_list, sample_idx * sample_len)
                copied_execute_model_req = execute_model_req.clone(
                    copied_seq_group_metadata_list)
            # else:
            #     copied_seq_group_metadata_list = self._fake_all_zeros_inputs(
            #         execute_model_req.seq_group_metadata_list, sample_idx * sample_len)
            #     copied_execute_model_req = execute_model_req.clone(
            #         copied_seq_group_metadata_list)  
            model_outputs = []
            for _ in range(sample_len):
                copied_execute_model_req.extra_outputs = "hidden_states"
                model_output = super().execute_model(
                    execute_model_req=copied_execute_model_req)
                model_output = model_output[0]

                self._append_new_tokens(model_output,
                                        copied_seq_group_metadata_list)
                model_outputs.append(model_output)
                copied_execute_model_req.extra_inputs = model_output.raw_extra_tensors

            model_outputs_topK.append(model_outputs)
        # [15, 5, 4]
        import numpy as np
        shape = (len(model_outputs_topK), len(model_outputs_topK[0]), len(model_outputs_topK[0][0]))

        tensor = np.empty(shape, dtype=object)

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    pass
                    # model_outputs_topK[i][j][k].samples[0].output_token = 0

        for i in range(shape[0] - 1):
            for j in range(shape[1]):
                pass
                # model_outputs_topK[i][j].sampled_token_ids.zero_()
                    # tensor[i, j, k] = model_outputs_topK[i][j][k].samples[0].output_token
                    # model_outputs_topK[i][j][k].samples[0].output_token = 0
        # print(tensor)

        return model_outputs_topK, True

    @staticmethod
    def _fake_all_zeros_inputs(
        seq_group_metadata_list: List[SequenceGroupMetadata],
        num_lookahead_slot_mapping_dirty_offset: int
    ) -> List[SequenceGroupMetadata]:
        """Copy input data structures to remove side-effects when input data
        structures are shared with other modules.

        Helpful when the vLLM scheduler runs in the same process as the worker.
        The alternative is deep-copying (or other form of deep copy); this has
        performance downsides.
        """

        # Shallow-copy the list of SequenceGroupMetadata. This allows us to
        # append tokens and change is_prompt without external side-effects.
        new_seq_group_metadata_list = []

        for old_seq_group_metadata in seq_group_metadata_list:
            # We must shallow-copy seq_group_metadata as is_prompt could change.
            seq_group_metadata = copy.copy(old_seq_group_metadata)
            new_seq_group_metadata_list.append(seq_group_metadata)

            # We must shallow-copy seq_data as we will append token ids
            new_seq_data = {}
            for seq_id, old_seq_data in seq_group_metadata.seq_data.items():
                new_seq_data[seq_id] = copy.copy(old_seq_data)
                new_seq_data[
                    seq_id].prompt_token_ids = [0] * len(old_seq_data.prompt_token_ids[:])
                new_seq_data[
                    seq_id].output_token_ids = [0] * len(old_seq_data.output_token_ids[:])

            seq_group_metadata.seq_data = new_seq_data
            seq_group_metadata.num_lookahead_slot_mapping_dirty_offset = num_lookahead_slot_mapping_dirty_offset
        return new_seq_group_metadata_list
