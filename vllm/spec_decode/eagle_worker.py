import copy
import weakref
from typing import Dict, List, Optional, Set, Tuple

import torch

from vllm.sequence import (ExecuteModelRequest, SamplerOutput, SequenceData,
                           SequenceGroupMetadata)
from vllm.spec_decode.draft_model_runner import TP1DraftModelRunner
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeProposer)
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.top1_proposer import Top1Proposer
from vllm.spec_decode.eagle_proposer import EagleProposer
from vllm.logger import init_logger

logger = init_logger(__name__)

class EagleWorker(MultiStepWorker):
    """The MultiStepWorker is equivalent to a Worker except that it allows
    multiple forward passes in a single call, assuming the scheduler has
    allocated enough space to store the additional KV. This reduces overhead
    by invoking the scheduler less.

    The MultiStepWorker does not support cache swap operations, or beam search.
    Cache swap operations do not require large modifications. On the other hand,
    beam search requires memory allocations during sequence forks and thus
    requires more thought for MultiStepWorker support.
    """

    def init_device(self) -> None:
        super().init_device()

        # NotsLazy initialization  self._proposer
        # self._proposer = Top1Proposer(
        #     weakref.proxy(self),  # type: ignore[arg-type]
        #     self.device,
        #     self.vocab_size,
        #     max_proposal_len=self.max_model_len,
        # )

    def load_model(self):
        self.model_runner.load_model()

        if self.model_runner.model.config.topk == 1:
            self._proposer = Top1Proposer(
                weakref.proxy(self),  # type: ignore[arg-type]
                self.device,
                self.vocab_size,
                max_proposal_len=self.max_model_len,
            )
        else:
            self._proposer = EagleProposer(
                weakref.proxy(self),  # type: ignore[arg-type]
                self.device,
                self.vocab_size,
                max_proposal_len=self.max_model_len,
            )

    @torch.inference_mode()
    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> Tuple[List[SamplerOutput], bool]:
        """Run the model forward pass sample_len times. Returns the list of
        sampler output, one per model forward pass, along with indicator of
        whether torch tensor in sampler output need to be transposed in latter
        sampler_output_to_torch logic.

        For multi step worker, this indicator shall be True.
        """
        self._raise_if_unsupported(execute_model_req)
        
        if self.model_runner.model.config.topk == 1:
            logger.info("Using Eagle-Chain Speculative decoding.")
            # Expand the batch for sequences with a bonus token.
            # Perform a forward pass on the expanded batch and filter the
            # response to retain only the original sequences' responses.
            expanded_request, indices_of_seq_with_bonus_tokens =\
                self._expand_execute_model_request(
                    execute_model_req, seq_ids_with_bonus_token_in_last_step)

            # FIXME(chenzhengda): Here has some erros with_bonus_tokens
            for seq_group_metadata in expanded_request.seq_group_metadata_list:
                seq_group_metadata.bind_spec_input_hidden_states(
                    execute_model_req.previous_hidden_states.hidden_states)

            # Run model sample_len times.
            model_outputs: List[SamplerOutput] = []
            if isinstance(
                    self.model_runner, TP1DraftModelRunner
            ) and self.model_runner.supports_gpu_multi_step(expanded_request):
                # Here we run the draft_model_runner with multi-step prepare
                # on the GPU directly
                expanded_request.num_steps = sample_len
                model_outputs = self.execute_model(
                    execute_model_req=expanded_request)
            else:
                # Here we run multi-step directly, with every step prepared
                # on the CPU.
                # TODO: Remove this branch once DraftModelRunner supports TP>1
                # and other restrictions that are part of DraftModelRunner's
                # supports_gpu_multi_step(..)
                for _ in range(sample_len):
                    model_output: List[SamplerOutput] = super().execute_model(
                        execute_model_req=expanded_request)
                    assert (len(model_output) == 1
                            ), "composing multistep workers not supported"
                    model_output = model_output[0]

                    self._append_new_tokens(
                        model_output, expanded_request.seq_group_metadata_list)
                    model_outputs.append(model_output)

                    # Update previous_hidden_states for next step
                    for seq_group_metadata in expanded_request.seq_group_metadata_list:
                        seq_group_metadata.bind_spec_input_hidden_states(
                            model_output.hidden_states)

            filtered_model_outputs = self._filter_model_output(
                model_outputs, indices_of_seq_with_bonus_tokens)
            return filtered_model_outputs, True
        else:
            # logger.info("Using Eagle-Tree Speculative decoding.")
            assert sample_len == (self.model_runner.model.config.total_tokens), (
            "Eagle's sample_len should be equal to total_tokens")
            # Expand the batch for sequences with a bonus token.
            # Perform a forward pass on the expanded batch and filter the
            # response to retain only the original sequences' responses.
            expanded_request, indices_of_seq_with_bonus_tokens =\
                self._expand_execute_model_request(
                    execute_model_req, seq_ids_with_bonus_token_in_last_step)

            # FIXME(chenzhengda): Here has some erros with_bonus_tokens
            for seq_group_metadata in expanded_request.seq_group_metadata_list:
                seq_group_metadata.bind_spec_input_hidden_states(
                    execute_model_req.previous_hidden_states.hidden_states)

            # Here we run the draft_model_runner with multi-step prepare
            # on the GPU directly
            expanded_request.num_steps = sample_len
            model_outputs = self.execute_model(
                execute_model_req=expanded_request)

            return model_outputs, False