import weakref
from typing import List, Tuple

import torch

from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.spec_decode.interfaces import SpeculativeProposals
from vllm.spec_decode.top1_proposer import Top1Proposer
from vllm.worker.worker import Worker


class MultiHeadWorker(Worker):
    """The MultiHeadWorker is equivalent to a Worker except that it allows
    multiple forward passes in a single call, assuming the scheduler has
    allocated enough space to store the additional KV. This reduces overhead
    by invoking the scheduler less.

    The MultiHeadWorker does not support cache swap operations, or beam search.
    Cache swap operations do not require large modifications. On the other hand,
    beam search requires memory allocations during sequence forks and thus
    requires more thought for MultiHeadWorker support.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Lazy initialization list.
        self._proposer: Top1Proposer

    def init_device(self):
        super().init_device()

        self._proposer = Top1Proposer(
            weakref.proxy(self),
            self.device,
            self.vocab_size,
            max_proposal_len=self.max_model_len,
        )

    def set_include_gpu_probs_tensor(self):
        # Need include_gpu_probs_tensor for multi_step_worker
        if hasattr(self.model_runner.model, "sampler"):
            self.model_runner.model.sampler.include_gpu_probs_tensor = True

    @torch.inference_mode()
    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
    ) -> Tuple[List[SamplerOutput], bool]:
        model_outputs, _ = super().execute_model(execute_model_req=execute_model_req)[0]
        return model_outputs, False

    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> SpeculativeProposals:
        """Produce speculations given an input batch of sequences. The number of
        speculative tokens per sequence is determined by max_proposal_len.
        """

        return self._proposer.get_proposals(execute_model_req)