from typing import List, Optional, Set, Tuple

import torch

from vllm.sequence import (ExecuteModelRequest, SamplerOutput,
                           SequenceGroupMetadata)
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeProposer)
from vllm.spec_decode.top1_proposer import Top1Proposer
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.util import sampler_output_to_torch

# Eagle Tree Proposer
class EagleProposer(Top1Proposer):
    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> SpeculativeProposals:
        """Get speculative proposals given the input batch.

        Sequences which would exceed the max model length are skipped during
        speculation.
        """
        proposal_len = execute_model_req.num_speculative_tokens
        seq_group_metadata_list = execute_model_req.seq_group_metadata_list

        # Split speculative- and non-speculative- sequences.
        (
            proposal_lens,
            nonzero_proposal_len_seqs,
            nonzero_proposal_len_indices,
        ) = self._split_by_proposal_len(seq_group_metadata_list, proposal_len)

        if nonzero_proposal_len_seqs:
            # Speculate tokens using the draft worker for the speculative
            # sequences.
            # If sampler_transposed is true, then maybe_sampler_output's
            # token_ids is like [batch] format in proposal_len size list,
            # while if it is false, the format would be [proposal_len]
            # in batch size list
            hidden_states = execute_model_req.previous_hidden_states
            if hidden_states is not None:
                hidden_states.prune(nonzero_proposal_len_seqs)
            nonzero_execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=nonzero_proposal_len_seqs,
                num_speculative_tokens=proposal_len,
                previous_hidden_states=hidden_states,
            )
            maybe_sampler_output, transposed = self._worker.sampler_output(
                execute_model_req=nonzero_execute_model_req,
                sample_len=proposal_len,
                seq_ids_with_bonus_token_in_last_step=\
                    seq_ids_with_bonus_token_in_last_step,
            )
            (
                proposal_lens,
                maybe_sampler_output,
                nonzero_proposal_len_indices,
            ) = self._remove_no_proposal_seqs(proposal_lens,
                                              maybe_sampler_output,
                                              nonzero_proposal_len_indices,
                                              transposed)
        else:
            # If no sequences can be speculated, set sampler output to None.
            maybe_sampler_output = None
            transposed = False

        # Combine speculative- and non-speculative sequences into the same
        # representation.
        proposal_tokens, proposal_probs, proposal_lens = self._merge_outputs(
            batch_size=len(seq_group_metadata_list),
            proposal_len=proposal_len,
            maybe_sampler_output=maybe_sampler_output,
            proposal_lens=proposal_lens,
            nonzero_proposal_len_indices=nonzero_proposal_len_indices,
            sampler_transposed=transposed,
        )

        proposals = SpeculativeProposals(
            proposal_token_ids=proposal_tokens,
            proposal_probs=proposal_probs,
            proposal_lens=proposal_lens,
            tree_token_ids=maybe_sampler_output[0].tree_token_ids,
            tree_positions=maybe_sampler_output[0].tree_positions,
            tree_attention_masks=maybe_sampler_output[0].tree_attention_masks,
            tree_retrieve_indices=maybe_sampler_output[0].tree_retrieve_indices,
            no_proposals=maybe_sampler_output is None
        )

        return proposals

    def sampler_output_to_torch_for_eagle_tree_proposal(
        self, sampler_output_list: List[SamplerOutput], sampler_transposed: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Utility function which converts a list of SamplerOutput to tensors.

            sampler_transposed here is used as the indicator for whether
            we need do additional tensor transpose logic here.

            Returns:
                sampled_token_ids: torch.Tensor
                    shape: [batch_size, len(sampler_output_list)]

                sampled_token_probs: torch.Tensor
                    shape: [batch_size, len(sampler_output_list), vocab_size]
            """

        # shape: [batch_size, num_sampler, num_sampler_output, vocab_size]
        sampled_token_probs = torch.stack(
            [
                sampler_output.sampled_token_probs
                for sampler_output in sampler_output_list
            ],
            dim=0,
        )

        # Transposing if needed
        if sampler_transposed:
            raise ValueError("Unconsidered sampler_transposed.")

        # shape: [batch_size, num_sampler, num_sampler_output, vocab_size]
        sampled_token_logprobs = torch.stack(
            [sampler_output.logprobs for sampler_output in sampler_output_list],
            dim=0,
        )

        # Transposing if needed
        if sampler_transposed:
            raise ValueError("Unconsidered sampler_transposed.")

        # shape: [batch_size, num_sampler, num_sampler_output]
        sampled_token_ids = torch.stack(
            [
                sampler_output.sampled_token_ids
                for sampler_output in sampler_output_list
            ],
            dim=0,
        )

        # Transposing if needed
        if sampler_transposed:
            raise ValueError("Unconsidered sampler_transposed.")

        return sampled_token_ids, sampled_token_probs, sampled_token_logprobs

    def _merge_outputs(
        self,
        batch_size: int,
        proposal_len: int,
        maybe_sampler_output: Optional[List[SamplerOutput]],
        proposal_lens: List[int],
        nonzero_proposal_len_indices: List[int],
        sampler_transposed: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """After speculations are produced, merge the speculation results with
        the skipped sequences.
        """
        if maybe_sampler_output is None:
            # If no speculative tokens, the sampler output will be None.
            # In this case we return empty proposals.
            proposal_tokens = torch.tensor(-1,
                                           dtype=torch.long,
                                           device=self._device).expand(
                                               batch_size, proposal_len)
            proposal_probs = torch.tensor(0,
                                          dtype=torch.float32,
                                          device=self._device).expand(
                                              batch_size, proposal_len,
                                              self._vocab_size)
            proposal_lens_tensor = torch.tensor(0,
                                                dtype=torch.long,
                                                device=self._device).expand(
                                                    len(proposal_lens))
            return proposal_tokens, proposal_probs, proposal_lens_tensor

        sampler_output = maybe_sampler_output
        proposal_tokens, proposal_probs, _ = self.sampler_output_to_torch_for_eagle_tree_proposal(
            sampler_output, sampler_transposed)

        # Now, reformat the output GPU tensors such that each sequence has
        # a proposal. the proposal can be empty, e.g. [-1, -1, -1]

        entire_proposal_tokens = proposal_tokens.new_full(
            size=(batch_size, *proposal_tokens.shape[1:]),
            fill_value=-1,
        )
        entire_proposal_tokens[nonzero_proposal_len_indices] = proposal_tokens
        entire_proposal_probs = proposal_probs.new_zeros(
            batch_size,
            *proposal_probs.shape[1:],
        )
        entire_proposal_probs[nonzero_proposal_len_indices] = proposal_probs

        proposal_tokens, proposal_probs = (
            entire_proposal_tokens,
            entire_proposal_probs,
        )

        proposal_lens_tensor = torch.zeros(batch_size,
                                           dtype=torch.long,
                                           device=self._device)
        proposal_lens_tensor[nonzero_proposal_len_indices] = proposal_len

        return proposal_tokens, proposal_probs, proposal_lens_tensor
