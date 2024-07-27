from itertools import chain, count
from typing import Iterator, List, Tuple

import torch

from vllm.sequence import (
    ExecuteModelRequest,
    SamplerOutput,
    SequenceData,
    SequenceGroupMetadata,
    SequenceGroupState,
    get_all_seq_ids,
)
from vllm.spec_decode.interfaces import (
    SpeculativeProposals,
    SpeculativeScorer,
    SpeculativeScores,
)
from vllm.spec_decode.batch_expansion import BatchExpansionTop1Scorer
from vllm.spec_decode.util import (
    nvtx_range,
    sampler_output_to_torch,
    split_batch_by_proposal_len,
)
from vllm.worker.worker_base import WorkerBase

SeqId = int
TargetSeqId = int
TokenId = int


class EagleScorer(BatchExpansionTop1Scorer):
    @nvtx_range("EagleScorer.score_proposals")
    def score_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        proposals: SpeculativeProposals,
    ) -> SpeculativeScores:
        # TODO(cade) perform this on GPU to remove blocking call.
        proposal_lens_list = proposals.proposal_lens.tolist()
        proposal_tree_token_ids_list = proposals.tree_token_ids.tolist()

        for batch_id, (seq_group_metadata, proposal_token_ids) in enumerate(
            zip(
                execute_model_req.seq_group_metadata_list,
                proposal_tree_token_ids_list,
            )
        ):
            seq_id, seq_data = next(iter(seq_group_metadata.seq_data.items()))
            if proposal_token_ids:
                seq_data.update_num_computed_tokens(
                    (seq_data.get_len() - 1) - seq_data.get_num_computed_tokens()
                )
                # for token in proposal_token_ids:
                # skip first token
                for token in proposal_token_ids[1:]:
                    seq_data._output_token_ids.append(token)
                    seq_data._cached_all_token_ids.append(token)
                # use the prompt mode for multi-query sampling
                # seq_group_metadata.token_chunk_size += len(proposal_token_ids)
                seq_group_metadata.token_chunk_size += len(proposal_token_ids[1:])
                seq_group_metadata.tree_positions = proposals.tree_positions[batch_id]
                seq_group_metadata.tree_attention_masks = (
                    proposals.tree_attention_masks[batch_id]
                )
        # assert(torch.stack((execute_model_req.seq_group_metadata_list[0].tree_positions, execute_model_req.seq_group_metadata_list[1].tree_positions, execute_model_req.seq_group_metadata_list[2].tree_positions)).is_contiguous())
        target_sampler_output = self._scorer_worker.execute_model(
            execute_model_req=execute_model_req.clone(
                seq_group_metadata_list=execute_model_req.seq_group_metadata_list
            )
        )

        for seq_group_metadata, proposal_token_ids in zip(
            execute_model_req.seq_group_metadata_list,
            proposal_tree_token_ids_list,
        ):
            seq_id, seq_data = next(iter(seq_group_metadata.seq_data.items()))
            if proposal_token_ids:
                for token in proposal_token_ids[1:]:
                    seq_data._output_token_ids.pop()
                    seq_data._cached_all_token_ids.pop()
                seq_group_metadata.token_chunk_size -= len(proposal_token_ids[1:])

        padding = (
            torch.zeros(len(proposal_tree_token_ids_list), 1, dtype=torch.long) - 1
        ).to(self._scorer_worker.model_runner.device)
        paded_tokens = torch.cat(
            (target_sampler_output[0].sampled_token_ids, padding), dim=1
        )
        retrieved_tokens = torch.stack(
            [
                paded_tokens[i, proposals.tree_retrieve_indices[i]]
                for i in range(len(proposals.tree_retrieve_indices))
            ]
        )

        retrieved_probs = torch.stack(
            [
                target_sampler_output[0].sampled_token_probs[
                    i, proposals.tree_retrieve_indices[i], :
                ]
                for i in range(len(proposals.tree_retrieve_indices))
            ]
        )

        paded_slot_mapping = torch.cat(
            (
                target_sampler_output[0].spec_slot_mapping.view(
                    len(execute_model_req.seq_group_metadata_list), -1
                ),
                padding,
            ),
            dim=1,
        )

        retrieved_slot_mapping = torch.stack(
            [
                paded_slot_mapping[i, proposals.tree_retrieve_indices[i]]
                for i in range(len(proposals.tree_retrieve_indices))
            ]
        )

        return SpeculativeScores(
            probs=retrieved_probs,
            token_ids=retrieved_tokens,
            flatten_slot_mapping=paded_slot_mapping,
            retrieved_slot_mapping=retrieved_slot_mapping,
            logprobs=None,
            hidden_states=target_sampler_output[0].hidden_states.view(
                len(proposals.tree_retrieve_indices),
                -1,
                target_sampler_output[0].hidden_states.shape[-1],
            ),
        )
