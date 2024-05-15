from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
from vllm.transformers_utils.configs.medusa import MedusaConfig


class ResidualBlock(nn.Module):

    def __init__(self, hidden_size: int, num_layers: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_layers)
        ])
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + self.act(layer(x))
        return x


class Medusa(nn.Module):

    def __init__(self, config: MedusaConfig, **_) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size=self.config.hidden_size,
                          num_layers=self.config.num_hidden_layers)
            for _ in range(self.config.num_heads)
        ])
        self.unpadded_vocab_size = config.vocab_size

        self.lm_heads = nn.ModuleList([
            ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
            ) for _ in range(self.config.num_heads)
        ])

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)

    def forward(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        return [block(hidden_states) for block in self.blocks]

    def compute_logits(
            self, hidden_states: list[torch.Tensor],
            sampling_metadata: SamplingMetadata) -> list[torch.Tensor]:
        return [
            self.logits_processor(lm_head.weight, hs, sampling_metadata)
            for hs, lm_head in zip(hidden_states, self.lm_heads)
        ]

    def sample(
        self,
        logits: List[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> List[SamplerOutput]:
        logits = torch.stack(logits, dim=0).float()
        logprobs = torch.log_softmax(logits, dim=-1)
        token_ids = logits.argmax(-1)  # support only top-1 for now
        probs = torch.softmax(logits, dim=-1)

        token_id_list = []
        token_prob_list = []
        token_logprob_list = []

        for idx, seq_group in enumerate(sampling_metadata.seq_groups):
            token_id_list.append(token_ids[:, seq_group.sample_indices])
            token_prob_list.append(probs[:, seq_group.sample_indices])
            token_logprob_list.append(logprobs[:, seq_group.sample_indices])

        outputs: List[Optional[SamplerOutput]] = []
        for idx in range(len(sampling_metadata.seq_groups)):
            outputs.append(
                SamplerOutput(
                    outputs=None,
                    sampled_token_probs=token_prob_list[idx].squeeze(),
                    logprobs=token_logprob_list[idx].squeeze(),
                    sampled_token_ids=token_id_list[idx].squeeze(),
                ))

        return outputs

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            name = name.replace("medusa_heads.", "")
            # Skip loading extra heads
            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
