import os
from typing import Union

from transformers import PretrainedConfig

from vllm.transformers_utils.configs.inputs_spec import (ExtraModelInputsSpec,
                                                         ModelInputSpec)


class EagleConfig(PretrainedConfig):
    model_type = "eagle"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        max_paths: int = 64,
        topk: int = 10,
        **kwargs,
    ):
        if "architectures" not in kwargs or "LlamaForCausalLM" in kwargs[
                "architectures"]:
            kwargs["architectures"] = ["EagleModel"]

        self.extra_inputs_spec: ExtraModelInputsSpec = {
            "hidden_states": ModelInputSpec(shape=(hidden_size, ))
        }

        # self.tree_choices = [[0], [1], [2], [3], [0, 0], [0, 1], [0, 2],
        #                      [1, 0], [1, 1], [2, 0], [2, 1], [3, 0], [0, 0, 0],
        #                      [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1],
        #                      [0, 2, 0], [0, 2, 1], [1, 0, 0], [0, 0, 0, 0],
        #                      [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 0, 0],
        #                      [0, 0, 0, 0, 1]]
        # self.tree_choices = [[0], [0, 0], [0, 1], [0, 2], [0, 3]] # # 1st depth we choose top 1 and 2nd depth we choose top 4
        self.tree_choices = [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0],
                             [0, 0, 0, 0, 0]]  # # 5st depth we choose top 1
        self.topk = topk

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling,
                          dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}")
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in [
                "linear", "dynamic"
        ]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(
                rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(
                f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}"
            )

    def set_num_lookahead_tokens(self, num_lookahead_tokens: int):
        self.num_heads = num_lookahead_tokens
