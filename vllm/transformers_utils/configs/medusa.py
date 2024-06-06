import os
from typing import Union

from transformers import PretrainedConfig

from vllm.transformers_utils.configs.inputs_spec import (ExtraModelInputsSpec,
                                                         ModelInputSpec)


class MedusaConfig(PretrainedConfig):
    model_type = "medusa"

    def __init__(self,
                 hidden_size: int = 4096,
                 vocab_size: int = 32000,
                 num_heads: int = 5,
                 num_hidden_layers: int = 1,
                 max_paths: int = 64,
                 topk: int = 10,
                 **kwargs):

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.max_paths = max_paths
        self.topk = topk
        self.max_seq_len = int(2**20)
        if "architectures" not in kwargs:
            kwargs["architectures"] = ["MedusaModel"]

        self.extra_inputs_spec: ExtraModelInputsSpec = {
            "hidden_states": ModelInputSpec(shape=(self.hidden_size, ))
        }

        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ) -> "MedusaConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs)
        for k in list(config_dict.keys()):
            if 'num' in k:
                if 'heads' in k:
                    config_dict["num_heads"] = config_dict.pop(k)
                elif 'layers' in k:
                    config_dict["num_hidden_layers"] = config_dict.pop(k)
        return cls.from_dict(config_dict, **kwargs)

    @property
    def num_attention_heads(self):
        return 0

    def set_num_lookahead_tokens(self, num_lookahead_tokens: int):
        self.num_heads = num_lookahead_tokens
