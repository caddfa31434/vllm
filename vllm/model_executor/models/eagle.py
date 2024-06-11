from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
from vllm.utils import is_hip
from vllm.transformers_utils.configs.eagle import EagleConfig


class EagleMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(input_size=intermediate_size,
                                           output_size=hidden_size,
                                           bias=bias,
                                           quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class EagleAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class EagleDecoderLayer(nn.Module):

    def __init__(
        self,
        config: EagleConfig,
        layer_id: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = EagleAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
        )
        self.mlp = EagleMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
        )
        # TODO: Star Code 1
        self.layer_id = layer_id  
        if self.layer_id != 0:
            self.input_layernorm = RMSNorm(config.hidden_size,
                                        eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        # TODO: Star Code 1
        if residual is None:
            residual = hidden_states
            if self.layer_id != 0:
                hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class EagleModel(nn.Module):

    def __init__(
        self,
        config: EagleConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        # TODO: Star Code 1
        self.layers = nn.ModuleList([
            EagleDecoderLayer(config=config,
                              layer_id=idx,
                              cache_config=cache_config,
                              quant_config=quant_config)
            for idx in range(config.num_hidden_layers)
        ])
        # TODO: Star Code 2
        self.fc=nn.Linear(2*config.hidden_size, config.hidden_size, bias=True)

        # TODO: Star Code 3
        # self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        hidden_input: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        # TODO: Star Code 2 
        # hidden_states = self.fc(torch.cat((hidden_states, hidden_input), dim=-1))
        hidden_states = self.fc(torch.cat((hidden_states, hidden_input.view(-1, hidden_input.shape[-1])), dim=-1))

        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )
        # TODO: Star Code 3
        # hidden_states, _ = self.norm(hidden_states, residual)
        hidden_states = hidden_states + residual
        return hidden_states

def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))

class node:
    def __init__(self,parent=None,value=None,dict_key=None):
        self.parent=parent
        self.value=value
        if parent:
            self.depth=parent.depth+1
            parent.children.append(self)
        else:
            self.depth=0
        self.children=[]
        self.dict_key=dict_key
    def is_leaf(self):
        return len(self.children)==0

    def all_index(self):
        if not self.parent.parent:
            return [self.index]
        else:
            return self.parent.all_index()+[self.index]


class Tree:
    def __init__(self,tree_list):
        sorted_tree_list = sorted(tree_list, key=lambda x: (len(x), x))
        self.root=node()
        self.node_dic={}
        for tree_node in sorted_tree_list:
            cur_value=tree_node[-1]
            if len(tree_node)==1:
                cur_node=node(parent=self.root,value=cur_value,dict_key=tuple(tree_node))
            else:
                cur_parent=self.node_dic[tuple(tree_node[:-1])]
                cur_node = node(parent=cur_parent, value=cur_value,dict_key=tuple(tree_node))
            self.node_dic[tuple(tree_node)] = cur_node
        self.indexnode()

    def max_depth(self):
        return max([item.depth for item in self.node_dic.values()])

    def num_node_wchild(self):
        num_c=0
        for item in self.node_dic.values():
            if not item.is_leaf():
                num_c+=1
        return num_c

    def get_node_wchild(self):
        ns=[]
        for item in self.node_dic.values():
            if not item.is_leaf():
                ns.append(item)
        return ns

    def indexnode(self):
        cur_index=0
        for key in self.node_dic:
            cur_node=self.node_dic[key]
            if not cur_node.is_leaf():
                cur_node.index=cur_index
                cur_index+=1

class Eagle(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "embed_tokens",
        "lm_head"
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(
        self,
        config: EagleConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = EagleModel(config,
                                cache_config,
                                quant_config,
                                lora_config=lora_config)
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = Sampler()

    def generate_tree_buffers(self, tree_choices):
        tree=Tree(tree_choices)
        sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
        tree_len = tree.num_node_wchild()

        max_depth=tree.max_depth()
        nodes_wc=tree.get_node_wchild()

        depth_counts=[0 for _ in range(max_depth-1)]
        for x in nodes_wc:
            depth_counts[x.depth-1]+=1
        depth_counts_sum = [sum(depth_counts[:i + 1]) for i in range(len(depth_counts))]

        tree_attn_mask = torch.eye(tree_len, tree_len)

        for id,x in enumerate(nodes_wc):
            tree_attn_mask[id,x.all_index()]=1

        tree_attn_mask_list0=[tree_attn_mask[:ml,:ml] for ml in depth_counts_sum]
        tree_attn_mask_list=[]
        for id,x in enumerate(tree_attn_mask_list0):
            x=x[-depth_counts[id]:]
            tree_attn_mask_list.append(x)

        tree_indices_list = [torch.zeros(ml, dtype=torch.long) for ml in depth_counts]
        repeat_nums=[[] for _ in depth_counts]
        start = 0
        bias = 0
        for i in range(len(depth_counts)):
            bias = 0
            repeat_j=0
            for j in range(depth_counts[i]):
                cur_node = nodes_wc[start + j]
                cur_parent = cur_node.parent

                if j != 0:
                    if cur_parent != parent:
                        bias += 1
                        parent = cur_parent
                        repeat_nums[i].append(j-repeat_j)
                        repeat_j=j
                else:
                    parent = cur_parent
                tree_indices_list[i][j] = cur_node.value + self.config.topk * (bias)
            repeat_nums[i].append(j - repeat_j+1)
            start += depth_counts[i]

        position_ids = [torch.zeros(ml, dtype=torch.long) for ml in depth_counts]

        tree_buffers = {
            "attn_mask": [i.unsqueeze(0).unsqueeze(0) for i in tree_attn_mask_list],
            "tree_indices": tree_indices_list,
            "position_ids":position_ids,
            "repeat_nums":repeat_nums
        }

        # Move the tensors in the dictionary to the specified device
        tree_buffers = {
            k: [i.clone() for i in v]
            if isinstance(v[0], torch.Tensor)
            else (
                torch.tensor(v)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in tree_buffers.items()
        }
        return tree_buffers

    def repeat_hidden(self, hidden_state, repeat_num):
        new_hidden = []
        hidden_state = hidden_state.unsqueeze(1)
        for id, i in enumerate(repeat_num):
            new_hidden.append(hidden_state[:, id:id + 1].repeat(1, i, 1))
        return torch.cat(new_hidden, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        sampling_metadata: SamplingMetadata,
    ) -> list[torch.Tensor]:
        if attn_metadata.num_prefill_tokens > 0 and attn_metadata.num_decode_tokens == 0 :
            out_hidden = self.model(hidden_states, input_ids, positions, kv_caches,
                                    attn_metadata)
        else:
            out_hidden = self.model(hidden_states, input_ids, positions, kv_caches, attn_metadata)
            logits = self.logits_processor(self.lm_head.weight, out_hidden, sampling_metadata)
            token_ids = torch.topk(logits, self.config.topk, dim=-1).indices
            logprobs = torch.topk(torch.log_softmax(logits, dim=-1), self.config.topk, dim=-1).values
            
            next_input_ids = token_ids[:, self.tree_buffer['tree_indices'][0]].view(-1)
            next_hidden_states = self.repeat_hidden(out_hidden, self.tree_buffer["repeat_nums"][0])
            next_position_ids = positions.unsqueeze(1).expand(positions.shape[0], len(self.tree_buffer['tree_indices'])) + self.tree_buffer["position_ids"][0].to(positions.device) + 1
            
            attn_metadata.slot_mapping += 1 
            attn_metadata.num_decode_tokens = len(next_input_ids)

            attn_metadata.decode_metadata.seq_lens_tensor += len(self.tree_buffer["tree_indices"][0])
            attn_metadata.decode_metadata.context_lens_tensor = attn_metadata.context_lens_tensor + 1
            attn_metadata.decode_metadata.max_decode_seq_len = torch.max(attn_metadata.decode_metadata.seq_lens_tensor).item()
            attn_metadata.decode_metadata.tree_width = len(self.tree_buffer["tree_indices"][0])
            
            next_out_hidden = self.model(next_hidden_states, next_input_ids, next_position_ids, kv_caches, attn_metadata)
            sampling_metadata.selected_token_indices = torch.arange(0, next_out_hidden.shape[0]).to(sampling_metadata.selected_token_indices.device)
            logits = self.logits_processor(self.lm_head.weight, next_out_hidden, sampling_metadata)
            token_ids = torch.topk(logits, self.config.topk, dim=-1).indices
            logprobs = torch.topk(torch.log_softmax(logits, dim=-1), self.config.topk, dim=-1).values
             
            next_input_ids = token_ids.view(4, 40)[:, self.tree_buffer['tree_indices'][0]]
            
        return out_hidden

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        token_ids = torch.topk(logits, self.config.topk, dim=-1).indices
        logprobs = torch.topk(torch.log_softmax(logits, dim=-1), self.config.topk, dim=-1).values
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        
        # Hard code for loadding eagle lmhead
        weight_loader = getattr(params_dict["lm_head.weight"], "weight_loader", default_weight_loader)
        weight_loader(
            params_dict["lm_head.weight"],
            torch.load(
                "/data/jieni/workspace/code/inference-toolboxes/hf_experimanets/Llama-2-7b-chat-hf/pytorch_model-00002-of-00002.bin"
            )["lm_head.weight"],
        )
        
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # TODO: Star Code 4, 注意模型参数名称和LlaMA不一样
                # param = params_dict[name]
                param = params_dict["model." + name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    # TODO: Star Code 5 (eagle draft has fc.bias)
                    if name != 'fc.bias':
                        continue
                # TODO: Star Code 4 (eagle checkpoints start with model.)
                # param = params_dict[name]
                param = params_dict["model." + name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

        self.tree_buffer = self.generate_tree_buffers(self.config.tree_choices)

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
                quantization_param_path, tp_rank, tp_size,
                self.config.num_hidden_layers,
                self.config.__class__.model_type):
            layer_self_attn = self.model.layers[layer_idx].self_attn

            if is_hip():
                # The scaling factor convention we are assuming is
                # quantized_value * scaling_factor ~= true_value
                # which is consistent with the practice of setting
                # scaling_factor = tensor_amax / FPtype_max
                scaling_factor *= 2
            if hasattr(layer_self_attn, "kv_scale"):
                layer_self_attn.attn._kv_scale = scaling_factor
            else:
                raise RuntimeError("Self attention has no KV cache scaling "
                                   "factor attribute!")
