# from vllm import LLM
# from vllm.sampling_params import SamplingParams

# import os
# os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'

# llm = LLM(
#     model="/data/jieni/workspace/models/llama-68m/",
#     use_v2_block_manager=True,
#     tensor_parallel_size=1,
#     gpu_memory_utilization=0.5,
#     enforce_eager=True,
# )

# sampling_params = SamplingParams(max_tokens=500, best_of=2)
# prompts = [
#     # "Hello, my name is",
#     # "The president of the United States is",
#     "The capital of France is",
#     # "The future of AI is",
# ]
# request_outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
# print(f"{request_outputs[0].outputs[0].text=}")

import torch
def create_tree_attention_mask(context_len, prompt_len, tree_width,
                               num_kv_head, dtype):
    prompt_mask = torch.zeros((num_kv_head, tree_width, prompt_len),
                              dtype=dtype)

    print(f"{prompt_mask=}")    

    none_mask_value = torch.arange(context_len - prompt_len).repeat(
        tree_width, 1) - torch.arange(tree_width)[:, None]
    none_mask_value = none_mask_value % tree_width
    none_mask_value = none_mask_value == 0

    min_value = torch.finfo(dtype).min

    generate_mask = torch.full(none_mask_value.shape, min_value, dtype=dtype)
    generate_mask[none_mask_value] = 0
    generate_mask = generate_mask.unsqueeze(0).repeat(num_kv_head, 1, 1)
    print(f"{generate_mask=}")   
    return torch.concat([prompt_mask, generate_mask], dim=2)

print(create_tree_attention_mask(6, 2, 4, 1, torch.float16))