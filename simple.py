from vllm import LLM
from vllm.sampling_params import SamplingParams

import os
os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'

llm = LLM(
    model="/data/jieni/workspace/code/inference-toolboxes/hf_experimanets/vicuna-7b-v1.3/",
    # speculative_model="/data/jieni/workspace/code/inference-toolboxes/vllm_master/flipkart-incubator/vllm/medusa-vicuna-7b-v1.3",
    # speculative_model="FasterDecoding/medusa-v1.0-vicuna-7b-v1.5",
    # num_speculative_tokens=3,
    use_v2_block_manager=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5,
    enforce_eager=True,
)

# llm = LLM(
#     model="/data/jieni/workspace/code/inference-toolboxes/hf_experimanets/Llama-2-7b-chat-hf",
#     speculative_model="/data/jieni/workspace/code/inference-toolboxes/hf_experimanets/EAGLE-llama2-chat-7B",
#     num_speculative_tokens=2,
#     use_v2_block_manager=True,
#     extra_inputs_for_draft_model="hidden_states",
#     tensor_parallel_size=1,
#     gpu_memory_utilization=0.5,
#     enforce_eager=True,
# )

# llm = LLM(
#     model="/data/jieni/workspace/code/inference-toolboxes/hf_experimanets/Llama-2-7b-chat-hf",
#     speculative_model="/data/jieni/workspace/models/llama-68m/",
#     num_speculative_tokens=3,
#     use_v2_block_manager=True,
#     tensor_parallel_size=1,
#     gpu_memory_utilization=0.5,
#     enforce_eager=True
# )

sampling_params = SamplingParams(temperature=0.0, max_tokens=500)
prompts = [
    # "Hello, my name is",
    # "The president of the United States is",
    "The capital of France is",
    # "The future of AI is",
]
request_outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
print(f"{request_outputs[0].outputs[0].text=}")
# print(f"{request_outputs[1].outputs[0].text=}")
# print(f"{request_outputs[2].outputs[0].text=}")
# print(f"{request_outputs[3].outputs[0].text=}")

# from transformers import AutoTokenizer
# tok = AutoTokenizer.from_pretrained("/data/jieni/workspace/code/inference-toolboxes/hf_experimanets/Llama-2-7b-chat-hf")
# print(tok.encode(prompts[0]))
# print(tok.encode(request_outputs[0].outputs[0].text))

