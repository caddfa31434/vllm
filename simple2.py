from vllm import LLM
from vllm.sampling_params import SamplingParams
import torch
import os

os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# llm = LLM(
#     model="/data/jieni/workspace/code/inference-toolboxes/hf_experimanets/vicuna-7b-v1.3/",
#     speculative_model="/data/jieni/workspace/code/inference-toolboxes/vllm_master/flipkart-incubator/vllm/medusa-vicuna-7b-v1.3",
#     num_speculative_tokens=3,
#     num_lookahead_slots=3,
#     use_v2_block_manager=True,
#     tensor_parallel_size=1,
#     gpu_memory_utilization=0.5,
#     enforce_eager=True,
# )

llm = LLM(
    model="/data/jieni/workspace/code/inference-toolboxes/hf_experimanets/Llama-2-7b-chat-hf",
    # "/data/jieni/workspace/models/meta-llama/Llama-2-70b-chat-hf",
    speculative_model="/data/jieni/workspace/code/inference-toolboxes/hf_experimanets/EAGLE-llama2-chat-7B",
    # "/data/jieni/workspace/models/yuhuili/EAGLE-llama2-chat-70B",
    num_speculative_candidates=15,
    num_speculative_tokens=5,
    num_lookahead_slots=75,
    use_v2_block_manager=True,
    tensor_parallel_size=1,
    # tensor_parallel_size=1,
    gpu_memory_utilization=0.5,
    enforce_eager=True,
)

# llm = LLM(
#     model="/data/jieni/workspace/code/inference-toolboxes/hf_experimanets/Llama-2-7b-chat-hf",
#     speculative_model="/data/jieni/workspace/models/llama-68m/",
#     num_speculative_tokens=3,
#     num_lookahead_slots=6,
#     use_v2_block_manager=True,
#     tensor_parallbash run.sh --svc_config_path ./svc.yaml  --process_timeout 50000 --enable_http_service 1 --port 9223 --health_port 9224
# el_size=1,
#     gpu_memory_utilization=0.5,
#     enforce_eager=True
# )

sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
prompts = [
    # "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is Paris. It is located in the",
    "hi" * 2040,
    # "The future of AI is",
]

# import pandas as pd

# df = pd.read_csv("/data/jieni/redbi/llm3_sql_prompt.csv")
# column_list = df["prompt"].tolist()
# prompts = column_list[1]

# with torch.profiler.profile(
#     activities=[
#         torch.profiler.ProfilerActivity.CPU,
#         torch.profiler.ProfilerActivity.CUDA,
#     ],
#     with_stack=True,
#     # on_trace_ready=torch.profiler.tensorboard_trace_handler(str("./")),
# ) as p:
request_outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
import time
start_time = time.perf_counter()
request_outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
end_time = time.perf_counter()
latency = end_time - start_time
print(latency)

# print(p.key_averages())
# p.export_chrome_trace("trace.json")

# request_outputs = llm.generate(prompts=prompts,
#                                sampling_params=sampling_params)

print(f"{request_outputs[0].outputs[0].text=}")
# print(f"{request_outputs[1].outputs[0].text=}")
# print(f"{request_outputs[2].outputs[0].text=}")
# print(f"{request_outputs[3].outputs[0].text=}")

# from transformers import AutoTokenizer
# tok = AutoTokenizer.from_pretrained("/data/jieni/workspace/code/inference-toolboxes/hf_experimanets/Llama-2-7b-chat-hf")
# print(tok.encode(prompts[0]))
# print(tok.encode(request_outputs[0].outputs[0].text))
