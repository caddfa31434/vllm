from vllm import LLM, SamplingParams
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
prompts = [
    "The future of AI is",
    "The president of the United States is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    # model="/data/jieni/workspace/models/lmsys/vicuna-7b-v1.3",
    model="/data/jieni/workspace/code/inference-toolboxes/hf_experimanets/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,
    speculative_model="/data/jieni/workspace/models/abhigoyal/vllm-medusa-vicuna-7b-v1.3",
    num_speculative_tokens=5,
    use_v2_block_manager=True,
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
