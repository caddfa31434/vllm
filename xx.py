from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    # "Hello, my name is",
    # "The president of the United States is",
    "The capital of France is",
    # "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(
    model="/data/jieni/workspace/models/Qwen/Qwen2-7B-Instruct/",
    distributed_executor_backend="ray",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.5,
)
# llm = LLM(model="/data/jieni/workspace/models/Qwen/Qwen2-7B-Instruct-AWQ/")
# llm = LLM(model="/data/jieni/workspace/models/Qwen/Qwen2-57B-A14B-Instruct/")
# llm = LLM(model="/data/jieni/workspace/models/Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4/")
# llm = LLM(model="/data/jieni/workspace/models/Qwen/Qwen2-72B-Instruct/", tensor_parallel_size=4)
# llm = LLM(model="/data/jieni/workspace/models/Qwen/Qwen2-72B-Instruct-AWQ/", tensor_parallel_size=4)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
