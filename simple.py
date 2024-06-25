from vllm import LLM
from vllm.sampling_params import SamplingParams
import torch
import os

os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
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
    num_speculative_candidates=1,
    num_speculative_tokens=5,
    num_lookahead_slots=5,
    use_v2_block_manager=True,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.5,
    enforce_eager=True,
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=500)

prompts = [
    # "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is Paris. It is located in the",
    # "The capital of France is",
    "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\nCompose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.   Title: Discovering the Aloha Spirit of Hawaii: A Cultural Journey\n\nIntroduction:\nHawaii, the Aloha State, is a tropical paradise known for its stunning beaches, lush greenery, and rich cultural heritage. Recently, I had the opportunity to embark on a journey to this beautiful island chain, and I must say, it was an unforgettable experience. In this blog post, I will share some of the cultural experiences and must-see attractions that made my trip to Hawaii truly special.\n\nCultural Experiences:\n\n1. Visit a Luau: A traditional Hawaiian luau is a must-do experience when visiting the islands. These festive gatherings are filled with music, dance, and delicious Hawaiian cuisine. I had the chance to attend a luau on the Big Island, where I enjoyed a sumptuous feast of kalua pig, poke, and haupia, while watching traditional hula dancers perform.\n2. Learn the Hula: The hula is an integral part of Hawaiian culture, and learning this traditional dance is a great way to immerse yourself in the local culture. I took a hula lesson with a local instructor on Oahu, who taught me the basic steps and rhythms of this beautiful dance.\n3. Explore a Hawaiian Temple: Hawaii is home to many beautiful temples, each with its own unique history and architecture. I visited the Kona Temple on the Big Island, which is dedicated to the Hawaiian goddess of love and fertility, Laka. The temple's tranquil gardens and peaceful atmosphere made for a serene and spiritual experience.\n\nMust-See Attractions:\n\n1. Waikiki Beach: No trip to Hawaii is complete without a visit to Waikiki Beach, one of the most famous beaches in the world. This stretch of sand is known for its white sands, crystal-clear waters, and stunning sunsets. I spent a leisurely afternoon swimming, sunbathing, and people-watching at this iconic beach.\n[/INST]",
    # "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\nAnalyze the following customer reviews from different sources for three different smartphones - the latest iPhone, Samsung Galaxy, and Google Pixel - and provide an overall rating for each phone on a scale of 1 to 10. Consider the following complex and contradictory reviews:\n- TechRadar's review of the latest iPhone: The new iPhone is a stunning triumph of engineering that sets a new bar for smartphone performance and camera quality. However, the incremental design and high price mean it lacks the 'wow' factor of previous iPhones. Still, its power and intelligence are unrivaled.\n- CNET's review of the latest Samsung Galaxy: The Samsung Galaxy phone has plenty of high points, including an amazing screen, fast performance, solid battery life and an impressive array of camera options. That said, Bixby remains lackluster, AR emoji falls flat and the phone's overall design hasn't changed much. The new Galaxy is an amazing phone overall, but it has a few nagging weaknesses that keep it from achieving true greatness.\n- The Verge's review of the latest Google Pixel: Google's Pixel packs cutting-edge specs, innovative AI-powered software, and a killer camera into a sleek design. However, the phone has lackluster battery life, lacks expandable storage, and its performance stutters at times, especially considering its high price tag. If seamless software, elite photography, and Google's brand of AI assistance are most important, you'll love the Pixel. But the overall experience isn't as well-rounded as some competitors. Return the answer as a JSON object with the overall ratings for each phone out of 10, to one decimal place.", "Can you change the ratings from numbers to letters? Capital letters MUST be used when writing the names of phones.",
    # "hi" * 2040,
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
# import time
# start_time = time.perf_counter()
# prompts = [
#     # "Hello, my name is",
#     # "The president of the United States is",
#     # "The capital of France is Paris. It is located in the",
#     "hi" * 2040,
#     # "The future of AI is",
# ]
request_outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
# end_time = time.perf_counter()
# latency = end_time - start_time
# print(latency)



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