import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from vllm import LLM, SamplingParams

# os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

prompts = [
    # "The future of AI is",
    # "The president of the United States is",
    "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\nCompose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions. Title: Discovering the Aloha Spirit of Hawaii: A Cultural Journey\n\nIntroduction:\nHawaii, the Aloha State, is a tropical paradise known for its stunning beaches, lush greenery, and rich cultural heritage. Recently, I had the opportunity to embark on a journey to this beautiful island chain, and I must say, it was an unforgettable experience. In this blog post, I will share some of the cultural experiences and must-see attractions that made my trip to Hawaii truly special.\n\nCultural Experiences:\n\n1. Visit a Luau: A traditional Hawaiian luau is a must-do experience when visiting the islands. These festive gatherings are filled with music, dance, and delicious Hawaiian cuisine. I had the chance to attend a luau on the Big Island, where I enjoyed a sumptuous feast of kalua pig, poke, and haupia, while watching traditional hula dancers perform.\n2. Learn the Hula: Hawaii is famous for its hula dancing, and I couldn't resist learning a few moves during my trip. I took a hula lesson with a local instructor on Oahu, who taught me the basics of this ancient dance form. It was a fun and rewarding experience, and I even got to perform in front of my fellow travelers!\n3. Explore a Hawaiian Temple: Hawaii is home to many beautiful temples, each with its own unique history and cultural significance. I visited the Kona Temple on the Big Island, which is dedicated to the Hawaiian goddess of the earth, Pele. The temple's tranquil gardens and stunning architecture were a peaceful oasis in the midst of a lush tropical landscape.\n\nMust-See Attractions:\n [/INST] "
    # "Sure, here are STAR stories for each of Amazon's Leadership Principles based on your professional experiences:\n\nTitle: Customer-Obsession\nSituation: As a Visual Management Trainee at Enedis, I received customer feedback on difficulty in navigating the management tool used by the GI2R division.\nTask: My task was to implement changes to make the tool more user-friendly based on the feedback received.\nAction: I conducted user testing, collected and analyzed feedback, and made necessary changes to the tool.\nResult: As a result, the user satisfaction and ease of use for the tool increased, and the customers were happier with the tool.\nLearnings: Through this experience, I learned to prioritize customer feedback and incorporate it into the product design.\n\nTitle: Ownership\nSituation: During my internship as a Process Engineer Trainee at Nystar, I noticed an inefficient process leading to high energy consumption and increased costs.\nTask: My task was to develop a plan to optimize the energy efficiency of the process.\nAction: I conducted data analysis, identified inefficiencies, and implemented changes to the process.\nResult: The implementation of these changes led to a reduction in energy consumption and costs.\nLearnings: This experience taught me to take ownership of identifying and solving problems in the workplace.\n\nTitle: Invent and Simplify\nSituation: As a Multipurpose Worker at Carbonex, I noticed missing sell-by dates of stock in the store, leading to waste.\nTask: My task was to define a process to identify stock near sell-by and to implement a solution to the problem.\nAction: I collected data, surveyed potential fixes, and implemented the changes necessary.\nResult: The changes implemented led to a 50% decrease in waste and were standardized across all stores.\nLearnings: This experience taught me to ensure that I challenge old ways of working when necessary and to think outside the box to simplify processes.\n\nTitle: Are Right, A Lot\nSituation: As a member of a Badminton Club, I noticed poor team performance in competitions.\nTask: My task was to develop a new strategy to improve the team's performance.\nAction: I conducted an analysis of past performances, identified weaknesses, and implemented a new strategy.\nResult: The new strategy led to increased team performance and success in competitions.\nLearnings: This experience taught me to rely on data and analysis to make informed decisions.\n\nTitle: Learn and Be Curious\nSituation: As a Webmaster, I noticed a lack of online presence for a local charity.\nTask: My task was to create and manage a website to increase visibility and outreach for the charity.\nAction: I conducted research on website design, created and launched a website for the charity.\nResult: The website increased the online visibility and outreach for the charity.\nLearnings: This experience taught me new skills in website design and management through self-directed learning.\n\nTitle: Hire and Develop the Best\nSituation: At Enedis, I noticed a lack of a formal mentorship program in the company.\nTask: My task was to develop and implement a mentorship program for the company.\nAction: I conducted research on mentorship programs, designed and launched the program.\nResult: The implementation of the mentorship program led to increased employee satisfaction and retention.\nLearnings: Through this experience, I learned to recognize and develop talent in others.\n\nTitle: Insist on the Highest Standards\nSituation: As a student of Quality Culture, I noticed a lack of focus on quality in the company culture.\nTask: My task was to develop a plan to promote a culture of quality within the company.\nAction: I conducted research on quality culture best practices, designed and launched the program.\nResult: The program increased the focus on quality in company culture and processes.\nLearnings: This experience taught me to prioritize and uphold high standards",
]

sampling_params = SamplingParams(temperature=0, max_tokens=500)

llm = LLM(
    # model="/data/jieni/workspace/models/Qwen2-57B-A14B-Instruct/",
    model=
    "/data/jieni/workspace/code/inference-toolboxes/hf_experimanets/Llama-2-7b-chat-hf",
    # tensor_parallel_size=4,
    speculative_model=
    "/data/jieni/workspace/code/inference-toolboxes/hf_experimanets/EAGLE-llama2-chat-7B",
    # speculative_model="/mnt/nj-larc/dataset/llm_ckpt/ckpt/Qwen2-moe_RedBI_v3.2/state_100/jieni_merged_ckpt",
    speculative_draft_tensor_parallel_size=1,
    # distributed_executor_backend="ray",
    # ray_workers_use_nsight=True,
    num_speculative_candidates=1,
    num_speculative_tokens=4,
    num_lookahead_slots=4,
    use_v2_block_manager=True,
    enforce_eager=False,
)
outputs = llm.generate(prompts, sampling_params)
outputs = llm.generate(prompts, sampling_params)
print(f"===============================================================")
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text!r}")
    # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# outputs = llm.generate(prompts, sampling_params)

# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# outputs = llm.generate(prompts, sampling_params)

# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
