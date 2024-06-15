from transformers import AutoTokenizer, AutoModelForCausalLM

import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['https_proxy'] = '10.7.4.2:3128'

model = AutoModelForCausalLM.from_pretrained("daryl149/llama-2-7b-chat-hf")
tokenizer = AutoTokenizer.from_pretrained("daryl149/llama-2-7b-chat-hf")

prompt = "What's your name?"
inputs = tokenizer(prompt, return_tensors="pt")

generate_ids = model.generate(inputs.input_ids, max_length=30)
print(
    tokenizer.batch_decode(generate_ids,
                           skip_special_tokens=True,
                           clean_up_tokenization_spaces=False)[0])
