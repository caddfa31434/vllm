import asyncio
import json
from pathlib import Path
from time import perf_counter
from typing import Optional, Set

import click
import pandas as pd
from tokenizers import Tokenizer
from tqdm import tqdm

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter

GENETATE_KEY = "__generate__"


def get_text(sample: dict, tokenizer: Tokenizer) -> str:
    if 'prompt' in sample and isinstance(sample['prompt'], list):
        conv = []
        for m in sample['prompt']:
            conv += [{'role': 'user', 'content': m}, {'role': 'assistant', 'content': GENETATE_KEY}]
    elif 'call_json' in sample:
        conv = sample['call_json']['messages']
        
        if conv[0]['role'] == "system":
            conv[1]['content'] = f"{conv[0]['content'].strip()}\n\n{conv[1]['content'].strip()}"
            conv = conv[1:]
        
        if conv[-1]['role'] != 'assistant':
            conv.append({'role': 'assistant', 'content': GENETATE_KEY})
    
    return tokenizer.apply_chat_template(conv, tokenize=False).split("<s>", 1)[-1]


def load_dataset(dataset_path: Path, tokenizer: Tokenizer) -> list[str]:
    try:
        with open(dataset_path, "r") as rf:
            data = [json.loads(line) for line in rf]
    except:
        with open(dataset_path, "r") as rf:
            data = json.load(rf)
        
        if isinstance(data, dict):
            data = list(data.values())
            
    return [get_text(sample, tokenizer=tokenizer) for sample in data]


async def run(llm: AsyncLLMEngine, dataset_path: Path, concurrency: Optional[int]):    
    dataset = load_dataset(dataset_path, await llm.get_tokenizer())
    if concurrency is None:
        concurrency = len(dataset)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    
    request_counter = Counter()
    sem = asyncio.Semaphore(concurrency)
    
    async def generate(prompt: str) -> str:
        results_generator = llm.generate(
            request_id=str(next(request_counter)),
            prompt=prompt,
            sampling_params=sampling_params
        )

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        return {
            "text": final_output.outputs[0].text,
            "prompt_tokens": len(final_output.prompt_token_ids),
            "completion_tokens": len(final_output.outputs[0].token_ids),
            "steps": getattr(request_output.outputs[0], "num_steps", len(final_output.outputs[0].token_ids)),
        }

    async def run_for_sample(sample: str) -> str:
        await sem.acquire()
        messages = sample.split(GENETATE_KEY)
        
        output = {
            "text": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "steps": 0,
        }
        
        for message in messages[:-1]:
            output["text"] += message
            response = await generate(prompt=output["text"])
            for k in output:
                output[k] += response[k]
                
        sem.release()
        output['text'] += messages[-1]
                
        return output
    
    async def run_for_dataset():
        stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "steps": 0
        }
            
        num_samples = 0
        start_time = perf_counter()
        tasks: Set[asyncio.Task] = set()
        for sample in dataset:
            tasks.add(asyncio.create_task(run_for_sample(sample)))
            
        for task in (pbar:=tqdm(asyncio.as_completed(tasks), total=len(tasks))):
            result = await task
            
            num_samples += 1
            for k in stats:
                stats[k] += result[k]
                
            pbar.set_postfix({"tokens/step": f"{stats['completion_tokens'] / stats['steps']:.2f}"})
            
        total_time = perf_counter() - start_time
        stats.update({
            'total_time': total_time,
            'input_tokens/s': stats['prompt_tokens'] / total_time,
            'output_tokens/step': stats['completion_tokens'] / stats['steps'],
            'output_tokens/s': stats['completion_tokens'] / total_time,
        })
        return stats
    
    sampling_params.temperature = 0.0
    greedy_stats = {k:f"{v:.2f}" for k,v in (await run_for_dataset()).items()}
    
    sampling_params.temperature = 1.0
    random_stats = {k:f"{v:.2f}" for k,v in (await run_for_dataset()).items()}
    
    return greedy_stats, random_stats


async def amain(dataset_path: Path, model: Path, speculative_model: Path, speculative_length: int, speculative_inputs: str):
    try:
        llm = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=model,
            speculative_model=speculative_model,
            num_speculative_tokens=speculative_length,
            use_v2_block_manager=speculative_model is not None,
            extra_inputs_for_draft_model=speculative_inputs,
            disable_log_requests=True,
            disable_log_stats=True,
        ))
    except:
        llm = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=model,
            disable_log_requests=True,
            disable_log_stats=True,
        ))
        
    greedy_stats = {}
    random_stats = {}

    concurrency = 2**6
    while concurrency > 0:
        greedy_stats[concurrency], random_stats[concurrency] = await run(
            llm=llm,
            dataset_path=dataset_path,
            concurrency=concurrency,
        )
        
        print(pd.DataFrame(greedy_stats).T.round(2))
        print(pd.DataFrame(random_stats).T.round(2))
        
        concurrency //= 2
        
    greedy_stats = pd.DataFrame(greedy_stats).T.round(2)
    random_stats = pd.DataFrame(random_stats).T.round(2)
    
    print(greedy_stats.to_csv())
    print(random_stats.to_csv())
    
    print("greedy," + ",".join(greedy_stats['output_tokens/s'].to_list()))
    print("random," + ",".join(random_stats['output_tokens/s'].to_list()))
    
    
@click.command()
@click.argument("dataset_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("model", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-s", "--speculative-model", default=None, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-k", "--speculative-length", default=5, type=int)
@click.option("-si", "--speculative-inputs", default="hidden_states", type=str)
def main(dataset_path: Path, model: Path, speculative_model: Path, speculative_length: int, speculative_inputs: str):
    asyncio.run(amain(
        dataset_path=dataset_path,
        model=model,
        speculative_model=speculative_model,
        speculative_length=speculative_length,
        speculative_inputs=speculative_inputs
    ))


if __name__ == "__main__":
    main()