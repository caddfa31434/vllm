import random
from unittest.mock import MagicMock

import pytest
from transformers import AutoTokenizer
import torch

from vllm.model_executor.utils import set_random_seed
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.top1_proposer import TopKProposer
from vllm.worker.worker import Worker

from .utils import (assert_logprobs_dict_allclose, create_batch,
                    create_seq_group_metadata_from_prompts, create_worker,
                    patch_execute_model_with_seeds, zero_kv_cache)

import os

os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'


@pytest.mark.parametrize("batch_size", list(range(1, 2)))
def test_prepare_prompt(batch_size):
    seed = 100
    model_name = "/data/jieni/workspace/code/inference-toolboxes/hf_experimanets/Llama-2-7b-chat-hf"

    block_size = 16
    num_gpu_blocks = 8192 // block_size

    worker = create_worker(
        Worker,
        model_name,
        block_size,
        num_gpu_blocks,
        seed,
    )

    # candidates
    # [[[  518,  1024, 29962,   322,   590,  1024],
    #     [  518,  1024, 29962,   322,   590,  5121],
    #     [  518,  1024, 29962,   322,   338,    -1],
    #     [  518,  1024, 29962,   322,   306,    -1],
    #     [  518,  1024, 29962,    13,    -1,    -1],
    #     [  518,  1024, 29962, 29871,    -1,    -1],
    #     [  518,  1024,   322,   590,    -1,    -1],
    #     [  518,  1024,   322,   306,    -1,    -1],
    #     [  518,  1024, 29889,   322,    -1,    -1],
    #     [  518,  1024, 29889,   590,    -1,    -1],
    #     [  518,    13,   322,   590,    -1,    -1],
    #     [  518,    13,   590,    -1,    -1,    -1],
    #     [  518, 29871,   590,    -1,    -1,    -1],
    #     [  518, 29871,    13,    -1,    -1,    -1],
    #     [  518,   322,   590,    -1,    -1,    -1]],

    #     [  278,  6673,   616,  6673,   310,   278],
    #     [  278,  6673,   616,  6673,   310,  6673],
    #     [  278,  6673,   616,  6673,   616,    -1],
    #     [  278,  6673,   616,  6673,  1058,    -1],
    #     [  278,  6673,   616,  5874,    -1,    -1],
    #     [  278,  6673,   616,  2343,    -1,    -1],
    #     [  278,  6673,   310,   278,    -1,    -1],
    #     [  278,  6673,   310,  6673,    -1,    -1],
    #     [  278,  6673,   322,   278,    -1,    -1],
    #     [  278,  6673,   322,  6673,    -1,    -1],
    #     [  278,  2343,   310,   278,    -1,    -1],
    #     [  278,  2343,   278,    -1,    -1,    -1],
    #     [  278,  9939,  6673,    -1,    -1,    -1],
    #     [  278,  9939,  5874,    -1,    -1,    -1],
    #     [  278, 11822,   310,    -1,    -1,    -1],

    # tree_candidates
    # [ 518,  1024,    13, 29871,   322, 29962,   322, 29889,   322,   590,
    #    590,    13,   590,   322,    13, 29871,   590,   306,   322,   590,
    #    590,   590,   338,   306,  1024,  5121],
    # [  278,  6673,  2343,  9939, 11822,   616,   310,   322,   310,   278,
    #   6673,  5874,   310,  6673,  5874,  2343,   278,  6673,   278,  6673,
    #    278,   310,   616,  1058,   278,  6673],

    # tree_candidates positions
    # [6,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9, 9,  9,  9, 10, 10, 10, 11, 11],
    # [8,  9,  9,  9,  9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 13, 13]]

    prompt_texts = [
        "Hello, my name is",
        "The president of the United States is",
    ]

    tok = AutoTokenizer.from_pretrained(model_name)

    prompts = [tok.encode(prompt_texts[0]), tok.encode(prompt_texts[1])]

    tree_candidates = [[
        518, 1024, 13, 29871, 322, 29962, 322, 29889, 322, 590, 590, 13, 590,
        322, 13, 29871, 590, 306, 322, 590, 590, 590, 338, 306, 1024, 5121
    ],
                       [
                           278, 6673, 2343, 9939, 11822, 616, 310, 322, 310,
                           278, 6673, 5874, 310, 6673, 5874, 2343, 278, 6673,
                           278, 6673, 278, 310, 616, 1058, 278, 6673
                       ]]

    num_candidates = 26
    final_prompt_lens = [len(prompt) + num_candidates for prompt in prompts]

    zero_kv_cache(worker.cache_engine)
    set_random_seed(seed)
    single_step_seq_group = create_seq_group_metadata_from_prompts(
        prompts,
        num_gpu_blocks,
        block_size,
        final_prompt_lens=final_prompt_lens)

    # hard codes for mock inputs
    single_step_seq_group[0].is_prompt = False
    single_step_seq_group[1].is_prompt = False
    single_step_seq_group[0].seq_data[0].candidate_token_ids = tree_candidates[
        0]
    single_step_seq_group[1].seq_data[1].candidate_token_ids = tree_candidates[
        1]

    expected_output = worker.execute_model(
        execute_model_req=ExecuteModelRequest(
            seq_group_metadata_list=single_step_seq_group))
