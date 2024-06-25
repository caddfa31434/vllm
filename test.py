# save it as `test.py` , and run it with `NCCL_DEBUG=TRACE torchrun --nproc-per-node=8 test.py`
# adjust `--nproc-per-node` to the number of GPUs you want to use.
import torch
import torch.distributed as dist
import os

# os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
dist.init_process_group(backend="nccl")
data = torch.FloatTensor([
    1,
] * 128).to(f"cuda:{dist.get_rank()}")
dist.all_reduce(data, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
value = data.mean().item()
assert value == dist.get_world_size()
