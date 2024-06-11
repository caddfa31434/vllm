import triton.language as tl
import triton
import torch


@triton.jit
def copy_kernel(x_ptr, z_ptr, n, bs: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * bs + tl.arange(0, bs)[:, None]
    mask = offs < n
    x = tl.load(x_ptr + offs, mask)
    tl.store(z_ptr + offs, x, mask)


x = torch.arange(0, 10).unsqueeze(1).cuda().contiguous().float()
z = torch.zeros_like(x)
copy_kernel[(3,)](x, z, n=10, bs=4)

print(x)
print(z)
