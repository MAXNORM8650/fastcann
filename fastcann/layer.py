import torch
import torch.nn as nn
from torch import Tensor

from . import _C


class CanonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, mix: Tensor):
        valid_dtypes = {torch.float32, torch.float16, torch.bfloat16}
        if x.dtype not in valid_dtypes:
            raise TypeError(f"x must be float32, float16, or bfloat16, got {x.dtype}")
        if mix.dtype not in valid_dtypes:
            raise TypeError(f"mix must be float32, float16, or bfloat16, got {mix.dtype}")
        if x.dtype != mix.dtype:
            raise TypeError(f"x and mix must have the same dtype, got {x.dtype} and {mix.dtype}")
        x = x.contiguous()
        mix = mix.contiguous()
        y = _C.forward(x, mix)
        ctx.save_for_backward(x, mix)
        return y

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, mix = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_x, grad_mix = _C.backward(grad_out, x, mix)
        return grad_x, grad_mix


class CanonLayerCUDA(nn.Module):
    """GPU implementation of CanonLayer for float32, float16, and bfloat16 CUDA/ROCm tensors."""

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        self.dim = dim
        self.kernel_size = kernel_size
        self.mix = nn.Parameter(torch.empty(kernel_size, dim, dtype=torch.float32))
        nn.init.normal_(self.mix, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B, L, D], got {tuple(x.shape)}")
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {x.shape[-1]}")
        return CanonFn.apply(x, self.mix)
