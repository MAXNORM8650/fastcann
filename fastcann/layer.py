import torch
import torch.nn as nn
from torch import Tensor

from . import _C


class CanonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, mix: Tensor):
        if x.dtype != torch.float32:
            raise TypeError(f"x must be float32, got {x.dtype}")
        if mix.dtype != torch.float32:
            raise TypeError(f"mix must be float32, got {mix.dtype}")
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
    """CUDA implementation of CanonLayer for kernel_size=4 and float32 CUDA tensors."""

    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        if kernel_size != 4:
            raise ValueError("This CUDA starter currently supports kernel_size=4 only.")
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
