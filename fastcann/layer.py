import torch
import torch.nn as nn
from torch import Tensor

from . import _C


def _check_inputs(x: Tensor, mix: Tensor) -> None:
    valid_dtypes = {torch.float32, torch.float16, torch.bfloat16}
    if x.dtype not in valid_dtypes:
        raise TypeError(f"x must be float32, float16, or bfloat16, got {x.dtype}")
    if mix.dtype not in valid_dtypes:
        raise TypeError(f"mix must be float32, float16, or bfloat16, got {mix.dtype}")
    if x.dtype != mix.dtype:
        raise TypeError(f"x and mix must have the same dtype, got {x.dtype} and {mix.dtype}")


@torch.library.custom_op("fastcann::forward", mutates_args=(), device_types="cuda")
def canon_forward_op(x: Tensor, mix: Tensor) -> Tensor:
    _check_inputs(x, mix)
    return _C.forward(x.contiguous(), mix.contiguous())


@torch.library.custom_op("fastcann::backward", mutates_args=(), device_types="cuda")
def canon_backward_op(grad_out: Tensor, x: Tensor, mix: Tensor) -> tuple[Tensor, Tensor]:
    _check_inputs(x, mix)
    if grad_out.dtype != x.dtype:
        raise TypeError(f"grad_out and x must have the same dtype, got {grad_out.dtype} and {x.dtype}")
    return _C.backward(grad_out.contiguous(), x.contiguous(), mix.contiguous())


@torch.library.register_fake("fastcann::forward")
def _(x: Tensor, mix: Tensor) -> Tensor:
    return torch.empty_like(x)


@torch.library.register_fake("fastcann::backward")
def _(grad_out: Tensor, x: Tensor, mix: Tensor) -> tuple[Tensor, Tensor]:
    return torch.empty_like(x), torch.empty_like(mix)


def _setup_context(ctx, inputs, output) -> None:
    x, mix = inputs
    ctx.save_for_backward(x, mix)


def _backward(ctx, grad_out: Tensor) -> tuple[Tensor, Tensor]:
    x, mix = ctx.saved_tensors
    return canon_backward_op(grad_out, x, mix)


torch.library.register_autograd("fastcann::forward", _backward, setup_context=_setup_context)


class CanonFn:
    @staticmethod
    def apply(x: Tensor, mix: Tensor) -> Tensor:
        return canon_forward_op(x, mix)


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
        return canon_forward_op(x, self.mix)
