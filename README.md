# fastcann

`fastcann` is a PyTorch extension package for a fast GPU implementation of the Canon local mixing layer with `kernel_size=4`.

It currently targets `float32` tensors on GPU and provides a small Python API on top of a compiled C++/CUDA-or-ROCm extension.

## What It Computes

For each batch `b`, token `t`, and channel `c`:

```text
y[b, t, c] = x[b, t, c]
           + mix[0, c] * x[b, t + 2, c]
           + mix[1, c] * x[b, t + 1, c]
           + mix[2, c] * x[b, t, c]
           + mix[3, c] * x[b, t - 1, c]
```

Out-of-range sequence positions are treated as zero.

## Install

From GitHub release/tag:

```bash
pip install --no-build-isolation git+https://github.com/MAXNORM8650/fastcann.git@v0.1.2
```

On this ROCm environment, the install that worked was:

```bash
PYTORCH_ROCM_ARCH=gfx90a uv pip install --python "/vast/users/hisham.cholakkal/documents/multiagent/GPA/codegent-pkg/.venv/bin/python" --no-build-isolation git+https://github.com/MAXNORM8650/fastcann.git@v0.1.4
```

That installed:

- `fastcann==0.1.0`
- built from commit `bb0ac36eb3847c7e15bbc3cd9cc43ffb86ba580c`

Once the package is published to PyPI, the install command will be:

```bash
pip install fastcann
```

For local development from this repo:

```bash
pip install -e . --no-build-isolation
```

Manual in-place extension build:

```bash
python setup.py build_ext --inplace
```

The environment used during development here was:

```bash
uv run --python "/vast/users/hisham.cholakkal/documents/multiagent/GPA/codegent-pkg/.venv/bin/python" python setup.py build_ext --inplace
```

## Usage

Primary import:

```python
from fastcann import CanonLayerCUDA
```

Example:

```python
import torch
from fastcann import CanonLayerCUDA

layer = CanonLayerCUDA(dim=768, kernel_size=4).cuda()
x = torch.randn(8, 2048, 768, device="cuda", dtype=torch.float32)
y = layer(x)
```

This package builds a Torch extension during installation, so the target environment must already have:

- `torch`
- a working CUDA or ROCm toolchain
- compiler/build tools
- `setuptools`

On ROCm, if the build still picks up unsupported default GPU targets such as `gfx950`, set `PYTORCH_ROCM_ARCH` explicitly during install. On this machine, forcing `PYTORCH_ROCM_ARCH=gfx90a` avoided the unsupported target failure.

Legacy import paths are still supported:

```python
from fast_canon_layers import CanonLayerCUDA
from canon import CanonLayerCUDA
```

## Project Layout

- `fastcann/` - primary installable package
- `fast_canon_layers/` - compatibility package
- `canon.py` - compatibility shim
- `canon.cpp` - Python bindings for the extension
- `canon_kernel.cu` - GPU kernels and launcher code
- `setup.py` - package build script
- `simple_test.py` - quick correctness and speed check
- `test_canon_cuda.py` - reference correctness and benchmark script

## Testing

Quick run:

```bash
uv run --python "/vast/users/hisham.cholakkal/documents/multiagent/GPA/codegent-pkg/.venv/bin/python" python -u simple_test.py
```

Reference test:

```bash
python test_canon_cuda.py
```

If you want to install into a specific interpreter with `uv`, this is the pattern used during validation:

```bash
PYTORCH_ROCM_ARCH=gfx90a uv pip install --python "/path/to/python" --no-build-isolation git+https://github.com/MAXNORM8650/fastcann.git@v0.1.4
```

## Current Limits

- GPU only
- contiguous tensors only
- `float32` only
- `kernel_size=4` only
- forward, `grad_x`, and `grad_mix` implemented

## Roadmap

- add `half` and `bfloat16`
- tune `TILE_L`, `CTILE`, `REDUCE_THREADS`, and `REDUCE_CTILE`
- replace the shared-memory reduction in `grad_mix` with a warp-level reduction
- vectorize loads when alignment allows
- consider a two-stage reduction for very large `B * L`
