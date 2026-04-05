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

Standard install:

```bash
pip install .
```

Editable install for development:

```bash
pip install -e . --no-build-isolation
```

Manual in-place extension build:

```bash
python setup.py build_ext --inplace
```

If you are building inside a managed environment in this repo, the working command used here was:

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
