# fastcann

This project packages a custom PyTorch CUDA extension for fast `CanonLayer` local mixing with `kernel_size=4` and `float32` CUDA tensors.

## Operator

For each batch `b`, token `t`, and channel `c`:

```text
y[b, t, c] = x[b, t, c]
           + mix[0, c] * x[b, t + 2, c]
           + mix[1, c] * x[b, t + 1, c]
           + mix[2, c] * x[b, t, c]
           + mix[3, c] * x[b, t - 1, c]
```

Values outside the sequence bounds are treated as zero.

## Package layout

- `setup.py` - package build script
- `fastcann/` - installable Python package
- `canon.cpp` - Python bindings
- `canon_kernel.cu` - CUDA kernels and launchers
- `fast_canon_layers/` - legacy compatibility package
- `canon.py` - backward-compatible import shim
- `test_canon_cuda.py` - correctness and speed test

## Install

From this directory:

```bash
pip install .
```

If you want an editable install during development:

```bash
pip install -e .
```

You can still build the extension in place if needed:

```bash
python setup.py build_ext --inplace
```

## Usage

```python
from fastcann import CanonLayerCUDA
```

The previous import paths remain available:

```python
from fast_canon_layers import CanonLayerCUDA
from canon import CanonLayerCUDA
```

## Test

```bash
python test_canon_cuda.py
```

## Current scope

- CUDA only
- contiguous tensors only
- `float32` only
- `kernel_size=4` only
- forward, `grad_x`, and `grad_mix` implemented in CUDA

## Good next upgrades

- add `half` and `bfloat16`
- tune `TILE_L`, `CTILE`, `REDUCE_THREADS`, `REDUCE_CTILE`
- replace block-wide shared-memory reduction with warp-level reduction in `grad_mix`
- vectorize loads when alignment allows
- consider a two-stage reduction for very large `B * L`

# fastcann
