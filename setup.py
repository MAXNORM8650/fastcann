from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

is_rocm = getattr(torch.version, "hip", None) is not None

cxx_flags = ["-O3", "-std=c++17"]
device_flags = ["-O3", "-std=c++17"]

if is_rocm:
    device_flags += ["-fno-gpu-rdc"]
else:
    device_flags += ["--use_fast_math"]

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="fastcann",
    version="0.1.0",
    description="Fast CUDA-backed Canon layers for PyTorch",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    ext_modules=[
        CUDAExtension(
            name="fastcann._C",
            sources=["canon.cpp", "canon_kernel.cu"],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": device_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
