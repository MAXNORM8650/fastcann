from pathlib import Path

from setuptools import find_packages, setup

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")


def get_extension_modules():
    import torch
    from torch.utils.cpp_extension import CUDAExtension

    is_rocm = getattr(torch.version, "hip", None) is not None

    cxx_flags = ["-O3", "-std=c++17"]
    device_flags = ["-O3", "-std=c++17"]

    if is_rocm:
        device_flags += ["-fno-gpu-rdc"]
    else:
        device_flags += ["--use_fast_math"]

    return [
        CUDAExtension(
            name="fastcann._C",
            sources=["canon.cpp", "canon_kernel.cu"],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": device_flags,
            },
        )
    ]


def get_cmdclass():
    from torch.utils.cpp_extension import BuildExtension

    return {"build_ext": BuildExtension}


setup(
    name="fastcann",
    version="0.1.0",
    description="Fast GPU-backed Canon layers for PyTorch",
    long_description=README,
    long_description_content_type="text/markdown",
    author="MAXNORM8650",
    url="https://github.com/MAXNORM8650/fastcann",
    project_urls={
        "Source": "https://github.com/MAXNORM8650/fastcann",
        "Releases": "https://github.com/MAXNORM8650/fastcann/releases",
        "Issues": "https://github.com/MAXNORM8650/fastcann/issues",
    },
    packages=find_packages(),
    py_modules=["canon"],
    include_package_data=True,
    python_requires=">=3.10",
    ext_modules=get_extension_modules(),
    cmdclass=get_cmdclass(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)
