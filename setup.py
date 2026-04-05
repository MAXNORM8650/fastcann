from pathlib import Path
import os
import shutil
import subprocess
import tempfile

from setuptools import find_packages, setup

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

ROCM_ARCH_ENV_VARS = (
    "PYTORCH_ROCM_ARCH",
    "HCC_AMDGPU_TARGET",
    "AMDGPU_TARGETS",
)


def _split_rocm_arches(value):
    return [arch for arch in value.replace(",", ";").split(";") if arch]


def _hipcc_path():
    return shutil.which("hipcc") or "/opt/rocm/bin/hipcc"


def _hipcc_supports_arch(arch):
    hipcc = _hipcc_path()
    if not Path(hipcc).exists():
        return True

    src = "__global__ void fastcann_probe() {}\n"
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "probe.hip"
        out_path = Path(tmpdir) / "probe.o"
        src_path.write_text(src, encoding="utf-8")
        cmd = [
            hipcc,
            "--cuda-device-only",
            f"--offload-arch={arch}",
            "-c",
            str(src_path),
            "-o",
            str(out_path),
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    return result.returncode == 0


def _sanitize_rocm_arch_env():
    for env_var in ROCM_ARCH_ENV_VARS:
        raw_value = os.environ.get(env_var)
        if not raw_value:
            continue

        requested_arches = _split_rocm_arches(raw_value)
        supported_arches = [arch for arch in requested_arches if _hipcc_supports_arch(arch)]
        dropped_arches = [arch for arch in requested_arches if arch not in supported_arches]

        if dropped_arches:
            print(
                f"fastcann: dropping unsupported ROCm archs from {env_var}: "
                + ", ".join(dropped_arches)
            )

        if supported_arches:
            os.environ[env_var] = ";".join(supported_arches)
        else:
            os.environ.pop(env_var, None)
            print(
                f"fastcann: removed {env_var} because none of its ROCm archs are supported by hipcc"
            )


def _resolve_rocm_arches(torch_module):
    requested_arches = []

    for env_var in ROCM_ARCH_ENV_VARS:
        raw_value = os.environ.get(env_var)
        if raw_value:
            requested_arches.extend(_split_rocm_arches(raw_value))

    if not requested_arches:
        arch_flags = torch_module._C._cuda_getArchFlags()
        if arch_flags:
            requested_arches.extend(
                flag.split("=", 1)[1]
                for flag in arch_flags.split()
                if flag.startswith("--offload-arch=")
            )

    if not requested_arches:
        return []

    supported_arches = []
    dropped_arches = []
    for arch in requested_arches:
        if arch in supported_arches or arch in dropped_arches:
            continue
        if _hipcc_supports_arch(arch):
            supported_arches.append(arch)
        else:
            dropped_arches.append(arch)

    if dropped_arches:
        print("fastcann: dropping unsupported ROCm archs: " + ", ".join(dropped_arches))

    return supported_arches


def get_extension_modules():
    import torch
    from torch.utils.cpp_extension import CUDAExtension

    is_rocm = getattr(torch.version, "hip", None) is not None

    cxx_flags = ["-O3", "-std=c++17"]
    device_flags = ["-O3", "-std=c++17"]

    if is_rocm:
        _sanitize_rocm_arch_env()
        rocm_arches = _resolve_rocm_arches(torch)
        device_flags += [f"--offload-arch={arch}" for arch in rocm_arches]
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
