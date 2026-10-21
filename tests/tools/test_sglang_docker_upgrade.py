# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check the SGLang image upgrade without importing GPU dependencies."""

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_sglang_requirements_align_with_sglang_dependency_requirements():
    requirements = set((ROOT / "requirements.txt").read_text().splitlines())
    assert {"numpy", "pyarrow>=16.0.0", "huggingface_hub>=1.10.0", "av==16.1.0"} <= requirements
    dockerfile = (ROOT / "docker/Dockerfile").read_text()
    assert "--requirements" not in dockerfile


def test_sglang_dockerfile_targets_official_cuda13_hopper_and_blackwell():
    dockerfile = (ROOT / "docker/Dockerfile").read_text()
    assert "lmsysorg/sglang:v0.5.21-cu130" in dockerfile
    assert "transformer_engine[core_cu13,pytorch]==2.18.0" in dockerfile
    assert "FLASH_ATTENTION_FORCE_BUILD=TRUE" in dockerfile
    assert 'FLASH_ATTN_CUDA_ARCHS="90;100"' in dockerfile
    assert dockerfile.count('TORCH_CUDA_ARCH_LIST="9.0;10.0;10.3"') == 2
    assert "ENV TORCH_CUDA_ARCH_LIST" not in dockerfile
    assert "FLASHINFER_CUDA_ARCH_LIST" not in dockerfile
    assert "FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE" in dockerfile
    assert "cd hopper/" in dockerfile
    assert "0f82fead0b2d59f6390c62a1d74e927159bc5a7b" in dockerfile
    assert "setup_memory_saver.py" in dockerfile
    assert "cu128/" not in dockerfile
    assert "nvidia-cudnn-cu12" not in dockerfile
    assert "hook_mode_preload_cu12" not in dockerfile
    assert "efa-installer" not in dockerfile
    assert "2.18.0-fa2-sm103.patch" in dockerfile
    assert "0.1.12-blackwell-fp16-store.patch" in dockerfile
    assert "PIP_CONSTRAINT" not in dockerfile
    assert "ENV CPLUS_INCLUDE_PATH" not in dockerfile
    assert "ENV CUDA_HOME" not in dockerfile


@pytest.mark.parametrize("build_gpu", [(9, 0), (10, 3)])
def test_int4_explicit_build_targets_override_visible_gpu(monkeypatch, build_gpu):
    """Exercise setup.py with stubs; never import Torch or compile an
    extension."""
    captured = {}
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1, get_device_capability=lambda _index: build_gpu
    )
    cpp_extension = ModuleType("torch.utils.cpp_extension")
    cpp_extension.BuildExtension = object
    cpp_extension.CUDAExtension = lambda **kwargs: kwargs
    setuptools = ModuleType("setuptools")
    setuptools.setup = lambda **kwargs: captured.update(kwargs)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", ModuleType("torch.utils"))
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", cpp_extension)
    monkeypatch.setitem(sys.modules, "setuptools", setuptools)
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "9.0;10.0;10.3")
    runpy.run_path(str(ROOT / "relax/backends/megatron/kernels/int4_qat/setup.py"))
    flags = captured["ext_modules"][0]["extra_compile_args"]["nvcc"]
    for arch in ("90", "100", "103"):
        assert f"-gencode=arch=compute_{arch},code=sm_{arch}" in flags


def test_sglang_dockerfile_uses_latest_patch_links():
    dockerfile = (ROOT / "docker/Dockerfile").read_text()
    assert "ARG PATCH_VERSION=latest" in dockerfile
    assert "SGLANG_PATCH_VERSION" not in dockerfile
    assert "COPY docker/patch/${PATCH_VERSION}/megatron.patch" in dockerfile
    assert "COPY docker/patch/${PATCH_VERSION}/sglang.patch" in dockerfile
    sglang_patch = ROOT / "docker/patch/latest/sglang.patch"
    assert sglang_patch.is_symlink()
    assert sglang_patch.readlink() == Path("../sglang/v0.5.21.patch")
    assert sglang_patch.resolve() == (ROOT / "docker/patch/sglang/v0.5.21.patch").resolve()
    assert (ROOT / "docker/patch/latest/megatron.patch").readlink() == Path("../megatron/20260915-4a4de8657.patch")
    assert "ARG MEGATRON_BRIDGE_COMMIT=6862ee170a2569607e8ebc5dde91f119779093ca" in dockerfile


def test_sglang_dockerfile_restores_official_nccl_override_last():
    dockerfile = (ROOT / "docker/Dockerfile").read_text()
    pip_installs = [line for line in dockerfile.splitlines() if "pip" in line.split() and "install" in line.split()]
    assert pip_installs[-1] == ("RUN pip install --no-cache-dir --force-reinstall --no-deps nvidia-nccl-cu13==2.30.7")
