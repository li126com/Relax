# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Check the SGLang image upgrade without importing GPU dependencies."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_sglang_requirements_align_with_sglang_dependency_requirements():
    requirements = set((ROOT / "requirements.txt").read_text().splitlines())
    assert {"numpy", "pyarrow>=16.0.0", "huggingface_hub>=1.10.0", "av==16.1.0"} <= requirements
    dockerfile = (ROOT / "docker/Dockerfile").read_text()
    assert "--requirements" not in dockerfile


def test_sglang_dockerfile_targets_official_cuda13_hopper():
    dockerfile = (ROOT / "docker/Dockerfile").read_text()
    assert "lmsysorg/sglang:v0.5.21-cu130" in dockerfile
    assert "transformer_engine[core_cu13,pytorch]==2.18.0" in dockerfile
    assert "FLASH_ATTENTION_FORCE_BUILD=TRUE" in dockerfile
    assert "cd hopper/" in dockerfile
    assert "0f82fead0b2d59f6390c62a1d74e927159bc5a7b" in dockerfile
    assert "setup_memory_saver.py" in dockerfile
    assert "cu128/" not in dockerfile
    assert "nvidia-cudnn-cu12" not in dockerfile
    assert "hook_mode_preload_cu12" not in dockerfile
    assert "efa-installer" not in dockerfile
    assert "sm103" not in dockerfile
    assert "PIP_CONSTRAINT" not in dockerfile
    assert "ENV CPLUS_INCLUDE_PATH" not in dockerfile
    assert "ENV CUDA_HOME" not in dockerfile


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
    commands = [line for line in dockerfile.splitlines() if line.startswith("RUN ")]
    assert commands[-1] == ("RUN pip install --no-cache-dir --force-reinstall --no-deps nvidia-nccl-cu13==2.30.7")
