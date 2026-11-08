# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Preload native MoE linear kernels before resident async communication starts."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Sequence
from typing import TYPE_CHECKING

from relax.utils.logging_utils import get_logger


if TYPE_CHECKING:
    import torch

logger = get_logger(__name__)


def warmup_native_moe_linears(args: Namespace, model: Sequence[torch.nn.Module]) -> None:
    """Exercise native router/gate forward and backward on temporary leaves.

    A cold cuBLAS call can wait for work on other CUDA streams. During pipeline
    training that work can itself be waiting for a peer's next MoE collective.
    Run during model initialization, before any actor starts training, rather
    than inserting synchronization into the training loop.

    These representative token sizes preload the observed GEMM families; they
    do not claim to enumerate every shape an arbitrary model might select.
    Original parameters, their gradients, and optimizer/DDP state are untouched.
    """
    if not (
        getattr(args, "fully_async", False)
        and not getattr(args, "hybrid", False)
        and not getattr(args, "offload_train", False)
        and getattr(args, "moe_token_dispatcher_type", None) == "flex"
        and getattr(args, "moe_flex_dispatcher_backend", None) == "deepep"
    ):
        return

    import torch
    import torch.nn.functional as functional
    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel.random import checkpoint
    from megatron.core.transformer.moe.moe_utils import RouterGatingLinearFunction
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

    specs = {}
    for chunk in model:
        for module in chunk.modules():
            if isinstance(module, SharedExpertMLP) and getattr(module, "gate_weight", None) is not None:
                weight, kind, router_dtype = module.gate_weight, "shared_gate", None
                input_dtype = weight.dtype
            elif isinstance(module, TopKRouter):
                weight, kind = module.weight, "router"
                input_dtype = module.config.params_dtype
                router_dtype = {"fp32": torch.float32, "fp64": torch.float64}.get(
                    module.config.moe_router_dtype, input_dtype
                )
            else:
                continue
            if weight.device.type != "cuda":
                continue
            key = (
                kind,
                weight.device,
                tuple(weight.shape),
                tuple(weight.stride()),
                weight.dtype,
                input_dtype,
                router_dtype,
            )
            specs.setdefault(key, weight)

    if not specs:
        return
    tp_size = mpu.get_tensor_model_parallel_world_size() if getattr(args, "sequence_parallel", False) else 1
    budget = getattr(args, "max_tokens_per_gpu", None) or 256 * tp_size
    local_budget = max(1, (budget + tp_size - 1) // tp_size)
    token_sizes = sorted({1, min(256, local_budget), min(2048, local_budget)})
    recompute = getattr(args, "recompute_granularity", None) == "full"
    devices = set()
    with torch.enable_grad(), torch.autocast(device_type="cuda", enabled=False):
        for key, original_weight in specs.items():
            kind, device, shape, _stride, _weight_dtype, input_dtype, router_dtype = key
            devices.add(device)

            def linear(
                inputs: torch.Tensor,
                temporary_weight: torch.Tensor,
                warmup_kind: str = kind,
                warmup_dtype: torch.dtype | None = router_dtype,
            ) -> torch.Tensor:
                if warmup_kind == "router":
                    return RouterGatingLinearFunction.apply(inputs, temporary_weight, None, warmup_dtype)
                return functional.linear(inputs, temporary_weight)

            for tokens in token_sizes:
                weight = torch.zeros_like(original_weight, requires_grad=True)
                inputs = torch.full((tokens, shape[1]), 1.0 / shape[1], dtype=input_dtype, device=device)
                inputs.requires_grad_()
                output = checkpoint(linear, False, inputs, weight) if recompute else linear(inputs, weight)
                output.backward(torch.ones_like(output))
                del output, inputs, weight
    for device in devices:
        torch.cuda.synchronize(device)
    logger.info("Preloaded %d native MoE linear kernel families at token sizes %s", len(specs), token_sizes)
