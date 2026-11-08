# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""GPU regression for initialization warmup's parameter and RNG isolation."""

from argparse import Namespace
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native MoE kernel warmup requires CUDA and Megatron")
def test_native_moe_warmup_preserves_training_state(tmp_path):
    import torch.distributed as dist
    import torch.nn.functional as functional

    pytest.importorskip("megatron.core")
    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel.random import checkpoint, get_cuda_rng_tracker, model_parallel_cuda_manual_seed
    from megatron.core.transformer.moe.moe_utils import RouterGatingLinearFunction
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

    from relax.backends.megatron.native_moe_gemm_warmup import warmup_native_moe_linears

    if dist.is_initialized():
        pytest.skip("This state-isolation regression requires its own single-rank process group")
    dist.init_process_group("nccl", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1)
    try:
        mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(1234)
        # Real module types with only the native parameters used by this helper.
        # The test does not claim to construct a complete expert/optimizer/DDP.
        router = TopKRouter.__new__(TopKRouter)
        torch.nn.Module.__init__(router)
        router.config = SimpleNamespace(moe_router_dtype="fp32", params_dtype=torch.bfloat16)
        router.weight = torch.nn.Parameter(torch.full((512, 4096), 1 / 256, device="cuda", dtype=torch.bfloat16))
        shared = SharedExpertMLP.__new__(SharedExpertMLP)
        torch.nn.Module.__init__(shared)
        shared.gate_weight = torch.nn.Parameter(torch.full((1, 4096), 1 / 256, device="cuda", dtype=torch.bfloat16))
        model = torch.nn.ModuleList([router, shared])
        hooks = []
        originals = []
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 1 / 128)
            parameter.main_grad = torch.full_like(parameter, 1 / 64, dtype=torch.float32)
            parameter.register_hook(lambda gradient: hooks.append(True))
            originals.append(
                [
                    (tensor.detach().clone(), tensor.data_ptr())
                    for tensor in (parameter, parameter.grad, parameter.main_grad)
                ]
            )

        def native_result():
            inputs = torch.full((23, 4096), 1 / 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            weight = router.weight.detach().clone().requires_grad_()
            gate = shared.gate_weight.detach().clone().requires_grad_()

            def forward(hidden, router_weight, gate_weight):
                return (
                    RouterGatingLinearFunction.apply(hidden, router_weight, None, torch.float32),
                    functional.linear(hidden, gate_weight),
                )

            routed, gated = checkpoint(forward, False, inputs, weight, gate)
            (routed.square().mean() + gated.float().square().mean()).backward()
            return [tensor.detach().clone() for tensor in (routed, gated, inputs.grad, weight.grad, gate.grad)]

        baseline = native_result()
        assert all(torch.count_nonzero(tensor) > 0 for tensor in baseline)
        for budget in (256, 1920):
            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state().clone()
            tracker_rng = {name: state.clone() for name, state in get_cuda_rng_tracker().get_states().items()}
            args = Namespace(
                fully_async=True,
                hybrid=False,
                offload_train=False,
                moe_token_dispatcher_type="flex",
                moe_flex_dispatcher_backend="deepep",
                sequence_parallel=True,
                max_tokens_per_gpu=budget,
                recompute_granularity="full",
            )
            with torch.no_grad():
                warmup_native_moe_linears(args, [model])
                assert not torch.is_grad_enabled()
            assert not hooks
            assert torch.equal(cpu_rng, torch.get_rng_state())
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
            current_rng = get_cuda_rng_tracker().get_states()
            assert current_rng.keys() == tracker_rng.keys()
            assert all(torch.equal(state, current_rng[name]) for name, state in tracker_rng.items())
            for parameter, original in zip(model.parameters(), originals, strict=True):
                for tensor, (value, pointer) in zip(
                    (parameter, parameter.grad, parameter.main_grad), original, strict=True
                ):
                    assert torch.equal(tensor, value)
                    assert tensor.data_ptr() == pointer
            for actual, expected in zip(native_result(), baseline, strict=True):
                assert torch.equal(actual, expected)
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()
