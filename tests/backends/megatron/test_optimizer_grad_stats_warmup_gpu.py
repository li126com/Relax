# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real two-rank MCore optimizer regression; requires two idle CUDA GPUs."""

import copy
import datetime
import time
from argparse import Namespace
from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class _DenseAndExpert(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.dense = torch.nn.Parameter((torch.arange(512, device=device) % 17 / 16).to(torch.bfloat16))
        self.expert = torch.nn.Parameter((torch.arange(256, device=device) % 13 / 16).to(torch.bfloat16))
        self.expert.allreduce = False

    def forward(self, rank: int, microbatch: int) -> torch.Tensor:
        return sum(
            (parameter * ((torch.arange(parameter.numel(), device=parameter.device) % 11 + rank + microbatch) / 8))
            .float()
            .sum()
            for parameter in self.parameters()
        )


def _assert_tree_equal(actual, expected) -> None:
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_tree_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_tree_equal(left, right)
    else:
        assert actual == expected


def _optimizer_warmup_worker(rank: int, rendezvous: str) -> None:
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.optimizer.optimizer import ChainedOptimizer
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    from relax.backends.megatron.grad_sync_warmup import warmup_optimizer_grad_stats

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=datetime.timedelta(seconds=35)
    )
    try:
        parallel_state.initialize_model_parallel(expert_model_parallel_size=2, expert_tensor_parallel_size=1)
        model_parallel_cuda_manual_seed(1234)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=1,
            num_moe_experts=2,
            expert_model_parallel_size=2,
            calculate_per_token_loss=True,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        models, optimizers = [], []
        for _ in range(2):
            ddp = DistributedDataParallel(
                config,
                DistributedDataParallelConfig(
                    use_distributed_optimizer=True,
                    overlap_grad_reduce=True,
                    overlap_param_gather=True,
                    grad_reduce_in_fp32=True,
                ),
                _DenseAndExpert(device),
            )
            ddp.disable_forward_pre_hook(param_sync=False)
            optimizer = get_megatron_optimizer(
                OptimizerConfig(
                    optimizer="adam",
                    lr=1 / 32,
                    min_lr=0,
                    weight_decay=0,
                    bf16=True,
                    use_distributed_optimizer=True,
                    clip_grad=1,
                ),
                [ddp],
                use_gloo_process_groups=False,
            )
            ddp.zero_grad_buffer()
            optimizer.zero_grad()
            models.append(ddp)
            optimizers.append(optimizer)

        candidate = optimizers[0]
        assert isinstance(candidate, ChainedOptimizer) and len(candidate.chained_optimizers) == 2
        leaves = candidate.chained_optimizers
        group = leaves[0].get_grad_stats_parallel_group()
        assert group is leaves[1].get_grad_stats_parallel_group()
        assert group is not dist.group.WORLD and dist.get_world_size(group) == 2
        # Same membership does not make DDP's dense reduction PG the norm PG.
        assert group is not parallel_state.get_data_parallel_group(with_context_parallel=True)
        nested = ChainedOptimizer([candidate])
        args = Namespace(fully_async=True, use_distributed_optimizer=True, overlap_grad_reduce=False)
        parameters = list(models[0].parameters())
        originals = [
            (p.detach().clone(), p.main_grad.clone(), p.data_ptr(), p.main_grad.data_ptr()) for p in parameters
        ]
        states = [copy.deepcopy(leaf.optimizer.state_dict()) for leaf in leaves]
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        tracked_rng = {name: state.clone() for name, state in get_cuda_rng_tracker().get_states().items()}
        # A transparent spy delegates every call to the real NCCL collective.
        with patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reductions:
            warmup_optimizer_grad_stats(args, nested)
            assert reductions.call_count == 1
            assert reductions.call_args.kwargs["group"] is group
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
        _assert_tree_equal(get_cuda_rng_tracker().get_states(), tracked_rng)
        for leaf, before in zip(leaves, states, strict=True):
            _assert_tree_equal(leaf.optimizer.state_dict(), before)
            assert not getattr(leaf, "_has_grad_norm_group_cache", None)
            assert not getattr(leaf, "grad_norms_by_group", None)
        for parameter, (value, gradient, pointer, gradient_pointer) in zip(parameters, originals, strict=True):
            assert torch.equal(parameter, value) and torch.equal(parameter.main_grad, gradient)
            assert parameter.data_ptr() == pointer and parameter.main_grad.data_ptr() == gradient_pointer
            assert parameter.grad is None

        updates, gradients = [], []
        for ddp, optimizer in zip(models, optimizers, strict=True):
            for microbatch in range(2):
                with ddp.no_sync() if microbatch == 0 else nullcontext():
                    (ddp(rank, microbatch) / 2).backward()
            ddp.finish_grad_sync()
            gradients.append([parameter.main_grad.clone() for parameter in ddp.parameters()])
            if optimizer is candidate:
                with patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reductions:
                    warmup_optimizer_grad_stats(args, nested)
                    assert reductions.call_count == 0
                for parameter, before in zip(ddp.parameters(), gradients[-1], strict=True):
                    assert torch.equal(parameter.main_grad, before)
            updates.append(optimizer.step())
            ddp.start_param_sync(force_sync=True)
        assert updates[0][0] and float(updates[0][1]) > 0
        _assert_tree_equal(gradients[0], gradients[1])
        _assert_tree_equal(updates[0], updates[1])
        for candidate_parameter, reference_parameter, original in zip(
            models[0].parameters(), models[1].parameters(), originals, strict=True
        ):
            assert torch.equal(candidate_parameter, reference_parameter)
            assert not torch.equal(candidate_parameter, original[0])
        for candidate_leaf, reference_leaf in zip(
            optimizers[0].chained_optimizers, optimizers[1].chained_optimizers, strict=True
        ):
            _assert_tree_equal(candidate_leaf.optimizer.state_dict(), reference_leaf.optimizer.state_dict())
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="requires two idle CUDA GPUs"
)
def test_real_optimizer_norm_warmup_preserves_state_and_first_update(tmp_path):
    pytest.importorskip("megatron.core.distributed")
    context = mp.spawn(_optimizer_warmup_worker, args=((tmp_path / "nccl-init").as_uri(),), nprocs=2, join=False)
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("optimizer gradient-statistics warmup exceeded 90 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
