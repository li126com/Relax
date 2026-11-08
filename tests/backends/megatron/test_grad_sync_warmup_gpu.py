# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real MCore/NCCL regression; requires two otherwise idle CUDA devices."""

import datetime
import time
from argparse import Namespace
from contextlib import nullcontext

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

    def forward(self, dense_input: torch.Tensor, expert_input: torch.Tensor) -> torch.Tensor:
        return (self.dense * dense_input).float().sum() + (self.expert * expert_input).float().sum()


def _inputs(rank: int, step: int, microbatch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(
        ((torch.arange(size, device=device) % 11 + rank + step + microbatch) / 8).to(torch.bfloat16)
        for size in (512, 256)
    )


def _warmup_worker(rank: int, rendezvous: str) -> None:
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.transformer.transformer_config import TransformerConfig

    from relax.backends.megatron.grad_sync_warmup import warmup_distributed_optimizer_grad_sync

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=datetime.timedelta(seconds=35)
    )
    try:
        parallel_state.initialize_model_parallel(expert_model_parallel_size=2, expert_tensor_parallel_size=1)
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
        module = _DenseAndExpert(device)
        original = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}
        ddp = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                use_distributed_optimizer=True,
                overlap_grad_reduce=True,
                overlap_param_gather=True,
                grad_reduce_in_fp32=True,
            ),
            module,
        )
        # Production disables param pre-hooks before the first train sub-step;
        # this test concerns gradient communication, not optimizer parameter AG.
        ddp.disable_forward_pre_hook(param_sync=False)
        groups = ddp.bucket_groups + ddp.expert_parallel_bucket_groups
        assert [group.intra_distributed_optimizer_instance_size for group in groups] == [2, 1]
        args = Namespace(fully_async=True, use_distributed_optimizer=True, overlap_grad_reduce=True)
        ddp.zero_grad_buffer()
        warmup_distributed_optimizer_grad_sync(args, [ddp])
        for group in groups:
            assert group.is_first_batch and not group.golden_per_param_grad_ready_counts
            assert not group.per_param_grad_ready_counts and not group.grad_reduce_finished
            assert group.grad_reduce_handle is None and group.param_gather_handle is None
            assert all(torch.count_nonzero(bucket.grad_data) == 0 for bucket in group.buckets)

        for step in range(2):
            ddp.zero_grad_buffer()
            if step:
                assert all(not group.is_first_batch for group in groups)
                assert all(set(group.golden_per_param_grad_ready_counts.values()) == {1} for group in groups)
            reference = {
                name: parameter.detach().float().cpu().requires_grad_() for name, parameter in original.items()
            }
            for source_rank in range(2):
                for microbatch in range(2):
                    dense_input, expert_input = _inputs(source_rank, step, microbatch, torch.device("cpu"))
                    loss = (reference["dense"] * dense_input.float()).sum() / 2
                    if source_rank == rank:
                        loss = loss + (reference["expert"] * expert_input.float()).sum() / 2
                    loss.backward()
            for microbatch in range(2):
                with ddp.no_sync() if microbatch == 0 else nullcontext():
                    (ddp(*_inputs(rank, step, microbatch, device)) / 2).backward()
            ddp.finish_grad_sync()

            # Compare only DistOpt's owned output shards, including padding.
            # Unowned input regions need not contain reduced gradients.
            for group in groups:
                for bucket in group.buckets:
                    expected = torch.zeros(bucket.grad_data.numel())
                    for name, parameter in module.named_parameters():
                        if parameter in bucket.param_to_index:
                            start, end = bucket.param_to_index[parameter]
                            expected[start:end] = reference[name].grad.flatten()
                    size = group.intra_distributed_optimizer_instance_size
                    shard_rank = group.intra_distributed_optimizer_instance_rank
                    actual = bucket.grad_data.chunk(size)[shard_rank].cpu()
                    torch.testing.assert_close(actual, expected.chunk(size)[shard_rank], rtol=0, atol=0)
                assert set(group.per_param_grad_ready_counts.values()) == {1}
            before = [bucket.grad_data.clone() for group in groups for bucket in group.buckets]
            warmup_distributed_optimizer_grad_sync(args, [ddp])
            for snapshot, bucket in zip(before, (bucket for group in groups for bucket in group.buckets), strict=True):
                torch.testing.assert_close(snapshot, bucket.grad_data, rtol=0, atol=0)
            for name, parameter in module.named_parameters():
                torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)

        # An unmarked live iteration must be rejected before touching its grads.
        del ddp._relax_grad_sync_warmed_up
        with pytest.raises(RuntimeError, match="untouched, idle first-batch"):
            warmup_distributed_optimizer_grad_sync(args, [ddp])
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="requires two idle CUDA GPUs"
)
def test_real_distopt_warmup_preserves_gradients_and_first_batch_calibration(tmp_path):
    pytest.importorskip("megatron.core.distributed")
    context = mp.spawn(_warmup_worker, args=((tmp_path / "nccl-init").as_uri(),), nprocs=2, join=False)
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("gradient warmup NCCL regression exceeded 90 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
