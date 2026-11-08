# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real CPU DDP regression for trainable conditional vision across updates.

This exercises PyTorch's reducer and autograd hooks, not Megatron's first-batch
bucket accounting or CUDA/CP collectives, which require separate integration.
"""

import datetime
import time
from contextlib import nullcontext
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .test_conditional_branch_sync import _CallbackVisionModel, _install_callback, _reference_forward


def _average_gradients(state, bucket):
    state["calls"] += 1
    tensor = bucket.buffer()
    dist.all_reduce(tensor, group=dist.group.WORLD)
    tensor.div_(dist.get_world_size(group=dist.group.WORLD))
    result = torch.futures.Future()
    result.set_result(tensor)
    return result


def _ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=datetime.timedelta(seconds=30)
    )
    handles = []
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            model = _CallbackVisionModel()
            reference = deepcopy(model)
            _install_callback(model, monkeypatch)
            ddp = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
            communication = {"calls": 0}
            ddp.register_comm_hook(communication, _average_gradients)
            counts = {name: 0 for name, _ in model.named_parameters()}

            def count_hook(name):
                def hook(grad):
                    counts[name] += 1

                return hook

            handles = [parameter.register_hook(count_hook(name)) for name, parameter in model.named_parameters()]
            # First step entirely empty, then DP-mixed; accumulated gradients
            # must survive an empty final microbatch. Last case has nonempty
            # original inputs on both ranks but an empty post-CP shard on one.
            scenarios = [
                [(False, False)],
                [(rank == 0, False)],
                [(True, False), (False, False)],
                [(True, rank == 1)],
            ]
            for step, microbatches in enumerate(scenarios):
                ddp.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                counts.update(dict.fromkeys(counts, 0))
                vision_calls_before = model.vision_model.forward_calls
                for index, (has_images, cp_empty) in enumerate(microbatches):
                    values = torch.tensor([[0.4 + 0.1 * rank, 0.7 + 0.05 * step]])
                    kwargs = {"cp_empty": cp_empty}
                    if has_images:
                        kwargs.update(pixel_values=torch.tensor([[0.3, 0.6]]), image_grid_thw=torch.ones(1, 3))
                    expected = _reference_forward(reference, values, **kwargs)
                    context = ddp.no_sync() if index + 1 < len(microbatches) else nullcontext()
                    with context:
                        output = ddp(values, **kwargs)
                        torch.testing.assert_close(output, expected)
                        (output.square().mean() / len(microbatches)).backward()
                    (expected.square().mean() / len(microbatches)).backward()
                assert communication["calls"] == step + 1
                assert model.vision_model.forward_calls - vision_calls_before == len(microbatches)
                assert all(count == len(microbatches) for count in counts.values()), counts
                for (name, parameter), (_, expected_parameter) in zip(
                    model.named_parameters(), reference.named_parameters()
                ):
                    expected_grad = expected_parameter.grad
                    if expected_grad is None:
                        expected_grad = torch.zeros_like(expected_parameter)
                    dist.all_reduce(expected_grad, group=dist.group.WORLD)
                    expected_grad.div_(2)
                    assert parameter.grad is not None, name
                    torch.testing.assert_close(parameter.grad, expected_grad)
                    with torch.no_grad():
                        parameter.add_(parameter.grad, alpha=-0.01)
                        expected_parameter.add_(expected_grad, alpha=-0.01)
                    torch.testing.assert_close(parameter, expected_parameter)
    finally:
        for handle in handles:
            handle.remove()
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires CPU Gloo")
def test_empty_vision_across_steps_and_accumulation_with_real_ddp(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    context = mp.spawn(_ddp_worker, args=((tmp_path / "gloo-init").as_uri(),), nprocs=2, join=False)
    deadline = time.monotonic() + 45
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("conditional-vision Gloo regression exceeded 45 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
