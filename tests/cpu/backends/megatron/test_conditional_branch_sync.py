# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


# Load this standalone wrapper without importing the Megatron backend's GPU patches.
_spec = importlib.util.spec_from_file_location(
    "conditional_branch_sync_under_test",
    Path(__file__).resolve().parents[4] / "relax/backends/megatron/conditional_branch_sync.py",
)
sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync)


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("local_images", [False, True])
def test_mixed_vision_forward_preserves_output_and_backward(monkeypatch, paired, local_images):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_weight = torch.nn.Parameter(torch.tensor(2.0))
            self.pg_collection = SimpleNamespace(dp=object())
            self.mask = torch.tensor([[1.0, 0.0]])

        def forward(self, input_ids, **kwargs):
            output = input_ids * (self.vision_weight if local_images else 3)
            return (output, self.mask) if paired else output

    model = Model()
    monkeypatch.setattr(sync, "_is_qwen3vl_with_conditional_vision", lambda _: True)
    monkeypatch.setattr(sync, "_dp_any", lambda *args: True)
    dummy_calls = []

    def dummy(model, device):
        dummy_calls.append(device)
        return model.vision_weight.square()

    monkeypatch.setattr(sync, "_run_dummy_vision", dummy)
    monkeypatch.setenv("RELAX_DISABLE_VISION_BRANCH_SYNC", "0")
    sync.install_conditional_branch_sync(Namespace(overlap_grad_reduce=True), model)
    inputs = torch.tensor([[1.0, 2.0]], requires_grad=True)
    kwargs = {"image_grid_thw": torch.ones(1, 3)} if local_images else {}
    result = model(inputs, **kwargs)
    if paired:
        assert isinstance(result, tuple) and len(result) == 2
        output, mask = result
        assert mask is model.mask
        assert not mask.requires_grad
    else:
        output = result
    torch.testing.assert_close(output, inputs * (2 if local_images else 3))
    output.sum().backward()
    torch.testing.assert_close(inputs.grad, torch.full_like(inputs, 2 if local_images else 3))
    assert model.vision_weight.grad is not None
    torch.testing.assert_close(model.vision_weight.grad, torch.tensor(3.0 if local_images else 0.0))
    assert len(dummy_calls) == (0 if local_images else 1)


def test_conditional_sync_rejects_unknown_output_contract():
    with pytest.raises(TypeError, match="tensor pair"):
        sync._attach_zero_dependency((torch.ones(1), None), torch.ones(1))


def test_text_only_opt_out_does_not_wrap_forward(monkeypatch):
    model = torch.nn.Linear(2, 2)
    original = model.forward
    monkeypatch.setenv("RELAX_DISABLE_VISION_BRANCH_SYNC", "1")
    monkeypatch.setattr(sync, "_is_qwen3vl_with_conditional_vision", lambda _: True)
    monkeypatch.setattr(sync, "_rank0_log_info", lambda _: None)
    sync.install_conditional_branch_sync(Namespace(overlap_grad_reduce=True), model)
    assert model.forward == original


class _TinyVision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = torch.nn.Module()
        self.patch_embed.proj = torch.nn.Linear(2, 2, bias=False)
        self.deepstack = torch.nn.Linear(2, 2, bias=False)
        self.forward_calls = 0
        with torch.no_grad():
            self.patch_embed.proj.weight.fill_(0.2)
            self.deepstack.weight.fill_(0.3)

    def forward(self, hidden_states, grid_thw):
        self.forward_calls += 1
        merged = self.patch_embed.proj(hidden_states)
        return merged, [self.deepstack(merged)]


class _CallbackVisionModel(torch.nn.Module):
    """Exercise the Bridge callback contracts with actual vision autograd.

    ``cp_empty`` models the post-split empty views consumed by CP AllGather;
    this small CPU model does not implement or claim to test CP transport.
    """

    _relax_empty_vision_forward_supported = True

    def __init__(self, *, paired=False):
        super().__init__()
        self.vision_model = _TinyVision()
        self.language_weight = torch.nn.Parameter(torch.eye(2) * 0.5)
        self.vision_transformer_config = SimpleNamespace(
            spatial_merge_size=1, in_channels=2, temporal_patch_size=1, patch_size=1
        )
        self.pg_collection = SimpleNamespace(dp=None)
        self.paired = paired
        self.mask = torch.tensor([[1.0, 0.0]])

    def forward(self, values, *, pixel_values=None, image_grid_thw=None, cp_empty=False):
        output = values @ self.language_weight
        local_values = pixel_values[:0] if cp_empty else pixel_values
        if local_values is not None and local_values.shape[0]:
            merged, deepstack = self.vision_model(local_values, image_grid_thw)
            output = output + merged.sum() + sum(feature.sum() for feature in deepstack)
        else:
            empty_outputs = self._relax_empty_vision_forward(values.device)
            if empty_outputs is not None:
                merged, deepstack = empty_outputs
                if cp_empty:
                    # Empty slices preserve the VJP edge through the real dummy
                    # forward, as the Bridge's CP AllGather inputs must do.
                    dependency = merged[:0].sum() + sum(feature[:0].sum() for feature in deepstack)
                else:
                    dependency = (merged.float().sum() + sum(feature.float().sum() for feature in deepstack)) * 0
                output = output + dependency.to(output.dtype)
        return (output, self.mask) if self.paired else output


def _reference_forward(model, values, *, pixel_values=None, image_grid_thw=None, cp_empty=False):
    """Ordinary conditional-model math, without an empty-branch callback."""
    output = values @ model.language_weight
    if pixel_values is not None and not cp_empty:
        merged, deepstack = model.vision_model(pixel_values, image_grid_thw)
        output = output + merged.sum() + sum(feature.sum() for feature in deepstack)
    return output


def _install_callback(model, monkeypatch):
    monkeypatch.setattr(sync, "_is_qwen3vl_with_conditional_vision", lambda _: True)
    monkeypatch.setattr(sync, "_dp_any", lambda *_: pytest.fail("Bridge callback must not add a DP collective"))
    monkeypatch.setenv("RELAX_DISABLE_VISION_BRANCH_SYNC", "0")
    original = model.forward
    sync.install_conditional_branch_sync(Namespace(overlap_grad_reduce=True, overlap_param_gather=True), model)
    sync.install_conditional_branch_sync(Namespace(overlap_grad_reduce=True, overlap_param_gather=True), model)
    assert model.forward == original
    assert callable(model._relax_empty_vision_forward)


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("branch", ["images", "no_images", "cp_empty"])
def test_bridge_callback_preserves_math_and_vision_hook_count(monkeypatch, paired, branch):
    model = _CallbackVisionModel(paired=paired)
    reference = deepcopy(model)
    _install_callback(model, monkeypatch)
    counts = {name: 0 for name, _ in model.vision_model.named_parameters()}

    def count_hook(name):
        def hook(grad):
            counts[name] += 1

        return hook

    handles = [parameter.register_hook(count_hook(name)) for name, parameter in model.vision_model.named_parameters()]
    values = torch.tensor([[1.0, 2.0]], requires_grad=True)
    expected_values = values.detach().clone().requires_grad_()
    kwargs = {}
    if branch != "no_images":
        kwargs = {"pixel_values": torch.ones(1, 2), "image_grid_thw": torch.ones(1, 3)}
    kwargs["cp_empty"] = branch == "cp_empty"
    try:
        result = model(values, **kwargs)
        if paired:
            output, mask = result
            assert mask is model.mask and not mask.requires_grad
        else:
            output = result
        expected = _reference_forward(reference, expected_values, **kwargs)
        torch.testing.assert_close(output, expected)
        output.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(values.grad, expected_values.grad)
        for (name, parameter), (_, expected_parameter) in zip(model.named_parameters(), reference.named_parameters()):
            expected_grad = expected_parameter.grad
            if expected_grad is None:
                expected_grad = torch.zeros_like(expected_parameter)
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, expected_grad)
        assert model.vision_model.forward_calls == 1
        assert all(count == 1 for count in counts.values())
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_bridge_empty_callback_skips_inference_and_frozen_vision(monkeypatch, frozen, grad_enabled):
    model = _CallbackVisionModel()
    _install_callback(model, monkeypatch)
    if frozen:
        model.vision_model.requires_grad_(False)
        sync.disable_frozen_conditional_branch_sync(model)
    values = torch.ones(1, 2)
    with torch.set_grad_enabled(grad_enabled):
        output = model(values)
        torch.testing.assert_close(output, values @ model.language_weight)
        assert output.requires_grad is grad_enabled
    assert model.vision_model.forward_calls == int(grad_enabled and not frozen)


def test_bridge_empty_callback_retains_trainable_projection(monkeypatch):
    model = _CallbackVisionModel()
    _install_callback(model, monkeypatch)
    model.vision_model.patch_embed.requires_grad_(False)
    sync.disable_frozen_conditional_branch_sync(model)
    model(torch.ones(1, 2)).sum().backward()
    assert model.vision_model.forward_calls == 1
    assert model.vision_model.patch_embed.proj.weight.grad is None
    torch.testing.assert_close(model.vision_model.deepstack.weight.grad, torch.zeros(2, 2))
