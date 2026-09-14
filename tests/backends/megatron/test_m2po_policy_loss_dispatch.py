# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""M2PO dispatch and scalar metrics through the Megatron loss path."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _load_loss_module():
    try:
        import megatron.core  # noqa: F401
    except ModuleNotFoundError:
        megatron = ModuleType("megatron")
        core = ModuleType("megatron.core")
        mpu = ModuleType("megatron.core.mpu")
        core.mpu = mpu
        sys.modules.update(
            {
                "megatron": megatron,
                "megatron.core": core,
                "megatron.core.mpu": mpu,
            }
        )
        try:
            return importlib.import_module("relax.backends.megatron.loss")
        finally:
            sys.modules.pop("megatron.core.mpu", None)
            sys.modules.pop("megatron.core", None)
            sys.modules.pop("megatron", None)
    return importlib.import_module("relax.backends.megatron.loss")


loss_module = _load_loss_module()


def _stub_policy_forward(monkeypatch, log_probs):
    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *_args, **_kwargs: (
            None,
            {"log_probs": [log_probs], "entropy": [torch.zeros_like(log_probs)]},
        ),
    )
    monkeypatch.setattr(loss_module, "resolve_opd_gather_topk_token_ids", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(loss_module, "compute_policy_opd_loss", lambda **_kwargs: (None, {}))
    monkeypatch.setattr(loss_module.mpu, "get_data_parallel_world_size", lambda **_kwargs: 1, raising=False)


def _args(*, calculate_per_token_loss):
    return SimpleNamespace(
        loss_type="policy_loss",
        advantage_estimator="m2po",
        calculate_per_token_loss=calculate_per_token_loss,
        qkv_format="thd",
        recompute_loss_function=False,
        allgather_cp=False,
        global_batch_size=2,
        true_on_policy_mode=False,
        use_rollout_logprobs=False,
        use_opsm=False,
        get_mismatch_metrics=False,
        use_tis=False,
        custom_pg_loss_reducer_function_path=None,
        entropy_coef=0.0,
        use_kl_loss=False,
        m2po_kl2_budget=0.01,
        m2po_miniclip_low=0.3,
        m2po_miniclip_high=0.5,
    )


def _batch(old_log_probs, advantages):
    response_lengths = [2, 3]
    return {
        "advantages": advantages,
        "log_probs": [old_log_probs],
        "response_lengths": response_lengths,
        "total_lengths": [length + 1 for length in response_lengths],
        "unconcat_tokens": [torch.arange(length + 1) for length in response_lengths],
        "loss_masks": [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 0.0, 1.0])],
        "dynamic_cp_size": 1,
        "dynamic_cp_rank": 0,
    }


@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
def test_m2po_scalar_metrics_use_the_framework_denominator(monkeypatch, calculate_per_token_loss):
    """One scalar per microbatch must not shrink after global aggregation."""
    from relax.algorithms import get_algorithm
    from relax.utils.training.ppo_utils import compute_m2po_loss

    log_probs = torch.tensor([-0.5, -1.2, -0.8, -1.1, -0.6])
    old_log_probs = torch.full_like(log_probs, -1.0)
    advantages = torch.tensor([1.0, 1.0, -1.0, -1.0, -1.0])
    batch = _batch(old_log_probs, advantages)
    _stub_policy_forward(monkeypatch, log_probs)

    _, normalizer, logging = loss_module.loss_function(
        _args(calculate_per_token_loss=calculate_per_token_loss),
        batch,
        num_microbatches=1,
        logits=torch.empty(1, 1, 1),
    )

    _, _, *raw_metrics = compute_m2po_loss(
        old_log_probs - log_probs,
        advantages,
        0.01,
        0.3,
        0.5,
        loss_mask=torch.cat(batch["loss_masks"]),
    )
    logged = dict(zip(logging["keys"], logging["values"][1:], strict=True))
    metric_names = get_algorithm("m2po").policy_scalar_metric_names
    framework_denominator = torch.cat(batch["loss_masks"]).sum() if calculate_per_token_loss else 2

    if calculate_per_token_loss:
        assert torch.equal(normalizer, framework_denominator)
    for name, raw in zip(metric_names, raw_metrics, strict=True):
        assert torch.equal(logged[name], torch.as_tensor(raw) * framework_denominator)
        assert torch.equal(logged[name] / framework_denominator, torch.as_tensor(raw))


def test_policy_scalar_metrics_cannot_overwrite_core_metrics(monkeypatch):
    log_probs = torch.zeros(2)
    old_log_probs = torch.zeros_like(log_probs)
    advantages = torch.ones_like(log_probs)
    _stub_policy_forward(monkeypatch, log_probs)
    monkeypatch.setattr(
        loss_module,
        "compute_policy_loss_for",
        lambda *_args, **_kwargs: (
            torch.zeros_like(log_probs),
            torch.zeros_like(log_probs),
            {"loss": torch.tensor(1.0)},
        ),
    )
    args = _args(calculate_per_token_loss=True)
    batch = {
        "advantages": advantages,
        "log_probs": [old_log_probs],
        "response_lengths": [2],
        "total_lengths": [3],
        "unconcat_tokens": [torch.arange(3)],
        "loss_masks": [torch.ones(2)],
    }

    with pytest.raises(ValueError, match="would overwrite existing metrics.*loss"):
        loss_module.policy_loss_function(args, batch, torch.empty(1, 1, 1), lambda values: values.sum())


@pytest.mark.parametrize(
    ("calculate_per_token_loss", "expected"),
    [
        pytest.param(False, (100.0 + 1.0) / 2, id="sample-weighted"),
        pytest.param(True, (100 * 100.0 + 10 * 1.0) / 110, id="token-weighted"),
    ],
)
def test_m2po_metrics_are_weighted_local_microbatch_statistics(monkeypatch, calculate_per_token_loss, expected):
    """Document that M2 diagnostics are not pooled by harmful-token count."""
    local_m2_values = iter((100.0, 1.0))

    def fake_policy_loss(*_args, **kwargs):
        zeros = torch.zeros_like(kwargs["ppo_kl"])
        return zeros, zeros, {"ppo_kl_m2_before": torch.tensor(next(local_m2_values))}

    monkeypatch.setattr(loss_module, "compute_policy_loss_for", fake_policy_loss)
    args = _args(calculate_per_token_loss=calculate_per_token_loss)
    metric_numerators = []
    token_denominator = 0

    for response_length in (100, 10):
        log_probs = torch.zeros(response_length)
        _stub_policy_forward(monkeypatch, log_probs)
        batch = {
            "advantages": torch.ones(response_length),
            "log_probs": [torch.zeros(response_length)],
            "response_lengths": [response_length],
            "total_lengths": [response_length + 1],
            "unconcat_tokens": [torch.arange(response_length + 1)],
            "loss_masks": [torch.ones(response_length)],
            "dynamic_cp_size": 1,
            "dynamic_cp_rank": 0,
        }
        _, _, logging = loss_module.loss_function(
            args,
            batch,
            num_microbatches=2,
            logits=torch.empty(1, 1, 1),
        )
        metric_index = logging["keys"].index("ppo_kl_m2_before") + 1
        metric_numerators.append(logging["values"][metric_index])
        token_denominator += response_length

    denominator = token_denominator if calculate_per_token_loss else args.global_batch_size
    actual = torch.stack(metric_numerators).sum().item() / denominator
    assert actual == pytest.approx(expected)
