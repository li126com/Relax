# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Policy loss selection must come from the registry."""

import ast
import dataclasses
import math
import pathlib
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from relax.algorithms.policy import POLICY_LOSS_FNS, compute_policy_loss_for  # noqa: E402
from relax.algorithms.spec import ALGORITHM_SPECS, get_algorithm, list_algorithm_names  # noqa: E402
from relax.utils.training.ppo_utils import (  # noqa: E402
    _solve_tau_from_sorted_delta2,
    compute_cispo_loss,
    compute_m2po_loss,
    compute_policy_loss,
    compute_sapo_loss,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LOSS_PATH = REPO_ROOT / "relax" / "backends" / "megatron" / "loss.py"
SERVE_PATH = REPO_ROOT / "relax" / "components" / "advantages.py"
NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


def _args(estimator, **overrides):
    base = dict(
        advantage_estimator=estimator,
        eps_clip=0.2,
        eps_clip_high=0.2,
        sapo_tau_pos=1.0,
        sapo_tau_neg=1.05,
        m2po_kl2_budget=0.01,
        m2po_miniclip_low=0.3,
        m2po_miniclip_high=0.5,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _tensors():
    torch.manual_seed(0)
    return torch.randn(8), torch.randn(8), torch.randn(8)


def _legacy_solve_tau_from_sorted_delta2(sorted_delta2, target_sum):
    """Frozen pre-refactor implementation used as a numerical oracle."""
    n = sorted_delta2.numel()
    total = float(sorted_delta2.sum().item())
    if target_sum >= total - 1e-12:
        return 100000.0, total / n
    if target_sum <= 1e-12:
        return 0.0, 0.0
    csum = torch.cumsum(sorted_delta2, dim=0)
    for k in range(n):
        left_sum = float(csum[k].item())
        rest = n - k - 1
        m2 = sorted_delta2[k].item() - 1e-12
        if m2 * rest + left_sum >= target_sum - 1e-12:
            if k == 0:
                return 0.0, float(csum[-1].item()) / n
            m2_after = (sorted_delta2[k - 1].item() * (rest + 1) + float(csum[k - 1].item())) / n
            return max(sorted_delta2[k - 1].item() - 1e-12, 0.0) ** 0.5, m2_after
    return 100000.0, total / n


def _legacy_compute_m2po_loss(ppo_kl, advantages, kl2_budget, miniclip_low, miniclip_high):
    """Frozen M2PO loss from main before the registry refactor."""
    ratio = (-ppo_kl).exp()
    pos_harmful = (advantages > 1e-12) & (ratio > 1.0 + 1e-12)
    neg_harmful = (advantages < -1e-12) & (ratio < 1.0 - 1e-12)
    tr_delta_sq = ppo_kl[pos_harmful | neg_harmful].pow(2)
    n = tr_delta_sq.numel()
    if n == 0:
        clip_low, clip_high, m2_now, m2_after = 0.0, 100000.0, 0.0, 0.0
    else:
        m2_now = float(tr_delta_sq.sum().detach().item() / n)
        if m2_now <= kl2_budget + 1e-12:
            clip_low, clip_high, m2_after = 0.0, 100000.0, m2_now
        else:
            sorted_delta2, _ = torch.sort(tr_delta_sq)
            tau, m2_after = _legacy_solve_tau_from_sorted_delta2(sorted_delta2, kl2_budget * float(n))
            clip_low, clip_high = math.exp(-tau), math.exp(tau)

    eps_low = max(1.0 - clip_low, miniclip_low)
    eps_high = max(clip_high - 1.0, miniclip_high)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * ratio.clamp(1.0 - eps_low, 1.0 + eps_high)
    pg_loss = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = (pg_losses2 > pg_losses1).float()
    return pg_loss, clipfrac, m2_now, m2_after, eps_low, eps_high


# ---------------- registry ----------------


def test_every_registered_loss_is_reachable_from_some_spec():
    """No dead entries: a loss no spec names can never be dispatched to.

    The inverse direction is `test_every_spec_policy_loss_id_is_registered`.
    Together they pin the table to exactly what the registry uses, which is
    what a hard-coded inventory of names was doing before -- except this
    version fails for a reason instead of failing on every addition.
    """
    referenced = {get_algorithm(name).policy_loss_fn for name in list_algorithm_names()}
    assert set(POLICY_LOSS_FNS) == referenced


def test_every_spec_policy_loss_id_is_registered():
    for name in list_algorithm_names():
        assert get_algorithm(name).policy_loss_fn in POLICY_LOSS_FNS


# ---------------- adapters match their kernels ----------------


def test_ppo_clip_matches_the_underlying_kernel():
    log_probs, ppo_kl, advantages = _tensors()
    args = _args("grpo")
    got = compute_policy_loss_for(args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])


def test_sapo_matches_the_underlying_kernel():
    log_probs, ppo_kl, advantages = _tensors()
    got = compute_policy_loss_for(_args("sapo"), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = compute_sapo_loss(ppo_kl=ppo_kl, advantages=advantages, tau_pos=1.0, tau_neg=1.05)
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])


def test_cispo_matches_the_underlying_kernel():
    log_probs, ppo_kl, advantages = _tensors()
    got = compute_policy_loss_for(_args("cispo"), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = compute_cispo_loss(
        log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages, eps_clip=0.2, eps_clip_high=0.2
    )
    assert torch.equal(got[0], want[0])
    assert torch.equal(got[1], want[1])


def test_m2po_matches_the_underlying_kernel_and_names_its_scalar_metrics():
    log_probs, ppo_kl, advantages = _tensors()
    args = _args("m2po", m2po_kl2_budget=0.02, m2po_miniclip_low=0.25, m2po_miniclip_high=0.4)

    got_loss, got_clipfrac, got_metrics = compute_policy_loss_for(
        args,
        log_probs=log_probs,
        ppo_kl=ppo_kl,
        advantages=advantages,
    )
    want_loss, want_clipfrac, *want_metrics = compute_m2po_loss(
        ppo_kl=ppo_kl,
        advantages=advantages,
        kl2_budget=args.m2po_kl2_budget,
        miniclip_low=args.m2po_miniclip_low,
        miniclip_high=args.m2po_miniclip_high,
    )

    assert torch.equal(got_loss, want_loss)
    assert torch.equal(got_clipfrac, want_clipfrac)
    assert list(got_metrics) == list(get_algorithm("m2po").policy_scalar_metric_names)
    for value, expected in zip(got_metrics.values(), want_metrics, strict=True):
        assert torch.equal(value, torch.as_tensor(expected, device=ppo_kl.device, dtype=torch.float32))
        assert value.shape == ()
        assert value.dtype == torch.float32
        assert value.requires_grad is False


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("values", "target_sum"),
    [
        ([1.0, 4.0, 9.0], 14.0),  # no clipping
        ([1.0, 4.0, 9.0], 0.0),  # clip everything
        ([1.0, 4.0, 9.0], 1.0),  # first breakpoint (k=0)
        ([1.0, 4.0, 9.0], 3.0),  # Python-float 1e-12 changes the selected breakpoint
        ([1.0, 4.0, 9.0], 8.0),  # interior breakpoint (k>0)
        ([1.0, 4.0, 9.0], 10.0),
        ([1.0, 1.0, 4.0, 4.0], 4.0),  # repeated values at the 1e-12 boundary
        ([0.1, 0.2, 0.4], 0.15),  # sum() and cumsum() round differently at k=0
        ([0.0, 0.0], 0.0),  # preserves the legacy branch order
    ],
)
def test_m2po_tau_solver_preserves_legacy_boundaries(dtype, values, target_sum):
    sorted_delta2 = torch.tensor(values, dtype=dtype)
    expected_tau, expected_m2 = _legacy_solve_tau_from_sorted_delta2(sorted_delta2, target_sum)
    actual_tau, actual_m2 = _solve_tau_from_sorted_delta2(sorted_delta2, target_sum)

    assert actual_tau == expected_tau
    assert actual_m2 == expected_m2


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_m2po_tau_solver_preserves_legacy_random_cases(dtype):
    generator = torch.Generator().manual_seed(2026)
    for size in (1, 2, 7, 31):
        sorted_delta2 = torch.sort(torch.rand(size, generator=generator).square().to(dtype=dtype)).values
        total = float(sorted_delta2.sum().item())
        for fraction in (0.0, 0.01, 0.25, 0.75, 1.0, 1.1):
            target_sum = total * fraction
            expected_tau, expected_m2 = _legacy_solve_tau_from_sorted_delta2(sorted_delta2, target_sum)
            actual_tau, actual_m2 = _solve_tau_from_sorted_delta2(sorted_delta2, target_sum)
            assert actual_tau == expected_tau
            assert actual_m2 == expected_m2


@pytest.mark.skipif(not NPU_AVAILABLE, reason="requires an Ascend NPU")
@pytest.mark.parametrize(
    ("ppo_kl", "advantages", "budget"),
    [
        ([0.0, 0.0], [1.0, -1.0], 0.02),
        ([-0.8, -0.4, 0.2, 0.7], [1.0, 1.0, -1.0, -1.0], 0.02),
        ([-1.0, -2.0, -3.0], [1.0, 1.0, 1.0], 1.0),
    ],
)
def test_m2po_npu_dispatch_preserves_legacy_loss_and_gradients(ppo_kl, advantages, budget):
    cpu_ppo_kl = torch.tensor(ppo_kl, dtype=torch.float32, requires_grad=True)
    cpu_advantages = torch.tensor(advantages, dtype=torch.float32)
    expected = _legacy_compute_m2po_loss(cpu_ppo_kl, cpu_advantages, budget, 0.3, 0.5)
    expected[0].sum().backward()

    npu_ppo_kl = cpu_ppo_kl.detach().to("npu").requires_grad_()
    npu_advantages = cpu_advantages.to("npu")
    actual = compute_policy_loss_for(
        _args("m2po", m2po_kl2_budget=budget),
        log_probs=-npu_ppo_kl,
        ppo_kl=npu_ppo_kl,
        advantages=npu_advantages,
    )

    torch.testing.assert_close(actual[0].cpu(), expected[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual[1].cpu(), expected[1], rtol=0, atol=0)
    for actual_metric, expected_metric in zip(actual[2].values(), expected[2:], strict=True):
        assert actual_metric.device.type == "npu"
        torch.testing.assert_close(actual_metric.cpu(), torch.tensor(expected_metric), rtol=1e-5, atol=1e-6)
    actual[0].sum().backward()
    torch.testing.assert_close(npu_ppo_kl.grad.cpu(), cpu_ppo_kl.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("ppo_kl", "advantages", "budget"),
    [
        pytest.param([-0.8, -0.4, -0.1, 0.2, 0.7], [1.0, 1.0, -1.0, -1.0, -1.0], 0.02, id="mixed-signs"),
        pytest.param([-0.8, -0.4, 0.2, 0.7], [1.0, 1.0, -1.0, -1.0], 0.0, id="zero-budget"),
        pytest.param([-1.0, -2.0, -3.0], [1.0, 1.0, 1.0], 1.0, id="exact-breakpoint"),
        pytest.param([-1.0, -1.0, -2.0, -2.0], [1.0, 1.0, 1.0, 1.0], 1.0, id="repeated-breakpoint"),
    ],
)
def test_m2po_dispatch_preserves_legacy_clipped_loss_metrics_and_gradients(dtype, ppo_kl, advantages, budget):
    _assert_m2po_dispatch_matches_legacy(dtype, ppo_kl, advantages, budget)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("ppo_kl", "advantages", "budget"),
    [
        pytest.param([0.8, -0.4, 0.0], [1.0, -1.0, 0.0], 0.02, id="no-harmful-tokens"),
        pytest.param([-0.1, -0.2, 0.1], [1.0, 1.0, -1.0], 1.0, id="within-budget"),
    ],
)
def test_m2po_dispatch_preserves_legacy_unclipped_loss_metrics_and_gradients(dtype, ppo_kl, advantages, budget):
    _assert_m2po_dispatch_matches_legacy(dtype, ppo_kl, advantages, budget)


def _assert_m2po_dispatch_matches_legacy(dtype, ppo_kl, advantages, budget):
    actual_ppo_kl = torch.tensor(ppo_kl, dtype=dtype, requires_grad=True)
    expected_ppo_kl = actual_ppo_kl.detach().clone().requires_grad_()
    advantages = torch.tensor(advantages, dtype=dtype)
    actual = compute_policy_loss_for(
        _args("m2po", m2po_kl2_budget=budget),
        log_probs=-actual_ppo_kl,
        ppo_kl=actual_ppo_kl,
        advantages=advantages,
    )
    expected = _legacy_compute_m2po_loss(expected_ppo_kl, advantages, budget, 0.3, 0.5)
    actual[0].sum().backward()
    expected[0].sum().backward()

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    assert list(actual[2]) == list(get_algorithm("m2po").policy_scalar_metric_names)
    for actual_metric, expected_metric in zip(actual[2].values(), expected[2:], strict=True):
        torch.testing.assert_close(actual_metric, torch.tensor(expected_metric, dtype=torch.float32), rtol=0, atol=0)
        assert actual_metric.requires_grad is False
    torch.testing.assert_close(actual_ppo_kl.grad, expected_ppo_kl.grad, rtol=0, atol=0)


@pytest.mark.parametrize("budget", [0.02, 1.0])
def test_m2po_dispatch_preserves_legacy_float16_unbounded_clamp_limit(budget):
    """The legacy no-clipping upper bound exceeds float16's finite range."""
    ppo_kl = torch.tensor([0.0, -0.1], dtype=torch.float16)
    advantages = torch.ones_like(ppo_kl)
    with pytest.raises(RuntimeError, match="overflow"):
        _legacy_compute_m2po_loss(ppo_kl, advantages, budget, 0.3, 0.5)
    with pytest.raises(RuntimeError, match="overflow"):
        compute_policy_loss_for(
            _args("m2po", m2po_kl2_budget=budget),
            log_probs=-ppo_kl,
            ppo_kl=ppo_kl,
            advantages=advantages,
        )


def test_dispatch_rejects_scalar_metric_count_drift(monkeypatch):
    log_probs, ppo_kl, advantages = _tensors()
    monkeypatch.setitem(
        POLICY_LOSS_FNS,
        "ppo_clip",
        lambda *args, **kwargs: (torch.zeros_like(ppo_kl), torch.zeros_like(ppo_kl), 1.0),
    )
    with pytest.raises(ValueError, match="returned 1 scalar metrics, but its spec declares 0"):
        compute_policy_loss_for(_args("grpo"), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)


def test_dispatch_rejects_a_non_scalar_declared_metric(monkeypatch):
    log_probs, ppo_kl, advantages = _tensors()
    monkeypatch.setitem(
        ALGORITHM_SPECS,
        "grpo",
        dataclasses.replace(get_algorithm("grpo"), policy_scalar_metric_names=("probe",)),
    )
    monkeypatch.setitem(
        POLICY_LOSS_FNS,
        "ppo_clip",
        lambda *args, **kwargs: (
            torch.zeros_like(ppo_kl),
            torch.zeros_like(ppo_kl),
            torch.ones(2),
        ),
    )
    with pytest.raises(ValueError, match="metric 'probe' must be scalar"):
        compute_policy_loss_for(_args("grpo"), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (torch.tensor(1.25, dtype=torch.float64), 1.25),
        (torch.tensor(2, dtype=torch.int64), 2.0),
        (torch.tensor(True), 1.0),
    ],
)
def test_dispatch_normalizes_real_scalar_metrics_to_float32(monkeypatch, value, expected):
    log_probs, ppo_kl, advantages = _tensors()
    monkeypatch.setitem(
        ALGORITHM_SPECS,
        "grpo",
        dataclasses.replace(get_algorithm("grpo"), policy_scalar_metric_names=("probe",)),
    )
    monkeypatch.setitem(
        POLICY_LOSS_FNS,
        "ppo_clip",
        lambda *args, **kwargs: (torch.zeros_like(ppo_kl), torch.zeros_like(ppo_kl), value),
    )

    _, _, metrics = compute_policy_loss_for(_args("grpo"), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)

    assert metrics["probe"].dtype == torch.float32
    assert metrics["probe"] == expected


def test_dispatch_rejects_a_complex_scalar_metric(monkeypatch):
    log_probs, ppo_kl, advantages = _tensors()
    monkeypatch.setitem(
        ALGORITHM_SPECS,
        "grpo",
        dataclasses.replace(get_algorithm("grpo"), policy_scalar_metric_names=("probe",)),
    )
    monkeypatch.setitem(
        POLICY_LOSS_FNS,
        "ppo_clip",
        lambda *args, **kwargs: (
            torch.zeros_like(ppo_kl),
            torch.zeros_like(ppo_kl),
            torch.tensor(1 + 2j),
        ),
    )
    with pytest.raises(ValueError, match="metric 'probe' must be real-valued"):
        compute_policy_loss_for(_args("grpo"), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)


def test_sapo_defaults_when_args_lack_tau_fields():
    log_probs, ppo_kl, advantages = _tensors()
    args = SimpleNamespace(advantage_estimator="sapo", eps_clip=0.2, eps_clip_high=0.2)
    got = compute_policy_loss_for(args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = compute_sapo_loss(ppo_kl=ppo_kl, advantages=advantages, tau_pos=1.0, tau_neg=1.05)
    assert torch.equal(got[0], want[0])


def test_sapo_taus_are_read_from_args():
    log_probs, ppo_kl, advantages = _tensors()
    args = _args("sapo", sapo_tau_pos=2.0, sapo_tau_neg=3.0)
    got = compute_policy_loss_for(args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    want = compute_sapo_loss(ppo_kl=ppo_kl, advantages=advantages, tau_pos=2.0, tau_neg=3.0)
    assert torch.equal(got[0], want[0])


@pytest.mark.parametrize("estimator", ["grpo", "gspo", "ppo", "reinforce_plus_plus"])
def test_ppo_clip_family_share_one_loss(estimator):
    log_probs, ppo_kl, advantages = _tensors()
    reference = compute_policy_loss_for(_args("grpo"), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    actual = compute_policy_loss_for(_args(estimator), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    assert torch.equal(reference[0], actual[0])


@pytest.mark.parametrize(
    "estimator",
    ["grpo", "gspo", "sapo", "cispo", "rloo", "ppo", "reinforce_plus_plus", "reinforce_plus_plus_baseline"],
)
def test_policy_losses_without_scalar_diagnostics_return_an_empty_mapping(estimator):
    log_probs, ppo_kl, advantages = _tensors()
    _, _, metrics = compute_policy_loss_for(
        _args(estimator), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages
    )
    assert metrics == {}


# ---------------- call sites no longer branch on names ----------------


def _is_raw_estimator_expression(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute):
        return node.attr == "advantage_estimator"
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "advantage_estimator"
    )


def _contains_string_literal(node: ast.AST) -> bool:
    return any(isinstance(child, ast.Constant) and isinstance(child.value, str) for child in ast.walk(node))


def _name_checks(source: str) -> list[str]:
    """Find raw estimator-vs-literal comparisons without matching spec
    fields."""
    lines = source.splitlines()
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        raw_estimator_positions = {i for i, operand in enumerate(operands) if _is_raw_estimator_expression(operand)}
        if raw_estimator_positions and any(
            _contains_string_literal(operand) for i, operand in enumerate(operands) if i not in raw_estimator_positions
        ):
            found.append(lines[node.lineno - 1].strip())
    return found


def test_loss_py_no_longer_branches_on_estimator_names():
    found = _name_checks(LOSS_PATH.read_text(encoding="utf-8"))
    assert found == [], f"loss.py still compares the estimator to literal names: {found}"


def test_serve_path_no_longer_branches_on_estimator_names():
    found = _name_checks(SERVE_PATH.read_text(encoding="utf-8"))
    assert found == [], f"components/advantages.py still compares the estimator to literal names: {found}"


def test_the_name_check_pattern_catches_every_spelling():
    """Guard the guard: the previous version of this test was blind to `in.

    {`.
    """
    for spelling in (
        'if args.advantage_estimator == "gspo":',
        'if args.advantage_estimator != "ppo":',
        'if args.advantage_estimator in ["grpo", "gspo"]:',
        'x = args.advantage_estimator in {"reinforce_plus_plus"}',
        'if self.config.advantage_estimator in ("ppo",):',
        'if getattr(args, "advantage_estimator", None) == "m2po":',
        'if "rloo" == args.advantage_estimator:',
    ):
        source = f"{spelling}\n    pass" if spelling.startswith("if ") else spelling
        assert _name_checks(source) == [spelling], spelling
    for allowed in (
        "spec = get_algorithm(args.advantage_estimator)",
        'if get_algorithm(args.advantage_estimator).advantage_normalization == "token_global":',
    ):
        source = f"{allowed}\n    pass" if allowed.startswith("if ") else allowed
        assert _name_checks(source) == [], allowed


def test_both_paths_delegate_to_the_shared_estimator():
    for path in (LOSS_PATH, SERVE_PATH):
        src = path.read_text(encoding="utf-8")
        assert "from relax.algorithms.advantages import" in src, f"{path.name} does not use the shared estimator"


def test_loss_py_reads_kl_level_and_full_log_probs_from_the_spec():
    src = LOSS_PATH.read_text(encoding="utf-8")
    assert 'kl_level == "sequence"' in src
    assert "needs_full_log_probs" in src


def test_neither_call_site_still_raises_not_implemented_for_estimators():
    for path in (LOSS_PATH, SERVE_PATH):
        src = path.read_text(encoding="utf-8")
        assert "advantage_estimator {" not in src, f"{path.name} still formats an estimator into NotImplementedError"
