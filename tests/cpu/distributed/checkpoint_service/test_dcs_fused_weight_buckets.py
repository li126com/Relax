# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""DCS bucket failures release the update lock without hiding the original
error."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture()
def dcs_update_bucket():
    path = Path(__file__).resolve().parents[4] / "relax/distributed/checkpoint_service/backends/device_direct.py"
    tree = ast.parse(path.read_text())
    backend = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeviceDirectBackend")
    method = next(
        node
        for node in backend.body
        if isinstance(node, ast.FunctionDef) and node.name == "_update_bucket_weights_from_distributed"
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("release_fails", [False, True])
def test_http_failure_releases_dcs_lock_preserves_error_and_does_not_advance_progress(
    dcs_update_bucket, release_fails
):
    calls = []
    original_error = RuntimeError("HTTP 400: partial weights")

    def get(value):
        if value == "http":
            raise original_error
        if value == "release":
            calls.append("release")
            if release_fails:
                raise RuntimeError("lock actor stopped")
        return value

    namespace = dcs_update_bucket
    namespace.update(
        ray=SimpleNamespace(get=get),
        dist=SimpleNamespace(broadcast=lambda *args, **kwargs: SimpleNamespace(wait=lambda: None)),
        logger=logging.getLogger(__name__),
    )
    state = SimpleNamespace(
        _group_name="slime-pp_7",
        _model_update_groups=object(),
        weight_version=1,
        lock=SimpleNamespace(
            acquire=SimpleNamespace(remote=lambda: True), release=SimpleNamespace(remote=lambda: "release")
        ),
        _batch_request=lambda *args: "http",
    )
    progress = SimpleNamespace(update=lambda count: calls.append("progress"))
    with pytest.raises(RuntimeError) as error:
        namespace["_update_bucket_weights_from_distributed"](state, [("weight", torch.ones(1))], progress)
    assert error.value is original_error
    assert calls == ["release"]


def test_distributed_bucket_requests_all_runners_without_duplicate_broadcast(dcs_update_bucket):
    events = []
    payloads = []
    weights = [("model.layers.0.weight", torch.ones(2)), ("mtp.layers.0.weight", torch.full((2,), 3.0))]

    def broadcast(tensor, src, *, group, async_op):
        assert src == 0 and group is state._model_update_groups and async_op
        events.append(("broadcast", tensor.clone()))
        return SimpleNamespace(wait=lambda: events.append("wait"))

    def request(endpoint, payload):
        assert endpoint == "/update_weights_from_distributed"
        payloads.append(payload)
        return "http"

    def get(value):
        if value == "http":
            assert events[-2:] == ["wait", "wait"]
            return [{"success": True}]
        if value == "release":
            events.append("release")
        return value

    namespace = dcs_update_bucket
    namespace.update(ray=SimpleNamespace(get=get), dist=SimpleNamespace(broadcast=broadcast))
    state = SimpleNamespace(
        _group_name="slime-pp_4",
        _model_update_groups=object(),
        weight_version=1,
        lock=SimpleNamespace(
            acquire=SimpleNamespace(remote=lambda: True), release=SimpleNamespace(remote=lambda: "release")
        ),
        _batch_request=request,
    )
    progress = SimpleNamespace(update=lambda count: events.append("progress"))
    namespace["_update_bucket_weights_from_distributed"](state, weights, progress)
    assert len(payloads) == 1
    assert payloads[0]["selector"] == "all"
    assert payloads[0]["names"] == [name for name, _ in weights]
    broadcasts = [event[1] for event in events if isinstance(event, tuple)]
    assert len(broadcasts) == len(weights)
    for actual, (_, expected) in zip(broadcasts, weights):
        torch.testing.assert_close(actual, expected)
    assert events[-2:] == ["release", "progress"]


@pytest.mark.parametrize("release_fails", [False, True])
def test_receiver_rejection_releases_dcs_lock_without_advancing_progress(dcs_update_bucket, release_fails):
    calls = []

    def get(value):
        if value == "http":
            return [{"success": True}, {"success": False, "message": "draft weight shape mismatch"}]
        if value == "release":
            calls.append("release")
            if release_fails:
                raise RuntimeError("lock actor stopped")
        return value

    namespace = dcs_update_bucket
    namespace.update(
        ray=SimpleNamespace(get=get),
        dist=SimpleNamespace(broadcast=lambda *args, **kwargs: SimpleNamespace(wait=lambda: None)),
        logger=logging.getLogger(__name__),
    )
    state = SimpleNamespace(
        _group_name="slime-pp_4",
        _model_update_groups=object(),
        weight_version=1,
        lock=SimpleNamespace(
            acquire=SimpleNamespace(remote=lambda: True), release=SimpleNamespace(remote=lambda: "release")
        ),
        _batch_request=lambda *args: "http",
    )
    progress = SimpleNamespace(update=lambda count: calls.append("progress"))
    with pytest.raises(RuntimeError, match="SGLang rejected distributed weights.*draft weight shape mismatch"):
        namespace["_update_bucket_weights_from_distributed"](state, [("mtp.weight", torch.ones(1))], progress)
    assert calls == ["release"]
