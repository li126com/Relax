# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Colocated weight-update logging must not free a paused TMS allocator's
blocks."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


_SOURCE = Path(__file__).resolve().parents[4] / "relax/backends/megatron/actor.py"


def _after_update_print() -> ast.Call:
    methods = [
        method
        for node in ast.parse(_SOURCE.read_text()).body
        if isinstance(node, ast.ClassDef)
        for method in node.body
        if isinstance(method, ast.FunctionDef) and method.name == "update_weights"
    ]
    assert len(methods) == 1
    calls = [
        node
        for node in ast.walk(methods[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print_memory"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "after update_weights"
    ]
    assert len(calls) == 1, "select update_weights, not the separate fully-async receiver API"
    return calls[0]


@pytest.mark.parametrize("uses_tms", [False, True])
@pytest.mark.parametrize("is_npu", [False, True])
def test_update_weights_prints_memory_without_clearing_paused_tms(uses_tms: bool, is_npu: bool) -> None:
    printed = []
    cleared = []

    def print_memory(message: str, *, clear_before_print: bool) -> None:
        assert isinstance(clear_before_print, bool)
        if clear_before_print:
            if uses_tms:
                raise RuntimeError("empty_cache attempted to free unmapped blocks in the paused TMS allocator")
            cleared.append("empty_cache")
        printed.append((message, clear_before_print))

    call = _after_update_print()
    module = ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=call)], type_ignores=[]))

    exec(
        compile(module, str(_SOURCE), "exec"),
        {
            "self": SimpleNamespace(_train_state_offloader=SimpleNamespace(uses_tms=uses_tms)),
            "device_utils": SimpleNamespace(is_npu_available=is_npu),
            "print_memory": print_memory,
        },
    )

    should_clear = not is_npu and not uses_tms
    assert printed == [("after update_weights", should_clear)]
    assert cleared == (["empty_cache"] if should_clear else [])
