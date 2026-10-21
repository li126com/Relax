# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Independently compare pre-update train/rollout LPs from existing ONLINE
dumps.

Init hook: scripts.tools.deepseek_v41_logprob_compare.initialize_reduced_experiment
CLI: python -m scripts.tools.deepseek_v41_logprob_compare --dump-details DIR
     --output-prefix PREFIX --log-file LOG --rollout-ids 0 1
Run postprocessing on the training host with its installed Torch, not locally.
"""

import argparse
import ast
import csv
import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def initialize_reduced_experiment(args: argparse.Namespace) -> None:
    """Apply the reduced experiment's DeepEP setting before recording
    metadata."""
    import torch

    if args.deterministic_mode:
        torch.utils.deterministic.fill_uninitialized_memory = False
    capture_parallel_metadata(args)


def capture_parallel_metadata(args: argparse.Namespace) -> None:
    """Existing Megatron init hook; records facts only, without changing
    execution."""
    import torch
    import torch.distributed as dist
    from megatron.core import mpu

    if not args.dump_details:
        raise ValueError("LP diagnostic init hook requires --dump-details.")
    metadata = {
        "rank": dist.get_rank(group=dist.group.WORLD),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
        "tp_rank": mpu.get_tensor_model_parallel_rank(),
        "tp_size": mpu.get_tensor_model_parallel_world_size(),
        "pp_rank": mpu.get_pipeline_model_parallel_rank(),
        "pp_size": mpu.get_pipeline_model_parallel_world_size(),
        "cp_rank": mpu.get_context_parallel_rank(),
        "cp_size": mpu.get_context_parallel_world_size(),
        "dp_rank": mpu.get_data_parallel_rank(with_context_parallel=False),
        "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
        "config": {
            name: getattr(args, name, None)
            for name in (
                "qkv_format",
                "allgather_cp",
                "cp_partition_mode",
                "data_pad_size_multiplier",
                "dynamic_context_parallel",
                "calculate_per_token_loss",
                "rollout_temperature",
                "rollout_seed",
                "seed",
                "rollout_batch_size",
                "n_samples_per_prompt",
                "use_rollout_logprobs",
                "get_mismatch_metrics",
                "use_rollout_routing_replay",
                "hf_checkpoint",
                "prompt_data",
                "recompute_granularity",
                "moe_token_dispatcher_type",
            )
        },
    }
    path = Path(args.dump_details) / "parallel_metadata" / f"rank_{metadata['rank']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != metadata:
            raise ValueError(f"Stale parallel metadata: {path}; use a new experiment directory.")
        return
    with path.open("x") as output:
        json.dump(metadata, output, indent=2)


def _native_cpu_helpers(torch: Any) -> dict[str, Any]:
    """Execute unchanged CPU helpers without importing backend initialization
    patches."""
    source = Path(__file__).resolve().parents[2] / "relax/backends/megatron/cp_utils.py"
    names = {
        "get_logits_and_tokens_offset_with_cp",
        "slice_log_prob_with_cp",
        "_slice_loss_mask_with_cp",
        "get_sum_of_sample_mean",
    }
    functions = [
        node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != names:
        raise ValueError("Native CP helpers changed; diagnostic must be reviewed.")
    namespace = {"torch": torch, "Callable": Callable}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def _logged_metrics(path: Path | None) -> dict[int, dict[str, Any]]:
    result = {}
    if path is not None:
        with path.open(errors="replace") as source:
            for line in source:
                match = re.search(r"rollout (\d+): (\{.*\})", line)
                if match:
                    result[int(match[1])] = ast.literal_eval(match[2])
    return result


def compare_rollout(root: Path, rollout_id: int, expected_samples: int, logged: dict[str, Any]) -> dict[str, Any]:
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError("Postprocessing must remain CPU-only.")
    native = _native_cpu_helpers(torch)
    rollout_path = root / "rollout_data" / f"{rollout_id}.pt"
    rollout = torch.load(rollout_path, map_location="cpu", weights_only=False)
    raw_samples = {sample["index"]: sample for sample in rollout["samples"]}
    assert rollout["rollout_id"] == rollout_id
    assert len(raw_samples) == len(rollout["samples"]) == expected_samples
    shards: dict[int, dict[int, dict[str, Any]]] = {}
    selected = []
    native_rank_values = {"log_probs": [], "rollout_log_probs": []}
    config = None
    for meta_path in sorted((root / "parallel_metadata").glob("rank_*.json")):
        meta = json.loads(meta_path.read_text())
        if meta["tp_rank"] != 0 or meta["pp_rank"] != meta["pp_size"] - 1:
            continue
        current_config = meta["config"]
        assert current_config["qkv_format"] == "bshd" and not current_config["dynamic_context_parallel"]
        if config is not None:
            assert config == current_config
        config = current_config
        path = root / "train_data" / f"{rollout_id}_{meta['rank']}.pt"
        dump = torch.load(path, map_location="cpu", weights_only=False)
        assert dump["rollout_id"] == rollout_id and dump["rank"] == meta["rank"]
        rows = dump["samples"]
        for sample in rows:
            index = int(sample["sample_indices"])
            peers = shards.setdefault(index, {})
            assert meta["cp_rank"] not in peers
            peers[meta["cp_rank"]] = {"sample": sample, "metadata": meta}
        for field in native_rank_values:
            values = [torch.tensor(row[field], dtype=torch.float32) for row in rows]
            assert all(torch.isfinite(value).all() for value in values), (path, field)
            reducer = native["get_sum_of_sample_mean"](
                [row["total_lengths"] for row in rows],
                [row["response_lengths"] for row in rows],
                [torch.tensor(row["loss_masks"], dtype=torch.int32) for row in rows],
                qkv_format="bshd",
                max_seq_lens=[row["max_seq_lens"] for row in rows],
                dynamic_cp_size=meta["cp_size"],
                dynamic_cp_rank=meta["cp_rank"],
                sample_denoms=[row["sample_index_mask_sums"] for row in rows],
            )
            value = meta["cp_size"] * reducer(torch.cat(values)) * meta["dp_size"] / expected_samples
            native_rank_values[field].append(float(value))
        selected.append({"path": str(path), "metadata": meta})
    assert config is not None and set(shards) == set(raw_samples)
    totals = {"mask": 0, "abs": 0.0, "sample_abs": 0.0, "train": 0.0, "sg": 0.0}
    lengths, masks, padded, versions = Counter(), Counter(), Counter(), Counter()
    hashes = {name: hashlib.sha256() for name in ("tokens", "loss_masks", "rollout_log_probs")}
    removed = 0
    for index in sorted(shards):
        peers = shards[index]
        first = peers[0]["sample"]
        cp_size = peers[0]["metadata"]["cp_size"]
        assert set(peers) == set(range(cp_size))
        raw = raw_samples[index]
        total, response, padding = first["total_lengths"], first["response_lengths"], first["max_seq_lens"]
        assert total == len(raw["tokens"]) and response == raw["response_length"]
        assert first["tokens"] == raw["tokens"]
        expected_mask = raw["loss_mask"] if raw["loss_mask"] is not None else [1] * response
        expected_mask = expected_mask + [1] * (response - len(expected_mask))
        if raw["remove_sample"]:
            expected_mask = [0] * response
        assert first["loss_masks"] == expected_mask
        mask = torch.tensor(expected_mask, dtype=torch.float64)
        assert torch.isfinite(mask).all() and ((mask == 0) | (mask == 1)).all()
        train = torch.empty(response, dtype=torch.float32)
        sg = torch.empty(response, dtype=torch.float32)
        coverage = torch.zeros(response, dtype=torch.int32)
        original_sg = torch.tensor(raw["rollout_log_probs"], dtype=torch.float32)
        for cp_rank, peer in peers.items():
            sample = peer["sample"]
            for field in (
                "tokens",
                "loss_masks",
                "total_lengths",
                "response_lengths",
                "max_seq_lens",
                "sample_index_mask_sums",
            ):
                assert sample[field] == first[field], (index, cp_rank, field)
            positions = native["slice_log_prob_with_cp"](
                torch.arange(response),
                total,
                response,
                "bshd",
                padding,
                dynamic_cp_size=cp_size,
                dynamic_cp_rank=cp_rank,
            )
            local_train = torch.tensor(sample["log_probs"], dtype=torch.float32)
            local_sg = torch.tensor(sample["rollout_log_probs"], dtype=torch.float32)
            assert len(positions) == len(local_train) == len(local_sg)
            assert torch.isfinite(local_train).all() and torch.isfinite(local_sg).all()
            assert torch.equal(local_sg.view(torch.int32), original_sg[positions].view(torch.int32))
            train[positions], sg[positions] = local_train, local_sg
            coverage[positions] += 1
        assert (coverage == 1).all(), (rollout_id, index, "CP coverage")
        assert torch.isfinite(train).all() and torch.isfinite(sg).all()
        assert torch.equal(sg.view(torch.int32), original_sg.view(torch.int32))
        denominator = max(first["sample_index_mask_sums"], 1)
        assert first["sample_index_mask_sums"] == int(mask.sum())
        absolute = float(((train.double() - sg.double()).abs() * mask).sum())
        totals["mask"] += int(mask.sum())
        totals["abs"] += absolute
        totals["sample_abs"] += absolute / denominator
        totals["train"] += float((train.double() * mask).sum()) / denominator
        totals["sg"] += float((sg.double() * mask).sum()) / denominator
        lengths[response] += 1
        masks[int(mask.sum())] += 1
        padded[padding] += 1
        removed += int(raw["remove_sample"])
        versions.update(str(version) for version in raw["weight_versions"])
        for name, value in (
            ("tokens", torch.tensor(raw["tokens"])),
            ("loss_masks", mask.int()),
            ("rollout_log_probs", original_sg),
        ):
            hashes[name].update(str((index, str(value.dtype), tuple(value.shape))).encode())
            hashes[name].update(value.contiguous().view(torch.uint8).numpy().tobytes())
    assert totals["mask"] > 0
    train_mean, sg_mean = totals["train"] / expected_samples, totals["sg"] / expected_samples
    native_means = {field: sum(values) / len(values) for field, values in native_rank_values.items()}
    log_comparison = {}
    for field, value in native_means.items():
        observed = logged.get(f"rollout/{field}")
        log_comparison[field] = {
            "native_fp32_recomputed": value,
            "logged": observed,
            "within_1e_minus_6": abs(value - observed) <= 1e-6 if observed is not None else None,
        }
    assert not torch.cuda.is_initialized()
    return {
        "rollout_id": rollout_id,
        "display_step": rollout_id + 1,
        "samples": expected_samples,
        "response_tokens": sum(length * count for length, count in lengths.items()),
        "effective_answer_tokens": totals["mask"],
        "remove_sample_count": removed,
        "user_token_weighted_logprob_mae": totals["abs"] / totals["mask"],
        "sample_weighted_logprob_mae": totals["sample_abs"] / expected_samples,
        "token_mae_minus_sample_mae": totals["abs"] / totals["mask"] - totals["sample_abs"] / expected_samples,
        "official_sample_mean_train_log_prob": train_mean,
        "official_sample_mean_inference_log_prob": sg_mean,
        "official_absolute_difference_of_means": abs(train_mean - sg_mean),
        "native_fp32_absolute_difference_of_means": abs(native_means["log_probs"] - native_means["rollout_log_probs"]),
        "official_log_comparison": log_comparison,
        "response_lengths": dict(lengths),
        "effective_mask_sums": dict(masks),
        "padded_lengths": dict(padded),
        "rollout_weight_version_call_counts": dict(versions),
        "weight_version_scope": "Rollout per-call strings only; no per-token spans or train-side version field.",
        "input_sha256": {name: value.hexdigest() for name, value in hashes.items()},
        "validation": "Finite LPs; original tokens/effective masks/SG LPs match; unique complete CP response coverage.",
        "config": config,
        "source_rollout": str(rollout_path),
        "source_train": selected,
        "cuda_initialized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-details", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--rollout-ids", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--expected-samples", type=int, default=128)
    args = parser.parse_args()
    logged = _logged_metrics(args.log_file)
    rows = [
        compare_rollout(args.dump_details, step, args.expected_samples, logged.get(step, {}))
        for step in args.rollout_ids
    ]
    result = {
        "scoring": "ONLINE same-rollout pre-update train LPs; saved after training, not post-update scoring.",
        "user_mae_definition": "sum(effective_answer_mask * abs(train_LP - inference_LP)) / sum(mask)",
        "official_definition": "abs(sample_mean(train_LP) - sample_mean(inference_LP)); not token MAE.",
        "no_mae_threshold_assumed": True,
        "rows": rows,
    }
    json_path, csv_path = Path(f"{args.output_prefix}.json"), Path(f"{args.output_prefix}.csv")
    if json_path.exists() or csv_path.exists():
        raise FileExistsError("Output already exists; choose a new --output-prefix.")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    columns = (
        "display_step",
        "rollout_id",
        "samples",
        "response_tokens",
        "effective_answer_tokens",
        "remove_sample_count",
        "user_token_weighted_logprob_mae",
        "official_absolute_difference_of_means",
        "native_fp32_absolute_difference_of_means",
    )
    with csv_path.open("x", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows({name: row[name] for name in columns} for row in rows)
    logger.info("Independent ONLINE LP metrics written to %s and %s", json_path, csv_path)


if __name__ == "__main__":
    main()
