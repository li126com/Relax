# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Initialize resident async training's gradient communicators before activations."""

from argparse import Namespace
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

if TYPE_CHECKING:
    from megatron.core.optimizer.optimizer import MegatronOptimizer


def warmup_optimizer_grad_stats(args: Namespace, optimizer: "MegatronOptimizer") -> None:
    """Initialize actual optimizer norm groups before resident training fills the cache.

    Norm reductions can use a different NCCL group from DDP gradient reductions,
    even when their members coincide. Reduce temporary scalars on the optimizer's
    own groups; do not inspect gradients or invoke optimizer methods that update
    parameters, state, or gradient-norm bookkeeping.
    """
    if not (
        getattr(args, "fully_async", False)
        and not getattr(args, "hybrid", False)
        and not getattr(args, "offload_train", False)
        and getattr(args, "use_distributed_optimizer", False)
    ) or getattr(optimizer, "_relax_grad_stats_warmed_up", False):
        return

    pending = [optimizer]
    groups = []
    while pending:
        current = pending.pop(0)
        children = getattr(current, "chained_optimizers", None)
        if children is not None:
            pending[0:0] = children
            continue
        group = current.get_grad_stats_parallel_group()
        # MCore permits None to mean WORLD. Keep that meaning while passing an
        # explicit group to the collective.
        if group is None:
            group = dist.group.WORLD
        if group is None:
            raise RuntimeError("Optimizer norm warmup requires an initialized gradient statistics process group")
        if all(group is not previous for previous in groups):
            groups.append(group)

    with torch.no_grad():
        scalar = torch.zeros(1, dtype=torch.float32, device=torch.device("cuda", torch.cuda.current_device()))
        for group in groups:
            dist.all_reduce(scalar, op=dist.ReduceOp.SUM, group=group)
        # Initialization boundary only: finish NCCL allocation before activations.
        torch.cuda.current_stream().synchronize()
    optimizer._relax_grad_stats_warmed_up = True
    logger.info("Initialized %d optimizer gradient norm groups before fully-async training", len(groups))


def warmup_distributed_optimizer_grad_sync(args: Namespace, model: Sequence[torch.nn.Module]) -> None:
    """Reduce already-zeroed real DDP buffers once, preserving first-backward calibration.

    Call immediately after zero_grad_buffer/optimizer.zero_grad and before the
    first forward. Using the real buffers initializes the same process groups,
    dtypes and collective sizes as training, before variable-shape activation
    caches can leave too little device memory for NCCL's own allocations.
    Resident fully-async actors keep these groups for the model's lifetime.
    """
    if not (
        getattr(args, "fully_async", False)
        and not getattr(args, "hybrid", False)
        and not getattr(args, "offload_train", False)
        and getattr(args, "use_distributed_optimizer", False)
        and getattr(args, "overlap_grad_reduce", False)
    ):
        return

    for model_chunk in model:
        if getattr(model_chunk, "_relax_grad_sync_warmed_up", False):
            continue
        groups = model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups
        for group in groups:
            if not (
                group.is_first_batch
                and not group.per_param_grad_ready_counts
                and not group.golden_per_param_grad_ready_counts
                and group.grad_reduce_handle is None
                and group.param_gather_handle is None
                and not group.grad_reduce_finished
            ):
                raise RuntimeError("Gradient communicator warmup requires untouched, idle first-batch DDP state")

        # These methods use DDP's actual DistOpt process groups, including the
        # expert groups. No backward hooks or optimizer step are run here.
        model_chunk.start_grad_sync()
        model_chunk.finish_grad_sync()
        # An empty ready-count map does not freeze MCore's golden counts on
        # reset: the first real backward still performs that calibration.
        model_chunk.zero_grad_buffer()
        model_chunk._relax_grad_sync_warmed_up = True
        logger.info("Initialized %d gradient bucket groups before the first fully-async forward", len(groups))
