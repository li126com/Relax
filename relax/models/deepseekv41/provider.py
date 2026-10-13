# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from dataclasses import dataclass, field
from typing import Any

from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.core.models.gpt import GPTModel


@dataclass
class DeepSeekV41ModelProvider(MLAModelProvider):
    """V4.1 architecture metadata; execution requires its own model/layer spec."""

    experimental_attention_variant: str | None = None
    mtp_num_layers: int | None = None
    moe_n_hash_layers: int = 0
    o_groups: int = 8
    o_lora_rank: int = 1024
    v41_hf_config: dict[str, Any] = field(default_factory=dict)
    v41_kv_source_layer_ids: tuple[int, ...] = ()
    v41_index_source_layer_ids: tuple[int, ...] = ()
    v41_candidate_source_layer_id: int = -1
    v41_candidate_topk_blocks: int = 0
    v41_candidate_block_size: int = 0
    v41_engram_layer_ids: tuple[int, ...] = ()
    v41_engram_num_embeddings: tuple[int, ...] = ()
    v41_engram_max_ngram_size: int = 0
    v41_engram_vocab_size: int = 0
    v41_engram_n_heads: int = 0
    v41_engram_head_dim: int = 0
    v41_engram_pad_token_id: int = 0
    v41_engram_compressed_vocab_size: int = 0
    v41_hc_eps: float = 1e-6

    def provide(
        self, pre_process: bool | None = None, post_process: bool | None = None, vp_stage: int | None = None
    ) -> GPTModel:
        # TODO(agent): Port the V4.1 execution spec, instance-owned cross-layer
        # state and frozen CPU-backed Engram tables before enabling construction.
        raise NotImplementedError(
            "DeepSeek V4.1 Megatron execution is not ported yet; only Bridge configuration and weight conversion "
            "are available. The upstream V4/MLA model is not a compatible execution fallback."
        )
