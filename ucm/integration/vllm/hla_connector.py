import copy
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from ucm.integration.vllm.device import create_device
from ucm.integration.vllm.request_hasher import RequestHasher
from ucm.integration.vllm.ucm_connector import (
    KVCacheLayout,
    PendingDumpTask,
    RequestDispatchMeta,
    RequestMeta,
    UCMConnectorMetadata,
    UCMDirectConnector,
    UCMLiteConnector,
    _record_counter,
    _scheduler_read_block_size,
    _short_list,
    _use_ucm_connector_cpu_affinity,
)
from ucm.logger import init_logger
from ucm.shared.metrics import ucmmetrics
from ucm.sparse.state import has_ucm_sparse
from ucm.store.factory_v1 import UcmConnectorFactoryV1
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class HLARequestMeta(RequestMeta):
    """RequestMeta extended with per-group block tracking for hybrid models."""

    group_ucm_block_ids: list[list[bytes]] = field(default_factory=list)
    group_vllm_block_ids: list[list[int]] = field(default_factory=list)


@dataclass
class HLARequestDispatchMeta(RequestDispatchMeta):
    """Extends RequestDispatchMeta with full-attn block count for MLA rank scoping."""

    load_full_attn_count: int = 0
    dump_full_attn_count: int = 0
    # Parallel to load/dump vLLM block ids. Selects the BLNHC source block.
    load_block_group_ids: list[int] = field(default_factory=list)
    dump_block_group_ids: list[int] = field(default_factory=list)


def layer_name_to_kv_cache_spec(
    kv_cache_config: "KVCacheConfig",
) -> dict[str, list[KVCacheSpec]]:
    """Map each model layer name to its concrete KVCacheSpec.

    Handles merged group specs and UniformTypeKVCacheSpecs (per-layer
    ``kv_cache_specs`` entries).
    """
    out: dict[str, list[KVCacheSpec]] = defaultdict(list)
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            by_name = spec.kv_cache_specs
            for name in group.layer_names:
                out[name].append(by_name[name])
        else:
            for name in group.layer_names:
                out[name].append(spec)
    return out


def block_size_from_kv_cache_spec(spec: KVCacheSpec) -> int:
    """Token block size used for KV scheduling / hashing for one group spec."""
    block_size = 0
    if isinstance(spec, UniformTypeKVCacheSpecs):
        block_size = next(iter(spec.kv_cache_specs.values())).block_size
    else:
        block_size = spec.block_size

    return block_size


def is_mamba_align_kv_cache_spec(spec: KVCacheSpec) -> bool:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        sample = next(iter(spec.kv_cache_specs.values()))
        return is_mamba_align_kv_cache_spec(sample)
    return isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align"


def extend_non_null(
    dst_ucm_block_ids: list[bytes],
    dst_vllm_block_ids: list[int],
    src_ucm_block_ids: list[bytes],
    src_vllm_block_ids: list[int],
    dst_group_ids: Optional[list[int]] = None,
    group_id: Optional[int] = None,
) -> None:
    # Skip vLLM null blocks (block_id=0) used as mamba-align placeholders.
    for ucm_block_id, vllm_block_id in zip(src_ucm_block_ids, src_vllm_block_ids):
        if vllm_block_id == 0:
            continue
        dst_ucm_block_ids.append(ucm_block_id)
        dst_vllm_block_ids.append(vllm_block_id)
        if dst_group_ids is not None:
            dst_group_ids.append(int(group_id))


def kv_cache_group_external_role(group) -> str:
    """How one KV cache group participates in external cache IO.

    ``raw_key`` stays in vLLM's block tables and is never hashed or copied.
    ``ple`` and ``linear`` are mamba-align state groups. ``full_attn`` is the
    token-hashed group. Compressed key shares that group's physical block.
    """
    names = [str(name).lower() for name in group.layer_names]
    if names and all("raw_key_cache" in name for name in names):
        return "raw_key"
    if is_mamba_align_kv_cache_spec(group.kv_cache_spec):
        if any(name.endswith(".ple") or ".ple." in name for name in names):
            return "ple"
        return "linear"
    return "full_attn"


def _preview_block_hash(block_id: bytes) -> str:
    if not block_id:
        return "empty"
    return block_id[:8].hex()


def _normalize_tensor_size_list(tensor_size_list: Any) -> list[int]:
    if isinstance(tensor_size_list, np.ndarray):
        return [int(v) for v in tensor_size_list.reshape(-1).tolist()]
    if isinstance(tensor_size_list, (list, tuple)):
        return [int(v) for v in tensor_size_list]
    return [int(tensor_size_list)]


def kv_cache_tensor_layer_names(raw_tensor) -> list[str]:
    """Layer names placed in one vLLM KV cache tensor.

    Current vLLM stores them on ``layers``. Older configs used ``shared_by``.
    """
    layers = getattr(raw_tensor, "layers", None)
    if layers:
        return list(layers)
    shared_by = getattr(raw_tensor, "shared_by", None)
    if shared_by:
        return list(shared_by)
    return []


def iter_group_kv_cache_specs(spec: KVCacheSpec):
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return spec.kv_cache_specs.values()
    return (spec,)


def _layer_name_is_flash_next_only(name: str) -> bool:
    lowered = str(name).lower()
    return (
        "raw_key_cache" in lowered
        or "compressed_key_cache" in lowered
        or lowered.endswith(".ple")
        or ".ple." in lowered
    )


def hybrid_kv_cache_layout_name(kv_cache_config, vllm_config=None) -> str:
    """``LBNHC`` is layer-first. ``BLNHC`` is block-first."""
    configured = getattr(kv_cache_config, "kv_cache_layout", None)
    if configured:
        return str(configured).split(".")[-1]
    cache_config = getattr(vllm_config, "cache_config", None) if vllm_config else None
    resolver = getattr(cache_config, "get_resolved_kv_cache_layout", None)
    if resolver is None:
        return "LBNHC"
    layout = resolver()
    return str(getattr(layout, "name", layout)).split(".")[-1]


def has_npu_qwen3_next_groups(kv_cache_config) -> bool:
    """Layer-first mamba models: full attention plus align-mode linear groups.

    Compressed key, raw key, and PLE belong to block-first flash-next.
    """
    has_full_attention = False
    has_mamba_align = False
    for group in kv_cache_config.kv_cache_groups:
        if any(_layer_name_is_flash_next_only(name) for name in group.layer_names):
            return False
        role = kv_cache_group_external_role(group)
        if role == "linear":
            has_mamba_align = True
        elif role == "full_attn":
            has_full_attention = True
        else:
            return False
    return has_full_attention and has_mamba_align


def has_cuda_hybrid_linear_attention_groups(kv_cache_config) -> bool:
    """True when CUDA groups contain full attention and align-mode mamba.

    Current vLLM gives each group its own tensor, so the two specs no longer
    share one ``KVCacheTensor``. Group membership is the stable signal.
    """
    has_full_attention = False
    has_mamba_align = False
    for group in kv_cache_config.kv_cache_groups:
        for spec in iter_group_kv_cache_specs(group.kv_cache_spec):
            if isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align":
                has_mamba_align = True
            elif isinstance(spec, FullAttentionSpec):
                has_full_attention = True
    return has_full_attention and has_mamba_align


@dataclass
class GroupInfo:
    """Per-group metadata used by :class:`KVCacheGroupManager`."""

    group_id: int
    block_size: int
    layer_names: tuple[str, ...]
    # Independent hash chain seed per group (see ``KVCacheGroupManager``).
    seed: bytes
    is_mamba_align: bool = False
    external_role: str = "full_attn"
    block_hasher: Optional[Callable[["Request"], list[bytes]]] = None

    @property
    def is_full_attention(self) -> bool:
        return self.external_role == "full_attn"

    @property
    def transfers_external_cache(self) -> bool:
        return self.external_role != "raw_key"


class KVCacheGroupManager:
    """Group-aware hashing and two-stage lookup for hybrid (HLA) connectors."""

    def __init__(
        self,
        kv_cache_config: "KVCacheConfig",
        request_hasher: "RequestHasher",
        base_seed: bytes,
    ) -> None:
        self.request_hasher = request_hasher
        self.groups_by_id: list[GroupInfo] = []
        self.full_attn_groups: list[GroupInfo] = []
        self.state_groups: list[GroupInfo] = []

        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            block_size = block_size_from_kv_cache_spec(spec)
            is_mamba_align = is_mamba_align_kv_cache_spec(spec)
            external_role = kv_cache_group_external_role(group)
            seed = request_hasher((b"UCM_GROUP_SEED", base_seed, group_id))
            info = GroupInfo(
                group_id=group_id,
                block_size=block_size,
                layer_names=tuple(group.layer_names),
                seed=seed,
                is_mamba_align=is_mamba_align,
                external_role=external_role,
            )
            if external_role == "full_attn":
                info.block_hasher = request_hasher.make_request_block_hasher(
                    block_size, seed
                )
            self.groups_by_id.append(info)
            if info.is_full_attention:
                self.full_attn_groups.append(info)
            elif external_role in ("linear", "ple"):
                self.state_groups.append(info)

        assert len(self.full_attn_groups) >= 1, (
            "UCMHybridLinearAttentionConnector expects at least one full-attention group in "
            "kv_cache_config.kv_cache_groups."
        )

        # raw_key is not part of the external cache, so it does not define
        # the resume alignment.
        aligned_groups = [g for g in self.groups_by_id if g.transfers_external_cache]
        all_block_sizes = [g.block_size for g in aligned_groups]
        self.lcm_block_size: int = math.lcm(*all_block_sizes)
        self.primary_full_attn_group = self.full_attn_groups[0]
        for group in self.full_attn_groups:
            if any(
                "compressed_key_cache" not in name.lower()
                and "raw_key_cache" not in name.lower()
                and "attn" in name.lower()
                for name in group.layer_names
            ):
                self.primary_full_attn_group = group
                break

        for g in aligned_groups:
            assert self.lcm_block_size % g.block_size == 0, (
                f"group {g.group_id} block_size={g.block_size} does not "
                f"divide LCM={self.lcm_block_size}"
            )
        for sg in self.state_groups:
            assert sg.is_mamba_align, (
                f"state group {sg.group_id} is not mamba-align; "
                f"UCMHybridLinearAttentionConnector only supports mamba-align "
                f"state groups."
            )

        logger.info(
            "KVCacheGroupManager initialized: "
            f"lcm_block_size={self.lcm_block_size}, "
            f"full_attn_groups="
            f"{[(g.group_id, g.block_size) for g in self.full_attn_groups]}, "
            f"state_groups="
            f"{[(g.group_id, g.block_size, g.external_role) for g in self.state_groups]}, "
            f"primary_full_attn={self.primary_full_attn_group.group_id}"
        )

    @property
    def num_groups(self) -> int:
        return len(self.groups_by_id)

    def compute_block_hashes(self, group: GroupInfo, request: "Request") -> list[bytes]:
        """Hash a request at one group's block boundaries and chain seed."""
        if group.external_role == "raw_key":
            return []
        if group.is_mamba_align:
            # mamba-align pads block table with null blocks; no per-block hash.
            return [b""] * (len(request.all_token_ids) // group.block_size)

        assert group.block_hasher is not None
        return group.block_hasher(request)

    def compute_all_group_block_ids(self, request: "Request") -> list[list[bytes]]:
        """Compute full block hashes for every group, indexed by group_id."""
        return [self.compute_block_hashes(g, request) for g in self.groups_by_id]

    def compute_mamba_align_state_hash(
        self,
        group: GroupInfo,
        seq_len: int,
        group_block_ids: list[list[bytes]],
    ) -> Optional[bytes]:
        """Derive the mamba-align state hash at ``seq_len`` from the prefix hash."""
        if seq_len <= 0 or seq_len % self.lcm_block_size != 0:
            return None
        primary = self.primary_full_attn_group
        prefix_idx = seq_len // primary.block_size - 1
        if prefix_idx < 0:
            return None
        try:
            prefix_hash = group_block_ids[primary.group_id][prefix_idx]
        except IndexError:
            logger.error(
                "mamba-align state hash missing primary prefix hash: "
                f"group_id={group.group_id}, seq_len={seq_len}, "
                f"primary_group_id={primary.group_id}, "
                f"prefix_idx={prefix_idx}, "
                f"num_primary_hashes="
                f"{len(group_block_ids[primary.group_id])}"
            )
            return None
        if not prefix_hash:
            return None
        return self.request_hasher(
            (group.seed, b"UCM_MAMBA_ALIGN_STATE", seq_len, prefix_hash)
        )

    def lookup_external_hit_tokens(
        self,
        num_computed_tokens: int,
        group_block_ids: list[list[bytes]],
        lookup_on_prefix: Callable[[list[bytes]], int],
        lookup_on_reverse: Callable[[list[bytes]], int],
    ) -> tuple[int, int, list[bytes]]:
        """Two-stage HLA lookup using precomputed per-group hashes.

        ``group_block_ids`` must have one entry per group, indexed by the
        original ``group_id`` (see :meth:`compute_all_group_block_ids`).

        Stage 1 — every full-attention group runs ``lookup_on_prefix``
        beyond its own ``hbm_hit_block_num``; the candidate hits are taken
        as a min and rounded down to ``lcm_block_size`` so the final
        external hit is consistent across all full-attn groups and aligns
        to the kv-cache page granularity expected by the scheduler.

        Stage 2 — mamba-align state groups are checked via
        ``lookup_on_reverse``: for each state group, the state hashes at
        all candidate LCM boundary positions (earliest-to-latest) are
        collected and a single reverse scan finds the rightmost hit.
        The min across state groups is the rightmost position where ALL
        state groups' states are present. If any state group has no hit
        at any candidate position, the external hit is downgraded to zero.

        Returns:
            Tuple of
            - ``external_hit_tokens``: tokens hit beyond ``num_computed_tokens``,
              aligned to ``lcm_block_size``. ``0`` if any check fails.
            - ``external_hit_lcm_blocks``: ``external_hit_tokens //
              lcm_block_size`` (also ``0`` on downgrade).
            - ``mamba_prefetch_hashes``: rank-0 mamba state hashes from
              ``num_computed_tokens + lcm_block_size`` to ``best_pos``,
              for GC heat update (rank-0 un-checked positions + other ranks).
        """
        assert len(group_block_ids) == self.num_groups, (
            f"group_block_ids length {len(group_block_ids)} does not match "
            f"num_groups {self.num_groups}"
        )
        assert num_computed_tokens % self.lcm_block_size == 0, (
            f"num_computed_tokens={num_computed_tokens} is not aligned to "
            f"lcm_block_size={self.lcm_block_size}"
        )

        # Stage 1: each full-attn group contributes a candidate hit count.
        candidates: list[int] = []
        stage1_detail: list[str] = []
        for fa in self.full_attn_groups:
            fa_block_ids = group_block_ids[fa.group_id]
            fa_hbm_blocks = num_computed_tokens // fa.block_size
            fa_external = fa_block_ids[fa_hbm_blocks:]
            if not fa_external:
                candidates.append(0)
                stage1_detail.append(
                    f"g{fa.group_id}:ext=0,hbm_blocks={fa_hbm_blocks},"
                    f"total={len(fa_block_ids)}"
                )
                continue
            try:
                fa_hit_blocks = lookup_on_prefix(fa_external) + 1
            except Exception as e:
                logger.error(
                    f"full-attn group {fa.group_id} lookup error. "
                    f"{type(e).__name__}: {e}"
                )
                _record_counter("connector_lookup_errors_total")
                candidates.append(0)
                stage1_detail.append(f"g{fa.group_id}:error")
                continue
            candidates.append(max(fa_hit_blocks, 0) * fa.block_size)
            stage1_detail.append(
                f"g{fa.group_id}:ext={len(fa_external)},"
                f"hit_blocks={max(fa_hit_blocks, 0)},"
                f"block_size={fa.block_size},"
                f"first={_preview_block_hash(fa_external[0])}"
            )

        # Resume boundary must be a multiple of lcm_block_size so every
        # group's tail/dispatch slicing lands on a real block boundary.
        min_external_hit_tokens = min(candidates) if candidates else 0
        external_hit_tokens = (
            min_external_hit_tokens // self.lcm_block_size
        ) * self.lcm_block_size
        if external_hit_tokens <= 0:
            logger.info(
                "HLA external miss at full-attention lookup: "
                f"computed_tokens={num_computed_tokens}, "
                f"lcm_block_size={self.lcm_block_size}, "
                f"candidates={candidates}, detail=[{'; '.join(stage1_detail)}]"
            )
            return 0, 0, []

        # Stage 2: reverse scan for mamba state at LCM boundaries.
        # For each state group, collect state hashes at all candidate
        # positions (earliest-to-latest) and use lookup_on_reverse to find
        # the rightmost hit.  The min across state groups is the rightmost
        # position where ALL states are present.
        total_hit_tokens = num_computed_tokens + external_hit_tokens

        if not self.state_groups:
            return (
                external_hit_tokens,
                external_hit_tokens // self.lcm_block_size,
                [],
            )

        positions = list(
            range(
                num_computed_tokens + self.lcm_block_size,
                total_hit_tokens + self.lcm_block_size,
                self.lcm_block_size,
            )
        )

        best_pos = total_hit_tokens
        for sg in self.state_groups:
            # Truncate to positions <= best_pos so earlier state groups
            # can shrink the search window for subsequent ones.
            sg_positions = [p for p in positions if p <= best_pos]
            sg_hashes: list[bytes] = []
            for pos in sg_positions:
                state_hash = self.compute_mamba_align_state_hash(
                    sg, pos, group_block_ids
                )
                sg_hashes.append(state_hash if state_hash is not None else b"")
            try:
                idx = lookup_on_reverse(sg_hashes)
            except Exception as e:
                logger.error(
                    f"mamba-align state reverse lookup error for "
                    f"group={sg.group_id}. {type(e).__name__}: {e}"
                )
                _record_counter("connector_lookup_errors_total")
                return 0, 0, []
            if idx < 0:
                # This state group has no state at any candidate position.
                empty_hashes = sum(1 for state_hash in sg_hashes if not state_hash)
                logger.info(
                    "HLA external miss at mamba-align lookup: "
                    f"group={sg.group_id}, positions={sg_positions}, "
                    f"hashes={len(sg_hashes)}, empty_hashes={empty_hashes}, "
                    f"full_attn_hit_tokens={external_hit_tokens}, "
                    f"first={_preview_block_hash(sg_hashes[0] if sg_hashes else b'')}"
                )
                return 0, 0, []
            sg_pos = sg_positions[idx]
            if sg_pos < best_pos:
                logger.info(
                    "HLA mamba-align lookup shortened the hit: "
                    f"group={sg.group_id}, hit_pos={sg_pos}, "
                    f"previous_best={best_pos}, positions={sg_positions}"
                )
                best_pos = sg_pos

        external_hit_tokens = best_pos - num_computed_tokens
        if external_hit_tokens <= 0:
            logger.info(
                "HLA external miss after mamba-align alignment: "
                f"best_pos={best_pos}, computed_tokens={num_computed_tokens}"
            )
            return 0, 0, []

        # Collect mamba state hashes for GC heat update.
        mamba_prefetch_hashes: list[bytes] = []
        for pos in range(
            self.lcm_block_size,
            best_pos + self.lcm_block_size,
            self.lcm_block_size,
        ):
            for sg in self.state_groups:
                state_hash = self.compute_mamba_align_state_hash(
                    sg, pos, group_block_ids
                )
                if state_hash is not None:
                    mamba_prefetch_hashes.append(state_hash)

        return (
            external_hit_tokens,
            external_hit_tokens // self.lcm_block_size,
            mamba_prefetch_hashes,
        )


class HybridLinearAttentionLayout(KVCacheLayout):
    """Physical layout for hybrid full-attention + linear-attention pages.

    vLLM backs full-attention and linear-attention groups with raw tensors.
    The physical layout is backend dependent:

    - Ascend Qwen3-Next stores one ``block_stride`` page per layer. Conv/ssm
      and K/V already sit inside that page. Whole-cache copies every page in
      one object. Layerwise gives each page its own shard, in forward order.
      Older Ascend hybrid tensors stay component-major:
        [conv_block_or_padding, k_or_ssm_block, v_block_or_padding]
      across all physical blocks.
    - CUDA LBNHC is layer-outermost, ``[L, B, N, H, C]``. Each layer is its
      own contiguous region of blocks. One logical block is one page from
      every layer, and those pages are not adjacent. Separate group tensors
      alias this same allocation, because one block id belongs to one group
      at a time.

    The store receives one unified tensor_size_list. Ascend exposes three
    physical slices. CUDA exposes one page segment per layer and drops
    aliased group tensors.
    """

    def __init__(
        self,
        kvcaches,
        ucm_config: dict,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
    ):
        # Layer-first models split one shard per page when layerwise is on.
        # Block-first flash-next never takes that mode.
        use_layerwise = bool(ucm_config.get("use_layerwise", True))
        hybrid_layerwise = bool(
            ucm_config.get("hybrid_linear_attention_layerwise", True)
        )
        self._layer_first_layerwise = use_layerwise and hybrid_layerwise
        super().__init__(kvcaches, ucm_config, vllm_config, kv_cache_config)

    @staticmethod
    def _dtype_size(dtype: torch.dtype) -> int:
        return torch.empty((), dtype=dtype).element_size()

    @staticmethod
    def _mamba_component_sizes(spec: MambaSpec) -> list[int]:
        return [
            math.prod(shape) * HybridLinearAttentionLayout._dtype_size(dtype)
            for shape, dtype in zip(spec.shapes, spec.dtypes)
        ]

    def _attention_component_sizes(self, spec: KVCacheSpec) -> tuple[int, int]:
        assert isinstance(spec, FullAttentionSpec)
        if isinstance(spec, MLAAttentionSpec):
            # MLA: head_size = kv_lora_rank + qk_rope_head_dim
            hf = self.vllm_config.model_config.hf_text_config
            k_dim = getattr(hf, "kv_lora_rank", spec.head_size)
            v_dim = getattr(hf, "qk_rope_head_dim", spec.head_size)
        else:
            k_dim = spec.head_size
            v_dim = getattr(spec, "head_size_v", spec.head_size)
        k_size = (
            spec.block_size * spec.num_kv_heads * k_dim * self._dtype_size(spec.dtype)
        )
        v_size = (
            spec.block_size * spec.num_kv_heads * v_dim * self._dtype_size(spec.dtype)
        )
        return k_size, v_size

    def _finalize_layout_arrays(
        self,
        base_ptrs: list[list[int]],
        buffer_size_rows: list[list[int]],
        tensor_size_lists: list[list[int]],
        block_stride_lists: list[list[int]],
    ) -> None:
        self.row_slices: list[slice] = []
        self.row_tensor_size_lists: list[list[int]] = [
            [int(size) for size in row] for row in tensor_size_lists
        ]
        self.row_shard_sizes: list[int] = [
            sum(row) for row in self.row_tensor_size_lists
        ]

        offset = 0
        for row in tensor_size_lists:
            next_offset = offset + len(row)
            self.row_slices.append(slice(offset, next_offset))
            offset = next_offset

        self.base_ptrs = np.asarray(
            [ptr for row in base_ptrs for ptr in row], dtype=np.uint64
        )
        self.buffer_sizes = np.asarray(
            [size for row in buffer_size_rows for size in row], dtype=np.uint64
        )
        self.tensor_size_lists = np.asarray(
            [size for row in tensor_size_lists for size in row], dtype=np.uint64
        )
        self.block_stride_lists = np.asarray(
            [stride for row in block_stride_lists for stride in row], dtype=np.uint64
        )

        all_block_ids = np.arange(self.num_blocks, dtype=np.uint64)
        self.row_addr_lookup: dict[int, np.ndarray] = {}
        for row_id, row_slice in enumerate(self.row_slices):
            stride = np.ascontiguousarray(self.block_stride_lists[row_slice])
            base = np.ascontiguousarray(self.base_ptrs[row_slice])
            self.row_addr_lookup[row_id] = np.ascontiguousarray(
                all_block_ids[:, None] * stride[None, :] + base[None, :]
            )

    def extract_block_addrs(
        self, vllm_block_ids: List[int], layer_first: bool = False
    ) -> np.ndarray:
        if layer_first:
            raise ValueError("layer_first is not supported for flattened hybrid layout")
        vllm_block_ids_np = np.asarray(vllm_block_ids, dtype=np.uint64)
        return (
            vllm_block_ids_np[:, None] * self.block_stride_lists[None, :]
            + self.base_ptrs[None, :]
        )

    def extract_block_addrs_for_row(
        self, vllm_block_ids: List[int], row_id: int
    ) -> np.ndarray:
        if row_id < 0 or row_id >= len(self.row_slices):
            raise ValueError(
                f"Invalid hybrid row_id={row_id}; row_count={len(self.row_slices)}"
            )
        lookup = self.row_addr_lookup.get(row_id)
        if lookup is not None:
            return lookup[np.asarray(vllm_block_ids, dtype=np.uint64)]
        row_slice = self.row_slices[row_id]
        vllm_block_ids_np = np.asarray(vllm_block_ids, dtype=np.uint64)
        return (
            vllm_block_ids_np[:, None] * self.block_stride_lists[row_slice][None, :]
            + self.base_ptrs[row_slice][None, :]
        )

    def _collect_shared_tensor_info(
        self,
        raw_tensor,
        kvcaches,
    ) -> tuple[list[KVCacheSpec], list[int]]:
        shared_specs: list[KVCacheSpec] = []
        shared_ptrs: list[int] = []
        layer_to_specs = layer_name_to_kv_cache_spec(self.kv_cache_config)
        for layer_name in kv_cache_tensor_layer_names(raw_tensor):
            kv_layer = kvcaches.get(layer_name)
            if kv_layer is None:
                continue
            shared_specs.extend(layer_to_specs[layer_name])
            if isinstance(kv_layer, torch.Tensor):
                shared_ptrs.append(kv_layer.data_ptr())
            elif isinstance(kv_layer, (tuple, list)):
                for tensor in kv_layer:
                    if isinstance(tensor, torch.Tensor):
                        shared_ptrs.append(tensor.data_ptr())
            else:
                logger.warning(f"unsupported kv_layer type: {type(kv_layer)}")
        return shared_specs, shared_ptrs

    def _log_component_adjacency(self, raw_tensor, kvcaches) -> None:
        """Log whether conv and ssm/K of one block sit next to each other."""
        layer_names = kv_cache_tensor_layer_names(raw_tensor)
        if not layer_names:
            return
        layer_name = layer_names[0]
        kv_layer = kvcaches.get(layer_name)
        if isinstance(kv_layer, torch.Tensor):
            components = [("tensor", kv_layer)]
        elif isinstance(kv_layer, (tuple, list)):
            components = [
                (f"c{index}", tensor)
                for index, tensor in enumerate(kv_layer)
                if isinstance(tensor, torch.Tensor)
            ]
        else:
            logger.info(
                "HLA component adjacency: "
                f"layer={layer_name}, unsupported_type={type(kv_layer)}"
            )
            return
        described = []
        for name, tensor in components:
            one_block = (
                int(tensor.nbytes) // self.num_blocks
                if self.num_blocks and int(tensor.nbytes) % self.num_blocks == 0
                else None
            )
            block1_ptr = None
            if tensor.dim() > 0 and tensor.shape[0] > 1:
                block1_ptr = int(tensor[1].data_ptr())
            described.append(
                {
                    "name": name,
                    "ptr": int(tensor.data_ptr()),
                    "nbytes": int(tensor.nbytes),
                    "shape": tuple(tensor.shape),
                    "one_block": one_block,
                    "block1_ptr": block1_ptr,
                }
            )
        ordered = sorted(described, key=lambda item: item["ptr"])
        gaps = []
        for prev, curr in zip(ordered, ordered[1:]):
            delta = curr["ptr"] - prev["ptr"]
            gaps.append(
                f"{prev['name']}->{curr['name']}: delta={delta}, "
                f"prev_nbytes={prev['nbytes']}, "
                f"prev_one_block={prev['one_block']}, "
                f"same_block_adjacent={delta == prev['one_block']}, "
                f"component_region_adjacent={delta == prev['nbytes']}"
            )
        logger.info(
            "HLA component adjacency: "
            f"layer={layer_name}, layer_count={len(layer_names)}, "
            f"components={described}, gaps=[{'; '.join(gaps)}], "
            f"block_stride={getattr(raw_tensor, 'block_stride', None)}, "
            f"layer_stride={getattr(raw_tensor, 'layer_stride', None)}, "
            f"offset={getattr(raw_tensor, 'offset', None)}, "
            f"size={getattr(raw_tensor, 'size', None)}"
        )

    def _append_contiguous_page_layout(
        self,
        raw_tensor,
        shared_ptrs: list[int],
        base_ptrs: list[list[int]],
        buffer_size_rows: list[list[int]],
        tensor_size_lists: list[list[int]],
        block_stride_lists: list[list[int]],
    ) -> None:
        if raw_tensor.size % self.num_blocks != 0:
            raise ValueError(
                "Invalid hybrid linear-attention raw tensor size: "
                f"raw_size={raw_tensor.size}, num_blocks={self.num_blocks}"
            )
        page_size = raw_tensor.size // self.num_blocks
        base = min(shared_ptrs)
        base_ptrs.append([base])
        buffer_size_rows.append([raw_tensor.size])
        tensor_size_lists.append([page_size])
        block_stride_lists.append([page_size])

    def _cuda_layer_bases(
        self, raw_tensor, layer_names: list[str], layer_ptrs: list[tuple[int, int]]
    ) -> tuple[list[int], int]:
        """Return per-layer block-0 pointers and the page size for LBNHC.

        Layer ``l`` occupies ``[base_l, base_l + layer_stride)``. Block ``b``
        of that layer starts at ``base_l + b * block_stride``.
        """
        block_stride = int(getattr(raw_tensor, "block_stride", 0) or 0)
        layer_stride = int(getattr(raw_tensor, "layer_stride", 0) or 0)
        if block_stride <= 0 or layer_stride <= 0:
            raise ValueError(
                "CUDA LBNHC tensor is missing block_stride or layer_stride: "
                f"block_stride={block_stride}, layer_stride={layer_stride}"
            )
        if layer_stride != block_stride * self.num_blocks:
            raise ValueError(
                "CUDA LBNHC layer region is not one contiguous run of blocks: "
                f"layer_stride={layer_stride}, block_stride={block_stride}, "
                f"num_blocks={self.num_blocks}"
            )
        span = len(layer_names) * layer_stride
        if span > int(raw_tensor.size):
            raise ValueError(
                "CUDA LBNHC layers exceed the raw allocation: "
                f"layers={len(layer_names)}, layer_stride={layer_stride}, "
                f"raw_size={raw_tensor.size}"
            )

        index, ptr = layer_ptrs[0]
        layer0_ptr = ptr - index * layer_stride
        if layer0_ptr < 0:
            raise ValueError(
                "CUDA LBNHC layer pointer falls before layer 0: "
                f"ptr={ptr}, layer_index={index}, layer_stride={layer_stride}"
            )
        for other_index, other_ptr in layer_ptrs[1:]:
            expected = layer0_ptr + other_index * layer_stride
            if other_ptr != expected:
                raise ValueError(
                    "CUDA LBNHC layers are not placed on layer_stride: "
                    f"layer_index={other_index}, ptr={other_ptr}, "
                    f"expected={expected}, layer_stride={layer_stride}"
                )
        bases = [
            layer0_ptr + layer_index * layer_stride
            for layer_index in range(len(layer_names))
        ]
        return bases, block_stride

    def _build_cuda_lbnhc_layout(self, kvcaches) -> None:
        """One page segment per layer for CUDA LBNHC whole-cache IO.

        Group tensors overlay one backing allocation. Recording each of them
        would copy the same layer pages once per group.
        """
        base_ptrs: list[list[int]] = []
        buffer_size_rows: list[list[int]] = []
        tensor_size_lists: list[list[int]] = []
        block_stride_lists: list[list[int]] = []
        self.layer_name_to_row = {}
        seen_blocks: dict[tuple[int, int], int] = {}
        tensor_count = 0

        for raw_tensor in self.kv_cache_config.kv_cache_tensors:
            layer_names = kv_cache_tensor_layer_names(raw_tensor)
            if not layer_names:
                continue
            layer_ptrs: list[tuple[int, int]] = []
            for index, layer_name in enumerate(layer_names):
                kv_layer = kvcaches.get(layer_name)
                if kv_layer is None:
                    continue
                ptr = self._layer_data_ptr(kv_layer)
                if ptr is None:
                    logger.warning(f"unsupported kv_layer type: {type(kv_layer)}")
                    continue
                layer_ptrs.append((index, ptr))
            if not layer_ptrs:
                logger.warning(f"no kv cache tensor found for layers {layer_names}")
                continue

            tensor_count += 1
            bases, page_size = self._cuda_layer_bases(raw_tensor, layer_names, layer_ptrs)
            block_key = (bases[0], int(raw_tensor.size))
            row_id = seen_blocks.get(block_key)
            if row_id is None:
                row_id = len(base_ptrs)
                seen_blocks[block_key] = row_id
                layer_region = page_size * self.num_blocks
                base_ptrs.append(bases)
                buffer_size_rows.append([layer_region] * len(bases))
                tensor_size_lists.append([page_size] * len(bases))
                block_stride_lists.append([page_size] * len(bases))
            for layer_name in layer_names:
                self.layer_name_to_row[layer_name] = row_id

        if tensor_count == 0:
            raise ValueError("CUDA hybrid layout found no KV cache tensors")

        logger.info(
            "CUDA LBNHC hybrid layout: "
            f"tensors={tensor_count}, packed_blocks={len(base_ptrs)}, "
            f"layers_per_block={len(tensor_size_lists[0])}, "
            f"page_size={int(tensor_size_lists[0][0])}"
        )
        self._finalize_layout_arrays(
            base_ptrs,
            buffer_size_rows,
            tensor_size_lists,
            block_stride_lists,
        )

    @staticmethod
    def _layer_data_ptr(kv_layer) -> Optional[int]:
        if isinstance(kv_layer, torch.Tensor):
            return int(kv_layer.data_ptr())
        if isinstance(kv_layer, (tuple, list)):
            ptrs = [
                int(tensor.data_ptr())
                for tensor in kv_layer
                if isinstance(tensor, torch.Tensor)
            ]
            return min(ptrs) if ptrs else None
        return None

    def _append_ascend_component_major_layout(
        self,
        raw_tensor,
        shared_ptrs: list[int],
        mamba_specs: list[MambaSpec],
        attn_specs: list[FullAttentionSpec],
        base_ptrs: list[list[int]],
        buffer_size_rows: list[list[int]],
        tensor_size_lists: list[list[int]],
        block_stride_lists: list[list[int]],
    ) -> None:
        mamba_sizes = self._mamba_component_sizes(mamba_specs[0])
        if len(mamba_sizes) < 2:
            logger.warning(
                f"unexpected mamba component sizes {mamba_sizes}; "
                "falling back to contiguous page layout"
            )
            self._append_contiguous_page_layout(
                raw_tensor,
                shared_ptrs,
                base_ptrs,
                buffer_size_rows,
                tensor_size_lists,
                block_stride_lists,
            )
            return

        conv_size = mamba_sizes[0]
        ssm_size = mamba_sizes[1]
        k_size, v_size = self._attention_component_sizes(attn_specs[0])
        middle_size = max(k_size, ssm_size)
        page_size = raw_tensor.size // self.num_blocks
        tail_size = page_size - conv_size - middle_size
        if tail_size <= 0:
            raise ValueError(
                "Invalid Ascend hybrid linear-attention page layout: "
                f"page_size={page_size}, conv_size={conv_size}, "
                f"middle_size={middle_size}, tail_size={tail_size}"
            )
        if tail_size < v_size:
            raise ValueError(
                "Ascend hybrid linear-attention tail cannot hold attention V: "
                f"tail_size={tail_size}, v_size={v_size}"
            )

        base = min(shared_ptrs)
        offsets = [
            0,
            conv_size * self.num_blocks,
            (conv_size + middle_size) * self.num_blocks,
        ]
        sizes = [conv_size, middle_size, tail_size]
        base_ptrs.append([base + offset for offset in offsets])
        buffer_size_rows.append([size * self.num_blocks for size in sizes])
        tensor_size_lists.append(sizes)
        block_stride_lists.append(sizes)

    def _append_ascend_attn_only_layout(
        self,
        raw_tensor,
        shared_ptrs: list[int],
        attn_specs: list[FullAttentionSpec],
        base_ptrs: list[list[int]],
        buffer_size_rows: list[list[int]],
        tensor_size_lists: list[list[int]],
        block_stride_lists: list[list[int]],
    ) -> None:
        """Component-major [conv_padding, K, V] layout for Ascend attn-only tensors."""
        k_size, v_size = self._attention_component_sizes(attn_specs[0])
        page_size = raw_tensor.size // self.num_blocks
        conv_padding_size = page_size - k_size - v_size
        if conv_padding_size <= 0:
            self._append_contiguous_page_layout(
                raw_tensor,
                shared_ptrs,
                base_ptrs,
                buffer_size_rows,
                tensor_size_lists,
                block_stride_lists,
            )
            return

        # K-cache view starts past conv_padding; subtract to get raw base.
        base = min(shared_ptrs) - conv_padding_size * self.num_blocks
        sizes = [conv_padding_size, k_size, v_size]
        offsets = [
            0,
            conv_padding_size * self.num_blocks,
            (conv_padding_size + k_size) * self.num_blocks,
        ]
        base_ptrs.append([base + offset for offset in offsets])
        buffer_size_rows.append([size * self.num_blocks for size in sizes])
        tensor_size_lists.append(sizes)
        block_stride_lists.append(sizes)

    def _layout_name(self) -> str:
        return hybrid_kv_cache_layout_name(self.kv_cache_config, self.vllm_config)

    @staticmethod
    def _require_tensor_offset(raw_tensor) -> int:
        if not hasattr(raw_tensor, "offset"):
            raise ValueError(
                "KVCacheTensor.offset is required for BLNHC; "
                "refusing to assume offset 0"
            )
        return int(raw_tensor.offset)

    def _blnhc_block0(
        self, raw_tensor, layer_views: list[tuple[int, int, Optional[torch.Tensor]]]
    ) -> tuple[int, int, int, int]:
        """Return layer 0 of block 0. ``data_ptr`` already includes ``offset``."""
        if not hasattr(raw_tensor, "block_stride") or not hasattr(
            raw_tensor, "layer_stride"
        ):
            raise ValueError("BLNHC tensor is missing block_stride or layer_stride")
        offset = self._require_tensor_offset(raw_tensor)
        block_stride = int(raw_tensor.block_stride)
        layer_stride = int(raw_tensor.layer_stride)
        if block_stride <= 0 or layer_stride <= 0:
            raise ValueError(
                "BLNHC tensor has non-positive stride: "
                f"block_stride={block_stride}, layer_stride={layer_stride}, "
                f"offset={offset}"
            )

        def storage_base(tensor: Optional[torch.Tensor]) -> Optional[int]:
            if tensor is None:
                return None
            try:
                return int(tensor.untyped_storage().data_ptr())
            except (AttributeError, RuntimeError):
                return None

        points_at_storage = all(
            storage_base(tensor) == ptr for _, ptr, tensor in layer_views
        )
        if points_at_storage:
            block0 = layer_views[0][1] + offset
        else:
            index, ptr, _ = layer_views[0]
            block0 = ptr - index * layer_stride
            for other_index, other_ptr, _ in layer_views[1:]:
                if other_ptr - other_index * layer_stride != block0:
                    raise ValueError(
                        "BLNHC layers are not placed at layer_stride: "
                        f"offset={offset}, layer_stride={layer_stride}, "
                        f"layer_index={other_index}, ptr={other_ptr}, "
                        f"block0={block0}"
                    )
        if block0 < 0:
            raise ValueError(
                f"BLNHC block base is negative: block0={block0}, offset={offset}"
            )
        return block0, block_stride, layer_stride, offset

    @staticmethod
    def _blnhc_tensor_kind(layer_names: list[str]) -> str:
        names = [name.lower() for name in layer_names]
        if any("raw_key_cache" in name for name in names):
            return "raw_key"
        if any("compressed_key_cache" in name for name in names):
            return "compressed"
        if any(name.endswith(".ple") or ".ple." in name for name in names):
            return "ple"
        return ""

    def _build_cuda_blnhc_layout(self, kvcaches) -> None:
        """BLNHC whole-cache layout for qwen3.8-flash-next.

        Each stored object is one physical block of ``block_stride`` bytes.
        Copy starts at ``block0``, which already includes the row-tensor
        ``offset``. Compressed key and full attention share one block, so
        that object starts at the earlier ``block0``. ``raw_key_cache`` is
        omitted.
        """
        records = []
        for raw_tensor in self.kv_cache_config.kv_cache_tensors:
            layer_names = kv_cache_tensor_layer_names(raw_tensor)
            if not layer_names:
                continue
            kind = self._blnhc_tensor_kind(layer_names)
            layer_views: list[tuple[int, int, Optional[torch.Tensor]]] = []
            for index, layer_name in enumerate(layer_names):
                kv_layer = kvcaches.get(layer_name)
                if kv_layer is None:
                    continue
                tensor = kv_layer if isinstance(kv_layer, torch.Tensor) else None
                ptr = self._layer_data_ptr(kv_layer)
                if ptr is None:
                    logger.warning(f"unsupported kv_layer type: {type(kv_layer)}")
                    continue
                layer_views.append((index, ptr, tensor))
            if not layer_views:
                logger.warning(f"no kv cache tensor found for layers {layer_names}")
                continue
            block0, block_stride, layer_stride, offset = self._blnhc_block0(
                raw_tensor, layer_views
            )
            if not kind:
                specs = []
                layer_to_specs = layer_name_to_kv_cache_spec(self.kv_cache_config)
                for layer_name in layer_names:
                    specs.extend(layer_to_specs.get(layer_name, []))
                kind = (
                    "linear"
                    if any(isinstance(spec, MambaSpec) for spec in specs)
                    else "full_attn"
                )
            records.append(
                {
                    "kind": kind,
                    "layer_names": layer_names,
                    "block0": block0,
                    "block_stride": block_stride,
                    "layer_stride": layer_stride,
                    "offset": offset,
                    "size": int(raw_tensor.size),
                }
            )

        if not records:
            raise ValueError("BLNHC hybrid layout found no KV cache tensors")

        transferable = [rec for rec in records if rec["kind"] != "raw_key"]
        main_records = [
            rec for rec in transferable if rec["kind"] in ("linear", "full_attn", "ple")
        ]
        compressed_records = [
            rec for rec in transferable if rec["kind"] == "compressed"
        ]
        full_records = [rec for rec in transferable if rec["kind"] == "full_attn"]
        if not main_records or len(compressed_records) != 1 or len(full_records) != 1:
            raise ValueError(
                "BLNHC layout needs one full-attention tensor, one compressed "
                f"tensor, and linear/PLE tensors; kinds="
                f"{[rec['kind'] for rec in records]}"
            )
        main_stride = int(main_records[0]["layer_stride"])
        if any(int(rec["layer_stride"]) != main_stride for rec in main_records):
            raise ValueError(
                "linear attention, full attention, and PLE must share layer_stride: "
                + ", ".join(
                    f"{rec['kind']}={rec['layer_stride']}" for rec in main_records
                )
            )
        compressed = compressed_records[0]
        full = full_records[0]
        block_strides = {int(rec["block_stride"]) for rec in transferable}
        if len(block_strides) != 1:
            raise ValueError(
                f"BLNHC tensors have different block_stride: {block_strides}"
            )
        block_stride = block_strides.pop()
        layer_count = len(full["layer_names"])
        if layer_count <= 0 or any(
            len(rec["layer_names"]) != layer_count
            for rec in transferable
            if rec["kind"] in ("linear", "compressed")
        ):
            raise ValueError(
                "BLNHC linear, compressed, and full-attention tensors must "
                f"cover {layer_count} layers"
            )
        full_block0 = int(full["block0"])
        compressed_block0 = int(compressed["block0"])
        shared_base = min(full_block0, compressed_block0)
        shared_gap = abs(full_block0 - compressed_block0)
        if shared_gap >= block_stride:
            raise ValueError(
                "compressed key and full attention are not in one physical block: "
                f"full_block0={full_block0}, compressed_block0={compressed_block0}, "
                f"block_stride={block_stride}"
            )
        live_end = max(
            full_block0 + layer_count * int(full["layer_stride"]),
            compressed_block0 + layer_count * int(compressed["layer_stride"]),
        )
        if live_end - shared_base > block_stride:
            raise ValueError(
                "one block_stride cannot hold compressed key and full attention: "
                f"shared_base={shared_base}, live_end={live_end}, "
                f"block_stride={block_stride}"
            )
        for rec in transferable:
            if rec["kind"] == "compressed":
                continue
            live_span = len(rec["layer_names"]) * int(rec["layer_stride"])
            if live_span > block_stride:
                raise ValueError(
                    "BLNHC live data exceeds one physical block: "
                    f"kind={rec['kind']}, live_span={live_span}, "
                    f"block_stride={block_stride}"
                )

        group_block0: dict[int, int] = {}
        for group_id, group in enumerate(self.kv_cache_config.kv_cache_groups):
            role = kv_cache_group_external_role(group)
            if role == "raw_key":
                continue
            if role == "full_attn":
                group_block0[group_id] = shared_base
                continue
            matched = [
                rec
                for rec in transferable
                if rec["kind"] == role
                and set(rec["layer_names"]).intersection(group.layer_names)
            ]
            if len(matched) != 1:
                raise ValueError(
                    f"BLNHC group {group_id} role={role} matched "
                    f"{len(matched)} tensors"
                )
            group_block0[group_id] = int(matched[0]["block0"])

        self.blnhc_group_block0 = group_block0
        self.blnhc_block_stride = block_stride
        self.registration_buffers = []
        seen_buffers = set()
        for rec in transferable:
            backing = int(rec["block0"]) - int(rec["offset"])
            if backing < 0:
                raise ValueError(
                    "BLNHC backing base is negative: "
                    f"kind={rec['kind']}, block0={rec['block0']}, "
                    f"offset={rec['offset']}"
                )
            key = (backing, int(rec["size"]))
            if key in seen_buffers:
                continue
            seen_buffers.add(key)
            self.registration_buffers.append(key)

        self.layer_name_to_row = {}
        for rec in transferable:
            for layer_name in rec["layer_names"]:
                self.layer_name_to_row[layer_name] = 0
        logger.info(
            "BLNHC block-first layout: "
            f"block_stride={block_stride}, layer_stride={main_stride}, "
            f"compressed_layer_stride={int(compressed['layer_stride'])}, "
            f"shared_base={shared_base}, full_block0={full_block0}, "
            f"compressed_block0={compressed_block0}, "
            f"offsets={[(rec['kind'], rec['offset']) for rec in records]}, "
            f"groups={sorted(group_block0)}"
        )
        self._finalize_layout_arrays(
            [[shared_base]],
            [[block_stride]],
            [[block_stride]],
            [[block_stride]],
        )

    def extract_blnhc_block_addrs(
        self, vllm_block_ids: List[int], group_ids: List[int]
    ) -> np.ndarray:
        if len(vllm_block_ids) != len(group_ids):
            raise ValueError(
                "BLNHC block ids and group ids have different lengths: "
                f"{len(vllm_block_ids)} vs {len(group_ids)}"
            )
        ptrs = np.empty((len(vllm_block_ids), 1), dtype=np.uint64)
        stride = int(self.blnhc_block_stride)
        for index, (block_id, group_id) in enumerate(zip(vllm_block_ids, group_ids)):
            try:
                block0 = self.blnhc_group_block0[int(group_id)]
            except KeyError as exc:
                raise ValueError(
                    f"No BLNHC block base for group_id={group_id}"
                ) from exc
            ptrs[index, 0] = int(block0) + int(block_id) * stride
        return ptrs

    @staticmethod
    def _component_tensors(kv_layer) -> list[torch.Tensor]:
        if isinstance(kv_layer, torch.Tensor):
            return [kv_layer]
        if isinstance(kv_layer, (tuple, list)):
            return [tensor for tensor in kv_layer if isinstance(tensor, torch.Tensor)]
        return []

    def _components_fit_one_page(self, tensors: list[torch.Tensor], block_stride: int) -> bool:
        """True when every component of block 0 lives inside one physical page."""
        if not tensors or self.num_blocks <= 0:
            return False
        pointers = [int(tensor.data_ptr()) for tensor in tensors]
        if max(pointers) - min(pointers) >= block_stride:
            return False
        for tensor in tensors:
            if tensor.dim() == 0 or int(tensor.shape[0]) <= 1:
                return False
            if int(tensor.shape[0]) % self.num_blocks != 0:
                return False
            slots_per_block = int(tensor.shape[0]) // self.num_blocks
            step = int(tensor[1].data_ptr()) - int(tensor.data_ptr())
            if step <= 0 or step * slots_per_block != block_stride:
                return False
        return True

    def _npu_layer_block0(self, raw_tensor, index: int, tensors: list[torch.Tensor]) -> int:
        """Layer 0, block 0. A view pointer already includes ``offset``."""
        offset = self._require_tensor_offset(raw_tensor)
        layer_stride = int(raw_tensor.layer_stride)
        anchor = min(tensors, key=lambda tensor: int(tensor.data_ptr()))
        ptr = int(anchor.data_ptr())
        try:
            storage = int(anchor.untyped_storage().data_ptr())
        except (AttributeError, RuntimeError):
            storage = None
        origin = ptr + offset if storage is not None and storage == ptr else ptr
        return origin - index * layer_stride

    def _npu_qwen3_next_page_layout(self, kvcaches) -> bool:
        """Layer-first pages: one ``block_stride`` page per layer, on CUDA and NPU."""
        if self._layout_name() != "LBNHC":
            return False
        if not has_npu_qwen3_next_groups(self.kv_cache_config):
            return False
        saw_tensor = False
        for raw_tensor in self.kv_cache_config.kv_cache_tensors:
            layer_names = kv_cache_tensor_layer_names(raw_tensor)
            if not layer_names:
                continue
            if any(_layer_name_is_flash_next_only(name) for name in layer_names):
                return False
            block_stride = int(getattr(raw_tensor, "block_stride", 0) or 0)
            layer_stride = int(getattr(raw_tensor, "layer_stride", 0) or 0)
            if block_stride <= 0 or layer_stride != block_stride * self.num_blocks:
                logger.info(
                    "LBNHC layer-first skipped: "
                    f"block_stride={block_stride}, layer_stride={layer_stride}, "
                    f"num_blocks={self.num_blocks}"
                )
                return False
            checked = False
            for layer_name in layer_names:
                tensors = self._component_tensors(kvcaches.get(layer_name))
                if not tensors:
                    continue
                checked = True
                pointers = [int(tensor.data_ptr()) for tensor in tensors]
                if max(pointers) - min(pointers) >= block_stride:
                    logger.info(
                        "LBNHC layer-first skipped: components are not in one page, "
                        f"layer={layer_name}, block_stride={block_stride}"
                    )
                    return False
                break
            if not checked:
                logger.info(
                    "LBNHC layer-first skipped: "
                    f"no kv cache tensor for layers {layer_names}"
                )
                return False
            saw_tensor = True
        return saw_tensor

    def _prepare_npu_qwen3_next_pages(
        self, kvcaches
    ) -> tuple[int, int, int, dict[str, int], int]:
        """Return ``block0``, page size, layer stride, layer index, tensor count.

        Group tensors alias one allocation. ``block0`` already includes offset.
        """
        block0: Optional[int] = None
        block_stride = 0
        layer_stride = 0
        layer_count = 0
        backing_key: Optional[tuple[int, int]] = None
        layer_to_index: dict[str, int] = {}
        tensor_count = 0

        for raw_tensor in self.kv_cache_config.kv_cache_tensors:
            layer_names = kv_cache_tensor_layer_names(raw_tensor)
            if not layer_names:
                continue
            page = int(raw_tensor.block_stride)
            stride = int(raw_tensor.layer_stride)
            offset = self._require_tensor_offset(raw_tensor)
            present: list[tuple[int, list[torch.Tensor]]] = []
            for index, layer_name in enumerate(layer_names):
                tensors = self._component_tensors(kvcaches.get(layer_name))
                if tensors:
                    present.append((index, tensors))
            if not present:
                raise ValueError(
                    f"Qwen3-Next NPU tensor has no layer pointers: {layer_names}"
                )
            this_block0 = self._npu_layer_block0(
                raw_tensor, present[0][0], present[0][1]
            )
            for index, tensors in present[1:]:
                other = self._npu_layer_block0(raw_tensor, index, tensors)
                if other != this_block0:
                    raise ValueError(
                        "Qwen3-Next NPU layers are not placed on layer_stride: "
                        f"layer_index={index}, block0={other}, expected={this_block0}, "
                        f"layer_stride={stride}, offset={offset}"
                    )
            if this_block0 < 0:
                raise ValueError(
                    f"Qwen3-Next NPU block base is negative: block0={this_block0}, "
                    f"offset={offset}"
                )
            if len(layer_names) * stride > int(raw_tensor.size):
                raise ValueError(
                    "Qwen3-Next NPU layers exceed the raw allocation: "
                    f"layers={len(layer_names)}, layer_stride={stride}, "
                    f"raw_size={raw_tensor.size}"
                )
            this_key = (this_block0 - offset, int(raw_tensor.size))
            if block0 is None:
                block0 = this_block0
                block_stride = page
                layer_stride = stride
                layer_count = len(layer_names)
                backing_key = this_key
            elif (
                this_block0 != block0
                or page != block_stride
                or stride != layer_stride
                or len(layer_names) != layer_count
                or this_key != backing_key
            ):
                raise ValueError(
                    "Qwen3-Next NPU tensors do not alias one allocation: "
                    f"block0={this_block0}, expected={block0}, "
                    f"layers={len(layer_names)}, expected_layers={layer_count}"
                )
            for index, layer_name in enumerate(layer_names):
                previous = layer_to_index.get(layer_name)
                if previous is not None and previous != index:
                    raise ValueError(
                        "Qwen3-Next NPU layer is listed at two page indexes: "
                        f"layer={layer_name}, indexes={previous},{index}"
                    )
                layer_to_index[layer_name] = index
            tensor_count += 1

        if block0 is None or tensor_count == 0:
            raise ValueError("Qwen3-Next NPU layout found no KV cache tensors")
        return block0, block_stride, layer_stride, layer_to_index, tensor_count

    def _npu_qwen3_next_forward_rows(
        self, layer_to_index: dict[str, int], layer_count: int
    ) -> list[int]:
        """Physical page indexes sorted by the earliest model layer on that page."""

        def page_position(index: int) -> tuple[int, int]:
            positions = [
                self.layer_name_to_id.get(name, extract_layer_index(name))
                for name, page in layer_to_index.items()
                if page == index
            ]
            if not positions:
                return (index, index)
            return (min(positions), index)

        return sorted(range(layer_count), key=page_position)

    def _build_npu_qwen3_next_layout(self, kvcaches) -> None:
        """One object: every layer page of ``block_stride``."""
        block0, block_stride, layer_stride, layer_to_index, tensor_count = (
            self._prepare_npu_qwen3_next_pages(kvcaches)
        )
        layer_count = max(layer_to_index.values()) + 1
        bases = [block0 + index * layer_stride for index in range(layer_count)]
        layer_region = block_stride * self.num_blocks
        self.layer_name_to_row = {name: 0 for name in layer_to_index}
        logger.info(
            "LBNHC layer-first layout: "
            f"tensors={tensor_count}, layers={layer_count}, "
            f"block_stride={block_stride}, block0={block0}, "
            f"layer_stride={layer_stride}"
        )
        self._finalize_layout_arrays(
            [bases],
            [[layer_region] * layer_count],
            [[block_stride] * layer_count],
            [[block_stride] * layer_count],
        )
        # One object holds every page, even if the generic flag defaults on.
        self.use_layerwise = False

    def _build_npu_qwen3_next_layerwise_layout(self, kvcaches) -> None:
        """One shard per physical page, ordered by forward progress.

        The i-th layer of every group shares that page. ``wait_for_layer_load``
        runs on the full-attention layer and pulls in the next shard.
        """
        block0, block_stride, layer_stride, layer_to_index, tensor_count = (
            self._prepare_npu_qwen3_next_pages(kvcaches)
        )
        layer_count = max(layer_to_index.values()) + 1
        forward_pages = self._npu_qwen3_next_forward_rows(layer_to_index, layer_count)
        row_of_page = {
            physical: row for row, physical in enumerate(forward_pages)
        }
        layer_region = block_stride * self.num_blocks
        base_ptrs = []
        buffer_size_rows = []
        tensor_size_lists = []
        block_stride_lists = []
        for physical in forward_pages:
            base_ptrs.append([block0 + physical * layer_stride])
            buffer_size_rows.append([layer_region])
            tensor_size_lists.append([block_stride])
            block_stride_lists.append([block_stride])
        self.layer_name_to_row = {
            name: row_of_page[index] for name, index in layer_to_index.items()
        }
        logger.info(
            "LBNHC layer-first layerwise layout: "
            f"tensors={tensor_count}, rows={layer_count}, "
            f"block_stride={block_stride}, block0={block0}, "
            f"layer_stride={layer_stride}, "
            f"forward_pages={forward_pages}"
        )
        self._finalize_layout_arrays(
            base_ptrs,
            buffer_size_rows,
            tensor_size_lists,
            block_stride_lists,
        )

    def _build_layout(self, kvcaches):
        layout_name = self._layout_name()
        if layout_name == "BLNHC":
            # Block-first flash-next is whole-cache only.
            self._build_cuda_blnhc_layout(kvcaches)
            return
        if layout_name == "LBNHC" and self._npu_qwen3_next_page_layout(kvcaches):
            if self._layer_first_layerwise:
                self._build_npu_qwen3_next_layerwise_layout(kvcaches)
            else:
                self._build_npu_qwen3_next_layout(kvcaches)
            return
        if current_platform.is_cuda_alike():
            self._build_cuda_lbnhc_layout(kvcaches)
            return

        base_ptrs = []
        buffer_size_rows = []
        tensor_size_lists = []
        block_stride_lists = []
        self.layer_name_to_row: dict[str, int] = {}

        is_npu = current_platform.device_type == "npu"

        for raw_tensor in self.kv_cache_config.kv_cache_tensors:
            layer_names = kv_cache_tensor_layer_names(raw_tensor)
            if not layer_names:
                continue

            shared_specs, shared_ptrs = self._collect_shared_tensor_info(
                raw_tensor, kvcaches
            )

            if not shared_ptrs:
                logger.warning(f"no kv cache tensor found for shared layers {layer_names}")
                continue

            self._log_component_adjacency(raw_tensor, kvcaches)
            row_id = len(base_ptrs)
            mamba_specs = [s for s in shared_specs if isinstance(s, MambaSpec)]
            attn_specs = [s for s in shared_specs if isinstance(s, FullAttentionSpec)]

            # Ascend: hybrid → component_major, attn-only → attn_only, else contiguous.
            if is_npu and mamba_specs and attn_specs:
                self._append_ascend_component_major_layout(
                    raw_tensor,
                    shared_ptrs,
                    mamba_specs,
                    attn_specs,
                    base_ptrs,
                    buffer_size_rows,
                    tensor_size_lists,
                    block_stride_lists,
                )
            elif is_npu and attn_specs:
                self._append_ascend_attn_only_layout(
                    raw_tensor,
                    shared_ptrs,
                    attn_specs,
                    base_ptrs,
                    buffer_size_rows,
                    tensor_size_lists,
                    block_stride_lists,
                )
            else:
                self._append_contiguous_page_layout(
                    raw_tensor,
                    shared_ptrs,
                    base_ptrs,
                    buffer_size_rows,
                    tensor_size_lists,
                    block_stride_lists,
                )

            for layer_name in layer_names:
                self.layer_name_to_row[layer_name] = row_id

        self._finalize_layout_arrays(
            base_ptrs,
            buffer_size_rows,
            tensor_size_lists,
            block_stride_lists,
        )


class UCMHybridLinearAttentionConnector(UCMDirectConnector, SupportsHMA):
    """UCM connector for hybrid multi-group KV cache layouts.

    Merges the former UCMHMAConnector logic (group-aware hashing, two-stage
    lookup, per-group dispatch) with the HybridLinearAttentionLayout
    specialization for shared KV tensor pages.
    """

    @classmethod
    def supports_kv_cache_layout(cls, kv_cache_config) -> bool:
        if kv_cache_config is None:
            return False

        if has_cuda_hybrid_linear_attention_groups(kv_cache_config):
            return True
        if current_platform.device_type != "npu":
            return False

        layer_to_specs = layer_name_to_kv_cache_spec(kv_cache_config)
        for raw_tensor in kv_cache_config.kv_cache_tensors:
            shared_specs = [
                spec
                for layer_name in kv_cache_tensor_layer_names(raw_tensor)
                for spec in layer_to_specs.get(layer_name, [])
            ]
            if any(
                isinstance(spec, FullAttentionSpec) for spec in shared_specs
            ) and any(
                isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align"
                for spec in shared_specs
            ):
                return True

        return False

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        self._skip_null_vllm_blocks = True
        # group manager only lives on the scheduler side, where ``self._seed``
        # and ``self.request_hasher`` are populated by the parent ctor.
        self.group_manager: Optional[KVCacheGroupManager] = None
        if role == KVConnectorRole.SCHEDULER:
            self.group_manager = KVCacheGroupManager(
                kv_cache_config=kv_cache_config,
                request_hasher=self.request_hasher,
                base_seed=self._seed,
            )
            lcm_block_size = self.group_manager.lcm_block_size
            self.block_size = lcm_block_size
            self.hash_block_size = lcm_block_size
            self._bind_request_block_hasher()

        logger.info(f"{type(self).__name__} initialized")

    def get_block_size(self) -> int:
        if self.group_manager is not None:
            return self.group_manager.lcm_block_size
        return self.block_size

    def _create_kv_cache_layout(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> KVCacheLayout:
        return HybridLinearAttentionLayout(
            kv_caches,
            self.launch_config,
            self._vllm_config,
            self._kv_cache_config,
        )

    def _create_store(
        self,
        kv_cache_layout: Optional[KVCacheLayout],
        cpu_affinity_cores: Optional[list[int]] = None,
        tensor_size_list_override: Optional[list[int]] = None,
        shard_size_override: Optional[int] = None,
        block_size_override: Optional[int] = None,
        unique_id_suffix: str = "",
    ) -> UcmKVStoreBaseV1:
        if len(self.connector_configs) != 1:
            raise RuntimeError(
                f"Expected exactly one connector config, "
                f"but got {len(self.connector_configs)}: "
                f"{self.connector_configs}"
            )

        name = self.connector_configs[0]["ucm_connector_name"]
        module_path = self.connector_configs[0].get("ucm_connector_module_path", None)
        config = copy.deepcopy(self.connector_configs[0]["ucm_connector_config"])
        config.setdefault("share_buffer_enable", self.is_mla)
        self._set_default_shm_buffer_capacity(config)
        if "storage_backends" in config:
            backends = [path for path in config["storage_backends"].split(":")]
            config["storage_backends"] = backends
        config["unique_id"] = f"{self.unique_id}{unique_id_suffix}"
        if self._role == KVConnectorRole.WORKER:
            config["device_id"] = self.device_id
            tensor_size_list = _normalize_tensor_size_list(
                tensor_size_list_override
                if tensor_size_list_override is not None
                else kv_cache_layout.tensor_size_list
            )
            config["tensor_size_list"] = tensor_size_list * self.blocks_per_chunk
            shard_size = (
                shard_size_override
                if shard_size_override is not None
                else kv_cache_layout.shard_size
            )
            block_size = (
                block_size_override
                if block_size_override is not None
                else kv_cache_layout.block_size
            )
            config["shard_size"] = shard_size * self.blocks_per_chunk
            config["block_size"] = block_size * self.blocks_per_chunk
            self._publish_block_size(config["block_size"])
            config["local_rank_size"] = self.tp_size if self.is_mla else 1
            registration_buffers = getattr(kv_cache_layout, "registration_buffers", None)
            if registration_buffers:
                buffer_addrs = [int(addr) for addr, _ in registration_buffers]
                buffer_sizes = [int(size) for _, size in registration_buffers]
            else:
                buffer_addrs = kv_cache_layout.base_ptrs.reshape(-1).tolist()
                buffer_sizes = kv_cache_layout.buffer_sizes.reshape(-1).tolist()
            gpu_kv_buffer_set = set()
            gpu_kv_buffer_addrs = []
            gpu_kv_buffer_sizes = []
            for addr, size in zip(buffer_addrs, buffer_sizes):
                # Layerwise padding is store metadata only. Never register a
                # ghost (nullptr, zero-sized) slot as a real device buffer.
                if int(addr) == 0 or int(size) == 0:
                    continue
                key = (int(addr), int(size))
                if key in gpu_kv_buffer_set:
                    continue
                gpu_kv_buffer_set.add(key)
                gpu_kv_buffer_addrs.append(key[0])
                gpu_kv_buffer_sizes.append(key[1])
            config["gpu_kv_buffer_addrs"] = gpu_kv_buffer_addrs
            config["gpu_kv_buffer_sizes"] = gpu_kv_buffer_sizes
            if cpu_affinity_cores:
                config["cpu_affinity_cores"] = list(cpu_affinity_cores)
        elif self._gc_owner:
            bs = _scheduler_read_block_size()
            if bs is None:
                config_base = self.block_size * self.element_size * self.head_size
                bs = (
                    config_base
                    * self.num_layers
                    * (1 if self.is_mla else self.num_head * 2)
                    * self.blocks_per_chunk
                )
                logger.warning(f"Falling back to manual block_size estimate: {bs}")
            config["block_size"] = bs
        config["posix_gc_enable"] = self._gc_owner
        logger.info(f"create {name} with config: {config}")
        return UcmConnectorFactoryV1.create_connector(name, config, module_path)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self.kv_caches = kv_caches
        self.kv_cache_layout = self._create_kv_cache_layout(self.kv_caches)
        self.block_data_size = self.kv_cache_layout.block_size
        self.device = create_device()

        enable_affinity = _use_ucm_connector_cpu_affinity()
        worker_cores, store_cores = (
            self.device.split_cores(self.device_id) if enable_affinity else (None, None)
        )

        self.store = self._create_store(
            kv_cache_layout=self.kv_cache_layout,
            cpu_affinity_cores=store_cores,
        )

        if worker_cores:
            try:
                os.sched_setaffinity(0, worker_cores)
                logger.info(f"[VLLM CPU Affinity] Worker bound to cores {worker_cores}")
            except Exception as e:
                logger.warning(f"Failed to bind worker: {e}")

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.group_manager is not None, (
            "get_num_new_matched_tokens must be called on the scheduler-side "
            "connector, where the group manager is initialized."
        )

        lcm_block_size = self.group_manager.lcm_block_size
        assert num_computed_tokens % lcm_block_size == 0, (
            f"num_computed_tokens={num_computed_tokens} is not aligned to "
            f"lcm_block_size={lcm_block_size}"
        )
        hbm_hit_block_num = num_computed_tokens // lcm_block_size

        if self.persist_token_threshold > request.num_tokens:
            logger.info(
                f"Skip persistence: req {request.request_id}, "
                f"input tokens ({request.num_tokens}) < threshold "
                f"({self.persist_token_threshold})."
            )
            return 0, False

        try:
            group_ucm_block_ids = self.group_manager.compute_all_group_block_ids(
                request
            )
        except Exception as e:
            logger.error(
                f"request {request.request_id} hash error. " f"{type(e).__name__}: {e}"
            )
            return 0, False
        primary_full_attn = self.group_manager.primary_full_attn_group
        primary_block_ids = group_ucm_block_ids[primary_full_attn.group_id]

        # Pre-lookup reduction: leave at least recompute_tokens for vLLM to
        # recompute, so the batch isn't dispatched as uniform decode into FULL
        # cudagraph. For hybrid this also avoids looking up the last block(s)
        # whose mamba state may not be valid in HBM at dump time.
        recompute_tokens = self._get_full_hit_recompute_tokens()
        max_hit_lcm_blocks = max(
            0, (request.num_tokens - recompute_tokens) // lcm_block_size
        )
        total_lcm_blocks = request.num_tokens // lcm_block_size

        if max_hit_lcm_blocks < total_lcm_blocks:
            lookup_block_ids = []
            for gid, group in enumerate(self.group_manager.groups_by_id):
                ids = group_ucm_block_ids[gid]
                group_max = max_hit_lcm_blocks * (lcm_block_size // group.block_size)
                lookup_block_ids.append(ids[:group_max])
        else:
            lookup_block_ids = group_ucm_block_ids

        external_hit_tokens, external_hit_lcm_blocks, mamba_prefetch_hashes = (
            self.group_manager.lookup_external_hit_tokens(
                num_computed_tokens,
                lookup_block_ids,
                lambda block_ids: self._rank_consistency.lookup_on_prefix(
                    self.store, block_ids
                ),
                lambda block_ids: self._rank_consistency.lookup_on_reverse(
                    self.store, block_ids
                ),
            )
        )

        if (
            self.enable_record_traces
            and request.request_id not in self.requests_meta
            and len(primary_block_ids) > 0
        ):
            hex_block_ids = [b.hex() for b in primary_block_ids]
            logger.info_once(
                f"timestamp: {time.perf_counter()}, "
                f"input_length: {request.num_tokens}, "
                f"output_length: {request.max_tokens}, "
                f"ucm_block_ids: {hex_block_ids}"
            )

        total_hit_block_num = hbm_hit_block_num + external_hit_lcm_blocks

        # GC heat update for all hit blocks across ranks.
        total_hit_tokens = total_hit_block_num * lcm_block_size
        hbm_hit_full_attn = num_computed_tokens // primary_full_attn.block_size
        total_hit_full_attn = total_hit_tokens // primary_full_attn.block_size
        all_hit_full_attn = primary_block_ids[0:total_hit_full_attn]
        hbm_full_attn = primary_block_ids[0:hbm_hit_full_attn]
        if hbm_full_attn:
            self.store.prefetch(hbm_full_attn)
        if mamba_prefetch_hashes:
            self.store.prefetch(mamba_prefetch_hashes)
        # MLA full-attn is TP-replicated (shared hash), no per-rank entries to prefetch.
        # Only mamba blocks have per-rank entries needing heat update.
        per_rank_hashes = mamba_prefetch_hashes
        if not self.is_mla:
            per_rank_hashes = all_hit_full_attn + mamba_prefetch_hashes
        self._prefetch_all_rank_hashes(per_rank_hashes)

        if len(primary_block_ids) > 0:
            ucmmetrics.update_stats(
                {
                    "interval_lookup_hit_rates": external_hit_lcm_blocks
                    * lcm_block_size
                    / (len(primary_block_ids) * primary_full_attn.block_size)
                },
            )

        # No post-lookup workaround: pre-lookup truncation already ensures
        # total_hit_tokens == external_hit_tokens, and the mamba state at
        # total_hit_tokens is a position the store actually verified.
        num_total_hit_tokens = total_hit_block_num * lcm_block_size

        logger.info(
            "HLA lookup: "
            f"request_id={request.request_id}, "
            f"num_tokens={request.num_tokens}, "
            f"computed_tokens={num_computed_tokens}, "
            f"lcm_block_size={lcm_block_size}, "
            f"recompute_tokens={recompute_tokens}, "
            f"max_hit_lcm_blocks={max_hit_lcm_blocks}, "
            f"primary_blocks={len(primary_block_ids)}, "
            f"hit_hbm_lcm_blocks={hbm_hit_block_num}, "
            f"hit_external_tokens={external_hit_tokens}, "
            f"hit_external_lcm_blocks={external_hit_lcm_blocks}"
        )

        self.requests_meta[request.request_id] = HLARequestMeta(
            ucm_block_ids=primary_block_ids,
            hbm_hit_block_num=hbm_hit_block_num,
            total_hit_block_num=total_hit_block_num,
            num_token_ids=len(request.all_token_ids),
            token_processed=num_total_hit_tokens,
            group_ucm_block_ids=group_ucm_block_ids,
            group_vllm_block_ids=[[] for _ in range(self.group_manager.num_groups)],
        )

        return external_hit_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        req_meta = self.requests_meta.get(request.request_id)
        if req_meta is None:
            return
        assert isinstance(req_meta, HLARequestMeta)
        block_ids = blocks.get_block_ids()
        if self.group_manager is not None:
            assert len(block_ids) == self.group_manager.num_groups, (
                f"allocated block group count {len(block_ids)} does not match "
                f"HLA group count {self.group_manager.num_groups}"
            )
        req_meta.group_vllm_block_ids = [list(group) for group in block_ids]

    def _append_mamba_align_state_block(
        self,
        dst_ucm_block_ids: list[bytes],
        dst_vllm_block_ids: list[int],
        req_meta: "HLARequestMeta",
        request_id: str,
        gid: int,
        seq_len: int,
        reason: str,
        dst_group_ids: Optional[list[int]] = None,
    ) -> None:
        group = self.group_manager.groups_by_id[gid]
        state_idx = max((seq_len - 1) // group.block_size, 0)
        vllm_state_idx = state_idx
        if reason == "load":
            block_ids = req_meta.group_vllm_block_ids[gid]
            for i in range(len(block_ids) - 1, -1, -1):
                if block_ids[i] != 0:
                    vllm_state_idx = i
                    break
        try:
            vllm_block_id = req_meta.group_vllm_block_ids[gid][vllm_state_idx]
        except IndexError:
            logger.error(
                "HLA mamba-align state vLLM block missing: "
                f"request_id={request_id}, group_id={gid}, reason={reason}, "
                f"seq_len={seq_len}, state_idx={state_idx}, "
                f"vllm_state_idx={vllm_state_idx}, "
                f"num_vllm_blocks={len(req_meta.group_vllm_block_ids[gid])}"
            )
            return
        if vllm_block_id == 0:
            return
        ucm_block_id = self.group_manager.compute_mamba_align_state_hash(
            group, seq_len, req_meta.group_ucm_block_ids
        )
        if ucm_block_id is None:
            logger.error(
                "HLA mamba-align state hash missing: "
                f"request_id={request_id}, group_id={gid}, reason={reason}, "
                f"seq_len={seq_len}, state_idx={state_idx}"
            )
            return
        dst_ucm_block_ids.append(ucm_block_id)
        dst_vllm_block_ids.append(vllm_block_id)
        if dst_group_ids is not None:
            dst_group_ids.append(gid)

    def _generate_hla_dispatch_meta(
        self,
        req_meta: "HLARequestMeta",
        new_tokens: int,
        new_vllm_block_ids_per_group: tuple[list[int], ...],
        need_load: bool = True,
        request_id: str = "",
        incoming_block_ids_are_full: bool = False,
    ) -> HLARequestDispatchMeta:
        """Build a flat (ucm, vllm) block id pair list across all groups."""
        assert self.group_manager is not None
        groups_by_id = self.group_manager.groups_by_id
        num_groups = self.group_manager.num_groups
        lcm_block_size = self.group_manager.lcm_block_size

        assert len(new_vllm_block_ids_per_group) == num_groups, (
            f"new_vllm_block_ids_per_group length "
            f"{len(new_vllm_block_ids_per_group)} does not match "
            f"num_groups {num_groups}"
        )
        for gid in range(num_groups):
            incoming_vllm_block_ids = list(new_vllm_block_ids_per_group[gid])
            existing_vllm_block_ids = req_meta.group_vllm_block_ids[gid]
            if incoming_block_ids_are_full:
                req_meta.group_vllm_block_ids[gid] = incoming_vllm_block_ids
            elif not existing_vllm_block_ids:
                req_meta.group_vllm_block_ids[gid] = incoming_vllm_block_ids
            elif incoming_vllm_block_ids:
                suffix_len = len(incoming_vllm_block_ids)
                if existing_vllm_block_ids[-suffix_len:] != incoming_vllm_block_ids:
                    existing_vllm_block_ids.extend(incoming_vllm_block_ids)

        load_ucm_block_ids: list[bytes] = []
        load_vllm_block_ids: list[int] = []
        load_block_group_ids: list[int] = []
        dump_ucm_block_ids: list[bytes] = []
        dump_vllm_block_ids: list[int] = []
        dump_block_group_ids: list[int] = []

        external_hit_lcm_blocks = (
            req_meta.total_hit_block_num - req_meta.hbm_hit_block_num
        )
        hbm_hit_tokens = req_meta.hbm_hit_block_num * lcm_block_size
        total_hit_tokens = req_meta.total_hit_block_num * lcm_block_size

        if need_load and external_hit_lcm_blocks > 0:
            # Pass 1: full-attention blocks first (for MLA rank-0-only dump)
            for gid, group in enumerate(groups_by_id):
                if not group.is_full_attention:
                    continue
                load_tok_start = hbm_hit_tokens
                load_tok_end = total_hit_tokens
                start_blk = load_tok_start // group.block_size
                end_blk = load_tok_end // group.block_size
                if start_blk >= end_blk:
                    continue
                extend_non_null(
                    load_ucm_block_ids,
                    load_vllm_block_ids,
                    req_meta.group_ucm_block_ids[gid][start_blk:end_blk],
                    req_meta.group_vllm_block_ids[gid][start_blk:end_blk],
                    load_block_group_ids,
                    gid,
                )
            load_full_attn_count = len(load_ucm_block_ids) if self.is_mla else 0
            # Pass 2: mamba state blocks, including the one-layer PLE group.
            for gid, group in enumerate(groups_by_id):
                if group.external_role not in ("linear", "ple"):
                    continue
                self._append_mamba_align_state_block(
                    load_ucm_block_ids,
                    load_vllm_block_ids,
                    req_meta,
                    request_id,
                    gid,
                    total_hit_tokens,
                    "load",
                    load_block_group_ids,
                )
        else:
            load_full_attn_count = 0

        if req_meta.token_processed < req_meta.num_token_ids:
            dump_tok_start = req_meta.token_processed
            dump_tok_end = min(
                req_meta.token_processed + new_tokens, req_meta.num_token_ids
            )
            first_lcm_b = (dump_tok_start // lcm_block_size + 1) * lcm_block_size
            last_lcm_b = (dump_tok_end // lcm_block_size) * lcm_block_size

            # Pass 1: full-attention blocks first
            for gid, group in enumerate(groups_by_id):
                if not group.is_full_attention:
                    continue
                start_blk = dump_tok_start // group.block_size
                end_blk = dump_tok_end // group.block_size
                if start_blk >= end_blk:
                    continue
                extend_non_null(
                    dump_ucm_block_ids,
                    dump_vllm_block_ids,
                    req_meta.group_ucm_block_ids[gid][start_blk:end_blk],
                    req_meta.group_vllm_block_ids[gid][start_blk:end_blk],
                    dump_block_group_ids,
                    gid,
                )
            dump_full_attn_count = len(dump_ucm_block_ids) if self.is_mla else 0
            # Pass 2: mamba state blocks, including the one-layer PLE group.
            for gid, group in enumerate(groups_by_id):
                if group.external_role not in ("linear", "ple"):
                    continue
                if dump_tok_end != last_lcm_b or last_lcm_b < first_lcm_b:
                    continue
                self._append_mamba_align_state_block(
                    dump_ucm_block_ids,
                    dump_vllm_block_ids,
                    req_meta,
                    request_id,
                    gid,
                    last_lcm_b,
                    "dump",
                    dump_block_group_ids,
                )
        else:
            dump_full_attn_count = 0

        req_meta.token_processed += new_tokens

        if (
            load_ucm_block_ids
            or dump_ucm_block_ids
            or new_tokens >= lcm_block_size
        ):
            logger.info(
                "HLA dispatch: "
                f"request_id={request_id}, new_tokens={new_tokens}, "
                f"token_processed={req_meta.token_processed}, "
                f"num_token_ids={req_meta.num_token_ids}, "
                f"external_hit_lcm_blocks={external_hit_lcm_blocks}, "
                f"load_blocks={len(load_ucm_block_ids)}, "
                f"dump_blocks={len(dump_ucm_block_ids)}, "
                f"dump_vllm_blocks={dump_vllm_block_ids[:4]}"
            )

        return HLARequestDispatchMeta(
            load_block_ids=(load_ucm_block_ids, load_vllm_block_ids),
            dump_block_ids=(dump_ucm_block_ids, dump_vllm_block_ids),
            load_full_attn_count=load_full_attn_count,
            dump_full_attn_count=dump_full_attn_count,
            load_block_group_ids=load_block_group_ids,
            dump_block_group_ids=dump_block_group_ids,
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        assert self.group_manager is not None
        num_groups = self.group_manager.num_groups
        empty_per_group: tuple[list[int], ...] = tuple([] for _ in range(num_groups))

        requests_dispatch_meta: dict[str, HLARequestDispatchMeta] = {}

        for request in scheduler_output.scheduled_new_reqs:
            request_id = request.req_id
            req_meta = self.requests_meta.get(request_id)
            if req_meta is None:
                continue
            assert isinstance(req_meta, HLARequestMeta)
            requests_dispatch_meta[request_id] = self._generate_hla_dispatch_meta(
                req_meta,
                scheduler_output.num_scheduled_tokens[request_id],
                request.block_ids,
                request_id=request_id,
                incoming_block_ids_are_full=True,
            )

        # Same three situations as the parent: chunked prefill (dump only),
        # resumed (load + dump), decode (no-op).
        scheduled_cached_reqs = scheduler_output.scheduled_cached_reqs
        if not isinstance(scheduled_cached_reqs, list):
            for i, request_id in enumerate(scheduled_cached_reqs.req_ids):
                req_meta = self.requests_meta.get(request_id)
                if req_meta is None:
                    continue
                assert isinstance(req_meta, HLARequestMeta)
                raw_new_block_ids = scheduled_cached_reqs.new_block_ids[i]
                new_block_ids = (
                    empty_per_group if raw_new_block_ids is None else raw_new_block_ids
                )
                if hasattr(scheduled_cached_reqs, "resumed_from_preemption"):
                    resumed_from_preemption = (
                        scheduled_cached_reqs.resumed_from_preemption[i]
                    )
                else:
                    resumed_from_preemption = (
                        request_id in scheduled_cached_reqs.resumed_req_ids
                    )
                requests_dispatch_meta[request_id] = self._generate_hla_dispatch_meta(
                    req_meta,
                    scheduler_output.num_scheduled_tokens[request_id],
                    new_block_ids,
                    resumed_from_preemption,
                    request_id=request_id,
                    incoming_block_ids_are_full=resumed_from_preemption,
                )
        else:
            for request in scheduled_cached_reqs:
                request_id = request.req_id
                req_meta = self.requests_meta.get(request_id)
                if req_meta is None:
                    continue
                assert isinstance(req_meta, HLARequestMeta)
                requests_dispatch_meta[request_id] = self._generate_hla_dispatch_meta(
                    req_meta,
                    scheduler_output.num_scheduled_tokens[request_id],
                    request.new_block_ids,
                    request.resumed_from_preemption,
                    request_id=request_id,
                    incoming_block_ids_are_full=request.resumed_from_preemption,
                )

        for request_id in scheduler_output.finished_req_ids:
            self.requests_meta.pop(request_id, None)

        return UCMConnectorMetadata(
            requests_dispatch_meta,
            scheduler_output.preempted_req_ids or set(),
        )

    def _mla_split_scope(self, ucm_ids, vllm_ids, full_attn_count, is_dump):
        """Split into MLA/KDA and apply rank scoping for MLA hybrid.

        Returns ``(rank0_ucm, scoped_ucm, scoped_vllm)`` where *rank0_ucm*
        holds the rank-0 hashes of the blocks that will actually be stored
        (for the rank-consistency tracker), *scoped_ucm* holds the per-rank
        store keys, and *scoped_vllm* holds the matching vLLM block IDs.
        """
        n = full_attn_count
        mla_ucm, kda_ucm = ucm_ids[:n], ucm_ids[n:]
        mla_vllm, kda_vllm = vllm_ids[:n], vllm_ids[n:]
        is_rank0 = self.tp_rank % self.tp_size == 0
        # MLA: shared hash for all ranks; KDA: rank0 shared, non-rank0 per-rank hash
        if is_rank0:
            kda_scoped = kda_ucm
        else:
            kda_scoped = [self.request_hasher(b) for b in kda_ucm]
        if is_dump and not is_rank0:
            return kda_ucm, kda_scoped, kda_vllm
        return mla_ucm + kda_ucm, mla_ucm + kda_scoped, mla_vllm + kda_vllm

    def _scope_blocks(self, ucm_ids, vllm_ids, full_attn_count, is_dump):
        """Rank-scope block IDs for dump or load.

        Returns ``(rank0_ucm, scoped_ucm, scoped_vllm)`` where *rank0_ucm*
        is the rank-0 hash (for tracker clear/mark), *scoped_ucm* is the
        per-rank store key, and *scoped_vllm* is the matching vLLM block IDs.
        """
        n = int(full_attn_count) if full_attn_count else 0
        if self.is_mla:
            return self._mla_split_scope(ucm_ids, vllm_ids, n, is_dump)
        if self.tp_rank % self.tp_size == 0:
            return ucm_ids, ucm_ids, vllm_ids
        scoped = [self.request_hasher(b) for b in ucm_ids]
        return ucm_ids, scoped, vllm_ids

    def _aligned_group_ids(self, group_ids, scoped_vllm, original_vllm) -> list[int]:
        if len(scoped_vllm) == len(original_vllm):
            return list(group_ids)
        if list(scoped_vllm) == list(original_vllm)[-len(scoped_vllm) :]:
            return list(group_ids)[-len(scoped_vllm) :]
        raise RuntimeError("BLNHC group ids do not line up with scoped blocks")

    def _transfer_block_addrs(self, scoped_vllm, group_ids, original_vllm):
        layout = self.kv_cache_layout
        if not getattr(layout, "blnhc_group_block0", None):
            return layout.extract_block_addrs(scoped_vllm)
        if group_ids is None:
            raise RuntimeError("BLNHC transfer is missing per-block group ids")
        aligned = self._aligned_group_ids(group_ids, scoped_vllm, original_vllm)
        return layout.extract_blnhc_block_addrs(scoped_vllm, aligned)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Bulk load override: MLA blocks shared hash, KDA blocks per-rank hash."""
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)
        load_requests = sum(
            1
            for request in metadata.request_meta.values()
            if len(request.load_block_ids[0]) > 0
        )
        if load_requests:
            logger.info(
                "HLA load begin: "
                f"requests={len(metadata.request_meta)}, "
                f"with_load_blocks={load_requests}, "
                f"block_data_size={self.block_data_size}"
            )
        request_to_task: dict[str, Task] = {}
        is_load = False
        num_loaded_block = 0
        num_loaded_request = 0
        load_start_time = time.perf_counter() * 1000
        request_to_load_blocks: dict[str, int] = {}
        all_load_ucm_ids: list[bytes] = []
        all_load_vllm_ids: list[int] = []
        # Ensure do_mamba_copy_block (from preprocess_mamba, compute stream)
        # has completed before submitting load DMA (store stream).  Without
        # this, the copy may land after the load and clobber loaded data.
        # At this point the previous step's forward is done, so the only
        # pending compute op is the mamba state copy — sync overhead is
        # negligible.
        self.device.synchronize()
        for request_id, request in metadata.request_meta.items():
            if len(request.load_block_ids[0]) == 0:
                continue
            is_load = True
            num_loaded_block += len(request.load_block_ids[0])
            num_loaded_request += 1
            n = getattr(request, "load_full_attn_count", 0)
            _, scoped_ucm, scoped_vllm = self._scope_blocks(
                request.load_block_ids[0], request.load_block_ids[1], n, is_dump=False
            )
            if not scoped_ucm:
                num_loaded_block -= len(request.load_block_ids[0])
                num_loaded_request -= 1
                continue
            num_loaded_block -= len(request.load_block_ids[0]) - len(scoped_ucm)
            try:
                ptrs = self._transfer_block_addrs(
                    scoped_vllm,
                    getattr(request, "load_block_group_ids", None),
                    request.load_block_ids[1],
                )
                ptrs = ptrs.reshape(ptrs.shape[0], -1)
                shard_indexs = [0] * len(scoped_ucm)
                task = self._rank_consistency.submit_load(
                    self.store,
                    {request_id: request.load_block_ids[0]},
                    scoped_ucm,
                    shard_indexs,
                    ptrs,
                )
                request_to_task[request_id] = task
                request_to_load_blocks[request_id] = len(scoped_ucm)
            except Exception as e:
                logger.error(
                    f"request {request_id} submit load task error. "
                    f"{type(e).__name__}: {e}"
                )
                self._record_load_error(
                    "connector_load_submit_errors_total",
                    metadata.request_meta[request_id].load_block_ids[1]
                    + metadata.request_meta[request_id].dump_block_ids[1],
                )
                self._connector_worker_meta.mark_failed(request_id)
                num_loaded_block -= len(scoped_ucm)

        for request_id, task in request_to_task.items():
            try:
                self._rank_consistency.wait_load(task)
            except Exception as e:
                logger.error(
                    f"request {request_id} wait load task error. "
                    f"{type(e).__name__}: {e}"
                )
                self._record_load_error(
                    "connector_load_wait_errors_total",
                    metadata.request_meta[request_id].load_block_ids[1]
                    + metadata.request_meta[request_id].dump_block_ids[1],
                )
                self._connector_worker_meta.mark_failed(request_id)
                num_loaded_block -= request_to_load_blocks.get(request_id, 0)

        if is_load:
            load_end_time = time.perf_counter() * 1000
            load_duration_ms = load_end_time - load_start_time
            load_bytes = num_loaded_block * self.block_data_size
            load_speed = load_bytes / max(load_duration_ms, 1) / 1024 / 1024
            ucmmetrics.update_stats(
                {
                    "load_requests_num": num_loaded_request,
                    "load_blocks_num": num_loaded_block,
                    "load_duration": load_duration_ms,
                    "load_speed": load_speed,
                    "load_bytes_total": load_bytes,
                }
            )

    def wait_for_save(self) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        total_ucm_block_ids: list[bytes] = []
        total_vllm_block_ids: list[int] = []
        total_group_ids: list[int] = []
        block_ids_by_request: dict[str, set[bytes]] = {}
        num_saved_block = 0
        skipped_empty = 0
        skipped_scope = 0
        for request_id, request in metadata.request_meta.items():
            if len(request.dump_block_ids[0]) == 0:
                skipped_empty += 1
                continue
            n = getattr(request, "dump_full_attn_count", 0)
            rank0_ucm, scoped_ucm, scoped_vllm = self._scope_blocks(
                request.dump_block_ids[0], request.dump_block_ids[1], n, is_dump=True
            )
            if not scoped_ucm:
                skipped_scope += 1
                logger.info(
                    "HLA dump scoped away: "
                    f"request_id={request_id}, "
                    f"dump_blocks={len(request.dump_block_ids[0])}, "
                    f"dump_full_attn_count={n}"
                )
                continue
            block_ids_by_request[request_id] = set(rank0_ucm)
            num_saved_block += len(scoped_ucm)
            total_ucm_block_ids.extend(scoped_ucm)
            total_vllm_block_ids.extend(scoped_vllm)
            if getattr(self.kv_cache_layout, "blnhc_group_block0", None):
                total_group_ids.extend(
                    self._aligned_group_ids(
                        getattr(request, "dump_block_group_ids", []),
                        scoped_vllm,
                        request.dump_block_ids[1],
                    )
                )

        if not total_ucm_block_ids:
            if skipped_scope:
                logger.info(
                    "HLA dump skipped: "
                    f"requests={len(metadata.request_meta)}, "
                    f"empty={skipped_empty}, scoped_away={skipped_scope}"
                )
            return

        event_handle = 0
        try:
            total_ptrs = self._transfer_block_addrs(
                total_vllm_block_ids,
                total_group_ids,
                total_vllm_block_ids,
            )
            total_ptrs = total_ptrs.reshape(total_ptrs.shape[0], -1)
            logger.info(
                "HLA dump submit: "
                f"blocks={len(total_ucm_block_ids)}, "
                f"ptr_cols={total_ptrs.shape[-1]}, "
                f"block_data_size={self.block_data_size}, "
                f"first_hash={_preview_block_hash(total_ucm_block_ids[0])}"
            )
            shard_indexs = [0] * len(total_ucm_block_ids)
            event_handle = self._get_dump_event_handle()
            save_start_time = time.perf_counter() * 1000
            task = self._rank_consistency.submit_dump(
                self.store,
                block_ids_by_request,
                total_ucm_block_ids,
                shard_indexs,
                total_ptrs,
                event_handle,
            )
        except Exception as e:
            logger.error(f"dump kv cache failed. {type(e).__name__}: {e}")
            if self.enable_event_sync and event_handle and self.device is not None:
                self.device.destroy_event_handle(event_handle)
            self._rank_consistency.finish_dump(set(block_ids_by_request))
            return

        try:
            self._rank_consistency.wait_dump(task)
            save_end_time = time.perf_counter() * 1000
        except Exception as e:
            logger.error_limit(
                f"wait for dump kv cache failed. {type(e).__name__}: {e}"
            )
            self._rank_consistency.finish_dump(set(block_ids_by_request))
            return
        finally:
            if self.enable_event_sync and event_handle and self.device is not None:
                self.device.destroy_event_handle(event_handle)

        self._rank_consistency.finish_dump(set(block_ids_by_request))
        save_bytes = num_saved_block * self.block_data_size
        ucmmetrics.update_stats(
            {
                "save_duration": save_end_time - save_start_time,
                "save_bytes_total": save_bytes,
            }
        )

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, object] | None]:
        return False, None


class UCMHybridLinearAttentionLayerWiseConnector(UCMHybridLinearAttentionConnector):
    """Layerwise connector for full-attention + linear-attention hybrid layouts."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self.launch_config = copy.deepcopy(self.launch_config)
        self.launch_config["use_layerwise"] = True
        self.use_layerwise = True
        self.load_tasks: dict[int, dict[str, Task]] = defaultdict(dict)
        self.dump_tasks: dict[int, list[PendingDumpTask]] = defaultdict(list)
        self.request_data: list[tuple[str, list[bytes], list[bytes], list[int]]] = []
        self._failure_req_ids: set[str] = set()
        self._submitted_load_rows: set[int] = set()
        # A hybrid KV row can be visited more than once in one model-runner
        # batch (for example by speculative decoding). Persist its first
        # successful submission only and reset this state for every batch.
        self._dumped_row_ids: set[int] = set()
        self._dump_transfer_data: (
            tuple[list[bytes], list[int], set[str], dict[str, set[bytes]]] | None
        ) = None
        self._row_shard_size = 0
        self._layerwise_load_bytes = 0
        self._layerwise_load_bytes_recorded = False
        self._layerwise_save_bytes = 0
        self._load_block_counts: dict[str, int] = {}
        prefetch_rows_config = self.launch_config.get(
            "hybrid_layerwise_prefetch_rows", 2
        )
        try:
            self._load_prefetch_rows = max(1, int(prefetch_rows_config))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid hybrid_layerwise_prefetch_rows=%r; fallback to 2.",
                prefetch_rows_config,
            )
            self._load_prefetch_rows = 2
        self.is_save = False
        self.need_load = False
        logger.info(
            "Init UCMHybridLinearAttentionLayerWiseConnector "
            f"with prefetch_rows={self._load_prefetch_rows}."
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if has_ucm_sparse() and os.getenv("VLLM_HASH_ATTENTION") == "1":
            for layer_name, value in kv_caches.items():
                kv_cache, _ = value
                self.kv_caches[layer_name] = kv_cache
        else:
            self.kv_caches = kv_caches

        self.kv_cache_layout = self._create_kv_cache_layout(self.kv_caches)
        self.block_data_size = int(self.kv_cache_layout.tensor_size_lists.sum())
        self.layer_name_to_id = self.kv_cache_layout.layer_name_to_id
        self.layer_ids = sorted(set(self.layer_name_to_id.values()))
        self.first_layer_id = self.layer_ids[0]
        self.layer_name_to_row = getattr(self.kv_cache_layout, "layer_name_to_row", {})
        self.row_ids = sorted(set(self.layer_name_to_row.values()))
        row_tensor_size_lists = getattr(
            self.kv_cache_layout, "row_tensor_size_lists", []
        )
        if not self.row_ids:
            raise RuntimeError("Hybrid layerwise layout has no cache rows.")
        if max(self.row_ids) >= len(row_tensor_size_lists):
            raise RuntimeError(
                "Hybrid layerwise row mapping is inconsistent with layout rows: "
                f"row_ids={_short_list(self.row_ids)}, "
                f"row_tensor_size_lists={len(row_tensor_size_lists)}"
            )

        first_row_id = self.row_ids[0]
        row_tensor_size_list = list(row_tensor_size_lists[first_row_id])
        row_shard_size = sum(row_tensor_size_list)
        self._row_shard_size = row_shard_size
        for row_id in self.row_ids:
            tensor_size_list = list(row_tensor_size_lists[row_id])
            if tensor_size_list != row_tensor_size_list:
                raise RuntimeError(
                    "Hybrid layerwise rows must share the same tensor layout for "
                    "one row-sharded store: "
                    f"row_id={row_id}, tensor_size_list={tensor_size_list}, "
                    f"expected={row_tensor_size_list}"
                )

        self.device = create_device()

        enable_affinity = _use_ucm_connector_cpu_affinity()
        worker_cores, store_cores = (
            self.device.split_cores(self.device_id) if enable_affinity else (None, None)
        )

        self.store = self._create_store(
            kv_cache_layout=self.kv_cache_layout,
            cpu_affinity_cores=store_cores,
            tensor_size_list_override=row_tensor_size_list,
            shard_size_override=row_shard_size,
            block_size_override=row_shard_size * (max(self.row_ids) + 1),
        )

        if worker_cores:
            try:
                os.sched_setaffinity(0, worker_cores)
                logger.info(f"[VLLM CPU Affinity] Worker bound to cores {worker_cores}")
            except Exception as e:
                logger.warning(f"Failed to bind worker: {e}")

        row_to_layers: dict[int, list[str]] = defaultdict(list)
        for layer_name, row_id in self.layer_name_to_row.items():
            row_to_layers[row_id].append(layer_name)
        self.row_save_layer = {
            row_id: max(
                layer_names,
                key=lambda name: self.layer_name_to_id.get(name, self.first_layer_id),
            )
            for row_id, layer_names in row_to_layers.items()
        }
        logger.info(
            "Hybrid layerwise layout: "
            f"rows={len(self.row_ids)}, row_ids={_short_list(self.row_ids)}, "
            f"row_shard_size={row_shard_size}, "
            f"row_tensor_size_list={row_tensor_size_list}, "
            f"row_save_layers={len(self.row_save_layer)}"
        )

    def _mark_load_failed(
        self,
        metadata: "UCMConnectorMetadata",
        request_id: str,
    ) -> None:
        request_meta = metadata.request_meta.get(request_id)
        if request_meta is not None:
            self._invalid_block_ids.update(request_meta.load_block_ids[1])
        self._failure_req_ids.add(request_id)
        self._connector_worker_meta.mark_failed(request_id)

    def _submit_request_load_tasks_for_row(
        self,
        row_id: int,
        metadata: "UCMConnectorMetadata",
    ) -> None:
        for (
            request_id,
            ucm_block_ids,
            store_block_ids,
            vllm_block_ids,
        ) in self.request_data:
            if request_id in self._failure_req_ids:
                continue
            try:
                row_ptrs = self.kv_cache_layout.extract_block_addrs_for_row(
                    vllm_block_ids, row_id
                )
                shard_indexs = [row_id] * len(store_block_ids)
                task = self._rank_consistency.submit_load(
                    self.store,
                    {request_id: ucm_block_ids},
                    store_block_ids,
                    shard_indexs,
                    row_ptrs,
                )
                self.load_tasks[row_id][request_id] = task
            except Exception as e:
                logger.error(
                    f"request {request_id} submit load task for row {row_id} "
                    f"error. {type(e).__name__}: {e}"
                )
                self._mark_load_failed(metadata, request_id)
        self._submitted_load_rows.add(row_id)

    def _submit_request_load_tasks_for_row_once(
        self,
        row_id: int,
        metadata: "UCMConnectorMetadata",
    ) -> None:
        if row_id in self._submitted_load_rows:
            return
        self._submit_request_load_tasks_for_row(row_id, metadata)

    def _wait_row_load(self, row_id: int, metadata: "UCMConnectorMetadata") -> int:
        """Pop and wait for a row's per-request load tasks, marking failures."""
        row_tasks = self.load_tasks.pop(row_id, {})
        for request_id, task in row_tasks.items():
            try:
                self._rank_consistency.wait_load(task)
            except Exception as e:
                logger.error(
                    f"request {request_id} wait row {row_id} "
                    f"load failed. {type(e).__name__}: {e}"
                )
                self._mark_load_failed(metadata, request_id)
            else:
                self._layerwise_load_bytes += (
                    self._load_block_counts.get(request_id, 0) * self._row_shard_size
                )
        return len(row_tasks)

    def _record_layerwise_load_bytes(self) -> None:
        if self._layerwise_load_bytes_recorded:
            return
        ucmmetrics.update_stats({"load_bytes_total": self._layerwise_load_bytes})
        self._layerwise_load_bytes_recorded = True

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)
        self.load_tasks.clear()
        self.request_data.clear()
        self._failure_req_ids.clear()
        self._submitted_load_rows.clear()
        self._dumped_row_ids.clear()
        self._dump_transfer_data = None
        self.need_load = False
        self._layerwise_load_bytes = 0
        self._layerwise_load_bytes_recorded = False
        self._layerwise_save_bytes = 0
        self._load_block_counts.clear()

        for request_id, request in metadata.request_meta.items():
            if len(request.load_block_ids[0]) == 0:
                continue
            n = getattr(request, "load_full_attn_count", 0)
            _, scoped_ucm, scoped_vllm = self._scope_blocks(
                request.load_block_ids[0], request.load_block_ids[1], n, is_dump=False
            )
            if not scoped_ucm:
                continue
            self.need_load = True
            self._load_block_counts[request_id] = len(scoped_ucm)
            self.request_data.append(
                (request_id, request.load_block_ids[0], scoped_ucm, scoped_vllm)
            )

        if self.need_load and self.row_ids:
            # Ensure do_mamba_copy_block (from preprocess_mamba, compute stream)
            # has completed before submitting load DMA (store stream).  Without
            # this, the copy may land after the load and clobber loaded data.
            # At this point the previous step's forward is done, so the only
            # pending compute op is the mamba state copy — sync overhead is
            # negligible.
            self.device.synchronize()
            # vLLM only calls wait_for_layer_load at full_attn (last layer of
            # each row), so row 0 must be loaded here before linear_attn begins.
            num_submit = min(self._load_prefetch_rows + 1, len(self.row_ids))
            for idx in range(num_submit):
                self._submit_request_load_tasks_for_row_once(idx, metadata)
            self._wait_row_load(0, metadata)
            if len(self.row_ids) == 1:
                self._record_layerwise_load_bytes()

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self._connector_metadata or not self.need_load:
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)
        row_id = self.layer_name_to_row.get(layer_name)
        if row_id is None:
            return

        # Wait for NEXT row so its linear_attn layers have KV loaded.
        next_row_id = row_id + 1
        if next_row_id >= len(self.row_ids):
            return

        self._submit_request_load_tasks_for_row_once(next_row_id, metadata)

        self._wait_row_load(next_row_id, metadata)
        if next_row_id == self.row_ids[-1]:
            self._record_layerwise_load_bytes()

        # Prefetch rows ahead.
        prefetch_start = next_row_id + 1
        prefetch_end = min(prefetch_start + self._load_prefetch_rows, len(self.row_ids))
        for idx in range(prefetch_start, prefetch_end):
            self._submit_request_load_tasks_for_row_once(idx, metadata)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        if not self._connector_metadata:
            return

        row_id = self.layer_name_to_row.get(layer_name)
        if row_id is None:
            return
        if self.row_save_layer.get(row_id) != layer_name:
            return
        if row_id in self._dumped_row_ids:
            logger.debug(
                "Skip duplicate hybrid layerwise dump in the same batch: "
                f"layer_name={layer_name}, row_id={row_id}"
            )
            return

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)
        if self._dump_transfer_data is None:
            self._dump_transfer_data = self._build_dump_transfer_data(metadata, row_id)
        (
            total_ucm_block_ids,
            total_vllm_block_ids,
            dump_request_ids,
            block_ids_by_request,
        ) = self._dump_transfer_data

        if not total_ucm_block_ids:
            return

        self.is_save = True

        row_ptrs = self.kv_cache_layout.extract_block_addrs_for_row(
            total_vllm_block_ids, row_id
        )
        shard_indexs = [row_id] * len(total_ucm_block_ids)
        try:
            row_ptrs = np.ascontiguousarray(row_ptrs)
            event_handle = self._get_dump_event_handle()
            task = self._rank_consistency.submit_dump(
                self.store,
                block_ids_by_request,
                total_ucm_block_ids,
                shard_indexs,
                row_ptrs,
                event_handle,
            )
            self.dump_tasks[row_id].append(
                PendingDumpTask(
                    task=task,
                    request_ids=set(dump_request_ids),
                    event_handle=event_handle,
                )
            )
            self._layerwise_save_bytes += (
                len(total_ucm_block_ids) * self._row_shard_size
            )
            self._dumped_row_ids.add(row_id)
        except Exception as e:
            logger.error(
                f"submit hybrid layerwise row {row_id} dump task failed. "
                f"{type(e).__name__}: {e}"
            )

    def _build_dump_transfer_data(
        self,
        metadata: "UCMConnectorMetadata",
        row_id: int,
    ) -> tuple[list[bytes], list[int], set[str], dict[str, set[bytes]]]:
        total_ucm_block_ids: list[bytes] = []
        total_vllm_block_ids: list[int] = []
        dump_request_ids: set[str] = set()
        block_ids_by_request: dict[str, set[bytes]] = {}
        for request_id, request in metadata.request_meta.items():
            if len(request.dump_block_ids[0]) == 0:
                continue
            dump_request_ids.add(request_id)
            n = getattr(request, "dump_full_attn_count", 0)
            rank0_ucm, scoped_ucm, scoped_vllm = self._scope_blocks(
                request.dump_block_ids[0], request.dump_block_ids[1], n, is_dump=True
            )
            if not scoped_ucm:
                continue
            block_ids_by_request[request_id] = set(rank0_ucm)
            total_ucm_block_ids.extend(scoped_ucm)
            total_vllm_block_ids.extend(scoped_vllm)
        return (
            total_ucm_block_ids,
            total_vllm_block_ids,
            dump_request_ids,
            block_ids_by_request,
        )

    def wait_for_save(self) -> None:
        if not self.is_save:
            return

        dump_request_ids = (
            self._dump_transfer_data[2]
            if self._dump_transfer_data is not None
            else set()
        )
        for row_id in self.row_ids:
            for pending_dump_task in self.dump_tasks.pop(row_id, []):
                try:
                    self._rank_consistency.wait_dump(pending_dump_task.task)
                except Exception as e:
                    logger.error_limit(
                        f"wait for dump kv cache failed. " f"{type(e).__name__}: {e}"
                    )
        self._rank_consistency.finish_dump(dump_request_ids)
        if self._layerwise_save_bytes > 0:
            ucmmetrics.update_stats({"save_bytes_total": self._layerwise_save_bytes})
            self._layerwise_save_bytes = 0
        self.dump_tasks.clear()
        self._dump_transfer_data = None
        self.is_save = False
        if self.enable_event_sync:
            self.device.destroy_event_handles()


class UCMHLALiteConnector(UCMLiteConnector, SupportsHMA):
    """UCM Lite connector for full-attention + linear-attention hybrids.

    A thin subclass of :class:`UCMLiteConnector`: reuses the full-attention
    prefix-chain logging (``ucm_block_ids``) and all no-op I/O hooks, and only
    adds a :class:`KVCacheGroupManager` so the startup
    ``KVCacheGroupManager initialized:`` line records the HLA topology (LCM,
    group block sizes, mamba-group count) for the offline hit-rate simulator.
    The simulator derives mamba state keys from the prefix chain (state sharing
    == prefix sharing), so this connector logs **only** the prefix chain per
    request — no per-request state hashes.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ) -> None:
        if kv_cache_config is None:
            raise RuntimeError("UCMHLALiteConnector requires kv_cache_config.")
        super().__init__(vllm_config, role, kv_cache_config)
        self.group_manager = KVCacheGroupManager(
            kv_cache_config=kv_cache_config,
            request_hasher=self.request_hasher,
            base_seed=self._seed,
        )
        self.block_size = self.group_manager.lcm_block_size
        self.hash_block_size = self.group_manager.lcm_block_size
        self.trace_hash_block_size = self.hash_block_size
        is_mla = getattr(vllm_config.model_config, "is_deepseek_mla", False)
        hbm_data_size = self._compute_hla_block_data_size(
            vllm_config, kv_cache_config, is_mla
        )
        logger.info(
            f"UCMTraceMeta: type=mamba, "
            f"is_mla={is_mla}, "
            f"vllm_hash_block_size={self.hash_block_size}, "
            f"trace_hash_block_size={self.trace_hash_block_size}, "
            f"hbm_block_data_size={hbm_data_size}, "
            f"lcm_block_size={self.group_manager.lcm_block_size}, "
            f"mamba_groups={len(self.group_manager.state_groups)}"
        )
        logger.info("Init UCMHLALiteConnector.")

    @staticmethod
    def _compute_hla_block_data_size(
        vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig", is_mla: bool
    ) -> int:
        """Compute hbm_block_data_size for HLA models from model config.

        Mirrors the calculator's deriveLinearHybridParams: derives block_size
        from mamba state alignment, computes page_size (FA per-token × block_size
        + conv on Ascend), then block_data_size = num_tensors × page_size.
        """
        mc = vllm_config.model_config
        hf = mc.hf_text_config
        tp = vllm_config.parallel_config.tensor_parallel_size

        num_full = getattr(hf, "num_full_attn_layers", None)
        num_linear = getattr(hf, "num_linear_layers", None)
        if num_full is None or num_linear is None:
            layer_types = getattr(hf, "layer_types", None)
            if layer_types and isinstance(layer_types, list):
                num_full = sum(1 for t in layer_types if t == "full_attention")
                num_linear = sum(1 for t in layer_types if t != "full_attention")
            else:
                return 0

        has_mtp = (
            vllm_config.speculative_config is not None
            and vllm_config.speculative_config.num_speculative_tokens > 0
        )
        num_full_eff = num_full + (1 if has_mtp else 0)
        num_tensors = min(num_full_eff, num_linear)
        if num_tensors <= 0:
            return 0

        dt_map = {"bfloat16": 2, "float16": 2, "float32": 4, "int8": 1}
        model_dt = dt_map.get(str(getattr(mc, "dtype", "bfloat16")), 2)
        ssm_dt = dt_map.get(
            str(getattr(hf, "mamba_ssm_dtype", getattr(mc, "dtype", "float32"))), 4
        )

        lin_kh = getattr(hf, "linear_num_key_heads", 0)
        lin_vh = getattr(hf, "linear_num_value_heads", 0)
        lin_kd = getattr(hf, "linear_key_head_dim", 0)
        lin_vd = getattr(hf, "linear_value_head_dim", 0)
        conv_k = getattr(hf, "linear_conv_kernel_dim", 4)

        v_heads_per_rank = lin_vh / tp if tp else lin_vh
        ssm_size = int(v_heads_per_rank * lin_vd * lin_kd * ssm_dt)
        conv_dim = lin_kd * lin_kh * 2 + lin_vd * lin_vh
        conv_dim_per_rank = conv_dim / tp if tp else conv_dim
        conv_size = int((conv_k - 1) * conv_dim_per_rank * model_dt)
        mamba_total = conv_size + ssm_size

        if is_mla:
            kv_lora = getattr(hf, "kv_lora_rank", 0)
            qk_rope = getattr(hf, "qk_rope_head_dim", 0)
            attn_per_tok = (kv_lora + qk_rope) * 1 * model_dt
            attn_single_k = kv_lora * 1 * model_dt
        else:
            head_dim = getattr(hf, "head_dim", 0) or (
                getattr(hf, "hidden_size", 0)
                // max(getattr(hf, "num_attention_heads", 1), 1)
            )
            kv_heads = getattr(
                hf, "num_key_value_heads", getattr(hf, "num_attention_heads", 0)
            )
            kv_heads_per_rank = kv_heads / tp if tp else kv_heads
            attn_per_tok = 2 * head_dim * int(kv_heads_per_rank) * model_dt
            attn_single_k = head_dim * int(kv_heads_per_rank) * model_dt

        if not attn_per_tok or not attn_single_k:
            return 0

        is_ascend = current_platform.device_type == "npu"

        bs = 0
        for g in kv_cache_config.kv_cache_groups:
            bs = max(bs, block_size_from_kv_cache_spec(g.kv_cache_spec))

        if not bs:
            if is_ascend:
                kernel = 128
                ratio = -(-ssm_size // (kernel * attn_single_k)) if attn_single_k else 1
                bs = kernel * ratio
            else:
                kernel = 16
                ratio = (
                    -(-mamba_total // (kernel * attn_per_tok)) if attn_per_tok else 1
                )
                bs = kernel * ratio

        if is_ascend:
            page_size = bs * attn_per_tok + conv_size
        else:
            page_size = bs * attn_per_tok

        return num_tensors * page_size

    def get_block_size(self) -> int:
        return self.group_manager.lcm_block_size

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, object] | None]:
        self.requests_meta.pop(request.request_id, None)
        return False, None
