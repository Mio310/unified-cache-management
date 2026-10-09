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
    """Extends RequestDispatchMeta with full-attn block count for MLA rank scoping.

    When compressed or PLE pages are present, ``load_full_attn_count`` and
    ``dump_full_attn_count`` are also the attention-record prefix length so
    the worker can split the flat block list. Companion ids are parallel to
    that prefix (compressed) and to the mamba suffix (PLE). A zero companion
    id selects the zero sink.
    """

    load_full_attn_count: int = 0
    dump_full_attn_count: int = 0
    load_compressed_block_ids: list[int] = field(default_factory=list)
    dump_compressed_block_ids: list[int] = field(default_factory=list)
    load_ple_block_ids: list[int] = field(default_factory=list)
    dump_ple_block_ids: list[int] = field(default_factory=list)


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
) -> None:
    # Skip vLLM null blocks (block_id=0) used as mamba-align placeholders.
    for ucm_block_id, vllm_block_id in zip(src_ucm_block_ids, src_vllm_block_ids):
        if vllm_block_id == 0:
            continue
        dst_ucm_block_ids.append(ucm_block_id)
        dst_vllm_block_ids.append(vllm_block_id)


def _hash_prefix(block_id: bytes) -> str:
    if not block_id:
        return "-"
    return block_id[:8].hex()


def _normalize_tensor_size_list(tensor_size_list: Any) -> list[int]:
    if isinstance(tensor_size_list, np.ndarray):
        return [int(v) for v in tensor_size_list.reshape(-1).tolist()]
    if isinstance(tensor_size_list, (list, tuple)):
        return [int(v) for v in tensor_size_list]
    return [int(tensor_size_list)]


_RAW_LAYER_COMPONENT = "raw_key_cache"
_COMPRESSED_LAYER_COMPONENT = "compressed_key_cache"
_PLE_LAYER_COMPONENT = "ple"
_SIDE_CACHE_ROLES = frozenset({"raw", "compressed", "ple"})


def _layer_components(layer_name: str) -> tuple[str, ...]:
    return tuple(part for part in str(layer_name).split(".") if part)


def _layer_role(layer_name: str) -> str:
    """Role of one layer from its name components.

    ``ple`` matches only a whole component, so names such as ``complete`` stay
    ordinary full-attention or mamba layers.
    """
    parts = _layer_components(layer_name)
    if _RAW_LAYER_COMPONENT in parts:
        return "raw"
    if _COMPRESSED_LAYER_COMPONENT in parts:
        return "compressed"
    if _PLE_LAYER_COMPONENT in parts:
        return "ple"
    return ""


def _layer_index(layer_name: str) -> Optional[int]:
    parts = _layer_components(layer_name)
    for idx, part in enumerate(parts[:-1]):
        if part in ("layers", "layer") and parts[idx + 1].isdigit():
            return int(parts[idx + 1])
    return None


def _spec_prefix_cacheable(spec: KVCacheSpec) -> bool:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return all(
            _spec_prefix_cacheable(child) for child in spec.kv_cache_specs.values()
        )
    return bool(getattr(spec, "prefix_cacheable", True))


def _group_kind(layer_names, spec: KVCacheSpec) -> str:
    """Classify a vLLM KV group for HLA dump and lookup.

    Qwen3.8-Flash-Next keeps compressed history, the QSA raw ring, and the
    PLE short-conv in their own groups. Those groups stay in ``groups_by_id``
    so allocated block-id tuples stay aligned, but they are not full-attention
    or mamba-align records.
    """
    roles = [_layer_role(name) for name in layer_names]
    if roles and all(role and role == roles[0] for role in roles):
        return roles[0]
    if is_mamba_align_kv_cache_spec(spec):
        return "mamba"
    if not _spec_prefix_cacheable(spec):
        return "raw"
    return "full_attn"


def _tensor_layers(raw_tensor) -> list[str]:
    """Layer names stored on one ``KVCacheTensor``.

    Current vLLM uses ``shared_by``. Newer block-outer configs expose the same
    names on ``layers`` and leave ``shared_by`` empty.
    """
    shared = getattr(raw_tensor, "shared_by", None)
    if shared:
        return [str(name) for name in shared]
    layers = getattr(raw_tensor, "layers", None)
    if not layers:
        return []
    if isinstance(layers, dict):
        return [str(name) for name in layers]
    return [str(name) for name in layers]


def _unwrap_kv_tensor(kv_layer) -> Optional[torch.Tensor]:
    if isinstance(kv_layer, torch.Tensor):
        return kv_layer
    if isinstance(kv_layer, (tuple, list)):
        for item in kv_layer:
            if isinstance(item, torch.Tensor):
                return item
    return None


def _block_stride_bytes(tensor: torch.Tensor) -> int:
    if tensor.ndim < 1 or int(tensor.shape[0]) == 0:
        return 0
    return int(tensor.stride(0)) * int(tensor.element_size())


def _view_region_bytes(tensor: torch.Tensor) -> int:
    base = int(tensor.data_ptr())
    try:
        storage = tensor.untyped_storage()
        storage_ptr = int(storage.data_ptr())
        storage_size = int(storage.nbytes())
        if storage_ptr and storage_size and storage_ptr <= base:
            return storage_ptr + storage_size - base
    except Exception:
        pass
    return int(tensor.numel()) * int(tensor.element_size())


def _spec_page_bytes(spec: KVCacheSpec) -> int:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        child = next(iter(spec.kv_cache_specs.values()), None)
        return _spec_page_bytes(child) if child is not None else 0
    try:
        page = int(getattr(spec, "page_size_bytes", 0) or 0)
    except (TypeError, ValueError):
        page = 0
    if page > 0:
        return page
    if isinstance(spec, MambaSpec):
        return sum(HybridLinearAttentionLayout._mamba_component_sizes(spec))
    return 0


def _ordered_role_layers(kv_cache_config, role: str) -> list[str]:
    """Layers of the first group with ``role``, ordered by layer index."""
    matched = []
    for group in kv_cache_config.kv_cache_groups:
        if _group_kind(group.layer_names, group.kv_cache_spec) == role:
            matched.append(group)
    if not matched:
        return []
    if len(matched) > 1:
        logger.warning(
            "HLA direct layout uses only the first %s group (group layers=%s); "
            "%d additional %s groups are not given their own store segments.",
            role,
            list(matched[0].layer_names),
            len(matched) - 1,
            role,
        )
    names = []
    seen = set()
    for name in matched[0].layer_names:
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    names.sort(
        key=lambda name: (
            _layer_index(name) is None,
            _layer_index(name) or 0,
            name,
        )
    )
    return names


def select_ple_carrier_group_id(
    state_groups: list["GroupInfo"],
    ple_groups: list["GroupInfo"],
    kv_cache_config,
) -> Optional[int]:
    """Mamba-align group whose state record also carries the PLE page.

    Prefer a kv tensor that lists the PLE layer together with a mamba layer.
    Otherwise match the PLE layer index. Otherwise use the first state group.
    """
    if not ple_groups or not state_groups:
        return None
    ple_names = {name for group in ple_groups for name in group.layer_names}
    mamba_owner: dict[str, int] = {}
    for group in state_groups:
        for name in group.layer_names:
            mamba_owner.setdefault(name, group.group_id)
    for raw_tensor in getattr(kv_cache_config, "kv_cache_tensors", []) or []:
        layers = _tensor_layers(raw_tensor)
        if not any(name in ple_names or _layer_role(name) == "ple" for name in layers):
            continue
        for name in layers:
            owner = mamba_owner.get(name)
            if owner is not None:
                return owner
    target_index = None
    for group in ple_groups:
        for name in group.layer_names:
            target_index = _layer_index(name)
            if target_index is not None:
                break
        if target_index is not None:
            break
    if target_index is not None:
        for group in state_groups:
            for name in group.layer_names:
                if _layer_index(name) == target_index:
                    return group.group_id
    return state_groups[0].group_id


def assemble_side_record_ptrs(
    segment_kinds,
    bases,
    strides,
    sink_ptr: int,
    record_kind: str,
    primary_block_id: int,
    compressed_block_id: int,
    ple_block_id: int,
) -> list[int]:
    """Device pointers for one uniform HLA store record.

    Hybrid segments use ``primary_block_id``. Compressed segments are live
    only on an attention record. The PLE segment is live only on the carrier
    mamba record. Every unused side segment points at the zero sink so a load
    cannot clobber another cache that shares the allocation.
    """
    ptrs: list[int] = []
    for kind, base, stride in zip(segment_kinds, bases, strides):
        if kind == "compressed":
            block_id = int(compressed_block_id) if record_kind == "attn" else 0
        elif kind == "ple":
            block_id = int(ple_block_id) if record_kind == "mamba" else 0
        else:
            block_id = int(primary_block_id)
        if kind in ("compressed", "ple") and block_id == 0:
            ptrs.append(int(sink_ptr))
        else:
            ptrs.append(int(base) + block_id * int(stride))
    return ptrs


@dataclass
class GroupInfo:
    """Per-group metadata used by :class:`KVCacheGroupManager`."""

    group_id: int
    block_size: int
    layer_names: tuple[str, ...]
    # Independent hash chain seed per group (see ``KVCacheGroupManager``).
    seed: bytes
    is_mamba_align: bool = False
    # full_attn, mamba, compressed, ple, or raw. Default keeps older tests
    # that only set ``is_mamba_align`` on a full-attention-shaped group.
    kind: str = "full_attn"
    block_hasher: Optional[Callable[["Request"], list[bytes]]] = None

    @property
    def is_full_attention(self) -> bool:
        return self.kind == "full_attn" and not self.is_mamba_align


def _layout_group_infos(kv_cache_config) -> list[GroupInfo]:
    """Group roles for layout construction. Hashes are not needed here."""
    infos = []
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        kind = _group_kind(group.layer_names, group.kv_cache_spec)
        infos.append(
            GroupInfo(
                group_id=group_id,
                block_size=block_size_from_kv_cache_spec(group.kv_cache_spec),
                layer_names=tuple(group.layer_names),
                seed=b"",
                is_mamba_align=kind == "mamba",
                kind=kind,
            )
        )
    return infos


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
        self.compressed_groups: list[GroupInfo] = []
        self.ple_groups: list[GroupInfo] = []

        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            block_size = block_size_from_kv_cache_spec(spec)
            kind = _group_kind(group.layer_names, spec)
            is_mamba_align = kind == "mamba"
            seed = request_hasher((b"UCM_GROUP_SEED", base_seed, group_id))
            info = GroupInfo(
                group_id=group_id,
                block_size=block_size,
                layer_names=tuple(group.layer_names),
                seed=seed,
                is_mamba_align=is_mamba_align,
                kind=kind,
            )
            if kind == "full_attn":
                info.block_hasher = request_hasher.make_request_block_hasher(
                    block_size, seed
                )
            self.groups_by_id.append(info)
            if kind == "full_attn":
                self.full_attn_groups.append(info)
            elif kind == "mamba":
                self.state_groups.append(info)
            elif kind == "compressed":
                self.compressed_groups.append(info)
            elif kind == "ple":
                self.ple_groups.append(info)

        assert len(self.full_attn_groups) >= 1, (
            "UCMHybridLinearAttentionConnector expects at least one full-attention group in "
            "kv_cache_config.kv_cache_groups."
        )

        # The QSA raw ring is not prefix-cacheable and must not change the
        # resume grid. Compressed history and PLE still have to land on it.
        scheduled_groups = self.full_attn_groups + self.state_groups
        self.lcm_block_size: int = math.lcm(
            *[g.block_size for g in scheduled_groups]
        )
        for group in scheduled_groups + self.compressed_groups + self.ple_groups:
            assert self.lcm_block_size % group.block_size == 0, (
                f"group {group.group_id} kind={group.kind} "
                f"block_size={group.block_size} does not "
                f"divide LCM={self.lcm_block_size}"
            )
        for sg in self.state_groups:
            assert sg.is_mamba_align, (
                f"state group {sg.group_id} is not mamba-align; "
                f"UCMHybridLinearAttentionConnector only supports mamba-align "
                f"state groups."
            )

        self.compressed_group = (
            self.compressed_groups[0] if self.compressed_groups else None
        )
        self.ple_group = self.ple_groups[0] if self.ple_groups else None
        primary = self.full_attn_groups[0]
        self.compressed_index_aligned = False
        if self.compressed_group is not None:
            if self.compressed_group.block_size != primary.block_size:
                raise ValueError(
                    "compressed_key_cache block_size must match the primary "
                    "full-attention block_size, got "
                    f"{self.compressed_group.block_size} vs {primary.block_size}"
                )
            self.compressed_index_aligned = True
        if len(self.compressed_groups) > 1:
            logger.warning(
                "Multiple compressed_key_cache groups found; dump/load pairs "
                "block ids from group %s only.",
                self.compressed_group.group_id if self.compressed_group else None,
            )
        self.ple_carrier_group_id = select_ple_carrier_group_id(
            self.state_groups, self.ple_groups, kv_cache_config
        )

        logger.info(
            "KVCacheGroupManager initialized: "
            f"lcm_block_size={self.lcm_block_size}, "
            f"full_attn_groups="
            f"{[(g.group_id, g.block_size) for g in self.full_attn_groups]}, "
            f"state_groups="
            f"{[(g.group_id, g.block_size, g.is_mamba_align) for g in self.state_groups]}, "
            f"compressed_groups="
            f"{[(g.group_id, g.block_size) for g in self.compressed_groups]}, "
            f"ple_groups="
            f"{[(g.group_id, g.block_size) for g in self.ple_groups]}, "
            f"ple_carrier_group_id={self.ple_carrier_group_id}"
        )

    @property
    def num_groups(self) -> int:
        return len(self.groups_by_id)

    @property
    def has_side_caches(self) -> bool:
        """True when attention/mamba records must also carry compressed or PLE."""
        return self.compressed_group is not None or self.ple_group is not None

    def compute_block_hashes(self, group: GroupInfo, request: "Request") -> list[bytes]:
        """Hash a request at one group's block boundaries and chain seed."""
        if group.is_mamba_align or group.kind in _SIDE_CACHE_ROLES:
            # mamba-align, PLE, compressed, and the raw ring do not own a
            # prefix-hash chain. Compressed pages reuse the full-attention
            # block hash; PLE rides on the carrier mamba state hash.
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
        primary = self.full_attn_groups[0]
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
        # prefix_hit is the last present index, or -1 when the first block
        # is absent.
        candidates: list[int] = []
        stage1: list[tuple] = []
        for fa in self.full_attn_groups:
            fa_block_ids = group_block_ids[fa.group_id]
            fa_hbm_blocks = num_computed_tokens // fa.block_size
            fa_external = fa_block_ids[fa_hbm_blocks:]
            if not fa_external:
                candidates.append(0)
                stage1.append((fa.group_id, 0, None, 0, "-"))
                continue
            try:
                prefix_hit = lookup_on_prefix(fa_external)
                fa_hit_blocks = prefix_hit + 1
            except Exception as e:
                logger.error(
                    f"full-attn group {fa.group_id} lookup error. "
                    f"{type(e).__name__}: {e}"
                )
                _record_counter("connector_lookup_errors_total")
                candidates.append(0)
                stage1.append(
                    (
                        fa.group_id,
                        len(fa_external),
                        "error",
                        0,
                        _hash_prefix(fa_external[0]),
                    )
                )
                continue
            hit_tokens = max(fa_hit_blocks, 0) * fa.block_size
            candidates.append(hit_tokens)
            stage1.append(
                (
                    fa.group_id,
                    len(fa_external),
                    prefix_hit,
                    hit_tokens,
                    _hash_prefix(fa_external[0]),
                )
            )

        # Resume boundary must be a multiple of lcm_block_size so every
        # group's tail/dispatch slicing lands on a real block boundary.
        min_external_hit_tokens = min(candidates) if candidates else 0
        external_hit_tokens = (
            min_external_hit_tokens // self.lcm_block_size
        ) * self.lcm_block_size
        if external_hit_tokens <= 0:
            if not stage1 or all(item[1] == 0 for item in stage1):
                reason = "stage1_no_external_blocks"
            elif min_external_hit_tokens <= 0:
                reason = "stage1_prefix_miss"
            else:
                reason = "stage1_below_lcm"
            logger.info(
                "HLA lookup miss: computed=%s lcm=%s reason=%s "
                "stage1=(group,external_blocks,prefix_hit,hit_tokens,first)=%s "
                "min_hit_tokens=%s",
                num_computed_tokens,
                self.lcm_block_size,
                reason,
                stage1,
                min_external_hit_tokens,
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
            empty_hashes = 0
            for pos in sg_positions:
                state_hash = self.compute_mamba_align_state_hash(
                    sg, pos, group_block_ids
                )
                if not state_hash:
                    empty_hashes += 1
                    sg_hashes.append(b"")
                else:
                    sg_hashes.append(state_hash)
            try:
                idx = lookup_on_reverse(sg_hashes)
            except Exception as e:
                logger.error(
                    f"mamba-align state reverse lookup error for "
                    f"group={sg.group_id}. {type(e).__name__}: {e}"
                )
                _record_counter("connector_lookup_errors_total")
                logger.info(
                    "HLA lookup miss: computed=%s lcm=%s reason=stage2_error "
                    "group=%s stage1_lcm_hit=%s",
                    num_computed_tokens,
                    self.lcm_block_size,
                    sg.group_id,
                    external_hit_tokens,
                )
                return 0, 0, []
            if idx < 0:
                # This state group has no state at any candidate position.
                logger.info(
                    "HLA lookup miss: computed=%s lcm=%s reason=stage2_state_miss "
                    "group=%s positions=%s empty_hashes=%s stage1_lcm_hit=%s "
                    "first_state=%s last_state=%s "
                    "stage1=(group,external_blocks,prefix_hit,hit_tokens,first)=%s",
                    num_computed_tokens,
                    self.lcm_block_size,
                    sg.group_id,
                    len(sg_positions),
                    empty_hashes,
                    external_hit_tokens,
                    _hash_prefix(sg_hashes[0] if sg_hashes else b""),
                    _hash_prefix(sg_hashes[-1] if sg_hashes else b""),
                    stage1,
                )
                return 0, 0, []
            sg_pos = sg_positions[idx]
            if sg_pos < best_pos:
                best_pos = sg_pos

        external_hit_tokens = best_pos - num_computed_tokens
        if external_hit_tokens <= 0:
            logger.info(
                "HLA lookup miss: computed=%s lcm=%s "
                "reason=stage2_not_beyond_hbm best_pos=%s",
                num_computed_tokens,
                self.lcm_block_size,
                best_pos,
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

    vLLM may back full-attention and linear-attention layers with one shared
    raw int8 tensor. The physical layout is backend dependent:

    - Ascend stores the shared page in component-major order:
        [conv_block_or_padding, k_or_ssm_block, v_block_or_padding]
      across all physical blocks.
    - CUDA stores one contiguous page per physical block. The same bytes are
      viewed as either attention [K, V] or mamba [conv, ssm, padding].
    - CUDA direct mode appends Qwen3.8-Flash-Next compressed-key and PLE
      segments after those hybrid pages. The raw QSA ring is not stored.
      Layerwise mode keeps one row per raw tensor and does not add them.

    The store receives one unified tensor_size_list, so we expose the three
    physical slices for Ascend, while CUDA is exposed as one contiguous page
    with a full-page stride.
    """

    def __init__(
        self,
        kvcaches,
        ucm_config: dict,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
    ):
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
        for layer_name in _tensor_layers(raw_tensor):
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

    def _append_role_segments(
        self,
        kvcaches,
        role: str,
        base_ptrs: list[list[int]],
        buffer_size_rows: list[list[int]],
        tensor_size_lists: list[list[int]],
        block_stride_lists: list[list[int]],
    ) -> None:
        """One store segment per layer, addressed by that group's block id."""
        layer_to_specs = layer_name_to_kv_cache_spec(self.kv_cache_config)
        names = _ordered_role_layers(self.kv_cache_config, role)
        found = 0
        for name in names:
            tensor = _unwrap_kv_tensor(kvcaches.get(name))
            if tensor is None:
                logger.warning("HLA side segment missing kv tensor for %s", name)
                continue
            specs = layer_to_specs.get(name, [])
            spec = specs[0] if specs else None
            stride = _block_stride_bytes(tensor)
            page = _spec_page_bytes(spec) if spec is not None else 0
            if page <= 0:
                page = stride
            region = _view_region_bytes(tensor)
            if stride <= 0 or page <= 0:
                logger.warning(
                    "HLA side segment has no positive stride for %s "
                    "(stride=%s, page=%s)",
                    name,
                    stride,
                    page,
                )
                continue
            if page > stride or page > region:
                raise ValueError(
                    f"HLA {role} page for {name} is {page} bytes, but the "
                    f"layer view stride is {stride} and the registered "
                    f"region is {region}. Refusing to truncate the page."
                )
            base_ptrs.append([int(tensor.data_ptr())])
            buffer_size_rows.append([int(region)])
            tensor_size_lists.append([int(page)])
            block_stride_lists.append([int(stride)])
            self.row_segment_kinds.append([role])
            found += 1
        if names and found == 0:
            logger.warning(
                "HLA direct layout found no device tensors for %s layers %s",
                role,
                names,
            )

    def record_ptrs(
        self,
        record_kind: str,
        primary_block_id: int,
        compressed_block_id: int,
        ple_block_id: int,
    ) -> list[int]:
        return assemble_side_record_ptrs(
            self.segment_kinds,
            [int(value) for value in self.base_ptrs.tolist()],
            [int(value) for value in self.block_stride_lists.tolist()],
            int(self.sink_ptr),
            record_kind,
            int(primary_block_id),
            int(compressed_block_id),
            int(ple_block_id),
        )

    def row_record_ptrs(
        self,
        row_id: int,
        record_kind: str,
        primary_block_id: int,
        compressed_block_id: int,
        ple_block_id: int,
    ) -> list[int]:
        """Pointers for one layerwise row of one attention or mamba record."""
        if row_id < 0 or row_id >= len(self.row_segment_kinds):
            raise ValueError(
                f"Invalid hybrid row_id={row_id}; "
                f"row_count={len(self.row_segment_kinds)}"
            )
        row_slice = self.row_slices[row_id]
        kinds = self.row_segment_kinds[row_id]
        bases = [int(value) for value in self.base_ptrs[row_slice].tolist()]
        strides = [int(value) for value in self.block_stride_lists[row_slice].tolist()]
        if len(kinds) != len(bases):
            raise RuntimeError(
                "Layerwise row segment kinds do not match the row layout: "
                f"row_id={row_id}, kinds={len(kinds)}, segments={len(bases)}"
            )
        return assemble_side_record_ptrs(
            kinds,
            bases,
            strides,
            int(self.sink_ptr),
            record_kind,
            int(primary_block_id),
            int(compressed_block_id),
            int(ple_block_id),
        )

    def _layer_view_segment(self, kvcaches, layer_name: str):
        """Return ``(base, stride, page, region)`` for one layer view."""
        tensor = _unwrap_kv_tensor(kvcaches.get(layer_name))
        if tensor is None:
            logger.warning("HLA side segment missing kv tensor for %s", layer_name)
            return None
        specs = layer_name_to_kv_cache_spec(self.kv_cache_config).get(layer_name, [])
        spec = specs[0] if specs else None
        stride = _block_stride_bytes(tensor)
        page = _spec_page_bytes(spec) if spec is not None else 0
        if page <= 0:
            page = stride
        region = _view_region_bytes(tensor)
        if stride <= 0 or page <= 0 or page > stride or page > region:
            raise ValueError(
                f"HLA page for {layer_name} is {page} bytes, but the layer "
                f"view stride is {stride} and the registered region is {region}."
            )
        return int(tensor.data_ptr()), int(stride), int(page), int(region)

    def _augment_layerwise_side_rows(
        self,
        kvcaches,
        base_ptrs,
        buffer_size_rows,
        tensor_size_lists,
        block_stride_lists,
    ) -> None:
        """Append identical compressed and PLE slots to every CUDA hybrid row.

        Slots that a row does not own stay at pointer 0 here and are retargeted
        at the zero sink after that buffer exists. A stride of 0 keeps every
        block id on that sink.
        """
        infos = _layout_group_infos(self.kv_cache_config)
        compressed_groups = [group for group in infos if group.kind == "compressed"]
        ple_groups = [group for group in infos if group.kind == "ple"]
        state_groups = [group for group in infos if group.kind == "mamba"]
        if not compressed_groups and not ple_groups:
            return
        if not tensor_size_lists:
            return

        compressed_by_index: dict[int, str] = {}
        for group in compressed_groups:
            for name in group.layer_names:
                layer_index = _layer_index(name)
                if layer_index is not None:
                    compressed_by_index[layer_index] = name

        carrier_layers: set[str] = set()
        carrier_id = select_ple_carrier_group_id(
            state_groups, ple_groups, self.kv_cache_config
        )
        if carrier_id is not None:
            carrier_layers = set(infos[carrier_id].layer_names)

        compressed_views: dict[str, tuple[int, int, int]] = {}
        compressed_page = 0
        for name in compressed_by_index.values():
            view = self._layer_view_segment(kvcaches, name)
            if view is None:
                continue
            base, stride, page, region = view
            if compressed_page == 0:
                compressed_page = page
            elif page != compressed_page:
                raise ValueError(
                    "CUDA layerwise compressed pages must share one copy size: "
                    f"{name} page={page}, expected={compressed_page}"
                )
            compressed_views[name] = (base, stride, region)
        if compressed_groups and compressed_page <= 0:
            raise ValueError(
                "CUDA layerwise layout found compressed_key_cache groups "
                "but no device page."
            )

        ple_page = 0
        ple_view = None
        ple_names = [name for group in ple_groups for name in group.layer_names]
        if ple_names:
            ple_view = self._layer_view_segment(kvcaches, ple_names[0])
            if ple_view is not None:
                ple_page = ple_view[2]
        if ple_groups and ple_page <= 0:
            raise ValueError(
                "CUDA layerwise layout found a PLE group but no device page."
            )

        for row_id, layer_names in enumerate(self._row_layers):
            if compressed_page > 0:
                matched = None
                for name in layer_names:
                    if _layer_role(name):
                        continue
                    matched = compressed_by_index.get(_layer_index(name))
                    if matched:
                        break
                view = compressed_views.get(matched) if matched else None
                if view is not None and compressed_page <= view[1]:
                    base, stride, region = view
                    base_ptrs[row_id].append(base)
                    buffer_size_rows[row_id].append(region)
                    block_stride_lists[row_id].append(stride)
                else:
                    if view is not None and compressed_page > view[1]:
                        raise ValueError(
                            f"CUDA layerwise compressed page {compressed_page} "
                            f"exceeds stride {view[1]} for {matched}"
                        )
                    base_ptrs[row_id].append(0)
                    buffer_size_rows[row_id].append(0)
                    block_stride_lists[row_id].append(0)
                    self._sink_slots.append((row_id, len(base_ptrs[row_id]) - 1))
                tensor_size_lists[row_id].append(compressed_page)
                self.row_segment_kinds[row_id].append("compressed")
            if ple_page > 0:
                is_carrier = any(name in carrier_layers for name in layer_names)
                if is_carrier and ple_view is not None and ple_page <= ple_view[1]:
                    base, stride, _page, region = ple_view
                    base_ptrs[row_id].append(base)
                    buffer_size_rows[row_id].append(region)
                    block_stride_lists[row_id].append(stride)
                else:
                    if (
                        is_carrier
                        and ple_view is not None
                        and ple_page > ple_view[1]
                    ):
                        raise ValueError(
                            f"CUDA layerwise PLE page {ple_page} exceeds "
                            f"stride {ple_view[1]}"
                        )
                    base_ptrs[row_id].append(0)
                    buffer_size_rows[row_id].append(0)
                    block_stride_lists[row_id].append(0)
                    self._sink_slots.append((row_id, len(base_ptrs[row_id]) - 1))
                tensor_size_lists[row_id].append(ple_page)
                self.row_segment_kinds[row_id].append("ple")
        logger.info(
            "CUDA layerwise side slots: "
            f"rows={len(tensor_size_lists)}, compressed_page={compressed_page}, "
            f"ple_page={ple_page}, carrier_layers={len(carrier_layers)}"
        )

    def _build_layout(self, kvcaches):
        base_ptrs = []
        buffer_size_rows = []
        tensor_size_lists = []
        block_stride_lists = []
        self.layer_name_to_row: dict[str, int] = {}
        self.segment_kinds: list[str] = []
        self.row_segment_kinds: list[list[str]] = []
        self._row_layers: list[list[str]] = []
        self._sink_slots: list[tuple[int, int]] = []
        self.has_side_segments = False
        self.sink_ptr = 0
        self.sink_bytes = 0
        self._sink_tensor = None

        is_npu = current_platform.device_type == "npu"
        # Direct mode appends compressed and PLE as their own record segments.
        # CUDA layerwise keeps one hybrid row and appends those pages onto it.
        emit_side = (not self.use_layerwise) and (not is_npu)

        for raw_tensor in self.kv_cache_config.kv_cache_tensors:
            layer_names = _tensor_layers(raw_tensor)
            if not layer_names:
                continue
            if not is_npu:
                roles = {_layer_role(name) for name in layer_names}
                if roles and roles <= _SIDE_CACHE_ROLES:
                    continue

            shared_specs, shared_ptrs = self._collect_shared_tensor_info(
                raw_tensor, kvcaches
            )

            if not shared_ptrs:
                logger.warning(
                    f"no kv cache tensor found for shared layers {layer_names}"
                )
                continue

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
            self.row_segment_kinds.append(["hybrid"] * len(tensor_size_lists[-1]))
            self._row_layers.append(list(layer_names))

            for layer_name in layer_names:
                self.layer_name_to_row[layer_name] = row_id

        if emit_side:
            self._append_role_segments(
                kvcaches,
                "compressed",
                base_ptrs,
                buffer_size_rows,
                tensor_size_lists,
                block_stride_lists,
            )
            self._append_role_segments(
                kvcaches,
                "ple",
                base_ptrs,
                buffer_size_rows,
                tensor_size_lists,
                block_stride_lists,
            )

        if self.use_layerwise and not is_npu:
            self._augment_layerwise_side_rows(
                kvcaches,
                base_ptrs,
                buffer_size_rows,
                tensor_size_lists,
                block_stride_lists,
            )
        self.segment_kinds = [
            kind for row in self.row_segment_kinds for kind in row
        ]

        flat_count = sum(len(row) for row in tensor_size_lists)
        if len(self.segment_kinds) != flat_count:
            raise RuntimeError(
                "HLA segment kinds do not match the flattened layout: "
                f"kinds={len(self.segment_kinds)}, segments={flat_count}"
            )
        self.has_side_segments = any(
            kind in ("compressed", "ple") for kind in self.segment_kinds
        )
        if self.has_side_segments:
            width = max(int(size) for row in tensor_size_lists for size in row)
            device = None
            for value in kvcaches.values():
                tensor = _unwrap_kv_tensor(value)
                if tensor is not None:
                    device = tensor.device
                    break
            if device is None:
                raise RuntimeError(
                    "HLA side-cache segments were built without a kv cache device."
                )
            self._sink_tensor = torch.zeros(width, dtype=torch.uint8, device=device)
            self.sink_ptr = int(self._sink_tensor.data_ptr())
            self.sink_bytes = width
            for row_idx, seg_idx in self._sink_slots:
                base_ptrs[row_idx][seg_idx] = int(self.sink_ptr)
                buffer_size_rows[row_idx][seg_idx] = int(self.sink_bytes)
            logger.info(
                "Hybrid side-cache layout: "
                f"hybrid={self.segment_kinds.count('hybrid')}, "
                f"compressed={self.segment_kinds.count('compressed')}, "
                f"ple={self.segment_kinds.count('ple')}, "
                f"block_bytes={sum(int(size) for row in tensor_size_lists for size in row)}, "
                f"sink_bytes={self.sink_bytes}"
            )

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

    Direct mode stores Qwen3.8-Flash-Next compressed-key pages in the
    full-attention record and the PLE short-conv in the same-layer mamba
    state record. CUDA layerwise keeps one row per hybrid pool and appends
    those pages as extra segments on that row. The QSA raw ring is left out
    of the store and the LCM.
    """

    @classmethod
    def supports_kv_cache_layout(cls, kv_cache_config) -> bool:
        if kv_cache_config is None:
            return False

        if (
            current_platform.device_type != "npu"
            and not current_platform.is_cuda_alike()
        ):
            return False

        layer_to_specs = layer_name_to_kv_cache_spec(kv_cache_config)
        for raw_tensor in kv_cache_config.kv_cache_tensors:
            shared_specs = [
                spec
                for layer_name in _tensor_layers(raw_tensor)
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
            sink_ptr = int(getattr(kv_cache_layout, "sink_ptr", 0) or 0)
            sink_bytes = int(getattr(kv_cache_layout, "sink_bytes", 0) or 0)
            if sink_ptr and sink_bytes:
                sink_key = (sink_ptr, sink_bytes)
                if sink_key not in gpu_kv_buffer_set:
                    gpu_kv_buffer_set.add(sink_key)
                    gpu_kv_buffer_addrs.append(sink_key[0])
                    gpu_kv_buffer_sizes.append(sink_key[1])
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
            logger.info_once(
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
        primary_full_attn = self.group_manager.full_attn_groups[0]
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
            "HLA match: req=%s tokens=%s computed=%s total_lcm_blocks=%s "
            "hit_hbm=%s hit_external=%s external_hit_tokens=%s first=%s",
            request.request_id,
            len(request.all_token_ids),
            num_computed_tokens,
            request.num_tokens // lcm_block_size,
            hbm_hit_block_num,
            total_hit_block_num - hbm_hit_block_num,
            external_hit_tokens,
            _hash_prefix(primary_block_ids[0] if primary_block_ids else b""),
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
    ) -> bool:
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
            return False
        if vllm_block_id == 0:
            return False
        ucm_block_id = self.group_manager.compute_mamba_align_state_hash(
            group, seq_len, req_meta.group_ucm_block_ids
        )
        if ucm_block_id is None:
            logger.error(
                "HLA mamba-align state hash missing: "
                f"request_id={request_id}, group_id={gid}, reason={reason}, "
                f"seq_len={seq_len}, state_idx={state_idx}"
            )
            return False
        dst_ucm_block_ids.append(ucm_block_id)
        dst_vllm_block_ids.append(vllm_block_id)
        return True

    def _tracked_full_attn_count(self, record_count: int) -> int:
        """Attention-record prefix length visible to the worker.

        Non-MLA models historically left this at 0 because rank scoping does
        not split the flat list. Side-cache loads need the split so compressed
        ids stay aligned with attention records.
        """
        manager = self.group_manager
        if self.is_mla or (manager is not None and manager.has_side_caches):
            return record_count
        return 0

    def _vllm_block_at(self, req_meta: "HLARequestMeta", gid: int, index: int) -> int:
        block_ids = req_meta.group_vllm_block_ids[gid]
        if index < 0 or index >= len(block_ids):
            return 0
        return int(block_ids[index])

    def _ple_state_block_id(
        self,
        req_meta: "HLARequestMeta",
        request_id: str,
        seq_len: int,
        reason: str,
    ) -> int:
        """PLE physical block paired with one carrier mamba state record."""
        manager = self.group_manager
        assert manager is not None
        ple = manager.ple_group
        if ple is None:
            return 0
        state_idx = max((seq_len - 1) // ple.block_size, 0)
        vllm_state_idx = state_idx
        block_ids = req_meta.group_vllm_block_ids[ple.group_id]
        if reason == "load":
            for i in range(len(block_ids) - 1, -1, -1):
                if block_ids[i] != 0:
                    vllm_state_idx = i
                    break
        if vllm_state_idx < 0 or vllm_state_idx >= len(block_ids):
            logger.error(
                "HLA PLE vLLM block missing: "
                f"request_id={request_id}, group_id={ple.group_id}, "
                f"reason={reason}, seq_len={seq_len}, state_idx={state_idx}, "
                f"vllm_state_idx={vllm_state_idx}, num_vllm_blocks={len(block_ids)}"
            )
            return 0
        return int(block_ids[vllm_state_idx])

    def _append_full_attn_window(
        self,
        dst_ucm: list[bytes],
        dst_vllm: list[int],
        dst_compressed: list[int],
        req_meta: "HLARequestMeta",
        tok_start: int,
        tok_end: int,
    ) -> None:
        manager = self.group_manager
        assert manager is not None
        primary_gid = manager.full_attn_groups[0].group_id
        compressed = manager.compressed_group
        collect_side = manager.has_side_caches
        for group in manager.groups_by_id:
            if group.kind != "full_attn":
                continue
            start_blk = tok_start // group.block_size
            end_blk = tok_end // group.block_size
            if start_blk >= end_blk:
                continue
            ucm_ids = req_meta.group_ucm_block_ids[group.group_id][start_blk:end_blk]
            vllm_ids = req_meta.group_vllm_block_ids[group.group_id][start_blk:end_blk]
            for offset, (ucm_id, vllm_id) in enumerate(zip(ucm_ids, vllm_ids)):
                if vllm_id == 0:
                    continue
                dst_ucm.append(ucm_id)
                dst_vllm.append(vllm_id)
                if not collect_side:
                    continue
                comp_id = 0
                if (
                    manager.compressed_index_aligned
                    and compressed is not None
                    and group.group_id == primary_gid
                ):
                    comp_id = self._vllm_block_at(
                        req_meta, compressed.group_id, start_blk + offset
                    )
                dst_compressed.append(comp_id)

    def _append_mamba_window(
        self,
        dst_ucm: list[bytes],
        dst_vllm: list[int],
        dst_ple: list[int],
        req_meta: "HLARequestMeta",
        request_id: str,
        seq_len: int,
        reason: str,
    ) -> None:
        manager = self.group_manager
        assert manager is not None
        for group in manager.groups_by_id:
            if group.kind != "mamba":
                continue
            appended = self._append_mamba_align_state_block(
                dst_ucm,
                dst_vllm,
                req_meta,
                request_id,
                group.group_id,
                seq_len,
                reason,
            )
            if not appended or not manager.has_side_caches:
                continue
            ple_id = 0
            if (
                manager.ple_group is not None
                and group.group_id == manager.ple_carrier_group_id
            ):
                ple_id = self._ple_state_block_id(
                    req_meta, request_id, seq_len, reason
                )
            dst_ple.append(ple_id)

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
        load_compressed_block_ids: list[int] = []
        load_ple_block_ids: list[int] = []
        dump_ucm_block_ids: list[bytes] = []
        dump_vllm_block_ids: list[int] = []
        dump_compressed_block_ids: list[int] = []
        dump_ple_block_ids: list[int] = []

        external_hit_lcm_blocks = (
            req_meta.total_hit_block_num - req_meta.hbm_hit_block_num
        )
        hbm_hit_tokens = req_meta.hbm_hit_block_num * lcm_block_size
        total_hit_tokens = req_meta.total_hit_block_num * lcm_block_size

        if need_load and external_hit_lcm_blocks > 0:
            # Pass 1: full-attention blocks first (for MLA rank-0-only dump).
            # Compressed pages share these hashes and are not their own records.
            self._append_full_attn_window(
                load_ucm_block_ids,
                load_vllm_block_ids,
                load_compressed_block_ids,
                req_meta,
                hbm_hit_tokens,
                total_hit_tokens,
            )
            load_full_attn_count = self._tracked_full_attn_count(
                len(load_ucm_block_ids)
            )
            # Pass 2: mamba state blocks. PLE is attached to the carrier only.
            self._append_mamba_window(
                load_ucm_block_ids,
                load_vllm_block_ids,
                load_ple_block_ids,
                req_meta,
                request_id,
                total_hit_tokens,
                "load",
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

            attn_before = len(dump_ucm_block_ids)
            self._append_full_attn_window(
                dump_ucm_block_ids,
                dump_vllm_block_ids,
                dump_compressed_block_ids,
                req_meta,
                dump_tok_start,
                dump_tok_end,
            )
            dump_attn_records = len(dump_ucm_block_ids) - attn_before
            dump_full_attn_count = self._tracked_full_attn_count(
                len(dump_ucm_block_ids)
            )
            mamba_eligible = dump_tok_end == last_lcm_b and last_lcm_b >= first_lcm_b
            mamba_before = len(dump_ucm_block_ids)
            if mamba_eligible:
                self._append_mamba_window(
                    dump_ucm_block_ids,
                    dump_vllm_block_ids,
                    dump_ple_block_ids,
                    req_meta,
                    request_id,
                    last_lcm_b,
                    "dump",
                )
            dump_mamba_records = len(dump_ucm_block_ids) - mamba_before
            if dump_attn_records or dump_mamba_records or new_tokens >= lcm_block_size:
                logger.info(
                    "HLA dump: req=%s window=%s..%s new_tokens=%s lcm=%s "
                    "attn_records=%s mamba_records=%s mamba_eligible=%s "
                    "last_lcm=%s state_groups=%s first=%s",
                    request_id,
                    dump_tok_start,
                    dump_tok_end,
                    new_tokens,
                    lcm_block_size,
                    dump_attn_records,
                    dump_mamba_records,
                    mamba_eligible,
                    last_lcm_b,
                    len(self.group_manager.state_groups),
                    _hash_prefix(
                        dump_ucm_block_ids[0] if dump_ucm_block_ids else b""
                    ),
                )
        else:
            dump_full_attn_count = 0

        req_meta.token_processed += new_tokens

        return HLARequestDispatchMeta(
            load_block_ids=(load_ucm_block_ids, load_vllm_block_ids),
            dump_block_ids=(dump_ucm_block_ids, dump_vllm_block_ids),
            load_full_attn_count=load_full_attn_count,
            dump_full_attn_count=dump_full_attn_count,
            load_compressed_block_ids=load_compressed_block_ids,
            dump_compressed_block_ids=dump_compressed_block_ids,
            load_ple_block_ids=load_ple_block_ids,
            dump_ple_block_ids=dump_ple_block_ids,
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

    def _uses_side_records(self) -> bool:
        layout = getattr(self, "kv_cache_layout", None)
        if not getattr(layout, "has_side_segments", False):
            return False
        if self.is_mla:
            if not getattr(self, "_side_mla_warned", False):
                logger.warning(
                    "Flash-Next side-cache records are not combined with MLA "
                    "rank scoping; using one block id for every segment."
                )
                self._side_mla_warned = True
            return False
        return True

    def _side_record_ptrs(
        self,
        request,
        vllm_block_ids: list[int],
        full_attn_count: int,
        is_dump: bool,
    ) -> np.ndarray:
        attn_count = int(full_attn_count or 0)
        if is_dump:
            compressed_ids = list(
                getattr(request, "dump_compressed_block_ids", []) or []
            )
            ple_ids = list(getattr(request, "dump_ple_block_ids", []) or [])
        else:
            compressed_ids = list(
                getattr(request, "load_compressed_block_ids", []) or []
            )
            ple_ids = list(getattr(request, "load_ple_block_ids", []) or [])
        rows = []
        for index, block_id in enumerate(vllm_block_ids):
            if index < attn_count:
                compressed_id = (
                    compressed_ids[index] if index < len(compressed_ids) else 0
                )
                rows.append(
                    self.kv_cache_layout.record_ptrs(
                        "attn", block_id, compressed_id, 0
                    )
                )
            else:
                ple_index = index - attn_count
                ple_id = ple_ids[ple_index] if ple_index < len(ple_ids) else 0
                rows.append(
                    self.kv_cache_layout.record_ptrs("mamba", block_id, 0, ple_id)
                )
        if not rows:
            return np.zeros((0, 0), dtype=np.uint64)
        return np.asarray(rows, dtype=np.uint64)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Bulk load override: MLA blocks shared hash, KDA blocks per-rank hash."""
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)
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
                if self._uses_side_records():
                    ptrs = self._side_record_ptrs(
                        request, scoped_vllm, n, is_dump=False
                    )
                else:
                    ptrs = self.kv_cache_layout.extract_block_addrs(scoped_vllm)
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
        total_ptr_chunks: list[np.ndarray] = []
        block_ids_by_request: dict[str, set[bytes]] = {}
        num_saved_block = 0
        use_side_records = self._uses_side_records()
        for request_id, request in metadata.request_meta.items():
            if len(request.dump_block_ids[0]) == 0:
                continue
            n = getattr(request, "dump_full_attn_count", 0)
            rank0_ucm, scoped_ucm, scoped_vllm = self._scope_blocks(
                request.dump_block_ids[0], request.dump_block_ids[1], n, is_dump=True
            )
            if not scoped_ucm:
                continue
            block_ids_by_request[request_id] = set(rank0_ucm)
            num_saved_block += len(scoped_ucm)
            total_ucm_block_ids.extend(scoped_ucm)
            total_vllm_block_ids.extend(scoped_vllm)
            if use_side_records:
                total_ptr_chunks.append(
                    self._side_record_ptrs(request, scoped_vllm, n, is_dump=True)
                )

        if not total_ucm_block_ids:
            return

        event_handle = 0
        try:
            if use_side_records:
                total_ptrs = np.concatenate(total_ptr_chunks, axis=0)
            else:
                total_ptrs = self.kv_cache_layout.extract_block_addrs(
                    total_vllm_block_ids
                )
                total_ptrs = total_ptrs.reshape(total_ptrs.shape[0], -1)
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

    def _layerwise_row_ptrs(
        self,
        request,
        vllm_block_ids: list[int],
        full_attn_count: int,
        row_id: int,
        is_dump: bool,
    ) -> np.ndarray:
        """Row pointers, with compressed and PLE selected per record."""
        layout = self.kv_cache_layout
        if not getattr(layout, "has_side_segments", False):
            return layout.extract_block_addrs_for_row(vllm_block_ids, row_id)
        attn_count = int(full_attn_count or 0)
        if is_dump:
            compressed_ids = list(
                getattr(request, "dump_compressed_block_ids", []) or []
            )
            ple_ids = list(getattr(request, "dump_ple_block_ids", []) or [])
        else:
            compressed_ids = list(
                getattr(request, "load_compressed_block_ids", []) or []
            )
            ple_ids = list(getattr(request, "load_ple_block_ids", []) or [])
        rows = []
        for index, block_id in enumerate(vllm_block_ids):
            if index < attn_count:
                compressed_id = (
                    compressed_ids[index] if index < len(compressed_ids) else 0
                )
                rows.append(
                    layout.row_record_ptrs(
                        row_id, "attn", block_id, compressed_id, 0
                    )
                )
            else:
                ple_index = index - attn_count
                ple_id = ple_ids[ple_index] if ple_index < len(ple_ids) else 0
                rows.append(
                    layout.row_record_ptrs(row_id, "mamba", block_id, 0, ple_id)
                )
        if not rows:
            return np.zeros((0, 0), dtype=np.uint64)
        return np.asarray(rows, dtype=np.uint64)

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
                request_meta = metadata.request_meta[request_id]
                if getattr(self.kv_cache_layout, "has_side_segments", False):
                    row_ptrs = self._layerwise_row_ptrs(
                        request_meta,
                        vllm_block_ids,
                        getattr(request_meta, "load_full_attn_count", 0),
                        row_id,
                        is_dump=False,
                    )
                else:
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
        self._dump_row_inputs = []
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
        logger.info(
            "HLA layerwise load: requests=%s blocks=%s side_segments=%s",
            len(self.request_data),
            sum(self._load_block_counts.values()),
            bool(getattr(self.kv_cache_layout, "has_side_segments", False)),
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

        if getattr(self.kv_cache_layout, "has_side_segments", False):
            row_ptr_chunks = [
                self._layerwise_row_ptrs(
                    request, vllm_ids, full_attn_count, row_id, is_dump=True
                )
                for request, vllm_ids, full_attn_count in self._dump_row_inputs
            ]
            row_ptrs = (
                np.concatenate(row_ptr_chunks, axis=0)
                if row_ptr_chunks
                else np.zeros((0, 0), dtype=np.uint64)
            )
        else:
            row_ptrs = self.kv_cache_layout.extract_block_addrs_for_row(
                total_vllm_block_ids, row_id
            )
        shard_indexs = [row_id] * len(total_ucm_block_ids)
        if row_id == self.row_ids[0]:
            attn_records = sum(count for _, _, count in self._dump_row_inputs)
            sink_ptr = int(getattr(self.kv_cache_layout, "sink_ptr", 0) or 0)
            first_ptrs = row_ptrs[0].tolist() if len(row_ptrs) else []
            sink_slots = (
                sum(1 for ptr in first_ptrs if int(ptr) == sink_ptr) if sink_ptr else 0
            )
            logger.info(
                "HLA layerwise save: row=%s layer=%s blocks=%s attn_records=%s "
                "mamba_records=%s side_segments=%s first=%s "
                "first_record_sink_slots=%s/%s",
                row_id,
                layer_name,
                len(total_ucm_block_ids),
                attn_records,
                len(total_ucm_block_ids) - attn_records,
                bool(getattr(self.kv_cache_layout, "has_side_segments", False)),
                _hash_prefix(total_ucm_block_ids[0]),
                sink_slots,
                len(first_ptrs),
            )
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
        self._dump_row_inputs = []
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
            self._dump_row_inputs.append((request, list(scoped_vllm), int(n or 0)))
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
