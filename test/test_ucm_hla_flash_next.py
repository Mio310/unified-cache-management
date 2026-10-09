from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

import numpy as np

from ucm.integration.vllm.hla_connector import (
    GroupInfo,
    HLARequestMeta,
    HybridLinearAttentionLayout,
    KVCacheGroupManager,
    UCMHybridLinearAttentionConnector,
    _group_kind,
    _layer_role,
    assemble_side_record_ptrs,
    select_ple_carrier_group_id,
)
from ucm.integration.vllm.request_hasher import RequestHasher


def _config():
    return SimpleNamespace(
        model_config=SimpleNamespace(model="org/model", dtype="bfloat16"),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        speculative_config=None,
        additional_config={},
    )


def _spec(block_size, prefix_cacheable=True):
    return SimpleNamespace(block_size=block_size, prefix_cacheable=prefix_cacheable)


def _group(layer_names, block_size, prefix_cacheable=True):
    return SimpleNamespace(
        layer_names=list(layer_names),
        kv_cache_spec=_spec(block_size, prefix_cacheable),
    )


def _manager(groups, tensors=None):
    hasher = RequestHasher(_config(), 0)
    config = SimpleNamespace(
        kv_cache_groups=groups,
        kv_cache_tensors=[] if tensors is None else tensors,
    )
    return KVCacheGroupManager(config, hasher, hasher.seed), hasher


def _info(group_id, block_size, names, seed, kind, is_mamba_align=False):
    return GroupInfo(
        group_id=group_id,
        block_size=block_size,
        layer_names=tuple(names),
        seed=seed,
        is_mamba_align=is_mamba_align,
        kind=kind,
    )


def test_layer_role_matches_components_not_substrings():
    assert _layer_role("model.layers.3.self_attn.raw_key_cache") == "raw"
    assert _layer_role("model.layers.3.self_attn.compressed_key_cache") == "compressed"
    assert _layer_role("model.layers.1.ple") == "ple"
    assert _layer_role("model.layers.1.complete") == ""
    assert (
        _group_kind(["model.layers.3.raw_key_cache"], _spec(8, prefix_cacheable=False))
        == "raw"
    )
    assert _group_kind(["model.layers.0.self_attn"], _spec(16)) == "full_attn"


def test_raw_ring_is_excluded_from_lcm_and_lookup_groups():
    manager, _ = _manager(
        [
            _group(["model.layers.0.self_attn"], 16),
            _group(["model.layers.3.raw_key_cache"], 10, prefix_cacheable=False),
            _group(["model.layers.4.self_attn"], 24),
            _group(["model.layers.3.compressed_key_cache"], 16),
            _group(["model.layers.1.ple"], 16),
        ]
    )

    assert manager.lcm_block_size == 48
    assert [group.group_id for group in manager.full_attn_groups] == [0, 2]
    assert manager.state_groups == []
    assert manager.compressed_group.group_id == 3
    assert manager.ple_group.group_id == 4
    assert manager.has_side_caches
    assert manager.compressed_index_aligned
    request = SimpleNamespace(all_token_ids=list(range(16)))
    assert manager.compute_block_hashes(manager.groups_by_id[1], request) == [b""]
    assert manager.compute_block_hashes(manager.groups_by_id[3], request) == [b""]
    assert manager.compute_block_hashes(manager.groups_by_id[4], request) == [b""]


def test_compressed_block_size_must_match_primary_attention():
    with pytest.raises(ValueError, match="compressed_key_cache block_size"):
        _manager(
            [
                _group(["model.layers.0.self_attn"], 16),
                _group(["model.layers.3.compressed_key_cache"], 8),
            ]
        )


def test_ple_carrier_prefers_shared_tensor_then_layer_index():
    ple = _info(3, 16, ["model.layers.1.ple"], b"ple", "ple")
    first = _info(1, 16, ["model.layers.0.linear_attn"], b"m0", "mamba", True)
    second = _info(2, 16, ["model.layers.1.linear_attn"], b"m1", "mamba", True)
    shared = SimpleNamespace(
        shared_by=["model.layers.1.linear_attn", "model.layers.1.ple"]
    )
    assert (
        select_ple_carrier_group_id(
            [first, second], [ple], SimpleNamespace(kv_cache_tensors=[shared])
        )
        == 2
    )

    indexed = select_ple_carrier_group_id(
        [first, second], [ple], SimpleNamespace(kv_cache_tensors=[])
    )
    assert indexed == 2

    other = _info(3, 16, ["model.layers.9.ple"], b"ple", "ple")
    assert (
        select_ple_carrier_group_id(
            [first, second], [other], SimpleNamespace(kv_cache_tensors=[])
        )
        == 1
    )


def test_side_record_pointers_fill_only_the_live_segments():
    kinds = ["hybrid", "hybrid", "compressed", "ple"]
    bases = [1000, 2000, 3000, 4000]
    strides = [100, 100, 50, 80]
    attention = assemble_side_record_ptrs(
        kinds, bases, strides, sink_ptr=9, record_kind="attn",
        primary_block_id=2, compressed_block_id=4, ple_block_id=5,
    )
    mamba = assemble_side_record_ptrs(
        kinds, bases, strides, sink_ptr=9, record_kind="mamba",
        primary_block_id=3, compressed_block_id=4, ple_block_id=5,
    )
    other = assemble_side_record_ptrs(
        kinds, bases, strides, sink_ptr=9, record_kind="mamba",
        primary_block_id=3, compressed_block_id=4, ple_block_id=0,
    )

    assert attention == [1200, 2200, 3200, 9]
    assert mamba == [1300, 2300, 9, 4400]
    assert other == [1300, 2300, 9, 9]


def _dispatch_connector(groups):
    connector = UCMHybridLinearAttentionConnector.__new__(
        UCMHybridLinearAttentionConnector
    )
    connector.group_manager = groups
    connector.is_mla = False
    return connector


def _dispatch_manager(hasher):
    attn = _info(0, 16, ["model.layers.0.self_attn"], hasher.seed, "full_attn")
    other = _info(
        1, 16, ["model.layers.0.linear_attn"], hasher((b"mamba", 1)), "mamba", True
    )
    carrier = _info(
        2, 16, ["model.layers.1.linear_attn"], hasher((b"mamba", 2)), "mamba", True
    )
    compressed = _info(
        3, 16, ["model.layers.0.compressed_key_cache"], hasher.seed, "compressed"
    )
    ple = _info(4, 16, ["model.layers.1.ple"], hasher.seed, "ple")
    raw = _info(5, 10, ["model.layers.0.raw_key_cache"], hasher.seed, "raw")
    manager = KVCacheGroupManager.__new__(KVCacheGroupManager)
    manager.request_hasher = hasher
    manager.groups_by_id = [attn, other, carrier, compressed, ple, raw]
    manager.full_attn_groups = [attn]
    manager.state_groups = [other, carrier]
    manager.compressed_group = compressed
    manager.ple_group = ple
    manager.ple_carrier_group_id = carrier.group_id
    manager.compressed_index_aligned = True
    manager.lcm_block_size = 16
    return manager


def test_dispatch_pairs_compressed_with_attention_and_ple_with_carrier_mamba():
    hasher = RequestHasher(_config(), 0)
    manager = _dispatch_manager(hasher)
    connector = _dispatch_connector(manager)
    req_meta = HLARequestMeta(
        ucm_block_ids=[b"skip", b"keep"],
        hbm_hit_block_num=0,
        total_hit_block_num=2,
        num_token_ids=32,
        token_processed=0,
        group_ucm_block_ids=[
            [b"skip", b"keep"],
            [b"", b""],
            [b"", b""],
            [b"compressed-key"],
            [b"ple-key"],
            [b"raw-key"],
        ],
        group_vllm_block_ids=[
            [0, 12],
            [30, 31],
            [0, 32],
            [51, 52],
            [0, 41],
            [7],
        ],
    )
    meta = connector._generate_hla_dispatch_meta(
        req_meta,
        32,
        tuple(list(ids) for ids in req_meta.group_vllm_block_ids),
        need_load=True,
        request_id="req",
        incoming_block_ids_are_full=True,
    )

    assert meta.load_full_attn_count == 1
    assert meta.dump_full_attn_count == 1
    assert meta.load_block_ids[1] == [12, 31, 32]
    assert meta.dump_block_ids[1] == [12, 31, 32]
    assert meta.load_compressed_block_ids == [52]
    assert meta.dump_compressed_block_ids == [52]
    assert meta.load_ple_block_ids == [0, 41]
    assert meta.dump_ple_block_ids == [0, 41]
    assert b"raw-key" not in meta.dump_block_ids[0]
    assert b"compressed-key" not in meta.dump_block_ids[0]
    assert b"ple-key" not in meta.dump_block_ids[0]


def test_legacy_hla_dispatch_keeps_full_attn_count_at_zero():
    hasher = RequestHasher(_config(), 0)
    attn = _info(0, 16, ["model.layers.0.self_attn"], hasher.seed, "full_attn")
    mamba = _info(
        1, 16, ["model.layers.0.linear_attn"], hasher((b"mamba", 1)), "mamba", True
    )
    manager = KVCacheGroupManager.__new__(KVCacheGroupManager)
    manager.request_hasher = hasher
    manager.groups_by_id = [attn, mamba]
    manager.full_attn_groups = [attn]
    manager.state_groups = [mamba]
    manager.compressed_group = None
    manager.ple_group = None
    manager.ple_carrier_group_id = None
    manager.compressed_index_aligned = False
    manager.lcm_block_size = 16
    connector = _dispatch_connector(manager)
    req_meta = HLARequestMeta(
        ucm_block_ids=[b"attn0"],
        hbm_hit_block_num=0,
        total_hit_block_num=1,
        num_token_ids=16,
        token_processed=0,
        group_ucm_block_ids=[[b"attn0"], [b""]],
        group_vllm_block_ids=[[11], [31]],
    )
    meta = connector._generate_hla_dispatch_meta(
        req_meta,
        16,
        ([11], [31]),
        need_load=True,
        request_id="req",
        incoming_block_ids_are_full=True,
    )

    assert meta.dump_full_attn_count == 0
    assert meta.load_full_attn_count == 0
    assert meta.dump_block_ids[1] == [11, 31]
    assert meta.dump_compressed_block_ids == []
    assert meta.dump_ple_block_ids == []


def test_layerwise_row_pointers_follow_the_record_kind():
    layout = HybridLinearAttentionLayout.__new__(HybridLinearAttentionLayout)
    layout.row_segment_kinds = [["hybrid", "compressed", "ple"]]
    layout.row_slices = [slice(0, 3)]
    layout.base_ptrs = np.array([1000, 3000, 4000], dtype=np.uint64)
    layout.block_stride_lists = np.array([100, 50, 80], dtype=np.uint64)
    layout.sink_ptr = 9

    attention = layout.row_record_ptrs(0, "attn", 2, 4, 0)
    mamba = layout.row_record_ptrs(0, "mamba", 3, 4, 5)

    assert attention == [1200, 3200, 9]
    assert mamba == [1300, 9, 4400]
