"""CPU byte-roundtrip tests for the CUDA HLA transfer contract.

Extract production classes without importing vLLM/CUDA, as in
test_kv_cache_layout.py. NumPy-backed tensors let the fake store perform the
actual gather/scatter, including padding and independent physical block IDs.
"""

import ast
import ctypes
import hashlib
import importlib.util
import math
import pickle
import sys
import time
import unittest
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HLA = ROOT / "ucm/integration/vllm/hla_connector.py"
spec = importlib.util.spec_from_file_location(
    "qsa_test_helpers", ROOT / "ucm/integration/vllm/qsa_layout.py"
)
qsa = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = qsa
spec.loader.exec_module(qsa)


class Tensor:
    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        self.device = "cuda:0"

    def __getitem__(self, index):
        return Tensor(self.data[index])

    def data_ptr(self):
        return self.data.ctypes.data

    def numel(self):
        return self.data.size

    def element_size(self):
        return self.data.itemsize

    def stride(self, dim):
        return self.data.strides[dim] // self.data.itemsize

    def is_contiguous(self):
        return self.data.flags.c_contiguous


TORCH = NS(
    Tensor=Tensor,
    uint8=np.uint8,
    zeros=lambda n, **kw: Tensor(np.zeros(n, dtype=np.uint8)),
    empty=lambda n, **kw: Tensor(np.empty(n, dtype=np.uint8)),
)


@dataclass
class FullAttentionSpec:
    block_size: int = 4
    tokens_per_state: int = 1


@dataclass
class MambaSpec:
    block_size: int = 4
    mamba_cache_mode: str = "align"


@dataclass
class CircularBufferSpec:
    block_size: int = 2


class Hasher:
    def __call__(self, value):
        return hashlib.md5(pickle.dumps(value)).digest()

    def make_request_block_hasher(self, block_size, seed):
        return lambda req: [
            self((seed, tuple(req.all_token_ids[:end])))
            for end in range(block_size, len(req.all_token_ids) + 1, block_size)
        ]


def load_symbols():
    ns = dict(vars(qsa))
    ns.update(
        __name__=__name__,
        torch=TORCH,
        np=np,
        math=math,
        time=time,
        dataclass=dataclass,
        field=field,
        defaultdict=defaultdict,
        FullAttentionSpec=FullAttentionSpec,
        MambaSpec=MambaSpec,
        UniformTypeKVCacheSpecs=type("UniformTypeKVCacheSpecs", (), {}),
        MLAAttentionSpec=type("MLAAttentionSpec", (FullAttentionSpec,), {}),
        UCMDirectConnector=type("Base", (), {}),
        UCMLiteConnector=type("Lite", (), {}),
        SupportsHMA=type("HMA", (), {}),
        KVConnectorMetadata=object,
        extract_layer_index=qsa.layer_index,
        logger=Mock(),
        _record_counter=Mock(),
        ucmmetrics=Mock(),
        current_platform=NS(device_type="cuda", is_cuda_alike=lambda: True),
    )
    selected = {
        "RequestMeta",
        "RequestDispatchMeta",
        "KVCacheLayout",
        "UCMConnectorMetadata",
    }
    base = ast.parse(
        (ROOT / "ucm/integration/vllm/ucm_connector.py").read_text("utf-8-sig")
    )
    nodes = [n for n in base.body if getattr(n, "name", None) in selected]
    nodes += [
        n
        for n in ast.parse(HLA.read_text("utf-8-sig")).body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(HLA), "exec"), ns)
    return ns


SYMS = load_symbols()


def fixture(rows=12):
    groups = [[] for _ in range(6)]
    caches, tensors = {}, []
    for row in range(rows):
        names = [f"model.layers.{row * 4 + j}.attn" for j in range(4)]
        main = np.full((32, 16), row + 1, dtype=np.uint8)
        for j, name in enumerate(names):
            groups[0 if j == 3 else j + 1].append(name)
            caches[name] = Tensor(main)
        tensors.append(NS(size=main.nbytes, shared_by=names))
    for row in range(rows):
        prefix = f"model.layers.{row * 4 + 3}.attn."
        names = [prefix + "compressed_key_cache", prefix + "raw_key_cache"]
        aux = np.zeros((32, 12), dtype=np.uint8)
        for i, name in enumerate(names):
            groups[4 + i].append(name)
            caches[name] = Tensor(aux[:, :4] if i == 0 else aux[:, 4:])
            for block in range(32):
                caches[name].data[block].fill(40 + block + 40 * i)
        tensors.append(NS(size=aux.nbytes, shared_by=names))
    specs = [
        FullAttentionSpec(),
        *[MambaSpec() for _ in range(3)],
        FullAttentionSpec(tokens_per_state=2),
        CircularBufferSpec(),
    ]
    config = NS(
        num_blocks=32,
        kv_cache_tensors=tensors,
        kv_cache_groups=[
            NS(layer_names=names, kv_cache_spec=spec)
            for names, spec in zip(groups, specs)
        ],
    )
    model = NS(
        parallel_config=NS(pipeline_parallel_size=1),
        model_config=NS(hf_text_config=NS(num_hidden_layers=rows * 4)),
    )
    layout = SYMS["HybridLinearAttentionLayout"](caches, {}, model, config)
    manager = SYMS["KVCacheGroupManager"](config, Hasher(), b"seed")
    connector = SYMS["UCMHybridLinearAttentionConnector"].__new__(
        SYMS["UCMHybridLinearAttentionConnector"]
    )
    connector.group_manager = manager
    connector.qsa = manager.qsa
    connector.request_hasher = Hasher()
    connector.is_mla = False
    connector.tp_rank, connector.tp_size = 0, 1
    hashes = manager.compute_all_group_block_ids(NS(all_token_ids=list(range(16))))
    tables = [
        [1, 2, 3, 4],
        [0, 0, 0, 5],
        [0, 0, 0, 6],
        [0, 0, 0, 7],
        [8, 9, 10, 11],
        [12],
    ]
    meta = SYMS["HLARequestMeta"](
        num_token_ids=16,
        group_ucm_block_ids=hashes,
        group_vllm_block_ids=tables,
    )
    return NS(
        layout=layout,
        manager=manager,
        connector=connector,
        meta=meta,
        tables=tables,
        hashes=hashes,
        config=config,
        caches=caches,
    )


def dispatch(f, tokens=16, **kwargs):
    return f.connector._generate_hla_dispatch_meta(
        f.meta, tokens, tuple([] for _ in f.tables), **kwargs
    )


def read_record(ptrs, sizes):
    return b"".join(ctypes.string_at(int(p), int(s)) for p, s in zip(ptrs, sizes))


class TestQSAHLA(unittest.TestCase):
    def test_qsa_keys_match_original_hla_for_the_same_prefix(self):
        f = fixture(1)
        original_config = NS(
            num_blocks=f.config.num_blocks,
            kv_cache_tensors=f.config.kv_cache_tensors[:1],
            kv_cache_groups=f.config.kv_cache_groups[:4],
        )
        original = SYMS["KVCacheGroupManager"](
            original_config, Hasher(), b"seed"
        )
        hashes = original.compute_all_group_block_ids(
            NS(all_token_ids=list(range(16)))
        )
        self.assertEqual(f.hashes[0], hashes[0])
        for group, old_group in zip(f.manager.groups_by_id[:4], original.groups_by_id):
            self.assertEqual(group.seed, old_group.seed)
        for group, old_group in zip(f.manager.state_groups, original.state_groups):
            for boundary in (4, 8, 12, 16):
                self.assertEqual(
                    f.manager.compute_mamba_align_state_hash(group, boundary, f.hashes),
                    original.compute_mamba_align_state_hash(old_group, boundary, hashes),
                )

    def test_non_qsa_layout_and_dispatch_keep_existing_behavior(self):
        f = fixture(1)
        f.config.kv_cache_groups = f.config.kv_cache_groups[:4]
        f.config.kv_cache_tensors = f.config.kv_cache_tensors[:1]
        manager = SYMS["KVCacheGroupManager"](f.config, Hasher(), b"seed")
        self.assertIsNone(manager.qsa)
        f.connector.group_manager = manager
        f.connector.qsa = None
        f.meta.group_ucm_block_ids = f.hashes[:4]
        f.meta.group_vllm_block_ids = f.tables[:4]
        f.tables = f.tables[:4]
        result = dispatch(f)
        self.assertEqual(len(result.dump_block_ids[0]), 7)
        self.assertEqual(result.dump_qsa_records, [])
        model = NS(
            parallel_config=NS(pipeline_parallel_size=1),
            model_config=NS(hf_text_config=NS(num_hidden_layers=4)),
        )
        caches = {n: t for n, t in f.caches.items() if qsa.qsa_role(n) is None}
        layout = SYMS["HybridLinearAttentionLayout"](caches, {}, model, f.config)
        self.assertEqual(layout.row_tensor_size_lists, [[16]])

    def test_24_physical_tensors_become_12_fixed_rows(self):
        f = fixture()
        self.assertEqual(len(f.config.kv_cache_tensors), 24)
        self.assertEqual(f.layout.row_tensor_size_lists, [[16, 4, 8]] * 12)
        self.assertEqual(f.layout.block_size, 12 * 28)
        self.assertEqual(len(f.manager.full_attn_groups), 1)
        self.assertEqual(len(f.manager.state_groups), 3)

    def test_only_final_original_key_has_raw_and_mamba_has_zero_tails(self):
        f = fixture()
        result = dispatch(f)
        keys, blocks = result.dump_block_ids
        records = result.dump_qsa_records
        self.assertEqual(len(records), 7)  # 4 attention + 3 state
        self.assertEqual([bool(r.raw) for r in records], [False] * 3 + [True] + [False] * 3)
        self.assertEqual(keys[:4], f.hashes[0])
        self.assertEqual(len(set(keys)), 7)
        addresses = f.layout.extract_qsa_addrs(records)
        payloads = [read_record(p, f.layout.tensor_size_lists) for p in addresses]
        self.assertEqual({len(p) for p in payloads}, {336})
        for payload in payloads[:3]:
            self.assertEqual(payload[20:28], bytes(8))
        for payload in payloads[4:7]:
            self.assertEqual(payload[16:28], bytes(12))
        self.assertEqual(payloads[0][16:20], bytes([48]) * 4)  # compressed block 8
        self.assertEqual(payloads[3][16:20], bytes([51]) * 4)  # compressed block 11
        self.assertEqual(payloads[3][20:28], bytes([92]) * 8)  # raw block 12

    def test_unaligned_end_never_labels_latest_ring_as_earlier_state(self):
        f = fixture(1)
        result = dispatch(f, 15)
        self.assertEqual(len(result.dump_qsa_records), 3)
        self.assertFalse(any(r.raw for r in result.dump_qsa_records))

    def test_chunk_finishing_partial_block_saves_final_ring(self):
        f = fixture(1)
        dispatch(f, 15)
        result = dispatch(f, 1, need_load=False)
        self.assertEqual(result.dump_qsa_records[0].raw, {5: 12})
        self.assertEqual(
            result.dump_block_ids[0][0], f.hashes[0][3]
        )

    def test_tokens_past_prompt_do_not_claim_prompt_ring_snapshot(self):
        f = fixture(1)
        result = dispatch(f, 17)
        self.assertFalse(any(r.raw for r in result.dump_qsa_records))

    def test_immutable_store_retains_padding_without_an_extra_key(self):
        f = fixture(1)
        result = dispatch(f)
        store = dict(zip(result.dump_block_ids[0], result.dump_qsa_records))
        earlier_key = f.hashes[0][0]
        self.assertFalse(store[earlier_key].raw)
        fresh = fixture(1)
        short = dispatch(fresh, 4)
        for key, record in zip(short.dump_block_ids[0], short.dump_qsa_records):
            store.setdefault(key, record)
        self.assertFalse(store[earlier_key].raw)
        self.assertEqual(short.dump_block_ids[0][0], earlier_key)
        self.assertTrue(short.dump_qsa_records[0].raw)
        self.assertEqual(len(short.dump_block_ids[0]), 1)

    def test_lookup_requires_original_attention_and_all_states(self):
        f = fixture(1)
        stored = set(f.hashes[0])
        for group in f.manager.state_groups:
            stored.add(f.manager.compute_mamba_align_state_hash(group, 16, f.hashes))
        reverse = lambda keys: max(
            (i for i, k in enumerate(keys) if k in stored), default=-1
        )
        prefix = lambda keys: next(
            (i - 1 for i, k in enumerate(keys) if k not in stored), len(keys) - 1
        )
        self.assertEqual(
            f.manager.lookup_external_hit_tokens(0, f.hashes, prefix, reverse)[0], 16
        )
        stored.remove(f.manager.compute_mamba_align_state_hash(f.manager.state_groups[0], 16, f.hashes))
        self.assertEqual(
            f.manager.lookup_external_hit_tokens(0, f.hashes, prefix, reverse)[0], 0
        )

    def test_restore_uses_new_aux_block_tables_and_only_final_raw(self):
        f = fixture(1)
        dumped = dispatch(f)
        addresses = f.layout.extract_qsa_addrs(dumped.dump_qsa_records)
        store = {
            k: read_record(p, f.layout.tensor_size_lists)
            for k, p in zip(dumped.dump_block_ids[0], addresses)
        }
        f.meta.token_processed = 16
        f.meta.total_hit_block_num = 4
        f.meta.group_vllm_block_ids = [
            [13, 14, 15, 16],
            [0, 0, 0, 17],
            [0, 0, 0, 18],
            [0, 0, 0, 19],
            [20, 21, 22, 23],
            [24],
        ]
        result = dispatch(f, 0)
        self.assertEqual(sum(bool(r.raw) for r in result.load_qsa_records), 1)
        for key, record in zip(result.load_block_ids[0], result.load_qsa_records):
            ptrs = f.layout.extract_qsa_addrs([record], is_load=True)[0]
            data, offset = store[key], 0
            for ptr, size in zip(ptrs, f.layout.tensor_size_lists):
                size = int(size)
                ctypes.memmove(int(ptr), data[offset : offset + size], size)
                offset += size
        raw = f.caches["model.layers.3.attn.raw_key_cache"].data
        compressed = f.caches["model.layers.3.attn.compressed_key_cache"].data
        self.assertEqual(raw[24].tobytes(), bytes([92]) * 8)
        self.assertEqual(compressed[23].tobytes(), bytes([51]) * 4)
        self.assertEqual(raw[12].tobytes(), bytes([92]) * 8)

    def test_rejects_missing_auxiliary_block_and_concurrent_discard_writes(self):
        f = fixture(1)
        f.meta.group_vllm_block_ids[4] = []
        with self.assertRaisesRegex(ValueError, "Missing QSA"):
            dispatch(f)
        with self.assertRaisesRegex(ValueError, "serialize"):
            f.layout.extract_qsa_addrs(
                [qsa.QSARecord(1), qsa.QSARecord(2)], is_load=True
            )

    def test_worker_dump_and_load_wait_before_reusing_buffers(self):
        f = fixture(1)
        result = dispatch(f)
        c = f.connector
        c.kv_cache_layout = f.layout
        c.block_data_size = f.layout.block_size
        c.enable_event_sync = True
        c.device = Mock()
        c._get_dump_event_handle = Mock(return_value=123)
        c._connector_worker_meta = Mock()
        c._record_load_error = Mock()
        metadata = SYMS["UCMConnectorMetadata"]({"req": result})
        c._get_connector_metadata = lambda: metadata
        stored, pending = {}, []
        sizes = f.layout.tensor_size_lists

        class Transfers:
            def submit_dump(self, store, context, keys, shards, ptrs, event):
                assert event == 123
                for key, addresses in zip(keys, ptrs):
                    stored.setdefault(key, read_record(addresses, sizes))
                pending.append("dump")
                return "dump"

            def wait_dump(self, task):
                assert pending.pop() == task

            def finish_dump(self, requests):
                assert not pending

            def submit_load(self, store, context, keys, shards, ptrs):
                assert not pending
                pending.append((keys, ptrs.copy()))
                return "load"

            def wait_load(self, task):
                keys, ptrs = pending.pop()
                for key, addresses in zip(keys, ptrs):
                    offset = 0
                    for ptr, size in zip(addresses, sizes):
                        size = int(size)
                        ctypes.memmove(
                            int(ptr), stored[key][offset : offset + size], size
                        )
                        offset += size

        c.store = object()
        c._rank_consistency = Transfers()
        c.wait_for_save()
        self.assertEqual(len(stored), 7)
        c.device.destroy_event_handle.assert_called_once_with(123)
        f.meta.total_hit_block_num = 4
        f.meta.group_vllm_block_ids = [
            [13, 14, 15, 16],
            [0, 0, 0, 17],
            [0, 0, 0, 18],
            [0, 0, 0, 19],
            [20, 21, 22, 23],
            [24],
        ]
        metadata.request_meta["req"] = dispatch(f, 0)
        c.start_load_kv(None)
        self.assertFalse(pending)
        c._connector_worker_meta.mark_failed.assert_not_called()
        raw = f.caches["model.layers.3.attn.raw_key_cache"].data
        self.assertEqual(raw[24].tobytes(), bytes([92]) * 8)

    def test_nonzero_tp_rank_scopes_original_keys(self):
        f = fixture(1)
        result = dispatch(f)
        f.connector.tp_rank, f.connector.tp_size = 1, 2
        keys, blocks = result.dump_block_ids
        original, scoped, physical = f.connector._scope_blocks(keys, blocks, 0, True)
        self.assertEqual(original, keys)
        self.assertEqual(scoped, [Hasher()(key) for key in keys])
        self.assertEqual(physical, blocks)


if __name__ == "__main__":
    unittest.main()
