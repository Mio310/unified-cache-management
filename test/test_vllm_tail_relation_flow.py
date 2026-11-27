"""CPU-only tests of the scheduler/worker relation data flow.

The connector imports accelerator-only dependencies, so these tests compile its
actual methods and metadata classes from the source AST without importing vLLM.
"""
import ast
from dataclasses import dataclass, field
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("block_relations", ROOT / "ucm/integration/vllm/block_relations.py")
relations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relations)
source = (ROOT / "ucm/integration/vllm/ucm_connector.py").read_text(encoding="utf-8")
tree = ast.parse(source)
names = {"_generate_dispatch_meta", "update_state_after_alloc", "_validate_tail_scope", "_store_block_relations", "_load_relation_options", "_dump_relation_options"}
classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in {"RequestMeta", "RequestDispatchMeta"}]
methods = [n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "UCMDirectConnector" for n in c.body if isinstance(n, ast.FunctionDef) and n.name in names]
assert {n.name for n in methods} == names
proxy = ast.ClassDef(name="Proxy", bases=[], keywords=[], body=methods, decorator_list=[])
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *classes, proxy], type_ignores=[])
namespace = {"dataclass": dataclass, "field": field, "math": math, "logger": SimpleNamespace(warning=lambda *a, **k: None), "relations_for_slice": relations.relations_for_slice, "select_relations": relations.select_relations, "map_relation_keys": relations.map_relation_keys}
exec(compile(ast.fix_missing_locations(module), str(ROOT / "ucm/integration/vllm/ucm_connector.py"), "exec"), namespace)
namespace["UCMDirectConnector"] = namespace["Proxy"]
namespace["UCMLayerWiseConnector"] = type("Layerwise", (), {})

manager_tree = ast.parse(
    (ROOT / "ucm/integration/vllm/rank_consistency.py").read_text(encoding="utf-8")
)
manager_methods = [
    method
    for cls in manager_tree.body
    if isinstance(cls, ast.ClassDef) and cls.name == "RankConsistencyManager"
    for method in cls.body
    if isinstance(method, ast.FunctionDef) and method.name in {"submit_load", "submit_dump"}
]
assert {method.name for method in manager_methods} == {"submit_load", "submit_dump"}
manager_module = ast.Module(
    body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        ast.ClassDef(name="ManagerProxy", bases=[], keywords=[], body=manager_methods, decorator_list=[]),
    ],
    type_ignores=[],
)
exec(compile(ast.fix_missing_locations(manager_module), "rank_consistency.py", "exec"), namespace)

class RelationFlowTest(unittest.TestCase):
    def make_connector(self, enabled=True, mla=True):
        c = namespace["Proxy"]()
        c._tail_relations_enabled = enabled
        c.cp_world_size = 1
        c.tp_size = 2
        c.tp_rank = 1
        c.is_mla = mla
        c.request_hasher = lambda b: b"rank:" + b
        c.block_size = 1
        c._vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=1)
        )
        return c

    def test_dispatch_slice_and_worker_key_transform(self):
        c = self.make_connector(mla=False)
        meta = namespace["RequestMeta"](ucm_block_ids=[b"a", b"b", b"c", b"d"], hbm_hit_block_num=1, total_hit_block_num=2, num_token_ids=4, token_processed=2)
        dispatch = c._generate_dispatch_meta(meta, 2, [1, 2, 3, 4])
        self.assertEqual(dispatch.load_relations, [(b"b", b"a")])
        self.assertEqual(dispatch.dump_relations, [(b"c", b"b"), (b"d", b"c")])
        self.assertEqual(c._load_relation_options(dispatch, [b"b"]), {"block_relations": [(b"rank:b", b"rank:a")]})

    def test_async_dispatch_preserves_parent(self):
        c = self.make_connector()
        c.use_request_async_load = True
        c.requests_meta = {"req": namespace["RequestMeta"](ucm_block_ids=[b"a", b"b", b"c"], hbm_hit_block_num=1)}
        c._pending_async_load_dispatches = {}
        c._async_load_req_ids = set()
        c.update_state_after_alloc(SimpleNamespace(request_id="req"), SimpleNamespace(get_block_ids=lambda: [[1, 2, 3]]), 2)
        self.assertEqual(c._pending_async_load_dispatches["req"].load_relations, [(b"b", b"a"), (b"c", b"b")])

    def test_mla_sample_and_merged_requests_keep_original_parents(self):
        c = self.make_connector()
        r1 = SimpleNamespace(dump_relations=[(b"a", None), (b"b", b"a"), (b"c", b"b")], picked=[b"a", b"c"])
        r2 = SimpleNamespace(dump_relations=[(b"x", None), (b"y", b"x")], picked=[b"y"])
        c._get_rank_dump_blocks = lambda req: (req.picked, [])
        options = c._dump_relation_options(SimpleNamespace(request_meta={"r1": r1, "r2": r2}), ["r1", "r2"], [b"a", b"c", b"y"])
        self.assertEqual(options["block_relations"], [(b"a", None), (b"c", b"b"), (b"y", b"x")])
        r2.dump_relations = [(b"c", b"x")]
        r2.picked = [b"c"]
        with self.assertRaisesRegex(ValueError, "conflicting parents"):
            c._dump_relation_options(SimpleNamespace(request_meta={"r1": r1, "r2": r2}), ["r1", "r2"], [b"a", b"c"])

    def test_opt_out_and_missing_relations_fail_closed(self):
        c = self.make_connector(enabled=False)
        self.assertEqual(c._load_relation_options(SimpleNamespace(), [b"a"]), {})
        meta = namespace["RequestMeta"](
            ucm_block_ids=[b"a", b"b", b"c"],
            hbm_hit_block_num=1,
            total_hit_block_num=2,
            num_token_ids=3,
            token_processed=2,
        )
        dispatch = c._generate_dispatch_meta(meta, 1, [10, 11, 12])
        self.assertEqual(dispatch.load_block_ids, ([b"b"], [11]))
        self.assertEqual(dispatch.dump_block_ids, ([b"c"], [12]))
        self.assertIsNone(dispatch.load_relations)
        self.assertIsNone(dispatch.dump_relations)
        c._tail_relations_enabled = True
        with self.assertRaisesRegex(ValueError, "missing block relations"):
            c._load_relation_options(SimpleNamespace(load_relations=None), [b"a"])
        c.cp_world_size = 2
        with self.assertRaisesRegex(ValueError, "context parallelism"):
            c._store_block_relations([], [])
        c.cp_world_size = 1
        c._vllm_config.parallel_config.pipeline_parallel_size = 2
        with self.assertRaisesRegex(ValueError, "pipeline parallelism"):
            c._store_block_relations([], [])
        c._vllm_config.parallel_config.pipeline_parallel_size = 1
        subclass = type("UnverifiedConnector", (namespace["Proxy"],), {})()
        with self.assertRaisesRegex(ValueError, "connector"):
            subclass._validate_tail_scope()

    def test_manager_omits_optional_argument_or_forwards_it(self):
        calls = []

        class Store:
            supports_distributed_backend_load = True

            def load_data(self, *args, **kwargs):
                calls.append(("load", args, kwargs))
                return object()

            def dump_data(self, *args, **kwargs):
                calls.append(("dump", args, kwargs))
                return object()

        manager = namespace["ManagerProxy"]()
        manager.enabled = False
        manager._load_task_contexts = {}
        manager._dump_task_contexts = {}
        store = Store()
        manager.submit_load(store, {}, [b"b"], [0], [], distribute_backend_load=True)
        manager.submit_dump(store, {}, [b"b"], [0], [], 0)
        self.assertEqual(calls[0][2], {"distribute_backend_load": True})
        self.assertEqual(calls[1][2], {})
        relation = [(b"b", b"a")]
        manager.submit_load(store, {}, [b"b"], [0], [], block_relations=relation)
        manager.submit_dump(store, {}, [b"b"], [0], [], 0, block_relations=relation)
        self.assertEqual(calls[2][2]["block_relations"], relation)
        self.assertEqual(calls[3][2]["block_relations"], relation)

if __name__ == "__main__":
    unittest.main()
