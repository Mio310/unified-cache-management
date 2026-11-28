import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


# Load this dependency-free helper without importing the vLLM integration.
_path = Path(__file__).resolve().parents[1] / "ucm/integration/vllm/block_relations.py"
_spec = importlib.util.spec_from_file_location("block_relations", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
map_relation_keys = _module.map_relation_keys
relations_for_slice = _module.relations_for_slice
select_relations = _module.select_relations
validate_tail_eviction_config = _module.validate_tail_eviction_config
validate_tail_eviction_scope = _module.validate_tail_eviction_scope
apply_tail_eviction_store_config = _module.apply_tail_eviction_store_config


class BlockRelationsTest(unittest.TestCase):
    def test_tail_eviction_config_and_store_propagation(self):
        self.assertFalse(validate_tail_eviction_config({}))
        self.assertFalse(validate_tail_eviction_config({"enable_dram_tail_eviction": False}))
        config = {
            "enable_dram_tail_eviction": True,
            "ucm_connectors": [{
                "ucm_connector_name": "UcmPipelineStore",
                "ucm_connector_config": {"store_pipeline": "Cache|Posix"},
            }],
        }
        self.assertTrue(validate_tail_eviction_config(config))
        store_config = {}
        apply_tail_eviction_store_config(config, store_config)
        self.assertEqual(store_config, {"enable_dram_tail_eviction": True})
        disabled_store_config = {}
        apply_tail_eviction_store_config({}, disabled_store_config)
        self.assertEqual(disabled_store_config, {})
        with self.assertRaisesRegex(ValueError, "store pipeline"):
            config["ucm_connectors"][0]["ucm_connector_config"]["store_pipeline"] = "Posix"
            validate_tail_eviction_config(config)
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            validate_tail_eviction_config({"enable_dram_tail_eviction": "false"})
        with self.assertRaisesRegex(ValueError, "top level"):
            validate_tail_eviction_config({
                "ucm_connectors": [{"ucm_connector_config": {
                    "enable_dram_tail_eviction": True,
                }}],
            })

    def test_tail_eviction_rejects_cp_pp_and_mamba(self):
        parallel = SimpleNamespace(pipeline_parallel_size=1,
                                   prefill_context_parallel_size=1,
                                   decode_context_parallel_size=1)
        validate_tail_eviction_scope(parallel)
        parallel.pipeline_parallel_size = 2
        with self.assertRaisesRegex(ValueError, "pipeline parallelism"):
            validate_tail_eviction_scope(parallel)
        parallel.pipeline_parallel_size = 1
        parallel.decode_context_parallel_size = 2
        with self.assertRaisesRegex(ValueError, "context parallelism"):
            validate_tail_eviction_scope(parallel)
        parallel.decode_context_parallel_size = 1
        class MambaSpec:
            pass
        class DerivedMamba(MambaSpec):
            pass
        grouped = SimpleNamespace(kv_cache_specs={"layer": DerivedMamba()})
        cache = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=grouped)])
        with self.assertRaisesRegex(ValueError, "Mamba"):
            validate_tail_eviction_scope(parallel, cache)

    def test_slice_preserves_preceding_parent_and_explicit_root(self):
        chain = [b"a", b"b", b"c", b"d"]
        self.assertEqual(relations_for_slice(chain, 0, 2), [(b"a", None), (b"b", b"a")])
        self.assertEqual(relations_for_slice(chain, 2, 4), [(b"c", b"b"), (b"d", b"c")])
        self.assertEqual(relations_for_slice(chain, 4, 4), [])
        for start, end in [(-1, 2), (2, 5), (3, 2)]:
            with self.assertRaises(ValueError):
                relations_for_slice(chain, start, end)

    def test_mla_sampling_keeps_original_direct_parent(self):
        chain = [b"a", b"b", b"c", b"d", b"e"]
        relations = relations_for_slice(chain, 0, len(chain))
        self.assertEqual(
            select_relations(relations, chain[::2]),
            [(b"a", None), (b"c", b"b"), (b"e", b"d")],
        )
        with self.assertRaisesRegex(ValueError, "missing parent"):
            select_relations(relations, [b"unknown"])

    def test_rank_transform_applies_to_both_ends_and_preserves_root(self):
        self.assertEqual(
            map_relation_keys(
                [(b"a", None), (b"b", b"a")], lambda value: b"rank1:" + value
            ),
            [(b"rank1:a", None), (b"rank1:b", b"rank1:a")],
        )

    def test_duplicate_relations_are_idempotent_and_conflicts_fail(self):
        self.assertEqual(
            select_relations(
                [(b"a", None), (b"b", b"a"), (b"b", b"a")], [b"b", b"b"]
            ),
            [(b"b", b"a")],
        )
        with self.assertRaisesRegex(ValueError, "conflicting parents"):
            select_relations([(b"b", b"a"), (b"b", b"c")], [b"b"])
        with self.assertRaisesRegex(ValueError, "transformation collision"):
            map_relation_keys([(b"a", None), (b"b", b"a")], lambda _: b"collision")


if __name__ == "__main__":
    unittest.main()
