"""Exercise the optional relation argument without native runtime dependencies."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class FakeArray:
    pass


class FakeNativeStore:
    def __init__(self):
        self.calls = []

    def Load(self, *args):
        self.calls.append(("load", args))
        return 10

    def Dump(self, *args):
        self.calls.append(("dump", args))
        return 11


def load_connector():
    numpy = types.ModuleType("numpy")
    numpy.ndarray = FakeArray
    numpy.uint8 = object()
    numpy.uint64 = object()
    numpy.frombuffer = lambda value, dtype: value
    numpy.array = lambda value, dtype: value
    store_base = types.ModuleType("ucm.store.ucmstore_v1")
    store_base.Task = type("Task", (), {})
    store_base.UcmKVStoreBaseV1 = type("UcmKVStoreBaseV1", (), {})
    native = types.ModuleType("ucm.store.pipeline.ucmpipelinestore")
    native.PipelineStore = FakeNativeStore
    native.StoreNotFoundError = type("StoreNotFoundError", (Exception,), {})
    native.StoreUnhealthyError = type("StoreUnhealthyError", (Exception,), {})
    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    modules = {
        "numpy": numpy,
        "torch": torch,
        "ucm.store.ucmstore_v1": store_base,
        "ucm.store.pipeline.ucmpipelinestore": native,
    }
    path = Path(__file__).resolve().parents[1] / "ucm/store/pipeline/connector.py"
    spec = importlib.util.spec_from_file_location("pipeline_connector_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            del sys.modules[spec.name]
    return module


class PipelineBlockRelationsTest(unittest.TestCase):
    def setUp(self):
        self.module = load_connector()
        self.store = self.module.UcmPipelineStore.__new__(self.module.UcmPipelineStore)
        self.store.store_ = FakeNativeStore()

    def test_default_calls_keep_previous_native_signature(self):
        self.store.load_data([b"a"], [0], [[123]])
        self.store.dump_data([b"a"], [0], [[123]])
        self.assertEqual(len(self.store.store_.calls[0][1]), 5)
        self.assertEqual(len(self.store.store_.calls[1][1]), 4)

    def test_relations_are_forwarded_for_load_and_dump(self):
        relations = [(b"b", b"a")]
        self.store.load_data([b"b"], [0], [[123]], block_relations=relations)
        self.store.dump_data([b"b"], [0], [[123]], block_relations=relations)
        self.assertIs(self.store.store_.calls[0][1][-1], relations)
        self.assertIs(self.store.store_.calls[1][1][-1], relations)


if __name__ == "__main__":
    unittest.main()
