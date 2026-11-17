"""Request attribution across queueing, batching and real Store submissions."""

import json
import os
import queue
import sys
import types
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from common.sglang_utils import adapter_module, linker_module, trace_module


class LoadTraceTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"UCM_SGLANG_LOAD_TRACE": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.linker = object.__new__(linker_module.UcmDirectLinker)
        self.linker._dp_rank = self.linker._attn_dp_rank = self.linker._tp_rank = 0
        self.linker._lookup_progress = {}
        self.linker._pending_loads = {}
        self.linker._expand = lambda transfers, **kwargs: transfers
        self.transfer = adapter_module.UcmLayerTransfer(
            store_group="kv",
            source_pool="kv",
            pools=("kv",),
            layers=(0, 1),
            shard_size=8,
            tensor_size_list=(8,),
            page_keys=("shared-page",),
            addresses=np.array([[[100]], [[200]]], dtype=np.uint64),
            fragments=((8, 0), (8, 0)),
        )

    def enqueue(self, rid, session):
        token = trace_module._request.set(
            types.SimpleNamespace(rid=rid, session_id=session)
        )
        try:
            return self.linker.load(rid, [self.transfer])
        finally:
            trace_module._request.reset(token)

    def records(self, captured):
        return [
            json.loads(record.getMessage().split("[UCMTrace] ", 1)[1])
            for record in captured.records
        ]

    def test_shared_key_keeps_each_request_and_task_owner(self):
        self.run_shared_key(False)

    def test_prepared_pipeline_path_records_actual_keys(self):
        self.run_shared_key(True)

    def run_shared_key(self, fast_path):
        adapter = adapter_module.UcmDirectStoreAdapter(
            connector_name="test", module_path=None, store_config={}, identity={}
        )
        submitted = []

        def load(ids, shards, addresses):
            submitted.append((ids, shards))
            return len(submitted)

        store = types.SimpleNamespace(
            load_data=load, wait=lambda task: None, check=lambda task: True
        )

        def native_load(ids, shards, addresses, prerequisite, distribute):
            return load(
                [row.tobytes() for row in ids.reshape(-1, 16)], shards, addresses
            )

        native = types.SimpleNamespace(Load=native_load)

        def prepare(transfers):
            items = [
                adapter_module._PreparedStoreTransfer(
                    transfer=t,
                    store=store,
                    block_ids=adapter._block_ids(t),
                    shards_by_layer={0: (0,), 1: (1,)},
                )
                for t in transfers
            ]
            if fast_path:
                items = [
                    replace(
                        item,
                        prepared_io=adapter_module._PreparedPipelineTransfer(
                            store=native,
                            ids=np.frombuffer(b"".join(item.block_ids), dtype=np.uint8),
                            page_count=len(item.block_ids),
                            task_type=int,
                            distribute_backend_load=False,
                        ),
                    )
                    for item in items
                ]
            return {0: items, 1: items}

        adapter.prepare_load = prepare
        self.linker.adapter = adapter
        self.linker.num_layers = 2
        self.linker._page_transfers = lambda transfers, **kwargs: transfers
        self.linker._load_queue = queue.Queue()
        self.linker._requires_load_finish = False
        self.linker._completed_loads = queue.Queue()
        failures = []
        self.linker.layer_done_counter = types.SimpleNamespace(
            complete=lambda *args: None,
            fail=lambda batch, error: failures.append(error),
        )
        with self.assertLogs(trace_module.logger, level="INFO") as captured:
            self.enqueue("r1", "s1")
            self.enqueue("r2", "s2")
            ready = [types.SimpleNamespace(synchronize=lambda: None)]
            self.linker._load_queue.put((7, self.linker._pending_loads, ready))
            self.linker._load_queue.put(None)
            self.linker._load_worker()
        self.assertEqual(failures, [])
        records = self.records(captured)
        tasks = [r for r in records if r["event"] == "task_submitted"]
        self.assertEqual(len(tasks), 4)
        self.assertEqual(
            {(r["rid"], r["session_id"], r["layer"], r["shard"]) for r in tasks},
            {
                ("r1", "s1", 0, 0),
                ("r1", "s1", 1, 1),
                ("r2", "s2", 0, 0),
                ("r2", "s2", 1, 1),
            },
        )
        self.assertEqual(len({r["task_trace_id"] for r in tasks}), 4)
        self.assertEqual(len({r["load_attempt_id"] for r in tasks}), 2)
        self.assertEqual(len({tuple(r["block_ids"]) for r in tasks}), 1)
        for record, (ids, shards) in zip(tasks, submitted):
            self.assertEqual(record["block_ids"], [key.hex() for key in ids])
            self.assertEqual(record["page_keys"], ["shared-page"])
            self.assertEqual(record["load_batch_id"], 7)
        completed = [r for r in records if r["event"] == "task_completed"]
        self.assertEqual(
            {r["task_trace_id"] for r in tasks}, {r["task_trace_id"] for r in completed}
        )
        self.assertEqual(sum(r["event"] == "load_completed" for r in records), 2)

    def test_retry_cancel_empty_and_missing_session(self):
        with self.assertLogs(trace_module.logger, level="INFO") as captured:
            self.enqueue("r", "s")
            self.assertTrue(self.linker.cancel_queued_load("r"))
            self.enqueue("r", "s")
            self.assertTrue(self.linker.cancel_queued_load("r"))
            self.assertFalse(self.linker.cancel_queued_load("r"))
            self.assertFalse(self.linker.load("empty", []))
        records = self.records(captured)
        queued = [r for r in records if r["event"] == "load_queued"]
        self.assertNotEqual(queued[0]["load_attempt_id"], queued[1]["load_attempt_id"])
        self.assertEqual(sum(r["event"] == "load_cancelled" for r in records), 2)
        self.assertIsNone(records[-1]["session_id"])
        self.assertEqual(records[-1]["event"], "load_skipped")

    def test_disabled_does_not_capture_keys_or_change_pending_type(self):
        with (
            patch.dict(os.environ, {"UCM_SGLANG_LOAD_TRACE": "0"}),
            patch.object(trace_module.logger, "info") as log,
        ):
            self.enqueue("r", "s")
            self.assertIs(type(self.linker._pending_loads["r"]), list)
            log.assert_not_called()

    def test_wait_failure_retains_task_correlation_and_reaps_remaining(self):
        adapter = object.__new__(adapter_module.UcmDirectStoreAdapter)
        reaped = []

        def wait(task):
            reaped.append(task)
            if task == 1:
                raise RuntimeError("IO failed")

        store = types.SimpleNamespace(wait=wait)
        tasks = [
            adapter_module._StoreTask(store, i, None, {"task_trace_id": str(i)})
            for i in (1, 2)
        ]
        with self.assertLogs(trace_module.logger, level="INFO") as captured:
            with self.assertRaisesRegex(RuntimeError, "IO failed"):
                adapter.wait(tasks)
        self.assertEqual(reaped, [1, 2])
        self.assertEqual(
            [(r["event"], r["task_trace_id"]) for r in self.records(captured)],
            [("task_failed", "1"), ("task_completed", "2")],
        )

    def test_submit_failure_records_actual_keys(self):
        adapter = object.__new__(adapter_module.UcmDirectStoreAdapter)

        def fail(*args):
            raise RuntimeError("submit failed")

        store = types.SimpleNamespace(load_data=fail)
        transfer = self.transfer
        transfer = replace(transfer, trace={"rid": "r", "session_id": "s"})
        prepared = adapter_module._PreparedStoreTransfer(
            transfer=transfer,
            store=store,
            block_ids=[b"actual-key"],
            shards_by_layer={0: (0,)},
        )
        with self.assertLogs(trace_module.logger, level="INFO") as captured:
            with self.assertRaisesRegex(RuntimeError, "submit failed"):
                adapter._submit_prepared([prepared], is_offload=False, logical_layer=0)
        record = self.records(captured)[0]
        self.assertEqual(record["event"], "task_submit_failed")
        self.assertEqual(record["block_ids"], [b"actual-key".hex()])
        self.assertEqual(record["session_id"], "s")

    def test_logging_failure_does_not_affect_storage(self):
        with patch.object(
            trace_module.logger, "info", side_effect=RuntimeError("log failed")
        ):
            self.assertTrue(self.enqueue("r", "s"))
            self.assertTrue(self.linker.cancel_queued_load("r"))

    def test_wrapper_captures_top_level_session_and_resets_on_error(self):
        linker = self.linker

        class Wrapper:
            cache_linker = linker

            def match(self, key, req, result):
                return result

            def load_back(self, req):
                linker.load(req.rid, [self_transfer])
                raise RuntimeError("after queue")

        self_transfer = self.transfer
        modules = {
            "sglang.srt.mem_cache.unified_cache.unified_cache_linker": types.SimpleNamespace(
                UnifiedCacheLinkerWrapper=Wrapper
            ),
            "ucm.integration.sglang.ucm_direct_linker": linker_module,
        }
        with patch.dict(sys.modules, modules):
            trace_module.install_load_trace()
            original = Wrapper.load_back
            trace_module.install_load_trace()
            self.assertIs(original, Wrapper.load_back)
            with self.assertLogs(trace_module.logger, level="INFO"):
                with self.assertRaisesRegex(RuntimeError, "after queue"):
                    Wrapper().load_back(types.SimpleNamespace(rid="r", session_id="s"))
        self.assertIsNone(trace_module._request.get())
        self.assertEqual(linker._pending_loads["r"].trace["session_id"], "s")


if __name__ == "__main__":
    unittest.main()
