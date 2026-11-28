"""CPU-only tests of the real connector dump lifecycle methods."""
import ast
import math
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]


def load_methods(path, class_name, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    methods = [
        method
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == class_name
        for method in cls.body
        if isinstance(method, ast.FunctionDef) and method.name in names
    ]
    assert {method.name for method in methods} == names
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            ast.ClassDef(name="Proxy", bases=[], keywords=[], body=methods, decorator_list=[]),
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["Proxy"]


class StoreUnhealthyError(Exception):
    pass


namespace = {
    "time": time,
    "math": math,
    "logger": SimpleNamespace(error=Mock(), error_limit=Mock()),
    "ucmmetrics": SimpleNamespace(update_stats=Mock()),
    "StoreUnhealthyError": StoreUnhealthyError,
}
Connector = load_methods(
    "ucm/integration/vllm/ucm_connector.py",
    "UCMDirectConnector",
    {
        "_wait_pending_dump_task", "_release_dump_event_handle",
        "_non_negative_ms",
        "_remember_tail_dump_failure", "_raise_tail_dump_failure",
        "_poll_pending_dump_tasks", "_flush_pending_dump_tasks", "get_finished",
    },
    namespace,
)
Manager = load_methods(
    "ucm/integration/vllm/rank_consistency.py",
    "RankConsistencyManager",
    {"wait_dump"},
    namespace,
)


class TailDumpFailureTest(unittest.TestCase):
    def make_connector(self, enabled=True):
        connector = Connector()
        connector._tail_relations_enabled = enabled
        connector._tail_dump_failure = None
        connector._pending_dump_tasks = []
        connector._async_dump_req_ids = set()
        connector._poll_pending_load_tasks = Mock(return_value=set())
        connector.enable_event_sync = True
        connector.device = SimpleNamespace(destroy_event_handle=Mock())
        connector.store = SimpleNamespace(check=Mock(return_value=True))
        connector._rank_consistency = SimpleNamespace(wait_dump=Mock(), finish_dump=Mock())
        return connector

    def task(self, name, request, event):
        return SimpleNamespace(task=name, request_ids={request}, event_handle=event,
                               wait_for_save_start_ms=0)

    def test_completed_wait_failure_is_sticky_and_cleans_event(self):
        c = self.make_connector()
        c._pending_dump_tasks = [self.task("failed", "r", 7)]
        c._async_dump_req_ids = {"r"}
        c._rank_consistency.wait_dump.side_effect = RuntimeError("disk error")
        with self.assertRaisesRegex(RuntimeError, "dump wait failed: disk error"):
            c._poll_pending_dump_tasks()
        self.assertEqual(c._pending_dump_tasks, [])
        c.device.destroy_event_handle.assert_called_once_with(7)
        c._rank_consistency.wait_dump.assert_called_once_with("failed", propagate_unhealthy=True)
        with self.assertRaisesRegex(RuntimeError, "disk error"):
            c.get_finished({"r"})
        c._rank_consistency.finish_dump.assert_not_called()

    def test_check_failure_drains_task_and_other_completed_tasks(self):
        c = self.make_connector()
        c._pending_dump_tasks = [self.task("check_error", "a", 11),
                                 self.task("ready", "b", 12),
                                 self.task("running", "c", 13)]
        def check(task):
            if task == "check_error":
                raise RuntimeError("check error")
            return task == "ready"
        c.store.check.side_effect = check
        with self.assertRaisesRegex(RuntimeError, "dump check failed: check error"):
            c._poll_pending_dump_tasks()
        self.assertEqual([x.task for x in c._pending_dump_tasks], ["running"])
        self.assertEqual(c._rank_consistency.wait_dump.call_count, 2)
        self.assertEqual(c.device.destroy_event_handle.call_count, 2)

    def test_get_finished_drains_all_matching_tasks_without_reporting_success(self):
        c = self.make_connector()
        c._async_dump_req_ids = {"r"}
        c._pending_dump_tasks = [self.task("bad", "r", 1),
                                 self.task("good", "r", 2),
                                 self.task("other", "s", 3)]
        def wait(task, **kwargs):
            if task == "bad":
                raise RuntimeError("device failure")
        c._rank_consistency.wait_dump.side_effect = wait
        with self.assertRaisesRegex(RuntimeError, "device failure"):
            c.get_finished({"r"})
        self.assertEqual([x.task for x in c._pending_dump_tasks], ["other"])
        self.assertEqual(c.device.destroy_event_handle.call_count, 2)
        self.assertEqual(c._async_dump_req_ids, {"r"})
        c._rank_consistency.finish_dump.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "device failure"):
            c.get_finished({"r"})

    def test_disabled_strategy_preserves_existing_best_effort_behavior(self):
        c = self.make_connector(enabled=False)
        c._async_dump_req_ids = {"r"}
        c._pending_dump_tasks = [self.task("bad", "r", 1)]
        c._rank_consistency.wait_dump.side_effect = RuntimeError("device failure")
        self.assertEqual(c.get_finished({"r"}), ({"r"}, None))
        c._rank_consistency.finish_dump.assert_called_once_with({"r"})
        c._rank_consistency.wait_dump.assert_called_once_with("bad")
        c.device.destroy_event_handle.assert_called_once_with(1)

    def test_unhealthy_dump_propagates_only_in_strict_mode(self):
        manager = Manager()
        manager._record_dump_failure = Mock()
        task = object()
        store = SimpleNamespace(wait=Mock(side_effect=StoreUnhealthyError("unhealthy")))
        manager._dump_task_contexts = {id(task): (store, {"r": {b"block"}})}
        with self.assertRaises(StoreUnhealthyError):
            manager.wait_dump(task, propagate_unhealthy=True)
        self.assertEqual(manager._dump_task_contexts, {})
        manager._dump_task_contexts = {id(task): (store, {"r": {b"block"}})}
        manager.wait_dump(task)
        self.assertEqual(manager._record_dump_failure.call_count, 2)


if __name__ == "__main__":
    unittest.main()
