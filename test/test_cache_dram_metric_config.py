"""Ensure DRAM metric names and types agree across default and example configs."""

import importlib.util
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "cache_metric_defaults", ROOT / "ucm/default_metrics_config.py"
)
defaults = importlib.util.module_from_spec(spec)
spec.loader.exec_module(defaults)


class CacheDramMetricConfigTest(unittest.TestCase):
    def test_metric_names_types_and_example_agree(self):
        expected = {
            "cache_dram_hit_shards_total": "counter",
            "cache_dram_load_shards_total": "counter",
            "cache_dram_evicted_bytes_total": "counter",
            "cache_dram_tail_metadata_gc_total": "counter",
            "cache_dram_capacity_bytes": "gauge",
            "cache_dram_used_bytes": "gauge",
            "cache_dram_tail_metadata_capacity_bytes": "gauge",
            "cache_dram_tail_metadata_used_bytes": "gauge",
        }
        config = defaults.get_default_metrics_config()
        example = (ROOT / "examples/metrics/metrics_configs.yaml").read_text(
            encoding="utf-8"
        )
        for name, kind in expected.items():
            with self.subTest(name=name):
                matches = [x for x in config[kind] if x["name"] == name]
                self.assertEqual(len(matches), 1)
                self.assertEqual(
                    len(re.findall(r'^\s*- name: "' + re.escape(name) + r'"$', example, re.M)),
                    1,
                )
                section = re.search(r"(?ms)^" + kind + r":\s*\n(.*?)(?=^[a-z_]+:|\Z)", example)
                self.assertIsNotNone(section)
                self.assertIn(f'- name: "{name}"', section.group(1))
                if kind == "gauge":
                    self.assertEqual(matches[0]["multiprocess_mode"], "livesum")

    def test_grafana_contains_tail_metadata_panels(self):
        dashboard = json.loads(
            (ROOT / "examples/metrics/grafana_store.json").read_text(encoding="utf-8")
        )
        expressions = "\n".join(
            target.get("expr", "")
            for panel in dashboard["panels"]
            for target in panel.get("targets", [])
        )
        for name in (
            "cache_dram_tail_metadata_used_bytes",
            "cache_dram_tail_metadata_capacity_bytes",
            "cache_dram_tail_metadata_gc_total",
        ):
            self.assertIn("ucm:" + name, expressions)


if __name__ == "__main__":
    unittest.main()
