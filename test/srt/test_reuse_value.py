"""Dependency-free tests of delayed feedback and bounded learner state."""

import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "python/sglang/srt/mem_cache/reuse_value.py"
)
SPEC = importlib.util.spec_from_file_location("reuse_value_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
OnlineReuseValue = MODULE.OnlineReuseValue


class TestOnlineReuseValue(unittest.TestCase):
    def test_prefix_identity_shared_and_namespace_isolated(self):
        prefix = list(range(64))
        first = MODULE.prefix_observations(prefix + [100] * 64, "a")
        second = MODULE.prefix_observations(prefix + [200] * 64, "a")
        self.assertEqual(len(set(first) & set(second)), 1)
        isolated = MODULE.prefix_observations(prefix, "b")
        self.assertFalse(set(first) & set(isolated))

    def test_delayed_labels_and_horizon_boundary(self):
        model = OnlineReuseValue(horizon=2)
        model.observe({"a": 32})
        self.assertEqual(model.positive_labels + model.negative_labels, 0)
        model.observe({})
        model.observe({"a": 32})
        self.assertEqual(model.positive_labels, 1)
        for _ in range(3):
            model.observe({})
        self.assertEqual(model.negative_labels, 1)

    def test_bounded_history_censors_without_false_negative(self):
        model = OnlineReuseValue(capacity=2, horizon=10)
        for i in range(20):
            model.observe({str(i): 32})
        self.assertEqual(len(model.records), 2)
        self.assertEqual(len(model.pending), 2)
        self.assertEqual(model.censored_labels, 18)
        self.assertEqual(model.negative_labels, 0)

    def test_predictions_learn_and_decay_when_hotspot_stops(self):
        model = OnlineReuseValue(horizon=4)
        model.observe({"hot": 32})
        before = model.retention_value("hot", 32, 32, 32)
        for _ in range(200):
            model.observe({"hot": 32})
        trained = model.retention_value("hot", 32, 32, 32)
        self.assertGreater(trained, before)
        for _ in range(20):
            model.observe({})
        self.assertLess(model.retention_value("hot", 32, 32, 32), trained / 10)

    def test_frozen_ablation_and_reset(self):
        model = OnlineReuseValue(learn=False)
        for _ in range(10):
            model.observe({"a": 64})
        self.assertEqual(model.weights, [0.0] * 4)
        self.assertEqual(model.positive_labels, 9)
        model.reset()
        self.assertEqual(len(model.records), 0)
        self.assertEqual(model.tick, 0)

    def test_reuse_interval_is_available_before_future_label(self):
        model = OnlineReuseValue(horizon=8)
        model.observe({"a": 64})
        model.observe({})
        model.observe({"a": 64})
        self.assertEqual(model.records["a"].interval, 2)
        self.assertEqual(model.records["a"].features[2], 0.25)
        self.assertNotEqual(model.weights[2], 0.0)


if __name__ == "__main__":
    unittest.main()
