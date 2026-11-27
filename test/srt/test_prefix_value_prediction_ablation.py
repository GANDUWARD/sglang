"""No GPU dependencies: verify delayed-feedback scoring and production parity."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
from prefix_value_prediction_ablation import PRODUCTION, ScoredPredictor


class TestPredictionAblation(unittest.TestCase):
    def test_score_uses_issued_probability_not_label_time_probability(self):
        model = ScoredPredictor(horizon=8)
        model.observe({"a": 64})
        model.weights[0] = 2.0  # Simulate intervening learning before a delayed label.
        model.observe({"a": 64})
        metrics, events = model.result()
        event = next(event for event in events if event["status"] == "resolved")
        self.assertEqual(event["probability"], 0.5)
        self.assertGreater(event["probability_at_label"], 0.8)
        self.assertAlmostEqual(metrics["brier"], 0.25)
        self.assertAlmostEqual(metrics["logloss"], math.log(2.0))

    def test_predictions_use_full_request_boundary_then_remain_frozen(self):
        model = ScoredPredictor(horizon=8)
        model.observe({"a": 64, "b": 128})
        model.observe({"a": 64, "b": 128})
        for record in model.pending.values():
            forecast = model.forecasts[id(record.features)]
            self.assertEqual(forecast["weights_at_prediction"], model.weights)
            self.assertEqual(forecast["probability"], model._probability(model.weights, record.features))
        before = {key: dict(value) for key, value in model.forecasts.items()}
        model.observe({"c": 64})
        for key, value in before.items():
            self.assertEqual(model.forecasts[key], value)

    def test_horizon_boundary_and_no_end_of_trace_negative_fill(self):
        model = ScoredPredictor(horizon=2)
        model.observe({"a": 64})
        model.observe({})
        model.observe({})
        metrics, _ = model.result()
        self.assertEqual(metrics["resolved_labels"], 0)
        self.assertEqual(metrics["end_of_trace_pending"], 1)
        self.assertEqual(metrics["pending_at_exact_horizon"], 1)
        model.observe({})
        metrics, _ = model.result()
        self.assertEqual(metrics["negative_labels"], 1)
        self.assertEqual(metrics["end_of_trace_pending"], 0)
        positive = ScoredPredictor(horizon=2)
        positive.observe({"a": 64})
        positive.observe({})
        positive.observe({"a": 64})
        self.assertEqual(positive.result()[0]["positive_labels"], 1)

    def test_capacity_censoring_including_within_request_is_unscored(self):
        model = ScoredPredictor(capacity=1, horizon=8)
        model.observe({"a": 64, "b": 128})
        model.observe({"a": 64})
        metrics, events = model.result()
        self.assertEqual(metrics["predictions_issued"], 3)
        self.assertEqual(metrics["capacity_censored"], 2)
        self.assertEqual(metrics["end_of_trace_pending"], 1)
        self.assertEqual(metrics["resolved_labels"], 0)
        self.assertIsNone(metrics["brier"])
        self.assertTrue(all(event["label"] is None for event in events))

    def test_instrumentation_preserves_production_state_and_updates(self):
        observations = [{"a": 64, "b": 128}, {"a": 64}, {"c": 64}, {},
                        {"c": 64, "d": 128}, {"d": 128}, {}, {}, {}, {}]
        for learn in (False, True):
            for rate in (0.01, 0.03, 0.1):
                with self.subTest(learn=learn, rate=rate):
                    kwargs = dict(learn=learn, learning_rate=rate, horizon=2, capacity=2)
                    production = PRODUCTION.OnlineReuseValue(**kwargs)
                    scored = ScoredPredictor(**kwargs)
                    for observation in observations:
                        production.observe(observation)
                        scored.observe(observation)
                        self.assertEqual(production.weights, scored.weights)
                        self.assertEqual(production.records, scored.records)
                        self.assertEqual(production.pending, scored.pending)
                        self.assertEqual(production.loss_sum, scored.loss_sum)
                    scored.result()  # Also asserts production/scored label counts.

    def test_frozen_baseline_and_reset(self):
        model = ScoredPredictor(learn=False, horizon=1)
        for observation in ({"a": 64}, {"a": 64}, {}, {}):
            model.observe(observation)
        metrics, _ = model.result()
        self.assertEqual(metrics["positive_labels"], 1)
        self.assertEqual(metrics["negative_labels"], 1)
        self.assertEqual(metrics["brier"], 0.25)
        self.assertAlmostEqual(metrics["logloss"], math.log(2.0))
        model.reset()
        self.assertEqual(model.result()[0]["predictions_issued"], 0)

    def test_single_feature_removal_applies_to_training_and_prediction(self):
        for feature, index in (("frequency", 1), ("interval", 2), ("depth", 3)):
            with self.subTest(feature=feature):
                model = ScoredPredictor(drop_feature=feature)
                for _ in range(5):
                    model.observe({"a": 64})
                original = PRODUCTION.OnlineReuseValue._features(
                    model, model.records["a"], 64
                )
                masked = model.records["a"].features
                self.assertEqual(masked[index], 0.0)
                for i in range(4):
                    if i != index:
                        self.assertEqual(masked[i], original[i])
                self.assertEqual(model.weights[index], 0.0)
                self.assertNotEqual(model.weights[0], 0.0)
                metrics, events = model.result()
                self.assertEqual(metrics["positive_labels"], 4)
                self.assertTrue(all(event["features"][index] == 0.0 for event in events))

    def test_full_window_cohort_excludes_late_early_positive_labels(self):
        model = ScoredPredictor(horizon=2, learn=False)
        for observation in ({"a": 64}, {}, {}, {"b": 64}, {"b": 64}):
            model.observe(observation)
        metrics, _ = model.result()
        self.assertEqual(metrics["positive_labels"], 1)
        self.assertEqual(metrics["negative_labels"], 1)
        self.assertEqual(metrics["full_window_issued_tick_max"], 2)
        self.assertEqual(metrics["full_window_predictions"], 1)
        self.assertEqual(metrics["full_window_resolved_labels"], 1)
        self.assertEqual(metrics["full_window_positive_labels"], 0)
        self.assertEqual(metrics["full_window_negative_labels"], 1)
        self.assertEqual(metrics["full_window_brier"], 0.25)


if __name__ == "__main__":
    unittest.main()
