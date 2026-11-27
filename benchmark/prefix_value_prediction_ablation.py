"""Offline, delayed-feedback prequential evaluation of the production predictor.

Run with ordinary Python; no SGLang/CUDA import or serving process is needed.
Predictions are frozen after observing the current request, before any later
request arrives. Labels and SGD updates come from the unmodified production
observe/_train implementation. End-of-trace and capacity censoring are unscored.
This evaluates prefix recurrence probabilities, not retention scores or latency.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

from prefix_value_replay import make_trace

ROOT = Path(__file__).resolve().parents[1]
PREDICTOR_PATH = ROOT / "python/sglang/srt/mem_cache/reuse_value.py"
TRACE_PATH = ROOT / "benchmark/prefix_value_replay.py"
SPEC = importlib.util.spec_from_file_location(
    "prefix_value_prediction_production", PREDICTOR_PATH
)
PRODUCTION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PRODUCTION
SPEC.loader.exec_module(PRODUCTION)


class _ObservedRecords(OrderedDict):
    """Record production-created forecasts without copying the training loop."""

    def __init__(self, observer):
        super().__init__()
        self.observer = observer

    def __setitem__(self, key, record):
        super().__setitem__(key, record)
        self.observer._record_forecast(key, record)


class ScoredPredictor(PRODUCTION.OnlineReuseValue):
    def __init__(self, *, bias_only=False, drop_feature=None, **kwargs):
        if drop_feature not in (None, "frequency", "interval", "depth"):
            raise ValueError("Unknown feature ablation")
        if bias_only and drop_feature is not None:
            raise ValueError("Choose bias-only or a single feature removal")
        self.bias_only = bias_only
        self.drop_feature = drop_feature
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self.records = _ObservedRecords(self)
        self.forecasts = {}
        self.events = []
        self.predictions_issued = 0

    def _features(self, record, depth):
        features = super()._features(record, depth)
        if self.bias_only:
            return (features[0], 0.0, 0.0, 0.0)
        if self.drop_feature is not None:
            index = {"frequency": 1, "interval": 2, "depth": 3}[self.drop_feature]
            return tuple(0.0 if i == index else value for i, value in enumerate(features))
        return features

    def _record_forecast(self, key, record):
        # Even an occurrence censored within this request is counted. Its
        # probability is unset because it never reaches a prediction boundary.
        self.forecasts[id(record.features)] = {
            "prefix": key.hex() if isinstance(key, bytes) else str(key),
            "issued_tick": self.tick,
            # Retain the tuple itself: its identity joins delayed labels to
            # forecasts and must not be recycled after capacity censoring.
            "features": record.features,
            "probability": None,
        }
        self.predictions_issued += 1

    def _train(self, features, label):
        forecast = self.forecasts.pop(id(features))
        probability = forecast["probability"]
        if probability is None or forecast["issued_tick"] >= self.tick:
            raise AssertionError("A label must follow a frozen earlier prediction")
        forecast.update(
            status="resolved",
            label=label,
            resolved_tick=self.tick,
            brier=(probability - label) ** 2,
            logloss=-math.log(probability if label else 1.0 - probability),
            # Diagnostic only: production loss_sum uses this later probability.
            probability_at_label=self._probability(self.weights, features),
        )
        self.events.append(forecast)
        super()._train(features, label)

    def observe(self, prefixes):
        super().observe(prefixes)
        pending_ids = {id(record.features) for record in self.pending.values()}
        for feature_id in list(self.forecasts):
            if feature_id not in pending_ids:
                forecast = self.forecasts.pop(feature_id)
                forecast.update(
                    status="capacity_censored", label=None, resolved_tick=self.tick
                )
                self.events.append(forecast)
        # Serving can consult the predictor after the request's entire observe()
        # call. Earlier labels learned within this same request are legitimate
        # history. Do not refresh forecasts from previous requests with new weights.
        for record in self.pending.values():
            if record.last_seen == self.tick:
                forecast = self.forecasts[id(record.features)]
                forecast["probability"] = self._probability(
                    self.weights, record.features
                )
                forecast["weights_at_prediction"] = list(self.weights)

    def result(self):
        events = list(self.events)
        for forecast in self.forecasts.values():
            events.append(
                dict(
                    forecast,
                    status="end_of_trace_pending",
                    label=None,
                    resolved_tick=None,
                    observed_followup_requests=self.tick - forecast["issued_tick"],
                )
            )
        resolved = [event for event in events if event["status"] == "resolved"]
        positive = sum(event["label"] for event in resolved)
        negative = len(resolved) - positive
        capacity_censored = sum(
            event["status"] == "capacity_censored" for event in events
        )
        if (positive, negative, capacity_censored) != (
            self.positive_labels,
            self.negative_labels,
            self.censored_labels,
        ):
            raise AssertionError("Scored labels differ from the production learner")
        if len(events) != self.predictions_issued:
            raise AssertionError("Every issued prediction needs one final status")
        # Require enough real follow-up requests for even a negative label to
        # mature under production's strict age > horizon rule. This cohort is
        # auxiliary; it never changes the predeclared development selection rule.
        cutoff = self.tick - self.horizon - 1
        full_window = [event for event in events if event["issued_tick"] <= cutoff]
        full_window_resolved = [event for event in full_window if event["status"] == "resolved"]
        if any(event["status"] == "end_of_trace_pending" for event in full_window):
            raise AssertionError("A full-window forecast cannot remain pending")
        metrics = {
            "observed_requests": self.tick,
            "predictions_issued": self.predictions_issued,
            "resolved_labels": len(resolved),
            "positive_labels": positive,
            "negative_labels": negative,
            "capacity_censored": capacity_censored,
            "end_of_trace_pending": len(self.forecasts),
            "pending_at_exact_horizon": sum(
                self.tick - event["issued_tick"] == self.horizon
                for event in self.forecasts.values()
            ),
            "brier": sum(event["brier"] for event in resolved) / len(resolved)
            if resolved
            else None,
            "logloss": sum(event["logloss"] for event in resolved) / len(resolved)
            if resolved
            else None,
            "final_weights": list(self.weights),
            "full_window_issued_tick_max": cutoff,
            "full_window_predictions": len(full_window),
            "full_window_resolved_labels": len(full_window_resolved),
            "full_window_positive_labels": sum(event["label"] for event in full_window_resolved),
            "full_window_negative_labels": sum(1 - event["label"] for event in full_window_resolved),
            "full_window_capacity_censored": sum(event["status"] == "capacity_censored" for event in full_window),
            "full_window_brier": sum(event["brier"] for event in full_window_resolved) / len(full_window_resolved) if full_window_resolved else None,
            "full_window_logloss": sum(event["logloss"] for event in full_window_resolved) / len(full_window_resolved) if full_window_resolved else None,
        }
        return metrics, events


def evaluate_trace(trace, *, learning_rate=0.03, learn=True, bias_only=False,
                   drop_feature=None, horizon=64, capacity=4096):
    model = ScoredPredictor(
        learning_rate=learning_rate, learn=learn, bias_only=bias_only,
        drop_feature=drop_feature, horizon=horizon, capacity=capacity,
    )
    for item in trace:
        model.observe(
            PRODUCTION.prefix_observations(item["input_ids"], item.get("extra_key"))
        )
    return model.result()


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; existing paths are rejected")
    parser.add_argument("--count", type=int, default=240)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--dev-seeds", type=int, nargs="+", default=[1729, 1730, 1731])
    parser.add_argument("--heldout-seeds", type=int, nargs="+", default=[1732, 1733, 1734])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    parser.add_argument("--bias-only-lr", type=float, default=0.03)
    parser.add_argument("--prior-heldout-results", type=Path,
                        help="Disclose an earlier result file that already evaluated these held-out seeds")
    args = parser.parse_args(argv)
    if min(args.count, args.horizon, args.capacity) < 1:
        parser.error("count, horizon and capacity must be positive")
    rates = args.learning_rates + [args.bias_only_lr]
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        parser.error("learning rates must be finite and positive")
    all_seeds = args.dev_seeds + args.heldout_seeds
    if len(set(all_seeds)) != len(all_seeds):
        parser.error("development and held-out seeds must be distinct")
    if len(set(args.learning_rates)) != len(args.learning_rates):
        parser.error("learning rates must be distinct")
    if args.prior_heldout_results is not None and not args.prior_heldout_results.is_file():
        parser.error("prior-heldout-results must be an existing file")
    variants = [{"variant": "frozen", "learn": False, "learning_rate": 0.03}]
    variants += [
        {"variant": f"learned_lr_{rate:g}", "learn": True, "learning_rate": rate}
        for rate in args.learning_rates
    ]
    variants.append({"variant": f"bias_only_lr_{args.bias_only_lr:g}", "learn": True,
                     "learning_rate": args.bias_only_lr, "bias_only": True})
    variants += [
        {"variant": f"drop_{feature}_lr_0.03", "learn": True,
         "learning_rate": 0.03, "drop_feature": feature}
        for feature in ("frequency", "interval", "depth")
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "validation": "Offline prefix recurrence prediction only; no GPU or cache policy replay",
        "prediction_boundary": "After the entire current request observe(), before the next request",
        "label_source": "Unmodified production observe() and _train() callbacks",
        "censoring": "Capacity-censored and pending forecasts are unscored; no artificial drain requests. Production resolves negatives only at age > horizon, including leaving exact-horizon pending at trace end.",
        "cold_start": "Fresh zero weights, history and request clock for every seed and variant",
        "holdout_protocol": "Variants fixed before this run; no training state crosses seeds. Select a full-feature learning rate by development mean per-seed Brier, then evaluate all variants on held-out seeds without reselection.",
        "feature_ablation": "At lr=0.03, set exactly one of F/I/D predictor inputs to zero in both production training and probability inference. Does not change the external retention cost or recency-decay formula.",
        "full_window_cohort": "Auxiliary metrics use issued_tick <= observed_requests - horizon - 1, ensuring even negative labels can mature. Capacity censoring remains unscored. Does not change learning-rate selection.",
        "prior_holdout_exposure": {
            "disclosed_result": str(args.prior_heldout_results) if args.prior_heldout_results else None,
            "sha256": _sha256(args.prior_heldout_results.read_bytes()) if args.prior_heldout_results else None,
            "note": "These held-out seeds were already evaluated in the disclosed earlier pass; they are not a never-computed fresh holdout. This run selects from development scores only." if args.prior_heldout_results else "No prior result disclosed. Execution order within this run is not proof that these seeds were never evaluated previously.",
        },
        "count": args.count, "horizon": args.horizon, "capacity": args.capacity,
        "dev_seeds": args.dev_seeds, "heldout_seeds": args.heldout_seeds,
        "variants": variants,
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path.read_bytes())
            for path in (PREDICTOR_PATH, TRACE_PATH, Path(__file__).resolve())
        },
        "traces": [], "runs": [],
    }
    with (args.output / "prediction-events.jsonl").open("w", encoding="utf-8") as log:
        for split, seeds in (("development", args.dev_seeds), ("heldout", args.heldout_seeds)):
            for seed in seeds:
                trace = make_trace(seed, args.count)
                encoded = json.dumps(trace, separators=(",", ":")).encode()
                trace_path = args.output / f"trace-{seed}.json"
                trace_path.write_bytes(encoded)
                trace_hash = _sha256(encoded)
                summary["traces"].append({"seed": seed, "split": split,
                                          "path": trace_path.name, "sha256": trace_hash})
                for variant in variants:
                    config = {key: value for key, value in variant.items() if key != "variant"}
                    metrics, events = evaluate_trace(
                        trace, horizon=args.horizon, capacity=args.capacity, **config
                    )
                    identity = {"seed": seed, "split": split, "variant": variant["variant"]}
                    summary["runs"].append(dict(identity, trace_sha256=trace_hash, **metrics))
                    for event in events:
                        log.write(json.dumps(dict(identity, **event)) + "\n")
                    print(f"{split} seed={seed} {variant['variant']}: "
                          f"Brier={metrics['brier']} logloss={metrics['logloss']} "
                          f"positive={metrics['positive_labels']} negative={metrics['negative_labels']} "
                          f"pending={metrics['end_of_trace_pending']} censored={metrics['capacity_censored']}")
            if split == "development":
                ranked = []
                for rate in args.learning_rates:
                    name = f"learned_lr_{rate:g}"
                    candidates = [run for run in summary["runs"]
                                  if run["split"] == "development" and run["variant"] == name]
                    if any(run["brier"] is None for run in candidates):
                        raise ValueError("Each development seed needs resolved labels for selection")
                    ranked.append({"variant": name, "learning_rate": rate,
                                   "mean_per_seed_brier": sum(run["brier"] for run in candidates) / len(candidates)})
                ranked.sort(key=lambda row: (row["mean_per_seed_brier"], row["learning_rate"]))
                summary["selection"] = {
                    "rule": "Minimum arithmetic mean per-seed development Brier among full-feature learning rates; tie goes to lower learning rate",
                    "development_seeds": args.dev_seeds,
                    "ranked_candidates": ranked,
                    "selected_variant": ranked[0]["variant"],
                    "selected_learning_rate": ranked[0]["learning_rate"],
                    "heldout_evaluated_in_this_run_at_selection": False,
                    "prior_holdout_exposure": summary["prior_holdout_exposure"],
                }
                log.flush()
                (args.output / "selection.json").write_text(
                    json.dumps(summary["selection"], indent=2), encoding="utf-8"
                )
    summary["aggregates"] = []
    for split in ("development", "heldout"):
        for variant in variants:
            runs = [run for run in summary["runs"]
                    if run["split"] == split and run["variant"] == variant["variant"]]
            resolved = sum(run["resolved_labels"] for run in runs)
            full_window_resolved = sum(run["full_window_resolved_labels"] for run in runs)
            summary["aggregates"].append({
                "split": split, "variant": variant["variant"], "seeds": len(runs),
                "resolved_labels": resolved,
                "pooled_brier": sum((run["brier"] or 0) * run["resolved_labels"] for run in runs) / resolved if resolved else None,
                "pooled_logloss": sum((run["logloss"] or 0) * run["resolved_labels"] for run in runs) / resolved if resolved else None,
                "mean_per_seed_brier": sum(run["brier"] for run in runs) / len(runs) if all(run["brier"] is not None for run in runs) else None,
                "mean_per_seed_logloss": sum(run["logloss"] for run in runs) / len(runs) if all(run["logloss"] is not None for run in runs) else None,
                "full_window_resolved_labels": full_window_resolved,
                "full_window_pooled_brier": sum((run["full_window_brier"] or 0) * run["full_window_resolved_labels"] for run in runs) / full_window_resolved if full_window_resolved else None,
                "full_window_pooled_logloss": sum((run["full_window_logloss"] or 0) * run["full_window_resolved_labels"] for run in runs) / full_window_resolved if full_window_resolved else None,
                "full_window_mean_per_seed_brier": sum(run["full_window_brier"] for run in runs) / len(runs) if all(run["full_window_brier"] is not None for run in runs) else None,
                "full_window_mean_per_seed_logloss": sum(run["full_window_logloss"] for run in runs) / len(runs) if all(run["full_window_logloss"] is not None for run in runs) else None,
            })
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = [{key: value for key, value in run.items() if key != "final_weights"}
            for run in summary["runs"]]
    with (args.output / "per-seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


if __name__ == "__main__":
    main()
