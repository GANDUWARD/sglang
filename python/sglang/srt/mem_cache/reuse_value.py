"""Experimental bounded online reuse predictor, independent of KV residency.

Labels mean reuse within the next ``horizon`` observations, not GPU time saved.
Capacity-censored observations are discarded rather than mislabeled as misses.
The runtime must call observe once per logical request, including cache misses.
"""

import hashlib
import math
import struct
from collections import OrderedDict
from dataclasses import dataclass


def prefix_observations(token_ids, namespace=None, quantum=64):
    """Stable fixed-token boundaries survive Radix node splits and KV eviction."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(repr(namespace).encode("utf-8"))
    result = {}
    for i, token in enumerate(token_ids, 1):
        digest.update(struct.pack("<q", token))
        if i % quantum == 0:
            result[digest.digest()] = i
    return result


@dataclass
class ReuseRecord:
    last_seen: int
    frequency: float
    features: tuple
    interval: int


class OnlineReuseValue:
    def __init__(self, horizon=64, capacity=4096, learning_rate=0.03, learn=True):
        if horizon < 1 or capacity < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("horizon, capacity and learning_rate must be positive")
        self.horizon = horizon
        self.capacity = capacity
        self.learning_rate = learning_rate
        self.learn = learn
        self.reset()

    def reset(self):
        self.tick = 0
        self.records = OrderedDict()
        self.pending = OrderedDict()
        self.weights = [0.0] * 4
        self.positive_labels = 0
        self.negative_labels = 0
        self.censored_labels = 0
        self.loss_sum = 0.0

    @staticmethod
    def _probability(weights, features):
        logit = max(-20.0, min(20.0, sum(w * x for w, x in zip(weights, features))))
        return 1.0 / (1.0 + math.exp(-logit))

    def _features(self, record, depth):
        interval = self.horizon if record is None else record.interval
        frequency = 0.0 if record is None else record.frequency
        return (
            1.0,
            min(math.log1p(frequency) / 8.0, 1.0),
            min(interval / self.horizon, 1.0),
            min(math.log1p(depth) / 12.0, 1.0),
        )

    def _train(self, features, label):
        probability = self._probability(self.weights, features)
        self.positive_labels += label
        self.negative_labels += 1 - label
        self.loss_sum -= math.log(probability if label else 1.0 - probability)
        if self.learn:
            # SGD on logistic cross entropy; no tensor runtime on scheduler path.
            for i, feature in enumerate(features):
                gradient = (probability - label) * feature + 1e-4 * self.weights[i]
                self.weights[i] -= self.learning_rate * gradient

    def observe(self, prefixes):
        """Observe {stable_namespace_and_prefix_hash: prefix_token_depth}.

        Reuse at exactly the horizon is positive. Predictions are made after
        observing an occurrence and labeled only by subsequent observations.
        """
        self.tick += 1
        while self.pending:
            key = next(iter(self.pending))
            record = self.pending[key]
            if self.tick - record.last_seen <= self.horizon:
                break
            self.pending.pop(key)
            self._train(record.features, 0)

        for key, depth in prefixes.items():
            if depth <= 0:
                raise ValueError("prefix depth must be positive")
            previous = self.records.pop(key, None)
            pending = self.pending.pop(key, None)
            if pending is not None:
                self._train(pending.features, 1)
            frequency = 1.0
            if previous is not None:
                age = self.tick - previous.last_seen
                frequency += previous.frequency * math.exp(-age / self.horizon)
            interval = (
                self.horizon if previous is None else self.tick - previous.last_seen
            )
            record = ReuseRecord(self.tick, frequency, (), interval)
            record.features = self._features(record, depth)
            self.records[key] = record
            self.pending[key] = record
            while len(self.records) > self.capacity:
                removed, _ = self.records.popitem(last=False)
                if self.pending.pop(removed, None) is not None:
                    self.censored_labels += 1

    def retention_value(
        self, key, depth, recompute_cost, resource_size, *, apply_recency_decay=True
    ):
        """Expected cost per resource unit; caller owns the cost model/units.

        Finite scores change eviction order only, never pin storage. Additional
        recency decay prevents old popularity from retaining a permanent bonus.
        Disabling that external decay does not alter predictor features or SGD.
        """
        if resource_size <= 0 or recompute_cost < 0:
            raise ValueError("invalid cost or resource size")
        record = self.records.get(key)
        probability = self._probability(self.weights, self._features(record, depth))
        age = self.horizon if record is None else self.tick - record.last_seen
        decay = math.exp(-age / self.horizon) if apply_recency_decay else 1.0
        return probability * decay * recompute_cost / resource_size
