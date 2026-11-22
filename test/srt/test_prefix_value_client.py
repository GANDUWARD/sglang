"""Client protocol tests with synthetic HTTP streams, not model experiments."""

import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
from prefix_value_replay import make_trace, replay_one


class TestPrefixValueClient(unittest.TestCase):
    def run_stream(self, reason="length", done=True, count=2):
        event = {
            "text": "test",
            "meta_info": {
                "completion_tokens": count,
                "finish_reason": {"type": reason},
            },
        }
        payload = b"data: " + json.dumps(event).encode() + b"\n\n"
        if done:
            payload += b"data: [DONE]\n\n"
        with patch("urllib.request.urlopen", return_value=io.BytesIO(payload)):
            return replay_one(
                "http://example.invalid", {"id": 0, "group": 1, "input_ids": [1]}, 2
            )

    def test_complete_stream(self):
        result = self.run_stream()
        self.assertEqual(result["meta_info"]["completion_tokens"], 2)
        self.assertGreaterEqual(result["latency_s"], result["ttft_s"])

    def test_truncated_stream_is_rejected_even_with_expected_token_count(self):
        with self.assertRaises(RuntimeError):
            self.run_stream(done=False)

    def test_abort_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.run_stream(reason="abort")

    def test_short_completion_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.run_stream(count=1)

    def test_trace_is_reproducible(self):
        self.assertEqual(make_trace(1729, 100), make_trace(1729, 100))
        self.assertNotEqual(make_trace(1729, 100), make_trace(1730, 100))


if __name__ == "__main__":
    unittest.main()
