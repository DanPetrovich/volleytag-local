import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_long_context_rally_state import (  # noqa: E402
    decode,
    safe_matches,
    window_summary,
)
from evaluate_long_context_boundaries import pair_intervals  # noqa: E402


class LongContextRallyStateTest(unittest.TestCase):
    def test_window_summary_keeps_shape_and_uses_neighbors(self):
        values = np.asarray([[0.0], [2.0], [4.0]])
        result = window_summary(values, (1,))
        self.assertEqual(result.shape, (3, 3))
        self.assertAlmostEqual(result[1, 1], 2.0)
        self.assertGreater(result[1, 2], 1.0)

    def test_decode_bridges_single_gap_and_adds_safe_padding(self):
        times = np.arange(0.0, 7.0)
        scores = np.asarray([0.0, 0.8, 0.9, 0.1, 0.9, 0.8, 0.0])
        rows = decode(times, scores, 1.0, threshold=0.5, smoothing_radius=0,
                      bridge_s=1.0, minimum_active_s=2.0, padding_s=2.0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["start_s"], -1.5)
        self.assertEqual(rows[0]["end_s"], 7.5)

    def test_safe_match_requires_complete_containment(self):
        truth = [{"id": "r1", "start_s": 10.0, "end_s": 20.0}]
        good = [{"start_s": 8.0, "end_s": 22.0}]
        clipped = [{"start_s": 11.0, "end_s": 22.0}]
        self.assertEqual(len(safe_matches(truth, good, 3.0)), 1)
        self.assertEqual(safe_matches(truth, clipped, 3.0), [])

    def test_boundary_pairing_uses_latest_start_and_padding(self):
        starts = [{"time_s": 8.0, "score": 0.6}, {"time_s": 10.0, "score": 0.9}]
        ends = [{"time_s": 20.0, "score": 0.8}]
        rows = pair_intervals(starts, ends, 2.0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["start_s"], 8.0)
        self.assertEqual(rows[0]["end_s"], 22.0)


if __name__ == "__main__":
    unittest.main()
