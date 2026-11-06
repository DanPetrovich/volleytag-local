import sys
import unittest
from pathlib import Path

try:
    import joblib  # noqa: F401
    import numpy  # noqa: F401
    import sklearn  # noqa: F401
except ImportError as error:  # pragma: no cover - optional vision environment
    raise unittest.SkipTest(str(error))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_rally_boundary_fusion import (  # noqa: E402
    directional_event_matches,
    safe_interval_matches,
)


class RallyBoundaryEvaluationTest(unittest.TestCase):
    def test_directional_events_reject_clipping(self):
        candidates = [{"time_s": 8.0}, {"time_s": 10.5}]
        self.assertEqual(8.0, directional_event_matches([10.0], candidates, "start", 3)[0]["candidate_s"])
        self.assertEqual(10.5, directional_event_matches([10.0], candidates, "end", 3)[0]["candidate_s"])

    def test_safe_interval_must_contain_truth(self):
        truth = [{"id": "r1", "start_s": 10.0, "end_s": 20.0}]
        predicted = [
            {"start_s": 9.0, "end_s": 19.9},
            {"start_s": 9.0, "end_s": 22.0},
        ]
        matches = safe_interval_matches(truth, predicted, 3.0)
        self.assertEqual([{"gt_id": "r1", "prediction_index": 1,
                           "start_lead_s": 1.0, "end_lag_s": 2.0}], matches)

    def test_safe_interval_rejects_excess_padding(self):
        truth = [{"id": "r1", "start_s": 10.0, "end_s": 20.0}]
        predicted = [{"start_s": 6.9, "end_s": 20.0}]
        self.assertEqual([], safe_interval_matches(truth, predicted, 3.0))


if __name__ == "__main__":
    unittest.main()
