import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_dense_referee_training_samples import build, merge_intervals  # noqa: E402


class DenseRefereeTrainingSamplesTest(unittest.TestCase):
    def test_merge_intervals_joins_overlap(self):
        self.assertEqual([(1, 4), (6, 7)], merge_intervals([(1, 3), (2, 4), (6, 7)]))

    def test_build_excludes_uncertain_window_and_transition_margin(self):
        common = {"source": 2, "source_name": "s", "video": "v.mp4",
                  "roi_box_norm": [0.1, 0.2, 0.3, 0.4]}
        queue = {"items": [
            {**common, "id": "positive", "window_start_s": 5, "window_end_s": 15},
            {**common, "id": "uncertain", "window_start_s": 20, "window_end_s": 25},
        ]}
        results = {"answers": {
            "positive": {"decision": "serve_authorization",
                         "gesture_start_s": 9, "gesture_end_s": 10},
            "uncertain": {"decision": "uncertain"},
        }}
        dense_queue, dense_results = build(queue, results)
        by_time = {row["time_s"]: dense_results["answers"][row["id"]]["decision"]
                   for row in dense_queue["items"]}
        self.assertEqual("serve_authorization", by_time[8.0])
        self.assertEqual("no_signal", by_time[5.0])
        self.assertNotIn(7.0, by_time)
        self.assertFalse(any(20 <= time <= 25 for time in by_time))


if __name__ == "__main__":
    unittest.main()
