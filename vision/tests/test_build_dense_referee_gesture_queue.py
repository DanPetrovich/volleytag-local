import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_dense_referee_gesture_queue import clamp_to_replays, make_item  # noqa: E402


class DenseRefereeGestureQueueTest(unittest.TestCase):
    def setUp(self):
        self.source = {"name": "training", "source": 2, "video": "source.mp4",
                       "roi_box_norm": [0.1, 0.2, 0.3, 0.4]}
        self.rally = {"id": "r1", "start_s": 20.0, "end_s": 30.0}

    def test_serve_window_searches_before_contact(self):
        row = make_item(self.source, self.rally, "serve_search", 1, [])
        self.assertEqual((12.0, 17.0, 21.0),
                         (row["window_start_s"], row["time_s"], row["window_end_s"]))
        self.assertTrue(row["requires_exact_time"])

    def test_replay_boundary_clips_context(self):
        self.assertEqual((12.0, 18.0), clamp_to_replays(
            12.0, 21.0, 17.0, [{"start_s": 18.0, "end_s": 20.0}]))

    def test_target_inside_replay_is_rejected(self):
        self.assertIsNone(clamp_to_replays(
            12.0, 21.0, 17.0, [{"start_s": 16.0, "end_s": 18.0}]))

    def test_short_rally_is_not_used_as_active_negative(self):
        short = {"id": "r2", "start_s": 20.0, "end_s": 24.0}
        self.assertIsNone(make_item(self.source, short, "active_negative", 2, []))

    def test_replay_clipped_negative_is_rejected(self):
        self.assertIsNone(make_item(
            self.source, self.rally, "active_negative", 1,
            [{"start_s": 25.5, "end_s": 28.0}],
        ))


if __name__ == "__main__":
    unittest.main()
