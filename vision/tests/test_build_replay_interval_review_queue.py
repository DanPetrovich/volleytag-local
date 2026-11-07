import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "build_replay_interval_review_queue.py"
SPEC = importlib.util.spec_from_file_location("build_replay_interval_review_queue", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ReplayIntervalReviewQueueTest(unittest.TestCase):
    def test_builds_sorted_training_only_queue(self):
        payload = {"frozen_tests_used": False, "sources": {"11": {
            "video": "v.mp4", "duration_s": 100,
            "replay_pairs": [
                {"start_s": 50, "end_s": 58},
                {"start_s": 10, "end_s": 18},
            ],
        }}}
        queue = MODULE.build_queue([payload])
        self.assertEqual([item["start_s"] for item in queue["items"]], [10, 50])
        self.assertEqual(queue["items"][0]["window_start_s"], 8)
        self.assertEqual(queue["counts"], {"items": 2, "sources": 1})

    def test_rejects_frozen_source(self):
        with self.assertRaisesRegex(ValueError, "frozen"):
            MODULE.build_queue([{"frozen_tests_used": True, "sources": {}}])


if __name__ == "__main__":
    unittest.main()
