import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "build_rally_interval_timeline_queue.py"
SPEC = importlib.util.spec_from_file_location("build_rally_interval_timeline_queue", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RallyIntervalTimelineQueueTest(unittest.TestCase):
    def test_marks_frozen_split_explicitly(self):
        payload = MODULE.build_queue("match.mp4", 100, 200, "frozen_eval")
        self.assertTrue(payload["frozen_tests_used"])
        self.assertEqual(payload["boundary_definition"]["start"],
                         "server contacts the ball for the live serve")

    def test_rejects_nonpositive_interval(self):
        with self.assertRaisesRegex(ValueError, "positive duration"):
            MODULE.build_queue("match.mp4", 200, 100, "train")

    def test_rejects_unknown_split(self):
        with self.assertRaisesRegex(ValueError, "split must be"):
            MODULE.build_queue("match.mp4", 100, 200, "test")


if __name__ == "__main__":
    unittest.main()
