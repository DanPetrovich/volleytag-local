import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "build_referee_roi_queue.py"
SPEC = importlib.util.spec_from_file_location("build_referee_roi_queue", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RefereeRoiQueueTest(unittest.TestCase):
    def test_uses_training_sources_only(self):
        playlist = [{"video": f"source-{index}.mp4"} for index in range(10)]
        events = [{"source": source, "action": "serve", "time_s": 20 + source}
                  for source in MODULE.TRAINING_SOURCES]
        events.append({"source": 5, "action": "serve", "time_s": 10})
        result = MODULE.build_queue({"playlist": playlist, "events": events})
        self.assertFalse(result["frozen_tests_used"])
        self.assertEqual([row["source"] for row in result["items"]],
                         list(MODULE.TRAINING_SOURCES))

    def test_requires_reviewed_serve_per_source(self):
        playlist = [{"video": f"source-{index}.mp4"} for index in range(10)]
        with self.assertRaisesRegex(ValueError, "without reviewed serves"):
            MODULE.build_queue({"playlist": playlist, "events": []})


if __name__ == "__main__":
    unittest.main()
