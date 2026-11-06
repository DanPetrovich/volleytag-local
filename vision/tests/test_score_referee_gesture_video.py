import importlib.util
import sys
import unittest
from pathlib import Path

try:
    import joblib  # noqa: F401
    import numpy  # noqa: F401
    import torch  # noqa: F401
    import torchvision  # noqa: F401
except (ModuleNotFoundError, RuntimeError) as error:
    raise unittest.SkipTest(f"optional vision dependency unavailable: {error}")


VISION_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(VISION_DIR))
MODULE_PATH = VISION_DIR / "score_referee_gesture_video.py"
SPEC = importlib.util.spec_from_file_location("score_referee_gesture_video", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RefereeBoundaryGroupingTest(unittest.TestCase):
    def test_groups_nearby_hits_and_keeps_highest_score(self):
        rows = [
            {"time_s": 10, "boundary_score": .4},
            {"time_s": 11, "boundary_score": .8},
            {"time_s": 15, "boundary_score": .6},
        ]
        grouped = MODULE.group_candidates(rows, maximum_gap_s=2)
        self.assertEqual(len(grouped), 2)
        self.assertEqual(grouped[0]["time_s"], 11)
        self.assertEqual(grouped[0]["raw_hits"], 2)


if __name__ == "__main__":
    unittest.main()
