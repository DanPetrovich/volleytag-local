import importlib.util
import sys
import unittest
from pathlib import Path

try:
    import cv2  # noqa: F401
    import numpy as np
except ModuleNotFoundError as error:
    raise unittest.SkipTest(f"optional vision dependency unavailable: {error.name}")


VISION_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(VISION_DIR))
MODULE_PATH = VISION_DIR / "build_referee_motion_candidates.py"
SPEC = importlib.util.spec_from_file_location("build_referee_motion_candidates", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RefereeMotionCandidatesTest(unittest.TestCase):
    def test_selects_strong_separated_non_replay_peaks(self):
        scores = np.asarray([0, .9, .8, 0, .7, 0, .95, 0], dtype=np.float32)
        peaks = MODULE.select_motion_peaks(
            scores, fps=1, start_s=1, end_s=8, count=2, separation_s=2,
            excluded=[{"start_s": 6, "end_s": 7}],
        )
        self.assertEqual([row[0] for row in peaks], [2, 5])


if __name__ == "__main__":
    unittest.main()
