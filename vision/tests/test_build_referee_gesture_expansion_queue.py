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
MODULE_PATH = VISION_DIR / "build_referee_gesture_expansion_queue.py"
SPEC = importlib.util.spec_from_file_location("build_referee_gesture_expansion_queue", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RefereeGestureExpansionQueueTest(unittest.TestCase):
    def test_diverse_selection_rejects_near_duplicate_motion(self):
        candidates = [(10, 1.0), (20, .99), (30, .8)]
        descriptors = np.asarray([
            [1.0, 0.0],
            [.999, .045],
            [0.0, 1.0],
        ], dtype=np.float32)
        descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
        self.assertEqual(MODULE.diverse_selection(candidates, descriptors, 2), [0, 2])

    def test_diverse_selection_honors_requested_count(self):
        candidates = [(10, 1.0), (20, .8)]
        descriptors = np.eye(2, dtype=np.float32)
        self.assertEqual(len(MODULE.diverse_selection(candidates, descriptors, 10)), 2)


if __name__ == "__main__":
    unittest.main()
