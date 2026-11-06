import unittest

try:
    import numpy as np
    import ultralytics  # noqa: F401
except ModuleNotFoundError as error:
    raise unittest.SkipTest(f"optional ball dependency unavailable: {error.name}")

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extract_rally_ball_features import trajectory_features  # noqa: E402


class RallyBallFeaturesTest(unittest.TestCase):
    def test_moving_candidates_are_detected_despite_static_distractor(self):
        frames = []
        for index in range(5):
            frames.append(np.asarray([
                [.9, .1, .1, .001],
                [.7, .3 + index * .05, .5, .001],
            ], dtype=np.float32))
        features = trajectory_features(frames)
        self.assertEqual(features.shape, (51,))
        self.assertGreater(features[-1], 0)
        self.assertGreater(features[-6], .04)

    def test_empty_sequence_has_zero_features(self):
        features = trajectory_features([np.empty((0, 4), dtype=np.float32) for _ in range(5)])
        self.assertTrue(np.all(features == 0))


if __name__ == "__main__":
    unittest.main()
