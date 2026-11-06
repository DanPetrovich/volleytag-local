import importlib.util
import unittest
from pathlib import Path

try:
    import numpy as np
    import sklearn  # noqa: F401
except ModuleNotFoundError as error:
    raise unittest.SkipTest(f"optional classifier dependency unavailable: {error.name}")


MODULE_PATH = Path(__file__).parents[1] / "evaluate_referee_gesture_classifier.py"
SPEC = importlib.util.spec_from_file_location("evaluate_referee_gesture_classifier", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RefereeGestureOperatingPointTest(unittest.TestCase):
    def test_maximizes_precision_at_maximum_recall(self):
        labels = np.asarray(["serve_authorization", "rally_end_team_to_serve",
                             "other", "other"])
        scores = np.asarray([.9, .6, .7, .1])
        result = MODULE.recall_first_operating_point(labels, scores)
        self.assertEqual(result["recall"], 1)
        self.assertEqual(result["true_positive"], 2)
        self.assertEqual(result["false_positive"], 1)
        self.assertAlmostEqual(result["threshold"], .6)


if __name__ == "__main__":
    unittest.main()
