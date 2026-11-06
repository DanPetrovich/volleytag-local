import importlib.util
import unittest
from pathlib import Path

try:
    import numpy as np
    import sklearn  # noqa: F401
except ModuleNotFoundError as error:
    raise unittest.SkipTest(f"optional vision dependency unavailable: {error.name}")


MODULE_PATH = Path(__file__).parents[1] / "evaluate_rally_state_classifier.py"
SPEC = importlib.util.spec_from_file_location("evaluate_rally_state_classifier", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RallyStateOperatingPointTest(unittest.TestCase):
    def test_keeps_required_recall_and_maximizes_precision(self):
        labels = np.asarray([1, 1, 0, 0])
        scores = np.asarray([.9, .7, .8, .1])
        point = MODULE.operating_point(labels, scores, min_recall=.5)
        self.assertEqual(point["recall"], .5)
        self.assertEqual(point["precision"], 1.0)

    def test_all_positive_baseline_is_available(self):
        labels = np.asarray([1, 0, 0])
        scores = np.asarray([.4, .3, .2])
        point = MODULE.operating_point(labels, scores, min_recall=1.0)
        self.assertEqual(point["tp"], 1)
        self.assertEqual(point["recall"], 1.0)

    def test_recall_first_point_keeps_precision_floor(self):
        labels = np.asarray([1, 1, 0, 0])
        scores = np.asarray([.9, .7, .8, .1])
        point = MODULE.recall_first_point(labels, scores, min_precision=.6)
        self.assertEqual(point["recall"], 1.0)
        self.assertGreaterEqual(point["precision"], .6)


if __name__ == "__main__":
    unittest.main()
