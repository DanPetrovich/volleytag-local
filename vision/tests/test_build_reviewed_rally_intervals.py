import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "build_reviewed_rally_intervals.py"
SPEC = importlib.util.spec_from_file_location("build_reviewed_rally_intervals", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ReviewedRallyIntervalsTest(unittest.TestCase):
    def test_builds_only_complete_intervals_between_markers(self):
        rows = MODULE.build_intervals([10, 25, 40])
        self.assertEqual(len(rows), 2)
        self.assertEqual((rows[0]["safe_start_s"], rows[0]["end_s"]), (10, 25))
        self.assertFalse(rows[0]["left_censored"])

    def test_rejects_unsorted_or_duplicate_markers(self):
        with self.assertRaises(ValueError):
            MODULE.build_intervals([10, 9])
        with self.assertRaises(ValueError):
            MODULE.build_intervals([10, 10])


if __name__ == "__main__":
    unittest.main()
