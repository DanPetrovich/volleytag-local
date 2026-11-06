import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "evaluate_rally_end_timestamps.py"
SPEC = importlib.util.spec_from_file_location("evaluate_rally_end_timestamps", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RallyEndTimestampEvaluationTest(unittest.TestCase):
    def test_maximizes_ordered_matches_then_minimizes_error(self):
        pairs = MODULE.ordered_matches([10, 20, 30], [11, 19, 40], 2)
        self.assertEqual(pairs, [(10, 11), (20, 19)])

    def test_does_not_reuse_a_timestamp(self):
        pairs = MODULE.ordered_matches([10, 11], [10.5], 1)
        self.assertEqual(len(pairs), 1)


if __name__ == "__main__":
    unittest.main()
