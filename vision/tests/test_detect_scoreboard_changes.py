import importlib.util
import unittest
from pathlib import Path

try:
    import cv2  # noqa: F401
    import numpy  # noqa: F401
except ModuleNotFoundError as error:
    raise unittest.SkipTest(f"optional vision dependency unavailable: {error.name}")


MODULE_PATH = Path(__file__).parents[1] / "detect_scoreboard_changes.py"
SPEC = importlib.util.spec_from_file_location("detect_scoreboard_changes", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ScoreboardChangesTest(unittest.TestCase):
    def test_thresholds_separates_and_excludes_replays(self):
        changes = MODULE.separated_changes(
            [(10, .08), (11, .07), (20, .04), (30, .02), (40, .09)],
            threshold=.03, separation_s=5,
            excluded=[{"start_s": 39, "end_s": 41}],
        )
        self.assertEqual([row[0] for row in changes], [10, 20])


if __name__ == "__main__":
    unittest.main()
