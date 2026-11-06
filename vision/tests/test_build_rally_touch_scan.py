import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "build_rally_touch_scan.py"
SPEC = importlib.util.spec_from_file_location("build_rally_touch_scan", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RallyTouchScanTest(unittest.TestCase):
    def test_samples_only_inside_clip_margins(self):
        rows = MODULE.build_samples([{"id": "a", "rally_id": "r1", "source": 5,
                                      "video": "v.mp4", "window_start_s": 10,
                                      "window_end_s": 13}], 1, .8)
        self.assertEqual([row["time_s"] for row in rows], [10.8, 11.8])

    def test_rejects_nonpositive_step(self):
        with self.assertRaisesRegex(ValueError, "step_s must be positive"):
            MODULE.build_samples([], 0, .8)


if __name__ == "__main__":
    unittest.main()
