import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "fuse_referee_boundary_candidates.py"
SPEC = importlib.util.spec_from_file_location("fuse_referee_boundary_candidates", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FuseRefereeBoundaryCandidatesTest(unittest.TestCase):
    def test_matched_motion_cannot_move_rally_end_before_score_change(self):
        events, used = MODULE.fuse_events(
            [{"time_s": 100}],
            [{"time_s": 94, "motion_score": .8}],
            before_s=10,
            after_s=6,
        )
        self.assertEqual(used, {0})
        self.assertEqual(events[0]["time_s"], 100)
        self.assertEqual(events[0]["matched_motion_time_s"], 94)
        self.assertEqual(events[0]["reason"], "scoreboard+motion")

    def test_unmatched_motion_remains_fallback(self):
        events, used = MODULE.fuse_events(
            [{"time_s": 100}],
            [{"time_s": 70, "motion_score": .4}],
            before_s=10,
            after_s=6,
        )
        self.assertEqual(used, set())
        self.assertEqual([row["reason"] for row in events],
                         ["motion_fallback", "scoreboard_only"])


if __name__ == "__main__":
    unittest.main()
