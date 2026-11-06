import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "evaluate_fused_referee_boundary.py"
SPEC = importlib.util.spec_from_file_location("evaluate_fused_referee_boundary", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FusedRefereeBoundaryEvaluationTest(unittest.TestCase):
    def test_recovers_missing_scoreboard_ends(self):
        route = {
            "expected_score_changes": 3, "scoreboard_events": 2,
            "candidate_density_per_minute": 3.5, "events": [{}, {}, {}, {}],
        }
        queue = {"items": [{"id": "a"}, {"id": "b"}]}
        results = {"answers": {
            "a": {"decision": "rally_end_team_to_serve"},
            "b": {"decision": "no_signal"},
        }}
        report = MODULE.evaluate(route, queue, results, 4)
        self.assertEqual(report["rally_end_recall"], 1)
        self.assertEqual(report["boundary_candidate_precision"], .75)
        self.assertTrue(report["passed"])


if __name__ == "__main__":
    unittest.main()
