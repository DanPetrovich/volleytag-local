import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "build_dense_rally_state_samples.py"
SPEC = importlib.util.spec_from_file_location("build_dense_rally_state_samples", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DenseRallyStateSamplesTest(unittest.TestCase):
    def test_builds_blocked_partitions_and_labels(self):
        queue = {"video": "v.mp4", "start_s": 0, "end_s": 20}
        results = {"pending_start": None, "rallies": [
            {"id": "r1", "start_s": 4, "end_s": 7,
             "start_confidence": "exact", "end_confidence": "exact"},
            {"id": "r2", "start_s": 12, "end_s": 16,
             "start_confidence": "inferred", "end_confidence": "exact"},
        ]}
        samples, labels = MODULE.build_samples(queue, results, 3, 1, 1)
        self.assertEqual(samples["split_s"], 9.5)
        by_time = {row["time_s"]: row for row in samples["items"]}
        self.assertEqual(labels["answers"][by_time[5]["id"]]["decision"], "active_play")
        self.assertEqual(by_time[5]["partition"], "development")
        self.assertEqual(by_time[13]["partition"], "evaluation")

    def test_rejects_unfinished_review(self):
        with self.assertRaisesRegex(ValueError, "unfinished"):
            MODULE.build_samples({"video": "v", "start_s": 0, "end_s": 10},
                                 {"pending_start": {"time_s": 5}, "rallies": []},
                                 1, 1, 1)

    def test_can_build_external_evaluation_only(self):
        queue = {"video": "v.mp4", "start_s": 0, "end_s": 20}
        results = {"pending_start": None, "rallies": [
            {"id": "r1", "start_s": 4, "end_s": 7,
             "start_confidence": "exact", "end_confidence": "exact"},
        ]}
        samples, _ = MODULE.build_samples(queue, results, 4, 1, 0)
        self.assertEqual(samples["development_rallies"], 0)
        self.assertTrue(all(row["partition"] == "evaluation" for row in samples["items"]))

    def test_excludes_replay_and_bumper_samples(self):
        queue = {"video": "v.mp4", "start_s": 0, "end_s": 20}
        results = {"pending_start": None, "rallies": [
            {"id": "r1", "start_s": 4, "end_s": 7,
             "start_confidence": "exact", "end_confidence": "exact"},
            {"id": "r2", "start_s": 12, "end_s": 16,
             "start_confidence": "exact", "end_confidence": "exact"},
        ]}
        samples, _ = MODULE.build_samples(
            queue, results, 3, 1, 1,
            [{"start_s": 8.5, "end_s": 10.5}],
        )
        times = {row["time_s"] for row in samples["items"]}
        self.assertNotIn(9, times)
        self.assertNotIn(10, times)
        self.assertEqual(samples["excluded_replay_samples"], 2)
        self.assertTrue(samples["replay_mask_enabled"])

    def test_rejects_frozen_replay_metadata(self):
        with self.assertRaisesRegex(ValueError, "frozen"):
            MODULE.replay_exclusions({"frozen_tests_used": True}, "v.mp4")


if __name__ == "__main__":
    unittest.main()
