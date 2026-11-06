import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "build_rally_touch_review_queue.py"


class RallyTouchReviewQueueTest(unittest.TestCase):
    def test_preserves_frozen_flag_and_safe_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "intervals.json", root / "queue.json"
            source.write_text(json.dumps({
                "video": "local.mp4", "frozen_tests_used": True,
                "intervals": [{"id": "rally-001", "safe_start_s": 10,
                               "end_s": 20, "left_censored": True,
                               "end_reason": "scoreboard_only"},
                              {"id": "rally-002", "safe_start_s": 20,
                               "end_s": 30, "left_censored": False,
                               "end_reason": "scoreboard_only"}],
            }))
            subprocess.run([sys.executable, str(SCRIPT), str(source), "--source", "5",
                            "--output", str(output)], check=True, capture_output=True)
            payload = json.loads(output.read_text())
            self.assertTrue(payload["frozen_tests_used"])
            self.assertEqual(payload["excluded_left_censored"], 1)
            self.assertEqual(len(payload["items"]), 1)
            self.assertEqual(payload["items"][0]["window_start_s"], 20)
            self.assertEqual(payload["items"][0]["window_end_s"], 30)


if __name__ == "__main__":
    unittest.main()
