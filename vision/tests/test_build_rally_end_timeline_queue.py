import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "build_rally_end_timeline_queue.py"


class RallyEndTimelineQueueTest(unittest.TestCase):
    def test_preserves_frozen_interval_without_candidate_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "route.json", root / "queue.json"
            source.write_text(json.dumps({
                "video": "local.mp4", "start_s": 600, "end_s": 900,
                "frozen_tests_used": True, "events": [{"time_s": 700}],
            }))
            subprocess.run([sys.executable, str(SCRIPT), str(source),
                            "--output", str(output)], check=True, capture_output=True)
            payload = json.loads(output.read_text())
            self.assertEqual((payload["start_s"], payload["end_s"]), (600, 900))
            self.assertTrue(payload["frozen_tests_used"])
            self.assertNotIn("events", payload)


if __name__ == "__main__":
    unittest.main()
