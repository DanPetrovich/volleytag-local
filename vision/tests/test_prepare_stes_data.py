import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "prepare_stes_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_stes_data", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PrepareStesDataTest(unittest.TestCase):
    def test_validates_split_and_counts_events(self):
        report = MODULE.validate_split(
            [{
                "video": "clip-1",
                "num_frames": 20,
                "events": [
                    {"frame": 2, "label": "Serve", "xy": [0.2, 0.3]},
                    {"frame": 18, "label": "score", "x": 0.4, "y": 0.8},
                ],
            }],
            split="train",
            allowed_classes=MODULE.EXPECTED_VNL_CLASSES,
        )
        self.assertEqual(report["video_count"], 1)
        self.assertEqual(report["event_count"], 2)
        self.assertEqual(report["events_by_class"], {"score": 1, "serve": 1})

    def test_rejects_split_leaks_and_invalid_frames(self):
        with self.assertRaisesRegex(ValueError, "frame out of bounds"):
            MODULE.validate_split(
                [{"video": "bad", "num_frames": 3,
                  "events": [{"frame": 3, "label": "serve"}]}],
                split="train",
                allowed_classes=MODULE.EXPECTED_VNL_CLASSES,
            )

    def test_exports_only_non_frozen_rally_boundaries(self):
        queue = {
            "frozen_tests_used": False,
            "rallies": [
                {"start_s": 1.0, "end_s": 4.0,
                 "start_confidence": "inferred", "end_confidence": "exact"},
                {"start_s": 7.0, "end_s": 9.0,
                 "start_confidence": "exact", "end_confidence": "exact"},
            ],
        }
        row = MODULE.export_queue(
            queue,
            video_id="training-a",
            meta={"fps": 25.0, "num_frames": 300, "width": 398, "height": 224},
        )
        self.assertEqual(row["num_events"], 4)
        self.assertEqual(
            [(event["frame"], event["label"]) for event in row["events"]],
            [(25, "serve"), (100, "score"), (175, "serve"), (225, "score")],
        )
        queue["frozen_tests_used"] = True
        with self.assertRaisesRegex(ValueError, "refusing"):
            MODULE.export_queue(
                queue,
                video_id="frozen",
                meta={"fps": 25.0, "num_frames": 300, "width": 398, "height": 224},
            )


if __name__ == "__main__":
    unittest.main()
