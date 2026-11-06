import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "build_rally_state_review_queue.py"
SPEC = importlib.util.spec_from_file_location("build_rally_state_review_queue", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RallyStateQueueValidationTest(unittest.TestCase):
    def test_accepts_unique_training_items(self):
        MODULE.validate_queue([{
            "id": "one", "source": 0, "window_start_s": 0.0,
        }, {
            "id": "two", "source": 9, "window_start_s": 1.0,
        }])

    def test_rejects_duplicate_ids(self):
        items = [
            {"id": "same", "source": 0, "window_start_s": 0.0},
            {"id": "same", "source": 1, "window_start_s": 1.0},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.validate_queue(items)

    def test_rejects_frozen_or_unknown_source(self):
        with self.assertRaisesRegex(ValueError, "non-training"):
            MODULE.validate_queue([{
                "id": "frozen", "source": 5, "window_start_s": 0.0,
            }])


if __name__ == "__main__":
    unittest.main()
