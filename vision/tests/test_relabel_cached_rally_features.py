import importlib.util
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as error:
    raise unittest.SkipTest(f"optional array dependency unavailable: {error.name}")


MODULE_PATH = Path(__file__).parents[1] / "relabel_cached_rally_features.py"
SPEC = importlib.util.spec_from_file_location("relabel_cached_rally_features", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RelabelCachedRallyFeaturesTest(unittest.TestCase):
    def test_selects_reviewed_subset_and_replaces_labels(self):
        queue = {"frozen_tests_used": False, "items": [
            {"id": "b", "source": 3}, {"id": "a", "source": 3},
        ]}
        results = {"answers": {
            "a": {"decision": "dead_ball"},
            "b": {"decision": "active_play"},
        }}
        cache = {
            "features": np.asarray([[1, 2], [3, 4], [5, 6]]),
            "labels": np.asarray(["old", "old", "old"]),
            "sources": np.asarray([3, 3, 3]),
            "item_ids": np.asarray(["a", "b", "discarded"]),
        }
        output = MODULE.relabel(queue, results, cache)
        np.testing.assert_array_equal(output["features"], [[3, 4], [1, 2]])
        self.assertEqual(output["labels"].tolist(), ["active_play", "dead_ball"])
        self.assertEqual(output["item_ids"].tolist(), ["b", "a"])

    def test_rejects_frozen_queue(self):
        with self.assertRaisesRegex(ValueError, "frozen"):
            MODULE.relabel({"frozen_tests_used": True}, {}, {})


if __name__ == "__main__":
    unittest.main()
