#!/usr/bin/env python3
"""Regression checks for the temporal player-51 route decoder."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "route_player51_temporal.py"
SPEC = importlib.util.spec_from_file_location("route_player51_temporal", MODULE_PATH)
assert SPEC and SPEC.loader
route = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(route)


def candidate(x: float, *, direct: float = 0.8, track_id: int = 1) -> dict:
    return {
        "box": [x, 0.0, x + 50.0, 100.0],
        "confidence": direct or 0.8,
        "direct_confidence": direct,
        "track_id": track_id,
        "source": "direct" if direct else "generic",
    }


class TemporalRouteTest(unittest.TestCase):
    def test_rejects_impossible_direct_teleport(self) -> None:
        sets = [[candidate(0), None], [candidate(500, track_id=2), None]]
        self.assertEqual(route.decode_anchor_first(sets, [0.0, 0.4], 6.0), [0, 1])

    def test_keeps_physically_plausible_motion(self) -> None:
        sets = [[candidate(0), None], [candidate(60), None]]
        self.assertEqual(route.decode_anchor_first(sets, [0.0, 0.4], 6.0), [0, 0])

    def test_bridges_short_gap_on_same_generic_track(self) -> None:
        sets = [[candidate(0), None], [candidate(30, direct=0.0), None]]
        self.assertEqual(route.decode_anchor_first(sets, [0.0, 0.4], 6.0), [0, 0])

    def test_direct_bridge_never_overrides_direct_top1(self) -> None:
        sets = [[candidate(0), None], [candidate(500, direct=.9, track_id=2), None]]
        self.assertEqual(route.decode_direct_bridge(sets, [0.0, 0.4], 6.0), [0, 0])

    def test_direct_bridge_uses_same_track_only_during_short_gap(self) -> None:
        sets = [[candidate(0), None], [candidate(30, direct=0.0), None],
                [candidate(40, direct=0.0), None]]
        self.assertEqual(route.decode_direct_bridge(sets, [0.0, 0.4, 2.0], 6.0), [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
