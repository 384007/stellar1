import sys
import unittest
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from routers.plus_analyze import _authoritative_chain_check


def _poses(n: int):
    return [{"frame_index": i * 5, "timestamp": i * 0.166} for i in range(n)]


class TestAuthoritativePhaseChainCheck(unittest.TestCase):
    def test_rejects_tiny_impact_follow_gap(self):
        poses = _poses(83)
        phase_map = {
            "address": 5,
            "takeaway": 12,
            "backswing": 22,
            "top": 35,
            "downswing": 45,
            "impact": 60,
            "follow_through": 61,  # tiny gap
            "finish": 70,
        }
        ok, reasons = _authoritative_chain_check(poses, phase_map)
        self.assertFalse(ok)
        self.assertIn("IMPACT_TO_FOLLOW_GAP_TOO_SMALL", reasons)

    def test_rejects_non_monotonic_chain(self):
        poses = _poses(83)
        phase_map = {
            "address": 5,
            "takeaway": 12,
            "backswing": 22,
            "top": 35,
            "downswing": 30,  # non-monotonic
            "impact": 60,
            "follow_through": 70,
            "finish": 80,
        }
        ok, reasons = _authoritative_chain_check(poses, phase_map)
        self.assertFalse(ok)
        self.assertIn("NON_MONOTONIC_PHASE_ORDER", reasons)


if __name__ == "__main__":
    unittest.main()

