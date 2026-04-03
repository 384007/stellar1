import unittest
import sys
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from services.swing_flow_utils import validate_impact_semantic_at_index


def _kin(speed_high: bool, unwinding: bool, strike_ok: bool) -> dict:
    n = 32
    speed = np.zeros(n, dtype=np.float64)
    speed[:] = 0.1
    speed[20] = 1.0 if speed_high else 0.2
    xf_d = np.zeros(n, dtype=np.float64)
    xf_d[20] = -0.5 if unwinding else 0.2
    hand_hip = np.full(n, 0.2, dtype=np.float64)
    if not strike_ok:
        hand_hip[20] = 1.0
    return {
        "n": n,
        "valid": np.ones(n, dtype=bool),
        "speed_s": speed,
        "xf_d": xf_d,
        "hand_hip": hand_hip,
        "kinematic_fail_codes": [],
    }


class TestImpactSemanticContract(unittest.TestCase):
    def test_impact_requires_unwinding_even_with_high_speed(self):
        kin = _kin(speed_high=True, unwinding=False, strike_ok=True)
        ok, checks = validate_impact_semantic_at_index(20, top_i=10, exc_apex=8, kin=kin)
        self.assertFalse(ok)
        self.assertTrue(checks["speed_high"])
        self.assertFalse(checks["unwinding"])

    def test_impact_requires_strike_zone_reasonable(self):
        kin = _kin(speed_high=True, unwinding=True, strike_ok=False)
        ok, checks = validate_impact_semantic_at_index(20, top_i=10, exc_apex=8, kin=kin)
        self.assertFalse(ok)
        self.assertFalse(checks["strike_zone_reasonable"])


if __name__ == "__main__":
    unittest.main()
