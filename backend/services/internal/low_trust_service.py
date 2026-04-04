from __future__ import annotations

from typing import Dict, List


def build_low_trust_result(analysis_id: str, keyframes: List[dict], fail_reasons: List[str]) -> Dict[str, object]:
    return {
        "analysis_id": analysis_id,
        "status": "low_trust",
        "trust_level": "low",
        "keyframes": keyframes,
        "fail_reasons": sorted(set(fail_reasons)),
    }
