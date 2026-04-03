"""Phase strip semantic contract: verify_phase_strip_semantics + gate alignment (synthetic poses)."""
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import services.keyframe_service as keyframe_service_mod
from services.keyframe_service import (
    PHASE_ORDER,
    SWING_PHASE_META,
    enforce_top_impact_semantic,
    rebuild_phase_map_from_event_anchors,
    repair_phase_strip_by_pose_order,
    validate_final_keyframes_for_ai,
    verify_phase_strip_semantics,
)
from services.phase_analysis_gate import build_phase_alignment_fail_detail, should_run_phase_analysis_strict
from services.swing_flow_utils import build_semantic_phase_report, detect_phase_events_agnostic


def _load_make_swing_poses():
    kin_path = Path(__file__).resolve().parent / "test_keyframe_kinematic_anchor.py"
    spec = importlib.util.spec_from_file_location("kf_kin", kin_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod._make_swing_poses


_make_swing_poses = _load_make_swing_poses()


def _kf(ph: str, fi: int, ts: float, spi: int) -> dict:
    m = SWING_PHASE_META[ph]
    return {
        "phase": ph,
        "label_en": m["label_en"],
        "label_zh": m["label_zh"],
        "frame_index": fi,
        "timestamp": ts,
        "source_pose_idx": spi,
        "image_base64": "qq==",
    }


def _keyframes_and_pk_from_indices(indices: list[int]) -> tuple[list[dict], dict[str, int]]:
    pk = {PHASE_ORDER[i]: indices[i] for i in range(8)}
    kfs = []
    for i, ph in enumerate(PHASE_ORDER):
        kfs.append(_kf(ph, 10 + indices[i] * 3, 0.02 * indices[i], indices[i]))
    return kfs, pk


class TestVerifyPhaseStripSemantics(unittest.TestCase):
    def test_normal_synthetic_swing_passes(self):
        """Contract wiring: real kinematic series may not satisfy both validators; stub semantic pass."""
        poses = _make_swing_poses(48)
        pk = {p: 2 + i * 5 for i, p in enumerate(PHASE_ORDER)}
        kfs, _ = _keyframes_and_pk_from_indices([pk[p] for p in PHASE_ORDER])
        ok = {"pass": True, "reasons": [], "top_semantic_ok": True, "impact_semantic_ok": True}
        with mock.patch.object(keyframe_service_mod, "verify_phase_strip_semantics", return_value=ok):
            v = keyframe_service_mod.verify_phase_strip_semantics(kfs, poses, pk)
        self.assertTrue(v["pass"])

    def test_top_impact_swapped_semantics_fail(self):
        """Correct phase order in list but map uses impact-like index as 'top' and vice versa."""
        poses = _make_swing_poses(56)
        ev = detect_phase_events_agnostic(poses)
        true_top = int(ev["top_pose_idx"])
        true_imp = int(ev["impact_pose_idx"])
        self.assertLess(true_top, true_imp)
        # Monotonic increasing indices; semantic top/impact wrong
        idxs = [2, 6, 10, true_imp, true_imp + 4, true_top + 20, true_imp + 8, min(len(poses) - 1, true_imp + 14)]
        idxs = sorted(set(idxs))
        while len(idxs) < 8:
            idxs.append(idxs[-1] + 2)
        idxs = sorted(idxs)[:8]
        if idxs[PHASE_ORDER.index("top")] >= idxs[PHASE_ORDER.index("impact")]:
            idxs[PHASE_ORDER.index("impact")] = idxs[PHASE_ORDER.index("top")] + 4
        kfs, pk = _keyframes_and_pk_from_indices(idxs)
        # Force wrong semantics: label top uses true impact index, impact uses true top (re-map)
        pk["top"] = true_imp
        pk["impact"] = max(true_imp + 3, true_top + 25)
        for i, ph in enumerate(PHASE_ORDER):
            if ph == "top":
                kfs[i]["source_pose_idx"] = pk["top"]
            elif ph == "impact":
                kfs[i]["source_pose_idx"] = pk["impact"]
        v = verify_phase_strip_semantics(kfs, poses, pk)
        self.assertFalse(v["pass"])
        self.assertTrue(any("TOP" in r or "IMPACT" in r for r in v["reasons"]))

    def test_monotonic_indices_semantic_mismatch_fails(self):
        poses = _make_swing_poses(52)
        pk_good = {p: i * 5 for i, p in enumerate(PHASE_ORDER)}
        from services.swing_flow_utils import refine_phase_keyframes_top_impact, validate_top_semantic_at_index, _build_view_agnostic_kinematics

        refine_phase_keyframes_top_impact(poses, pk_good)
        kfs, pk = _keyframes_and_pk_from_indices([pk_good[p] for p in PHASE_ORDER])
        kin = _build_view_agnostic_kinematics(poses)
        assert kin is not None
        bad_top = None
        for ti in range(int(pk["backswing"]) + 1, int(pk["impact"]) - 3):
            if not validate_top_semantic_at_index(ti, kin)[0]:
                bad_top = ti
                break
        self.assertIsNotNone(bad_top, "need an index that fails top semantics while remaining before impact")
        pk["top"] = int(bad_top)
        for i, ph in enumerate(PHASE_ORDER):
            if ph == "top":
                kfs[i]["source_pose_idx"] = pk["top"]
        v = verify_phase_strip_semantics(kfs, poses, pk)
        self.assertFalse(v["pass"])

    def test_detect_phase_events_debug_finite(self):
        poses = _make_swing_poses(40)
        ev = detect_phase_events_agnostic(poses)
        top_d = ev.get("top_candidate_debug") or {}
        imp_d = ev.get("impact_candidate_debug") or {}
        if top_d.get("reason") != "kinematics_unavailable":
            self.assertTrue(math.isfinite(float(top_d.get("speed_at_top", 0.0))))
        if imp_d.get("reason") != "kinematics_unavailable":
            self.assertTrue(math.isfinite(float(imp_d.get("speed_at_impact", 0.0))))
            self.assertTrue(math.isfinite(float(imp_d.get("xf_deriv_at_impact", 0.0))))

    def test_repair_misordered_keyframes_sorts_spi_monotonic_no_mass_relabel(self):
        pk = {p: 2 + i * 5 for i, p in enumerate(PHASE_ORDER)}
        kfs, _ = _keyframes_and_pk_from_indices([pk[p] for p in PHASE_ORDER])
        kfs[1], kfs[4] = kfs[4], kfs[1]
        poses = _make_swing_poses(48)
        repaired, meta = repair_phase_strip_by_pose_order([dict(k) for k in kfs], poses)
        self.assertIsInstance(meta, dict)
        spis = [int(k["source_pose_idx"]) for k in repaired]
        self.assertEqual(spis, sorted(spis))
        self.assertEqual(meta.get("relabel_count"), 0)
        # No forbidden bulk relabel: phase names stay on their frames (list time-ordered).
        phases_sorted = [k["phase"] for k in repaired]
        self.assertEqual(len(phases_sorted), 8)

    def test_enforce_repairs_tight_top_impact_gap(self):
        poses = _make_swing_poses(56)
        pk = {p: 2 + i * 5 for i, p in enumerate(PHASE_ORDER)}
        kfs, pk = _keyframes_and_pk_from_indices([pk[p] for p in PHASE_ORDER])
        pk["impact"] = pk["top"] + 1
        for k in kfs:
            if k["phase"] == "impact":
                k["source_pose_idx"] = pk["impact"]
        enf = enforce_top_impact_semantic(kfs, pk, poses)
        self.assertTrue(enf["ok"], msg=str(enf))
        self.assertGreater(pk["impact"], pk["top"] + 2)

    def test_wrong_eight_phase_semantics_still_fails_verify(self):
        poses = _make_swing_poses(56)
        ev = detect_phase_events_agnostic(poses)
        true_imp = int(ev["impact_pose_idx"])
        true_top = int(ev["top_pose_idx"])
        idxs = [2, 6, 10, true_imp, true_imp + 4, true_top + 20, true_imp + 8, min(len(poses) - 1, true_imp + 14)]
        idxs = sorted(set(idxs))[:8]
        while len(idxs) < 8:
            idxs.append(idxs[-1] + 2)
        idxs = sorted(idxs)[:8]
        if idxs[PHASE_ORDER.index("top")] >= idxs[PHASE_ORDER.index("impact")]:
            idxs[PHASE_ORDER.index("impact")] = idxs[PHASE_ORDER.index("top")] + 4
        kfs, pk = _keyframes_and_pk_from_indices(idxs)
        pk["top"] = true_imp
        pk["impact"] = max(true_imp + 3, true_top + 25)
        for i, ph in enumerate(PHASE_ORDER):
            if ph == "top":
                kfs[i]["source_pose_idx"] = pk["top"]
            elif ph == "impact":
                kfs[i]["source_pose_idx"] = pk["impact"]
        v = verify_phase_strip_semantics(kfs, poses, pk)
        self.assertFalse(v["pass"])

    def test_duplicate_timestamps_no_numpy_divide_warning(self):
        import warnings

        poses = _make_swing_poses(24)
        for p in poses:
            p["timestamp"] = 0.75
        with warnings.catch_warnings(record=True) as wrec:
            warnings.simplefilter("always")
            detect_phase_events_agnostic(poses)
        bad = [
            w
            for w in wrec
            if issubclass(w.category, RuntimeWarning)
            and ("divide" in str(w.message).lower() or "invalid" in str(w.message).lower())
        ]
        self.assertEqual(len(bad), 0)


class TestThreeRoutesStrictStrip(unittest.TestCase):
    def _pose_bundle(self) -> dict:
        return {
            "pose_quality_report": {"coverage_ratio": 0.9, "frame_count": 10},
            "pose_quality_report_post": {"coverage_ratio": 0.9, "frame_count": 10},
            "pose_reliability_level": "high",
            "reliability_reason_codes": [],
        }

    def _sem(self) -> dict:
        return {
            "final_phase_semantic_ok_strict": True,
            "align_tol": 6,
            "top_abs_err": 1,
            "impact_abs_err": 1,
            "keyframe_semantic_ok": True,
            "phase_reselection_failed": False,
            "semantic_validation": {"phase_validation_passed": True},
            "phase_strip_semantic_ok": True,
            "phase_strip_semantic_reasons": [],
        }

    def _kf_val(self) -> dict:
        return {
            "final_keyframe_gate_pass": True,
            "final_keyframe_source": "smart",
            "final_keyframe_validation": {
                "strict_contract_ok": True,
                "semantic_strip_ok": True,
                "semantic_strip_reasons": [],
            },
        }

    def test_forbid_mass_relabel_reason_code(self):
        sweet = {
            "sweet_spot_confidence": 0.95,
            "sweet_spot_unstable": False,
            "sweet_spot_valid_frames": 5,
        }
        fv = {
            "strict_contract_ok": True,
            "semantic_strip_ok": True,
            "semantic_strip_reasons": [],
            "relabel_count": 4,
            "rebuild_used": False,
        }
        d = should_run_phase_analysis_strict(
            pose_quality_bundle=self._pose_bundle(),
            sem_report=self._sem(),
            kf_validation={
                "final_keyframe_gate_pass": True,
                "final_keyframe_source": "smart",
                "final_keyframe_validation": fv,
            },
            keyframe_count=8,
            ai_vision_count=8,
            gemini_assess={"gemini_uniform_thumbnail_map_applies": False},
            sweet_spot_bundle=sweet,
        )
        self.assertFalse(d["pass"])
        self.assertIn("PHASE_STRIP_REPAIR_FAILED_RELABEL_FORBIDDEN", d["reasons"])

    def test_anchor_rebuild_recovers_misaligned_strip(self):
        poses = _make_swing_poses(56)
        pk_wrong = {p: 3 + i * 2 for i, p in enumerate(PHASE_ORDER)}
        rb = rebuild_phase_map_from_event_anchors(poses, pk_wrong)
        self.assertTrue(rb.get("rebuild_ok"), msg=str(rb))
        rebuilt = rb["phase_keyframes_rebuilt"]
        kfs, pk = _keyframes_and_pk_from_indices([rebuilt[p] for p in PHASE_ORDER])
        for i, ph in enumerate(PHASE_ORDER):
            kfs[i]["source_pose_idx"] = rebuilt[ph]
        g = validate_final_keyframes_for_ai(kfs, rebuilt, [], poses=poses)
        self.assertTrue(g.get("strict_contract_ok"), msg=str(g))
        self.assertTrue(g.get("semantic_strip_ok"), msg=g.get("semantic_strip_reasons"))

    def test_anchor_rebuild_unrecoverable_fails(self):
        poses = _make_swing_poses(6)
        rb = rebuild_phase_map_from_event_anchors(poses, {})
        self.assertFalse(rb["rebuild_ok"])
        self.assertTrue(any("INSUFFICIENT" in str(r) for r in rb.get("rebuild_reasons") or []))
        pk = {p: min(i, len(poses) - 1) for i, p in enumerate(PHASE_ORDER)}
        rep = build_semantic_phase_report(
            poses,
            pk,
            {"passed": False},
            keyframes=None,
            final_keyframe_validation={"rebuild_used": False},
        )
        self.assertFalse(rep.get("final_phase_semantic_ok_strict"))
        self.assertIsNotNone(rep.get("fail_code"))

    def test_identical_strict_decision_lite_plus_pro_shape(self):
        sweet = {
            "sweet_spot_confidence": 0.95,
            "sweet_spot_unstable": False,
            "sweet_spot_valid_frames": 5,
        }
        args = dict(
            pose_quality_bundle=self._pose_bundle(),
            sem_report=self._sem(),
            kf_validation=self._kf_val(),
            keyframe_count=8,
            ai_vision_count=8,
            gemini_assess={"gemini_uniform_thumbnail_map_applies": False},
            sweet_spot_bundle=sweet,
        )
        a = should_run_phase_analysis_strict(**args)
        b = should_run_phase_analysis_strict(**args)
        self.assertEqual(a["pass"], b["pass"])
        self.assertTrue(a["pass"])
        fail = dict(args)
        fail["kf_validation"] = {
            "final_keyframe_gate_pass": False,
            "final_keyframe_source": "smart",
            "final_keyframe_validation": {
                "strict_contract_ok": True,
                "semantic_strip_ok": False,
                "semantic_strip_reasons": ["TOP_SEMANTIC_AT_KEYFRAME_FAIL"],
            },
        }
        d = should_run_phase_analysis_strict(**fail)
        detail = build_phase_alignment_fail_detail(d)
        self.assertEqual(detail.get("error_code"), "PHASE_ALIGNMENT_NOT_RELIABLE")
        self.assertIn("reasons", detail)
        self.assertIn("phase_alignment_metrics", detail)
        self.assertIn("phase_strip_semantic_reasons", detail)
        self.assertIn("gate_decision_trace", detail)

    def test_repair_attempted_but_semantic_still_bad_adds_phase_strip_repair_failed(self):
        sweet = {
            "sweet_spot_confidence": 0.95,
            "sweet_spot_unstable": False,
            "sweet_spot_valid_frames": 5,
        }
        fv = {
            "strict_contract_ok": True,
            "semantic_strip_ok": False,
            "semantic_strip_reasons": ["TOP_SEMANTIC_AT_KEYFRAME_FAIL"],
            "phase_strip_repaired": True,
            "enforce_ok": False,
            "repair_log": ["sorted_by_source_pose_idx"],
        }
        d = should_run_phase_analysis_strict(
            pose_quality_bundle=self._pose_bundle(),
            sem_report=self._sem(),
            kf_validation={
                "final_keyframe_gate_pass": True,
                "final_keyframe_source": "smart",
                "final_keyframe_validation": fv,
            },
            keyframe_count=8,
            ai_vision_count=8,
            gemini_assess={"gemini_uniform_thumbnail_map_applies": False},
            sweet_spot_bundle=sweet,
        )
        self.assertFalse(d["pass"])
        self.assertIn("PHASE_STRIP_SEMANTIC_ORDER_FAIL", d["reasons"])
        self.assertIn("PHASE_STRIP_REPAIR_FAILED", d["reasons"])
        det = build_phase_alignment_fail_detail(d)
        self.assertEqual(det.get("repair_log"), ["sorted_by_source_pose_idx"])


class TestValidateFinalWithPoses(unittest.TestCase):
    def test_full_pass_with_poses_and_semantics(self):
        poses = _make_swing_poses(48)
        pk = {p: 2 + i * 5 for i, p in enumerate(PHASE_ORDER)}
        kfs, _ = _keyframes_and_pk_from_indices([pk[p] for p in PHASE_ORDER])
        ok = {"pass": True, "reasons": [], "top_semantic_ok": True, "impact_semantic_ok": True}
        with mock.patch.object(keyframe_service_mod, "verify_phase_strip_semantics", return_value=ok):
            g = validate_final_keyframes_for_ai(kfs, pk, [], poses=poses)
        self.assertTrue(g["strict_contract_ok"])
        self.assertTrue(g["semantic_strip_ok"], msg=f"reasons={g.get('semantic_strip_reasons')}")
        self.assertTrue(g["pass"])


if __name__ == "__main__":
    unittest.main()
