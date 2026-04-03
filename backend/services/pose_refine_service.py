"""
Temporal pose refinement: 1€ filter + light Kalman, short occlusion fill, body-scale metrics.

Runs after MediaPipe detection (and optional EMA smooth in pose_service). Quality gating
uses pre-refine statistics for coverage/truncation; post-refine jitter contributes to
final reliability so refinement can recover marginal temporal noise without hiding
structural truncation/occlusion failures.
"""

from __future__ import annotations

import logging
import math
import copy
from typing import Any

import numpy as np

from services.pose_strict_config import (
    COVERAGE_JOINTS,
    COVERAGE_HIGH,
    COVERAGE_LOW_FAIL,
    JITTER_JOINTS,
    JITTER_SCORE_HIGH,
    KALMAN_Q,
    KALMAN_R,
    KALMAN_R_LOW_VISIBILITY_MULT,
    OCCLUSION_MAX_GAP_FRAMES,
    ONE_EURO_BETA,
    ONE_EURO_BETA_LOW_VISIBILITY_MULT,
    ONE_EURO_D_CUTOFF_HZ,
    ONE_EURO_MIN_CUTOFF_HZ,
    SHOULDER_WIDTH_MIN_NORM,
    SMOOTHING_LAG_HIGH_THRESH,
    TRACK_CONSISTENCY_LOW,
    TRACK_JUMP_THRESH_NORM,
    TRUNCATION_BODY_PARTS,
    TRUNCATION_MARGIN,
    TRUNCATION_SCORE_HIGH,
    VISIBILITY_LOW,
    WRIST_TORSO_JUMP_MULT,
)

logger = logging.getLogger(__name__)


def _alpha(cutoff_hz: float, dt: float) -> float:
    """Exponential smoothing alpha from cutoff frequency and sample period (1€ filter)."""
    tau = 1.0 / (2.0 * math.pi * max(cutoff_hz, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-9))


class _OneEuro1D:
    """Single-axis One Euro Filter (Casiez et al., CHI 2012)."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: float | None = None
        self._t_prev: float | None = None
        self._dx_hat: float = 0.0

    def filter(
        self,
        x: float,
        t: float,
        min_cutoff: float | None = None,
        beta: float | None = None,
    ) -> float:
        mc = float(self.min_cutoff if min_cutoff is None else min_cutoff)
        b = float(self.beta if beta is None else beta)
        if self._t_prev is None:
            self._t_prev = t
            self._x_prev = x
            return x
        dt = max(t - self._t_prev, 1e-6)
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx_hat = a_d * dx + (1.0 - a_d) * self._dx_hat
        cutoff = mc + b * abs(self._dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._t_prev = t
        return x_hat


class _KalmanCV1D:
    """Constant-velocity Kalman filter on scalar observation (tracking-style second stage)."""

    def __init__(self, q: float, r: float):
        self.x = 0.0
        self.p = 1.0
        self.q = float(q)
        self.r = float(r)

    def update(self, z: float, r: float | None = None) -> float:
        r_eff = float(self.r if r is None else r)
        self.p = self.p + self.q
        k = self.p / (self.p + r_eff)
        self.x = self.x + k * (z - self.x)
        self.p = (1.0 - k) * self.p
        return self.x


def _joint_nx_ny_v(pose: dict, name: str) -> tuple[float, float, float]:
    for j in pose.get("joints", []):
        if j.get("name") == name:
            n = j.get("normalized") or {}
            return (
                float(n.get("x", 0.5)),
                float(n.get("y", 0.5)),
                float(j.get("visibility", 0.0)),
            )
    return 0.5, 0.5, 0.0


def _adaptive_filter_params(visibility: float) -> tuple[float, float, float, float]:
    """Visibility-driven 1€ + Kalman params: low v → less observation trust, less aggressive beta."""
    v = float(visibility)
    if v < VISIBILITY_LOW:
        return (
            ONE_EURO_MIN_CUTOFF_HZ,
            ONE_EURO_BETA * ONE_EURO_BETA_LOW_VISIBILITY_MULT,
            KALMAN_Q,
            KALMAN_R * KALMAN_R_LOW_VISIBILITY_MULT,
        )
    return (ONE_EURO_MIN_CUTOFF_HZ, ONE_EURO_BETA, KALMAN_Q, KALMAN_R)


def _torso_anchor(pose: dict) -> tuple[float, float, float, float]:
    """Shoulder mid and hip mid (normalized); falls back to 0.5 if shoulders weak."""
    lsx, lsy, lv = _joint_nx_ny_v(pose, "left_shoulder")
    rsx, rsy, rv = _joint_nx_ny_v(pose, "right_shoulder")
    lhx, lhy, _ = _joint_nx_ny_v(pose, "left_hip")
    rhx, rhy, _ = _joint_nx_ny_v(pose, "right_hip")
    if lv >= VISIBILITY_LOW and rv >= VISIBILITY_LOW:
        sx, sy = (lsx + rsx) / 2.0, (lsy + rsy) / 2.0
    else:
        sx, sy = 0.5, (lsy + rsy) / 2.0 if (lv + rv) > 0 else 0.5
    hx, hy = (lhx + rhx) / 2.0, (lhy + rhy) / 2.0
    return sx, sy, hx, hy


def _wrist_speed_peak_index(poses: list[dict]) -> int:
    """Index of max wrist speed (norm coords / s); 0 if too short."""
    n = len(poses)
    if n < 2:
        return 0
    best_i, best_sp = 0, -1.0
    for i in range(1, n):
        dt = float(poses[i]["timestamp"]) - float(poses[i - 1]["timestamp"])
        dt = max(dt, 1e-6)
        for name in ("left_wrist", "right_wrist"):
            x0, y0, _ = _joint_nx_ny_v(poses[i - 1], name)
            x1, y1, _ = _joint_nx_ny_v(poses[i], name)
            sp = math.hypot(x1 - x0, y1 - y0) / dt
            if sp > best_sp:
                best_sp = sp
                best_i = i
    return best_i


def _shoulder_scale(pose: dict) -> float:
    lx, _, lv = _joint_nx_ny_v(pose, "left_shoulder")
    rx, _, rv = _joint_nx_ny_v(pose, "right_shoulder")
    if lv < VISIBILITY_LOW or rv < VISIBILITY_LOW:
        return 1.0
    w = abs(rx - lx)
    return max(w, SHOULDER_WIDTH_MIN_NORM)


def compute_pose_quality_report(poses: list[dict]) -> dict[str, Any]:
    """Scalar quality metrics for gating (machine-readable)."""
    n = len(poses)
    if n == 0:
        return {
            "coverage_ratio": 0.0,
            "truncation_score": 1.0,
            "jitter_score": 0.0,
            "track_consistency": 0.0,
            "frame_count": 0,
        }

    cov_vals: list[float] = []
    trunc_hits = 0
    trunc_frames = 0
    for p in poses:
        vis = []
        for name in COVERAGE_JOINTS:
            _, _, v = _joint_nx_ny_v(p, name)
            vis.append(1.0 if v >= VISIBILITY_LOW else 0.0)
        cov_vals.append(float(np.mean(vis)) if vis else 0.0)
        lo = TRUNCATION_MARGIN
        hi = 1.0 - TRUNCATION_MARGIN
        bad = False
        for name in TRUNCATION_BODY_PARTS:
            nx, ny, v = _joint_nx_ny_v(p, name)
            if v < VISIBILITY_LOW:
                continue
            if nx < lo or nx > hi or ny < lo or ny > hi:
                bad = True
                break
        trunc_frames += 1
        if bad:
            trunc_hits += 1

    coverage_ratio = float(np.mean(cov_vals)) if cov_vals else 0.0
    truncation_score = float(trunc_hits / max(trunc_frames, 1))

    # Jitter: median over frames of max joint speed (normalized / second)
    speeds: list[float] = []
    for i in range(1, n):
        dt = float(poses[i]["timestamp"]) - float(poses[i - 1]["timestamp"])
        dt = max(dt, 1e-6)
        mx = 0.0
        for name in JITTER_JOINTS:
            x0, y0, _ = _joint_nx_ny_v(poses[i - 1], name)
            x1, y1, _ = _joint_nx_ny_v(poses[i], name)
            sp = math.hypot(x1 - x0, y1 - y0) / dt
            mx = max(mx, sp)
        speeds.append(mx)
    jitter_score = float(np.median(speeds)) if speeds else 0.0

    # Track consistency: penalize single-frame norm jumps (proxy for ID / tracking breaks)
    jumps = 0
    for i in range(1, n):
        for name in ("right_wrist", "left_wrist"):
            x0, y0, _ = _joint_nx_ny_v(poses[i - 1], name)
            x1, y1, _ = _joint_nx_ny_v(poses[i], name)
            if math.hypot(x1 - x0, y1 - y0) > TRACK_JUMP_THRESH_NORM:
                jumps += 1
                break
    track_consistency = 1.0 - min(1.0, jumps / max(n - 1, 1))

    return {
        "coverage_ratio": round(coverage_ratio, 4),
        "truncation_score": round(truncation_score, 4),
        "jitter_score": round(jitter_score, 4),
        "track_consistency": round(track_consistency, 4),
        "frame_count": n,
    }


def assess_pose_reliability(pre_report: dict, post_report: dict) -> tuple[str, list[str]]:
    """Return (high|medium|low, reason_codes). Low => strict phase-AI must FAIL."""
    reasons: list[str] = []
    cov_pre = float(pre_report.get("coverage_ratio", 0.0))
    tr_pre = float(pre_report.get("truncation_score", 1.0))
    jit_pre = float(pre_report.get("jitter_score", 0.0))
    tc_pre = float(pre_report.get("track_consistency", 0.0))
    jit_post = float(post_report.get("jitter_score", jit_pre))
    tc_post = float(post_report.get("track_consistency", tc_pre))

    if cov_pre < COVERAGE_LOW_FAIL:
        reasons.append("COVERAGE_TOO_LOW")
    if tr_pre > TRUNCATION_SCORE_HIGH:
        reasons.append("TRUNCATION_EXCESSIVE")
    if tc_pre < TRACK_CONSISTENCY_LOW and tc_post < TRACK_CONSISTENCY_LOW:
        reasons.append("TRACK_INCONSISTENT")
    if jit_pre > JITTER_SCORE_HIGH and jit_post > JITTER_SCORE_HIGH:
        reasons.append("JITTER_EXCESSIVE")

    hard = {"COVERAGE_TOO_LOW", "TRUNCATION_EXCESSIVE", "TRACK_INCONSISTENT", "JITTER_EXCESSIVE"}
    if reasons and hard.intersection(reasons):
        return "low", reasons

    if cov_pre >= COVERAGE_HIGH and tr_pre <= 0.12 and jit_post <= JITTER_SCORE_HIGH * 0.55 and tc_post >= 0.82:
        return "high", reasons

    if cov_pre < COVERAGE_HIGH or tr_pre > 0.12 or jit_post > JITTER_SCORE_HIGH * 0.75 or tc_post < 0.72:
        return "medium", reasons

    return "high", reasons


def _interpolate_short_gaps(poses: list[dict]) -> None:
    """Linear interp nx,ny for low-visibility runs <= K frames; in-place."""
    if len(poses) < 2:
        return
    names = list({j["name"] for p in poses for j in p.get("joints", [])})
    for name in names:
        vseq = [_joint_nx_ny_v(p, name)[2] for p in poses]
        i = 0
        while i < len(vseq):
            if vseq[i] >= VISIBILITY_LOW:
                i += 1
                continue
            j = i
            while j < len(vseq) and vseq[j] < VISIBILITY_LOW:
                j += 1
            gap = j - i
            if gap <= 0 or gap > OCCLUSION_MAX_GAP_FRAMES:
                i = j if j > i else i + 1
                continue
            i0 = i - 1
            j0 = j
            if i0 < 0 or j0 >= len(poses):
                i = j
                continue
            p_a = poses[i0]
            p_b = poses[j0]
            xa, ya, _ = _joint_nx_ny_v(p_a, name)
            xb, yb, _ = _joint_nx_ny_v(p_b, name)
            for k in range(i, j):
                t = (k - i0) / max(j0 - i0, 1)
                nx = xa + (xb - xa) * t
                ny = ya + (yb - ya) * t
                for joint in poses[k].get("joints", []):
                    if joint.get("name") != name:
                        continue
                    joint.setdefault("normalized", {})
                    joint["normalized"]["x"] = round(nx, 4)
                    joint["normalized"]["y"] = round(ny, 4)
                    joint["visibility"] = round(
                        min(1.0, float(joint.get("visibility", 0.0)) + 0.15), 3
                    )
                    fs = poses[k].get("frame_size") or {}
                    w = int(fs.get("width", 1))
                    h = int(fs.get("height", 1))
                    joint["x"] = round(nx * w, 2)
                    joint["y"] = round(ny * h, 2)
            i = j


def _apply_wrist_torso_clamp(poses: list[dict]) -> int:
    """Limit low-visibility wrist jumps relative to torso anchor; returns clamp event count."""
    clamp_lim = TRACK_JUMP_THRESH_NORM * WRIST_TORSO_JUMP_MULT
    clamps = 0
    prev_rel: dict[str, tuple[float, float]] = {}
    for pose in poses:
        sx, sy, _, _ = _torso_anchor(pose)
        fs = pose.get("frame_size") or {}
        w = int(fs.get("width", 1))
        h = int(fs.get("height", 1))
        for name in ("left_wrist", "right_wrist"):
            wx, wy, vis = _joint_nx_ny_v(pose, name)
            rx, ry = wx - sx, wy - sy
            if name in prev_rel and vis < VISIBILITY_LOW:
                prx, pry = prev_rel[name]
                d = math.hypot(rx - prx, ry - pry)
                if d > clamp_lim and d > 1e-9:
                    t = clamp_lim / d
                    rx = prx + (rx - prx) * t
                    ry = pry + (ry - pry) * t
                    clamps += 1
                    wx, wy = rx + sx, ry + sy
            prev_rel[name] = (rx, ry)
            for joint in pose.get("joints", []):
                if joint.get("name") != name:
                    continue
                joint.setdefault("normalized", {})
                joint["normalized"]["x"] = round(max(0.0, min(1.0, wx)), 4)
                joint["normalized"]["y"] = round(max(0.0, min(1.0, wy)), 4)
                joint["x"] = round(wx * w, 2)
                joint["y"] = round(wy * h, 2)
                break
    return clamps


def _apply_filters_and_reproject(poses: list[dict], fps: float) -> None:
    """1€ + Kalman (visibility-adaptive), reproject + angles (wrist–torso clamp runs before this)."""
    from services.pose_service import compute_golf_angles, compute_golf_angles_3d

    if len(poses) < 2:
        return
    dt = 1.0 / max(fps, 1e-6)
    names = list({j["name"] for p in poses for j in p.get("joints", [])})
    filters: dict[tuple[str, str], tuple[_OneEuro1D, _KalmanCV1D]] = {}
    for name in names:
        filters[(name, "x")] = (
            _OneEuro1D(ONE_EURO_MIN_CUTOFF_HZ, ONE_EURO_BETA, ONE_EURO_D_CUTOFF_HZ),
            _KalmanCV1D(KALMAN_Q, KALMAN_R),
        )
        filters[(name, "y")] = (
            _OneEuro1D(ONE_EURO_MIN_CUTOFF_HZ, ONE_EURO_BETA, ONE_EURO_D_CUTOFF_HZ),
            _KalmanCV1D(KALMAN_Q, KALMAN_R),
        )

    for pi, pose in enumerate(poses):
        t = float(pose.get("timestamp", pi * dt))
        sc = _shoulder_scale(pose)
        fs = pose.get("frame_size") or {}
        w = int(fs.get("width", 1))
        h = int(fs.get("height", 1))
        for joint in pose.get("joints", []):
            name = str(joint.get("name", ""))
            if name not in names:
                continue
            vis = float(joint.get("visibility", 0.0))
            mc, beta, _q, r_k = _adaptive_filter_params(vis)
            n = joint.get("normalized") or {}
            raw_x = float(n.get("x", 0.5))
            raw_y = float(n.get("y", 0.5))
            sx = raw_x / sc
            sy = raw_y / sc
            fx, kx = filters[(name, "x")]
            fy, ky = filters[(name, "y")]
            sx2 = kx.update(fx.filter(sx, t, min_cutoff=mc, beta=beta), r=r_k)
            sy2 = ky.update(fy.filter(sy, t, min_cutoff=mc, beta=beta), r=r_k)
            nx = max(0.0, min(1.0, sx2 * sc))
            ny = max(0.0, min(1.0, sy2 * sc))
            joint.setdefault("normalized", {})
            joint["normalized"]["x"] = round(nx, 4)
            joint["normalized"]["y"] = round(ny, 4)
            joint["x"] = round(nx * w, 2)
            joint["y"] = round(ny * h, 2)

        wl = pose.get("world_landmarks")
        if wl and isinstance(wl, dict) and len(wl) >= 10:
            pose["angles"] = compute_golf_angles_3d(wl)
        else:
            pose["angles"] = compute_golf_angles(pose.get("joints", []))

    for pose in poses:
        wl = pose.get("world_landmarks")
        if wl and isinstance(wl, dict) and len(wl) >= 10:
            pose["angles"] = compute_golf_angles_3d(wl)
        else:
            pose["angles"] = compute_golf_angles(pose.get("joints", []))
    return 0


def empty_pose_quality_bundle(reason_code: str) -> dict[str, Any]:
    z = compute_pose_quality_report([])
    z["wrist_jump_clamped_count"] = 0
    z["smoothing_lag_score"] = 0.0
    return {
        "pose_quality_report": z,
        "pose_quality_report_post": dict(z),
        "pose_reliability_level": "low",
        "reliability_reason_codes": [reason_code],
        "refine_applied": False,
    }


def refine_pose_sequence_pipeline(poses: list[dict], fps: float) -> dict[str, Any]:
    """
    In-place refinement on ``poses``. Returns bundle for routers / gates.
    """
    if len(poses) < 2:
        pre = compute_pose_quality_report(poses)
        pre["wrist_jump_clamped_count"] = 0
        pre["smoothing_lag_score"] = 0.0
        rel, codes = assess_pose_reliability(pre, pre)
        return {
            "pose_quality_report": pre,
            "pose_quality_report_post": dict(pre),
            "pose_reliability_level": rel,
            "reliability_reason_codes": codes,
            "refine_applied": False,
        }

    raw_joints_snapshot = [copy.deepcopy(p.get("joints", [])) for p in poses]
    pre = compute_pose_quality_report(poses)
    pre_cov = float(pre.get("coverage_ratio", 0.0))
    pre_jit = float(pre.get("jitter_score", 0.0))
    pre_tc = float(pre.get("track_consistency", 0.0))
    if pre_cov >= COVERAGE_HIGH and pre_jit <= JITTER_SCORE_HIGH * 0.45 and pre_tc >= 0.92:
        logger.info(
            "[pose_refine] skip refine on high-quality track pre_cov=%.3f pre_jit=%.3f pre_tc=%.3f",
            pre_cov,
            pre_jit,
            pre_tc,
        )
        pre["wrist_jump_clamped_count"] = 0
        pre["smoothing_lag_score"] = 0.0
        rel, codes = assess_pose_reliability(pre, pre)
        return {
            "pose_quality_report": pre,
            "pose_quality_report_post": dict(pre),
            "pose_reliability_level": rel,
            "reliability_reason_codes": codes,
            "refine_applied": False,
            "refine_rollback_to_raw": True,
            "refine_skip_reason": "high_quality_track",
        }
    peak_pre = _wrist_speed_peak_index(poses)
    _interpolate_short_gaps(poses)
    # Clamp wrist–torso jumps before temporal filters, which otherwise suppress low-vis spikes.
    n_clamp = _apply_wrist_torso_clamp(poses)
    _apply_filters_and_reproject(poses, fps)
    post = compute_pose_quality_report(poses)
    post["wrist_jump_clamped_count"] = int(n_clamp)
    peak_post = _wrist_speed_peak_index(poses)
    nlen = len(poses)
    sls = abs(peak_pre - peak_post) / max(nlen, 1)
    post["smoothing_lag_score"] = round(float(sls), 4)
    rel, codes = assess_pose_reliability(pre, post)
    codes = list(codes)
    if sls > SMOOTHING_LAG_HIGH_THRESH:
        codes.append("SMOOTHING_LAG_HIGH")

    logger.info(
        "[pose_refine] pre_cov=%.3f pre_jit=%.3f post_jit=%.3f lag=%.3f clamps=%d rel=%s codes=%s",
        pre["coverage_ratio"],
        pre["jitter_score"],
        post["jitter_score"],
        sls,
        n_clamp,
        rel,
        codes,
    )

    if float(post.get("jitter_score", pre_jit)) > pre_jit * 1.25:
        for i, pose in enumerate(poses):
            pose["joints"] = copy.deepcopy(raw_joints_snapshot[i])
        rollback_post = compute_pose_quality_report(poses)
        rollback_post["wrist_jump_clamped_count"] = int(n_clamp)
        rollback_post["smoothing_lag_score"] = round(float(sls), 4)
        rel_rb, codes_rb = assess_pose_reliability(pre, rollback_post)
        codes_rb = list(codes_rb) + ["REFINE_ROLLBACK_JITTER_REGRESSION"]
        logger.warning(
            "[pose_refine] rollback to raw joints post_jit %.3f > pre_jit*1.25 %.3f",
            float(post.get("jitter_score", pre_jit)),
            pre_jit * 1.25,
        )
        return {
            "pose_quality_report": pre,
            "pose_quality_report_post": rollback_post,
            "pose_reliability_level": rel_rb,
            "reliability_reason_codes": codes_rb,
            "refine_applied": False,
            "refine_rollback_to_raw": True,
            "refine_skip_reason": "jitter_regression",
        }

    return {
        "pose_quality_report": pre,
        "pose_quality_report_post": post,
        "pose_reliability_level": rel,
        "reliability_reason_codes": codes,
        "refine_applied": True,
        "refine_rollback_to_raw": False,
    }
