"""Pro v3 — Gemini motion-metadata report (single vision call; local fallback if weak)."""

from __future__ import annotations

import logging
from typing import Any, Optional

from services.gemini_service import analyze_prov3_motion_report_only

logger = logging.getLogger(__name__)

# Minimum bar after pass 1 / pass 2 (aligned with product expectations for screen mode).
_MIN_SUMMARY_EN_WORDS = 280
_MIN_SUMMARY_ZH_CHARS = 360
_MIN_LIST_ITEMS = 3

# Limited mode: slightly lower prose bar; quality judged by length + lists, not a fixed disclaimer phrase.
_MIN_SUMMARY_EN_WORDS_LIMITED = 200
_MIN_SUMMARY_ZH_CHARS_LIMITED = 280

_META_KEY = "__prov3_report_meta__"


def _pad_club_images_b64(raw: Optional[list[str]]) -> Optional[list[str]]:
    if not raw:
        return None
    xs = [x for x in raw if isinstance(x, str) and x.strip()][:3]
    if not xs:
        return None
    while len(xs) < 3:
        xs.append(xs[-1])
    return xs


def _nonempty_str_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if str(x).strip()]


def _report_is_weak_limited(out: dict[str, Any]) -> bool:
    summary_en = str(out.get("summary") or "").strip()
    summary_zh = str(out.get("summary_zh") or "").strip()
    if not summary_en or not summary_zh:
        return True
    en_words = len(summary_en.split())
    zh_chars = len(summary_zh.replace(" ", ""))
    if en_words < _MIN_SUMMARY_EN_WORDS_LIMITED or zh_chars < _MIN_SUMMARY_ZH_CHARS_LIMITED:
        return True
    issues = _nonempty_str_list(out.get("issues"))
    issues_zh = _nonempty_str_list(out.get("issues_zh"))
    sug = _nonempty_str_list(out.get("suggestions"))
    sug_zh = _nonempty_str_list(out.get("suggestions_zh"))
    if len(issues) < _MIN_LIST_ITEMS or len(issues_zh) < _MIN_LIST_ITEMS:
        return True
    if len(sug) < _MIN_LIST_ITEMS or len(sug_zh) < _MIN_LIST_ITEMS:
        return True
    return False


def _report_is_weak(out: dict[str, Any]) -> bool:
    summary_en = str(out.get("summary") or "").strip()
    summary_zh = str(out.get("summary_zh") or "").strip()
    if not summary_en or not summary_zh:
        return True
    en_words = len(summary_en.split())
    zh_chars = len(summary_zh.replace(" ", ""))
    if en_words < _MIN_SUMMARY_EN_WORDS or zh_chars < _MIN_SUMMARY_ZH_CHARS:
        return True
    issues = _nonempty_str_list(out.get("issues"))
    issues_zh = _nonempty_str_list(out.get("issues_zh"))
    sug = _nonempty_str_list(out.get("suggestions"))
    sug_zh = _nonempty_str_list(out.get("suggestions_zh"))
    if len(issues) < _MIN_LIST_ITEMS or len(issues_zh) < _MIN_LIST_ITEMS:
        return True
    if len(sug) < _MIN_LIST_ITEMS or len(sug_zh) < _MIN_LIST_ITEMS:
        return True
    return False


def _kf_by_phase(motion_context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = motion_context.get("keyframes") or []
    by: dict[str, dict[str, Any]] = {}
    if not isinstance(rows, list):
        return by
    for r in rows:
        if not isinstance(r, dict):
            continue
        p = str(r.get("phase") or "").strip()
        if p:
            by[p] = r
    return by


def _synthetic_keyframe_evaluations(
    motion_context: dict[str, Any],
    *,
    low_trust: bool = False,
) -> list[dict[str, Any]]:
    """Ordered per-row scores/text when Gemini is unavailable (metadata only)."""
    rows = motion_context.get("keyframes") or []
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    lt_en = " Phase labels follow the pipeline; read proxy and spacing only." if low_trust else ""
    lt_zh = " 阶段标签以管道为准，请结合能量代理与时间间隔理解。" if low_trust else ""
    for r in rows:
        if not isinstance(r, dict):
            continue
        ph = str(r.get("phase") or "").strip()
        if not ph:
            continue
        try:
            proxy = float(r.get("dense_motion_proxy") or 0.0)
        except (TypeError, ValueError):
            proxy = 0.0
        out.append(
            {
                "phase": ph,
                "score": int(max(35, min(78, 48 + min(20, proxy * 12)))),
                "action_assessment_en": (
                    f"Metadata-only fallback: dense_motion_proxy≈{proxy:.4f} at this phase; no vision.{lt_en}"
                ),
                "action_assessment_zh": (
                    f"本地 metadata 兜底：该阶段 dense_motion_proxy 约 {proxy:.4f}，无画面。{lt_zh}"
                ),
            }
        )
    return out


def build_prov3_fallback_report(motion_context: dict[str, Any]) -> dict[str, Any]:
    """Deterministic coaching-shaped report when AI fails or returns thin output."""
    by = _kf_by_phase(motion_context)
    swing = motion_context.get("swing_window_s") or [0.0, 0.0]
    try:
        w0, w1 = float(swing[0]), float(swing[1])
    except (TypeError, ValueError, IndexError):
        w0, w1 = 0.0, 0.0
    win_dur = max(0.0, w1 - w0)
    fps = float(motion_context.get("fps") or 240.0)

    def _proxy(phase_key: str) -> float:
        row = by.get(phase_key) or {}
        try:
            return float(row.get("dense_motion_proxy") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _ts(phase_key: str) -> float:
        row = by.get(phase_key) or {}
        try:
            return float(row.get("timestamp_s") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    p_imp = _proxy("impact")
    p_ft = _proxy("follow_through")
    p_fin = _proxy("finish")
    p_ds = _proxy("downswing")
    t_imp = _ts("impact")
    t_ft = _ts("follow_through")
    t_fin = _ts("finish")
    gap_if = max(0.0, t_ft - t_imp)
    gap_ff = max(0.0, t_fin - t_ft)

    # Heuristic flags (screen-safe; no vision claims).
    compressed_follow = gap_if < (4.5 / max(fps, 1.0))
    weak_finish_energy = p_fin < max(p_imp * 0.38, 1e-6) and p_imp > 1e-6
    strike_burst_mismatch = p_ds > 1e-6 and p_imp < p_ds * 0.55

    issues_en: list[str] = []
    issues_zh: list[str] = []
    if strike_burst_mismatch:
        issues_en.append(
            "Impact: strike timing appears unstable relative to the main downswing motion burst "
            "(impact proxy is low vs downswing proxy in the motion summary)."
        )
        issues_zh.append(
            "触球：根据当前 motion summary，下杆段能量偏高而触球处代理值相对偏低，触球时机相对主下杆爆发显得不够稳定。"
        )
    if compressed_follow:
        issues_en.append(
            "Follow-through: post-impact release looks compressed — very little time between Impact "
            "and Follow-through timestamps in the captured swing window."
        )
        issues_zh.append(
            "送杆：触球与送杆关键帧之间的时间间隔偏短，送杆释放可能在时间线上显得偏“挤压”。"
        )
    if weak_finish_energy:
        issues_en.append(
            "Finish: exit-phase motion proxy stays low vs Impact — the finish frame may not yet show "
            "a fully settled completion within this clip."
        )
        issues_zh.append(
            "收杆：收杆相对触球的 dense motion 代理值明显偏低，收杆画面在片段内可能尚未充分稳定完成。"
        )
    if win_dur > 0 and win_dur < (12.0 / max(fps, 1.0)):
        issues_en.append(
            "Downswing: the detected swing window is short — ensure the clip includes a full backswing "
            "and finish for cleaner phase spacing in future captures."
        )
        issues_zh.append(
            "下杆：检测到的挥杆窗口偏短，建议下次录制包含更完整的上杆与收杆，以便阶段间隔更可靠。"
        )

    while len(issues_en) < 4:
        issues_en.append(
            "Top: based on the current motion summary only, maintain width and tempo into the transition "
            "so Downswing can sequence without rushing."
        )
        issues_zh.append(
            "顶点：根据当前 motion summary，建议保持上杆幅度与节奏，避免过渡阶段过急影响下杆顺序。"
        )

    suggestions_en = [
        "Impact: hit short pitch shots focusing on crisp contact timing; use a metronome cadence through impact.",
        "Follow-through: rehearse full extension drills stopping only after chest faces target for two seconds.",
        "Finish: hold finish for three seconds each rep to verify balance and complete release.",
        "Downswing: pump drill — shallow takeaway then smooth transition, three reps without a ball then one with.",
    ]
    suggestions_zh = [
        "触球：短切练习强调干脆触球时机，可配合节拍器稳定通过触球点。",
        "送杆：做送杆充分伸展练习，胸朝向目标后停两秒再收。",
        "收杆：每一杆收杆后停三秒，检查重心与释放是否完整。",
        "下杆：无球三下“泵式”过渡再上杆击球，减少过渡抢快。",
    ]

    score_base = 58
    if compressed_follow:
        score_base -= 4
    if weak_finish_energy:
        score_base -= 3
    if strike_burst_mismatch:
        score_base -= 4
    total_score = int(max(48, min(72, score_base)))

    summary_en = (
        f"Based on the current motion summary (Pro v3 dense scan at approximately {fps:.0f} fps), this report is generated "
        f"without any image selection or frame picking. The extracted swing window spans about {win_dur:.3f} seconds "
        "between pipeline boundaries, which anchors every phase timestamp you see in the structured context.\n\n"
        "Address: use a stable setup reference — ball position, posture, and grip pressure should stay repeatable "
        "across reps because the later timing story is read from motion energy, not from pixels in this mode.\n\n"
        "Takeaway / Backswing: the early chain should build width without snatching; if the window is clipped, "
        "the first phases may crowd together, so favor slightly longer captures next time.\n\n"
        "Top: pause the feeling of completion before starting down; the metadata cannot see your face, "
        "but it can show whether energy rises into the strike band in an orderly way.\n\n"
        "Downswing / Transition: the motion proxy along the downswing segment should relate cleanly to Impact; "
        f"{'here the impact proxy trails the downswing burst, so prioritize sequencing, shallowing feels, and low-point control with mid-irons before speed.' if strike_burst_mismatch else 'keep the transition smooth so the handle leads without flipping early.'}\n\n"
        "Impact: "
        f"{'strike timing relative to the main burst looks inconsistent — rehearse half-swings with a crisp strike thought and a steady cadence so peak motion and contact line up.' if strike_burst_mismatch else 'treat the impact timestamp as your contact checkpoint; add small pitch swings to validate crispness without overswinging.'}\n\n"
        "Follow-through: "
        f"{'post-impact spacing on the timeline looks tight — let the club exit longer along the line, keep the chest rotating, and avoid folding the elbows immediately after contact.' if compressed_follow else 'extend through the ball, let the lead arm stretch, and keep turning so release is a rotation story, not a hands-only flip.'}\n\n"
        "Finish: "
        f"{'exit-phase proxy stays soft versus impact — hold a balanced finish, check weight on the lead side, and verify the belt buckle faces short of target before you walk out of the pose.' if weak_finish_energy else 'settle into a complete pose with most weight forward and the shaft wrapping comfortably around the neck — consistency matters more than flash.'}\n\n"
        "Practice design: alternate ball and no-ball reps, film from face-on and down-the-line when possible, "
        "and compare spacing between Impact, Follow-through, and Finish across sessions. "
        "This fallback report stays conservative: it does not invent ball flight, start line, or clubface aim. "
        "If the swing window was very short, re-record with a wider clip so each phase can breathe on the timeline."
    )
    summary_zh = (
        f"根据当前 motion summary（Pro v3 密集扫描，约 {fps:.0f} fps），本报告不依赖任何选图或附图，也不进行 AI 选帧。"
        f"管道给出的挥杆窗口约 {win_dur:.3f} 秒，这一边界用于对齐八个阶段时间戳。\n\n"
        "站姿：以可重复的站位与握压为基准；屏幕模式主要读的是能量与时间线，而不是画面细节，因此准备姿势的稳定性会直接反映在后续阶段间隔上。\n\n"
        "起杆 / 上杆：早段应追求宽度与顺序，避免“抢”；若窗口偏短，前段阶段可能在时间线上显得拥挤，建议下次录制留足上杆空间。\n\n"
        "顶点：在上杆顶点保持清晰的完成感再启动下杆；metadata 看不到表情，但能反映进入击球带前能量是否有序爬升。\n\n"
        "下杆 / 过渡：下杆段与触球的代理关系应协调；"
        f"{'当前触球相对下杆峰值偏弱，建议先用中铁杆练习顺序、浅下杆与最低点控制，再逐步加快速度。' if strike_burst_mismatch else '保持过渡平顺，让杆身与身体转动协同，避免过早甩腕。'}\n\n"
        "触球："
        f"{'从数据看触球与主爆发对齐度一般，可用半挥杆与固定节拍巩固触球点。' if strike_burst_mismatch else '以触球时间戳为检查点，配合短切验证干脆触球。'}\n\n"
        "送杆："
        f"{'触球后间隔偏短，建议沿目标线更充分送出，让胸部持续转动，避免触球后立刻收臂。' if compressed_follow else '通过触球后继续伸展与转动完成释放，而不是只用手臂甩动。'}\n\n"
        "收杆："
        f"{'收杆能量相对触球偏弱，建议完成收杆并检查重心是否留在前脚一侧、骨盆是否转足。' if weak_finish_energy else '收杆应稳定停住，重心前移，杆身自然环绕，重复性优先于幅度。'}\n\n"
        "训练安排建议：有球与无球交替，尽量拍摄正面与身后视角，对比多次录制中触球、送杆、收杆的间隔变化。"
        "本段为本地保底报告，不臆测球路、起始方向或杆面瞄准；若窗口过短或片段不完整，请换更长、更清晰的挥杆视频重试。"
    )

    sub = max(40, total_score - 10)
    scores = {
        "grip": sub,
        "stance": sub + 2,
        "backswing": sub + 4,
        "downswing": sub + 3,
        "follow_through": sub + 1,
    }
    training_plan = {
        f"day{i}": {
            "focus": suggestions_zh[(i - 1) % 4][:80],
            "drills": [
                suggestions_zh[(i - 1) % 4][:120],
                suggestions_zh[i % 4][:120],
            ],
            "duration": "20–30 min",
        }
        for i in range(1, 8)
    }
    training_plan["day7"] = {
        "focus": "复习与录像对比",
        "drills": [
            "对照本次触球/送杆/收杆要点回看录像",
            "用正面与身后视角各录一杆，对比阶段间隔是否更舒展",
        ],
        "duration": "20 min",
    }

    return {
        "total_score": total_score,
        "scores": scores,
        "issues": issues_en[:6],
        "issues_zh": issues_zh[:6],
        "suggestions": suggestions_en[:6],
        "suggestions_zh": suggestions_zh[:6],
        "summary": summary_en,
        "summary_zh": summary_zh,
        "keyframe_evaluations": _synthetic_keyframe_evaluations(motion_context, low_trust=False),
        "training_plan": training_plan,
        "ai_provider": "prov3_fallback",
    }


def build_prov3_limited_fallback(motion_context: dict[str, Any]) -> dict[str, Any]:
    """Motion-based report when limited-mode Gemini is weak; one short caveat then substantive coaching."""
    fb = build_prov3_fallback_report(motion_context)
    notice_zh = (
        "【提示】自动关键帧未达最高置信档；下文根据当前挥杆时间线与能量代理撰写，不臆测杆面与球路。"
        "建议尽量真机直拍、全身入镜、光线稳定。"
    )
    notice_en = (
        "[Note] Keyframe confidence is below the studio bar. "
        "The coaching below follows the motion timeline and energy proxies only — no face-angle or ball-flight claims. "
        "Prefer direct camera capture, full body in frame, stable lighting."
    )
    fb["summary_zh"] = notice_zh + "\n\n" + str(fb.get("summary_zh") or "").strip()
    fb["summary"] = notice_en + "\n\n" + str(fb.get("summary") or "").strip()
    fb["ai_provider"] = "prov3_limited_fallback"
    try:
        ts = int(fb.get("total_score") or 58)
    except (TypeError, ValueError):
        ts = 58
    fb["total_score"] = int(max(45, min(72, ts - 5)))
    fb["keyframe_evaluations"] = _synthetic_keyframe_evaluations(motion_context, low_trust=True)
    return fb


async def write_prov3_ai_report(
    motion_context: dict[str, Any],
    *,
    region: str = "global",
    report_mode: str = "formal",
    club_images_b64: Optional[list[str]] = None,
) -> dict[str, Any]:
    """One Gemini vision call (optional 3 club JPEGs in-call); local fallback if output is weak."""
    lim_lbl, p1_lbl = (
        "prov3_report_limited",
        "prov3_report",
    )
    ct = "PROV3"
    meta: dict[str, Any] = {
        "pass1_weak": False,
        "pass2_used": False,
        "pass2_weak": False,
        "fallback_used": False,
        "report_chain": "prov3",
    }
    club_kw = _pad_club_images_b64(club_images_b64)

    if (report_mode or "").strip().lower() == "limited":
        logger.info(
            "[%s][REPORT_MODE] report_mode=limited pass=single chain=%s",
            ct,
            meta["report_chain"],
        )
        out1 = await analyze_prov3_motion_report_only(
            motion_context,
            region=region,
            use_strong_prompt=False,
            max_tokens=12288,
            call_label=lim_lbl,
            report_mode="limited",
            club_images_b64=club_kw,
        )
        weak1 = _report_is_weak_limited(out1)
        meta["pass1_weak"] = weak1
        chosen = build_prov3_limited_fallback(motion_context) if weak1 else out1
        if weak1:
            meta["fallback_used"] = True
            logger.warning("[%s][REPORT_MODE] limited_fallback_used=true", ct)
        else:
            logger.info("[%s][REPORT_MODE] limited_fallback_used=false", ct)
        summary_en = str(chosen.get("summary") or "").strip()
        summary_zh = str(chosen.get("summary_zh") or "").strip()
        logger.info(
            "[%s][REPORT] total_score=%s provider=%s en_words=%s zh_chars=%s mode=limited",
            ct,
            chosen.get("total_score"),
            chosen.get("ai_provider"),
            len(summary_en.split()),
            len(summary_zh.replace(" ", "")),
        )
        chosen[_META_KEY] = meta
        return chosen

    logger.info("[%s][REPORT] single_pass_started report_mode=formal chain=%s", ct, meta["report_chain"])
    out1 = await analyze_prov3_motion_report_only(
        motion_context,
        region=region,
        use_strong_prompt=False,
        max_tokens=12288,
        call_label=p1_lbl,
        report_mode="formal",
        club_images_b64=club_kw,
    )
    weak1 = _report_is_weak(out1)
    meta["pass1_weak"] = weak1
    logger.info("[%s][REPORT] pass1_weak=%s (no pass2 — single Gemini)", ct, weak1)

    chosen = out1
    if _report_is_weak(chosen):
        fb = build_prov3_fallback_report(motion_context)
        meta["fallback_used"] = True
        logger.warning("[%s][REPORT] fallback_used=true (single-pass output weak)", ct)
        chosen = fb
    else:
        logger.info("[%s][REPORT] fallback_used=false", ct)

    summary_en = str(chosen.get("summary") or "").strip()
    summary_zh = str(chosen.get("summary_zh") or "").strip()
    logger.info(
        "[%s][REPORT] total_score=%s provider=%s en_words=%s zh_chars=%s",
        ct,
        chosen.get("total_score"),
        chosen.get("ai_provider"),
        len(summary_en.split()),
        len(summary_zh.replace(" ", "")),
    )

    chosen[_META_KEY] = meta
    return chosen


def pop_prov3_report_meta(report: dict[str, Any]) -> dict[str, Any]:
    """Remove internal meta from report dict; returns meta for logging."""
    raw = report.pop(_META_KEY, None)
    return raw if isinstance(raw, dict) else {}
