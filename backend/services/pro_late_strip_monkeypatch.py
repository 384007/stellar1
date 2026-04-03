from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_pro_late_strip_fix_applied"

_LATE_STRIP_PHASES = ("downswing", "impact", "follow_through", "finish")


def _remaining_phases_from_validation(kf_validation: dict[str, Any]) -> list[str]:
    fv = dict(kf_validation.get("final_keyframe_validation") or {})
    rem = list(fv.get("remaining_near_duplicate_phases") or [])
    if rem:
        return [str(x) for x in rem]
    details = kf_validation.get("details") or []
    out: list[str] = []
    for d in details:
        if not isinstance(d, dict):
            continue
        ph = str(d.get("phase") or "")
        if ph in _LATE_STRIP_PHASES and (d.get("is_near_duplicate") or d.get("time_too_close")):
            out.append(ph)
    return out


def _should_run_cleanup(
    keyframes_out: list[dict],
    poses: list[dict],
    gate_pass: bool,
    near_dup: int,
    time_tc: int,
    remaining: list[str],
) -> bool:
    if len(keyframes_out) != 8 or not poses:
        return False
    has_issue = (not gate_pass) or near_dup > 0 or time_tc > 0 or len(remaining) > 0
    return bool(has_issue)


def _cleanup_accepted(
    *,
    before_gate: bool,
    before_dup: int,
    before_tc: int,
    before_remaining: list[str],
    new_validation: dict[str, Any],
) -> bool:
    after_gate = bool(new_validation.get("final_keyframe_gate_pass", False))
    after_dup = int(new_validation.get("near_duplicates", 0))
    after_tc = int(new_validation.get("time_too_close", 0))
    after_remaining = _remaining_phases_from_validation(new_validation)
    if after_gate:
        return True
    if len(after_remaining) < len(before_remaining):
        return True
    if after_dup < before_dup:
        return True
    if after_tc < before_tc:
        return True
    return False


def apply_pro_late_strip_fix_patch() -> None:
    import services.keyframe_service as keyframe_service

    if getattr(keyframe_service, _PATCH_FLAG, False):
        return

    from services.pro_late_strip_cleanup_service import cleanup_pro_late_strip_duplicates

    original_ensure = keyframe_service.ensure_keyframes_ordered_for_ai

    def patched_ensure_keyframes_ordered_for_ai(
        video_path: str,
        poses: list[dict],
        swing_phases: list[dict],
        phase_keyframes_ordered_snapshot: dict[str, int],
        keyframes: list[dict],
        kf_validation: dict,
        phase_keyframes: dict[str, int],
        keyframe_width: int = 320,
        **kwargs: Any,
    ):
        keyframes_out, kf_validation_out, phase_keyframes_out, final_source = original_ensure(
            video_path,
            poses,
            swing_phases,
            phase_keyframes_ordered_snapshot,
            keyframes,
            kf_validation,
            phase_keyframes,
            keyframe_width,
            **kwargs,
        )
        try:
            before_gate = bool(kf_validation_out.get("final_keyframe_gate_pass", False))
            before_dup = int(kf_validation_out.get("near_duplicates", 0))
            before_tc = int(kf_validation_out.get("time_too_close", 0))
            before_remaining = _remaining_phases_from_validation(kf_validation_out)

            if not _should_run_cleanup(
                keyframes_out,
                poses,
                before_gate,
                before_dup,
                before_tc,
                before_remaining,
            ):
                return keyframes_out, kf_validation_out, phase_keyframes_out, final_source

            logger.info(
                "[keyframe] monkeypatch enter late-strip cleanup gate_pass=%s near_duplicates=%s "
                "time_too_close=%s remaining=%s source=%s",
                before_gate,
                before_dup,
                before_tc,
                before_remaining,
                final_source,
            )

            new_keyframes, new_validation, new_phase_keyframes = cleanup_pro_late_strip_duplicates(
                video_path,
                poses,
                keyframes_out,
                phase_keyframes_out,
                kf_validation_out,
                keyframe_width=keyframe_width,
            )

            after_gate = bool(new_validation.get("final_keyframe_gate_pass", False))
            after_dup = int(new_validation.get("near_duplicates", 0))
            after_tc = int(new_validation.get("time_too_close", 0))
            after_remaining = _remaining_phases_from_validation(new_validation)
            fv_new = dict(new_validation.get("final_keyframe_validation") or {})
            accepted = bool(fv_new.get("late_strip_cleanup_accepted_by_service")) or _cleanup_accepted(
                before_gate=before_gate,
                before_dup=before_dup,
                before_tc=before_tc,
                before_remaining=before_remaining,
                new_validation=new_validation,
            )

            logger.info(
                "[keyframe] monkeypatch_cleanup_compare before_gate=%s before_near_duplicates=%s "
                "before_time_too_close=%s before_remaining_phases=%s after_gate=%s after_near_duplicates=%s "
                "after_time_too_close=%s after_remaining_phases=%s final_accepted=%s",
                before_gate,
                before_dup,
                before_tc,
                before_remaining,
                after_gate,
                after_dup,
                after_tc,
                after_remaining,
                accepted,
            )

            if accepted:
                new_src = str(new_validation.get("final_keyframe_source") or final_source)
                fv = dict(new_validation.get("final_keyframe_validation") or {})
                logger.info(
                    "[keyframe] monkeypatch exit late-strip cleanup gate_pass=%s resolved=%s remaining=%s "
                    "cleanup_applied=%s source=%s",
                    after_gate,
                    bool(fv.get("late_strip_cleanup_resolved", False)),
                    after_remaining,
                    bool(fv.get("late_strip_cleanup_applied", False)),
                    new_src,
                )
                return new_keyframes, new_validation, new_phase_keyframes, new_src
        except Exception as exc:
            logger.exception("[keyframe] monkeypatch late-strip cleanup failed: %s", exc)
        return keyframes_out, kf_validation_out, phase_keyframes_out, final_source

    keyframe_service.ensure_keyframes_ordered_for_ai = patched_ensure_keyframes_ordered_for_ai
    setattr(keyframe_service, _PATCH_FLAG, True)
    logger.info("[pro_fix] applied monkeypatch to services.keyframe_service.ensure_keyframes_ordered_for_ai")
