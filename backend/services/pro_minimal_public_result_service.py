from __future__ import annotations

from typing import Any

from services.keyframe_service import SWING_PHASE_META

# Stellar Pro public contract: client-facing keyframe fields only (no internal debug).
_KF_PUBLIC_KEYS = (
    'phase',
    'label_en',
    'label_zh',
    'source_pose_idx',
    'source_frame_index',
    'timestamp',
    'image_base64',
    'pose_snapshot',
)


def _minimal_keyframes(keyframes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kf in keyframes or []:
        d = dict(kf)
        ph = str(d.get('phase') or '')
        meta = SWING_PHASE_META.get(ph, {})
        row: dict[str, Any] = {
            'phase': ph or d.get('phase'),
            'label_en': meta.get('label_en', ph),
            'label_zh': meta.get('label_zh', ph),
            'source_pose_idx': d.get('source_pose_idx'),
            'source_frame_index': d.get('source_frame_index', d.get('frame_index')),
            'timestamp': d.get('timestamp'),
            'image_base64': d.get('image_base64'),
            'pose_snapshot': d.get('pose_snapshot'),
        }
        rows.append({k: row[k] for k in _KF_PUBLIC_KEYS})
    return rows


def pack_pro_minimal_public_result(raw: dict[str, Any]) -> dict[str, Any]:
    """Minimal top-level keys for /stellar-pro/analyze (no provider / gate / fallback noise)."""
    raw = dict(raw or {})
    summ_en = raw.get('summary')
    summ_zh = raw.get('summary_zh')
    if summ_en is None and summ_zh is not None:
        summ_en = summ_zh
    if summ_zh is None and summ_en is not None:
        summ_zh = summ_en
    out: dict[str, Any] = {
        'analysis_id': raw.get('analysis_id'),
        'type': raw.get('type', 'stellar_pro'),
        'status': raw.get('status', 'completed'),
        'summary': summ_en,
        'summary_zh': summ_zh,
        'total_score': raw.get('total_score', 0),
        'keyframes': _minimal_keyframes(list(raw.get('keyframes') or [])),
        'contact_sheet_url': raw.get('contact_sheet_url'),
        'video_url': raw.get('video_url'),
    }
    for k in (
        'issues',
        'issues_zh',
        'suggestions',
        'suggestions_zh',
        'scores',
        'advanced_metrics',
        'training_plan',
        'skeleton_data',
        'pose_frames',
        'prediction',
        'trajectory',
        'video_meta',
        'phase_keyframes',
        'hand_detection',
        'detected_club',
        'hand_assumed',
        'hand_warning',
        'club_assumed',
        'club_warning',
    ):
        if k in raw and raw[k] is not None:
            out[k] = raw[k]
    return out
