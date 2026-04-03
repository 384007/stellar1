from __future__ import annotations

from typing import Any

_INTERNAL_KEYS = {
    'ai_provider',
    'ai_key',
    'phase_debug',
    'keyframe_validation',
    'final_phase_keyframes',
    'final_keyframe_validation',
    'final_keyframe_order_ok',
    'final_keyframe_time_order_ok',
    'final_keyframe_source',
    'final_keyframe_gate_pass',
    'phase_detector_version',
    'phase_detector_confidence',
    'top_candidate_debug',
    'impact_candidate_debug',
    'top_keyframe_vs_event',
    'impact_keyframe_vs_event',
    'top_semantic_at_keyframe',
    'impact_semantic_at_keyframe',
    'top_semantic_ok',
    'impact_semantic_ok',
    'semantic_validation',
    'phase_boundary',
    'analysis_reliability',
    'sweet_spot_warning',
    'sweet_spot_confidence',
    'segmentation_available',
    'world_3d_available',
    'provider_meta',
    'error_reason',
}


_KEYFRAME_PUBLIC_KEYS = {
    'phase',
    'label_en',
    'label_zh',
    'timestamp',
    'image_base64',
    'image_url',
    'status',
    'action_summary',
}


def _public_keyframes(keyframes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for kf in keyframes or []:
        row = {k: v for k, v in dict(kf).items() if k in _KEYFRAME_PUBLIC_KEYS}
        if 'status' not in row:
            row['status'] = 'ok'
        out.append(row)
    return out


def pack_pro_public_result(raw: dict[str, Any]) -> dict[str, Any]:
    raw = dict(raw or {})
    out = {
        'analysis_id': raw.get('analysis_id'),
        'type': 'pro',
        'status': raw.get('status', 'completed'),
        'summary': raw.get('summary'),
        'summary_zh': raw.get('summary_zh'),
        'issues': raw.get('issues', []),
        'issues_zh': raw.get('issues_zh', []),
        'suggestions': raw.get('suggestions', []),
        'suggestions_zh': raw.get('suggestions_zh', []),
        'scores': raw.get('scores', {}),
        'total_score': raw.get('total_score', 0),
        'training_plan': raw.get('training_plan', {}),
        'prediction': raw.get('prediction', {}),
        'keyframes': _public_keyframes(list(raw.get('keyframes') or [])),
        'contact_sheet_url': raw.get('contact_sheet_url'),
        'video_url': raw.get('video_url'),
        'video_meta': raw.get('video_meta', {}),
        'overall_action_summary': raw.get('overall_action_summary') or raw.get('summary_zh') or raw.get('summary'),
    }
    for k, v in raw.items():
        if k in out or k in _INTERNAL_KEYS:
            continue
        if k.startswith('debug_') or k.endswith('_debug'):
            continue
        out[k] = v
    return out
