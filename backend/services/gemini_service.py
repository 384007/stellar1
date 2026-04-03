import asyncio
import base64
import logging
import os
import json
import re
import threading
from functools import partial
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Business-level hard timeouts (seconds). Override via env if needed.
# Prevents hung Gemini/Qwen/network from blocking Lite / Plus / Pro / shared pipelines.
PHASE_DETECT_TIMEOUT_S = float(os.getenv("STELLAR_PHASE_DETECT_TIMEOUT_S", "90"))
LITE_AI_TIMEOUT_S = float(os.getenv("STELLAR_LITE_AI_TIMEOUT_S", "120"))
PLUS_AI_TIMEOUT_S = float(os.getenv("STELLAR_PLUS_AI_TIMEOUT_S", "180"))
PRO_AI_TIMEOUT_S = float(os.getenv("STELLAR_PRO_AI_TIMEOUT_S", "240"))
IMAGE_ONLY_TIMEOUT_S = float(os.getenv("STELLAR_IMAGE_ONLY_TIMEOUT_S", "120"))
QWEN_TIMEOUT_S = float(os.getenv("STELLAR_QWEN_TIMEOUT_S", "120"))
PLUS_OBSERVATION_TIMEOUT_S = float(os.getenv("STELLAR_PLUS_OBSERVE_TIMEOUT_S", "90"))

# Developer API keys: GEMINI_API_KEY (required if not using Vertex), optional GEMINI_API_KEY_2 … _10.
# Egress / geo: set GEMINI_HTTPS_PROXY or HTTPS_PROXY to tunnel via SG/JP etc.; code forces REST transport when set.
# Vertex AI (optional): set GEMINI_BACKEND=vertex, VERTEX_AI_PROJECT or GOOGLE_CLOUD_PROJECT,
# VERTEX_AI_LOCATION (default us-central1; use asia-southeast1 / asia-northeast1 for SG/Tokyo), VERTEX_GEMINI_MODEL.
# Auth: Application Default Credentials — GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json or workload identity.
# Optional extra keys: GEMINI_API_KEY_2 … GEMINI_API_KEY_10 (same format as GEMINI_API_KEY). On quota/429, try next.

_genai = None
_vertex_inited = False
_gemini_dev_lock = threading.Lock()


_gemini_proxy_logged = False


def _effective_gemini_https_proxy() -> str:
    """Prefer dedicated override so tests can use SG/JP egress without touching global env first."""
    return (
        (os.getenv("GEMINI_HTTPS_PROXY") or "").strip()
        or (os.getenv("HTTPS_PROXY") or "").strip()
        or (os.getenv("https_proxy") or "").strip()
    )


def _apply_gemini_outbound_proxy_env() -> bool:
    """Normalize proxy into HTTPS_PROXY for google-generativeai (REST) and urllib3."""
    p = _effective_gemini_https_proxy()
    if not p:
        return False
    os.environ["HTTPS_PROXY"] = p
    os.environ["https_proxy"] = p
    hp = (os.getenv("GEMINI_HTTP_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("http_proxy") or "").strip()
    if hp:
        os.environ["HTTP_PROXY"] = hp
        os.environ["http_proxy"] = hp
    return True


def _log_gemini_proxy_once() -> None:
    global _gemini_proxy_logged
    if _gemini_proxy_logged:
        return
    _gemini_proxy_logged = True
    if _effective_gemini_https_proxy():
        logger.info("[gemini] outbound proxy active → using REST transport for developer API")
        print("[stellar-ai] gemini developer_api: proxy + transport=rest", flush=True)


def _genai_configure_developer(api_key: str) -> None:
    """Configure google-generativeai; force REST when proxy set so traffic goes through SG/JP tunnel."""
    import google.generativeai as genai

    _apply_gemini_outbound_proxy_env()
    use_rest = bool(_effective_gemini_https_proxy())
    if use_rest:
        _log_gemini_proxy_once()
    kwargs: dict = {"api_key": api_key}
    if use_rest:
        kwargs["transport"] = "rest"
    genai.configure(**kwargs)


def _collect_developer_api_keys() -> list[str]:
    """GEMINI_API_KEY first (unchanged), then GEMINI_API_KEY_2 … GEMINI_API_KEY_10."""
    keys: list[str] = []
    primary = os.getenv("GEMINI_API_KEY", "").strip()
    if primary:
        keys.append(primary)
    for n in range(2, 11):
        k = os.getenv(f"GEMINI_API_KEY_{n}", "").strip()
        if k and k not in keys:
            keys.append(k)
    return keys


def developer_key_label(slot: int) -> str:
    """API-facing label for which env key was used: 'key', 'key2', 'key3', … (slot 1 → key)."""
    if slot <= 1:
        return "key"
    return f"key{slot}"


def _is_gemini_quota_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "429" in msg or "resource exhausted" in msg or "quota" in msg or "rate limit" in msg:
        return True
    try:
        import google.api_core.exceptions as gexc

        return isinstance(exc, gexc.ResourceExhausted)
    except ImportError:
        return False


def _call_gemini_developer_sync(
    prompt: str,
    images: list[str],
    max_tokens: int,
    temperature: float,
) -> tuple[str, int]:
    keys = _collect_developer_api_keys()
    if not keys:
        raise RuntimeError("GEMINI_API_KEY not configured on server")

    import google.generativeai as genai

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    content_parts: list = [prompt]
    for img_b64 in images:
        content_parts.append({"mime_type": "image/jpeg", "data": img_b64})

    last_err: BaseException | None = None
    with _gemini_dev_lock:
        for idx, key in enumerate(keys):
            try:
                _genai_configure_developer(key)
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    content_parts,
                    generation_config=genai.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        response_mime_type="application/json",
                    ),
                )
                if not response.candidates or not response.text:
                    raise RuntimeError("Gemini returned empty response")
                slot = idx + 1
                ak = developer_key_label(slot)
                logger.info("[gemini] developer_api ok ai_key=%s", ak)
                # Modal / Render often only surface stdout; search logs for [stellar-ai] ai_key=
                print(f"[stellar-ai] ai_key={ak}", flush=True)
                return response.text, slot
            except Exception as e:
                last_err = e
                if idx < len(keys) - 1 and _is_gemini_quota_error(e):
                    logger.warning(
                        "[gemini] key slot %s quota/rate limited (%s), trying next",
                        idx + 1,
                        e,
                    )
                    print(
                        f"[stellar-ai] ai_key slot {idx + 1} hit quota/rate limit, trying next key",
                        flush=True,
                    )
                    continue
                raise
    if last_err:
        raise last_err
    raise RuntimeError("Gemini developer API call failed")


def _use_vertex() -> bool:
    """Vertex AI when GEMINI_BACKEND=vertex (explicit). Uses GCP ADC, not API key."""
    return (os.getenv("GEMINI_BACKEND") or "").strip().lower() == "vertex"


def _vertex_project() -> str:
    return (os.getenv("VERTEX_AI_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()


def _vertex_location() -> str:
    # Examples: us-central1 | asia-southeast1 (Singapore) | asia-northeast1 (Tokyo)
    return (os.getenv("VERTEX_AI_LOCATION") or "us-central1").strip()


def _vertex_model_id() -> str:
    # Vertex model IDs differ from AI Studio; override with VERTEX_GEMINI_MODEL.
    return (os.getenv("VERTEX_GEMINI_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.0-flash-001").strip()


def _init_vertex_if_needed() -> None:
    global _vertex_inited
    if _vertex_inited:
        return
    project = _vertex_project()
    if not project:
        raise RuntimeError(
            "Vertex AI: set VERTEX_AI_PROJECT or GOOGLE_CLOUD_PROJECT, "
            "and GEMINI_BACKEND=vertex"
        )
    import vertexai

    vertexai.init(project=project, location=_vertex_location())
    _vertex_inited = True
    logger.info(
        "[gemini] Vertex AI init project=%s location=%s model=%s",
        project,
        _vertex_location(),
        _vertex_model_id(),
    )


def _call_gemini_vertex_sync(
    prompt: str,
    images: list[str],
    max_tokens: int,
    temperature: float,
) -> str:
    _init_vertex_if_needed()
    from vertexai.generative_models import GenerativeModel, Part, GenerationConfig

    model = GenerativeModel(_vertex_model_id())
    contents: list = [prompt]
    for img_b64 in images:
        contents.append(
            Part.from_data(mime_type="image/jpeg", data=base64.b64decode(img_b64))
        )
    gen_cfg = GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
    )
    response = model.generate_content(contents, generation_config=gen_cfg)
    if not response.candidates:
        raise RuntimeError("Vertex Gemini returned no candidates")
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Vertex Gemini returned empty response")
    return text

QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-vl-max-latest"


def _get_genai():
    global _genai
    if _genai is None:
        import google.generativeai as genai

        _genai = genai
        if not _collect_developer_api_keys():
            print("[gemini_service] WARNING: GEMINI_API_KEY not set")
    return _genai


LITE_PROMPT = """You are a professional golf coach and biomechanics expert. Analyze the following golf swing data.

Up to eight still images may be attached in chronological swing order (address→finish), matching the server's keyframe strip when provided. This is a Lite analysis (not Plus): base scores primarily on the pose JSON. **Do not** output Plus-style per-image phase critiques (no frame-by-frame claims that image N is definitively top/impact/finish). Use images only as coarse supporting context.

Skeleton angle data: {pose_data}

Provide your analysis in the following JSON format ONLY (no markdown, no extra text):
{{
  "scores": {{
    "grip": <0-100>,
    "stance": <0-100>,
    "backswing": <0-100>,
    "downswing": <0-100>,
    "follow_through": <0-100>
  }},
  "total_score": <0-100>,
  "issues": ["issue 1", "issue 2", "issue 3"],
  "issues_zh": ["问题1", "问题2", "问题3"],
  "suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"],
  "suggestions_zh": ["建议1", "建议2", "建议3"],
  "summary": "English summary (200 words max)",
  "summary_zh": "中文总结（200字以内）"
}}

Base your scoring on these biomechanical principles:
- Grip: Neutral grip position, consistent pressure
- Stance: Shoulder-width apart, proper ball position, balanced weight
- Backswing: Full shoulder turn (>=90°), maintained spine angle, proper wrist hinge
- Downswing: Proper sequencing (hips→torso→arms→club), lag retention
- Follow Through: Full rotation, balanced finish, high hands

Analyze the angle data carefully. Consider:
- Shoulder rotation angle relative to hip rotation (X-factor)
- Knee flex maintenance through the swing
- Elbow angles at key positions
- Spine tilt consistency"""

LITE_PROMPT_APPEND_PHASE_UNRELIABLE = """
[phase_images_reliable=FALSE — Lite route]
Strip images may be incomplete, monotonic-fallback, or failed semantic/order gates. Rely mainly on pose_data. **Forbidden:** Plus-style per-image phase-by-phase critique; do not pin faults to "image 4 = top" etc. Keep summaries global, not indexed by frame number.
"""

PRO_PROMPT = """You are an elite PGA-level golf coach with expertise in biomechanics and swing analysis. Provide a comprehensive professional analysis.

Up to eight still images are attached IN THIS EXACT ORDER (same as the product UI keyframe strip):
(1) address → (2) takeaway → (3) backswing → (4) top of swing → (5) downswing → (6) impact → (7) follow_through → (8) finish.
When you describe a phase, it MUST match what is visible in that numbered frame. If fewer than eight images are provided, only describe phases that have a corresponding image.

VISUAL-FIRST (these frames drive coaching quality — read pixels before JSON):
- For (1) address, (4) top, (6) impact, and (8) finish: write at least one concrete, visible observation each in BOTH English and Chinese somewhere in your analysis (issues/summary/advanced_metrics or implied in detailed issues). Examples: shaft plane vs horizon, hands height vs sternum, clubhead position, belt/hip openness toward target, head stability, trail elbow fold.
- For every major fault, LABEL it explicitly as either a POSTURE problem (setup/shape that is wrong even in a still photo) or a TIMING/SEQUENCE problem (transition order, early extension, casting) — use what you SEE in the images plus angles; do not blame "timing" without visual or kinematic support.
- Do not paraphrase the JSON numbers generically; every numeric claim should connect to something visible or mechanically implied in the strip.

Skeleton angle data: {pose_data}

Provide your analysis in the following JSON format ONLY (no markdown, no extra text):
{{
  "scores": {{
    "grip": <0-100>,
    "stance": <0-100>,
    "backswing": <0-100>,
    "downswing": <0-100>,
    "follow_through": <0-100>
  }},
  "total_score": <0-100>,
  "issues": ["detailed issue 1", "detailed issue 2", "detailed issue 3", "detailed issue 4", "detailed issue 5"],
  "issues_zh": ["详细问题1", "详细问题2", "详细问题3", "详细问题4", "详细问题5"],
  "suggestions": ["detailed suggestion 1", "detailed suggestion 2", "detailed suggestion 3", "detailed suggestion 4", "detailed suggestion 5"],
  "suggestions_zh": ["详细建议1", "详细建议2", "详细建议3", "详细建议4", "详细建议5"],
  "summary": "Detailed English analysis (800 words)",
  "summary_zh": "详细中文分析（800字）",
  "advanced_metrics": {{
    "swing_tempo": "backswing_to_downswing_ratio",
    "x_factor": <degrees>,
    "hip_slide": "description",
    "shaft_lean_at_impact": "description",
    "release_point": "description"
  }},
  "training_plan": {{
    "day1": {{"focus": "topic", "drills": ["drill1", "drill2"], "duration": "30 min"}},
    "day2": {{"focus": "topic", "drills": ["drill1", "drill2"], "duration": "30 min"}},
    "day3": {{"focus": "topic", "drills": ["drill1", "drill2"], "duration": "30 min"}},
    "day4": {{"focus": "topic", "drills": ["drill1", "drill2"], "duration": "30 min"}},
    "day5": {{"focus": "topic", "drills": ["drill1", "drill2"], "duration": "30 min"}},
    "day6": {{"focus": "topic", "drills": ["drill1", "drill2"], "duration": "30 min"}},
    "day7": {{"focus": "topic", "drills": ["drill1", "drill2"], "duration": "rest/review"}}
  }},
  "detected_club": {{
    "club_type": "<1W|3W|5W|3I|4I|5I|6I|7I|8I|9I|PW|AW|SW|LW|PT|UNKNOWN>",
    "club_group": "<WOOD|IRON|WEDGE|PUTTER|UNKNOWN>",
    "confidence": <0.0-1.0>
  }}
}}

Analyze with extreme detail, referencing specific angle values and comparing to tour-player benchmarks.
detected_club: identify the club from the images — drivers (1W) have large round heads; irons have thin blade/cavity heads; wedges are short with high loft; putters are distinctive. Use UNKNOWN only if truly impossible to tell."""

PLUS_PROMPT = """You are an elite golf biomechanics diagnostician conducting a premium Plus analysis.

POSE DATA (JSON — required):
{pose_data}

Skeleton / pose_data: MediaPipe body landmarks and derived angles (degrees). Joints are normalized 0–1 in the frame; use them together with the images — if a limb is low visibility in an image, trust the numeric visibility and angles for that side.

Up to eight still images are attached IN THIS EXACT ORDER (same as the product keyframes):
(1) address → (2) takeaway → (3) backswing → (4) top → (5) downswing → (6) impact → (7) follow_through → (8) finish.
Only when phase_images_reliable=true should swing_phase_evaluations map image index to exact phases. If reliability is false, do not claim strict phase mapping.

VISUAL-FIRST (Plus users paid for images — treat pixels as primary evidence):
- Phases (1) address, (4) top, (6) impact, (8) finish: swing_phase_evaluations note_zh and note_en MUST each name at least one visible cue (club, hands, torso, hips, head, shaft). If the frame is unclear, say what is unclear instead of generic advice.
- Separate POSTURE faults (what looks wrong in a still) from TIMING/SEQUENCE faults (what the strip progression suggests). Mention both in primary_diagnosis or additional_issues when supported.
- problem_description_zh/en must reference at least two different visible regions (e.g. lower body + arms), not only restated angle labels.
- Do not output template phrases; each sentence should be traceable to a visible detail or a specific angle value from the pose JSON above.

Respond with ONLY this JSON (no markdown, no backticks):
{{
  "posture_score": <float 1.0-10.0, one decimal, amateurs typically 4.0-7.5>,
  "primary_diagnosis": {{
    "title_zh": "<concise issue name, e.g. 上杆顶点上半身抬起>",
    "title_en": "<English equivalent>",
    "status_zh": "<one of: 完美|做得好|再接再厉|需要注意|需要改进>",
    "status_en": "<one of: Perfect|Good|Keep trying|Needs attention|Needs improvement>",
    "ai_confidence": <integer 0-100>
  }},
  "additional_issues": [
    {{
      "title_zh": "<issue>",
      "title_en": "<issue>",
      "status_zh": "<status>",
      "status_en": "<status>"
    }}
  ],
  "quick_tip_zh": "<one actionable 3-second coaching cue in Chinese>",
  "quick_tip_en": "<same in English>",
  "problem_description_zh": "<2-3 sentences explaining the primary issue in Chinese>",
  "problem_description_en": "<same in English>",
  "swing_phase_evaluations": [
    {{"phase": "address", "status": "<pass or error>", "note_zh": "<short>", "note_en": "<short>"}},
    {{"phase": "takeaway", "status": "<pass or error>", "note_zh": "", "note_en": ""}},
    {{"phase": "backswing", "status": "<pass or error>", "note_zh": "", "note_en": ""}},
    {{"phase": "top", "status": "<pass or error>", "note_zh": "", "note_en": ""}},
    {{"phase": "downswing", "status": "<pass or error>", "note_zh": "", "note_en": ""}},
    {{"phase": "impact", "status": "<pass or error>", "note_zh": "", "note_en": ""}},
    {{"phase": "follow_through", "status": "<pass or error>", "note_zh": "", "note_en": ""}},
    {{"phase": "finish", "status": "<pass or error>", "note_zh": "", "note_en": ""}}
  ],
  "training": {{
    "title_zh": "<training focus in Chinese>",
    "title_en": "<training focus in English>",
    "description_zh": "<when/why this issue occurs>",
    "description_en": "<same>",
    "difficulty": "<easy|normal|hard>",
    "frequency_percent": <float 0-100, estimated occurrence rate>
  }},
  "recommended_videos": [
    {{"title": "<real YouTube video title about this issue>", "creator": "<channel>", "search_query": "<YouTube search string>"}},
    {{"title": "<another>", "creator": "<channel>", "search_query": "<search>"}}
  ],
  "scores": {{"grip": <0-100>, "stance": <0-100>, "backswing": <0-100>, "downswing": <0-100>, "follow_through": <0-100>}},
  "total_score": <0-100>,
  "issues": ["<issue 1 en>", "<issue 2>", "<issue 3>"],
  "issues_zh": ["<问题1>", "<问题2>", "<问题3>"],
  "suggestions": ["<suggestion 1 en>", "<suggestion 2>", "<suggestion 3>"],
  "suggestions_zh": ["<建议1>", "<建议2>", "<建议3>"],
  "summary": "<English analysis, 300 words>",
  "summary_zh": "<中文分析，300字>",
  "detected_club": {{
    "club_type": "<1W|3W|5W|3I|4I|5I|6I|7I|8I|9I|PW|AW|SW|LW|PT|UNKNOWN>",
    "club_group": "<WOOD|IRON|WEDGE|PUTTER|UNKNOWN>",
    "confidence": <0.0-1.0>
  }}
}}

Rules:
- posture_score: 0.0-10.0 with one decimal. Amateurs are typically 4.0-7.5. Be honest.
- swing_phase_evaluations MUST have exactly 8 entries covering all phases.
- At least 1 and at most 4 phases should have status "error".
- primary_diagnosis: the SINGLE most impactful swing fault you observe.
- quick_tip: immediately actionable, like a real coach's 3-second instruction.
- difficulty: easy = fixable in 1-2 sessions, normal = 1-2 weeks, hard = months.
- frequency_percent: how often this fault likely appears based on the angle data.
- recommended_videos: default to [] unless clearly supported by reliable phase evidence.
- detected_club: identify the club from the images — look at shaft length, head shape, loft angle. Use UNKNOWN only if truly impossible to tell. Drivers (1W) have large round heads; irons have thin blade/cavity heads; wedges are short with high loft; putters are distinctive. If unsure between two adjacent irons, pick the more common one.
- Do NOT inflate scores. Be brutally honest for amateur golfers."""

PLUS_PROMPT_APPEND_PHASE_C = """
TEMPORAL_PIPELINE (Phase-C — backend fused kinematics ± optional action prior; JSON, no raw frames):
{phase_c_json}

Calibration rules (do not override phase_images_reliable for strict image↔phase mapping):
- Treat global_segmentation_confidence (when present) as overall segment-strip trust. Below ~0.55: use cautious diagnosis wording; avoid primary_diagnosis.ai_confidence above ~78 unless pose_data and images are exceptionally clear.
- Higher temporal_prior_strength with action_backend.status "ok" means an auxiliary temporal model partially agreed with boundaries; still prioritize visible evidence and pose_data over this block.
- If action_backend.status is not "ok", ignore action_backend for certainty; rely on phase_confidence.boundary_* only.
"""

PLUS_PROMPT_APPEND_PHASE_UNRELIABLE = """
[phase_images_reliable=FALSE — enforced by server]
The 8 images are chronological samples from the swing but are NOT verified as true address→finish phase moments (may be uniform time samples or misaligned labels).

Hard rules:
- swing_phase_evaluations: exactly 8 objects (address…finish). Every "status" MUST be the string unknown (not pass/error). note_zh and note_en must briefly state (bilingual) that per-phase image claims are disabled.
- Do NOT assert that a specific image index is top, impact, or follow-through.
- posture_score, primary_diagnosis, scores, summaries: base mainly on the pose_data JSON; use images only for coarse visual context.
- recommended_videos MUST be an empty array [].
"""

PRO_PROMPT_APPEND_PHASE_UNRELIABLE = """
[phase_images_reliable=FALSE]
The 8 frames are not verified as true phase-labelled strips. Do not write detailed phase-by-phase image critique as if (1)=address…(8)=finish were guaranteed. Emphasize pose_data and overall swing; avoid pinning faults to a numbered phase image.
"""

STELLAR_PRO_REPORT_PROMPT = """You are an elite PGA-level golf coach. **No images are attached.** Do not claim you viewed, selected, ranked, or verified any keyframe stills.

Authoritative facts:
- Phase timing and the eight keyframe moments were chosen by an in-house motion engine from 240fps pose time-series and biomechanical rules.
- Your only job is to write the coaching JSON from the numeric data below. Never instruct the client to "use a different frame" or second-guess which image is impact/top/etc.

POSE_TIME_SERIES (JSON):
{pose_data}

KEYFRAME_MOTION_SUMMARY (phase → timestamp, frame index, pose index — sequence metadata only, not pictures):
{keyframe_metrics}

Respond with ONLY this JSON (no markdown, no backticks). Use the same shape as standard Pro analysis:
{{
  "scores": {{
    "grip": <0-100>,
    "stance": <0-100>,
    "backswing": <0-100>,
    "downswing": <0-100>,
    "follow_through": <0-100>
  }},
  "total_score": <0-100>,
  "issues": ["detailed issue 1", "detailed issue 2", "detailed issue 3", "detailed issue 4", "detailed issue 5"],
  "issues_zh": ["详细问题1", "详细问题2", "详细问题3", "详细问题4", "详细问题5"],
  "suggestions": ["detailed suggestion 1", "detailed suggestion 2", "detailed suggestion 3", "detailed suggestion 4", "detailed suggestion 5"],
  "suggestions_zh": ["详细建议1", "详细建议2", "详细建议3", "详细建议4", "详细建议5"],
  "summary": "Detailed English analysis (600-800 words; kinematics from numbers only)",
  "summary_zh": "详细中文分析（600-800字；仅依据数值）",
  "advanced_metrics": {{
    "swing_tempo": "backswing_to_downswing_ratio estimate from series",
    "x_factor": <degrees or 0>,
    "hip_slide": "description",
    "shaft_lean_at_impact": "inferred from angles only",
    "release_point": "description"
  }},
  "training_plan": {{
    "day1": {{"focus": "topic", "drills": ["drill1", "drill2"], "duration": "30 min"}},
    "day2": {{"focus": "topic", "drills": ["drill1", "drill2"], "duration": "30 min"}},
    "day3": {{"focus": "topic", "drills": ["drill1", "drill2"], "duration": "30 min"}},
    "day4": {{"focus": "topic", "drills": ["drill1", "drill2"], "duration": "30 min"}},
    "day5": {{"focus": "topic", "drills": ["drill1", "drill2"], "duration": "30 min"}},
    "day6": {{"focus": "topic", "drills": ["drill1", "drill2"], "duration": "30 min"}},
    "day7": {{"focus": "topic", "drills": ["drill1", "drill2"], "duration": "rest/review"}}
  }},
  "detected_club": {{
    "club_type": "UNKNOWN",
    "club_group": "UNKNOWN",
    "confidence": 0.0
  }}
}}

Rules:
- detected_club MUST stay UNKNOWN / 0.0 confidence (no club images).
- Do not reference "image 1…8" or "the attached strip".
- Ground every major claim in angle series or keyframe_metrics timing; avoid fictional visuals.
"""

PRO_V2_REPORT_PROMPT = """You are an elite PGA-level coach writing a truthful coaching report from motion metadata only.

No image selection task. Do NOT choose frames. Do NOT mention contact sheets, "image 1-8", or any AI frame picking.

SCREEN / NO-IMAGES MODE: There are NO attached photos. That is expected. MOTION_CONTEXT alone (phase timestamps + dense motion proxy per phase + swing window) is ENOUGH to write a full phase-based coaching report. Do NOT shorten the report or refuse depth because of "no images". Never output a trivial one-paragraph reply.

You are given MOTION_CONTEXT for fixed 8 phases at 240fps:
Address, Takeaway, Backswing, Top, Downswing, Impact, Follow-through, Finish.
Treat these phases and their order/timing as ground truth.

Coaching style requirements:
1) Write like a real PGA coach: specific, technical, direct, no marketing language or vague filler ("整体不错", "需要多练习" alone is NOT acceptable).
2) Be factual: do NOT invent invisible visual details (clubface, ball flight, exact spine angle). Infer only from timing, spacing, and dense_motion_proxy trends in MOTION_CONTEXT.
3) If evidence is limited, you MUST still write a full report: hedge with "Based on the current motion summary..." / "根据当前 motion summary 判断..." but continue with concrete phase-by-phase coaching — not excuses for brevity.
4) Late-strip honesty: if Impact, Follow-through, or Finish timing looks compressed or weak in the data, name **Impact**, **Follow-through**, or **Finish** explicitly in prose and in issues — not generic "swing unstable".

Phase binding (mandatory in issues and in summary narrative):
- If the problem is around strike timing vs downswing burst → say **Impact** explicitly.
- If post-impact release or spacing is tight → say **Follow-through** explicitly.
- If exit / settle looks weak vs earlier phases → say **Finish** explicitly.
- Avoid vague labels like "动作不稳定" without naming the phase.

Content requirements:
- issues / issues_zh: phase-specific and concrete; EACH item must START with a phase name (English list: Address, Takeaway, Backswing, Top, Downswing, Impact, Follow-through, Finish — Chinese: 站姿, 起杆, 上杆, 顶点, 下杆, 触球, 送杆, 收杆).
- suggestions / suggestions_zh: each item must be a drill/cue the player can practice this week; start with the same phase naming rule.
- Minimum length: **at least 3** items in EACH of issues, issues_zh, suggestions, suggestions_zh (prefer 4–5 if data supports it).
- summary:
  - English: **450-700 words**, multiple paragraphs, clear phase flow.
  - Must explicitly discuss at least 4 of: Address, Takeaway, Top, Impact, Follow-through, Finish (Downswing/Backswing as needed from data).
  - If Impact / Follow-through / Finish show timing concerns in MOTION_CONTEXT, dedicate a full paragraph to each affected phase.
- summary_zh:
  - **500-900 汉字**（不含空白），多段落，同样覆盖至少 4 个关键阶段；触球/送杆/收杆有问题时必须分段展开，不得只用两三句敷衍。

MOTION_CONTEXT (JSON):
{motion_context}

Return ONLY valid JSON:
{{
  "total_score": <0-100>,
  "scores": {{"grip": <0-100>, "stance": <0-100>, "backswing": <0-100>, "downswing": <0-100>, "follow_through": <0-100>}},
  "issues": ["Phase: specific issue", "...", "..."],
  "issues_zh": ["阶段：具体问题", "...", "..."],
  "suggestions": ["Phase: actionable drill/cue", "...", "..."],
  "suggestions_zh": ["阶段：可执行练习或口令", "...", "..."],
  "summary": "450-700 words English coaching report",
  "summary_zh": "500-900字中文教练报告",
  "training_plan": {{
    "day1": {{"focus": "topic (Chinese)", "drills": ["drill1", "drill2"], "duration": "30 min"}},
    "day2": {{"focus": "...", "drills": ["...", "..."], "duration": "30 min"}},
    "day3": {{"focus": "...", "drills": ["...", "..."], "duration": "30 min"}},
    "day4": {{"focus": "...", "drills": ["...", "..."], "duration": "30 min"}},
    "day5": {{"focus": "...", "drills": ["...", "..."], "duration": "30 min"}},
    "day6": {{"focus": "...", "drills": ["...", "..."], "duration": "30 min"}},
    "day7": {{"focus": "复习与录像对比", "drills": ["drill1"], "duration": "20 min"}}
  }}
}}

training_plan MUST include day1 through day7; focus can be Chinese; drills concrete and short.
"""

PRO_V2_REPORT_PROMPT_PASS2 = """PASS 2 — Your previous output was REJECTED for being too thin, empty, or non-phase-specific.

You are an elite PGA coach. MOTION_CONTEXT JSON is the ONLY source. No images. Do NOT choose frames.

STRICT OUTPUT RULES:
1) issues, issues_zh, suggestions, suggestions_zh: **minimum 4 items each** (not 2, not 3). Every line MUST start with a phase name (English: Address, Takeaway, Backswing, Top, Downswing, Impact, Follow-through, Finish / Chinese: 站姿, 起杆, 上杆, 顶点, 下杆, 触球, 送杆, 收杆).
2) NO vague filler. Each issue names a phase + a concrete timing/motion-proxy observation from the JSON.
3) If Impact timing looks tight vs the downswing burst, write "**Impact**: ..." explicitly. Same for **Follow-through** and **Finish** when post-impact spacing or exit proxy is weak.
4) summary: **500-750 English words**, multiple paragraphs, phase-ordered narrative (Address → … → Finish), honest hedging only as a phrase — not as a substitute for content.
5) summary_zh: **600-950 汉字**，多段落，阶段清晰；禁止仅用两三句概括。

Do not claim you lack information because there are no pictures — the numeric phase timeline is sufficient.

MOTION_CONTEXT (JSON):
{motion_context}

Return ONLY the same JSON schema as before (total_score, scores, issues, issues_zh, suggestions, suggestions_zh, summary, summary_zh, training_plan day1-day7).
"""

IMAGE_ONLY_PROMPT = """You are an expert PGA-level golf coach and biomechanics analyst.

CRITICAL RULE — DETECTION FIRST:
1. Describe what you actually see in the image(s).
2. Decide: does this show a real golf swing? Set "is_golf_swing" accordingly.
3. If "is_golf_swing" is false, you MUST return all scores as 0, total_score as 0, empty issues/suggestions arrays, and state in the summary that no golf swing was detected. Do NOT fabricate analysis for non-golf content.

Only if "is_golf_swing" is true, evaluate these 5 dimensions (0-100):
1. Grip 2. Stance 3. Backswing 4. Downswing 5. Follow-through

Be brutally honest. Amateur golfers typically score 40-75. Do NOT inflate scores.

Respond with ONLY this JSON (no markdown, no backticks):
{
  "what_i_see": "<1-2 sentences describing what is visible>",
  "what_i_see_zh": "<中文描述看到了什么>",
  "is_golf_swing": true or false,
  "scores": {"grip": <0-100 or 0 if not a swing>, "stance": <0-100 or 0>, "backswing": <0-100 or 0>, "downswing": <0-100 or 0>, "follow_through": <0-100 or 0>},
  "total_score": <weighted average or 0 if not a swing>,
  "issues": ["<issue 1>", "<issue 2>", "<issue 3>"] or [] if not a swing,
  "issues_zh": ["<问题1>", "<问题2>", "<问题3>"] or [] if not a swing,
  "suggestions": ["<fix 1>", "<fix 2>", "<fix 3>"] or [] if not a swing,
  "suggestions_zh": ["<建议1>", "<建议2>", "<建议3>"] or [] if not a swing,
  "summary": "<Detailed English analysis, 150-200 words, or state no swing detected>",
  "summary_zh": "<详细中文分析，150-200字，或说明未检测到挥杆>",
  "prediction": {
    "predicted_distance": <yards or 0>,
    "lateral_offset": <yards or 0>,
    "shot_shape": "<Fade/Draw/Straight/Hook/Slice or N/A>",
    "shot_shape_zh": "<右曲/左曲/直球 or 未知>",
    "club_head_speed": <mph or 0>,
    "ball_speed": <mph or 0>,
    "launch_angle": <degrees or 0>,
    "spin_rate": <rpm or 0>,
    "smash_factor": <ratio or 0>
  },
  "detected_club": {
    "club_type": "<1W|3W|5W|3I|4I|5I|6I|7I|8I|9I|PW|AW|SW|LW|PT|UNKNOWN>",
    "club_group": "<WOOD|IRON|WEDGE|PUTTER|UNKNOWN>"
  }
}"""


def extract_json_from_response(text: str) -> dict:
    # Strip <think>...</think> blocks emitted by reasoning models
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "scores": {"grip": 0, "stance": 0, "backswing": 0, "downswing": 0, "follow_through": 0},
            "total_score": 0,
            "issues": ["Unable to parse AI response"],
            "issues_zh": ["无法解析AI响应"],
            "suggestions": ["Please try again with a clearer video"],
            "suggestions_zh": ["请使用更清晰的视频重试"],
            "summary": "Analysis could not be completed. Please try again.",
            "summary_zh": "分析未能完成，请重试。",
        }


# ── Gemini helpers ──

def _get_model(name: str = ""):
    if not name:
        name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    keys = _collect_developer_api_keys()
    if not keys:
        raise RuntimeError("GEMINI_API_KEY not configured on server")
    _get_genai()
    with _gemini_dev_lock:
        _genai_configure_developer(keys[0])
        genai = _get_genai()
        return genai.GenerativeModel(name)


async def _call_gemini(
    prompt: str,
    images: list[str],
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> tuple[str, Optional[int]]:
    """Call Gemini; returns (raw_text, key_slot). key_slot is 1..N for developer API keys, None for Vertex."""
    if _use_vertex():
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None,
            partial(_call_gemini_vertex_sync, prompt, images, max_tokens, temperature),
        )
        return text, None

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        partial(_call_gemini_developer_sync, prompt, images, max_tokens, temperature),
    )


async def run_gemini_vision(
    prompt: str,
    images_b64: list[str],
    *,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> tuple[str, Optional[int]]:
    """Vision call via Developer API (GEMINI_API_KEY…) or Vertex (GEMINI_BACKEND=vertex)."""
    return await _call_gemini(prompt, images_b64, max_tokens, temperature)


# ── Qwen (通义千问) helper via OpenAI-compatible DashScope API ──

_qwen_available: bool | None = None


def _has_qwen() -> bool:
    global _qwen_available
    if _qwen_available is None:
        _qwen_available = bool(os.getenv("QWEN_API_KEY", ""))
    return _qwen_available


async def _call_qwen(
    prompt: str,
    images: list[str],
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> str:
    """Call Qwen VL via DashScope OpenAI-compatible endpoint."""
    qwen_key = os.getenv("QWEN_API_KEY", "")
    if not qwen_key:
        raise RuntimeError("QWEN_API_KEY not configured on server")

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=qwen_key,
        base_url=QWEN_BASE_URL,
        timeout=QWEN_TIMEOUT_S,
    )

    content: list = [{"type": "text", "text": prompt}]
    for img_b64 in images[:8]:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })

    resp = await client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    raw = resp.choices[0].message.content or ""
    return re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()


# ── Unified Gemini-first, Qwen-fallback caller ──

async def _call_vision_ai(
    prompt: str,
    images: list[str],
    max_tokens: int = 4096,
    temperature: float = 0.3,
    label: str = "vision",
    *,
    timeout_s: float,
) -> tuple[str, str, Optional[int]]:
    """Try Gemini first; on failure fall back to Qwen if available.

    Returns (response_text, provider, key_slot). key_slot is 1-based index into GEMINI_API_KEY,
    GEMINI_API_KEY_2, … when provider is gemini and developer API; None for Vertex or Qwen.

    ``timeout_s`` caps wall-clock time for the whole provider chain (Gemini attempt + optional Qwen).
    """
    async def _inner() -> tuple[str, str, Optional[int]]:
        gemini_err = None
        try:
            text, key_slot = await _call_gemini(prompt, images, max_tokens, temperature)
            backend = "vertex" if _use_vertex() else "developer_api"
            ak = developer_key_label(key_slot) if key_slot is not None else "vertex"
            logger.info(
                "[ai] vision_ok provider=gemini backend=%s label=%s ai_key=%s",
                backend,
                label,
                ak,
            )
            print(f"[stellar-ai] vision label={label} ai_key={ak}", flush=True)
            return text, "gemini", key_slot
        except Exception as e:
            gemini_err = e
            logger.warning("[ai] gemini_fail label=%s err=%s", label, e)

        if _has_qwen():
            try:
                text = await _call_qwen(prompt, images, max_tokens, temperature)
                logger.info("[ai] vision_ok provider=qwen label=%s (gemini failed first)", label)
                print(f"[stellar-ai] vision label={label} provider=qwen ai_key=n/a", flush=True)
                return text, "qwen", None
            except Exception as e2:
                logger.warning("[ai] qwen_fail label=%s err=%s", label, e2)

        raise RuntimeError(f"All AI providers failed for {label}: {gemini_err}")

    try:
        return await asyncio.wait_for(_inner(), timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.error("[ai] vision_timeout label=%s timeout_s=%s", label, timeout_s)
        raise RuntimeError(
            f"AI vision timed out after {timeout_s}s (stage={label})",
        ) from None


# ── Public analysis functions ──

async def analyze_swing_lite(
    pose_data: dict,
    keyframe_images: Optional[list[str]] = None,
    region: str = "global",
    phase_images_reliable: bool = True,
) -> dict:
    """Lite: keyframe_images from phase strip; phase_images_reliable matches router phase_evaluations_reliable."""
    base = LITE_PROMPT.format(pose_data=json.dumps(pose_data, indent=2))
    prompt = base if phase_images_reliable else (base + LITE_PROMPT_APPEND_PHASE_UNRELIABLE)
    images = list(keyframe_images or [])[:8]
    try:
        text, provider, key_slot = await _call_vision_ai(
            prompt, images, 2048, 0.3, "lite", timeout_s=LITE_AI_TIMEOUT_S,
        )
        out = extract_json_from_response(text)
        out["ai_provider"] = provider
        if key_slot is not None:
            out["ai_key"] = developer_key_label(key_slot)
        return out
    except Exception as e:
        logger.error("[ai] analyze_swing_lite all providers failed: %s", e)
        return _fallback_result(str(e))


async def analyze_swing_pro(
    pose_data: dict,
    keyframe_images: Optional[list[str]] = None,
    region: str = "global",
    phase_images_reliable: bool = True,
) -> dict:
    """Pro: keyframe_images are resized phase strip frames, not uniform samples; reliable flag matches router gate."""
    base = PRO_PROMPT.format(pose_data=json.dumps(pose_data, indent=2))
    prompt = base if phase_images_reliable else (base + PRO_PROMPT_APPEND_PHASE_UNRELIABLE)
    images = list(keyframe_images or [])[:8]
    try:
        text, provider, key_slot = await _call_vision_ai(
            prompt, images, 8192, 0.2, "pro", timeout_s=PRO_AI_TIMEOUT_S,
        )
        out = extract_json_from_response(text)
        out["ai_provider"] = provider
        if key_slot is not None:
            out["ai_key"] = developer_key_label(key_slot)
        return out
    except Exception as e:
        logger.error("[ai] analyze_swing_pro all providers failed: %s", e)
        return _fallback_result(str(e))


async def analyze_stellar_pro_report_only(
    pose_data: dict,
    keyframe_metrics: dict,
    region: str = "global",
) -> dict:
    """Stellar Pro: text-only report from pose series + motion keyframe metadata — no images (no AI frame picking)."""
    _ = region
    prompt = STELLAR_PRO_REPORT_PROMPT.format(
        pose_data=json.dumps(pose_data, indent=2, ensure_ascii=False),
        keyframe_metrics=json.dumps(keyframe_metrics, indent=2, ensure_ascii=False),
    )
    try:
        text, provider, key_slot = await _call_vision_ai(
            prompt, [], 8192, 0.2, "stellar_pro_report", timeout_s=PRO_AI_TIMEOUT_S,
        )
        out = extract_json_from_response(text)
        out["ai_provider"] = provider
        if key_slot is not None:
            out["ai_key"] = developer_key_label(key_slot)
        return out
    except Exception as e:
        logger.error("[ai] analyze_stellar_pro_report_only failed: %s", e)
        return _fallback_result(str(e))


async def analyze_pro_v2_report_only(
    motion_context: dict,
    region: str = "global",
    *,
    use_strong_prompt: bool = False,
    max_tokens: int = 10240,
    call_label: str = "pro_v2_report",
) -> dict:
    """Pro v2: text-only report from motion keyframe metadata — no images (no AI frame picking)."""
    _ = region
    template = PRO_V2_REPORT_PROMPT_PASS2 if use_strong_prompt else PRO_V2_REPORT_PROMPT
    prompt = template.format(
        motion_context=json.dumps(motion_context, indent=2, ensure_ascii=False),
    )
    try:
        text, provider, key_slot = await _call_vision_ai(
            prompt, [], max_tokens, 0.2 if not use_strong_prompt else 0.15, call_label, timeout_s=PRO_AI_TIMEOUT_S,
        )
        out = extract_json_from_response(text)
        out["ai_provider"] = provider
        if key_slot is not None:
            out["ai_key"] = developer_key_label(key_slot)
        return out
    except Exception as e:
        logger.error("[ai] analyze_pro_v2_report_only failed: %s", e)
        return _fallback_result(str(e))


async def analyze_swing_plus(
    pose_data: dict,
    keyframe_images: Optional[list[str]] = None,
    region: str = "global",
    phase_images_reliable: bool = True,
    phase_c_context: Optional[dict[str, Any]] = None,
) -> dict:
    """Plus: images are the 8 phase keyframes (same as UI); reliable == phase_evaluations_reliable from routers."""
    base = PLUS_PROMPT.format(pose_data=json.dumps(pose_data, indent=2))
    prompt = base if phase_images_reliable else (base + PLUS_PROMPT_APPEND_PHASE_UNRELIABLE)
    if phase_c_context:
        prompt = prompt + PLUS_PROMPT_APPEND_PHASE_C.format(
            phase_c_json=json.dumps(phase_c_context, ensure_ascii=False, separators=(",", ":")),
        )
    images = list(keyframe_images or [])[:8]
    try:
        text, provider, key_slot = await _call_vision_ai(
            prompt, images, 8192, 0.2, "plus", timeout_s=PLUS_AI_TIMEOUT_S,
        )
        out = _normalize_plus_result(extract_json_from_response(text))
        if not phase_images_reliable:
            out = _force_unknown_phase_evals_for_unreliable(out)
        out["ai_provider"] = provider
        if key_slot is not None:
            out["ai_key"] = developer_key_label(key_slot)
        return out
    except Exception as e:
        logger.error("[ai] analyze_swing_plus all providers failed: %s", e)
        return _fallback_plus_result(str(e))


async def analyze_plus_visual_observation(
    frame_images: list[str],
    *,
    frame_labels: Optional[list[Optional[str]]] = None,
    phase_labels_trusted: bool = False,
    source: str = "other_actual_frames",
    issues: Optional[list[str]] = None,
) -> dict:
    """Generate non-authoritative visual commentary from actual visible frames."""
    imgs = list(frame_images or [])[:8]
    if not imgs:
        return {
            "available": False,
            "mode": "observation_only",
            "source": source,
            "phase_labels_trusted": False,
            "summary_zh": "",
            "summary_en": "",
            "bullets_zh": [],
            "bullets_en": [],
            "frame_notes": [],
            "issues": list(issues or []),
            "validation_issues": list(issues or []),
            "used_as_authoritative_source": False,
        }

    labels = list(frame_labels or [])[: len(imgs)]
    if len(labels) < len(imgs):
        labels.extend([None] * (len(imgs) - len(labels)))
    if not phase_labels_trusted:
        labels = [None for _ in labels]

    lines = [f"- frame {i + 1}: label={labels[i] if labels[i] else 'unknown'}" for i in range(len(labels))]
    trust_note = (
        "Phase labels are trusted and can be referenced."
        if phase_labels_trusted
        else "Phase labels are NOT trusted. Use visible-frame wording only and avoid definitive phase claims."
    )
    prompt = f"""
You are a golf visual observer. Describe what is visible from the provided chronological frames.
{trust_note}
If uncertain, explicitly state uncertainty. Do not invent exact club-ball contact certainty.

Frame list:
{chr(10).join(lines)}

Return ONLY JSON:
{{
  "summary_zh": "...",
  "summary_en": "...",
  "bullets_zh": ["..."],
  "bullets_en": ["..."],
  "frame_notes": [
    {{"index": 1, "note_zh": "...", "note_en": "..."}}
  ]
}}
""".strip()
    try:
        text, provider, key_slot = await _call_vision_ai(
            prompt, imgs, 2048, 0.2, "plus_observation", timeout_s=PLUS_OBSERVATION_TIMEOUT_S,
        )
        out = extract_json_from_response(text)
        summary_zh = str((out or {}).get("summary_zh") or "").strip() or "基于当前可见帧给出观察性结论（非正式报告）。"
        summary_en = str((out or {}).get("summary_en") or "").strip() or "Observation-only notes from current visible frames (non-authoritative)."
        bullets_zh = [str(x).strip() for x in list((out or {}).get("bullets_zh") or []) if str(x).strip()] or [
            "当前展示帧可见姿态变化，但关键阶段标签可能不可靠。"
        ]
        bullets_en = [str(x).strip() for x in list((out or {}).get("bullets_en") or []) if str(x).strip()] or [
            "Visible frames show posture changes, while key phase labels may be unreliable."
        ]
        raw_notes = list((out or {}).get("frame_notes") or [])
        frame_notes = []
        for i in range(len(imgs)):
            src = raw_notes[i] if i < len(raw_notes) and isinstance(raw_notes[i], dict) else {}
            frame_notes.append(
                {
                    "index": i + 1,
                    "label": labels[i],
                    "label_trusted": bool(phase_labels_trusted and labels[i]),
                    "note_zh": str(src.get("note_zh") or f"第{i+1}张可见帧：可见动作特征，但阶段结论仅供观察。"),
                    "note_en": str(src.get("note_en") or f"Visible frame {i+1}: motion cues are visible, phase conclusion is observational only."),
                }
            )
        return {
            "available": True,
            "mode": "authoritative_phase_report" if phase_labels_trusted else "observation_only",
            "source": source,
            "phase_labels_trusted": bool(phase_labels_trusted),
            "summary_zh": summary_zh,
            "summary_en": summary_en,
            "bullets_zh": bullets_zh[:8],
            "bullets_en": bullets_en[:8],
            "frame_notes": frame_notes,
            "issues": list(issues or []),
            "validation_issues": list(issues or []),
            "used_as_authoritative_source": False,
            "provider": provider,
            "ai_key": developer_key_label(key_slot) if key_slot is not None else None,
        }
    except Exception as e:
        logger.warning("[ai] plus_observation failed: %s", e)
        return {
            "available": True,
            "mode": "observation_only",
            "source": source,
            "phase_labels_trusted": False,
            "summary_zh": "基于当前可见帧可见到挥杆动作，但阶段标签不可靠，仅作观察参考。",
            "summary_en": "Swing motion is visible in current frames, but phase labels are unreliable; observation only.",
            "bullets_zh": ["当前可见帧可用于观察趋势，正式相位判断仍以后端门控结果为准。"],
            "bullets_en": ["Visible frames support trend observation; formal phase judgment still follows backend gate results."],
            "frame_notes": [
                {
                    "index": i + 1,
                    "label": None,
                    "label_trusted": False,
                    "note_zh": f"第{i+1}张可见帧：观察性描述，不作为正式结论。",
                    "note_en": f"Visible frame {i+1}: observational note only, not formal conclusion.",
                }
                for i in range(len(imgs))
            ],
            "issues": list(issues or []) + [f"OBSERVATION_FALLBACK:{e}"],
            "validation_issues": list(issues or []) + [f"OBSERVATION_FALLBACK:{e}"],
            "used_as_authoritative_source": False,
        }


async def analyze_with_images_only(
    frame_images: list[str],
    region: str = "global",
) -> dict:
    """Fallback: send raw frame images to AI without pose data."""
    if not frame_images:
        return _fallback_result("No frames available for analysis")
    images = frame_images[:5]
    try:
        text, provider, key_slot = await _call_vision_ai(
            IMAGE_ONLY_PROMPT, images, 4096, 0.3, "images_only", timeout_s=IMAGE_ONLY_TIMEOUT_S,
        )
        out = extract_json_from_response(text)
        out["ai_provider"] = provider
        if key_slot is not None:
            out["ai_key"] = developer_key_label(key_slot)
        return out
    except Exception as e:
        logger.error("[ai] analyze_with_images_only all providers failed: %s", e)
        return _fallback_result(str(e))


# ── Gemini-based phase detection ──

_PHASE_DETECT_PROMPT = """You are a golf swing phase analyst. These {n} images are evenly sampled from a golf swing video, numbered 1 to {n} in chronological order.

For each swing phase, identify which image number BEST represents that exact moment.
Each phase must use a different image number, and they must be in chronological order (address < takeaway < backswing < top < downswing < impact < follow_through < finish).

Camera note: the ball or clubface may be hidden (down-the-line, zoom, blur). Use body rotation, shaft direction, arm extension, and belt/hip turn — do not skip a phase because the ball is invisible.

Separation: consecutive phases should be several thumbnails apart when possible, especially impact → follow_through → finish (never pick the same thumbnail for follow_through and finish; finish must be clearly later in time).

Phase definitions:
- address: golfer standing still, club behind ball, ready to swing
- takeaway: club just starting to move back, hands below waist
- backswing: club moving up, hands between waist and shoulder height
- top: hands at highest point, maximum shoulder turn, club parallel or past parallel
- downswing: hands dropping from top, club accelerating down toward ball
- impact: moment of strike or just after — maximum hand/club speed, shaft near ball height (infer if ball not visible)
- follow_through: club past the ball, arms extending, body rotating through
- finish: swing complete, belt buckle facing target, hands high and behind head (must be after follow_through in time)

Return ONLY this JSON:
{{"address": <int>, "takeaway": <int>, "backswing": <int>, "top": <int>, "downswing": <int>, "impact": <int>, "follow_through": <int>, "finish": <int>}}"""


def _parse_phase_result(result: dict, n: int) -> dict[str, int] | None:
    phases = {}
    phase_ids = ["address", "takeaway", "backswing", "top",
                 "downswing", "impact", "follow_through", "finish"]
    prev = -1
    min_gap = max(1, n // 16)  # Minimum gap between adjacent phases
    for pid in phase_ids:
        v = result.get(pid)
        if not isinstance(v, (int, float)):
            return None
        idx = int(v) - 1
        if idx < 0 or idx >= n or idx <= prev:
            return None
        # Reject if gap to previous phase is too small (near-duplicate)
        if prev >= 0 and (idx - prev) < min_gap:
            logger.warning("[gemini] phase %s gap too small: %d - %d = %d < %d",
                           pid, idx, prev, idx - prev, min_gap)
            return None
        phases[pid] = idx
        prev = idx
    return phases


async def detect_phases_from_frames(
    frame_images: list[str],
    region: str = "global",
) -> dict[str, int] | None:
    """Use vision AI to identify which uniformly-sampled frame best
    represents each of the 8 swing phases. Returns {phase_id: frame_index}
    (0-based) or None on failure. Tries Gemini first, Qwen fallback."""
    n = len(frame_images)
    if n < 8:
        return None

    prompt = _PHASE_DETECT_PROMPT.format(n=n)
    images = frame_images[:n]

    try:
        text, provider, key_slot = await _call_vision_ai(
            prompt, images, 256, 0.1, "phase_detect", timeout_s=PHASE_DETECT_TIMEOUT_S,
        )
        logger.info(
            "[ai] phase_detect vision provider=%s ai_key=%s",
            provider,
            developer_key_label(key_slot) if key_slot is not None else "-",
        )
        result = extract_json_from_response(text)
        return _parse_phase_result(result, n)
    except Exception as e:
        logger.warning("[ai] detect_phases_from_frames all providers failed: %s", e)
        return None


# ── Result helpers ──

def _normalize_plus_result(result: dict) -> dict:
    if "posture_score" not in result:
        result["posture_score"] = round(result.get("total_score", 50) / 10, 1)
    if "primary_diagnosis" not in result:
        result["primary_diagnosis"] = {
            "title_zh": (result.get("issues_zh") or ["动作需要优化"])[0],
            "title_en": (result.get("issues") or ["Swing needs optimization"])[0],
            "status_zh": "需要注意",
            "status_en": "Needs attention",
            "ai_confidence": 60,
        }
    else:
        # Fix title/status contradiction: if title describes a problem, status cannot be positive
        pd = result["primary_diagnosis"]
        _fix_diagnosis_consistency(pd)
    if "swing_phase_evaluations" not in result:
        result["swing_phase_evaluations"] = [
            {"phase": p, "status": "pass", "note_zh": "", "note_en": ""}
            for p in ["address", "takeaway", "backswing", "top", "downswing", "impact", "follow_through", "finish"]
        ]

    # ── Score / diagnosis cross-validation ──
    total_score = result.get("total_score", 50)
    pd = result.get("primary_diagnosis", {})
    status_zh = pd.get("status_zh", "")
    status_en = pd.get("status_en", "")
    if isinstance(total_score, (int, float)):
        # High score but negative diagnosis → cap score
        if total_score >= 80 and status_zh in ("需要注意", "需要改进") or status_en.lower() in ("needs attention", "needs improvement"):
            if total_score >= 80:
                result["total_score"] = min(total_score, 72)
                result["posture_score"] = round(result["total_score"] / 10, 1)
                logger.info("[gemini] score capped %d→%d: diagnosis says needs attention", total_score, result["total_score"])
        # Low score but positive diagnosis → override status
        if total_score < 50 and (status_zh in _POSITIVE_STATUS_ZH or status_en in _POSITIVE_STATUS_EN):
            pd["status_zh"] = _CORRECTED_STATUS_ZH
            pd["status_en"] = _CORRECTED_STATUS_EN
            pd["_status_corrected"] = True
            logger.info("[gemini] status overridden: score=%d but status was positive", total_score)

    # ── Strip hallucinated fields ──
    result.pop("recommended_videos", None)
    result.pop("frequency_percent", None)
    # Also strip from primary_diagnosis if Gemini put them there
    pd.pop("recommended_videos", None)
    pd.pop("frequency_percent", None)

    return result


def _force_unknown_phase_evals_for_unreliable(result: dict) -> dict:
    phases = ["address", "takeaway", "backswing", "top", "downswing", "impact", "follow_through", "finish"]
    result["swing_phase_evaluations"] = [
        {
            "phase": p,
            "status": "unknown",
            "note_zh": "该图未通过严格相位对齐校验，仅供粗略参考。",
            "note_en": "Phase-image alignment is unverified; this note is coarse only.",
        }
        for p in phases
    ]
    result["recommended_videos"] = []
    return result


# Negative keywords in diagnosis titles that indicate a problem
_NEGATIVE_TITLE_KEYWORDS_ZH = [
    "过度", "不足", "偏", "错", "失", "弱", "差", "歪", "塌", "抬",
    "晃", "滑", "翻", "倒", "紧", "僵", "快", "慢", "短", "窄",
    "需要改进", "需要注意", "问题",
]
_NEGATIVE_TITLE_KEYWORDS_EN = [
    "excessive", "poor", "weak", "incorrect", "over-", "under-",
    "lack", "loss", "limited", "insufficient", "unstable", "inconsistent",
    "early", "late", "restricted", "collapsed", "sway", "slide",
    "needs improvement", "needs attention", "problem", "fault", "error",
]
_POSITIVE_STATUS_ZH = ["完美", "做得好"]
_POSITIVE_STATUS_EN = ["Perfect", "Good"]
_CORRECTED_STATUS_ZH = "需要注意"
_CORRECTED_STATUS_EN = "Needs attention"


def _fix_diagnosis_consistency(pd: dict) -> None:
    """If title describes a problem but status says 'Good'/'Perfect', fix the contradiction."""
    title_zh = pd.get("title_zh", "")
    title_en = (pd.get("title_en", "") or "").lower()
    status_zh = pd.get("status_zh", "")
    status_en = pd.get("status_en", "")

    title_is_negative = (
        any(kw in title_zh for kw in _NEGATIVE_TITLE_KEYWORDS_ZH)
        or any(kw in title_en for kw in _NEGATIVE_TITLE_KEYWORDS_EN)
    )
    status_is_positive = status_zh in _POSITIVE_STATUS_ZH or status_en in _POSITIVE_STATUS_EN

    if title_is_negative and status_is_positive:
        logger.warning(
            "[gemini] diagnosis contradiction: title='%s' but status='%s', correcting",
            title_zh, status_zh,
        )
        pd["status_zh"] = _CORRECTED_STATUS_ZH
        pd["status_en"] = _CORRECTED_STATUS_EN
        pd["_status_corrected"] = True


def cap_confidence(
    ai_result: dict,
    *,
    phase_validation: dict | None = None,
    hand: str = "UNKNOWN",
    club_type: str | None = None,
    tracking_quality: float = 1.0,
    **kwargs,
) -> dict:
    """Cap ai_confidence based on evidence quality. Returns analysis_reliability dict.

    Penalties:
      - phase_validation failed (soft): -30, reason ``phase_validation_soft_fail``
      - hand unknown: -15
      - club unknown: -10
      - tracking weak (<0.5): -10
      - diagnosis was auto-corrected: -10
      - sweet spot unstable or low confidence: -20 (both: -25), reason ``sweet_spot_unstable``
    Minimum confidence: 20
    """
    pd = ai_result.get("primary_diagnosis", {})
    original_conf = int(pd.get("ai_confidence", 50))
    penalty = 0
    reasons: list[str] = []

    if phase_validation is not None and not phase_validation.get("passed", True):
        penalty += 30
        reasons.append("phase_validation_soft_fail")
    if kwargs.get("phase_vision_reliable") is False:
        penalty += 20
        reasons.append("phase_vision_unreliable")
    if not hand or hand == "UNKNOWN":
        penalty += 15
        reasons.append("hand_unknown")
    if not club_type or club_type in (None, "null", "UNKNOWN", "None"):
        penalty += 10
        reasons.append("club_unknown")
    if kwargs.get("club_assumed"):
        penalty += 5
        reasons.append("club_assumed_7I")
    if tracking_quality < 0.5:
        penalty += 10
        reasons.append("tracking_weak")
    if pd.get("_status_corrected"):
        penalty += 10
        reasons.append("diagnosis_corrected")

    if kwargs.get("sweet_spot_unstable") is not None or kwargs.get("sweet_spot_confidence") is not None:
        from services.pose_strict_config import SWEET_SPOT_CONFIDENCE_LOW

        ss_u = bool(kwargs.get("sweet_spot_unstable") or False)
        ss_c_raw = kwargs.get("sweet_spot_confidence")
        try:
            ss_c = float(ss_c_raw) if ss_c_raw is not None else 1.0
        except (TypeError, ValueError):
            ss_c = 1.0
        low_c = ss_c < SWEET_SPOT_CONFIDENCE_LOW
        if ss_u or low_c:
            penalty += 25 if (ss_u and low_c) else 20
            reasons.append("sweet_spot_unstable")

    capped = max(20, original_conf - penalty)
    if capped != original_conf:
        pd["ai_confidence"] = capped
        logger.info(
            "[gemini] confidence capped: %d -> %d (penalty=%d reasons=%s)",
            original_conf, capped, penalty, reasons,
        )

    reliability = "high" if capped >= 75 else ("medium" if capped >= 50 else "low")

    return {
        "level": reliability,
        "original_confidence": original_conf,
        "capped_confidence": capped,
        "penalty": penalty,
        "reasons": reasons,
        "phase_validation": phase_validation,
    }


def _fallback_plus_result(reason: str = "") -> dict:
    base = _fallback_result(reason)
    base.update({
        "posture_score": 0.0,
        "primary_diagnosis": {
            "title_zh": f"分析失败：{reason}" if reason else "AI分析暂时不可用",
            "title_en": f"Analysis failed: {reason}" if reason else "AI analysis temporarily unavailable",
            "status_zh": "需要改进",
            "status_en": "Needs improvement",
            "ai_confidence": 0,
        },
        "additional_issues": [],
        "quick_tip_zh": "请重试",
        "quick_tip_en": "Please try again",
        "problem_description_zh": reason or "分析未能完成",
        "problem_description_en": reason or "Analysis could not be completed",
        "swing_phase_evaluations": [
            {"phase": p, "status": "pass", "note_zh": "", "note_en": ""}
            for p in ["address", "takeaway", "backswing", "top", "downswing", "impact", "follow_through", "finish"]
        ],
        "training": {
            "title_zh": "基础姿势练习",
            "title_en": "Basic Posture Practice",
            "description_zh": "从基础动作开始",
            "description_en": "Start with the basics",
            "difficulty": "normal",
            "frequency_percent": 0,
        },
        "recommended_videos": [],
    })
    return base


def _fallback_result(reason: str = "") -> dict:
    return {
        "ai_provider": "none",
        "scores": {"grip": 0, "stance": 0, "backswing": 0, "downswing": 0, "follow_through": 0},
        "total_score": 0,
        "issues": [f"AI analysis failed: {reason}" if reason else "AI analysis temporarily unavailable"],
        "issues_zh": [f"AI分析失败：{reason}" if reason else "AI分析暂时不可用"],
        "suggestions": ["Please check API key configuration and try again"],
        "suggestions_zh": ["请检查API密钥配置后重试"],
        "summary": f"Analysis could not be completed. {reason}",
        "summary_zh": f"分析未能完成。{reason}",
    }
