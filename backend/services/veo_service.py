"""
Posture practice video generation service.

Provider priority:
  1. Replicate  — REPLICATE_API_TOKEN 存在时优先使用
  2. Gemini Veo — GEMINI_API_KEY / VEO_API_KEY 回退

Replicate 模型默认 bytedance/seedance-1-lite（可用 REPLICATE_VIDEO_MODEL 覆盖）。
Veo 模型默认 veo-3.1-generate-preview（可用 VEO_MODEL 覆盖）。
多 key 轮换：遇 429 自动换 GEMINI_API_KEY_2 … _10。
"""

import asyncio
import base64
import logging
import os

import httpx

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

REPLICATE_API_BASE = "https://api.replicate.com/v1"

# bytedance/seedance-1-lite: supports 3-12s, duration param, good for instructional sports content
# override via REPLICATE_VIDEO_MODEL env var, e.g.:
#   bytedance/seedance-1-pro   (higher quality, slower)
#   minimax/video-01           (6s max, no duration param)
#   wavespeedai/wan-2.1-t2v-720p
REPLICATE_MODEL_DEFAULT = "bytedance/seedance-1-lite"

VEO_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
VEO_MODEL_DEFAULT = "veo-3.1-generate-preview"

POSTURE_VIDEO_TEMPLATES = [
    {
        "id": "stance",
        "title_zh": "站姿与膝盖弯曲",
        "title_en": "Stance & Knee Flex",
        "focus_zh": "正确的站姿宽度、脚位、重心平衡和膝盖弯曲角度",
        "focus_en": "Proper stance width, foot position, weight balance and knee flex angle",
    },
    {
        "id": "spine",
        "title_zh": "脊柱倾斜与肩髋对齐",
        "title_en": "Spine Tilt & Shoulder-Hip Alignment",
        "focus_zh": "正确的脊柱前倾角度和肩髋对齐方式",
        "focus_en": "Correct spine forward tilt angle and shoulder-hip alignment",
    },
    {
        "id": "grip",
        "title_zh": "握杆与手部设置",
        "title_en": "Grip & Hand Setup",
        "focus_zh": "正确的握杆压力、手指位置和手臂自然下垂",
        "focus_en": "Proper grip pressure, finger placement and natural arm hang",
    },
]


# ── Config helpers ─────────────────────────────────────────────────────────────

def _replicate_token() -> str:
    return os.getenv("REPLICATE_API_TOKEN", "").strip()


def _replicate_model() -> str:
    return os.getenv("REPLICATE_VIDEO_MODEL", REPLICATE_MODEL_DEFAULT).strip()


def _collect_veo_api_keys() -> list[str]:
    """VEO_API_KEY* → GEMINI_API_KEY* fallback, mirrors gemini_service."""
    keys: list[str] = []
    veo_primary = os.getenv("VEO_API_KEY", "").strip()
    if veo_primary:
        keys.append(veo_primary)
        for n in range(2, 11):
            k = os.getenv(f"VEO_API_KEY_{n}", "").strip()
            if k and k not in keys:
                keys.append(k)
        return keys
    primary = os.getenv("GEMINI_API_KEY", "").strip()
    if primary:
        keys.append(primary)
    for n in range(2, 11):
        k = os.getenv(f"GEMINI_API_KEY_{n}", "").strip()
        if k and k not in keys:
            keys.append(k)
    return keys


def _veo_model() -> str:
    return os.getenv("VEO_MODEL", VEO_MODEL_DEFAULT).strip()


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_posture_prompt(video_id: str, analysis_data: dict) -> str:
    issues = analysis_data.get("issues", [])
    scores = analysis_data.get("scores", {})
    phase_evals = analysis_data.get("swing_phase_evaluations", [])
    primary_diag = analysis_data.get("primary_diagnosis", {})

    address_eval = next((e for e in phase_evals if e.get("phase") == "address"), {})
    address_note = address_eval.get("note_en", "")

    base = (
        "Professional golf coaching instructional video. "
        "An experienced PGA golf instructor demonstrates the correct "
        "address and setup posture on a well-lit driving range. "
        "Side-angle camera view for clear visibility of body alignment. "
        "The instructor wears professional golf attire. "
        "Clean, focused, slow-motion demonstration. "
        "Cinematic quality, steady camera, natural daylight. "
    )

    if video_id == "stance":
        stance_score = scores.get("stance", 50)
        specific = (
            "FOCUS: Stance width and knee flex in address position. "
            "Instructor places feet shoulder-width apart, slight knee flex (20-25 degrees), "
            "weight evenly on balls of feet, correct ball position. "
            "Demonstrates athletic ready position step by step. "
        )
        if stance_score < 70:
            specific += "Emphasizes common stance errors and self-check for width and knee bend. "
    elif video_id == "spine":
        specific = (
            "FOCUS: Spine tilt and shoulder-hip alignment in address position. "
            "Instructor demonstrates proper hip hinge forward bend, "
            "straight spine, 30-35 degrees forward tilt, "
            "shoulders parallel to target line, hips square and level. "
            "Hip hinge motion shown clearly and slowly. "
        )
        diag_en = primary_diag.get("title_en", "")
        if any(kw in diag_en.lower() for kw in ("spine", "tilt", "posture")):
            specific += f"Special emphasis on correcting: {diag_en}. "
    elif video_id == "grip":
        grip_score = scores.get("grip", 50)
        specific = (
            "FOCUS: Grip and hand setup. Close-up of instructor placing hands on club: "
            "lead hand fingers wrapping grip, trail hand fitting naturally below, "
            "V-grooves pointing toward trail shoulder, firm but relaxed grip pressure, "
            "arms hanging naturally. Transitions to side view showing arm position. "
        )
        if grip_score < 70:
            specific += "Emphasizes relaxed grip pressure and natural arm hang. "
    else:
        specific = "Demonstrates proper overall address position setup. "

    correction = ""
    if address_note:
        correction = f"Key correction: {address_note}. "
    elif issues:
        idx = {"stance": 0, "spine": 1, "grip": 2}.get(video_id, 0)
        if idx < len(issues):
            correction = f"Key correction: {issues[idx]}. "

    return base + specific + correction


# ── Replicate provider ─────────────────────────────────────────────────────────

def _is_replicate_quota_error(status_code: int, body: str) -> bool:
    if status_code == 429:
        return True
    lo = body.lower()
    return "rate limit" in lo or "quota" in lo or "too many requests" in lo


def _replicate_terminal_success(status: str) -> bool:
    """Replicate HTTP API uses 'succeeded'; some responses use 'successful'."""
    return status in ("succeeded", "successful")


def _extract_replicate_video_url(output: object) -> str | None:
    """
    Normalize Replicate model output to a single HTTPS URL.

    Seedance / many T2V models return a string URI, or a list of FileOutput dicts
    like [{"url": "https://replicate.delivery/..."}] — older code only handled list[str].
    """
    if output is None:
        return None
    if isinstance(output, str):
        return output.strip() if output.strip().startswith("http") else None
    if isinstance(output, list):
        for item in output:
            if isinstance(item, str) and item.startswith("http"):
                return item
            if isinstance(item, dict):
                u = item.get("url") or item.get("uri")
                if isinstance(u, str) and u.startswith("http"):
                    return u
        return None
    if isinstance(output, dict):
        u = output.get("url") or output.get("uri") or output.get("video")
        if isinstance(u, str) and u.startswith("http"):
            return u
        nested = output.get("file")
        if isinstance(nested, dict):
            u2 = nested.get("url") or nested.get("uri")
            if isinstance(u2, str) and u2.startswith("http"):
                return u2
    return None


def _guess_video_mime(content: bytes, url: str, content_type: str | None) -> str:
    """Pick a Blob-compatible MIME for the browser <video> element."""
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in ("video/mp4", "video/webm", "video/quicktime", "video/x-matroska"):
        return ct
    ul = url.lower()
    if ".webm" in ul:
        return "video/webm"
    if ".mp4" in ul or ".m4v" in ul:
        return "video/mp4"
    if len(content) >= 12 and content[4:8] == b"ftyp":
        return "video/mp4"
    if len(content) >= 4 and content[:4] == b"\x1a\x45\xdf\xa3":
        return "video/webm"
    return "video/mp4"


async def _replicate_generate(
    client: httpx.AsyncClient,
    prompt: str,
    timeout_seconds: int,
) -> tuple[bytes, str]:
    """
    Create a Replicate prediction and poll until succeeded.
    Returns (raw_video_bytes, mime_type_for_browser_blob).
    """
    token = _replicate_token()
    model = _replicate_model()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Prefer": "wait=5",          # short sync wait to get prediction ID fast
    }

    # Build input payload — model-specific parameters
    owner, name = (model.split("/", 1) + [""])[:2]
    lo_model = model.lower()
    input_payload: dict = {"prompt": prompt}

    if "seedance" in lo_model:
        # seedance-1-lite / seedance-1-pro: supports 3-12s duration + aspect_ratio
        input_payload["duration"] = 8
        input_payload["aspect_ratio"] = "9:16"
    elif "minimax" in lo_model:
        # minimax/video-01 / video-01-director: max 6s, no duration param
        input_payload["prompt_optimizer"] = True
    elif "wan" in lo_model:
        # wavespeedai/wan-2.1-t2v-*: aspect_ratio supported
        input_payload["aspect_ratio"] = "9:16"
    # else: pass prompt only for unknown models

    # Use models.predictions.create (no version needed for official/latest)
    create_url = f"{REPLICATE_API_BASE}/models/{owner}/{name}/predictions"
    logger.info("[replicate] Creating prediction: model=%s", model)

    create_resp = await client.post(
        create_url,
        headers=headers,
        json={"input": input_payload},
    )

    # 200/201 = sync result ready; 202 = async accepted (prediction created, poll for result)
    if create_resp.status_code not in (200, 201, 202):
        body = create_resp.text[:600]
        if _is_replicate_quota_error(create_resp.status_code, body):
            raise _ReplicateQuotaError(
                f"Replicate 429 on model {model}: {body}"
            )
        raise RuntimeError(
            f"Replicate API error (HTTP {create_resp.status_code}): {body}"
        )

    pred = create_resp.json()
    pred_id: str = pred.get("id", "")
    get_url: str = (pred.get("urls") or {}).get("get") or f"{REPLICATE_API_BASE}/predictions/{pred_id}"

    if not pred_id:
        raise RuntimeError("Replicate returned no prediction ID")

    logger.info(
        "[replicate] Prediction created: id=%s initial_status=%s output_present=%s",
        pred_id,
        pred.get("status"),
        pred.get("output") is not None,
    )

    # ── Poll until we have a terminal success *and* a non-null output ──
    max_polls = timeout_seconds // 3
    for poll_i in range(max_polls):
        status = pred.get("status", "starting")
        out = pred.get("output")

        if _replicate_terminal_success(status) and out is not None:
            break
        if _replicate_terminal_success(status) and out is None:
            logger.warning(
                "[replicate] status=%s but output still null (poll %s), waiting…",
                status,
                poll_i,
            )
        if status in ("failed", "canceled", "cancelled", "aborted"):
            err = pred.get("error") or pred.get("detail") or "Prediction failed"
            raise RuntimeError(f"Replicate prediction {status}: {err}")

        await asyncio.sleep(3)

        try:
            poll_resp = await client.get(
                get_url,
                headers={"Authorization": f"Bearer {token}"},
            )
        except httpx.ReadTimeout:
            logger.warning("[replicate] Poll timeout, retrying...")
            continue

        if poll_resp.status_code != 200:
            logger.warning("[replicate] Poll HTTP %s, retrying...", poll_resp.status_code)
            continue

        pred = poll_resp.json()
        if poll_i % 10 == 0:
            logger.info(
                "[replicate] Polling… status=%s has_output=%s (%ds elapsed)",
                pred.get("status"),
                pred.get("output") is not None,
                (poll_i + 1) * 3,
            )
    else:
        raise RuntimeError(
            f"Replicate prediction timed out after {timeout_seconds}s (id={pred_id})"
        )

    final_status = pred.get("status", "")
    if not _replicate_terminal_success(final_status):
        raise RuntimeError(
            f"Replicate prediction not successful: status={final_status!r} id={pred_id}"
        )

    output = pred.get("output")
    video_url = _extract_replicate_video_url(output)

    if not video_url:
        logger.error("[replicate] No URL in output. raw=%s", repr(output)[:800])
        raise RuntimeError(
            f"Replicate succeeded but no video URL in output: {repr(output)[:400]}"
        )

    host = video_url.split("/")[2] if "://" in video_url else "?"
    logger.info("[replicate] Downloading video host=%s", host)

    auth_hdr = {"Authorization": f"Bearer {token}"}
    dl = await client.get(video_url, follow_redirects=True)
    if dl.status_code in (401, 403):
        logger.info("[replicate] Retrying download with Authorization header")
        dl = await client.get(
            video_url,
            headers=auth_hdr,
            follow_redirects=True,
        )
    if dl.status_code != 200:
        raise RuntimeError(
            f"Replicate video download failed: HTTP {dl.status_code} url={video_url[:100]}"
        )

    content = dl.content
    if len(content) < 1024:
        logger.error("[replicate] Body too small: %d bytes", len(content))
        raise RuntimeError(
            f"Replicate download too small ({len(content)} bytes), not a valid video"
        )

    mime = _guess_video_mime(content, video_url, dl.headers.get("content-type"))
    logger.info(
        "[replicate] OK bytes=%d header_ct=%r guessed_mime=%s magic_hex=%s",
        len(content),
        dl.headers.get("content-type"),
        mime,
        content[:12].hex(),
    )
    return content, mime


class _ReplicateQuotaError(RuntimeError):
    """Raised when Replicate returns 429 so caller can skip to next provider."""


# ── Veo provider ──────────────────────────────────────────────────────────────

def _is_veo_quota_error(status_code: int, body: str) -> bool:
    if status_code == 429:
        return True
    lo = body.lower()
    return "resource_exhausted" in lo or "quota" in lo or "rate limit" in lo


async def _veo_start_operation(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    api_key: str,
) -> str:
    url = f"{VEO_BASE_URL}/models/{model}:predictLongRunning"
    resp = await client.post(
        url,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json={"instances": [{"prompt": prompt}], "parameters": {"aspectRatio": "9:16"}},
    )
    if resp.status_code == 200:
        data = resp.json()
        op_name = data.get("name")
        if op_name:
            return op_name
        raise RuntimeError("Veo returned no operation name")
    body = resp.text[:600]
    raise RuntimeError(f"Veo start failed (HTTP {resp.status_code}): {body}")


async def _veo_generate(
    client: httpx.AsyncClient,
    prompt: str,
    timeout_seconds: int,
) -> bytes:
    """Generate via Veo with multi-key rotation on 429. Returns raw video bytes."""
    keys = _collect_veo_api_keys()
    if not keys:
        raise RuntimeError(
            "No Gemini/Veo API key configured (GEMINI_API_KEY or VEO_API_KEY)"
        )

    model = _veo_model()
    operation_name: str | None = None
    used_key: str | None = None
    last_err: Exception | None = None

    for idx, key in enumerate(keys):
        key_label = "key" if idx == 0 else f"key{idx + 1}"
        try:
            operation_name = await _veo_start_operation(client, model, prompt, key)
            used_key = key
            logger.info("[veo] Operation started with %s: %s", key_label, operation_name)
            break
        except RuntimeError as e:
            last_err = e
            if _is_veo_quota_error(0, str(e)) or "429" in str(e):
                logger.warning("[veo] %s quota-limited (429), trying next key...", key_label)
                continue
            raise

    if not operation_name or not used_key:
        raise last_err or RuntimeError("All Veo API keys exhausted (429)")

    max_polls = timeout_seconds // 10
    for poll_i in range(max_polls):
        await asyncio.sleep(10)
        try:
            poll_resp = await client.get(
                f"{VEO_BASE_URL}/{operation_name}",
                headers={"x-goog-api-key": used_key},
            )
        except httpx.ReadTimeout:
            continue

        if poll_resp.status_code != 200:
            continue

        data = poll_resp.json()
        if data.get("error"):
            raise RuntimeError(f"Veo generation error: {data['error'].get('message')}")
        if not data.get("done"):
            if poll_i % 6 == 0:
                logger.info("[veo] Still generating... (%ds)", (poll_i + 1) * 10)
            continue

        samples = (data.get("response") or {}).get(
            "generateVideoResponse", {}
        ).get("generatedSamples", [])
        if not samples:
            raise RuntimeError("Veo returned empty generatedSamples")

        video_uri = (samples[0].get("video") or {}).get("uri")
        if not video_uri:
            raise RuntimeError("Veo response missing video URI")

        logger.info("[veo] Video ready, downloading...")
        dl = await client.get(
            video_uri, headers={"x-goog-api-key": used_key}, follow_redirects=True
        )
        if dl.status_code != 200:
            raise RuntimeError(f"Veo video download failed: HTTP {dl.status_code}")

        logger.info("[veo] Downloaded %d bytes", len(dl.content))
        return dl.content

    raise RuntimeError(f"Veo generation timed out after {timeout_seconds}s")


# ── Public API ────────────────────────────────────────────────────────────────

def get_posture_templates() -> list[dict]:
    return [
        {
            "id": t["id"],
            "title_zh": t["title_zh"],
            "title_en": t["title_en"],
            "focus_zh": t["focus_zh"],
            "focus_en": t["focus_en"],
        }
        for t in POSTURE_VIDEO_TEMPLATES
    ]


async def generate_posture_video(
    video_id: str,
    analysis_data: dict,
    timeout_seconds: int = 600,
) -> dict:
    """
    Generate a single ~8s posture teaching video.

    Provider order:
      1. Replicate (REPLICATE_API_TOKEN) — seedance-1-lite by default
      2. Gemini Veo — multi-key rotation on 429

    Returns dict including video_content_type for correct <video> Blob MIME.
    """
    template = next((t for t in POSTURE_VIDEO_TEMPLATES if t["id"] == video_id), None)
    if not template:
        raise ValueError(f"Unknown posture video ID: {video_id}")

    prompt = build_posture_prompt(video_id, analysis_data)
    replicate_token = _replicate_token()
    veo_keys = _collect_veo_api_keys()

    if not replicate_token and not veo_keys:
        raise RuntimeError(
            "No video generation provider configured. "
            "Set REPLICATE_API_TOKEN and/or GEMINI_API_KEY."
        )

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_seconds, connect=30.0)
    ) as client:

        video_bytes: bytes | None = None
        video_mime = "video/mp4"
        provider_used = ""
        errors: list[str] = []

        # ── 1. Try Replicate ──
        if replicate_token:
            model = _replicate_model()
            logger.info(
                "[posture-video] Trying Replicate: id=%s model=%s", video_id, model
            )
            try:
                video_bytes, video_mime = await _replicate_generate(
                    client, prompt, timeout_seconds
                )
                provider_used = f"replicate/{model}"
            except _ReplicateQuotaError as e:
                msg = f"Replicate 429: {e}"
                logger.warning("[posture-video] %s — falling back to Veo", msg)
                errors.append(msg)
            except RuntimeError as e:
                msg = f"Replicate error: {e}"
                logger.warning("[posture-video] %s — falling back to Veo", msg)
                errors.append(msg)

        # ── 2. Fall back to Veo ──
        if video_bytes is None:
            if not veo_keys:
                raise RuntimeError(
                    "Replicate failed and no Veo key available. "
                    + " | ".join(errors)
                )
            model = _veo_model()
            logger.info("[posture-video] Trying Veo: id=%s model=%s", video_id, model)
            try:
                video_bytes = await _veo_generate(client, prompt, timeout_seconds)
                video_mime = _guess_video_mime(video_bytes, "", None)
                provider_used = f"veo/{model}"
            except RuntimeError as e:
                errors.append(f"Veo error: {e}")
                raise RuntimeError(
                    "All video generation providers failed. "
                    + " | ".join(errors)
                ) from e

    video_b64 = base64.b64encode(video_bytes).decode("utf-8")
    logger.info(
        "[posture-video] Done: id=%s provider=%s size=%d bytes",
        video_id, provider_used, len(video_bytes),
    )

    return {
        "video_id": video_id,
        "title_zh": template["title_zh"],
        "title_en": template["title_en"],
        "focus_zh": template["focus_zh"],
        "focus_en": template["focus_en"],
        "video_base64": video_b64,
        "video_size_bytes": len(video_bytes),
        "video_content_type": video_mime,
        "provider": provider_used,
        "status": "completed",
    }
