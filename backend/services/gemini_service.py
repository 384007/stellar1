import asyncio
import base64
import contextvars
import logging
import os
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
from functools import partial
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Business-level hard timeouts (seconds). Override via env if needed.
# Prevents hung external AI/network calls from blocking Lite / Plus / Pro / shared pipelines.
PHASE_DETECT_TIMEOUT_S = float(os.getenv("STELLAR_PHASE_DETECT_TIMEOUT_S", "90"))
LITE_AI_TIMEOUT_S = float(os.getenv("STELLAR_LITE_AI_TIMEOUT_S", "120"))
PLUS_AI_TIMEOUT_S = float(os.getenv("STELLAR_PLUS_AI_TIMEOUT_S", "180"))
PRO_AI_TIMEOUT_S = float(os.getenv("STELLAR_PRO_AI_TIMEOUT_S", "240"))
IMAGE_ONLY_TIMEOUT_S = float(os.getenv("STELLAR_IMAGE_ONLY_TIMEOUT_S", "120"))
PLUS_OBSERVATION_TIMEOUT_S = float(os.getenv("STELLAR_PLUS_OBSERVE_TIMEOUT_S", "90"))
NVIDIA_TIMEOUT_S = float(os.getenv("STELLAR_NVIDIA_TIMEOUT_S", "180"))

# Developer API keys: GEMINI_API_KEY (required if not using Vertex), optional GEMINI_API_KEY_2 … _10.
# Egress / geo: set GEMINI_HTTPS_PROXY or HTTPS_PROXY to tunnel via SG/JP etc.; code forces REST transport when set.
# Vertex AI (optional): set GEMINI_BACKEND=vertex, VERTEX_AI_PROJECT or GOOGLE_CLOUD_PROJECT,
# VERTEX_AI_LOCATION (default us-central1; use asia-southeast1 / asia-northeast1 for SG/Tokyo), VERTEX_GEMINI_MODEL.
# Auth: Application Default Credentials — GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json or workload identity.
# Optional extra keys: GEMINI_API_KEY_2 … GEMINI_API_KEY_10 (same format as GEMINI_API_KEY). On quota/429, try next.
#
# Reverse-proxy hosts (same env names as ``frontend/lib/gemini-proxy.ts``):
#   GEMINI_PROXY_ALI, GEMINI_PROXY_JD — mirror ``generativelanguage.googleapis.com`` REST paths.
# Modal + **CN network hint** (``gemini_modal_cn_proxy_first_context``): prefer ``GEMINI_PROXY_*`` before direct Google.
# **CF Pages forward** (``/api/modal-gemini-forward``) from Modal→Pages is usually **HTTP 403 / error 1010**
# (server-to-server / bot rules). It is **off by default**; set ``STELLAR_MODAL_CF_GEMINI_FORWARD=1`` to opt in.
# Lite/浏览器走 Pages 正常；Modal 工人应直连 Google 或配置 ``GEMINI_PROXY_*`` / ``GEMINI_HTTPS_PROXY``。
# STELLAR_CF_GEMINI_FORWARD=0 disables the forward path even when opt-in is set.
# Otherwise: **Google direct** first (``STELLAR_GEMINI_DIRECT_FIRST_TIMEOUT_S``, default 10s),
# then local GEMINI_PROXY_* when set.
#   STELLAR_GEMINI_PROXY_PHASE_TIMEOUT_S — defaults to PRO_AI_TIMEOUT_S for reverse-proxy attempts.

GEMINI_DEVELOPER_API_ORIGIN = "https://generativelanguage.googleapis.com"
NVIDIA_API_BASE_DEFAULT = "https://integrate.api.nvidia.com/v1"
NVIDIA_VIDEO_MODEL_DEFAULT = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
NVIDIA_VIDEO_MODEL_CANDIDATES = (
    NVIDIA_VIDEO_MODEL_DEFAULT,
    "nvidia/nemotron-nano-12b-v2-vl",
)
NVIDIA_KNOWN_UNHOSTED_VIDEO_MODELS = {
    "qwen/qwen3.6-35b-a3b",
}

_genai = None
_vertex_inited = False
_gemini_dev_lock = threading.Lock()
_video_ai_lock = threading.Lock()
_nvidia_rr_cursor = 0

# Set by ``gemini_modal_cn_proxy_first_context`` during Pro v3 enrich when the request hints China
# (``X-Stellar-Network-Hint: cn`` and/or ``CF-IPCountry: CN`` from Edge) and this process is the
# Modal worker — developer API then prefers ``GEMINI_PROXY_*`` before the short direct-Google attempt.
_gemini_modal_cn_proxy_first: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "gemini_modal_cn_proxy_first", default=False
)


def _stellar_modal_runtime() -> bool:
    """True on Modal Pro workers (aligned with ``prov3_analyze_single_flight_active`` heuristics)."""
    if (os.getenv("STELLAR_RUNTIME") or "").strip().lower() == "modal":
        return True
    if (os.getenv("MODAL_REGION") or "").strip():
        return True
    # Modal injects task identity in containers; covers cases where STELLAR_RUNTIME is unset.
    if (os.getenv("MODAL_TASK_ID") or "").strip():
        return True
    v3 = (os.getenv("STELLAR_MODAL_PRO_V3_ONLY") or "").strip().lower()
    return v3 in ("1", "true", "yes")


@contextmanager
def gemini_modal_cn_proxy_first_context(cn_network_hint: bool):
    """Scope for Pro v3 Gemini: Modal + CN hint → hit reverse proxies before short direct Google."""
    enabled = bool(cn_network_hint) and _stellar_modal_runtime()
    tok = _gemini_modal_cn_proxy_first.set(enabled)
    try:
        yield
    finally:
        _gemini_modal_cn_proxy_first.reset(tok)


_gemini_proxy_logged = False

_gemini_reverse_proxy_logged = False


def _reverse_proxy_origins_from_env() -> list[str]:
    out: list[str] = []
    for key in ("GEMINI_PROXY_ALI", "GEMINI_PROXY_JD"):
        v = (os.getenv(key) or "").strip().rstrip("/")
        if v and v not in out:
            out.append(v)
    return out


def _endpoint_log_label(endpoint: str) -> str:
    try:
        netloc = urlparse(endpoint).netloc
        return netloc or endpoint[:64]
    except Exception:
        return endpoint[:64]


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


def _genai_configure_developer(api_key: str, *, api_endpoint: str | None = None) -> None:
    """Configure google-generativeai.

    * ``HTTPS_PROXY`` / ``GEMINI_HTTPS_PROXY`` — tunnel egress (unchanged).
    * ``api_endpoint`` — reverse-proxy origin (``GEMINI_PROXY_*``) replacing Google host for REST paths.
    """
    import google.generativeai as genai

    global _gemini_reverse_proxy_logged
    ep = (api_endpoint or GEMINI_DEVELOPER_API_ORIGIN).strip().rstrip("/")
    direct = GEMINI_DEVELOPER_API_ORIGIN.rstrip("/")
    custom_origin = ep != direct

    _apply_gemini_outbound_proxy_env()
    has_tunnel = bool(_effective_gemini_https_proxy())
    use_rest = has_tunnel or custom_origin
    if use_rest:
        _log_gemini_proxy_once()
    if custom_origin and not _gemini_reverse_proxy_logged:
        _gemini_reverse_proxy_logged = True
        logger.info("[gemini] reverse-proxy api_endpoint=%s (ALI/JD or custom)", _endpoint_log_label(ep))

    kwargs: dict = {"api_key": api_key}
    if use_rest:
        kwargs["transport"] = "rest"
    if custom_origin:
        kwargs["client_options"] = {"api_endpoint": ep}
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


def nvidia_key_label(slot: int) -> str:
    """API-facing NVIDIA key label: nvidia_key, nvidia_key2, …."""
    if slot <= 1:
        return "nvidia_key"
    return f"nvidia_key{slot}"


def provider_key_label(provider: str, slot: Optional[int], fallback: str | None = None) -> str | None:
    if fallback:
        return fallback
    if slot is None:
        return None
    if provider == "nvidia":
        return nvidia_key_label(slot)
    if provider == "gemini":
        return developer_key_label(slot)
    return f"{provider}_key{slot if slot > 1 else ''}"


class VideoAiProviderHttpError(RuntimeError):
    def __init__(self, provider: str, status_code: int, body: str) -> None:
        super().__init__(f"{provider} HTTP {status_code}: {body}")
        self.provider = provider
        self.status_code = status_code


def _split_env_keys(raw: str) -> list[str]:
    parts = re.split(r"[,;\n\r]+", raw or "")
    return [p.strip() for p in parts if p.strip()]


def _append_unique_key(keys: list[str], key: str) -> None:
    k = (key or "").strip()
    if k and k not in keys:
        keys.append(k)


def _collect_nvidia_api_keys() -> list[tuple[int, str, str]]:
    """PatentPaper-compatible NVIDIA key discovery.

    Reads NVIDIA_API_KEY / NVIDIA_KEY, numbered _2 … _20 variants, and plural
    NVIDIA_API_KEYS / NVIDIA_KEYS. Values are never logged.
    """
    keys: list[str] = []
    env_names: list[str] = []

    def add(env_name: str, val: str) -> None:
        before = len(keys)
        _append_unique_key(keys, val)
        if len(keys) > before:
            env_names.append(env_name)

    for env_name in ("NVIDIA_API_KEY", "NVIDIA_KEY"):
        add(env_name, os.getenv(env_name, ""))
    for n in range(2, 21):
        for env_name in (f"NVIDIA_API_KEY_{n}", f"NVIDIA_KEY_{n}"):
            add(env_name, os.getenv(env_name, ""))
    for env_name in ("NVIDIA_API_KEYS", "NVIDIA_KEYS"):
        for idx, k in enumerate(_split_env_keys(os.getenv(env_name, ""))):
            add(f"{env_name}[{idx}]", k)
    return [(idx + 1, key, env_names[idx] if idx < len(env_names) else nvidia_key_label(idx + 1)) for idx, key in enumerate(keys)]


def _nvidia_api_base() -> str:
    return (
        (os.getenv("NVIDIA_API_BASE") or "").strip()
        or (os.getenv("NVIDIA_BASE_URL") or "").strip()
        or NVIDIA_API_BASE_DEFAULT
    ).rstrip("/")


def _model_looks_video_capable(model: str) -> bool:
    m = (model or "").strip().lower()
    if not m:
        return False
    if "qwen3.6-27b" in m or m in NVIDIA_KNOWN_UNHOSTED_VIDEO_MODELS:
        return False
    needles = (
        "cosmos",
        "omni",
        "vision",
        "video",
        "-vl",
        "_vl",
        "vl-",
        "vl_",
    )
    return any(x in m for x in needles)


def _nvidia_model_allowed(model: str, *, base: str) -> bool:
    m = (model or "").strip().lower()
    if not m:
        return False
    # The hosted NVIDIA API returns 404 for this older default. Keep custom NIM/self-hosted bases opt-in.
    if base.rstrip("/") == NVIDIA_API_BASE_DEFAULT and m in NVIDIA_KNOWN_UNHOSTED_VIDEO_MODELS:
        return False
    return True


def _nvidia_video_models(*, base: str | None = None) -> list[str]:
    api_base = (base or _nvidia_api_base()).rstrip("/")
    for env_name in ("NVIDIA_VIDEO_MODEL", "STELLAR_NVIDIA_VIDEO_MODEL"):
        v = (os.getenv(env_name) or "").strip()
        if v and _nvidia_model_allowed(v, base=api_base):
            return [v]
        if v:
            logger.warning("[ai] ignore_unhosted_nvidia_video_model env=%s model=%s base=%s", env_name, v, api_base)
    inherited = (os.getenv("NVIDIA_MODEL") or "").strip()
    if _model_looks_video_capable(inherited) and _nvidia_model_allowed(inherited, base=api_base):
        return [inherited]
    return list(NVIDIA_VIDEO_MODEL_CANDIDATES)


def _nvidia_video_model() -> str:
    return _nvidia_video_models()[0]


def _provider_base_default(provider: str) -> str:
    p = (provider or "").strip().lower()
    if p == "nvidia":
        return NVIDIA_API_BASE_DEFAULT
    if p == "openrouter":
        return "https://openrouter.ai/api/v1"
    if p == "openai":
        return "https://api.openai.com/v1"
    if p == "mistral":
        return "https://api.mistral.ai/v1"
    return ""


def _truthy_value(v: Any) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on", "video", "vision")


def _entry_declares_video(entry: dict[str, Any], *, explicit_video_pool: bool = False) -> bool:
    if explicit_video_pool:
        return True
    if _truthy_value(entry.get("video")) or _truthy_value(entry.get("vision")):
        return True
    caps = entry.get("capabilities")
    if isinstance(caps, str):
        return "video" in caps.lower() or "vision" in caps.lower()
    if isinstance(caps, list):
        return any("video" in str(x).lower() or "vision" in str(x).lower() for x in caps)
    role = str(entry.get("role") or "").lower()
    if "video" in role or "vision" in role or "vlm" in role:
        return True
    return _model_looks_video_capable(str(entry.get("model") or ""))


def _collect_json_video_ai_providers() -> list[dict[str, Any]]:
    """Optional explicit non-Gemini video fallback pool.

    Supports STELLAR_VIDEO_AI_KEYS_JSON / VIDEO_AI_KEYS_JSON directly. AI_KEYS_JSON is accepted only
    when an entry explicitly declares video/vision capability or uses a video-looking model name.
    Gemini, Groq, and Qwen/DashScope are intentionally excluded.
    """
    providers: list[dict[str, Any]] = []
    banned = {"gemini", "groq", "qwen", "dashscope"}
    for env_name in ("STELLAR_VIDEO_AI_KEYS_JSON", "VIDEO_AI_KEYS_JSON", "AI_KEYS_JSON"):
        raw = (os.getenv(env_name) or "").strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            logger.warning("[ai] ignore malformed %s", env_name)
            continue
        items = parsed if isinstance(parsed, list) else [parsed]
        explicit = env_name != "AI_KEYS_JSON"
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            provider = str(item.get("provider") or item.get("tier") or "openai").strip().lower()
            if provider in banned:
                continue
            key = str(item.get("api_key") or item.get("key") or "").strip()
            model = str(item.get("model") or "").strip()
            base = str(item.get("base_url") or item.get("base") or _provider_base_default(provider)).strip().rstrip("/")
            if not key or not model or not base:
                continue
            if not _entry_declares_video(item, explicit_video_pool=explicit):
                continue
            providers.append(
                {
                    "provider": provider,
                    "key": key,
                    "model": model,
                    "base": base,
                    "label": f"{provider}:{env_name}[{idx}]",
                    "env_name": f"{env_name}[{idx}]",
                }
            )
    return providers


def _collect_numbered_video_ai_providers() -> list[dict[str, Any]]:
    providers: list[dict[str, Any]] = []
    banned = {"gemini", "groq", "qwen", "dashscope"}
    for i in range(1, 21):
        key = (os.getenv(f"AI_KEY_{i}") or "").strip()
        if not key:
            continue
        provider = str(os.getenv(f"AI_KEY_{i}_TIER") or os.getenv(f"AI_KEY_{i}_PROVIDER") or "openai").strip().lower()
        if provider in banned:
            continue
        model = (os.getenv(f"AI_KEY_{i}_MODEL") or "").strip()
        base = (
            (os.getenv(f"AI_KEY_{i}_BASE") or "").strip()
            or (os.getenv(f"AI_KEY_{i}_BASE_URL") or "").strip()
            or _provider_base_default(provider)
        ).rstrip("/")
        if not model or not base:
            continue
        if not (
            _truthy_value(os.getenv(f"AI_KEY_{i}_VIDEO"))
            or _truthy_value(os.getenv(f"AI_KEY_{i}_VISION"))
            or _model_looks_video_capable(model)
        ):
            continue
        providers.append(
            {
                "provider": provider,
                "key": key,
                "model": model,
                "base": base,
                "label": f"{provider}:AI_KEY_{i}",
                "env_name": f"AI_KEY_{i}",
            }
        )
    return providers


def _ordered_video_ai_providers() -> list[dict[str, Any]]:
    """NVIDIA keys first, round-robin start slot; explicit video fallback providers after."""
    global _nvidia_rr_cursor
    nvidia_key_entries = _collect_nvidia_api_keys()
    nvidia_entries: list[dict[str, Any]] = []
    if nvidia_key_entries:
        with _video_ai_lock:
            start = _nvidia_rr_cursor % len(nvidia_key_entries)
            _nvidia_rr_cursor += 1
        nvidia_key_entries = nvidia_key_entries[start:] + nvidia_key_entries[:start]
        base = _nvidia_api_base()
        for model in _nvidia_video_models(base=base):
            for slot, key, env_name in nvidia_key_entries:
                nvidia_entries.append(
                    {
                        "provider": "nvidia",
                        "key": key,
                        "model": model,
                        "base": base,
                        "label": nvidia_key_label(slot),
                        "slot": slot,
                        "env_name": env_name,
                    }
                )

    providers = nvidia_entries + _collect_json_video_ai_providers() + _collect_numbered_video_ai_providers()
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for p in providers:
        ident = (
            str(p.get("provider") or ""),
            str(p.get("base") or ""),
            str(p.get("key") or ""),
            str(p.get("model") or ""),
        )
        if not ident[2] or ident in seen:
            continue
        seen.add(ident)
        out.append(p)
    return out


def _is_gemini_quota_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "429" in msg or "resource exhausted" in msg or "quota" in msg or "rate limit" in msg:
        return True
    try:
        import google.api_core.exceptions as gexc

        return isinstance(exc, gexc.ResourceExhausted)
    except ImportError:
        return False


def _cf_gemini_forward_enabled() -> bool:
    v = (os.getenv("STELLAR_CF_GEMINI_FORWARD") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _modal_cf_gemini_forward_enabled() -> bool:
    """Modal→CF Pages Gemini is opt-in; default off (avoids wasted round-trips and 1010)."""
    v = (os.getenv("STELLAR_MODAL_CF_GEMINI_FORWARD") or "").strip().lower()
    return v in ("1", "true", "yes")


def _cf_pages_origin_for_gemini_forward() -> str:
    """Pages origin for ``/api/modal-gemini-forward`` (wrangler project name default)."""
    for k in ("STELLAR_CF_PAGES_ORIGIN", "FRONTEND_URL"):
        v = (os.getenv(k) or "").strip().rstrip("/")
        if v.startswith("http://") or v.startswith("https://"):
            return v
    return "https://stellar-ai.pages.dev"


def _gemini_via_cf_pages_generate_sync(
    prompt: str,
    images: list[str],
    max_tokens: int,
    temperature: float,
    keys: list[str],
    model_name: str,
) -> tuple[str, int]:
    """Call Gemini through Cloudflare Pages Edge (same ``getGeminiHosts`` as Lite — zero GEMINI_PROXY_* on Modal)."""
    import json
    import urllib.error
    import urllib.request

    base = _cf_pages_origin_for_gemini_forward()
    url = f"{base}/api/modal-gemini-forward"
    parts: list[dict] = [{"text": prompt}]
    for img_b64 in images:
        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": img_b64}})
    body_obj = {
        "model": model_name,
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
        "cn_network_hint": True,
    }
    payload = json.dumps(body_obj).encode("utf-8")
    auth_key = keys[0]
    timeout_s = max(30, int(float(os.getenv("STELLAR_CF_GEMINI_FORWARD_TIMEOUT_S", str(int(PRO_AI_TIMEOUT_S))))))
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_key}",
            "X-Stellar-Modal-Gemini-Forward": "1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"CF Gemini forward HTTP {e.code}: {detail}") from e

    data = json.loads(raw)
    if isinstance(data, dict) and "detail" in data and "candidates" not in data:
        raise RuntimeError(str(data.get("detail", data)))
    cands = data.get("candidates") if isinstance(data, dict) else None
    if not cands:
        raise RuntimeError("CF forward: empty candidates")
    parts_out = (cands[0].get("content") or {}).get("parts") or []
    if not parts_out:
        raise RuntimeError("CF forward: empty parts")
    text = (parts_out[0].get("text") or "").strip()
    if not text:
        raise RuntimeError("CF forward: empty text")
    slot = int(data.get("_stellar_key_slot") or 1) if isinstance(data, dict) else 1
    ak = developer_key_label(slot)
    logger.info("[gemini] developer_api via_cf_pages ok ai_key=%s", ak)
    print(f"[stellar-ai] ai_key={ak} gemini_via=cf_pages base={base}", flush=True)
    return text, slot


def _call_gemini_developer_sync(
    prompt: str,
    images: list[str],
    max_tokens: int,
    temperature: float,
) -> tuple[str, int]:
    keys = _collect_developer_api_keys()
    if not keys:
        raise RuntimeError("GEMINI_API_KEY not configured on server")

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    proxies = _reverse_proxy_origins_from_env()
    cn_modal_route = bool(_gemini_modal_cn_proxy_first.get())
    # CF Pages forward: Modal egress → Pages is commonly blocked (1010); default skip unless explicitly enabled.
    cf_forward = (
        not _use_vertex()
        and not proxies
        and _stellar_modal_runtime()
        and _cf_gemini_forward_enabled()
        and _modal_cf_gemini_forward_enabled()
    )
    if cf_forward:
        logger.info(
            "[gemini] Modal: STELLAR_MODAL_CF_GEMINI_FORWARD on — trying CF Pages Gemini forward (%s)",
            _cf_pages_origin_for_gemini_forward(),
        )
        try:
            return _gemini_via_cf_pages_generate_sync(
                prompt, images, max_tokens, temperature, keys, model_name,
            )
        except Exception as e:
            logger.warning(
                "[gemini] CF Pages Gemini forward failed: %s — falling back to direct Google / proxy path",
                e,
            )

    import google.generativeai as genai

    content_parts: list = [prompt]
    for img_b64 in images:
        content_parts.append({"mime_type": "image/jpeg", "data": img_b64})

    direct = GEMINI_DEVELOPER_API_ORIGIN.rstrip("/")
    proxy_first = cn_modal_route and bool(proxies)
    if _stellar_modal_runtime() and not proxies and cn_modal_route and not _cf_gemini_forward_enabled():
        logger.warning(
            "[gemini] Modal+CN hint: STELLAR_CF_GEMINI_FORWARD off and no GEMINI_PROXY_* — using direct Google only",
        )
    direct_timeout = max(1.0, float(os.getenv("STELLAR_GEMINI_DIRECT_FIRST_TIMEOUT_S", "10")))
    proxy_timeout = max(direct_timeout, float(os.getenv("STELLAR_GEMINI_PROXY_PHASE_TIMEOUT_S", str(PRO_AI_TIMEOUT_S))))

    def _do_generate(key: str, endpoint: str):
        _genai_configure_developer(key, api_endpoint=endpoint)
        model = genai.GenerativeModel(model_name)
        return model.generate_content(
            content_parts,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            ),
        )

    def _finalize_response(response: Any, idx: int, ep: str) -> tuple[str, int]:
        if not response.candidates or not response.text:
            raise RuntimeError("Gemini returned empty response")
        slot = idx + 1
        ak = developer_key_label(slot)
        logger.info(
            "[gemini] developer_api ok endpoint=%s ai_key=%s",
            _endpoint_log_label(ep),
            ak,
        )
        print(f"[stellar-ai] ai_key={ak} gemini_endpoint={_endpoint_log_label(ep)}", flush=True)
        return response.text, slot

    last_err: BaseException | None = None
    direct_timed_out = False

    def _run_with_deadline(fn_submit, timeout_s: float):
        """Run one generate in a side thread; on timeout do not wait for the orphan call (Modal-friendly)."""
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            fut = fn_submit(pool)
            return fut.result(timeout=timeout_s)
        finally:
            pool.shutdown(wait=False)

    with _gemini_dev_lock:
        # Phase 1 — Google direct only, short deadline (skipped on Modal when client hints CN + proxies exist).
        if not proxy_first:
            for idx, key in enumerate(keys):
                try:
                    response = _run_with_deadline(
                        lambda p, k=key: p.submit(_do_generate, k, direct),
                        direct_timeout,
                    )
                    return _finalize_response(response, idx, direct)
                except FuturesTimeoutError:
                    direct_timed_out = True
                    last_err = TimeoutError(f"gemini direct first-try timeout {direct_timeout}s")
                    logger.warning(
                        "[gemini] direct Google no response within %ss (key_slot=%s) → %s",
                        direct_timeout,
                        idx + 1,
                        "GEMINI_PROXY_* phase" if proxies else f"retry direct timeout={proxy_timeout}s",
                    )
                    break
                except Exception as e:
                    last_err = e
                    if idx < len(keys) - 1 and _is_gemini_quota_error(e):
                        logger.warning(
                            "[gemini] direct key slot %s quota/rate limited (%s), trying next key",
                            idx + 1,
                            e,
                        )
                        print(
                            f"[stellar-ai] ai_key slot {idx + 1} hit quota/rate limit, trying next key",
                            flush=True,
                        )
                        continue
                    if proxies:
                        logger.warning(
                            "[gemini] direct failed (%s), trying reverse-proxy hosts",
                            e,
                        )
                        break
                    raise
        else:
            logger.info(
                "[gemini] Modal + CN network hint: using GEMINI_PROXY_* first (skip direct %ss phase)",
                int(direct_timeout),
            )

        # Phase 2a — ALI / JD with long timeout
        for ep in proxies:
            for idx, key in enumerate(keys):
                try:
                    response = _run_with_deadline(
                        lambda p, k=key, e=ep: p.submit(_do_generate, k, e),
                        proxy_timeout,
                    )
                    return _finalize_response(response, idx, ep)
                except Exception as e:
                    last_err = e
                    if idx < len(keys) - 1 and _is_gemini_quota_error(e):
                        logger.warning(
                            "[gemini] proxy=%s slot %s quota (%s), next key",
                            _endpoint_log_label(ep),
                            idx + 1,
                            e,
                        )
                        continue
                    logger.warning(
                        "[gemini] proxy=%s failed (%s), next origin or fallback",
                        _endpoint_log_label(ep),
                        e,
                    )
                    break

        # Phase 2b — direct timed out but no proxies: one long-timeout direct pass per key
        if direct_timed_out and not proxies:
            for idx, key in enumerate(keys):
                try:
                    response = _run_with_deadline(
                        lambda p, k=key: p.submit(_do_generate, k, direct),
                        proxy_timeout,
                    )
                    return _finalize_response(response, idx, direct)
                except Exception as e:
                    last_err = e
                    if idx < len(keys) - 1 and _is_gemini_quota_error(e):
                        continue
                    break

        # Phase 2c — proxies failed: final direct attempts with long timeout
        if proxies and last_err is not None:
            for idx, key in enumerate(keys):
                try:
                    response = _run_with_deadline(
                        lambda p, k=key: p.submit(_do_generate, k, direct),
                        proxy_timeout,
                    )
                    return _finalize_response(response, idx, direct)
                except Exception as e:
                    last_err = e
                    if idx < len(keys) - 1 and _is_gemini_quota_error(e):
                        continue
                    break

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
  "summary_zh": "中文总结（200字以内）",
  "detected_club": {{
    "club_type": "<UNKNOWN or 1W|3W|5W|3I|4I|5I|6I|7I|8I|9I|PW|AW|SW|LW|PT>",
    "club_group": "<WOOD|IRON|WEDGE|PUTTER>",
    "confidence": <0.0-1.0>,
    "hand": "<R or L>"
  }}
}}

If more than eight JPEGs are attached: images 1–8 are the phase strip (address→finish); images 9–11 (when present) are extra samples from ~25%, ~40%, and ~60% of the **source video** — use them only to refine detected_club, not to relabel which strip frame is which phase.

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

**Required:** You must still return the **full** JSON schema above with substantive content: non-zero scores where justified by pose_data, at least three issues and three suggestions in **both** EN and ZH, and full summary/summary_zh — same depth as a normal coaching report. Uncertainty affects how you describe image-to-phase alignment, not whether you analyze.
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
    "day1": {{"focus": "中文主题", "drills": ["中文练习1", "中文练习2"], "duration": "30 min"}},
    "day2": {{"focus": "...", "drills": ["...", "..."], "duration": "30 min"}},
    "day3": {{"focus": "...", "drills": ["...", "..."], "duration": "30 min"}},
    "day4": {{"focus": "...", "drills": ["...", "..."], "duration": "30 min"}},
    "day5": {{"focus": "...", "drills": ["...", "..."], "duration": "30 min"}},
    "day6": {{"focus": "...", "drills": ["...", "..."], "duration": "30 min"}},
    "day7": {{"focus": "复习", "drills": ["中文复习1", "中文复习2"], "duration": "rest/review"}}
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
- training_plan: each day's "focus" and both "drills" strings MUST be Simplified Chinese.
"""

PROV3_REPORT_PROMPT = """You are an elite PGA-level coach writing a truthful coaching report from motion metadata only.

No image selection task. Do NOT choose frames. Do NOT mention contact sheets, "image 1-8", or any AI frame picking.

SCREEN / NO-IMAGES MODE: There are NO attached photos. That is expected. MOTION_CONTEXT alone (phase timestamps + dense motion proxy per phase + swing window) is ENOUGH to write a full phase-based coaching report. Do NOT shorten the report or refuse depth because of "no images". Never output a trivial one-paragraph reply.

You are given MOTION_CONTEXT for fixed 8 phases at 240fps:
Address, Takeaway, Backswing, Top, Downswing, Impact, Follow-through, Finish.
Treat these phases and their order/timing as ground truth.

TRUTH FIRST (non-negotiable):
- Say only what MOTION_CONTEXT supports. If proxies, swing_window_s, or phase spacing suggest there is NO credible golf swing (e.g. near-zero dense_motion_proxy everywhere, collapsed window, nonsensical timing), say so clearly in summary/summary_zh and in keyframe_evaluations — do NOT fabricate tour-level swing coaching.
- If prov3_screen_pipeline, low_trust_preview_only, or prov3_fail_reasons indicate screen/recapture or untrusted phases, state honestly that quality/alignment may be poor (blur, moiré, untrusted labels) when inferring from metadata.
- You have NO pixels: never claim you "saw" the golfer/club/ball; infer from numbers only. If uncertain, say so in both languages.

PER-KEYFRAME OUTPUT (mandatory):
- Include "keyframe_evaluations": an array with EXACTLY one object per row in MOTION_CONTEXT.keyframes, in THE SAME ORDER as that array.
- Each object: {{"phase": "<same phase string as row>", "score": <0-100>, "action_assessment_en": "1-4 sentences", "action_assessment_zh": "1-4句"}}
- Assess that phase using timestamp_s, frame_index, dense_motion_proxy vs neighbors and swing_window_s. If signal is insufficient, use a low score and say "insufficient motion signal" / "数据不足以判断" in both languages.

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
  "keyframe_evaluations": [
    {{"phase": "address", "score": 0, "action_assessment_en": "...", "action_assessment_zh": "..."}}
  ],
  "training_plan": {{
    "day1": {{"focus": "中文训练主题", "drills": ["中文练习要点1", "中文练习要点2"], "duration": "30 min"}},
    "day2": {{"focus": "...", "drills": ["...", "..."], "duration": "30 min"}},
    "day3": {{"focus": "...", "drills": ["...", "..."], "duration": "30 min"}},
    "day4": {{"focus": "...", "drills": ["...", "..."], "duration": "30 min"}},
    "day5": {{"focus": "...", "drills": ["...", "..."], "duration": "30 min"}},
    "day6": {{"focus": "...", "drills": ["...", "..."], "duration": "30 min"}},
    "day7": {{"focus": "复习与录像对比", "drills": ["中文复习要点1", "中文复习要点2"], "duration": "20 min"}}
  }}
}}

training_plan MUST include day1 through day7; focus and BOTH drill strings MUST be Simplified Chinese (specific cues).
keyframe_evaluations length MUST equal len(MOTION_CONTEXT.keyframes); order MUST match that array.
"""

PROV3_REPORT_PROMPT_PASS2 = """PASS 2 — Your previous output was REJECTED for being too thin, empty, or non-phase-specific.

You are an elite PGA coach. MOTION_CONTEXT JSON is the ONLY source. No images. Do NOT choose frames.

STRICT OUTPUT RULES:
1) issues, issues_zh, suggestions, suggestions_zh: **minimum 4 items each** (not 2, not 3). Every line MUST start with a phase name (English: Address, Takeaway, Backswing, Top, Downswing, Impact, Follow-through, Finish / Chinese: 站姿, 起杆, 上杆, 顶点, 下杆, 触球, 送杆, 收杆).
2) NO vague filler. Each issue names a phase + a concrete timing/motion-proxy observation from the JSON.
3) If Impact timing looks tight vs the downswing burst, write "**Impact**: ..." explicitly. Same for **Follow-through** and **Finish** when post-impact spacing or exit proxy is weak.
4) summary: **500-750 English words**, multiple paragraphs, phase-ordered narrative (Address → … → Finish), honest hedging only as a phrase — not as a substitute for content.
5) summary_zh: **600-950 汉字**，多段落，阶段清晰；禁止仅用两三句概括。
6) keyframe_evaluations: same rules as pass-1 formal prompt — one object per MOTION_CONTEXT.keyframes row, same order; honest scores and bilingual action text from metadata only.

Do not claim you lack information because there are no pictures — the numeric phase timeline is sufficient.

MOTION_CONTEXT (JSON):
{motion_context}

Return ONLY the same JSON schema as before (total_score, scores, issues, issues_zh, suggestions, suggestions_zh, summary, summary_zh, keyframe_evaluations, training_plan day1-day7). training_plan: each day "focus" and both "drills" entries MUST be Simplified Chinese.
"""

PROV3_REPORT_LIMITED_PROMPT = """BELOW HIGH-TRUST BAR — Stellar Pro v3 (screen / keyframe verification did not reach the studio gate).

You still have MOTION_CONTEXT: phase timestamps, dense_motion_proxy per phase, swing_window_s, fps. Write a **substantive** coaching report from that data — same structure as formal mode. The player should read **what the timeline and proxies suggest**, not a wall of "low trust" disclaimers.

One-line disclosure only (do not repeat "低信任 / limited trust" throughout the prose):
- Start summary_zh with ONE short sentence, e.g. 「自动关键帧未达最高置信档；下文依据时间线与能量代理。」
- Start summary with ONE short English sentence with the same meaning.
- After that, write like a real coach: phase-by-phase observations (Address → … → Finish), concrete issues, and drills. Do not pad summaries with repeated trust warnings.

Truth (non-negotiable):
- Ground claims in MOTION_CONTEXT only. Do NOT invent ball flight, clubface aim, or details that need pixels.
- If the window is collapsed or proxies show no credible swing, say so **once**, then still give useful tempo, balance, and re-recording guidance.
- Do NOT claim you "saw" the golfer; infer from numbers only.

PER-KEYFRAME (mandatory):
- "keyframe_evaluations": EXACTLY one object per MOTION_CONTEXT.keyframes row, SAME ORDER.
- Each: {{"phase": "<same as row>", "score": <0-100>, "action_assessment_en": "1-4 sentences: what proxy/timing imply", "action_assessment_zh": "1-4句：该阶段能量/间隔说明了什么"}}
- Avoid repeating "低信任" in every row; at most one mild hedge in the whole array if needed.

training_plan day1–day7:
- "focus": Chinese training theme.
- "drills": exactly **two strings, both in Simplified Chinese** (specific cues or steps).

issues / issues_zh / suggestions / suggestions_zh: minimum 3 each; phase-prefixed like formal mode (English phases: Address, Takeaway, … / Chinese: 站姿, 起杆, …).

summary: **380–650 English words** after the opening disclosure line — phase-ordered, concrete.
summary_zh: **480–800 汉字** after the opening disclosure line — 同样按阶段展开，写清楚数据支撑的结论。

MOTION_CONTEXT (JSON):
{motion_context}

Return ONLY valid JSON with the SAME schema as formal Pro v3 reports:
{{
  "total_score": <0-100>,
  "scores": {{"grip": <0-100>, "stance": <0-100>, "backswing": <0-100>, "downswing": <0-100>, "follow_through": <0-100>}},
  "issues": ["Phase: ...", "..."],
  "issues_zh": ["阶段：...", "..."],
  "suggestions": ["Phase: ...", "..."],
  "suggestions_zh": ["阶段：...", "..."],
  "summary": "...",
  "summary_zh": "...",
  "keyframe_evaluations": [{{"phase": "address", "score": 0, "action_assessment_en": "...", "action_assessment_zh": "..."}}],
  "training_plan": {{ "day1": {{"focus": "中文主题", "drills": ["中文练习1", "中文练习2"], "duration": "30 min"}}, ... "day7": {{...}} }}
}}

keyframe_evaluations length MUST equal len(MOTION_CONTEXT.keyframes).
"""

PROV3_DETECTED_CLUB_APPEND = """
CLUB IMAGES (required in the SAME JSON object; narrative coaching still grounded only in MOTION_CONTEXT):
Exactly three JPEGs are attached in chronological order from the same clip at approximately 25%, 40%, and 60% of the video duration.
Use them ONLY to fill this top-level field (do not use them to override phase timing in MOTION_CONTEXT):
  "detected_club": {{"club_type": "<1W|3W|...|UNKNOWN>", "club_group": "<WOOD|IRON|WEDGE|PUTTER>", "confidence": <0-1>, "hand": "<R|L>"}}
If the club is not visible, use UNKNOWN, IRON, 0.0, R.
"""

PLUS_PROMPT_APPEND_GEMINI_VISUAL_OBS = """
SECOND OUTPUT (same single JSON response — same eight phase images as above):
Include top-level key "gemini_visual_observation" with this shape (observational commentary; may be softer than the formal Plus block):
{{
  "summary_zh": "<short>",
  "summary_en": "<short>",
  "bullets_zh": ["...", "..."],
  "bullets_en": ["...", "..."],
  "frame_notes": [{{"index": 1, "note_zh": "...", "note_en": "..."}}, ... one per image in order]
}}
Rules match the product visual-observer: if phase_images_reliable is false, avoid definitive per-phase claims; use uncertainty wording. frame_notes length should match the number of attached strip images (up to 8).
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
        _genai_configure_developer(keys[0], api_endpoint=None)
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
    ctx = contextvars.copy_context()

    def _runner() -> tuple[str, int]:
        return _call_gemini_developer_sync(prompt, images, max_tokens, temperature)

    return await loop.run_in_executor(None, ctx.run, _runner)


async def run_gemini_vision(
    prompt: str,
    images_b64: list[str],
    *,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> tuple[str, Optional[int]]:
    """Backward-compatible name; now routes to NVIDIA/video-capable AI, not Gemini."""
    text, _provider, _key_label = await _call_video_ai(
        prompt,
        images_b64,
        [],
        max_tokens,
        temperature,
        "compat_run_gemini_vision",
    )
    return text, None


# ── NVIDIA / explicit video AI pool ──

def _media_content_parts(
    prompt: str,
    images: list[str] | None = None,
    videos: list[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img_b64 in (images or [])[:8]:
        b64 = (img_b64 or "").strip()
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
    for vid_b64, mime in (videos or [])[:1]:
        b64 = (vid_b64 or "").strip()
        if not b64:
            continue
        mt = (mime or "video/mp4").strip() or "video/mp4"
        content.append({
            "type": "video_url",
            "video_url": {"url": f"data:{mt};base64,{b64}"},
        })
    return content


def _openai_compat_extract_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        return ""
    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(part))
        text = "\n".join(x for x in parts if x)
    else:
        text = str(content or "")
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


def _openai_compat_chat_sync(
    provider: dict[str, Any],
    prompt: str,
    images: list[str],
    videos: list[tuple[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    import httpx

    base = str(provider.get("base") or "").rstrip("/")
    model = str(provider.get("model") or "").strip()
    key = str(provider.get("key") or "").strip()
    name = str(provider.get("provider") or "video_ai").strip().lower()
    if not base or not model or not key:
        raise RuntimeError(f"{name}: incomplete provider config")

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": _media_content_parts(prompt, images, videos)}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if name == "nvidia":
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    timeout_s = max(30.0, float(os.getenv("STELLAR_VIDEO_AI_PROVIDER_TIMEOUT_S", str(NVIDIA_TIMEOUT_S))))
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(f"{base}/chat/completions", headers=headers, json=payload)
    if resp.status_code != 200:
        body = resp.text[:500]
        raise VideoAiProviderHttpError(name, resp.status_code, body)
    data = resp.json()
    text = _openai_compat_extract_text(data)
    if not text:
        raise RuntimeError(f"{name} returned empty response")
    return text


def _call_video_ai_sync(
    prompt: str,
    images: list[str] | None = None,
    videos: list[tuple[str, str]] | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    label: str = "vision",
) -> tuple[str, str, Optional[str]]:
    providers = _ordered_video_ai_providers()
    if not providers:
        raise RuntimeError("NVIDIA_API_KEY not configured; no explicit video AI provider configured")

    last_err: BaseException | None = None
    skipped_404_models: set[tuple[str, str, str]] = set()
    for p in providers:
        provider = str(p.get("provider") or "video_ai").lower()
        key_label = str(p.get("label") or provider_key_label(provider, p.get("slot")) or provider)
        model = str(p.get("model") or "")
        base = str(p.get("base") or "").rstrip("/")
        model_ident = (provider, base, model)
        if model_ident in skipped_404_models:
            continue
        try:
            text = _openai_compat_chat_sync(
                p,
                prompt,
                list(images or []),
                list(videos or []),
                max_tokens,
                temperature,
            )
            logger.info("[ai] video_ai_ok provider=%s label=%s ai_key=%s model=%s", provider, label, key_label, model)
            print(f"[stellar-ai] vision label={label} provider={provider} ai_key={key_label}", flush=True)
            return text, provider, key_label
        except Exception as e:
            last_err = e
            logger.warning(
                "[ai] video_ai_fail provider=%s label=%s ai_key=%s model=%s err=%s",
                provider,
                label,
                key_label,
                model,
                e,
            )
            if isinstance(e, VideoAiProviderHttpError) and e.status_code == 404:
                skipped_404_models.add(model_ident)
            continue
    raise RuntimeError(f"All video AI providers failed for {label}: {last_err}")


async def _call_video_ai(
    prompt: str,
    images: list[str] | None = None,
    videos: list[tuple[str, str]] | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    label: str = "vision",
) -> tuple[str, str, Optional[str]]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        partial(
            _call_video_ai_sync,
            prompt,
            list(images or []),
            list(videos or []),
            max_tokens,
            temperature,
            label,
        ),
    )


def run_video_ai_sync(
    prompt: str,
    images_b64: Optional[list[str]] = None,
    videos_b64: Optional[list[tuple[str, str]]] = None,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    label: str = "vision",
) -> tuple[str, str, Optional[str]]:
    """Synchronous public video-capable AI call. NVIDIA keys are primary and round-robin."""
    return _call_video_ai_sync(
        prompt,
        list(images_b64 or []),
        list(videos_b64 or []),
        max_tokens,
        temperature,
        label,
    )


# ── Unified NVIDIA video-capable caller ──

async def _call_vision_ai(
    prompt: str,
    images: list[str],
    max_tokens: int = 4096,
    temperature: float = 0.3,
    label: str = "vision",
    *,
    timeout_s: float,
) -> tuple[str, str, Optional[str]]:
    """Use NVIDIA/video-capable AI keys only.

    Returns (response_text, provider, key_label). NVIDIA keys are the primary pool and are
    round-robin. Gemini, Groq, and Qwen are intentionally not used here.

    ``timeout_s`` caps wall-clock time for the whole provider chain.
    """
    async def _inner() -> tuple[str, str, Optional[str]]:
        return await _call_video_ai(prompt, images, [], max_tokens, temperature, label)

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
    club_sample_images_b64: Optional[list[str]] = None,
) -> dict:
    """Lite: one vision call — optional three extra JPEGs (clip 25/40/60 positions) plus 8-strip for ``detected_club``."""
    _ = region
    base = LITE_PROMPT.format(pose_data=json.dumps(pose_data, indent=2))
    prompt = base if phase_images_reliable else (base + LITE_PROMPT_APPEND_PHASE_UNRELIABLE)
    strip = [x for x in (keyframe_images or []) if isinstance(x, str) and x.strip()][:8]
    extras = [x for x in (club_sample_images_b64 or []) if isinstance(x, str) and x.strip()][:3]
    while len(extras) < 3 and extras:
        extras.append(extras[-1])
    extras = extras[:3]
    images = strip + extras
    label = "lite_unified" if extras else "lite"
    try:
        text, provider, key_label = await _call_vision_ai(
            prompt, images, 2304, 0.3, label, timeout_s=LITE_AI_TIMEOUT_S,
        )
        out = extract_json_from_response(text)
        out["ai_provider"] = provider
        if key_label:
            out["ai_key"] = key_label
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
        text, provider, key_label = await _call_vision_ai(
            prompt, images, 8192, 0.2, "pro", timeout_s=PRO_AI_TIMEOUT_S,
        )
        out = extract_json_from_response(text)
        out["ai_provider"] = provider
        if key_label:
            out["ai_key"] = key_label
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
        text, provider, key_label = await _call_vision_ai(
            prompt, [], 8192, 0.2, "stellar_pro_report", timeout_s=PRO_AI_TIMEOUT_S,
        )
        out = extract_json_from_response(text)
        out["ai_provider"] = provider
        if key_label:
            out["ai_key"] = key_label
        return out
    except Exception as e:
        logger.error("[ai] analyze_stellar_pro_report_only failed: %s", e)
        return _fallback_result(str(e))


async def analyze_prov3_motion_report_only(
    motion_context: dict,
    region: str = "global",
    *,
    use_strong_prompt: bool = False,
    max_tokens: int = 10240,
    call_label: str = "prov3_report",
    report_mode: str = "formal",
    club_images_b64: Optional[list[str]] = None,
) -> dict:
    """Pro v3: motion JSON report; optional 3 club JPEGs in the **same** vision call as the report."""
    _ = region
    if (report_mode or "").strip().lower() == "limited":
        template = PROV3_REPORT_LIMITED_PROMPT
        temp = 0.25
    else:
        template = PROV3_REPORT_PROMPT_PASS2 if use_strong_prompt else PROV3_REPORT_PROMPT
        temp = 0.2 if not use_strong_prompt else 0.15
    prompt = template.format(
        motion_context=json.dumps(motion_context, indent=2, ensure_ascii=False),
    )
    imgs: list[str] = [x for x in (club_images_b64 or []) if isinstance(x, str) and x.strip()][:3]
    while len(imgs) < 3 and imgs:
        imgs.append(imgs[-1])
    imgs = imgs[:3]
    if imgs:
        prompt = prompt + PROV3_DETECTED_CLUB_APPEND
    try:
        text, provider, key_label = await _call_vision_ai(
            prompt, imgs, max_tokens, temp, call_label, timeout_s=PRO_AI_TIMEOUT_S,
        )
        out = extract_json_from_response(text)
        out["ai_provider"] = provider
        if key_label:
            out["ai_key"] = key_label
        return out
    except Exception as e:
        logger.error("[ai] analyze_prov3_motion_report_only failed: %s", e)
        return _fallback_result(str(e))


async def analyze_swing_plus(
    pose_data: dict,
    keyframe_images: Optional[list[str]] = None,
    region: str = "global",
    phase_images_reliable: bool = True,
    phase_c_context: Optional[dict[str, Any]] = None,
    *,
    include_visual_observation_bundle: bool = False,
) -> dict:
    """Plus: one vision call; optional ``gemini_visual_observation`` subtree in the same JSON."""
    _ = region
    base = PLUS_PROMPT.format(pose_data=json.dumps(pose_data, indent=2))
    prompt = base if phase_images_reliable else (base + PLUS_PROMPT_APPEND_PHASE_UNRELIABLE)
    if phase_c_context:
        prompt = prompt + PLUS_PROMPT_APPEND_PHASE_C.format(
            phase_c_json=json.dumps(phase_c_context, ensure_ascii=False, separators=(",", ":")),
        )
    if include_visual_observation_bundle:
        prompt = prompt + PLUS_PROMPT_APPEND_GEMINI_VISUAL_OBS
    images = list(keyframe_images or [])[:8]
    label = "plus_unified" if include_visual_observation_bundle else "plus"
    try:
        text, provider, key_label = await _call_vision_ai(
            prompt, images, 9216, 0.2, label, timeout_s=PLUS_AI_TIMEOUT_S,
        )
        out = _normalize_plus_result(extract_json_from_response(text))
        if not phase_images_reliable:
            out = _force_unknown_phase_evals_for_unreliable(out)
        out["ai_provider"] = provider
        if key_label:
            out["ai_key"] = key_label
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
        text, provider, key_label = await _call_vision_ai(
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
            "ai_key": key_label,
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
        text, provider, key_label = await _call_vision_ai(
            IMAGE_ONLY_PROMPT, images, 4096, 0.3, "images_only", timeout_s=IMAGE_ONLY_TIMEOUT_S,
        )
        out = extract_json_from_response(text)
        out["ai_provider"] = provider
        if key_label:
            out["ai_key"] = key_label
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
    (0-based) or None on failure. Uses NVIDIA/video-capable AI only."""
    n = len(frame_images)
    if n < 8:
        return None

    prompt = _PHASE_DETECT_PROMPT.format(n=n)
    images = frame_images[:n]

    try:
        text, provider, key_label = await _call_vision_ai(
            prompt, images, 256, 0.1, "phase_detect", timeout_s=PHASE_DETECT_TIMEOUT_S,
        )
        logger.info(
            "[ai] phase_detect vision provider=%s ai_key=%s",
            provider,
            key_label or "-",
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
      - Lite ``lite_trust_tier=medium`` (A fail, B pass): -14, reason ``lite_trust_medium``
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
    # Lite orchestrator: medium trust tier (A fail, B pass) — product reasons sanitized in pack.
    lt = kwargs.get("lite_trust_tier")
    if lt == "medium":
        penalty += 14
        reasons.append("lite_trust_medium")
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
