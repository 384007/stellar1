/**
 * NVIDIA video AI helpers.
 *
 * The filename is retained for compatibility with existing imports.
 */

export const NVIDIA_DIRECT = "https://integrate.api.nvidia.com/v1";
export const NVIDIA_VIDEO_MODEL_DEFAULT = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning";

export function getNvidiaKeys(getCfEnv: (key: string) => string): string[] {
  const keys: string[] = [];
  const add = (raw: string) => {
    const k = (raw || "").trim();
    if (k && !keys.includes(k)) keys.push(k);
  };
  add(getCfEnv("NVIDIA_API_KEY"));
  add(getCfEnv("NVIDIA_KEY"));
  for (let n = 2; n <= 20; n++) {
    add(getCfEnv(`NVIDIA_API_KEY_${n}`));
    add(getCfEnv(`NVIDIA_KEY_${n}`));
  }
  for (const envName of ["NVIDIA_API_KEYS", "NVIDIA_KEYS"]) {
    const raw = getCfEnv(envName);
    for (const part of raw.split(/[,;\n\r]+/)) add(part);
  }
  return keys;
}

export function getNvidiaApiBase(getCfEnv: (key: string) => string): string {
  return (
    getCfEnv("NVIDIA_API_BASE") ||
    getCfEnv("NVIDIA_BASE_URL") ||
    NVIDIA_DIRECT
  ).replace(/\/+$/, "");
}

function modelLooksVideoCapable(model: string): boolean {
  const m = (model || "").trim().toLowerCase();
  if (!m || m.includes("qwen3.6-27b") || m === "qwen/qwen3.6-35b-a3b") return false;
  return [
    "cosmos",
    "omni",
    "vision",
    "video",
    "-vl",
    "_vl",
    "vl-",
    "vl_",
  ].some((needle) => m.includes(needle));
}

export function getNvidiaVideoModel(getCfEnv: (key: string) => string): string {
  const explicit = getCfEnv("NVIDIA_VIDEO_MODEL") || getCfEnv("STELLAR_NVIDIA_VIDEO_MODEL");
  if (explicit) return explicit;
  const inherited = getCfEnv("NVIDIA_MODEL");
  if (modelLooksVideoCapable(inherited)) return inherited;
  return NVIDIA_VIDEO_MODEL_DEFAULT;
}
