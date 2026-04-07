import {
  isProv3StrictMediaPolicyResult,
  prov3DisplayKeyframeRows,
  type Prov3ResultLike,
} from "@/lib/prov3-keyframe-media";

/**
 * Single source for keyframe rows on **分析页**与**历史展开**：与 ``prov3DisplayKeyframeRows`` 及
 * 非 prov3 的 fallback 顺序（keyframes → preview → official）一致，避免 Lite 在历史侧与即时结果不同步。
 */
export function displayKeyframesForResult(r: Prov3ResultLike | null | undefined): unknown[] {
  if (!r) return [];
  if (isProv3StrictMediaPolicyResult(r)) {
    return prov3DisplayKeyframeRows(r);
  }
  const o = r as Record<string, unknown>;
  const kf = o.keyframes;
  const prev = o.preview_keyframes;
  const off = o.official_phase_keyframes;
  if (Array.isArray(kf) && kf.length > 0) return kf;
  if (Array.isArray(prev) && prev.length > 0) return prev;
  if (Array.isArray(off) && off.length > 0) return off;
  return [];
}
