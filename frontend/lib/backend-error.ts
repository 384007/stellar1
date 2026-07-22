/** Format gateway JSON bodies for user-visible analyze errors. */

export type AnalyzeProduct = "plus" | "pro";
export type AnalyzeLang = "zh" | "en";

function hintZh(code: string | undefined): string | null {
  if (!code) return null;
  if (code === "KEYFRAME_QUALITY_STRIP_INVALID") {
    return "关键帧未通过质量校验，请换一段更清晰、完整的挥杆视频重试。";
  }
  return null;
}

function hintEn(code: string | undefined): string | null {
  if (!code) return null;
  if (code === "KEYFRAME_QUALITY_STRIP_INVALID") {
    return "Keyframes did not pass quality checks. Try a clearer, full swing clip.";
  }
  return null;
}

/** Human-readable message from /analyze/* error JSON (handles nested `detail` dict). */
export function formatHttpAnalyzeError(
  status: number,
  body: unknown,
  lang: AnalyzeLang,
  product: AnalyzeProduct
): string {
  const head =
    lang === "zh"
      ? product === "plus"
        ? `Plus 分析失败 [${status}]`
        : `Pro分析失败 [${status}]`
      : product === "plus"
        ? `Plus analysis failed [${status}]`
        : `Pro analysis failed [${status}]`;

  const tail = (msg: string) => (msg ? `${head}: ${msg}` : head);

  if (body == null || typeof body !== "object") {
    return tail(lang === "zh" ? "服务器返回异常" : "Unexpected response");
  }

  const o = body as Record<string, unknown>;
  let d: unknown = o.detail;
  if (d === undefined && (o.reasons != null || o.hint_code != null || o.error_code != null)) {
    d = o;
  }

  if (typeof d === "string" && d.trim()) {
    return tail(d.trim());
  }

  if (d && typeof d === "object") {
    const inner = d as Record<string, unknown>;
    const hintCode = typeof inner.hint_code === "string" ? inner.hint_code : undefined;
    const hintLine =
      lang === "zh" ? hintZh(hintCode) : hintEn(hintCode);
    const reasons = Array.isArray(inner.reasons)
      ? inner.reasons.filter((x): x is string => typeof x === "string")
      : [];
    const parts: string[] = [];
    if (hintLine) parts.push(hintLine);
    else if (hintCode && lang === "en") parts.push(hintCode);
    if (reasons.length) parts.push(reasons.join("; "));
    const msg = typeof inner.message === "string" ? inner.message.trim() : "";
    if (msg) parts.push(msg);
    if (parts.length) return tail(parts.join(" · "));
    return tail(lang === "zh" ? "请稍后重试" : "Please try again");
  }

  return tail(lang === "zh" ? "未知错误" : "Unknown error");
}
