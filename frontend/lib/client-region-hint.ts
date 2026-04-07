/**
 * Browser-side hint for mainland-China–friendly client behavior (Pro job poll cap, Lite ``/api/lite/analyze-proxy``, etc.).
 * Does not replace CF ``CF-IPCountry`` on the server — augments when precheck/geo is unknown (e.g. VPN).
 */
export function clientLikelyMainlandChinaUser(): boolean {
  if (typeof window === "undefined") return false;
  const lang = (navigator.language || "").toLowerCase();
  if (lang === "zh-cn" || lang.startsWith("zh-cn-")) return true;
  try {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || "";
    return /shanghai|chongqing|urumqi|harbin|asia\/shanghai|asia\/chongqing|asia\/urumqi/i.test(tz);
  } catch {
    return false;
  }
}
