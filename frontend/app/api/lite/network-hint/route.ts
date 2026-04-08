import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";

/**
 * Lite 分析页选路：仅依据 CF ``CF-IPCountry``，无副用（不扣配额、不校验 JWT）。
 * 大陆 CN → 客户端走同源 ``/api/lite/analyze-proxy``（仅转发至独立 ``/analyze/lite``）；非 CN → 浏览器直连 Modal ``/analyze/lite``。
 *
 * ``lite_geo`` 三态：``cn`` | ``intl``（CF 已给出非 CN 国家码）| ``unknown``（无国家头）。
 * 客户端仅在 ``unknown`` 或请求失败时用 ``clientLikelyMainlandChinaUser()`` 回退，避免境外手机语言设为简体中文时被误走长连接代理（易约 1–2 分钟断开）。
 */
export async function GET(request: NextRequest) {
  const raw = (request.headers.get("cf-ipcountry") || request.headers.get("CF-IPCountry") || "").trim();
  const c = raw.toUpperCase();
  if (c === "CN") {
    return NextResponse.json({ network_hint: "cn" as const, lite_geo: "cn" as const });
  }
  if (raw === "") {
    return NextResponse.json({ lite_geo: "unknown" as const });
  }
  return NextResponse.json({ lite_geo: "intl" as const });
}
