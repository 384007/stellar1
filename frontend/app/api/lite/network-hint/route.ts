import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";

/**
 * 地理提示：仅依据 CF ``CF-IPCountry``（不扣配额、不校验 JWT）。
 * Lite 分析已统一走同源 ``/api/lite/analyze-proxy``；本接口仍供 Pro 预检等判断 ``network_hint`` / ``lite_geo``。
 *
 * ``lite_geo``：``cn`` | ``intl`` | ``unknown``（无国家头）。
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
