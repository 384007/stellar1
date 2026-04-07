import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";

/**
 * Lite 分析页选路：仅依据 CF ``CF-IPCountry``，无副用（不扣配额、不校验 JWT）。
 * 大陆 CN → 客户端走同源 ``/api/analyze``（多代理 Gemini + 多密钥 + Qwen）；非 CN → Modal ``/analyze/lite`` 直连。
 */
export async function GET(request: NextRequest) {
  const c = (request.headers.get("cf-ipcountry") || request.headers.get("CF-IPCountry") || "")
    .trim()
    .toUpperCase();
  return NextResponse.json(c === "CN" ? { network_hint: "cn" as const } : {});
}
