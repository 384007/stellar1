/**
 * POST /api/prov3/analyze/cancel — proxies to Modal cooperative cancel (same-origin for the browser).
 */

import { NextRequest, NextResponse } from "next/server";
import { getRequestContext } from "@cloudflare/next-on-pages";
import { requireProUserForProv3Edge } from "@/lib/prov3-edge-route-auth";
import { buildProv3ModalUrlList, normalizeProHttpApiBase } from "@/lib/prov3-endpoints";
import { sanitizeProductJson } from "@/lib/chains/sanitize";

export const runtime = "edge";

function getCfEnv(key: string): string {
  try {
    return ((getRequestContext().env as Record<string, string>)[key] as string) || "";
  } catch {
    return process.env[key] || "";
  }
}

export async function POST(request: NextRequest) {
  const auth = await requireProUserForProv3Edge(request);
  if (!auth.ok) return auth.response;

  const modalUrls = buildProv3ModalUrlList(getCfEnv, auth.cnHint);
  const base = normalizeProHttpApiBase(modalUrls[0] || "");
  if (!base) {
    return NextResponse.json({ detail: "分析服务暂时不可用，请稍后重试。" }, { status: 503 });
  }

  const url = `${base}/pro-v3/analyze/cancel`;
  const headers: Record<string, string> = {
    Authorization: request.headers.get("authorization") || "",
  };
  if (auth.cnHint) headers["X-Stellar-Network-Hint"] = "cn";

  const fr = await fetch(url, { method: "POST", headers });
  const text = await fr.text();
  const ct = fr.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    try {
      const data = JSON.parse(text) as unknown;
      return NextResponse.json(sanitizeProductJson(data, "analysis"), {
        status: fr.status,
        headers: { "content-type": "application/json; charset=utf-8" },
      });
    } catch {
      if (!fr.ok) {
        return NextResponse.json({ detail: "服务暂时不可用，请稍后重试。" }, { status: fr.status });
      }
      return new NextResponse(text, {
        status: fr.status,
        headers: { "content-type": ct },
      });
    }
  }
  if (!fr.ok) {
    return NextResponse.json({ detail: "服务暂时不可用，请稍后重试。" }, { status: fr.status });
  }
  return new NextResponse(text, {
    status: fr.status,
    headers: { "content-type": ct || "application/octet-stream" },
  });
}
