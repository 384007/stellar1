/**
 * POST /api/prov3/analyze/start — forwards a small JSON body to Modal ``POST /pro-v3/analyze/start``.
 * Returns ``job_id`` quickly so the browser never holds a long cross-origin analyze connection.
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

  let body: {
    source_r2_key?: string;
    screen_mode?: boolean;
    rough_impact_time_s?: number;
  };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ detail: "Invalid JSON body" }, { status: 400 });
  }

  const source_r2_key = String(body.source_r2_key || "").trim();
  if (!source_r2_key) {
    return NextResponse.json({ detail: "source_r2_key required" }, { status: 400 });
  }

  const clientCnHint =
    (request.headers.get("x-stellar-network-hint") || "").trim().toLowerCase() === "cn";
  const cnHint = auth.cnHint || clientCnHint;

  const modalUrls = buildProv3ModalUrlList(getCfEnv, cnHint);
  const base = normalizeProHttpApiBase(modalUrls[0] || "");
  if (!base) {
    return NextResponse.json({ detail: "分析服务暂时不可用，请稍后重试。" }, { status: 503 });
  }

  const url = `${base}/pro-v3/analyze/start`;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Authorization: request.headers.get("authorization") || "",
  };
  if (cnHint) headers["X-Stellar-Network-Hint"] = "cn";
  const cfCountry = request.headers.get("cf-ipcountry") || request.headers.get("CF-IPCountry");
  if (cfCountry?.trim()) {
    headers["CF-IPCountry"] = cfCountry.trim();
  }

  const fr = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify({
      source_r2_key,
      screen_mode: !!body.screen_mode,
      rough_impact_time_s:
        typeof body.rough_impact_time_s === "number" ? body.rough_impact_time_s : undefined,
    }),
  });

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
