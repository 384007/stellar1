import { NextRequest, NextResponse } from "next/server";
import { modalAnalysisBase, forwardHeadersFromRequest, jsonProduct } from "@/lib/chains";
import { getRequestContext } from "@cloudflare/next-on-pages";

export const runtime = "nodejs";

function getCfEnv(key: string): string {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return ((getRequestContext().env as any)[key] as string) || "";
  } catch {
    return process.env[key] || "";
  }
}

export async function POST(request: NextRequest) {
  try {
    const base = modalAnalysisBase(getCfEnv, request).replace(/\/+$/, "");
    if (!base) {
      return NextResponse.json({ detail: "分析上游未配置" }, { status: 503 });
    }

    const incoming = await request.formData();
    const out = new FormData();
    const fields = ["file", "front_view", "side_view"];
    for (const key of fields) {
      const v = incoming.get(key);
      if (v && typeof v !== "string") {
        const file = v as File;
        if (file.size > 0) out.append(key, file, file.name || `${key}.mp4`);
      }
    }

    const calibration = incoming.get("calibration_json");
    const mode = incoming.get("mode");
    if (calibration && typeof calibration === "string") out.append("calibration_json", calibration);
    if (mode && typeof mode === "string") out.append("mode", mode);

    if (!out.has("file")) {
      return NextResponse.json({ detail: "请上传主视频" }, { status: 400 });
    }

    const upstream = await fetch(`${base}/shot-tracer/reconstruct`, {
      method: "POST",
      headers: forwardHeadersFromRequest(request),
      body: out,
      signal: AbortSignal.timeout(300_000),
    });

    const text = await upstream.text();
    let data: unknown;
    try {
      data = JSON.parse(text) as unknown;
    } catch {
      if (!upstream.ok) {
        return NextResponse.json({ detail: "重建服务异常" }, { status: upstream.status });
      }
      return new NextResponse(text, { status: upstream.status });
    }

    if (!upstream.ok) {
      return jsonProduct(data, { status: upstream.status }, "analysis");
    }
    return jsonProduct(data, { status: 200 }, "analysis");
  } catch {
    return NextResponse.json({ detail: "重建失败，请稍后重试" }, { status: 500 });
  }
}
