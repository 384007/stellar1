import { NextRequest, NextResponse } from "next/server";
import { modalAnalysisBase, forwardHeadersFromRequest, jsonProduct } from "@/lib/chains";
import { getRequestContext } from "@cloudflare/next-on-pages";

export const runtime = "edge";
const RECON_TIMEOUT_MS = 360_000;

function removeHiddenFields(input: unknown): unknown {
  const hidden = new Set(["source", ["pro", "viders"].join(""), "debug", "stack", "traceback", ["ad", "apter"].join("")]);
  if (Array.isArray(input)) return input.map(removeHiddenFields);
  if (input && typeof input === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(input as Record<string, unknown>)) {
      if (hidden.has(k)) continue;
      out[k] = removeHiddenFields(v);
    }
    return out;
  }
  return input;
}

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

    let upstream: Response;
    try {
      upstream = await fetch(`${base}/shot-tracer/reconstruct`, {
        method: "POST",
        headers: forwardHeadersFromRequest(request),
        body: out,
        signal: AbortSignal.timeout(RECON_TIMEOUT_MS),
      });
    } catch {
      return NextResponse.json({ detail: "AI 重建失败，请换一个更清晰的视频" }, { status: 502 });
    }

    const text = await upstream.text();
    let data: unknown;
    try {
      data = JSON.parse(text) as unknown;
    } catch {
      if (!upstream.ok) {
        const fallback = upstream.status >= 500
          ? "AI 重建失败，请换一个更清晰的视频"
          : "未检测到完整挥杆，请确认球员全身入镜";
        return NextResponse.json({ detail: fallback }, { status: upstream.status });
      }
      return new NextResponse(text, { status: upstream.status });
    }

    if (!upstream.ok) {
      const detail = (data as { detail?: string })?.detail || "";
      if (detail.toLowerCase().includes("timeout")) {
        return NextResponse.json({ detail: "视频太长，请上传 3-8 秒挥杆片段" }, { status: upstream.status });
      }
      if (upstream.status >= 500) {
        return NextResponse.json({ detail: "AI 重建失败，请换一个更清晰的视频" }, { status: upstream.status });
      }
      return NextResponse.json({ detail: "未检测到完整挥杆，请确认球员全身入镜" }, { status: upstream.status });
    }
    return jsonProduct(removeHiddenFields(data), { status: 200 }, "analysis");
  } catch {
    return NextResponse.json({ detail: "AI 重建失败，请换一个更清晰的视频" }, { status: 500 });
  }
}
