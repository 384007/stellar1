/**
 * POST /api/analyze — Classic Lite vision: multipart → Modal ``POST /analyze/vision-classic``.
 * Opaque ``upload_token`` from ``/api/upload-video`` is expanded server-side (never forwarded as raw file_uri).
 */

import { NextRequest, NextResponse } from "next/server";
import { getRequestContext } from "@cloudflare/next-on-pages";
import { jwtVerify } from "jose";
import { forwardHeadersFromRequest, jsonProduct, modalAnalysisBase } from "@/lib/chains";
import { getEdgeJwtSecret } from "@/lib/chains/jwt-secret";
import { unsealUploadSession } from "@/lib/chains/upload-session";

export const runtime = "edge";

const VISION_CLASSIC_TIMEOUT_MS = 180_000;

function getCfEnv(key: string): string {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return ((getRequestContext().env as any)[key] as string) || "";
  } catch {
    return process.env[key] || "";
  }
}

async function requireAuth(request: NextRequest): Promise<NextResponse | null> {
  const authHeader = request.headers.get("authorization") || "";
  const token = authHeader.startsWith("Bearer ") ? authHeader.slice(7) : authHeader;

  if (!token) {
    return NextResponse.json({ detail: "请先登录后再使用分析功能" }, { status: 401 });
  }
  if (token.startsWith("guest-")) {
    return NextResponse.json(
      { detail: "游客模式已关闭，请注册或登录后使用分析功能" },
      { status: 403 },
    );
  }
  if (token.startsWith("local-")) return null;
  if (!token.includes(".")) {
    return NextResponse.json({ detail: "登录状态无效，请重新登录" }, { status: 401 });
  }

  let jwtSecret = "";
  try {
    jwtSecret = (getRequestContext().env as Record<string, string>).JWT_SECRET || "";
  } catch {
    /* not in CF context */
  }
  if (!jwtSecret) jwtSecret = process.env.JWT_SECRET || "";

  if (!jwtSecret) {
    console.error("[analyze] JWT_SECRET not configured, skipping JWT verification");
    return null;
  }

  try {
    const secret = new TextEncoder().encode(jwtSecret);
    const { payload } = await jwtVerify(token, secret);
    if (payload.is_guest) {
      return NextResponse.json(
        { detail: "游客模式已关闭，请注册或登录后使用分析功能" },
        { status: 403 },
      );
    }
    return null;
  } catch {
    return NextResponse.json({ detail: "登录已过期，请重新登录" }, { status: 401 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const authErr = await requireAuth(request);
    if (authErr) return authErr;

    const base = modalAnalysisBase(getCfEnv, request).replace(/\/+$/, "");
    if (!base) {
      return NextResponse.json(
        { detail: "分析上游未配置 (MODAL_BACKEND_URL / LITE_BACKEND_URL)" },
        { status: 503 },
      );
    }

    const formData = await request.formData();
    const out = new FormData();

    const uploadToken = formData.get("upload_token");
    if (uploadToken && typeof uploadToken === "string") {
      const secret = getEdgeJwtSecret();
      if (!secret) {
        return NextResponse.json({ detail: "上传会话不可用" }, { status: 503 });
      }
      const sess = await unsealUploadSession(uploadToken, secret);
      if (!sess) {
        return NextResponse.json({ detail: "上传会话无效或已过期" }, { status: 400 });
      }
      out.append("file_uri", sess.file_uri);
      out.append("mime_type", sess.mime_type);
      out.append("gemini_key_index", String(sess.gemini_key_index));
    } else {
      const fu = formData.get("file_uri");
      const mt = formData.get("mime_type");
      const ki = formData.get("gemini_key_index");
      if (fu && typeof fu === "string") out.append("file_uri", fu);
      if (mt && typeof mt === "string") out.append("mime_type", mt);
      if (ki !== null && ki !== "") out.append("gemini_key_index", String(ki));
    }

    const file = formData.get("file");
    if (file && typeof file !== "string" && "arrayBuffer" in file && (file as File).size > 0) {
      const f = file as File;
      out.append("file", f, f.name || "video.mp4");
    }

    if (!out.has("file") && !out.has("file_uri")) {
      return NextResponse.json({ detail: "请上传文件" }, { status: 400 });
    }

    const url = `${base}/analyze/vision-classic`;
    let upstream: Response;
    try {
      upstream = await fetch(url, {
        method: "POST",
        headers: forwardHeadersFromRequest(request),
        body: out,
        signal: AbortSignal.timeout(VISION_CLASSIC_TIMEOUT_MS),
      });
    } catch {
      return NextResponse.json({ detail: "网络异常，请稍后重试。" }, { status: 502 });
    }

    const text = await upstream.text();
    let data: unknown;
    try {
      data = JSON.parse(text) as unknown;
    } catch {
      if (!upstream.ok) {
        return NextResponse.json({ detail: "分析服务返回异常，请稍后重试。" }, { status: upstream.status });
      }
      return new NextResponse(text, {
        status: upstream.status,
        headers: { "content-type": upstream.headers.get("content-type") || "text/plain; charset=utf-8" },
      });
    }

    if (!upstream.ok) {
      return jsonProduct(data, { status: upstream.status }, "analysis");
    }

    return jsonProduct(data, { status: 200 }, "analysis");
  } catch {
    return NextResponse.json({ detail: "分析异常，请稍后重试。" }, { status: 500 });
  }
}
