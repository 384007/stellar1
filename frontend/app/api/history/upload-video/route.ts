import { NextRequest, NextResponse } from "next/server";
import { jwtVerify } from "jose";
import { getRequestContext } from "@cloudflare/next-on-pages";

export const runtime = "edge";

function getJwtSecret(): Uint8Array {
  let secret = process.env.JWT_SECRET || "";
  if (!secret) {
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      secret = (getRequestContext().env as any).JWT_SECRET || "";
    } catch { /* ignore */ }
  }
  return new TextEncoder().encode(secret);
}

function getR2() {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (getRequestContext().env as any).R2_BUCKET || null;
  } catch {
    return null;
  }
}

async function getUserId(request: NextRequest): Promise<string | null> {
  const auth = request.headers.get("authorization");
  if (!auth?.startsWith("Bearer ")) return null;
  const token = auth.slice(7);
  if (token.startsWith("local-")) return null;
  try {
    const { payload } = await jwtVerify(token, getJwtSecret());
    return (payload.user_id as string) || null;
  } catch {
    return null;
  }
}

/** Pro v3 / large uploads: raw body streamed to R2 (no multipart parse, no full-file buffer in Worker). */
const STREAM_ANALYSIS_HEADER = "x-stellar-upload-analysis-id";

function safeClientFilename(raw: string, fallbackBase: string): { safeName: string; ext: string } {
  const stripped = (raw || "").replace(/^.*[/\\]/, "").trim().slice(0, 220) || fallbackBase;
  const parts = stripped.split(".");
  const ext = (parts.length > 1 ? parts.pop() : null)?.toLowerCase() || "mp4";
  const base = parts.join(".") || fallbackBase.replace(/\.[^.]+$/, "");
  const safe = `${base.replace(/[^\w.-]+/g, "_").slice(0, 180) || "video"}.${ext}`;
  return { safeName: safe, ext };
}

export async function POST(request: NextRequest) {
  const userId = await getUserId(request);
  if (!userId) return NextResponse.json({ detail: "未登录" }, { status: 401 });

  const r2 = getR2();
  if (!r2) return NextResponse.json({ detail: "存储不可用" }, { status: 503 });

  const streamAnalysisId = (request.headers.get(STREAM_ANALYSIS_HEADER) || "").trim();
  if (streamAnalysisId) {
    if (!/^[a-zA-Z0-9._-]{4,128}$/.test(streamAnalysisId)) {
      return NextResponse.json({ detail: "参数错误: analysis_id" }, { status: 400 });
    }
    const body = request.body;
    if (!body) {
      return NextResponse.json({ detail: "请求体为空" }, { status: 400 });
    }
    const headerFn = (request.headers.get("x-stellar-upload-filename") || "").trim();
    const { safeName, ext } = safeClientFilename(headerFn, `${streamAnalysisId}.${extFromContentType(request)}`);
    const key = `videos/${userId}/${streamAnalysisId}.${ext}`;
    const contentType =
      request.headers.get("content-type")?.split(";")[0]?.trim() || "video/mp4";
    try {
      await r2.put(key, body, {
        httpMetadata: {
          contentType,
          contentDisposition: `inline; filename="${safeName.replace(/"/g, "")}"`,
        },
      });
      return NextResponse.json({ success: true, video_r2_key: key });
    } catch (e) {
      return NextResponse.json(
        { detail: "上传失败，请稍后重试。" },
        { status: 500 },
      );
    }
  }

  try {
    const form = await request.formData();
    const analysisId = String(form.get("analysis_id") || "");
    const file = form.get("file");
    if (!analysisId || !(file instanceof File)) {
      return NextResponse.json({ detail: "参数错误" }, { status: 400 });
    }

    const ext = (file.name.split(".").pop() || "mp4").toLowerCase();
    const key = `videos/${userId}/${analysisId}.${ext}`;
    const bytes = await file.arrayBuffer();
    await r2.put(key, bytes, {
      httpMetadata: {
        contentType: file.type || "video/mp4",
        contentDisposition: `inline; filename="${file.name || `${analysisId}.${ext}`}"`,
      },
    });

    return NextResponse.json({ success: true, video_r2_key: key });
  } catch (e) {
    return NextResponse.json(
      { detail: "上传失败，请稍后重试。" },
      { status: 500 }
    );
  }
}

function extFromContentType(request: NextRequest): string {
  const ct = (request.headers.get("content-type") || "").toLowerCase();
  if (ct.includes("quicktime")) return "mov";
  if (ct.includes("webm")) return "webm";
  return "mp4";
}

