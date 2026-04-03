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

export async function POST(request: NextRequest) {
  const userId = await getUserId(request);
  if (!userId) return NextResponse.json({ detail: "未登录" }, { status: 401 });

  const r2 = getR2();
  if (!r2) return NextResponse.json({ detail: "存储不可用" }, { status: 503 });

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
      { detail: `上传失败: ${e instanceof Error ? e.message : "未知错误"}` },
      { status: 500 }
    );
  }
}

