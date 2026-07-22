/**
 * Legacy Gemini Files upload route.
 *
 * Disabled: analysis now sends the original file to Modal, where NVIDIA video AI keys are used.
 */

import { NextRequest, NextResponse } from "next/server";
import { getRequestContext } from "@cloudflare/next-on-pages";
import { jwtVerify } from "jose";

export const runtime = "edge";

async function requireAuth(request: NextRequest): Promise<NextResponse | null> {
  const authHeader = request.headers.get("authorization") || "";
  const token = authHeader.startsWith("Bearer ") ? authHeader.slice(7) : authHeader;
  if (!token) return NextResponse.json({ detail: "未登录" }, { status: 401 });
  if (token.startsWith("guest-")) return NextResponse.json({ detail: "游客不支持" }, { status: 403 });
  if (token.startsWith("local-")) return null;
  if (!token.includes(".")) return NextResponse.json({ detail: "登录无效" }, { status: 401 });

  let jwtSecret = "";
  try {
    jwtSecret = ((getRequestContext().env as Record<string, string>).JWT_SECRET) || "";
  } catch { /* not in CF context */ }
  if (!jwtSecret) jwtSecret = process.env.JWT_SECRET || "";
  if (!jwtSecret) return null;

  try {
    await jwtVerify(token, new TextEncoder().encode(jwtSecret));
    return null;
  } catch {
    return NextResponse.json({ detail: "登录已过期" }, { status: 401 });
  }
}

async function disabled(request: NextRequest, detail: string) {
  const authErr = await requireAuth(request);
  if (authErr) return authErr;
  return NextResponse.json({ detail }, { status: 410 });
}

export async function POST(request: NextRequest) {
  return disabled(request, "旧 Gemini Files 上传通道已停用，请直接上传原始文件进行 NVIDIA 视频分析。");
}

export async function GET(request: NextRequest) {
  return disabled(request, "旧 Gemini Files 状态查询通道已停用。");
}
