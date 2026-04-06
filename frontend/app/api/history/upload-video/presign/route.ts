/**
 * POST /api/history/upload-video/presign — **CN-only** SigV4 presigned PUT URL.
 *
 * Without ``X-Stellar-Network-Hint: cn`` → ``{ mode: "proxy" }`` (no signing). Matches Pro v3 CN hint.
 *
 * Browser PUTs the video **directly to R2** (same bucket as Worker binding), so upload duration
 * does not consume Cloudflare Pages Function wall-clock (~100s). Modal analysis remains async after
 * ``/api/prov3/analyze/start`` and is unrelated to that limit.
 *
 * Pages secrets (same values as Modal/backend R2 S3 API): ``R2_ACCOUNT_ID`` or ``CLOUDFLARE_ACCOUNT_ID``,
 * ``R2_BUCKET`` (bucket name, e.g. stellar-golf-media), ``R2_ACCESS_KEY`` + ``R2_SECRET_KEY``
 * (or ``R2_ACCESS_KEY_ID`` / ``R2_SECRET_ACCESS_KEY``). If any missing → ``{ mode: "proxy" }`` and
 * client falls back to ``/api/history/upload-video`` streaming proxy.
 *
 * R2 bucket CORS must allow ``PUT`` from your Pages origin (and ``AllowedHeaders: *``), or browser
 * direct upload fails and the client retries the proxy path.
 */

import { NextRequest, NextResponse } from "next/server";
import { jwtVerify } from "jose";
import { getRequestContext } from "@cloudflare/next-on-pages";
import { AwsClient } from "aws4fetch";

export const runtime = "edge";

const PRESIGN_EXPIRES_SEC = 900;

function getCfEnv(key: string): string {
  try {
    return ((getRequestContext().env as Record<string, string>)[key] as string) || "";
  } catch {
    return process.env[key] || "";
  }
}

function getJwtSecret(): Uint8Array {
  let secret = process.env.JWT_SECRET || "";
  if (!secret) {
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      secret = (getRequestContext().env as any).JWT_SECRET || "";
    } catch {
      /* ignore */
    }
  }
  return new TextEncoder().encode(secret);
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

function r2PresignEnv():
  | { accountId: string; bucket: string; accessKeyId: string; secretAccessKey: string }
  | null {
  const accountId = (getCfEnv("R2_ACCOUNT_ID") || getCfEnv("CLOUDFLARE_ACCOUNT_ID")).trim();
  const bucket = (getCfEnv("R2_BUCKET_NAME") || getCfEnv("R2_BUCKET") || "stellar-golf-media").trim();
  const accessKeyId = (getCfEnv("R2_ACCESS_KEY_ID") || getCfEnv("R2_ACCESS_KEY")).trim();
  const secretAccessKey = (getCfEnv("R2_SECRET_ACCESS_KEY") || getCfEnv("R2_SECRET_KEY")).trim();
  if (!accountId || !bucket || !accessKeyId || !secretAccessKey) return null;
  return { accountId, bucket, accessKeyId, secretAccessKey };
}

function extensionForUpload(filename: string, contentType: string): string {
  const low = filename.toLowerCase();
  if (low.endsWith(".mov")) return "mov";
  if (low.endsWith(".webm")) return "webm";
  if (low.endsWith(".mp4")) return "mp4";
  const ct = contentType.toLowerCase();
  if (ct.includes("quicktime")) return "mov";
  if (ct.includes("webm")) return "webm";
  return "mp4";
}

export async function POST(request: NextRequest) {
  const userId = await getUserId(request);
  if (!userId) return NextResponse.json({ detail: "未登录" }, { status: 401 });

  const cnHint =
    (request.headers.get("x-stellar-network-hint") || "").trim().toLowerCase() === "cn";
  if (!cnHint) {
    return NextResponse.json({ mode: "proxy" as const });
  }

  const cfg = r2PresignEnv();
  if (!cfg) {
    return NextResponse.json({ mode: "proxy" as const });
  }

  let body: {
    analysis_id?: string;
    filename?: string;
    content_type?: string;
    byte_length?: number;
  };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ detail: "Invalid JSON" }, { status: 400 });
  }

  const analysisId = String(body.analysis_id || "").trim();
  if (!/^[a-zA-Z0-9._-]{4,128}$/.test(analysisId)) {
    return NextResponse.json({ detail: "参数错误: analysis_id" }, { status: 400 });
  }

  const filename = String(body.filename || "video.mp4").trim() || "video.mp4";
  const contentType = String(body.content_type || "video/mp4").trim() || "video/mp4";
  const ext = extensionForUpload(filename, contentType);
  const key = `videos/${userId}/${analysisId}.${ext}`;
  const encodedObjectKey = key.split("/").map((seg) => encodeURIComponent(seg)).join("/");

  const baseUrl = `https://${cfg.accountId}.r2.cloudflarestorage.com/${cfg.bucket}/${encodedObjectKey}`;
  const urlWithExpiry = `${baseUrl}?X-Amz-Expires=${PRESIGN_EXPIRES_SEC}`;

  try {
    const aws = new AwsClient({
      accessKeyId: cfg.accessKeyId,
      secretAccessKey: cfg.secretAccessKey,
      service: "s3",
      region: "auto",
    });
    const signed = await aws.sign(new Request(urlWithExpiry, { method: "PUT" }), {
      aws: { signQuery: true },
    });
    return NextResponse.json({
      mode: "direct" as const,
      upload_url: signed.url,
      video_r2_key: key,
      content_type: contentType,
    });
  } catch (e) {
    console.error("[upload-video/presign] sign failed:", e);
    return NextResponse.json({ mode: "proxy" as const });
  }
}
