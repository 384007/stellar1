/**
 * POST /api/upload-video — Streaming upload proxy to Gemini Files API.
 *
 * Streams the raw request body directly to Google without buffering,
 * so 100 MB videos don't OOM the Worker (128 MB memory limit).
 *
 * GET /api/upload-video?name=files/xxx — Poll Gemini file processing status.
 */

import { NextRequest, NextResponse } from "next/server";
import { getRequestContext } from "@cloudflare/next-on-pages";
import { jwtVerify } from "jose";
import { getGeminiHosts, getGeminiKeys, rewriteGoogleUrl } from "@/lib/gemini-proxy";

export const runtime = "edge";

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

// POST: Stream video bytes to Gemini resumable upload
export async function POST(request: NextRequest) {
  try {
    const authErr = await requireAuth(request);
    if (authErr) return authErr;

    const keys = getGeminiKeys(getCfEnv);
    if (keys.length === 0) {
      return NextResponse.json({ detail: "AI服务未配置 (GEMINI_API_KEY)" }, { status: 503 });
    }

    const mimeType =
      request.headers.get("x-upload-content-type") ||
      request.headers.get("content-type") ||
      "video/mp4";
    const filename =
      request.headers.get("x-upload-filename") || "video.mp4";
    const byteLength =
      request.headers.get("x-upload-byte-length") ||
      request.headers.get("content-length") ||
      "0";

    const country = (request.headers.get("cf-ipcountry") || "").toUpperCase();
    const isCN = country === "CN";
    const hosts = getGeminiHosts(getCfEnv, isCN);
    let lastErr = "";
    for (const host of hosts) {
     for (let keyIndex = 0; keyIndex < keys.length; keyIndex++) {
      const apiKey = keys[keyIndex]!;
      try {
        const initRes = await fetch(
          `${host}/upload/v1beta/files?key=${apiKey}`,
          {
            method: "POST",
            headers: {
              "X-Goog-Upload-Protocol": "resumable",
              "X-Goog-Upload-Command": "start",
              "X-Goog-Upload-Header-Content-Length": byteLength,
              "X-Goog-Upload-Header-Content-Type": mimeType,
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ file: { displayName: filename } }),
            signal: AbortSignal.timeout(20_000),
          },
        );

        if (!initRes.ok) {
          lastErr = `上传初始化失败 [${initRes.status}]`;
          if (initRes.status === 429) { console.log(`[upload-video] key quota on ${host}`); continue; }
          if (initRes.status < 500) {
            const err = await initRes.text().catch(() => "");
            return NextResponse.json(
              { detail: `${lastErr}: ${err.substring(0, 200)}` },
              { status: 502 },
            );
          }
          break; // 5xx → next host
        }

        const rawUploadUrl = initRes.headers.get("x-goog-upload-url");
        const uploadUrl = rawUploadUrl ? rewriteGoogleUrl(rawUploadUrl, host) : null;
        if (!uploadUrl) { lastErr = "未获取上传URL"; continue; }

        const uploadRes = await fetch(uploadUrl, {
          method: "POST",
          headers: {
            "X-Goog-Upload-Command": "upload, finalize",
            "X-Goog-Upload-Offset": "0",
            "Content-Type": mimeType,
          },
          body: request.body,
          signal: AbortSignal.timeout(60_000),
        });

        if (!uploadRes.ok) {
          const err = await uploadRes.text().catch(() => "");
          return NextResponse.json(
            { detail: `视频上传失败 [${uploadRes.status}]: ${err.substring(0, 200)}` },
            { status: 502 },
          );
        }

        const data = await uploadRes.json();
        return NextResponse.json({
          file_uri: data.file?.uri || null,
          file_name: data.file?.name || null,
          mime_type: mimeType,
          /** 0-based index into ordered keys (GEMINI_API_KEY, _2, _3, …). Analyze must use this for file_uri. */
          gemini_key_index: keyIndex,
        });
      } catch (e) {
        lastErr = e instanceof Error ? e.message : "网络错误";
        console.log(`[upload-video] ${host} failed: ${lastErr}`);
        break; // network error → next host
      }
     } // end key loop
    } // end host loop
    return NextResponse.json({ detail: `视频上传失败: ${lastErr}` }, { status: 502 });
  } catch (err) {
    return NextResponse.json(
      { detail: err instanceof Error ? err.message : "上传错误" },
      { status: 500 },
    );
  }
}

// GET: Check Gemini file processing status
export async function GET(request: NextRequest) {
  try {
    const authErr = await requireAuth(request);
    if (authErr) return authErr;

    const keys = getGeminiKeys(getCfEnv);
    if (keys.length === 0) {
      return NextResponse.json({ detail: "AI服务未配置" }, { status: 503 });
    }

    const fileName = request.nextUrl.searchParams.get("name");
    if (!fileName) {
      return NextResponse.json({ detail: "缺少 name 参数" }, { status: 400 });
    }

    const keyHintRaw = request.nextUrl.searchParams.get("key_index");
    const keyHint = keyHintRaw !== null ? parseInt(keyHintRaw, 10) : NaN;
    const keysOrdered =
      !Number.isNaN(keyHint) && keyHint >= 0 && keyHint < keys.length
        ? [...keys.slice(keyHint), ...keys.slice(0, keyHint)]
        : keys;

    const countryGet = (request.headers.get("cf-ipcountry") || "").toUpperCase();
    const hostsGet = getGeminiHosts(getCfEnv, countryGet === "CN");
    for (const host of hostsGet) {
      for (const key of keysOrdered) {
        try {
          const res = await fetch(
            `${host}/v1beta/${fileName}?key=${key}`,
            { signal: AbortSignal.timeout(10_000) },
          );
          if (res.ok) {
            const info = await res.json();
            return NextResponse.json({
              state: info.state || "UNKNOWN",
              uri: info.uri || null,
            });
          }
          if (res.status === 429 || res.status === 403 || res.status === 404) continue;
          break;
        } catch { break; }
      }
    }
    return NextResponse.json({ state: "UNKNOWN", uri: null });
  } catch (err) {
    return NextResponse.json(
      { detail: err instanceof Error ? err.message : "状态查询错误" },
      { status: 500 },
    );
  }
}
