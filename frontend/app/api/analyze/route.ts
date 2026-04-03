import { NextRequest, NextResponse } from "next/server";
import { getRequestContext } from "@cloudflare/next-on-pages";
import { jwtVerify } from "jose";
import { getGeminiHosts, getGeminiKeys, rewriteGoogleUrl } from "@/lib/gemini-proxy";

export const runtime = "edge";

const QWEN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions";
const QWEN_MODEL = "qwen-vl-max-latest";

function getCfEnv(key: string): string {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return ((getRequestContext().env as any)[key] as string) || "";
  } catch {
    return process.env[key] || "";
  }
}

/** Preserve selected Gemini 4xx on our response so the host×key loop can try the next key. */
function statusFromGeminiUpstream(upstreamStatus: number): number {
  if (upstreamStatus === 429 || upstreamStatus === 401 || upstreamStatus === 403) {
    return upstreamStatus;
  }
  return 502;
}

function shouldRetryNextGeminiKey(responseStatus: number): boolean {
  return responseStatus === 429 || responseStatus === 401 || responseStatus === 403;
}

const ANALYSIS_PROMPT = `You are an expert PGA-level golf coach and biomechanics analyst.

STEP 1 — DETECTION (CRITICAL):
Describe what you actually see. Determine whether the image/video shows a REAL golf swing.
Set "is_golf_swing" to true ONLY if you can clearly see a person swinging a golf club.
If the content shows anything else — a person standing, a landscape, a non-golf activity, a random object, or an unclear image — you MUST set "is_golf_swing" to false.

STEP 2 — ANALYSIS (only if is_golf_swing is true):
Evaluate these 5 dimensions (0-100):
1. Grip 2. Stance 3. Backswing 4. Downswing 5. Follow-through
Be brutally honest. Amateur golfers typically score 40-75. Do NOT inflate scores.

If is_golf_swing is false: set all scores to 0, total_score to 0, issues/suggestions to empty arrays, and describe what you see in summary.

Respond with ONLY this JSON (no markdown, no backticks):
{
  "what_i_see": "<describe what is visible>",
  "what_i_see_zh": "<中文描述>",
  "is_golf_swing": true or false,
  "scores": {"grip": <0-100>, "stance": <0-100>, "backswing": <0-100>, "downswing": <0-100>, "follow_through": <0-100>},
  "total_score": <weighted average>,
  "issues": ["<issue 1>","<issue 2>","<issue 3>"],
  "issues_zh": ["<问题1>","<问题2>","<问题3>"],
  "suggestions": ["<fix 1>","<fix 2>","<fix 3>"],
  "suggestions_zh": ["<建议1>","<建议2>","<建议3>"],
  "summary": "<English analysis 150-200 words>",
  "summary_zh": "<中文分析150-200字>",
  "prediction": {"predicted_distance":<yards>,"lateral_offset":<yards>,"shot_shape":"<shape>","shot_shape_zh":"<中文>","club_head_speed":<mph>,"ball_speed":<mph>,"launch_angle":<deg>,"spin_rate":<rpm>,"smash_factor":<ratio>}
}`;

function uint8ToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 8192) {
    const end = Math.min(i + 8192, bytes.length);
    for (let j = i; j < end; j++) binary += String.fromCharCode(bytes[j]);
  }
  return btoa(binary);
}

function stripThinkingBlocks(text: string): string {
  return text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
}

function buildAnalysisResponse(
  parsed: Record<string, unknown>,
  model: string,
  aiProvider: "gemini" | "qwen" = "gemini",
  keyIndex = 0,
  host = "",
) {
  const pred = (parsed.prediction as Record<string, unknown>) || {};
  const hostHostname = host
    ? (() => {
        try {
          return new URL(host).hostname;
        } catch {
          return host;
        }
      })()
    : "";
  return NextResponse.json({
    analysis_id: `stellar-${Date.now()}`,
    type: "lite",
    ai_provider: aiProvider,
    model_used: model,
    /** Gemini only: 1 = GEMINI_API_KEY, 2 = GEMINI_API_KEY_2. null when Qwen fallback. */
    key_used: aiProvider === "qwen" ? null : keyIndex + 1,
    host_used: hostHostname || (aiProvider === "qwen" ? "dashscope.aliyuncs.com" : ""),
    what_i_see: parsed.what_i_see || "",
    what_i_see_zh: parsed.what_i_see_zh || "",
    is_golf_swing: parsed.is_golf_swing === true,
    scores: parsed.scores || { grip: 0, stance: 0, backswing: 0, downswing: 0, follow_through: 0 },
    total_score: parsed.total_score || 0,
    issues: parsed.issues || [],
    issues_zh: parsed.issues_zh || [],
    suggestions: parsed.suggestions || [],
    suggestions_zh: parsed.suggestions_zh || [],
    summary: parsed.summary || "",
    summary_zh: parsed.summary_zh || "",
    keyframes: Array.isArray(parsed.keyframes) ? parsed.keyframes : [],
    skeleton_data: { frames: [], total_frames: 0 },
    prediction: {
      predicted_distance: (pred.predicted_distance as number) ?? 0,
      lateral_offset: (pred.lateral_offset as number) ?? 0,
      shot_shape: (pred.shot_shape as string) || "N/A",
      shot_shape_zh: (pred.shot_shape_zh as string) || "未知",
      club_head_speed: (pred.club_head_speed as number) ?? 0,
      ball_speed: (pred.ball_speed as number) ?? 0,
      launch_angle: (pred.launch_angle as number) ?? 0,
      spin_rate: (pred.spin_rate as number) ?? 0,
      smash_factor: (pred.smash_factor as number) ?? 0,
      trajectory: [],
    },
  });
}

// ── CN 分析: 通义千问 qwen3-vl-235b-a22b（图片 + 视频均支持）──

async function qwenAnalysis(file: File): Promise<NextResponse> {
  const apiKey = getCfEnv("QWEN_API_KEY");
  if (!apiKey) {
    return NextResponse.json(
      { detail: "通义千问 API 密钥未配置 (QWEN_API_KEY)" },
      { status: 503 }
    );
  }

  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const base64 = uint8ToBase64(bytes);
  const rawType = file.type || "";
  const isVideo =
    rawType.startsWith("video/") || /\.(mp4|mov|webm|avi)$/i.test(file.name || "");
  const mimeType =
    rawType === "video/quicktime" ? "video/mp4"
      : rawType || (isVideo ? "video/mp4" : "image/jpeg");

  // Qwen VL: image_url for images, video_url for videos
  const mediaPart = isVideo
    ? { type: "video_url", video_url: { url: `data:${mimeType};base64,${base64}` } }
    : { type: "image_url", image_url: { url: `data:${mimeType};base64,${base64}` } };

  try {
    const res = await fetch(QWEN_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: QWEN_MODEL,
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: ANALYSIS_PROMPT },
              mediaPart,
            ],
          },
        ],
        temperature: 0.3,
        max_tokens: 4096,
      }),
      signal: AbortSignal.timeout(60_000),
    });

    if (!res.ok) {
      const body = await res.text().catch(() => "");
      return NextResponse.json(
        { detail: `Qwen 分析错误 [${res.status}]: ${body.substring(0, 200)}` },
        { status: 502 }
      );
    }

    const data = await res.json();
    const raw: string = data.choices?.[0]?.message?.content || "";
    const text = stripThinkingBlocks(raw);
    const match = text.match(/\{[\s\S]*\}/);
    if (!match) {
      return NextResponse.json(
        { detail: `Qwen 返回了非JSON响应: ${text.substring(0, 150)}` },
        { status: 502 }
      );
    }

    const parsed = JSON.parse(match[0]);
    return buildAnalysisResponse(parsed, QWEN_MODEL);
  } catch (e) {
    return NextResponse.json(
      { detail: `Qwen 分析失败: ${e instanceof Error ? e.message : "未知错误"}` },
      { status: 502 }
    );
  }
}

// ── 全球分析: Gemini 2.5 Flash ──

async function geminiAnalysis(file: File, host: string, apiKey: string, keyIndex = 0): Promise<NextResponse> {
  const model = getCfEnv("GEMINI_MODEL") || "gemini-2.5-flash-lite";
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const rawType = file.type || "";
  const isVideo =
    rawType.startsWith("video/") || /\.(mp4|mov|webm|avi)$/i.test(file.name || "");
  const mimeType =
    rawType === "video/quicktime" ? "video/mp4"
      : rawType || (isVideo ? "video/mp4" : "image/jpeg");

  try {
    let contentParts: unknown[];

    if (isVideo) {
      // Step 1: Start resumable upload
      const initRes = await fetch(
        `${host}/upload/v1beta/files?key=${apiKey}`,
        {
          method: "POST",
          headers: {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": String(bytes.length),
            "X-Goog-Upload-Header-Content-Type": mimeType,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ file: { displayName: file.name || "swing.mp4" } }),
          signal: AbortSignal.timeout(20_000),
        }
      );
      if (!initRes.ok) {
        const errBody = await initRes.text().catch(() => "");
        return NextResponse.json(
          { detail: `AI上传初始化失败 [${initRes.status}]: ${errBody.substring(0, 200)}` },
          { status: statusFromGeminiUpstream(initRes.status) }
        );
      }

      const rawUploadUri = initRes.headers.get("x-goog-upload-url");
      const uploadUri = rawUploadUri ? rewriteGoogleUrl(rawUploadUri, host) : null;
      if (!uploadUri) {
        return NextResponse.json({ detail: "AI服务未返回上传URI" }, { status: 502 });
      }

      // Step 2: Upload bytes
      const uploadRes = await fetch(uploadUri, {
        method: "POST",
        headers: {
          "X-Goog-Upload-Command": "upload, finalize",
          "X-Goog-Upload-Offset": "0",
          "Content-Type": mimeType,
        },
        body: bytes,
        signal: AbortSignal.timeout(30_000),
      });
      if (!uploadRes.ok) {
        const errBody = await uploadRes.text().catch(() => "");
        return NextResponse.json(
          { detail: `视频上传失败 [${uploadRes.status}]: ${errBody.substring(0, 200)}` },
          { status: statusFromGeminiUpstream(uploadRes.status) }
        );
      }

      const uploadData = await uploadRes.json();
      const fileUri = uploadData.file?.uri;
      const fileName = uploadData.file?.name;
      if (!fileUri) {
        return NextResponse.json({ detail: "文件上传成功但未返回URI" }, { status: 502 });
      }

      // Step 3: Poll until ACTIVE
      if (fileName) {
        for (let i = 0; i < 20; i++) {
          const checkRes = await fetch(
            `${host}/v1beta/${fileName}?key=${apiKey}`
          );
          if (checkRes.ok) {
            const fileInfo = await checkRes.json();
            if (fileInfo.state === "ACTIVE") break;
            if (fileInfo.state === "FAILED") {
              return NextResponse.json(
                { detail: "视频处理失败，请尝试压缩视频或缩短时长" },
                { status: 422 }
              );
            }
          }
          await new Promise(r => setTimeout(r, 3000));
        }
      }

      contentParts = [{ text: ANALYSIS_PROMPT }, { fileData: { mimeType, fileUri } }];
    } else {
      const base64 = uint8ToBase64(bytes);
      contentParts = [{ text: ANALYSIS_PROMPT }, { inlineData: { mimeType, data: base64 } }];
    }

    const res = await fetch(
      `${host}/v1beta/models/${model}:generateContent?key=${apiKey}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [{ parts: contentParts }],
          generationConfig: { temperature: 0.3, maxOutputTokens: 4096 },
        }),
        signal: AbortSignal.timeout(60_000),
      }
    );

    if (!res.ok) {
      const body = await res.text().catch(() => "");
      return NextResponse.json(
        { detail: `AI 分析错误 [${res.status}]: ${body.substring(0, 200)}` },
        { status: statusFromGeminiUpstream(res.status) }
      );
    }

    const data = await res.json();
    const text = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
    const match = text.match(/\{[\s\S]*\}/);
    if (!match) {
      return NextResponse.json(
        { detail: `AI 返回了非JSON响应: ${text.substring(0, 150)}` },
        { status: 502 }
      );
    }

    const parsed = JSON.parse(match[0]);
    return buildAnalysisResponse(parsed, model, "gemini", keyIndex, host);
  } catch (e) {
    return NextResponse.json(
      { detail: `分析失败: ${e instanceof Error ? e.message : "未知错误"}` },
      { status: 502 }
    );
  }
}

// ── Auth guard ──

async function requireAuth(request: NextRequest): Promise<NextResponse | null> {
  const authHeader = request.headers.get("authorization") || "";
  const token = authHeader.startsWith("Bearer ") ? authHeader.slice(7) : authHeader;

  if (!token) {
    return NextResponse.json({ detail: "请先登录后再使用分析功能" }, { status: 401 });
  }
  if (token.startsWith("guest-")) {
    return NextResponse.json(
      { detail: "游客模式已关闭，请注册或登录后使用分析功能" },
      { status: 403 }
    );
  }
  if (token.startsWith("local-")) return null;
  if (!token.includes(".")) {
    return NextResponse.json({ detail: "登录状态无效，请重新登录" }, { status: 401 });
  }

  let jwtSecret = "";
  try { jwtSecret = (getRequestContext().env as Record<string, string>).JWT_SECRET || ""; } catch { /* not in CF context */ }
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
        { status: 403 }
      );
    }
    return null;
  } catch {
    return NextResponse.json({ detail: "登录已过期，请重新登录" }, { status: 401 });
  }
}

// ── URI-based Gemini analysis (video already uploaded via /api/upload-video) ──

async function geminiAnalysisWithUri(fileUri: string, mimeType: string, host: string, apiKey: string, keyIndex = 0): Promise<NextResponse> {
  const model = getCfEnv("GEMINI_MODEL") || "gemini-2.5-flash-lite";

  try {
    const res = await fetch(
      `${host}/v1beta/models/${model}:generateContent?key=${apiKey}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [{ parts: [{ text: ANALYSIS_PROMPT }, { fileData: { mimeType, fileUri } }] }],
          generationConfig: { temperature: 0.3, maxOutputTokens: 4096 },
        }),
        signal: AbortSignal.timeout(60_000),
      }
    );

    if (!res.ok) {
      const body = await res.text().catch(() => "");
      return NextResponse.json(
        { detail: `AI 分析错误 [${res.status}]: ${body.substring(0, 200)}` },
        { status: statusFromGeminiUpstream(res.status) }
      );
    }

    const data = await res.json();
    const text = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
    const match = text.match(/\{[\s\S]*\}/);
    if (!match) {
      return NextResponse.json(
        { detail: `AI 返回了非JSON响应: ${text.substring(0, 150)}` },
        { status: 502 }
      );
    }

    const parsed = JSON.parse(match[0]);
    return buildAnalysisResponse(parsed, model, "gemini", keyIndex, host);
  } catch (e) {
    return NextResponse.json(
      { detail: `分析失败: ${e instanceof Error ? e.message : "未知错误"}` },
      { status: 502 }
    );
  }
}

// ── Main handler ──

export async function POST(request: NextRequest) {
  try {
    const authErr = await requireAuth(request);
    if (authErr) return authErr;

    const formData = await request.formData();
    const file = formData.get("file") as File | null;
    const fileUri = formData.get("file_uri") as string | null;
    const fileMimeType = formData.get("mime_type") as string | null;

    if (!file && !fileUri) {
      return NextResponse.json({ detail: "请上传文件" }, { status: 400 });
    }

    const country = (request.headers.get("cf-ipcountry") || "").toUpperCase();
    const isCN = country === "CN";
    const hosts = getGeminiHosts(getCfEnv, isCN);
    const keys = getGeminiKeys(getCfEnv);
    const keyHintRaw = formData.get("gemini_key_index");
    const keyHintParsed =
      keyHintRaw !== null && String(keyHintRaw) !== ""
        ? parseInt(String(keyHintRaw), 10)
        : NaN;
    console.log(`[analyze] country=${country} isCN=${isCN} hosts=${hosts.map(h => new URL(h).hostname).join(",")} keys=${keys.length}`);

    if (keys.length === 0) {
      // No Gemini keys — go straight to Qwen
      const qwenKey = getCfEnv("QWEN_API_KEY");
      if (qwenKey) return fileUri
        ? NextResponse.json({ detail: "AI 服务密钥未配置" }, { status: 503 })
        : await qwenAnalysis(file!);
      return NextResponse.json({ detail: "AI 服务密钥未配置 (GEMINI_API_KEY)" }, { status: 503 });
    }

    // Pre-uploaded video: host × key loop (prefer key that created the Files API object — avoids 403)
    if (fileUri) {
      const keysOrdered =
        !Number.isNaN(keyHintParsed) && keyHintParsed >= 0 && keyHintParsed < keys.length
          ? [...keys.slice(keyHintParsed), ...keys.slice(0, keyHintParsed)]
          : keys;
      let uriResult: NextResponse | null = null;
      for (const host of hosts) {
        for (let ki = 0; ki < keysOrdered.length; ki++) {
          const apiKey = keysOrdered[ki]!;
          const originalKeyIndex = keys.indexOf(apiKey);
          uriResult = await geminiAnalysisWithUri(
            fileUri,
            fileMimeType || "video/mp4",
            host,
            apiKey,
            originalKeyIndex >= 0 ? originalKeyIndex : ki,
          );
          if (uriResult.status < 400) {
            console.log(
              `[analyze] ✓ Gemini via ${new URL(host).hostname} key${(originalKeyIndex >= 0 ? originalKeyIndex : ki) + 1}`,
            );
            return uriResult;
          }
          if (shouldRetryNextGeminiKey(uriResult.status)) {
            console.log(
              `[analyze] key retry HTTP ${uriResult.status} on ${new URL(host).hostname} → try next key`,
            );
            continue;
          }
          break;
        }
      }
      return uriResult || NextResponse.json({ detail: "AI 服务不可用" }, { status: 503 });
    }

    // Gemini: host × key loop → Qwen final fallback
    let geminiResult: NextResponse | null = null;
    for (const host of hosts) {
      for (let ki = 0; ki < keys.length; ki++) {
        geminiResult = await geminiAnalysis(file!, host, keys[ki], ki);
        if (geminiResult.status < 400) {
          console.log(`[analyze] ✓ Gemini via ${new URL(host).hostname} key${ki + 1}`);
          return geminiResult;
        }
        if (shouldRetryNextGeminiKey(geminiResult.status)) {
          console.log(`[analyze] key${ki + 1} HTTP ${geminiResult.status} on ${new URL(host).hostname} → try next key`);
          continue;
        }
        console.log(`[analyze] Gemini via ${new URL(host).hostname} key${ki + 1} → ${geminiResult.status}`);
        break;
      }
    }

    const qwenKey = getCfEnv("QWEN_API_KEY");
    if (qwenKey) {
      console.log("[analyze] All Gemini hosts/keys failed → Qwen fallback");
      return qwenAnalysis(file!);
    }
    return geminiResult || NextResponse.json({ detail: "AI 服务不可用" }, { status: 503 });
  } catch (err) {
    return NextResponse.json(
      { detail: err instanceof Error ? err.message : "分析错误" },
      { status: 500 }
    );
  }
}

