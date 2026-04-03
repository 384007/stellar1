/**
 * POST /api/lab — Create a Shot Lab analysis job.
 *
 * Accepts FormData with a `file` field (video/image).
 * Optional `job_id` field for client-side idempotency on retry.
 *
 * Flow: auth → feature flag → quota check → create job → AI analysis → store → respond
 * Quota is decremented idempotently using job_id.
 */

import { NextRequest, NextResponse } from "next/server";
import {
  isLabEnabled,
  getCfEnvVal,
  LAB_FREE_DAILY_LIMIT,
  type LabTier,
} from "@/lib/lab-config";
import {
  ensureLabSchema,
  getLabUsageToday,
  incrementLabUsage,
  createLabJob,
  markLabJobRetryProcessing,
  updateLabJobResult,
  getLabJob,
  updateLabJobSummary,
} from "@/lib/lab-db";
import {
  authenticateRequest,
  getDB,
  filterForTier,
  buildQuota,
  labError,
} from "@/lib/lab-auth";
import {
  getGeminiHosts,
  getGeminiKeys,
  isStaleGeminiFileReference,
  redactGeminiFileRefForLog,
  rewriteGoogleUrl,
  shouldRetryNextGeminiKey,
} from "@/lib/gemini-proxy";

export const runtime = "edge";

let _schemaEnsured = false;

const QWEN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions";
const QWEN_MODEL = "qwen-vl-max-latest";

// ── Lab-specific analysis prompt ──

const LAB_PROMPT = `You are an expert PGA-level golf biomechanics analyst performing phone-camera-only video analysis for a product called "Shot Lab".
IMPORTANT: All values are ESTIMATES from video analysis — NOT radar/launch-monitor measurements. Be honest about what you can and cannot determine from video.

Analyze this golf swing and return ONLY the following JSON (no markdown, no backticks):
{
  "is_golf_swing": true or false,
  "what_i_see": "<describe what is visible in the video>",
  "what_i_see_zh": "<中文描述>",
  "metrics": {
    "ball_speed_mph": <number or null if not estimable>,
    "ball_speed_confidence": <0.0-1.0>,
    "launch_angle_deg": <number or null>,
    "launch_angle_confidence": <0.0-1.0>,
    "launch_direction_deg": <number or null, positive=right of target>,
    "launch_direction_confidence": <0.0-1.0>,
    "backswing_time_sec": <number or null>,
    "downswing_time_sec": <number or null>,
    "tempo_ratio": <backswing/downswing ratio or null>,
    "tempo_confidence": <0.0-1.0>,
    "carry_distance_yards": <number or null>,
    "carry_distance_confidence": <0.0-1.0>,
    "contact_quality_score": <0-100 or null>,
    "contact_quality_confidence": <0.0-1.0>
  },
  "issues": [
    {
      "id": "<snake_case_id>",
      "title": "<English title>",
      "title_zh": "<中文标题>",
      "description": "<English description 1-2 sentences>",
      "description_zh": "<中文描述>",
      "severity": "high" or "medium" or "low",
      "drill": "<English drill recommendation>",
      "drill_zh": "<中文训练建议>"
    }
  ],
  "summary": "<English analysis 100-200 words>",
  "summary_zh": "<中文分析总结100-200字>",
  "full_report": "<English detailed structured report 300-500 words covering setup, backswing, transition, downswing, impact, follow-through with specific observations>",
  "full_report_zh": "<中文详细报告300-500字>",
  "drills": [
    {
      "title": "<English drill title>",
      "title_zh": "<中文训练名称>",
      "description": "<English description 2-3 sentences>",
      "description_zh": "<中文描述>"
    }
  ]
}

Rules:
- Identify at least 3 issues if it IS a golf swing, up to 10
- Provide at least 3 drills
- For metrics you cannot estimate from the video, use null (do NOT fabricate numbers)
- Ball speed/carry distance are rough estimates based on visual club speed and contact quality
- Tempo is measurable from frame timing if the video has sufficient frame rate
- All metric sources must be video-based estimation, label them honestly`;

// ── Helpers ──

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

function toNum(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

/** Parse `[503]`-style status from `labGeminiAnalysis*` error messages. */
function httpStatusFromBracketMessage(message: string): number {
  const m = message.match(/\[(\d{3})\]/);
  if (!m) return 0;
  const n = parseInt(m[1]!, 10);
  return Number.isFinite(n) ? n : 0;
}

function applyUnifiedPrediction(
  parsed: Record<string, unknown>,
  unifiedPredictionRaw: string | null
): Record<string, unknown> {
  if (!unifiedPredictionRaw) return parsed;
  let prediction: Record<string, unknown>;
  try {
    prediction = JSON.parse(unifiedPredictionRaw);
  } catch {
    return parsed;
  }
  if (typeof prediction !== "object" || prediction === null) return parsed;
  const predictedDistance = toNum(prediction.predicted_distance, NaN);
  if (!Number.isFinite(predictedDistance)) return parsed;

  const metrics = ((parsed.metrics as Record<string, unknown>) || {});
  const distanceConfidence = toNum(prediction.distance_confidence, metrics.carry_distance_confidence as number || 0);
  const mergedPrediction: Record<string, unknown> = {
    ...prediction,
    club_type: (prediction.club_type as string) || (parsed.club_type as string) || "UNKNOWN",
    club_group: (prediction.club_group as string) || (parsed.club_group as string) || "UNKNOWN",
    hand: (prediction.hand as string) || (parsed.hand as string) || "UNKNOWN",
  };

  return {
    ...parsed,
    prediction: mergedPrediction,
    club_type: mergedPrediction.club_type,
    club_group: mergedPrediction.club_group,
    hand: mergedPrediction.hand,
    metrics: {
      ...metrics,
      ball_speed_mph: toNum(prediction.ball_speed, metrics.ball_speed_mph as number),
      launch_angle_deg: toNum(prediction.launch_angle, metrics.launch_angle_deg as number),
      carry_distance_yards: predictedDistance,
      carry_distance_confidence: distanceConfidence,
    },
  };
}

// ── AI analysis backends (reuse same pattern as /api/analyze) ──

async function labGeminiAnalysis(file: File, host: string, apiKey: string): Promise<Record<string, unknown>> {
  const model = getCfEnvVal("GEMINI_MODEL") || "gemini-2.5-flash-lite";
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const rawType = file.type || "";
  const isVideo = rawType.startsWith("video/") || /\.(mp4|mov|webm|avi)$/i.test(file.name || "");
  const mimeType = rawType === "video/quicktime" ? "video/mp4" : rawType || (isVideo ? "video/mp4" : "image/jpeg");

  let contentParts: unknown[];

  if (isVideo) {
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
      throw new Error(`AI上传初始化失败 [${initRes.status}]: ${errBody.substring(0, 200)}`);
    }

    const rawUploadUri = initRes.headers.get("x-goog-upload-url");
    const uploadUri = rawUploadUri ? rewriteGoogleUrl(rawUploadUri, host) : null;
    if (!uploadUri) throw new Error("AI服务未返回上传URI");

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
      throw new Error(`视频上传失败 [${uploadRes.status}]: ${errBody.substring(0, 200)}`);
    }

    const uploadData = await uploadRes.json();
    const fileUri = uploadData.file?.uri;
    const fileName = uploadData.file?.name;
    if (!fileUri) throw new Error("文件上传成功但未返回URI");

    if (fileName) {
      for (let i = 0; i < 20; i++) {
        const checkRes = await fetch(
          `${host}/v1beta/${fileName}?key=${apiKey}`
        );
        if (checkRes.ok) {
          const fileInfo = await checkRes.json();
          if (fileInfo.state === "ACTIVE") break;
          if (fileInfo.state === "FAILED") throw new Error("视频处理失败，请尝试压缩视频或缩短时长");
        }
        await new Promise(r => setTimeout(r, 3000));
      }
    }

    contentParts = [{ text: LAB_PROMPT }, { fileData: { mimeType, fileUri } }];
  } else {
    const base64 = uint8ToBase64(bytes);
    contentParts = [{ text: LAB_PROMPT }, { inlineData: { mimeType, data: base64 } }];
  }

  const res = await fetch(
    `${host}/v1beta/models/${model}:generateContent?key=${apiKey}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts: contentParts }],
        generationConfig: { temperature: 0.3, maxOutputTokens: 8192 },
      }),
      signal: AbortSignal.timeout(60_000),
    }
  );

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`AI 分析错误 [${res.status}]: ${body.substring(0, 200)}`);
  }

  const data = await res.json();
  const text = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) throw new Error(`AI 返回了非JSON响应: ${text.substring(0, 150)}`);

  return JSON.parse(match[0]);
}

async function labQwenAnalysis(file: File): Promise<Record<string, unknown>> {
  const apiKey = getCfEnvVal("QWEN_API_KEY");
  if (!apiKey) throw new Error("通义千问 API 密钥未配置 (QWEN_API_KEY)");

  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const base64 = uint8ToBase64(bytes);
  const rawType = file.type || "";
  const isVideo = rawType.startsWith("video/") || /\.(mp4|mov|webm|avi)$/i.test(file.name || "");
  const mimeType = rawType === "video/quicktime" ? "video/mp4" : rawType || (isVideo ? "video/mp4" : "image/jpeg");

  const mediaPart = isVideo
    ? { type: "video_url", video_url: { url: `data:${mimeType};base64,${base64}` } }
    : { type: "image_url", image_url: { url: `data:${mimeType};base64,${base64}` } };

  const res = await fetch(QWEN_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: QWEN_MODEL,
      messages: [{ role: "user", content: [{ type: "text", text: LAB_PROMPT }, mediaPart] }],
      temperature: 0.3,
      max_tokens: 8192,
    }),
    signal: AbortSignal.timeout(60_000),
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Qwen 分析错误 [${res.status}]: ${body.substring(0, 200)}`);
  }

  const data = await res.json();
  const raw: string = data.choices?.[0]?.message?.content || "";
  const text = stripThinkingBlocks(raw);
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) throw new Error(`Qwen 返回了非JSON响应: ${text.substring(0, 150)}`);

  return JSON.parse(match[0]);
}

// ── URI-based Gemini analysis (video already uploaded via /api/upload-video) ──

async function labGeminiAnalysisWithUri(fileUri: string, mimeType: string, host: string, apiKey: string): Promise<Record<string, unknown>> {
  const model = getCfEnvVal("GEMINI_MODEL") || "gemini-2.5-flash-lite";

  const res = await fetch(
    `${host}/v1beta/models/${model}:generateContent?key=${apiKey}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts: [{ text: LAB_PROMPT }, { fileData: { mimeType, fileUri } }] }],
        generationConfig: { temperature: 0.3, maxOutputTokens: 8192 },
      }),
      signal: AbortSignal.timeout(60_000),
    }
  );

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`AI 分析错误 [${res.status}]: ${body.substring(0, 200)}`);
  }

  const data = await res.json();
  const text = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) throw new Error(`AI 返回了非JSON响应: ${text.substring(0, 150)}`);

  return JSON.parse(match[0]);
}

// ── Main handler ──

export async function POST(request: NextRequest) {
  try {
    if (!isLabEnabled()) {
      return NextResponse.json(
        { error: "FEATURE_DISABLED", detail: "Shot Lab 功能暂未开放" },
        { status: 404 }
      );
    }

    const authResult = await authenticateRequest(request);
    if (authResult instanceof NextResponse) return authResult;
    const { user_id, is_pro } = authResult;
    const tier: LabTier = is_pro ? "pro" : "free";

    const db = getDB();

    // Parse form data
    const formData = await request.formData();
    const file = formData.get("file") as File | null;
    const fileUri = formData.get("file_uri") as string | null;
    const fileMimeType = formData.get("mime_type") as string | null;
    const geminiKeyHintRaw = formData.get("gemini_key_index");
    const clientJobId = formData.get("job_id") as string | null;
    const unifiedPredictionRaw = formData.get("unified_prediction") as string | null;

    if (!file && !fileUri) {
      return NextResponse.json({ error: "BAD_REQUEST", detail: "请上传文件" }, { status: 400 });
    }

    const jobId = clientJobId || `lab-${crypto.randomUUID()}`;

    // ── D1 block: quota gating + idempotency ──
    // Each D1 op is wrapped so a transient DB error never blocks AI analysis.
    // Quota is enforced when D1 works; on DB failure we log + continue (best-effort).
    let dbReady = false;
    if (db) {
      try {
        if (!_schemaEnsured) {
          await ensureLabSchema(db);
          _schemaEnsured = true;
        }
        dbReady = true;
      } catch (schemaErr) {
        console.error("[lab] ensureLabSchema failed:", schemaErr instanceof Error ? schemaErr.message : schemaErr);
      }
    }

    if (db && dbReady) {
      try {
        const existingJob = await getLabJob(db, jobId);

        if (existingJob && (existingJob.user_id as string) !== user_id) {
          return NextResponse.json(
            { error: "FORBIDDEN", detail: "无权访问该分析任务" },
            { status: 403 }
          );
        }

        // Idempotency: return completed job immediately (no re-analysis)
        if (existingJob && existingJob.status === "completed" && existingJob.result_json) {
          const stored = JSON.parse(existingJob.result_json as string);
          const userTier: LabTier = (existingJob.tier as LabTier) || tier;
          const usage = await getLabUsageToday(db, user_id);
          return NextResponse.json({
            job_id: jobId,
            status: "completed",
            tier: userTier,
            report_tier: userTier === "pro" ? "pro" : "free",
            result: filterForTier(stored, userTier),
            quota: buildQuota(usage, is_pro),
          });
        }

        const isRetry = !!existingJob;

        if (!isRetry) {
          // New job: enforce quota first
          if (!is_pro) {
            const usage = await getLabUsageToday(db, user_id);
            if (usage >= LAB_FREE_DAILY_LIMIT) {
              return NextResponse.json(
                {
                  error: "QUOTA_EXCEEDED",
                  detail: "今日免费分析次数已用完。明天再来，或升级 Pro 继续练习。",
                  detail_en: "You've used today's included analyses. Come back tomorrow—or continue with Pro.",
                  quota: { used: usage, limit: LAB_FREE_DAILY_LIMIT, remaining: 0 },
                },
                { status: 429 }
              );
            }
          }
          await incrementLabUsage(db, user_id, jobId);
          await createLabJob(db, { id: jobId, user_id, tier });
        } else {
          // Retry of a failed/stuck job: UPDATE, never re-INSERT (avoids UNIQUE constraint)
          await markLabJobRetryProcessing(db, jobId, tier);
        }
      } catch (dbErr) {
        // D1 error during quota/job ops — log and continue to AI analysis without tracking
        console.error("[lab] D1 quota/job error (non-fatal):", dbErr instanceof Error ? dbErr.message : dbErr);
        dbReady = false;
      }
    }

    // Run AI analysis — Gemini host×key loop → Qwen final fallback
    const countryLab = (request.headers.get("cf-ipcountry") || "").toUpperCase();
    const hosts = getGeminiHosts(getCfEnvVal, countryLab === "CN");
    const keys = getGeminiKeys(getCfEnvVal);
    const keyHintParsed =
      geminiKeyHintRaw !== null && String(geminiKeyHintRaw) !== ""
        ? parseInt(String(geminiKeyHintRaw), 10)
        : NaN;
    const keysOrderedForUri =
      !!fileUri &&
      !Number.isNaN(keyHintParsed) &&
      keyHintParsed >= 0 &&
      keyHintParsed < keys.length
        ? [...keys.slice(keyHintParsed), ...keys.slice(0, keyHintParsed)]
        : keys;
    let parsed: Record<string, unknown> | null = null;
    let aiProvider: "gemini" | "qwen" = "gemini";
    let usedUriThenMultipart = false;
    try {
      if (fileUri) {
        console.log(`[AI][FILE] using_existing_file_id=${redactGeminiFileRefForLog(fileUri)}`);
        let lastUriErr: Error | null = null;
        let done = false;
        for (const host of hosts) {
          for (const key of keysOrderedForUri) {
            try {
              parsed = await labGeminiAnalysisWithUri(fileUri, fileMimeType || "video/mp4", host, key);
              done = true;
              break;
            } catch (e) {
              lastUriErr = e as Error;
              const em = lastUriErr.message;
              const httpSt = httpStatusFromBracketMessage(em);
              if (httpSt && isStaleGeminiFileReference(httpSt, em)) {
                console.log(
                  `[AI][FILE] existing_file_id_failed code=${httpSt} snippet=${em.substring(0, 160).replace(/\s+/g, " ")}`,
                );
              }
              if (shouldRetryNextGeminiKey(httpSt)) {
                console.log(`[lab] Gemini URI key retry on ${host}: ${em.substring(0, 120)}`);
                continue;
              }
              console.log(`[lab] Gemini URI via ${host} failed: ${em}`);
              break;
            }
          }
          if (done) break;
        }
        if (!done && !file) {
          throw lastUriErr || new Error("AI 服务不可用");
        }
        if (!done && file) {
          usedUriThenMultipart = true;
          console.log(
            `[AI][FILE] fallback_from_stale_file_reference=true reupload_started bytes=${file.size}`,
          );
          parsed = null;
        }
      }

      if (!parsed && file) {
        let geminiOk = false;
        let lastGeminiErr: Error | null = null;
        for (const host of hosts) {
          for (const key of keys) {
            try {
              parsed = await labGeminiAnalysis(file, host, key);
              geminiOk = true;
              console.log("[AI][FILE] analyze_with_new_file_id ok (lab multipart path)");
              break;
            } catch (e) {
              lastGeminiErr = e as Error;
              const httpSt = httpStatusFromBracketMessage(lastGeminiErr.message);
              if (shouldRetryNextGeminiKey(httpSt)) {
                console.log(`[lab] Gemini multipart key retry HTTP ${httpSt} on ${host}`);
                continue;
              }
              console.log(`[lab] Gemini via ${host} failed: ${lastGeminiErr.message}`);
              break;
            }
          }
          if (geminiOk) break;
        }
        if (!geminiOk) {
          if (getCfEnvVal("QWEN_API_KEY")) {
            console.log("[lab] All Gemini hosts/keys failed, falling back to Qwen");
            parsed = await labQwenAnalysis(file);
            aiProvider = "qwen";
          } else {
            throw lastGeminiErr || new Error("AI 服务不可用");
          }
        }
        if (usedUriThenMultipart && parsed) {
          console.log("[AI][FILE] reupload_success");
        }
      }

      if (!parsed) {
        throw new Error("AI 服务不可用");
      }
    } catch (aiErr) {
      if (db && dbReady) {
        try {
          await updateLabJobResult(db, jobId, "failed", JSON.stringify({ error: (aiErr as Error).message }));
        } catch { /* non-fatal */ }
      }
      return NextResponse.json(
        { error: "ANALYSIS_FAILED", detail: (aiErr as Error).message, job_id: jobId },
        { status: 502 }
      );
    }

    const normalized = applyUnifiedPrediction(parsed, unifiedPredictionRaw);

    // Store raw result in D1 (unfiltered — tier filtering happens on read)
    if (db && dbReady) {
      try {
        await updateLabJobResult(db, jobId, "completed", JSON.stringify(normalized));
        const snippet = (normalized.summary_zh as string) || (normalized.summary as string) || "";
        if (snippet) await updateLabJobSummary(db, jobId, snippet);
      } catch (storeErr) {
        console.error("[lab] D1 store result error (non-fatal):", storeErr instanceof Error ? storeErr.message : storeErr);
      }
    }

    // Build response with tier filtering
    let usage = 0;
    if (db && dbReady) {
      try {
        usage = await getLabUsageToday(db, user_id);
      } catch { /* non-fatal */ }
    }
    const filtered = filterForTier(normalized, tier);

    return NextResponse.json({
      job_id: jobId,
      status: "completed",
      tier,
      report_tier: tier === "pro" ? "pro" : "free",
      ai_provider: aiProvider,
      result: filtered,
      quota: buildQuota(usage, is_pro),
    });
  } catch (err) {
    console.error("[lab] POST error:", err);
    return NextResponse.json(
      { error: "INTERNAL", detail: err instanceof Error ? err.message : "分析错误" },
      { status: 500 }
    );
  }
}
