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
import { getRequestContext } from "@cloudflare/next-on-pages";
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
  updateLabJobVideoR2Key,
} from "@/lib/lab-db";
import {
  authenticateRequest,
  getDB,
  filterForTier,
  buildQuota,
} from "@/lib/lab-auth";
import {
  getGeminiHosts,
  getGeminiKeys,
  isStaleGeminiFileReference,
  redactGeminiFileRefForLog,
  shouldRetryNextGeminiKey,
} from "@/lib/gemini-proxy";
import { jsonProduct } from "@/lib/chains";
import { getEdgeJwtSecret } from "@/lib/chains/jwt-secret";
import { unsealUploadSession } from "@/lib/chains/upload-session";
import {
  httpStatusFromBracketMessage,
  labGeminiAnalysis,
  labGeminiAnalysisWithUri,
  labQwenAnalysis,
} from "@/lib/chains/analysis/lab-gemini";

export const runtime = "edge";

let _schemaEnsured = false;

function getR2Bucket() {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (getRequestContext().env as any).R2_BUCKET || null;
  } catch {
    return null;
  }
}

function labSourceFileExtension(file: File): string {
  const ext = (file.name?.split(".").pop() || "").toLowerCase();
  if (["mp4", "webm", "mov", "m4v", "avi", "jpg", "jpeg", "png", "webp"].includes(ext)) return ext === "jpeg" ? "jpg" : ext;
  const t = (file.type || "").toLowerCase();
  if (t.includes("webm")) return "webm";
  if (t.includes("quicktime") || t.includes("mov")) return "mov";
  if (t.includes("png")) return "png";
  if (t.startsWith("image/")) return "jpg";
  return "mp4";
}

function toNum(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
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
    let fileUri = formData.get("file_uri") as string | null;
    let fileMimeType = formData.get("mime_type") as string | null;
    let geminiKeyHintRaw = formData.get("gemini_key_index");
    const clientJobId = formData.get("job_id") as string | null;
    const unifiedPredictionRaw = formData.get("unified_prediction") as string | null;
    const uploadTokenRaw = formData.get("upload_token") as string | null;

    if (uploadTokenRaw && typeof uploadTokenRaw === "string") {
      const secret = getEdgeJwtSecret();
      if (!secret) {
        return NextResponse.json(
          { error: "SERVER_MISCONFIGURED", detail: "上传会话不可用" },
          { status: 503 },
        );
      }
      const sess = await unsealUploadSession(uploadTokenRaw, secret);
      if (!sess) {
        return NextResponse.json(
          { error: "BAD_REQUEST", detail: "上传会话无效或已过期" },
          { status: 400 },
        );
      }
      if (!fileUri) fileUri = sess.file_uri;
      if (!fileMimeType) fileMimeType = sess.mime_type;
      if (geminiKeyHintRaw === null || geminiKeyHintRaw === "") {
        geminiKeyHintRaw = String(sess.gemini_key_index);
      }
    }

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
          return jsonProduct(
            {
              job_id: jobId,
              status: "completed",
              tier: userTier,
              report_tier: userTier === "pro" ? "pro" : "free",
              result: filterForTier(stored, userTier),
              quota: buildQuota(usage, is_pro),
            },
            undefined,
            "lab",
          );
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
        { error: "ANALYSIS_FAILED", detail: "分析未完成，请稍后重试。", job_id: jobId },
        { status: 502 },
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

    // Best-effort: keep source media in R2 so unified history can "re-analyze" Shot Lab later.
    if (db && dbReady && file && file.size > 0) {
      try {
        const r2 = getR2Bucket();
        if (r2) {
          const buf = await file.arrayBuffer();
          const ext = labSourceFileExtension(file);
          const key = `lab-videos/${user_id}/${jobId}.${ext}`;
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          await (r2 as any).put(key, buf, {
            httpMetadata: {
              contentType: file.type || "application/octet-stream",
            },
          });
          await updateLabJobVideoR2Key(db, jobId, key);
        }
      } catch (persistErr) {
        console.warn(
          "[lab] R2 source persist (non-fatal):",
          persistErr instanceof Error ? persistErr.message : persistErr,
        );
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

    return jsonProduct(
      {
        job_id: jobId,
        status: "completed",
        tier,
        report_tier: tier === "pro" ? "pro" : "free",
        result: filtered,
        quota: buildQuota(usage, is_pro),
      },
      undefined,
      "lab",
    );
  } catch (err) {
    console.error("[lab] POST error:", err);
    return NextResponse.json({ error: "INTERNAL", detail: "分析异常，请稍后重试。" }, { status: 500 });
  }
}
