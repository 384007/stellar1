/**
 * GET /api/lab/:id — Retrieve a Shot Lab job result.
 *
 * Returns the job status and tier-filtered result.
 * Auth required; user can only access their own jobs.
 */

import { NextRequest, NextResponse } from "next/server";
import { isLabEnabled, type LabTier } from "@/lib/lab-config";
import { ensureLabSchema, getLabJob, getLabUsageToday } from "@/lib/lab-db";
import {
  authenticateRequest,
  getDB,
  filterForTier,
  buildQuota,
  labError,
} from "@/lib/lab-auth";
import { jsonProduct } from "@/lib/chains";

export const runtime = "edge";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    if (!isLabEnabled()) {
      return labError("FEATURE_DISABLED", "Shot Lab 功能暂未开放", 404);
    }

    const authResult = await authenticateRequest(request);
    if (authResult instanceof NextResponse) return authResult;
    const { user_id, is_pro } = authResult;

    const db = getDB();
    if (!db) {
      return labError("DB_UNAVAILABLE", "数据库不可用", 503);
    }

    await ensureLabSchema(db);

    const { id: jobId } = await params;
    const job = await getLabJob(db, jobId);

    if (!job) {
      return labError("NOT_FOUND", "任务不存在", 404);
    }

    if (job.user_id !== user_id) {
      return labError("FORBIDDEN", "无权访问此任务", 403);
    }

    const tier: LabTier = is_pro ? "pro" : (job.tier as LabTier) || "free";
    const usage = await getLabUsageToday(db, user_id);

    const response: Record<string, unknown> = {
      job_id: jobId,
      status: job.status,
      tier,
      created_at: job.created_at,
      updated_at: job.updated_at,
      quota: buildQuota(usage, is_pro),
    };

    if (job.status === "completed" && job.result_json) {
      const stored = JSON.parse(job.result_json as string);
      response.report_tier = tier === "pro" ? "pro" : "free";
      response.result = filterForTier(stored, tier);
    }

    return jsonProduct(response, undefined, "lab");
  } catch (err) {
    console.error("[lab] GET job error:", err);
    return labError("INTERNAL", "查询失败，请稍后重试。", 500);
  }
}
