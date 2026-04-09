/**
 * GET /api/lab/compare?a=<jobId>&b=<jobId> — Pro-only swing comparison.
 *
 * Returns two completed lab jobs with their metrics side-by-side
 * and a diff object showing per-metric deltas.
 */

import { NextRequest, NextResponse } from "next/server";
import { isLabEnabled } from "@/lib/lab-config";
import { ensureLabSchema, getLabJobsForCompare } from "@/lib/lab-db";
import {
  authenticateRequest,
  getDB,
  requirePro,
  labError,
} from "@/lib/lab-auth";
import { jsonProduct } from "@/lib/chains";

export const runtime = "edge";

const COMPARE_METRIC_KEYS = [
  "ball_speed_mph",
  "launch_angle_deg",
  "launch_direction_deg",
  "tempo_ratio",
  "carry_distance_yards",
  "contact_quality_score",
  "backswing_time_sec",
  "downswing_time_sec",
] as const;

export async function GET(request: NextRequest) {
  try {
    if (!isLabEnabled()) {
      return labError("FEATURE_DISABLED", "Shot Lab 功能暂未开放", 404);
    }

    const authResult = await authenticateRequest(request);
    if (authResult instanceof NextResponse) return authResult;

    const proGate = requirePro(authResult);
    if (proGate) return proGate;

    const { searchParams } = new URL(request.url);
    const idA = searchParams.get("a");
    const idB = searchParams.get("b");

    if (!idA || !idB) {
      return labError("BAD_REQUEST", "需要提供两个分析 ID（参数 a 和 b）", 400);
    }
    if (idA === idB) {
      return labError("BAD_REQUEST", "请选择两个不同的分析进行对比", 400);
    }

    const db = getDB();
    if (!db) {
      return labError("DB_UNAVAILABLE", "数据库不可用", 503);
    }

    await ensureLabSchema(db);

    const jobs = await getLabJobsForCompare(db, authResult.user_id, [idA, idB]);

    if (jobs.length !== 2) {
      return labError("NOT_FOUND", "无法找到两个已完成的分析记录", 404);
    }

    const results = jobs.map((job) => {
      const parsed = JSON.parse(job.result_json as string);
      return {
        job_id: job.id,
        created_at: job.created_at,
        tier: job.tier,
        metrics: parsed.metrics || {},
        summary: parsed.summary,
        summary_zh: parsed.summary_zh,
        issues_count: (parsed.issues || []).length,
      };
    });

    const diff: Record<string, { a: number | null; b: number | null; delta: number | null }> = {};
    for (const key of COMPARE_METRIC_KEYS) {
      const valA = (results[0].metrics[key] as number) ?? null;
      const valB = (results[1].metrics[key] as number) ?? null;
      diff[key] = {
        a: valA,
        b: valB,
        delta: valA != null && valB != null ? valB - valA : null,
      };
    }

    return jsonProduct({ shots: results, diff }, { status: 200 }, "lab");
  } catch (err) {
    console.error("[lab] compare error:", err);
    return labError("INTERNAL", "对比失败，请稍后重试。", 500);
  }
}
