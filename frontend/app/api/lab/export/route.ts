/**
 * POST /api/lab/export — Pro-only: export a lab job as structured JSON report.
 *
 * Body: { job_id: string }
 * Returns the full unfiltered result for Pro users.
 */

import { NextRequest, NextResponse } from "next/server";
import { isLabEnabled } from "@/lib/lab-config";
import { ensureLabSchema, getLabJobForExport } from "@/lib/lab-db";
import {
  authenticateRequest,
  getDB,
  requirePro,
  labError,
} from "@/lib/lab-auth";

export const runtime = "edge";

export async function POST(request: NextRequest) {
  try {
    if (!isLabEnabled()) {
      return labError("FEATURE_DISABLED", "Shot Lab 功能暂未开放", 404);
    }

    const authResult = await authenticateRequest(request);
    if (authResult instanceof NextResponse) return authResult;

    const proGate = requirePro(authResult);
    if (proGate) return proGate;

    let body: Record<string, unknown>;
    try {
      body = await request.json();
    } catch {
      return labError("BAD_REQUEST", "请求体格式错误", 400);
    }

    const jobId = body.job_id as string;
    if (!jobId) {
      return labError("BAD_REQUEST", "缺少 job_id 参数", 400);
    }

    const db = getDB();
    if (!db) {
      return labError("DB_UNAVAILABLE", "数据库不可用", 503);
    }

    await ensureLabSchema(db);

    const job = await getLabJobForExport(db, authResult.user_id, jobId);
    if (!job) {
      return labError("NOT_FOUND", "未找到该分析记录或尚未完成", 404);
    }

    const parsed = JSON.parse(job.result_json as string);

    return NextResponse.json({
      job_id: job.id,
      created_at: job.created_at,
      tier: "pro",
      report_tier: "pro",
      result: parsed,
      export_format: "json",
      exported_at: new Date().toISOString(),
    });
  } catch (err) {
    console.error("[lab] export error:", err);
    return labError("INTERNAL", err instanceof Error ? err.message : "导出失败", 500);
  }
}
