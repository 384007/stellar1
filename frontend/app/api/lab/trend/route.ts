/**
 * GET /api/lab/trend — Pro-only trend analytics.
 *
 * Returns time-series metrics from completed lab jobs
 * for charting progress over time.
 */

import { NextRequest, NextResponse } from "next/server";
import { isLabEnabled, LAB_TREND_MAX_DAYS, LAB_TREND_MAX_POINTS } from "@/lib/lab-config";
import { ensureLabSchema, getLabTrendData } from "@/lib/lab-db";
import {
  authenticateRequest,
  getDB,
  requirePro,
  labError,
} from "@/lib/lab-auth";
import type { LabTrendPoint } from "@/lib/lab-types";
import { jsonProduct } from "@/lib/chains";

export const runtime = "edge";

export async function GET(request: NextRequest) {
  try {
    if (!isLabEnabled()) {
      return labError("FEATURE_DISABLED", "Shot Lab 功能暂未开放", 404);
    }

    const authResult = await authenticateRequest(request);
    if (authResult instanceof NextResponse) return authResult;

    const proGate = requirePro(authResult);
    if (proGate) return proGate;

    const db = getDB();
    if (!db) {
      return labError("DB_UNAVAILABLE", "数据库不可用", 503);
    }

    await ensureLabSchema(db);

    const { searchParams } = new URL(request.url);
    const days = Math.min(
      parseInt(searchParams.get("days") || String(LAB_TREND_MAX_DAYS), 10) || LAB_TREND_MAX_DAYS,
      LAB_TREND_MAX_DAYS
    );

    const rows = await getLabTrendData(db, authResult.user_id, {
      maxDays: days,
      maxPoints: LAB_TREND_MAX_POINTS,
    });

    const points: LabTrendPoint[] = rows.map((row) => {
      let metrics: Record<string, unknown> = {};
      try {
        const parsed = JSON.parse(row.result_json as string);
        metrics = parsed.metrics || {};
      } catch { /* corrupted row */ }

      return {
        job_id: row.id as string,
        date: row.created_at as string,
        ball_speed_mph: (metrics.ball_speed_mph as number) ?? null,
        launch_angle_deg: (metrics.launch_angle_deg as number) ?? null,
        tempo_ratio: (metrics.tempo_ratio as number) ?? null,
        carry_distance_yards: (metrics.carry_distance_yards as number) ?? null,
        contact_quality_score: (metrics.contact_quality_score as number) ?? null,
      };
    });

    return jsonProduct(
      {
        points,
        period_days: days,
        total_sessions: points.length,
      },
      { status: 200 },
      "lab",
    );
  } catch (err) {
    console.error("[lab] trend error:", err);
    return labError("INTERNAL", "趋势查询失败，请稍后重试。", 500);
  }
}
