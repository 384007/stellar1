/**
 * Shared Shot Lab authentication and tier-filtering helpers.
 * Used by all /api/lab/* routes to avoid duplication.
 */

import { NextRequest, NextResponse } from "next/server";
import { getRequestContext } from "@cloudflare/next-on-pages";
import { jwtVerify } from "jose";
import {
  getCfEnvVal,
  LAB_FREE_DAILY_LIMIT,
  LAB_FREE_MAX_ISSUES,
  LAB_FREE_MAX_DRILLS,
  type LabTier,
} from "./lab-config";
import type { FieldsVisibility, LabQuotaResponse, LabErrorCode } from "./lab-types";

export interface AuthResult {
  user_id: string;
  is_pro: boolean;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function getDB(): any {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (getRequestContext().env as any).DB;
  } catch {
    return null;
  }
}

export function getJwtSecret(): Uint8Array {
  const secret = getCfEnvVal("JWT_SECRET");
  return new TextEncoder().encode(secret);
}

export async function authenticateRequest(
  request: NextRequest
): Promise<AuthResult | NextResponse> {
  const auth = request.headers.get("authorization") || "";
  const token = auth.startsWith("Bearer ") ? auth.slice(7) : "";

  if (!token) {
    return labError("UNAUTHORIZED", "请先登录", 401);
  }
  if (token.startsWith("guest-")) {
    return labError("UNAUTHORIZED", "游客模式不支持 Shot Lab，请注册或登录", 403);
  }
  if (token.startsWith("local-")) {
    return { user_id: token, is_pro: false };
  }
  if (!token.includes(".")) {
    return labError("UNAUTHORIZED", "登录状态无效", 401);
  }

  const secret = getJwtSecret();
  if (secret.length === 0) {
    console.error("[lab] JWT_SECRET not configured, skipping verification");
    return { user_id: "unknown", is_pro: false };
  }

  try {
    const { payload } = await jwtVerify(token, secret);
    return {
      user_id: (payload.user_id as string) || "unknown",
      is_pro: !!payload.is_pro,
    };
  } catch {
    return labError("UNAUTHORIZED", "登录已过期，请重新登录", 401);
  }
}

export function labError(
  code: LabErrorCode,
  detail: string,
  status: number,
  extra?: Record<string, unknown>
): NextResponse {
  return NextResponse.json({ error: code, detail, ...extra }, { status });
}

export function requirePro(auth: AuthResult): NextResponse | null {
  if (auth.is_pro) return null;
  return labError(
    "PRO_REQUIRED",
    "此功能为 Pro 专属。升级 Pro 解锁完整分析能力。",
    403,
    { detail_en: "This feature requires Pro. Upgrade to unlock full analysis capabilities." }
  );
}

export function buildQuota(usage: number, is_pro: boolean): LabQuotaResponse {
  return {
    used: usage,
    limit: is_pro ? null : LAB_FREE_DAILY_LIMIT,
    remaining: is_pro ? -1 : Math.max(0, LAB_FREE_DAILY_LIMIT - usage),
    is_pro,
  };
}

export function buildFieldsVisibility(tier: LabTier): FieldsVisibility {
  if (tier === "pro") {
    return {
      backswing_time: "visible",
      downswing_time: "visible",
      full_report: "visible",
      full_issues: "visible",
      full_drills: "visible",
      compare: "visible",
      trend: "visible",
      export: "visible",
      trajectory_full: "visible",
    };
  }
  return {
    backswing_time: "locked",
    downswing_time: "locked",
    full_report: "preview",
    full_issues: "locked",
    full_drills: "locked",
    compare: "locked",
    trend: "locked",
    export: "locked",
    trajectory_full: "locked",
  };
}

/**
 * Tier-based response filtering.
 * Pro: full data. Free: capped issues/drills, locked metrics, stripped reports.
 */
export function filterForTier(
  parsed: Record<string, unknown>,
  tier: LabTier
): Record<string, unknown> {
  const visibility = buildFieldsVisibility(tier);

  if (tier === "pro") {
    return { ...parsed, report_tier: "pro", fields_visibility: visibility };
  }

  const issues = (parsed.issues as Array<Record<string, unknown>>) || [];
  const drills = (parsed.drills as Array<Record<string, unknown>>) || [];
  const metrics = (parsed.metrics as Record<string, unknown>) || {};

  const fullReportZh = (parsed.full_report_zh as string) || "";
  const fullReport = (parsed.full_report as string) || "";
  const previewLen = Math.floor(fullReportZh.length * 0.3);
  const previewLenEn = Math.floor(fullReport.length * 0.3);

  return {
    ...parsed,
    report_tier: "free" as const,
    fields_visibility: visibility,
    issues: issues.slice(0, LAB_FREE_MAX_ISSUES),
    issues_total: issues.length,
    drills: drills.slice(0, LAB_FREE_MAX_DRILLS),
    drills_total: drills.length,
    metrics: {
      ...metrics,
      backswing_time_sec: null,
      downswing_time_sec: null,
    },
    full_report: previewLenEn > 0 ? fullReport.slice(0, previewLenEn) + "…" : undefined,
    full_report_zh: previewLen > 0 ? fullReportZh.slice(0, previewLen) + "…" : undefined,
    full_report_preview: true,
  };
}

export function getRegion(request: NextRequest): "CN" | "global" {
  return request.headers.get("CF-IPCountry")?.toUpperCase() === "CN" ? "CN" : "global";
}
