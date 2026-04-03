/**
 * Shot Lab feature flag and tier configuration.
 * Toggle FEATURE_SHOT_LAB env var to "0" to disable the entire module
 * without affecting /analyze or any other existing product surface.
 */

import { getRequestContext } from "@cloudflare/next-on-pages";

// ── Quota & retention defaults (configurable via env in future) ──

export const LAB_FREE_DAILY_LIMIT = 3;
export const LAB_PRO_DAILY_LIMIT = 999;
export const LAB_FREE_HISTORY_DAYS = 7;
export const LAB_FREE_HISTORY_MAX_ITEMS = 10;
export const LAB_FREE_MAX_ISSUES = 3;
export const LAB_FREE_MAX_DRILLS = 2;
export const LAB_PRO_HISTORY_MAX_ITEMS = 200;
export const LAB_TREND_MAX_DAYS = 90;
export const LAB_TREND_MAX_POINTS = 50;

// ── Feature flag ──

export function getCfEnvVal(key: string): string {
  try {
    return ((getRequestContext().env as Record<string, string>)[key] as string) || "";
  } catch {
    return process.env[key] || "";
  }
}

export function isLabEnabled(): boolean {
  const flag = getCfEnvVal("FEATURE_SHOT_LAB");
  // Enabled by default; disabled only when explicitly set to "0" or "false"
  return flag !== "0" && flag.toLowerCase() !== "false";
}

// ── Tier helpers ──

export type LabTier = "free" | "pro";

export interface LabQuotaInfo {
  used: number;
  limit: number | null; // null = unlimited
  remaining: number; // -1 = unlimited
  is_pro: boolean;
}
