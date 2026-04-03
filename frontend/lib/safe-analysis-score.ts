/**
 * Never coerce withheld / partial Plus scores to 0 — use null and explicit UI copy.
 */

export function normalizedTotalScoreForStorage(v: unknown): number | null {
  if (v === null || v === undefined) return null;
  if (typeof v === "number" && Number.isFinite(v)) return v;
  return null;
}

export function isPlusScoreWithheld(result: {
  report_status?: string | null;
  final_ui_safe_score_state?: string | null;
  total_score?: number | null;
  posture_score?: number | null;
}): boolean {
  if (result.report_status === "unavailable_due_to_unreliable_keyframes") return true;
  if (result.final_ui_safe_score_state === "null_withheld_unreliable_keyframes") return true;
  if (result.total_score === null || result.total_score === undefined) return true;
  if (result.posture_score === null || result.posture_score === undefined) return true;
  return false;
}
