/**
 * Pro analyze (`/pro-v2/analyze`) returns a minimal JSON contract. Expand to the shape
 * the Pro UI still expects (skeleton / prediction placeholders).
 */

import type { PlusAnalysisResult } from "@/components/PlusResultView";

const SWING_PHASE_IDS = [
  "address",
  "takeaway",
  "backswing",
  "top",
  "downswing",
  "impact",
  "follow_through",
  "finish",
] as const;

function derivePhaseKeyframesFromStrip(
  keyframes: Array<{ phase?: string; source_pose_idx?: unknown; source_frame_index?: unknown }>,
): Record<string, number> | undefined {
  const out: Record<string, number> = {};
  for (const kf of keyframes) {
    const ph = String(kf.phase ?? "")
      .toLowerCase()
      .replace(/[\s-]+/g, "_");
    const spi = kf.source_pose_idx;
    const sfi = kf.source_frame_index;
    const idx =
      typeof spi === "number" && Number.isFinite(spi)
        ? spi
        : typeof sfi === "number" && Number.isFinite(sfi)
          ? sfi
          : undefined;
    if (ph && idx !== undefined) {
      out[ph] = idx;
    }
  }
  return Object.keys(out).length ? out : undefined;
}

/** 后端若未返回 analysis_id，历史 POST 会 400；在此生成稳定 id 并写回 raw。 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function ensureAnalysisIdOnRaw(raw: Record<string, any>): void {
  const cur = String(raw.analysis_id ?? "").trim();
  if (cur) return;
  raw.analysis_id =
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `local-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

function mergePhaseKeyframeMaps(
  rawPk: unknown,
  keyframes: Array<{ phase?: string; source_pose_idx?: unknown; source_frame_index?: unknown }>,
): Record<string, number> | undefined {
  const fromStrip = derivePhaseKeyframesFromStrip(keyframes) ?? {};
  const fromRaw =
    rawPk && typeof rawPk === "object" && !Array.isArray(rawPk)
      ? { ...(rawPk as Record<string, number>) }
      : {};
  const merged = { ...fromStrip, ...fromRaw };
  return Object.keys(merged).length ? merged : undefined;
}

/** Coerce API total_score (number or numeric string) for UI / withheld checks. */
function parseProTotalScore(v: unknown): number {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string") {
    const n = parseFloat(v.replace(/,/g, "").trim());
    return Number.isFinite(n) ? n : 0;
  }
  return 0;
}

/**
 * Map Stellar Pro expanded payload → PlusResultView model (same tabs / skeleton / video).
 * Analysis text and scores still come from Stellar Pro; Plus-only fields are synthesized.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function proExpandedToPlusViewModel(r: Record<string, any>): PlusAnalysisResult {
  ensureAnalysisIdOnRaw(r);
  const issues = Array.isArray(r.issues) ? (r.issues as string[]) : [];
  const issues_zh = Array.isArray(r.issues_zh) ? (r.issues_zh as string[]) : [];
  const suggestions = Array.isArray(r.suggestions) ? (r.suggestions as string[]) : [];
  const suggestions_zh = Array.isArray(r.suggestions_zh) ? (r.suggestions_zh as string[]) : [];
  const summary = String(r.summary ?? "").trim();
  const summary_zh = String(r.summary_zh ?? r.summary ?? "").trim();
  const total = parseProTotalScore(r.total_score);
  const posture_score = Math.min(10, Math.max(0, total / 10));

  const firstIssZh = issues_zh[0] || (summary_zh ? summary_zh.slice(0, 100) : "");
  const firstIssEn = issues[0] || (summary ? summary.slice(0, 100) : "");

  const primary_diagnosis = {
    title_zh: firstIssZh || "挥杆分析摘要",
    title_en: firstIssEn || "Swing analysis summary",
    status_zh: "需要关注",
    status_en: "Needs attention",
    ai_confidence: 72,
  };

  const additional_issues = issues_zh.slice(1, 8).map((title_zh, i) => ({
    title_zh,
    title_en: issues[i + 1] || title_zh,
    status_zh: "需注意",
    status_en: "Note",
  }));

  const keyframes = Array.isArray(r.keyframes) ? r.keyframes : [];
  const phasesWithKf = new Set(
    keyframes.map((k: { phase?: string }) =>
      String(k.phase ?? "")
        .toLowerCase()
        .replace(/[\s-]+/g, "_"),
    ),
  );
  const swing_phase_evaluations = SWING_PHASE_IDS.map((phase) => ({
    phase,
    status: phasesWithKf.has(phase) ? "ok" : "ok",
    note_zh: "",
    note_en: "",
  }));

  const plan = r.training_plan as
    | Record<string, { focus: string; drills: string[]; duration?: string }>
    | undefined;
  const day1 = plan && typeof plan === "object" ? Object.values(plan)[0] : undefined;
  const drillZh = day1?.drills?.filter(Boolean).join("；") ?? "";
  const drillEn = day1?.drills?.filter(Boolean).join("; ") ?? "";
  const training = {
    title_zh: day1?.focus || "针对性训练",
    title_en: day1?.focus || "Targeted training",
    description_zh: drillZh || summary_zh.slice(0, 280) || "根据报告中的建议完成练习。",
    description_en: drillEn || summary.slice(0, 280) || "Follow the report suggestions.",
    difficulty: "normal",
    frequency_percent: 0,
  };

  const scoresRaw = r.scores && typeof r.scores === "object" ? (r.scores as Record<string, number>) : null;
  const pk = mergePhaseKeyframeMaps(r.phase_keyframes, keyframes);

  return {
    analysis_id: String(r.analysis_id ?? ""),
    type: "pro",
    posture_score,
    primary_diagnosis,
    additional_issues,
    quick_tip_zh: suggestions_zh[0] || summary_zh.slice(0, 200),
    quick_tip_en: suggestions[0] || summary.slice(0, 200),
    problem_description_zh: summary_zh.slice(0, 500) || summary_zh,
    problem_description_en: summary.slice(0, 500) || summary,
    swing_phase_evaluations,
    training,
    recommended_videos: [],
    scores: scoresRaw && Object.keys(scoresRaw).length > 0 ? scoresRaw : null,
    total_score: total,
    issues,
    issues_zh,
    suggestions,
    suggestions_zh,
    summary,
    summary_zh,
    keyframes: keyframes as PlusAnalysisResult["keyframes"],
    skeleton_data:
      r.skeleton_data && typeof r.skeleton_data === "object"
        ? (r.skeleton_data as PlusAnalysisResult["skeleton_data"])
        : { frames: [], total_frames: 0 },
    pose_frames: Array.isArray(r.pose_frames) ? r.pose_frames : [],
    phase_keyframes: pk,
    prediction: r.prediction && typeof r.prediction === "object" ? r.prediction : undefined,
    video_meta: r.video_meta && typeof r.video_meta === "object" ? r.video_meta : undefined,
    keyframes_degraded: r.keyframes_degraded,
    keyframe_display_mode: r.keyframe_display_mode,
    final_keyframe_gate_pass: r.final_keyframe_gate_pass,
    training_plan:
      r.training_plan && typeof r.training_plan === "object"
        ? (r.training_plan as PlusAnalysisResult["training_plan"])
        : undefined,
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function expandStellarProForUi(raw: Record<string, any>): Record<string, any> {
  ensureAnalysisIdOnRaw(raw);
  const summary = String(raw.summary ?? "").trim();
  const summary_zh = String(raw.summary_zh ?? raw.summary ?? "").trim();
  const keyframes = Array.isArray(raw.keyframes) ? raw.keyframes : [];
  const emptyPrediction = {
    predicted_distance: 0,
    lateral_offset: 0,
    shot_shape: "unknown",
    shot_shape_zh: "未知",
    club_head_speed: 0,
    ball_speed: 0,
    launch_angle: 0,
    spin_rate: 0,
    smash_factor: 0,
    trajectory: [] as { t: number; x: number; y: number; lateral: number }[],
  };

  return {
    ...raw,
    type: raw.type ?? "pro",
    analysis_id: String(raw.analysis_id ?? ""),
    status: raw.status ?? "completed",
    summary,
    summary_zh,
    scores: raw.scores ?? {},
    total_score: parseProTotalScore(raw.total_score),
    issues: raw.issues ?? [],
    issues_zh: raw.issues_zh ?? [],
    suggestions: raw.suggestions ?? [],
    suggestions_zh: raw.suggestions_zh ?? [],
    advanced_metrics: raw.advanced_metrics ?? {},
    training_plan: raw.training_plan ?? {},
    keyframes: keyframes.map((kf: Record<string, unknown>) => {
      const phase = String(kf.phase ?? "unknown");
      return {
        ...kf,
        phase,
        label_en: String(kf.label_en ?? phase),
        label_zh: String(kf.label_zh ?? phase),
        timestamp: Number(kf.timestamp ?? 0),
        image_base64: String(kf.image_base64 ?? ""),
      };
    }),
    skeleton_data: raw.skeleton_data ?? { frames: [], total_frames: 0 },
    prediction: raw.prediction ?? emptyPrediction,
    trajectory: raw.trajectory ?? [],
    pose_frames: Array.isArray(raw.pose_frames) ? raw.pose_frames : [],
    phase_keyframes: mergePhaseKeyframeMaps(
      raw.phase_keyframes,
      Array.isArray(raw.keyframes) ? raw.keyframes : [],
    ),
    video_meta: raw.video_meta ?? {},
    contact_sheet_url: raw.contact_sheet_url ?? null,
    video_url: raw.video_url ?? null,
  };
}
