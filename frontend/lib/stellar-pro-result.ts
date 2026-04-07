/**
 * Pro v3 product response (`POST /pro-v3/analyze` — see `routers.prov3_api`). Expand to the shape
 * the Pro UI still expects (skeleton / prediction placeholders).
 */

import type { PlusAnalysisResult } from "@/components/PlusResultView";
import { normalizeProv3MediaInRaw } from "@/lib/prov3-media-url";

/**
 * Low-trust / preview-only strip: only when ``final_status`` is **explicitly** non-``pass``,
 * or trust is ``low_trust``, or ``low_trust_preview_only``.
 * Missing/empty ``final_status`` → **not** low trust (matches ``PlusResultView.isLowTrustPreviewOnly``).
 */
export function stellarProTrustIsLow(r: {
  final_status?: unknown;
  analysis_trust?: unknown;
  trust_level?: unknown;
  low_trust_preview_only?: unknown;
}): boolean {
  const fs = String(r.final_status ?? "").trim();
  const explicitNonPass = fs.length > 0 && fs !== "pass";
  const trust = String(r.analysis_trust ?? r.trust_level ?? "");
  return explicitNonPass || trust === "low_trust" || r.low_trust_preview_only === true;
}

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
  keyframes: Array<{ phase?: string; frame_index?: unknown; source_pose_idx?: unknown; source_frame_index?: unknown }>,
): Record<string, number> | undefined {
  const out: Record<string, number> = {};
  for (const kf of keyframes) {
    const ph = String(kf.phase ?? "")
      .toLowerCase()
      .replace(/[\s-]+/g, "_");
    const spi = kf.source_pose_idx;
    const sfi = kf.source_frame_index;
    const fi = kf.frame_index;
    const idx =
      typeof fi === "number" && Number.isFinite(fi)
        ? fi
        : typeof spi === "number" && Number.isFinite(spi)
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

/**
 * 与历史页 ``parseResult`` 一致：按信任度整理顶层 ``keyframes``、``official_phase_keyframes``、``preview_keyframes``。
 * 低信任且 ``preview_keyframes`` 为空但顶层 ``keyframes`` 仍有数据时，迁入 ``preview_keyframes`` 并清空顶层（be60305 契约 + 旧后端兼容）。
 * 原地修改 ``r``。须在 ``normalizeProv3MediaInRaw`` 之前调用。
 */
export function normalizeProResultKeyframeArraysForTrust(r: Record<string, unknown>): void {
  if (!r || typeof r !== "object") return;
  const lowTrust = stellarProTrustIsLow(r);
  const keyframes = Array.isArray(r.keyframes) ? r.keyframes : [];
  const official = Array.isArray(r.official_phase_keyframes) ? r.official_phase_keyframes : [];
  const preview = Array.isArray(r.preview_keyframes) ? r.preview_keyframes : [];
  if (!lowTrust) {
    if (official.length > 0) {
      r.keyframes = official;
    }
  } else {
    if (preview.length === 0 && keyframes.length > 0) {
      r.preview_keyframes = keyframes;
    }
    r.keyframes = [];
  }
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
  keyframes: Array<{ phase?: string; frame_index?: unknown; source_pose_idx?: unknown; source_frame_index?: unknown }>,
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
  normalizeProResultKeyframeArraysForTrust(r as Record<string, unknown>);
  normalizeProv3MediaInRaw(r as Record<string, unknown>);
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

  const official = Array.isArray(r.official_phase_keyframes)
    ? r.official_phase_keyframes.filter((k: unknown) => k != null && typeof k === "object" && !Array.isArray(k))
    : [];
  const keyframesTop = Array.isArray(r.keyframes)
    ? r.keyframes.filter((k: unknown) => k != null && typeof k === "object" && !Array.isArray(k))
    : [];
  const preview = Array.isArray(r.preview_keyframes)
    ? r.preview_keyframes.filter((k: unknown) => k != null && typeof k === "object" && !Array.isArray(k))
    : [];
  const isLowTrust = stellarProTrustIsLow(r);
  const stripKfB64 = String(r.pipeline ?? "") === "prov3";
  /** High trust: prefer ``official_phase_keyframes``; some payloads only populate top-level ``keyframes``. */
  const officialForModel = isLowTrust
    ? official
    : official.length > 0
      ? official
      : keyframesTop;
  const displayKeyframesRaw = isLowTrust ? [] : officialForModel;
  type NormalizedKf = {
    phase: string;
    label_en: string;
    label_zh: string;
    frame_index: number;
    timestamp: number;
    image_base64: string;
    keyframe_image_url?: string;
    keyframe_image_source?: string;
  } & Record<string, unknown>;
  const normalizeKeyframes = (arr: Array<Record<string, unknown>>) =>
    arr.map((kf) => {
      const phaseRaw = String((kf as { phase?: unknown }).phase ?? "").trim().toLowerCase().replace(/[\s-]+/g, "_");
      const fallbackLabel = phaseRaw || "frame";
      const frameIndexRaw = Number((kf as { frame_index?: unknown }).frame_index ?? 0);
      const timestampRaw = Number((kf as { timestamp?: unknown }).timestamp ?? 0);
      return {
        ...(kf as Record<string, unknown>),
        phase: phaseRaw || fallbackLabel,
        label_en:
          typeof (kf as { label_en?: unknown }).label_en === "string"
            ? String((kf as { label_en?: string }).label_en)
            : fallbackLabel,
        label_zh:
          typeof (kf as { label_zh?: unknown }).label_zh === "string"
            ? String((kf as { label_zh?: string }).label_zh)
            : fallbackLabel,
        frame_index: Number.isFinite(frameIndexRaw) ? frameIndexRaw : 0,
        timestamp: Number.isFinite(timestampRaw) ? timestampRaw : 0,
        image_base64: stripKfB64
          ? ""
          : typeof (kf as { image_base64?: unknown }).image_base64 === "string"
            ? String((kf as { image_base64?: string }).image_base64)
            : "",
        keyframe_image_url:
          typeof (kf as { keyframe_image_url?: unknown }).keyframe_image_url === "string"
            ? String((kf as { keyframe_image_url?: string }).keyframe_image_url)
            : undefined,
        keyframe_image_source:
          typeof (kf as { keyframe_image_source?: unknown }).keyframe_image_source === "string"
            ? String((kf as { keyframe_image_source?: string }).keyframe_image_source)
            : undefined,
      };
    });
  const normalizedOfficial = normalizeKeyframes(officialForModel as Array<Record<string, unknown>>) as NormalizedKf[];
  const normalizedPreview = normalizeKeyframes(preview as Array<Record<string, unknown>>) as NormalizedKf[];
  const normalizedDisplay = normalizeKeyframes(displayKeyframesRaw as Array<Record<string, unknown>>) as NormalizedKf[];
  /** 低信任 Pro v3：Plus 部分逻辑仍读 ``keyframes``；与条图一致用 ``preview`` */
  const keyframesForPlusModel =
    isLowTrust && normalizedPreview.length > 0 ? normalizedPreview : normalizedDisplay;
  const phasesWithKf = new Set(
    keyframesForPlusModel.map((k) => String(k.phase ?? "").toLowerCase().replace(/[\s-]+/g, "_")),
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
  /** Gemini / limited 报告可能省略 drills 或非数组 — 避免 `undefined.join` 抛错整页白屏 */
  const drillsArr = day1 && Array.isArray(day1.drills) ? day1.drills : [];
  const trimmedDrills = drillsArr.map((x) => String(x ?? "").trim()).filter(Boolean);
  const drillZh = trimmedDrills.join("；");
  /** Pro v3 训练计划 drills 多为纯中文：英文区用 summary 摘要，避免把中文塞进 description_en */
  const drillsLatin = trimmedDrills.filter((x) => /[A-Za-z]/.test(x));
  const drillEn =
    drillsLatin.length > 0
      ? drillsLatin.join("; ")
      : summary.slice(0, 280) || "Follow the report suggestions.";
  const training = {
    title_zh: day1?.focus || "针对性训练",
    title_en: day1?.focus || "Targeted training",
    description_zh: drillZh || summary_zh.slice(0, 280) || "根据报告中的建议完成练习。",
    description_en: drillEn || summary.slice(0, 280) || "Follow the report suggestions.",
    difficulty: "normal",
    frequency_percent: 0,
  };

  const scoresRaw = r.scores && typeof r.scores === "object" ? (r.scores as Record<string, number>) : null;
  /** Low-trust prov3 often has empty official strip; preview keyframes still map phases on the 240 timeline for overlay/HUD. */
  const stripForPhaseKeyframes = isLowTrust
    ? normalizedPreview.length > 0
      ? normalizedPreview
      : normalizedOfficial
    : normalizedOfficial.length > 0
      ? normalizedOfficial
      : normalizedPreview;
  const pk = mergePhaseKeyframeMaps(r.phase_keyframes, stripForPhaseKeyframes);

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
    keyframes: keyframesForPlusModel as PlusAnalysisResult["keyframes"],
    official_phase_keyframes: normalizedOfficial as PlusAnalysisResult["keyframes"],
    preview_keyframes: normalizedPreview as PlusAnalysisResult["keyframes"],
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
    screen_mode: Boolean(r.screen_mode),
    screen_fallback_raw: Boolean(r.screen_fallback_raw),
    screen_fallback_hint_zh:
      typeof r.screen_fallback_hint_zh === "string" ? r.screen_fallback_hint_zh : undefined,
    analysis_trust: typeof r.analysis_trust === "string" ? r.analysis_trust : undefined,
    report_mode: typeof r.report_mode === "string" ? r.report_mode : undefined,
    review_round: typeof r.review_round === "number" ? r.review_round : undefined,
    core_frame_scores:
      r.core_frame_scores && typeof r.core_frame_scores === "object" && !Array.isArray(r.core_frame_scores)
        ? (r.core_frame_scores as PlusAnalysisResult["core_frame_scores"])
        : undefined,
    keyframe_mismatch_notice: Boolean(r.keyframe_mismatch_notice),
    warning: typeof r.warning === "string" ? r.warning : undefined,
    final_status: typeof r.final_status === "string" ? r.final_status : undefined,
    low_trust_preview_only: Boolean(r.low_trust_preview_only),
    screen_cropped_video_url:
      typeof r.screen_cropped_video_url === "string" ? r.screen_cropped_video_url : undefined,
    screen_clean_video_url:
      typeof r.screen_clean_video_url === "string" ? r.screen_clean_video_url : undefined,
    playback_video_url: typeof r.playback_video_url === "string" ? r.playback_video_url : undefined,
    /** Must pass through for PlusResultView ``originalVideoDownloadUrl`` / video tab (was dropped → no URL fallback). */
    video_url: typeof r.video_url === "string" ? r.video_url : undefined,
    original_video_url: typeof r.original_video_url === "string" ? r.original_video_url : undefined,
    analysis_video_url: typeof r.analysis_video_url === "string" ? r.analysis_video_url : undefined,
    routing_execution:
      r.routing_execution && typeof r.routing_execution === "object" && !Array.isArray(r.routing_execution)
        ? (r.routing_execution as PlusAnalysisResult["routing_execution"])
        : undefined,
    screen_keyframe_audit:
      r.screen_keyframe_audit && typeof r.screen_keyframe_audit === "object" && !Array.isArray(r.screen_keyframe_audit)
        ? (r.screen_keyframe_audit as PlusAnalysisResult["screen_keyframe_audit"])
        : undefined,
    prov3_debug:
      r.prov3_debug && typeof r.prov3_debug === "object" && !Array.isArray(r.prov3_debug)
        ? (r.prov3_debug as PlusAnalysisResult["prov3_debug"])
        : undefined,
    /** Preserved for history / PlusResultView Prov3 interactive keyframe branch */
    pipeline: typeof r.pipeline === "string" ? r.pipeline : undefined,
    keyframes_strip:
      r.keyframes_strip && typeof r.keyframes_strip === "object" && !Array.isArray(r.keyframes_strip)
        ? (r.keyframes_strip as PlusAnalysisResult["keyframes_strip"])
        : undefined,
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function expandStellarProForUi(raw: Record<string, any>): Record<string, any> {
  ensureAnalysisIdOnRaw(raw);
  normalizeProResultKeyframeArraysForTrust(raw as Record<string, unknown>);
  normalizeProv3MediaInRaw(raw as Record<string, unknown>);
  const summary = String(raw.summary ?? "").trim();
  const summary_zh = String(raw.summary_zh ?? raw.summary ?? "").trim();
  const keyframes = Array.isArray(raw.keyframes) ? raw.keyframes : [];
  const official_phase_keyframes = Array.isArray(raw.official_phase_keyframes) ? raw.official_phase_keyframes : [];
  const preview_keyframes = Array.isArray(raw.preview_keyframes) ? raw.preview_keyframes : [];
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

  const lowTrustExpand = stellarProTrustIsLow(raw as Record<string, unknown>);

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
    keyframes: keyframes
      .filter((kf): kf is Record<string, unknown> => kf != null && typeof kf === "object" && !Array.isArray(kf))
      .map((kf) => {
        const phase = String(kf.phase ?? "unknown");
        return {
          ...kf,
          phase,
          label_en: String(kf.label_en ?? phase),
          label_zh: String(kf.label_zh ?? phase),
          frame_index: Number(kf.frame_index ?? 0),
          timestamp: Number(kf.timestamp ?? 0),
          image_base64: String(raw.pipeline ?? "") === "prov3" ? "" : String(kf.image_base64 ?? ""),
          keyframe_image_url:
            typeof kf.keyframe_image_url === "string" ? kf.keyframe_image_url : undefined,
          keyframe_image_source:
            typeof kf.keyframe_image_source === "string" ? kf.keyframe_image_source : "analysis_video",
        };
      }),
    official_phase_keyframes,
    preview_keyframes,
    skeleton_data: raw.skeleton_data ?? { frames: [], total_frames: 0 },
    prediction: raw.prediction ?? emptyPrediction,
    trajectory: raw.trajectory ?? [],
    pose_frames: Array.isArray(raw.pose_frames) ? raw.pose_frames : [],
    phase_keyframes: mergePhaseKeyframeMaps(
      raw.phase_keyframes,
      lowTrustExpand && preview_keyframes.length > 0
        ? preview_keyframes
        : official_phase_keyframes.length > 0
          ? official_phase_keyframes
          : preview_keyframes,
    ),
    video_meta: raw.video_meta ?? {},
    contact_sheet_url: raw.contact_sheet_url ?? null,
    video_url: raw.video_url ?? null,
    screen_mode: raw.screen_mode,
    screen_fallback_raw: raw.screen_fallback_raw,
    screen_fallback_hint_zh: raw.screen_fallback_hint_zh,
    analysis_trust: raw.analysis_trust,
    report_mode: raw.report_mode,
    review_round: raw.review_round,
    core_frame_scores: raw.core_frame_scores,
    retry_required: raw.retry_required,
    retry_reasons: raw.retry_reasons,
    keyframe_mismatch_notice: raw.keyframe_mismatch_notice,
    warning: raw.warning,
    final_status: raw.final_status,
    low_trust_preview_only: raw.low_trust_preview_only,
    screen_clean_video_url: raw.screen_clean_video_url,
    screen_keyframe_review_applied: raw.screen_keyframe_review_applied,
    routing_strategy: raw.routing_strategy,
    routing_execution: raw.routing_execution,
    screen_keyframe_audit: raw.screen_keyframe_audit,
    prov3_debug: raw.prov3_debug,
    pipeline: raw.pipeline,
    keyframes_strip: raw.keyframes_strip,
  };
}
