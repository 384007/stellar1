"use client";

import React, { useState, useRef, useEffect, useCallback, useMemo } from "react";
import ProComparison from "@/components/ProComparison";
import SimAnimation from "@/components/SimAnimation";
import Skeleton3DViewer from "@/components/Skeleton3DViewer";
import VideoAnalysisOverlay from "@/components/VideoAnalysisOverlay";
import { normalizePoseFramesForOverlay } from "@/lib/analysis-pose-storage";
import { coachingTipsFromParsed } from "@/lib/video-analysis-coaching";
import {
  drawPlusStyleSkeletonOverlay,
  letterboxPoseInContainer,
  plusSkeletonScale,
} from "@/lib/plus-skeleton-canvas-draw";
import { getAnalysisVideoBlob } from "@/lib/video-store";
import { isPlusScoreWithheld } from "@/lib/safe-analysis-score";
import { keyframeImageDataUrl } from "@/lib/image-base64";
import KeyframeProv3InteractiveViewer from "@/components/KeyframeProv3InteractiveViewer";
import {
  isProv3StrictMediaPolicyResult,
  useProv3KeyframeDisplayGate,
  PROV3_KEYFRAME_MEDIA_FAIL_ZH,
  PROV3_KEYFRAME_MEDIA_FAIL_EN,
  type Prov3KeyframeGateState,
} from "@/lib/prov3-keyframe-media";
import { resolveProv3ProductMediaUrl } from "@/lib/prov3-media-url";
import { downloadHrefAsFile } from "@/lib/download-href-as-file";

/* ═══════════════ Types ═══════════════ */

export interface PlusAnalysisResult {
  analysis_id: string;
  type: "plus" | "pro";
  posture_score: number | null;
  primary_diagnosis: { title_zh: string; title_en: string; status_zh: string; status_en: string; ai_confidence: number };
  additional_issues: Array<{ title_zh: string; title_en: string; status_zh: string; status_en: string }>;
  quick_tip_zh: string;
  quick_tip_en: string;
  problem_description_zh: string;
  problem_description_en: string;
  swing_phase_evaluations: Array<{ phase: string; status: string; note_zh: string; note_en: string }>;
  training: { title_zh: string; title_en: string; description_zh: string; description_en: string; difficulty: string; frequency_percent: number };
  recommended_videos: Array<{ title: string; creator: string; search_query: string }>;
  scores: Record<string, number> | null;
  total_score: number | null;
  gemini_observation?: {
    available?: boolean;
    mode?: "authoritative_phase_report" | "observation_only" | string;
    source?: string;
    phase_labels_trusted?: boolean;
    summary_zh?: string;
    summary_en?: string;
    bullets_zh?: string[];
    bullets_en?: string[];
    frame_notes?: Array<{
      index: number;
      label?: string | null;
      label_trusted?: boolean;
      note_zh?: string;
      note_en?: string;
    }>;
    issues?: string[];
    observed_phase_keyframes?: Record<string, number>;
    used_as_authoritative_source?: boolean;
  };
  report_status?: string | null;
  report_error_code?: string | null;
  final_ui_safe_score_state?: string | null;
  issues: string[];
  issues_zh: string[];
  suggestions: string[];
  suggestions_zh: string[];
  summary: string;
  summary_zh: string;
  keyframes: Array<{
    phase: string;
    label_en: string;
    label_zh: string;
    timestamp: number;
    image_base64: string;
    keyframe_image_url?: string;
    keyframe_image_source?: string;
    frame_index?: number;
    analysis_timestamp?: number;
    display_source_kind?: string;
    display_source_timestamp?: number;
    display_source_frame_index?: number;
    display_render_ok?: boolean;
    display_render_error?: string;
    width?: number;
    height?: number;
    pose_snapshot?: { joints: Array<{ name: string; nx: number; ny: number; v: number }>; connections: number[][] };
  }>;
  official_phase_keyframes?: PlusAnalysisResult["keyframes"];
  preview_keyframes?: PlusAnalysisResult["keyframes"];
  final_status?: string;
  low_trust_preview_only?: boolean;
  skeleton_data: { frames: Array<Record<string, unknown>>; total_frames: number };
  pose_frames?: Array<PoseFrame>;
  /** Maps swing phase id → pose_frames index (same moment as keyframe image) */
  phase_keyframes?: Record<string, number>;
  prediction?: Record<string, unknown>;
  video_meta?: {
    fps?: number;
    total_pose_frames?: number;
    duration_s?: number;
    /** OpenCV frame count — aligns video scrubber with pose frame_index */
    source_frame_count?: number;
  };
  /** Stellar Pro: multi-day plan from analyzer (optional). */
  training_plan?: Record<string, { focus: string; drills: string[]; duration: string }>;
  keyframes_degraded?: boolean;
  keyframe_display_mode?: "product_ready" | "degraded_debug_strip" | string;
  final_keyframe_gate_pass?: boolean;
  _plus_usage?: { used: number; remaining: number; limit: number | null; is_pro: boolean };
  /** Pro v3 product pipeline (`POST /pro-v3/analyze` — `routers.prov3_api`) */
  screen_mode?: boolean;
  /** 八关键帧条图：真 240 分析 MP4 上的解码帧号，与 SwingNet frame_index 一致；历史同步时一并保存 */
  keyframes_strip?: {
    timeline?: string;
    analysis_fps?: number;
    thumbnails_from_analysis_video?: boolean;
  };
  screen_fallback_raw?: boolean;
  screen_fallback_hint_zh?: string;
  analysis_trust?: "high_trust" | "low_trust" | string;
  report_mode?: "formal" | "limited" | string;
  review_round?: number;
  core_frame_scores?: Record<
    string,
    {
      score?: number | null;
      pass_90?: boolean | null;
      confidence?: number | null;
      reason_codes?: string[];
      comment_zh?: string;
      comment_en?: string;
    }
  >;
  keyframe_mismatch_notice?: boolean;
  warning?: string;
  screen_cropped_video_url?: string | null;
  screen_clean_video_url?: string | null;
  playback_video_url?: string | null;
  video_url?: string | null;
  original_video_url?: string | null;
  analysis_video_url?: string | null;
  /** Screen pipeline: how routing_strategy mapped to last backend pass (debug / transparency). */
  routing_execution?: Record<string, unknown> | null;
  /** ROI / dense motion / visual-dedupe gate outcome + reasons (Screen Mode). */
  screen_keyframe_audit?: {
    structural_gates_passed?: boolean;
    all_core_ai_pass_90?: boolean;
    roi_passed?: boolean;
    dense_motion_passed?: boolean;
    visual_gate_passed?: boolean;
    formal_report_allowed?: boolean;
    reason_codes?: string[];
    duplicate_pairs?: string[][];
    summary_zh?: string;
    summary_en?: string;
  };
  /** Backend debug bundle: paths, dense stats, keyframe lineup, visual gate (Screen Mode). */
  prov3_debug?: Record<string, unknown>;
  /** Pro v3 analyze pipeline — enables interactive keyframe viewer + local annotation sync */
  pipeline?: string;
}

export interface PoseFrame {
  joints: Array<{ name: string; x: number; y: number; z: number; visibility: number; normalized: { x: number; y: number } }>;
  connections: number[][];
  angles: Record<string, number>;
  frame_size: { width: number; height: number };
  frame_index: number;
  timestamp: number;
  image_base64?: string;
}

export type PlusResultTabKey = "diagnosis" | "fullswing" | "compare" | "video";

interface Props {
  result: PlusAnalysisResult;
  lang: "en" | "zh";
  externalVideoSrc?: string | null;
  backendUrl?: string;
  /** Overlay coaching copy: Pro uses issue/summary-style tips like the legacy Pro video tab. */
  coachingMode?: "plus" | "pro";
  /** e.g. Stellar Pro opens on video so VideoAnalysisOverlay (trajectory / yardage HUD) shows first */
  initialActiveTab?: PlusResultTabKey;
}
type TabKey = PlusResultTabKey;

/* ─── Posture Practice Video Types ─── */

interface PostureVideoCard {
  id: string;
  title_zh: string;
  title_en: string;
  focus_zh: string;
  focus_en: string;
  correction_zh: string;
  correction_en: string;
  status: "idle" | "generating" | "completed" | "failed";
  videoBlobUrl: string | null;
  error: string | null;
}

type PracticeStatus = "idle" | "generating" | "partial_ready" | "completed" | "failed";

/* ═══════════════ Constants ═══════════════ */

const PHASE_LABELS: Record<string, { en: string; zh: string }> = {
  address: { en: "Address", zh: "准备" }, takeaway: { en: "Takeaway", zh: "起杆" },
  backswing: { en: "Backswing", zh: "上杆" }, top: { en: "Top", zh: "顶点" },
  downswing: { en: "Downswing", zh: "下杆" }, impact: { en: "Impact", zh: "击球" },
  follow_through: { en: "Follow-Through", zh: "送杆" }, finish: { en: "Finish", zh: "收杆" },
};

const SWING_PHASES = ["address", "takeaway", "backswing", "top", "downswing", "impact", "follow_through", "finish"];

const PROV3_CORE_SCORE_LABELS: Record<string, { en: string; zh: string }> = {
  takeaway: { en: "Takeaway", zh: "起杆" },
  backswing_mid: { en: "Backswing (mid)", zh: "上杆（中段）" },
  top: { en: "Top", zh: "顶点" },
  early_downswing: { en: "Early downswing", zh: "下杆（早段）" },
  impact: { en: "Impact", zh: "触球" },
  release: { en: "Release", zh: "释放/送杆" },
};

/** Unify strip labels (8-frame UX) with core review phase IDs — same swing moments, different naming granularity. */
const PROV3_PHASE_NAMING = {
  stripHint: {
    zh: "上杆 / 下杆 / 送杆 与下方「核心关键帧评分」中的上杆（中段）、下杆（早段）、释放 为同一挥杆阶段，只是命名粗细不同。",
    en: "Backswing / Downswing / Follow-through here are the same moments as mid-backswing, early downswing, and release in core scores — friendlier strip names vs. review IDs.",
  },
  coreFootnote: {
    zh: "对应关系：主展示「上杆」≈ 审核「上杆（中段）」；「下杆」≈「下杆（早段）」；「送杆」≈「释放」。并非两套互不相关的关键帧。",
    en: "Mapping: strip Backswing ≈ review mid-backswing; strip Downswing ≈ early downswing; strip Follow-through ≈ release. These are not two independent sets of frames.",
  },
} as const;

/** Hover: tie audit phase keys to 8-frame strip names (tooltip). */
const PROV3_CORE_STRIP_TOOLTIP: Record<
  "takeaway" | "backswing_mid" | "top" | "early_downswing" | "impact" | "release",
  { en: string; zh: string }
> = {
  takeaway: { en: "8-frame strip: Takeaway", zh: "8 帧条：起杆" },
  backswing_mid: { en: "Same moment as strip “Backswing”", zh: "与 8 帧条「上杆」为同一时刻（审核称上杆中段）" },
  top: { en: "8-frame strip: Top", zh: "8 帧条：顶点" },
  early_downswing: { en: "Same moment as strip “Downswing”", zh: "与 8 帧条「下杆」为同一时刻（审核称下杆早段）" },
  impact: { en: "8-frame strip: Impact", zh: "8 帧条：击球" },
  release: { en: "Same moment as strip “Follow-through”", zh: "与 8 帧条「送杆」为同一时刻（审核称释放）" },
};

/**
 * Convert a keyframe's embedded pose_snapshot into a full PoseFrame.
 * This ensures the skeleton always matches the JPEG image, even when
 * the backend used an alternative frame due to decode failure.
 */
function _snapshotToPoseFrame(
  snap: { joints: Array<{ name: string; nx: number; ny: number; v: number }>; connections: number[][] },
  timestamp: number,
  imgWidth?: number,
  imgHeight?: number,
): PoseFrame | null {
  if (!snap.joints?.length) return null;
  const w = imgWidth || 320;
  const h = imgHeight || 568;
  return {
    joints: snap.joints.map((j) => ({
      name: j.name,
      x: j.nx * w,
      y: j.ny * h,
      z: 0,
      visibility: j.v,
      normalized: { x: j.nx, y: j.ny },
    })),
    connections: snap.connections || [],
    angles: {},
    frame_size: { width: w, height: h },
    frame_index: 0,
    timestamp: timestamp ?? 0,
  };
}

function keyframeForPhase(
  keyframes: PlusAnalysisResult["keyframes"] | undefined,
  phaseId: string,
) {
  const pid = phaseId.toLowerCase().replace(/[\s-]+/g, "_");
  return (
    keyframes?.find((k) => {
      const ph = String((k as { phase?: string }).phase ?? "")
        .toLowerCase()
        .replace(/[\s-]+/g, "_");
      return ph === pid;
    }) ?? null
  );
}

/** Pose / overlay alignment: official A/B strip only — never ``preview_keyframes``. */
function officialKeyframeStripForPose(result: PlusAnalysisResult): PlusAnalysisResult["keyframes"] | undefined {
  if (Array.isArray(result.official_phase_keyframes) && result.official_phase_keyframes.length > 0) {
    return result.official_phase_keyframes;
  }
  return result.keyframes;
}

function isLowTrustPreviewOnly(result: PlusAnalysisResult): boolean {
  if (result.low_trust_preview_only) return true;
  if (String(result.final_status || "") && String(result.final_status) !== "pass") return true;
  return String(result.analysis_trust || "") === "low_trust";
}

function plusKeyframeImageSrc(
  kf: { keyframe_image_url?: string; image_base64?: string } | null | undefined,
  urlOnly?: boolean,
): string | null {
  const u = resolveProv3ProductMediaUrl(String(kf?.keyframe_image_url ?? "").trim());
  if (u) return u;
  if (urlOnly) return null;
  return keyframeImageDataUrl(kf?.image_base64) ?? null;
}

function plusKeyframeImageUsable(
  kf: { keyframe_image_url?: string; image_base64?: string } | null | undefined,
  urlOnly?: boolean,
): boolean {
  return plusKeyframeImageSrc(kf, urlOnly) !== null;
}

function PlusKeyframePhoto({
  keyframe_image_url,
  image_base64,
  alt,
  className,
  placeholderClassName,
  lang = "zh",
  urlOnly = false,
}: {
  keyframe_image_url?: string;
  image_base64?: string;
  alt: string;
  className?: string;
  placeholderClassName?: string;
  lang?: "en" | "zh";
  /** Pro v3: never use base64 as visible source */
  urlOnly?: boolean;
}) {
  const [broken, setBroken] = useState(false);
  const dataUrl = plusKeyframeImageSrc({ keyframe_image_url, image_base64 }, urlOnly);
  const usable = dataUrl !== null;
  const ph = urlOnly
    ? lang === "en"
      ? "Timeline JPG missing"
      : "时间线关键帧图缺失"
    : lang === "en"
      ? "No image"
      : "无图";
  if (!usable || broken) {
    return (
      <div
        className={`flex items-center justify-center bg-white/10 text-center text-[9px] leading-tight text-white/40 px-1 ${
          placeholderClassName ?? className ?? ""
        }`}
      >
        {ph}
      </div>
    );
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      key={dataUrl}
      src={dataUrl}
      alt={alt}
      className={className}
      onError={() => setBroken(true)}
    />
  );
}

function poseIndexForPhase(result: PlusAnalysisResult, phaseId: string): number | null {
  const pk = result.phase_keyframes;
  if (!pk || typeof pk !== "object") return null;
  const pid = phaseId.toLowerCase().replace(/[\s-]+/g, "_");
  // Try exact key first, then normalized lookup
  if (typeof pk[phaseId] === "number") return pk[phaseId] as number;
  for (const [key, val] of Object.entries(pk)) {
    if (key.toLowerCase().replace(/[\s-]+/g, "_") === pid && typeof val === "number") {
      return val as number;
    }
  }
  return null;
}

function poseForPhase(
  result: PlusAnalysisResult,
  phaseId: string,
  phaseSliderIndex: number,
): PoseFrame | null {
  const poses = result.pose_frames;
  if (!poses?.length) return null;

  // 0. Prefer pose_snapshot embedded in the keyframe (guaranteed same moment as JPEG)
  const kf = keyframeForPhase(officialKeyframeStripForPose(result), phaseId);
  if (kf?.pose_snapshot) {
    const snap = kf.pose_snapshot;
    const syntheticPose = _snapshotToPoseFrame(snap, kf.timestamp, kf.width, kf.height);
    if (syntheticPose) return syntheticPose;
  }

  // 1. Use phase_keyframes mapping (exact backend match)
  const pi = poseIndexForPhase(result, phaseId);
  if (pi != null && poses[pi]) return poses[pi];

  // 2. Match by keyframe timestamp — find the pose closest to the keyframe image
  if (kf && typeof kf.timestamp === "number") {
    let bestIdx = 0;
    let bestDiff = Infinity;
    for (let i = 0; i < poses.length; i++) {
      const row = poses[i];
      if (!row || typeof row !== "object") continue;
      const d = Math.abs((row.timestamp ?? 0) - kf.timestamp);
      if (d < bestDiff) {
        bestDiff = d;
        bestIdx = i;
      }
    }
    return poses[bestIdx] ?? null;
  }

  // 3. Fallback: distribute evenly across pose frames
  const n = poses.length;
  const idx = Math.min(
    Math.floor((phaseSliderIndex / Math.max(SWING_PHASES.length - 1, 1)) * (n - 1)),
    n - 1,
  );
  return poses[idx] ?? null;
}

const STATUS_COLORS: Record<string, string> = {
  "完美": "text-green-400", "Perfect": "text-green-400",
  "做得好": "text-emerald-400", "Good": "text-emerald-400",
  "再接再厉": "text-amber-400", "Keep trying": "text-amber-400",
  "需要注意": "text-orange-400", "Needs attention": "text-orange-400",
  "需要改进": "text-red-400", "Needs improvement": "text-red-400",
};

const DIFFICULTY_MAP: Record<string, { en: string; zh: string; color: string }> = {
  easy: { en: "Easy", zh: "简单", color: "text-green-400" },
  normal: { en: "Normal", zh: "普通", color: "text-blue-400" },
  hard: { en: "Hard", zh: "困难", color: "text-red-400" },
};

/* Pro reference angles per phase for 动作校对 */
const PRO_ANGLES_BY_PHASE: Record<string, Record<string, number>> = {
  address:      { left_elbow: 168, right_elbow: 162, left_knee: 168, right_knee: 165, left_shoulder: 45, right_shoulder: 45 },
  takeaway:     { left_elbow: 165, right_elbow: 155, left_knee: 166, right_knee: 162, left_shoulder: 68, right_shoulder: 60 },
  backswing:    { left_elbow: 155, right_elbow: 130, left_knee: 160, right_knee: 155, left_shoulder: 95, right_shoulder: 72 },
  top:          { left_elbow: 130, right_elbow: 85,  left_knee: 153, right_knee: 148, left_shoulder: 115, right_shoulder: 85 },
  downswing:    { left_elbow: 148, right_elbow: 100, left_knee: 158, right_knee: 155, left_shoulder: 85,  right_shoulder: 60 },
  impact:       { left_elbow: 168, right_elbow: 155, left_knee: 168, right_knee: 162, left_shoulder: 45,  right_shoulder: 48 },
  follow_through:{ left_elbow: 155, right_elbow: 168, left_knee: 162, right_knee: 168, left_shoulder: 60, right_shoulder: 95 },
  finish:       { left_elbow: 125, right_elbow: 168, left_knee: 155, right_knee: 165, left_shoulder: 80,  right_shoulder: 115 },
};

/* Pro reference skeleton — side-view golfer matching real swing camera angle */
/* Right-handed golfer: trail shoulder lower, spine tilted forward, hands at grip */
const PRO_SKELETON_BY_PHASE: Record<string, Record<string, [number, number]>> = {
  address: {
    head:[.50,.15], lsh:[.40,.28], rsh:[.56,.30],
    lel:[.38,.42], rel:[.54,.42],
    lwri:[.46,.55], rwri:[.48,.56],
    lhip:[.42,.50], rhip:[.54,.51],
    lkn:[.38,.68], rkn:[.56,.68], lank:[.35,.86], rank:[.58,.86],
  },
  takeaway: {
    head:[.50,.15], lsh:[.42,.28], rsh:[.55,.29],
    lel:[.38,.40], rel:[.58,.38],
    lwri:[.44,.50], rwri:[.52,.48],
    lhip:[.42,.50], rhip:[.54,.51],
    lkn:[.38,.68], rkn:[.56,.68], lank:[.35,.86], rank:[.58,.86],
  },
  backswing: {
    head:[.50,.15], lsh:[.44,.27], rsh:[.54,.26],
    lel:[.36,.35], rel:[.60,.30],
    lwri:[.34,.28], rwri:[.50,.24],
    lhip:[.42,.50], rhip:[.54,.50],
    lkn:[.40,.68], rkn:[.56,.68], lank:[.36,.86], rank:[.58,.86],
  },
  top: {
    head:[.50,.15], lsh:[.46,.26], rsh:[.52,.24],
    lel:[.36,.28], rel:[.58,.20],
    lwri:[.32,.16], rwri:[.44,.12],
    lhip:[.42,.50], rhip:[.54,.49],
    lkn:[.40,.68], rkn:[.56,.68], lank:[.36,.86], rank:[.58,.86],
  },
  downswing: {
    head:[.50,.15], lsh:[.42,.28], rsh:[.55,.28],
    lel:[.37,.38], rel:[.56,.34],
    lwri:[.44,.48], rwri:[.50,.44],
    lhip:[.40,.50], rhip:[.55,.50],
    lkn:[.38,.68], rkn:[.56,.68], lank:[.35,.86], rank:[.58,.86],
  },
  impact: {
    head:[.50,.16], lsh:[.40,.28], rsh:[.56,.30],
    lel:[.38,.42], rel:[.54,.42],
    lwri:[.46,.55], rwri:[.48,.56],
    lhip:[.38,.50], rhip:[.54,.51],
    lkn:[.36,.68], rkn:[.56,.68], lank:[.34,.86], rank:[.58,.86],
  },
  follow_through: {
    head:[.50,.14], lsh:[.38,.27], rsh:[.55,.28],
    lel:[.34,.34], rel:[.52,.34],
    lwri:[.30,.24], rwri:[.40,.22],
    lhip:[.36,.50], rhip:[.52,.51],
    lkn:[.34,.68], rkn:[.56,.68], lank:[.34,.86], rank:[.56,.86],
  },
  finish: {
    head:[.50,.14], lsh:[.36,.26], rsh:[.54,.27],
    lel:[.34,.30], rel:[.50,.26],
    lwri:[.36,.18], rwri:[.44,.14],
    lhip:[.34,.50], rhip:[.50,.50],
    lkn:[.34,.68], rkn:[.54,.70], lank:[.34,.86], rank:[.55,.86],
  },
};

const PRO_SKELETON_CONNS: [string, string][] = [
  ["head","lsh"],["head","rsh"],["lsh","rsh"],["lsh","lel"],["rsh","rel"],
  ["lel","lwri"],["rel","rwri"],["lsh","lhip"],["rsh","rhip"],["lhip","rhip"],
  ["lhip","lkn"],["rhip","rkn"],["lkn","lank"],["rkn","rank"],
];

const PRO_CLUB_POS: Record<string, [number, number]> = {
  address: [0.47, 0.72], takeaway: [0.60, 0.56], backswing: [0.62, 0.12],
  top: [0.36, 0.02], downswing: [0.56, 0.40], impact: [0.47, 0.72],
  follow_through: [0.22, 0.14], finish: [0.30, 0.04],
};

/* ═══════════════ Canvas Skeleton ═══════════════ */

function drawAnglePill(
  ctx: CanvasRenderingContext2D,
  x: number, y: number,
  value: number, suffix: string,
  color: string, sc: number,
) {
  const text = `${Math.abs(Math.round(value))}${suffix}`;
  const fs = Math.max(10, 13 * sc);
  ctx.save();
  ctx.font = `700 ${fs}px ui-sans-serif,system-ui,sans-serif`;
  const tw = ctx.measureText(text).width;
  const pad = 5 * sc;
  const h = fs + pad * 2;
  const w = tw + pad * 2;
  const r = h / 2;
  ctx.globalAlpha = 0.75;
  ctx.fillStyle = "rgba(0,0,0,0.65)";
  ctx.beginPath();
  ctx.moveTo(x - w / 2 + r, y - h / 2);
  ctx.lineTo(x + w / 2 - r, y - h / 2);
  ctx.arc(x + w / 2 - r, y, r, -Math.PI / 2, Math.PI / 2);
  ctx.lineTo(x - w / 2 + r, y + h / 2);
  ctx.arc(x - w / 2 + r, y, r, Math.PI / 2, -Math.PI / 2);
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = 0.95;
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x, y);
  ctx.restore();
}

function drawSkeletonOnCanvas(
  ctx: CanvasRenderingContext2D,
  cW: number,
  cH: number,
  poseFrame: PoseFrame,
  showSkeleton: boolean,
  showGuideLines: boolean,
) {
  ctx.clearRect(0, 0, cW, cH);
  if (!poseFrame?.joints?.length) return;

  const { offsetX, offsetY, renderW, renderH } = letterboxPoseInContainer(
    poseFrame.frame_size?.width || cW,
    poseFrame.frame_size?.height || cH,
    cW,
    cH,
  );
  const px = (nx: number, ny: number): [number, number] => [
    offsetX + nx * renderW,
    offsetY + ny * renderH,
  ];
  const s = plusSkeletonScale(renderW, renderH);
  drawPlusStyleSkeletonOverlay(
    ctx,
    poseFrame,
    px,
    s,
    offsetY,
    renderH,
    showSkeleton,
    showGuideLines,
  );
}

/* ─── SkeletonCanvas component ─── */
function SkeletonCanvas({
  poseFrame, showSkeleton, showGuideLines,
}: { poseFrame: PoseFrame | null; showSkeleton: boolean; showGuideLines: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    // find the closest positioned ancestor (the aspect-video container)
    const parent = canvas.parentElement as HTMLElement;
    if (!parent) return;
    containerRef.current = parent as HTMLDivElement;
    const dpr = window.devicePixelRatio || 1;
    const cW = parent.offsetWidth;
    const cH = parent.offsetHeight;
    canvas.width = cW * dpr;
    canvas.height = cH * dpr;
    canvas.style.width = `${cW}px`;
    canvas.style.height = `${cH}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    if (poseFrame) {
      drawSkeletonOnCanvas(ctx, cW, cH, poseFrame, showSkeleton, showGuideLines);
    }
  }, [poseFrame, showSkeleton, showGuideLines]);

  return <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" style={{ zIndex: 10 }} />;
}

/* ─── Pro reference skeleton (3D animated robot) ─── */
function ProRefCanvas({ phase }: { phase: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stRef = useRef({ phase, progress: 1, prevSkel: null as Record<string, [number, number]> | null, prevClub: null as [number, number] | null });
  const rafRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement as HTMLElement;
    if (!parent) return;
    const st = stRef.current;
    if (st.phase !== phase) {
      st.prevSkel = { ...(PRO_SKELETON_BY_PHASE[st.phase] || PRO_SKELETON_BY_PHASE.address) };
      st.prevClub = PRO_CLUB_POS[st.phase] || PRO_CLUB_POS.address;
      st.phase = phase; st.progress = 0;
    }
    const tgtS = PRO_SKELETON_BY_PHASE[phase] || PRO_SKELETON_BY_PHASE.address;
    const tgtC = PRO_CLUB_POS[phase] || PRO_CLUB_POS.address;
    const srcS = st.prevSkel || tgtS;
    const srcC = st.prevClub || tgtC;
    const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
    const lerpPt = (a: [number, number], b: [number, number], t: number): [number, number] => [lerp(a[0], b[0], t), lerp(a[1], b[1], t)];
    const ease = (t: number) => t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;

    function render() {
      const cvs = canvasRef.current;
      if (!cvs) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const cW = parent.offsetWidth; const cH = parent.offsetHeight;
      if (!cW || !cH) return;
      cvs.width = cW * dpr; cvs.height = cH * dpr;
      cvs.style.width = `${cW}px`; cvs.style.height = `${cH}px`;
      const ctx = cvs.getContext("2d");
      if (!ctx) return;
      ctx.scale(dpr, dpr); ctx.clearRect(0, 0, cW, cH);

      const t = ease(Math.min(1, st.progress));
      const sk: Record<string, [number, number]> = {};
      for (const k of Object.keys(tgtS)) sk[k] = lerpPt(srcS[k] || tgtS[k], tgtS[k], t);
      const club = lerpPt(srcC, tgtC, t);
      const tp = (nx: number, ny: number): [number, number] => [nx * cW, ny * cH];
      const sc = Math.max(0.4, Math.min(1.8, Math.min(cW, cH) / 320));
      const jR = Math.max(3.5, 7 * sc);
      const limbW = Math.max(2.5, 5 * sc);

      // Ground shadow
      const ankY = Math.max(sk.lank?.[1] ?? 0.88, sk.rank?.[1] ?? 0.88);
      const hipMx = sk.lhip && sk.rhip ? (sk.lhip[0] + sk.rhip[0]) / 2 : 0.5;
      const [gx, gy] = tp(hipMx, ankY + 0.04);
      ctx.save(); ctx.globalAlpha = 0.18;
      const shG = ctx.createRadialGradient(gx, gy, 0, gx, gy, cW * 0.22);
      shG.addColorStop(0, "#f5c518"); shG.addColorStop(1, "transparent");
      ctx.fillStyle = shG; ctx.beginPath();
      ctx.ellipse(gx, gy, cW * 0.22, cH * 0.022, 0, 0, Math.PI * 2); ctx.fill(); ctx.restore();

      // Club shaft
      if (sk.lwri && sk.rwri) {
        const [hx, hy] = tp((sk.lwri[0] + sk.rwri[0]) / 2, (sk.lwri[1] + sk.rwri[1]) / 2);
        const [cx, cy] = tp(club[0], club[1]);
        ctx.save();
        ctx.globalAlpha = 0.08; ctx.strokeStyle = "#c0c0c0"; ctx.lineWidth = limbW * 2;
        ctx.lineCap = "round"; ctx.beginPath(); ctx.moveTo(hx, hy); ctx.lineTo(cx, cy); ctx.stroke();
        const sG = ctx.createLinearGradient(hx, hy, cx, cy);
        sG.addColorStop(0, "#b0b0b0"); sG.addColorStop(0.5, "#e8e8e8"); sG.addColorStop(1, "#808080");
        ctx.globalAlpha = 0.6; ctx.strokeStyle = sG; ctx.lineWidth = Math.max(1.5, 2.5 * sc);
        ctx.beginPath(); ctx.moveTo(hx, hy); ctx.lineTo(cx, cy); ctx.stroke();
        ctx.globalAlpha = 0.7; ctx.fillStyle = "#a8a8a8";
        const hr = Math.max(3, 5 * sc);
        ctx.beginPath(); ctx.arc(cx, cy, hr, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = "#d0d0d0"; ctx.lineWidth = 1; ctx.stroke();
        ctx.restore();
      }

      // Limbs (3D cylinder)
      for (const [a, b] of PRO_SKELETON_CONNS) {
        const pa = sk[a]; const pb = sk[b];
        if (!pa || !pb) continue;
        const [x1, y1] = tp(pa[0], pa[1]); const [x2, y2] = tp(pb[0], pb[1]);
        const ang = Math.atan2(y2 - y1, x2 - x1);
        const px2 = Math.sin(ang); const py2 = -Math.cos(ang);
        ctx.save();
        ctx.globalAlpha = 0.1; ctx.strokeStyle = "#f5c518"; ctx.lineWidth = limbW * 2.5;
        ctx.lineCap = "round"; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        const lG = ctx.createLinearGradient(
          (x1 + x2) / 2 + px2 * limbW, (y1 + y2) / 2 + py2 * limbW,
          (x1 + x2) / 2 - px2 * limbW, (y1 + y2) / 2 - py2 * limbW,
        );
        lG.addColorStop(0, "rgba(200,160,0,0.3)"); lG.addColorStop(0.35, "rgba(245,197,24,0.85)");
        lG.addColorStop(0.55, "rgba(255,220,80,0.95)"); lG.addColorStop(1, "rgba(200,160,0,0.4)");
        ctx.globalAlpha = 0.8; ctx.strokeStyle = lG; ctx.lineWidth = limbW;
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        ctx.globalAlpha = 0.25; ctx.strokeStyle = "#fff"; ctx.lineWidth = Math.max(0.5, limbW * 0.25);
        ctx.beginPath(); ctx.moveTo(x1 + px2 * limbW * 0.15, y1 + py2 * limbW * 0.15);
        ctx.lineTo(x2 + px2 * limbW * 0.15, y2 + py2 * limbW * 0.15); ctx.stroke();
        ctx.restore();
      }

      // Joints (3D spheres)
      for (const [key, pt] of Object.entries(sk)) {
        const [x, y] = tp(pt[0], pt[1]);
        const isH = key === "head";
        const r = isH ? jR * 1.8 : jR;
        ctx.save();
        const oG = ctx.createRadialGradient(x, y, 0, x, y, r * 3.5);
        oG.addColorStop(0, isH ? "rgba(245,197,24,0.4)" : "rgba(245,197,24,0.18)"); oG.addColorStop(1, "transparent");
        ctx.beginPath(); ctx.arc(x, y, r * 3.5, 0, Math.PI * 2); ctx.fillStyle = oG; ctx.fill();
        const sG = ctx.createRadialGradient(x - r * 0.3, y - r * 0.35, r * 0.05, x, y, r);
        sG.addColorStop(0, isH ? "#fff8dc" : "#ffe066");
        sG.addColorStop(0.5, isH ? "#ffd700" : "rgba(245,197,24,0.8)");
        sG.addColorStop(1, isH ? "#b8860b" : "rgba(180,140,0,0.5)");
        ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fillStyle = sG; ctx.fill();
        ctx.strokeStyle = "rgba(245,197,24,0.7)"; ctx.lineWidth = Math.max(0.5, 1.2 * sc); ctx.stroke();
        ctx.beginPath(); ctx.arc(x - r * 0.25, y - r * 0.3, r * 0.3, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255,255,255,0.55)"; ctx.fill();
        if (isH) {
          ctx.beginPath(); ctx.arc(x, y, r * 0.7, -0.3, Math.PI * 0.6);
          ctx.strokeStyle = "rgba(100,200,255,0.6)"; ctx.lineWidth = Math.max(1, 2 * sc); ctx.stroke();
        }
        ctx.restore();
      }

      // Motion trail (ghost during animation)
      if (t > 0.02 && t < 0.98) {
        ctx.save(); ctx.globalAlpha = 0.1;
        for (const k of ["lwri", "rwri"]) {
          const from = srcS[k]; const to = tgtS[k];
          if (!from || !to) continue;
          for (let i = 1; i <= 3; i++) {
            const tt = Math.max(0, t - i * 0.12);
            const [gx2, gy2] = tp(lerp(from[0], to[0], tt), lerp(from[1], to[1], tt));
            ctx.beginPath(); ctx.arc(gx2, gy2, jR * (0.7 - i * 0.15), 0, Math.PI * 2);
            ctx.fillStyle = `rgba(245,197,24,${0.3 - i * 0.08})`; ctx.fill();
          }
        }
        ctx.restore();
      }

      if (st.progress < 1) { st.progress = Math.min(1, st.progress + 0.04); rafRef.current = requestAnimationFrame(render); }
    }

    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    render();
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [phase]);

  return <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" />;
}

/* ═══════════════ Motion Correction Panel ═══════════════ */

const CORRECTION_UPPER = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"] as const;
const CORRECTION_LOWER = ["left_knee", "right_knee"] as const;
const CORRECTION_LABELS: Record<string, { zh: string; en: string }> = {
  left_elbow: { zh: "左肘", en: "L.Elbow" }, right_elbow: { zh: "右肘", en: "R.Elbow" },
  left_knee: { zh: "左膝", en: "L.Knee" }, right_knee: { zh: "右膝", en: "R.Knee" },
  left_shoulder: { zh: "左肩", en: "L.Shoulder" }, right_shoulder: { zh: "右肩", en: "R.Shoulder" },
  spine_tilt: { zh: "脊柱倾角", en: "Spine tilt" },
  x_factor: { zh: "X因子", en: "X-factor" },
  hip_rotation: { zh: "髋旋转", en: "Hip rot." },
};

/** Tour-style reference spine tilt (deg) by phase — rough coaching band */
const PRO_SPINE_BY_PHASE: Record<string, number> = {
  address: 32, takeaway: 30, backswing: 28, top: 26, downswing: 28, impact: 34, follow_through: 30, finish: 28,
};

function CorrectionCell({
  name, user, pro, lang,
}: { name: string; user: number; pro: number | null; lang: "en" | "zh" }) {
  const diff = pro !== null ? Math.abs(user - pro) : null;
  const status = diff === null ? "none" : diff < 12 ? "good" : diff < 28 ? "fair" : "poor";
  const colors = { good: "#22c55e", fair: "#eab308", poor: "#ef4444", none: "#666" };
  const c = colors[status as keyof typeof colors];
  const lab = CORRECTION_LABELS[name] || { zh: name, en: name };
  return (
    <div className="rounded-lg border bg-black/30 px-2.5 py-2" style={{ borderColor: `${c}44` }}>
      <p className="mb-0.5 text-[10px] text-white/40">{lang === "zh" ? lab.zh : lab.en}</p>
      <p className="text-sm font-bold" style={{ color: c }}>{user.toFixed(0)}°</p>
      {pro !== null && (
        <p className="text-[9px] text-white/25">
          Pro {pro}° {diff !== null && <span style={{ color: c }}>Δ{diff.toFixed(0)}°</span>}
        </p>
      )}
    </div>
  );
}

function MotionCorrectionPanel({ poseFrame, phase, lang }: { poseFrame: PoseFrame | null; phase: string; lang: "en" | "zh" }) {
  if (!poseFrame?.angles) return null;
  const proAngles = PRO_ANGLES_BY_PHASE[phase] || PRO_ANGLES_BY_PHASE.address;
  const refSpine = PRO_SPINE_BY_PHASE[phase] ?? 30;

  const upper = CORRECTION_UPPER.map((name) => {
    const user = poseFrame.angles?.[name];
    const pro = proAngles[name] ?? null;
    if (user == null) return null;
    return <CorrectionCell key={name} name={name} user={user} pro={pro} lang={lang} />;
  }).filter(Boolean);

  const lower = CORRECTION_LOWER.map((name) => {
    const user = poseFrame.angles?.[name];
    const pro = proAngles[name] ?? null;
    if (user == null) return null;
    return <CorrectionCell key={name} name={name} user={user} pro={pro} lang={lang} />;
  }).filter(Boolean);

  const spine = poseFrame.angles?.spine_tilt;
  const xf = poseFrame.angles?.x_factor;
  const hip = poseFrame.angles?.hip_rotation;

  if (!upper.length && !lower.length && spine == null && xf == null && hip == null) return null;

  return (
    <div className="glass-card space-y-4 p-4">
      <h4 className="flex items-center gap-2 text-xs font-bold text-white/60">
        <span className="h-2 w-2 rounded-full bg-brand-gold" />
        {lang === "zh" ? "动作校对（头颈 · 上身 · 下身）" : "Alignment (head/spine · upper · lower)"}
      </h4>

      {(spine != null || xf != null || hip != null) && (
        <div>
          <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-amber-400/80">
            {lang === "zh" ? "头部 / 躯干" : "Head / trunk"}
          </p>
          <div className="grid grid-cols-3 gap-2">
            {spine != null && (
              <CorrectionCell name="spine_tilt" user={spine} pro={refSpine} lang={lang} />
            )}
            {xf != null && (
              <div className="rounded-lg border border-white/10 bg-black/30 px-2.5 py-2">
                <p className="mb-0.5 text-[10px] text-white/40">{lang === "zh" ? CORRECTION_LABELS.x_factor.zh : CORRECTION_LABELS.x_factor.en}</p>
                <p className="text-sm font-bold text-cyan-400">{xf.toFixed(0)}°</p>
              </div>
            )}
            {hip != null && (
              <div className="rounded-lg border border-white/10 bg-black/30 px-2.5 py-2">
                <p className="mb-0.5 text-[10px] text-white/40">{lang === "zh" ? CORRECTION_LABELS.hip_rotation.zh : CORRECTION_LABELS.hip_rotation.en}</p>
                <p className="text-sm font-bold text-emerald-400">{hip.toFixed(0)}°</p>
              </div>
            )}
          </div>
        </div>
      )}

      {upper.length > 0 && (
        <div>
          <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-cyan-400/80">
            {lang === "zh" ? "上半身" : "Upper body"}
          </p>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">{upper}</div>
        </div>
      )}

      {lower.length > 0 && (
        <div>
          <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-green-400/80">
            {lang === "zh" ? "下半身" : "Lower body"}
          </p>
          <div className="grid grid-cols-2 gap-2">{lower}</div>
        </div>
      )}
    </div>
  );
}

/* ═══════════════ Save highlight ═══════════════ */

async function saveHighlight(
  keyframe: { keyframe_image_url?: string; image_base64?: string },
  label: string,
  options?: { urlOnly?: boolean },
) {
  const urlOnly = Boolean(options?.urlOnly);
  const remote = String(keyframe.keyframe_image_url ?? "").trim();
  if (remote) {
    try {
      const r = await fetch(remote);
      if (r.ok) {
        const blob = await r.blob();
        const obj = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = obj;
        link.download = `stellar-plus-${label.replace(/\s+/g, "-")}-${Date.now()}.jpg`;
        link.click();
        URL.revokeObjectURL(obj);
        return;
      }
    } catch {
      if (urlOnly) return;
    }
  }
  if (urlOnly) return;
  const dataUrl = keyframeImageDataUrl(keyframe.image_base64);
  if (!dataUrl) return;
  const link = document.createElement("a");
  link.href = dataUrl;
  const ext = dataUrl.startsWith("data:image/png") ? "png" : dataUrl.startsWith("data:image/webp") ? "webp" : "jpg";
  link.download = `stellar-plus-${label.replace(/\s+/g, "-")}-${Date.now()}.${ext}`;
  link.click();
}

/* ═══════════════ Toggle Buttons ═══════════════ */

function SkeletonToggles({
  showSkeleton,
  showGuideLines,
  onSkel,
  onGuide,
  lang,
  /** Prov3 左侧竖条：与关键帧工具同宽、淡化，仅图标 */
  rail,
}: {
  showSkeleton: boolean;
  showGuideLines: boolean;
  onSkel: () => void;
  onGuide: () => void;
  lang: "en" | "zh";
  rail?: boolean;
}) {
  const skelLabel = lang === "zh" ? "骨架" : "Skeleton";
  const guideLabel = lang === "zh" ? "辅助线" : "Guides";

  if (rail) {
    const railBtn = (active: boolean, tone: "purple" | "amber") =>
      `flex h-9 w-9 shrink-0 items-center justify-center rounded-md border backdrop-blur-md transition ${
        active
          ? tone === "purple"
            ? "border-purple-400/25 bg-purple-500/12 text-purple-200/90 shadow-[inset_0_0_0_1px_rgba(168,85,247,0.12)]"
            : "border-amber-400/25 bg-amber-500/12 text-amber-100/90 shadow-[inset_0_0_0_1px_rgba(245,158,11,0.12)]"
          : "border-white/[0.06] bg-black/25 text-white/38 hover:border-white/12 hover:bg-black/38 hover:text-white/60"
      }`;
    return (
      <div className="flex flex-col items-center gap-1.5">
        <button
          type="button"
          onClick={onSkel}
          className={railBtn(showSkeleton, "purple")}
          title={skelLabel}
          aria-label={skelLabel}
          aria-pressed={showSkeleton}
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0" />
          </svg>
        </button>
        <button
          type="button"
          onClick={onGuide}
          className={railBtn(showGuideLines, "amber")}
          title={guideLabel}
          aria-label={guideLabel}
          aria-pressed={showGuideLines}
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
          </svg>
        </button>
      </div>
    );
  }

  return (
    <div className="pointer-events-auto absolute left-3 top-3 z-20 flex flex-col gap-1.5">
      <button
        type="button"
        onClick={onSkel}
        className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-[10px] font-medium backdrop-blur-sm transition ${
          showSkeleton
            ? "border-purple-400/35 bg-purple-500/25 text-purple-200"
            : "border-white/10 bg-black/35 text-white/50 hover:bg-black/45"
        }`}
      >
        <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0" />
        </svg>
        {skelLabel}
      </button>
      <button
        type="button"
        onClick={onGuide}
        className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-[10px] font-medium backdrop-blur-sm transition ${
          showGuideLines
            ? "border-amber-400/35 bg-amber-500/25 text-amber-200"
            : "border-white/10 bg-black/35 text-white/50 hover:bg-black/45"
        }`}
      >
        <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
        </svg>
        {guideLabel}
      </button>
    </div>
  );
}

/* ═══════════════ Full Swing Tab ═══════════════ */

type FullSwingViewProps = Props & { prov3Strict: boolean; prov3KfGate: Prov3KeyframeGateState };

function FullSwingView({ result, lang, prov3Strict, prov3KfGate }: FullSwingViewProps) {
  const [activePhase, setActivePhase] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [showGuideLines, setShowGuideLines] = useState(true);
  const [showProRef, setShowProRef] = useState(false);
  const playRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (playing) {
      playRef.current = setInterval(() => {
        setActivePhase(p => { if (p >= SWING_PHASES.length - 1) { setPlaying(false); return 0; } return p + 1; });
      }, 900);
    }
    return () => { if (playRef.current) clearInterval(playRef.current); };
  }, [playing]);

  const lowTrustPreviewOnly = isLowTrustPreviewOnly(result);
  const officialKeyframes =
    Array.isArray(result.official_phase_keyframes) && result.official_phase_keyframes.length > 0
      ? result.official_phase_keyframes
      : [];
  const previewKeyframes = Array.isArray(result.preview_keyframes) ? result.preview_keyframes : [];
  const displayKeyframes = lowTrustPreviewOnly ? previewKeyframes : officialKeyframes;

  const phaseKey = SWING_PHASES[activePhase];
  const currentKf = keyframeForPhase(displayKeyframes, phaseKey);
  const currentPose = poseForPhase(result, phaseKey, activePhase);
  const phaseLabel = PHASE_LABELS[phaseKey] || { en: phaseKey, zh: phaseKey };
  const phaseEval = result.swing_phase_evaluations?.find(e => e.phase === phaseKey);

  if (prov3Strict) {
    if (prov3KfGate === "checking" || prov3KfGate === "idle") {
      return (
        <div className="flex flex-col items-center justify-center rounded-xl border border-white/10 bg-black/30 py-20">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-white/15 border-t-brand-purple/80" />
          <p className="mt-4 max-w-sm px-6 text-center text-xs text-white/50">
            {lang === "zh" ? "正在校验真 240 时间线关键帧图片…" : "Verifying true-240 timeline keyframe images…"}
          </p>
        </div>
      );
    }
    if (prov3KfGate === "fail") {
      return (
        <div className="rounded-xl border border-red-500/35 bg-red-500/10 p-4 text-sm leading-relaxed text-red-100/95">
          {lang === "zh" ? PROV3_KEYFRAME_MEDIA_FAIL_ZH : PROV3_KEYFRAME_MEDIA_FAIL_EN}
        </div>
      );
    }
  }

  return (
    <div className="space-y-4">
      {lowTrustPreviewOnly && (
        <div className="glass-card border border-amber-400/35 bg-amber-500/10 p-3 text-xs text-amber-200">
          {lang === "zh" ? "低信任，关键帧未通过验证（仅预览，不作为正式相位关键帧）" : "Low trust: keyframes not validated (preview only, not official phase keyframes)."}
        </div>
      )}
      {displayKeyframes.length === 0 && (
        <div className="glass-card border border-amber-400/30 bg-amber-500/10 p-3 text-xs text-amber-200">
          {lang === "zh" ? "本次分析低信任，暂无可用正式关键帧。" : "Low-trust analysis: no usable official keyframes."}
        </div>
      )}
      {/* Main viewer */}
      <div className="glass-card overflow-hidden">
        {/* Split / single toggle */}
        <div className="flex items-center justify-between px-3 pt-3 pb-1">
          <h4 className="text-xs font-bold text-white/60">
            {lang === "zh" ? phaseLabel.zh : phaseLabel.en}
            {phaseEval && (
              <span className={`ml-2 text-[10px] ${phaseEval.status === "error" ? "text-orange-400" : "text-green-400"}`}>
                {phaseEval.status === "error" ? (lang === "zh" ? "失误" : "Error") : (lang === "zh" ? "通过" : "Pass")}
              </span>
            )}
          </h4>
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input type="checkbox" checked={showProRef} onChange={e => setShowProRef(e.target.checked)} className="sr-only" />
            <div className={`h-4 w-8 rounded-full transition-colors ${showProRef ? "bg-brand-gold/70" : "bg-white/10"}`}>
              <div className={`h-4 w-4 rounded-full bg-white transition-transform ${showProRef ? "translate-x-4" : "translate-x-0"}`} />
            </div>
            <span className="text-[10px] text-brand-gold/70">{lang === "zh" ? "Pro对比" : "Pro Side"}</span>
          </label>
        </div>

        <div className="flex flex-col gap-1 bg-black">
          {/* User side */}
          <div className="relative bg-black w-full overflow-hidden" style={{ minHeight: showProRef ? "50vh" : "65vh" }}>
            {currentKf && (
              <PlusKeyframePhoto
                keyframe_image_url={currentKf.keyframe_image_url}
                image_base64={currentKf.image_base64}
                alt={phaseLabel.en}
                className="w-full h-full object-contain absolute inset-0"
                lang={lang}
                urlOnly={prov3Strict}
              />
            )}
            <SkeletonCanvas key={`skel-${showProRef}`} poseFrame={currentPose} showSkeleton={showSkeleton} showGuideLines={showGuideLines} />
            <SkeletonToggles showSkeleton={showSkeleton} showGuideLines={showGuideLines}
              onSkel={() => setShowSkeleton(s => !s)} onGuide={() => setShowGuideLines(g => !g)} lang={lang} />
            {currentKf && plusKeyframeImageUsable(currentKf, prov3Strict) && (
              <button onClick={() => void saveHighlight(currentKf, phaseLabel.en, { urlOnly: prov3Strict })}
                className="absolute top-3 right-3 rounded-lg bg-black/40 backdrop-blur-sm p-2 text-white/50 hover:text-white border border-white/10 transition"
                style={{ zIndex: 20 }}>
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" /></svg>
              </button>
            )}
            {showProRef && (
              <div className="absolute bottom-2 left-2 rounded-md bg-black/60 backdrop-blur-sm px-2.5 py-1 text-xs text-brand-purple font-medium">
                {lang === "zh" ? "你的挥杆" : "Your swing"}
              </div>
            )}
          </div>

          {/* Pro reference — stacked below */}
          {showProRef && (
            <div className="relative w-full overflow-hidden bg-[#0a0818]" style={{ minHeight: "50vh" }}>
              <div className="absolute inset-0 bg-gradient-to-br from-brand-purple/5 to-brand-gold/5" />
              <ProRefCanvas phase={phaseKey} />
              <div className="absolute bottom-2 left-2 rounded-md bg-black/60 backdrop-blur-sm px-2.5 py-1 text-xs text-brand-gold font-medium">
                {lang === "zh" ? "Pro 参考姿势" : "Pro reference"}
              </div>
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-center pointer-events-none">
                <p className="text-xs text-white/20">{lang === "zh" ? phaseLabel.zh : phaseLabel.en}</p>
              </div>
            </div>
          )}
        </div>

        {/* Play controls */}
        <div className="p-3">
          <div className="flex items-center gap-3 mb-3">
            <button onClick={() => setPlaying(!playing)}
              className="flex h-9 w-9 items-center justify-center rounded-full bg-brand-purple/20 border border-brand-purple/30 text-brand-purple hover:bg-brand-purple/30 transition">
              {playing ? (
                <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24"><rect x="6" y="4" width="4" height="16" /><rect x="14" y="4" width="4" height="16" /></svg>
              ) : (
                <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z" /></svg>
              )}
            </button>
            <div className="flex-1">
              <input type="range" min={0} max={SWING_PHASES.length - 1} value={activePhase}
                onChange={e => { setPlaying(false); setActivePhase(Number(e.target.value)); }}
                className="w-full h-1.5 rounded-full appearance-none bg-white/10 accent-brand-purple cursor-pointer" />
            </div>
            <span className="text-[10px] text-white/40 w-8 text-right">{activePhase + 1}/{SWING_PHASES.length}</span>
          </div>

          {/* Phase thumbnail strip */}
          <div className="flex gap-1.5 overflow-x-auto pb-1">
            {SWING_PHASES.map((phase, i) => {
              const kf = keyframeForPhase(displayKeyframes, phase);
              const isErr = result.swing_phase_evaluations?.find(e => e.phase === phase)?.status === "error";
              const pl = PHASE_LABELS[phase] || { en: phase, zh: phase };
              return (
                <button key={phase} onClick={() => { setPlaying(false); setActivePhase(i); }}
                  className={`flex-shrink-0 text-center transition ${activePhase === i ? "opacity-100 scale-105" : "opacity-50 hover:opacity-75"}`}
                  style={{ width: 64 }}>
                  <div className={`h-12 w-full rounded-lg overflow-hidden border-2 mb-0.5 ${activePhase === i ? "border-brand-purple" : isErr ? "border-orange-400/30" : "border-transparent"}`}>
                    {kf ? (
                      <PlusKeyframePhoto
                        keyframe_image_url={kf.keyframe_image_url}
                        image_base64={kf.image_base64}
                        alt={pl.en}
                        className="w-full h-full object-cover"
                        placeholderClassName="h-12 w-full"
                        lang={lang}
                        urlOnly={prov3Strict}
                      />
                    ) : (
                      <div className="w-full h-full bg-white/5" />
                    )}
                  </div>
                  <p className="text-[9px] text-white/50 truncate">{lang === "zh" ? pl.zh : pl.en}</p>
                  <div className={`mx-auto mt-0.5 h-1 w-1 rounded-full ${isErr ? "bg-orange-400" : "bg-green-400/50"}`} />
                </button>
              );
            })}
          </div>
          {result.screen_mode ? (
            <p className="text-[9px] text-white/38 leading-snug px-0.5 mt-1">
              {lang === "zh" ? PROV3_PHASE_NAMING.stripHint.zh : PROV3_PHASE_NAMING.stripHint.en}
            </p>
          ) : null}
        </div>
      </div>

      {lowTrustPreviewOnly && previewKeyframes.length > 0 && (
        <div className="glass-card p-3">
          <p className="mb-2 text-[11px] text-amber-200/90">
            {lang === "zh" ? "低信任预览图（不绑定正式 Address/Top/Impact/Finish）" : "Low-trust preview strip (not bound as official Address/Top/Impact/Finish)."}
          </p>
          <div className="grid grid-cols-4 gap-2">
            {previewKeyframes.map((kf, idx) => (
              <PlusKeyframePhoto
                key={`${idx}-${kf.label_en}`}
                keyframe_image_url={kf.keyframe_image_url}
                image_base64={kf.image_base64}
                urlOnly={prov3Strict}
                alt={kf.label_en}
                className="h-20 w-full rounded object-cover"
                placeholderClassName="h-20 w-full rounded"
                lang={lang}
              />
            ))}
          </div>
        </div>
      )}

      {/* Phase note */}
      {phaseEval && (lang === "zh" ? phaseEval.note_zh : phaseEval.note_en) && (
        <div className="glass-card p-3">
          <p className="text-xs text-white/50">{lang === "zh" ? phaseEval.note_zh : phaseEval.note_en}</p>
        </div>
      )}

      {/* Motion correction panel */}
      <MotionCorrectionPanel poseFrame={currentPose} phase={phaseKey} lang={lang} />

      {result.type === "pro" && result.prediction && (
        <SimAnimation
          prediction={
            result.prediction as {
              predicted_distance: number;
              lateral_offset: number;
              shot_shape: string;
              shot_shape_zh?: string;
              club_head_speed: number;
              ball_speed: number;
              launch_angle: number;
              spin_rate: number;
              smash_factor: number;
              trajectory?: Array<{ t: number; x: number; y: number; lateral: number }>;
            }
          }
          lang={lang}
          isPro
        />
      )}

      {result.type === "pro" && result.pose_frames && result.pose_frames.length > 0 && (
        <div className="space-y-4">
          <Skeleton3DViewer frames={result.pose_frames} lang={lang} />
          <div className="glass-card p-5">
            <h3 className="mb-4 text-sm font-semibold text-white">
              {lang === "zh" ? "3D 运动数据" : "3D Motion Data"}
            </h3>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {Object.entries(result.pose_frames[0]?.angles || {}).map(([key, val]) => {
                const labels: Record<string, { en: string; zh: string }> = {
                  left_elbow: { en: "L.Elbow", zh: "左肘" },
                  right_elbow: { en: "R.Elbow", zh: "右肘" },
                  left_knee: { en: "L.Knee", zh: "左膝" },
                  right_knee: { en: "R.Knee", zh: "右膝" },
                  left_shoulder: { en: "L.Shoulder", zh: "左肩" },
                  right_shoulder: { en: "R.Shoulder", zh: "右肩" },
                  shoulder_rotation: { en: "Shoulder Rot.", zh: "肩旋转" },
                  hip_rotation: { en: "Hip Rot.", zh: "髋旋转" },
                  x_factor: { en: "X-Factor", zh: "X因子" },
                  spine_tilt: { en: "Spine Tilt", zh: "脊柱倾斜" },
                };
                const l = labels[key] || { en: key, zh: key };
                return (
                  <div key={key} className="rounded-xl border border-white/5 bg-black/30 p-3 text-center">
                    <p className="text-[10px] text-white/40">{lang === "zh" ? l.zh : l.en}</p>
                    <p className="mt-1 text-lg font-bold text-brand-purple">
                      {typeof val === "number" ? val.toFixed(1) : val}°
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ═══════════════ Posture Practice Panel ═══════════════ */

function PosturePracticePanel({
  result, lang, backendUrl,
}: { result: PlusAnalysisResult; lang: "en" | "zh"; backendUrl: string }) {
  const [videos, setVideos] = useState<PostureVideoCard[]>(() => {
    const addressEval = result.swing_phase_evaluations?.find(e => e.phase === "address");
    const issuesZh = result.issues_zh || [];
    const issuesEn = result.issues || [];
    const sugZh = result.suggestions_zh || [];
    const sugEn = result.suggestions || [];

    return [
      {
        id: "stance",
        title_zh: "站姿与膝盖弯曲",
        title_en: "Stance & Knee Flex",
        focus_zh: "正确的站姿宽度、脚位、重心平衡和膝盖弯曲角度",
        focus_en: "Proper stance width, foot position, weight balance and knee flex angle",
        correction_zh: addressEval?.note_zh || issuesZh[0] || "站姿需要优化",
        correction_en: addressEval?.note_en || issuesEn[0] || "Stance needs optimization",
        status: "idle", videoBlobUrl: null, error: null,
      },
      {
        id: "spine",
        title_zh: "脊柱倾斜与肩髋对齐",
        title_en: "Spine Tilt & Shoulder-Hip Alignment",
        focus_zh: "正确的脊柱前倾角度和肩髋对齐方式",
        focus_en: "Correct spine forward tilt angle and shoulder-hip alignment",
        correction_zh: issuesZh[1] || sugZh[0] || "脊柱姿态需要优化",
        correction_en: issuesEn[1] || sugEn[0] || "Spine posture needs optimization",
        status: "idle", videoBlobUrl: null, error: null,
      },
      {
        id: "grip",
        title_zh: "握杆与手部设置",
        title_en: "Grip & Hand Setup",
        focus_zh: "正确的握杆压力、手指位置和手臂自然下垂",
        focus_en: "Proper grip pressure, finger placement and natural arm hang",
        correction_zh: issuesZh[2] || sugZh[1] || "握杆需要优化",
        correction_en: issuesEn[2] || sugEn[1] || "Grip needs optimization",
        status: "idle", videoBlobUrl: null, error: null,
      },
    ];
  });

  const blobUrlsRef = useRef<string[]>([]);

  useEffect(() => {
    const urls = blobUrlsRef.current;
    return () => { urls.forEach(u => URL.revokeObjectURL(u)); };
  }, []);

  const anyGenerating = videos.some(v => v.status === "generating");

  const practiceStatus: PracticeStatus = (() => {
    const gen = videos.some(v => v.status === "generating");
    const done = videos.some(v => v.status === "completed");
    const allDone = videos.every(v => v.status === "completed");
    const fail = videos.some(v => v.status === "failed");
    if (gen) return "generating";
    if (allDone) return "completed";
    if (done) return "partial_ready";
    if (fail && !done) return "failed";
    return "idle";
  })();

  async function handleGenerate(videoId: string) {
    setVideos(prev => prev.map(v =>
      v.id === videoId ? { ...v, status: "generating", error: null } : v
    ));

    try {
      const token = localStorage.getItem("stellar_token");
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 660_000);

      const res = await fetch(`${backendUrl}/analyze/plus/posture-video`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          video_id: videoId,
          analysis_data: {
            issues: result.issues,
            issues_zh: result.issues_zh,
            scores: result.scores,
            primary_diagnosis: result.primary_diagnosis,
            swing_phase_evaluations: result.swing_phase_evaluations,
            suggestions: result.suggestions,
            suggestions_zh: result.suggestions_zh,
          },
        }),
        signal: controller.signal,
      });

      clearTimeout(timer);

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(err.detail || `Video generation failed [${res.status}]`);
      }

      const data = await res.json();

      if (!data.video_base64) {
        throw new Error("Server returned no video data");
      }

      const byteChars = atob(data.video_base64);
      const byteArr = new Uint8Array(byteChars.length);
      for (let i = 0; i < byteChars.length; i++) byteArr[i] = byteChars.charCodeAt(i);
      const sniffVideoMime = (buf: Uint8Array): string | null => {
        if (buf.length >= 4 && buf[0] === 0x1a && buf[1] === 0x45 && buf[2] === 0xdf && buf[3] === 0xa3) return "video/webm";
        if (buf.length >= 12 && buf[4] === 0x66 && buf[5] === 0x74 && buf[6] === 0x79 && buf[7] === 0x70) return "video/mp4";
        return null;
      };
      const serverMime =
        typeof data.video_content_type === "string" && data.video_content_type.startsWith("video/")
          ? data.video_content_type
          : null;
      const mime = sniffVideoMime(byteArr) || serverMime || "video/mp4";
      const blob = new Blob([byteArr], { type: mime });
      const blobUrl = URL.createObjectURL(blob);
      blobUrlsRef.current.push(blobUrl);

      setVideos(prev => prev.map(v =>
        v.id === videoId ? { ...v, status: "completed", videoBlobUrl: blobUrl } : v
      ));
    } catch (err) {
      const isAbort = err instanceof DOMException && err.name === "AbortError";
      const msg = isAbort
        ? (lang === "zh" ? "视频生成超时，请重试" : "Video generation timed out")
        : (err instanceof Error ? err.message : (lang === "zh" ? "生成失败" : "Generation failed"));

      setVideos(prev => prev.map(v =>
        v.id === videoId ? { ...v, status: "failed", error: msg } : v
      ));
    }
  }

  const STATUS_BADGE: Record<PracticeStatus, { zh: string; en: string; color: string }> = {
    idle: { zh: "等待开始", en: "Ready", color: "text-white/40" },
    generating: { zh: "AI 生成中…", en: "Generating…", color: "text-brand-purple" },
    partial_ready: { zh: "部分就绪", en: "Partial", color: "text-amber-400" },
    completed: { zh: "全部完成", en: "All Done", color: "text-green-400" },
    failed: { zh: "生成失败", en: "Failed", color: "text-red-400" },
  };

  const badge = STATUS_BADGE[practiceStatus];

  return (
    <div className="space-y-3 animate-fade-in">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-bold text-white flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-brand-purple animate-pulse" />
          {lang === "zh" ? "AI 姿势教学视频" : "AI Posture Teaching Videos"}
        </h4>
        <span className={`text-[10px] font-medium ${badge.color}`}>
          {lang === "zh" ? badge.zh : badge.en}
        </span>
      </div>

      <p className="text-xs text-white/40">
        {lang === "zh"
          ? "基于您的分析结果，AI 将为每个纠正点生成 8 秒专属教学视频。点击下方卡片逐个生成。"
          : "Based on your analysis, AI generates 8-second teaching videos for each correction point. Click each card to generate."}
      </p>

      {videos.map((video) => (
        <div key={video.id} className="glass-card overflow-hidden">
          <div className="p-4">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <h5 className="text-sm font-bold text-white">
                  {lang === "zh" ? video.title_zh : video.title_en}
                </h5>
                <p className="text-[11px] text-brand-gold/70 mt-0.5">
                  {lang === "zh" ? video.focus_zh : video.focus_en}
                </p>
              </div>
              {video.status === "completed" && (
                <span className="flex-shrink-0 ml-2 h-5 w-5 rounded-full bg-green-400/20 flex items-center justify-center">
                  <svg className="h-3 w-3 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                  </svg>
                </span>
              )}
            </div>

            <div className="mt-2 rounded-lg bg-white/[0.03] border border-white/5 px-3 py-2">
              <p className="text-[10px] text-white/30 mb-0.5">
                {lang === "zh" ? "纠正点" : "Correction"}
              </p>
              <p className="text-xs text-white/60">
                {lang === "zh" ? video.correction_zh : video.correction_en}
              </p>
            </div>
          </div>

          {video.status === "idle" && (
            <div className="px-4 pb-4">
              <button
                onClick={() => handleGenerate(video.id)}
                disabled={anyGenerating}
                className={`w-full rounded-xl py-3 text-sm font-medium transition ${
                  anyGenerating
                    ? "bg-white/5 text-white/20 cursor-not-allowed border border-white/5"
                    : "bg-brand-purple/20 text-brand-purple border border-brand-purple/30 hover:bg-brand-purple/30"
                }`}
              >
                {anyGenerating
                  ? (lang === "zh" ? "请等待当前视频完成" : "Wait for current video")
                  : (lang === "zh" ? "生成教学视频" : "Generate Video")}
              </button>
            </div>
          )}

          {video.status === "generating" && (
            <div className="px-4 pb-4">
              <div className="h-40 flex items-center justify-center rounded-xl bg-black/30 border border-brand-purple/20">
                <div className="text-center">
                  <div className="relative mx-auto mb-3 h-10 w-10">
                    <div className="absolute inset-0 rounded-full border-2 border-brand-purple/20" />
                    <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-brand-purple animate-spin" />
                    <div className="absolute inset-2 rounded-full border-2 border-transparent border-b-brand-gold animate-spin" style={{ animationDirection: "reverse", animationDuration: "1.5s" }} />
                  </div>
                  <p className="text-xs text-white/60">{lang === "zh" ? "AI 正在生成视频…" : "AI generating video…"}</p>
                  <p className="text-[10px] text-white/25 mt-1">{lang === "zh" ? "预计需要 1-3 分钟" : "Estimated 1-3 minutes"}</p>
                </div>
              </div>
            </div>
          )}

          {video.status === "completed" && video.videoBlobUrl && (
            <div className="px-4 pb-4">
              <video
                controls
                playsInline
                preload="auto"
                className="w-full rounded-xl bg-black"
                src={video.videoBlobUrl}
              />
            </div>
          )}

          {video.status === "failed" && (
            <div className="px-4 pb-4">
              <div className="rounded-xl bg-red-500/10 border border-red-500/20 p-3">
                <p className="text-xs text-red-400 break-words">
                  {video.error || (lang === "zh" ? "视频生成失败" : "Video generation failed")}
                </p>
                <button
                  onClick={() => handleGenerate(video.id)}
                  disabled={anyGenerating}
                  className="mt-2 text-xs text-white/40 underline hover:text-white/60 transition disabled:opacity-30"
                >
                  {lang === "zh" ? "重试" : "Retry"}
                </button>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

/* ═══════════════ Main Component ═══════════════ */

function isProv3KeyframeViewer(result: PlusAnalysisResult): boolean {
  if (result.pipeline === "prov3") return true;
  if (String(result.analysis_id || "").startsWith("prov3_")) return true;
  if (result.prov3_debug && typeof result.prov3_debug === "object" && !Array.isArray(result.prov3_debug)) {
    return true;
  }
  return false;
}

export default function PlusResultView({ result, lang, externalVideoSrc, backendUrl, coachingMode, initialActiveTab }: Props) {
  const overlayCoachingMode = coachingMode ?? (result.type === "pro" ? "pro" : "plus");
  const prov3Strict = isProv3StrictMediaPolicyResult(result);
  const prov3KfGate = useProv3KeyframeDisplayGate(result);
  const [activeTab, setActiveTab] = useState<TabKey>(() => initialActiveTab ?? "diagnosis");
  const [activeKeyframe, setActiveKeyframe] = useState(0);
  const [showAllIssues, setShowAllIssues] = useState(false);
  const [showMoreDesc, setShowMoreDesc] = useState(false);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [showGuideLines, setShowGuideLines] = useState(true);
  const [videoSrc, setVideoSrc] = useState<string | null>(externalVideoSrc ?? null);
  const videoSrcLoaded = useRef(!!externalVideoSrc);
  const [videoIdbExhausted, setVideoIdbExhausted] = useState(false);
  const [originalVideoDownloadBusy, setOriginalVideoDownloadBusy] = useState(false);
  const lastVideoAnalysisIdRef = useRef<string>("");
  const [showPosturePractice, setShowPosturePractice] = useState(false);
  const lowTrustPreviewOnly = isLowTrustPreviewOnly(result);
  const officialKeyframes =
    Array.isArray(result.official_phase_keyframes) && result.official_phase_keyframes.length > 0
      ? result.official_phase_keyframes
      : [];
  const previewKeyframes = Array.isArray(result.preview_keyframes) ? result.preview_keyframes : [];
  const displayKeyframes = lowTrustPreviewOnly ? previewKeyframes : officialKeyframes;
  const safeActiveKeyframe =
    displayKeyframes.length > 0 ? Math.min(activeKeyframe, displayKeyframes.length - 1) : 0;

  const originalVideoDownloadUrl = useMemo(() => {
    const raw = String(
      result.original_video_url ||
        result.video_url ||
        result.playback_video_url ||
        result.analysis_video_url ||
        "",
    ).trim();
    const u = resolveProv3ProductMediaUrl(raw);
    return u || null;
  }, [
    result.original_video_url,
    result.video_url,
    result.playback_video_url,
    result.analysis_video_url,
  ]);

  const onDownloadOriginalVideo = useCallback(async () => {
    if (!originalVideoDownloadUrl || originalVideoDownloadBusy) return;
    setOriginalVideoDownloadBusy(true);
    try {
      await downloadHrefAsFile(
        originalVideoDownloadUrl,
        `stellar_${String(result.analysis_id || "analysis")}_video.mp4`,
        true,
      );
    } finally {
      setOriginalVideoDownloadBusy(false);
    }
  }, [originalVideoDownloadBusy, originalVideoDownloadUrl, result.analysis_id]);

  useEffect(() => {
    if (displayKeyframes.length === 0) {
      if (activeKeyframe !== 0) setActiveKeyframe(0);
      return;
    }
    if (activeKeyframe > displayKeyframes.length - 1) {
      setActiveKeyframe(displayKeyframes.length - 1);
    }
  }, [activeKeyframe, displayKeyframes.length]);

  useEffect(() => {
    const id = String(result.analysis_id ?? "");
    if (id === lastVideoAnalysisIdRef.current) return;
    lastVideoAnalysisIdRef.current = id;
    videoSrcLoaded.current = false;
    setVideoIdbExhausted(false);
    setVideoSrc((prev) => {
      const next = externalVideoSrc ?? null;
      if (prev && prev.startsWith("blob:") && prev !== next) {
        try {
          URL.revokeObjectURL(prev);
        } catch {
          /* ignore */
        }
      }
      return next;
    });
  }, [result.analysis_id, externalVideoSrc]);

  // Video tab: parent blob URL first; else IndexedDB with short backoff (saveAnalysisVideo may lag).
  useEffect(() => {
    if (activeTab !== "video") return;

    if (externalVideoSrc) {
      setVideoSrc((prev) => {
        if (prev && prev !== externalVideoSrc) {
          try { URL.revokeObjectURL(prev); } catch { /* */ }
        }
        return externalVideoSrc;
      });
      videoSrcLoaded.current = true;
      setVideoIdbExhausted(false);
      return;
    }

    if (videoSrcLoaded.current) return;
    const id = result.analysis_id;
    if (!id) {
      videoSrcLoaded.current = true;
      setVideoIdbExhausted(true);
      return;
    }

    let cancelled = false;
    const gapsMs = [0, 150, 400, 1000];

    void (async () => {
      for (let i = 0; i < gapsMs.length; i++) {
        if (i > 0) await new Promise((r) => setTimeout(r, gapsMs[i] - gapsMs[i - 1]));
        if (cancelled) return;
        const blob = await getAnalysisVideoBlob(id).catch(() => null);
        if (blob) {
          if (!cancelled) {
            setVideoSrc((prev) => {
              if (prev) {
                try { URL.revokeObjectURL(prev); } catch { /* */ }
              }
              return URL.createObjectURL(blob);
            });
            setVideoIdbExhausted(false);
          }
          videoSrcLoaded.current = true;
          return;
        }
      }
      if (!cancelled) {
        videoSrcLoaded.current = true;
        setVideoIdbExhausted(true);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [activeTab, result.analysis_id, externalVideoSrc]);

  useEffect(() => {
    return () => {
      if (videoSrc && videoSrc !== externalVideoSrc) URL.revokeObjectURL(videoSrc);
    };
  }, [videoSrc, externalVideoSrc]);

  const scoreWithheld = isPlusScoreWithheld(result);
  const diag = result.primary_diagnosis || { title_zh: "暂无诊断", title_en: "No diagnosis", status_zh: "—", status_en: "—", ai_confidence: 0 };
  const statusText = lang === "zh" ? diag.status_zh : diag.status_en;
  const statusColor = STATUS_COLORS[statusText] || "text-amber-400";
  const diffInfo = DIFFICULTY_MAP[result.training?.difficulty] || DIFFICULTY_MAP.normal;
  const postureScore = scoreWithheld ? null : (typeof result.posture_score === "number" ? result.posture_score : null);
  const totalScore = scoreWithheld ? null : (typeof result.total_score === "number" ? result.total_score : null);
  const postureNum = postureScore ?? 0;
  const scoreColor = postureNum >= 7 ? "text-green-400" : postureNum >= 5 ? "text-amber-400" : "text-red-400";
  const additionalIssues = Array.isArray(result.additional_issues) ? result.additional_issues : [];
  const swingPhaseEvals = Array.isArray(result.swing_phase_evaluations) ? result.swing_phase_evaluations : [];
  const issuesArr = Array.isArray(lang === "zh" ? result.issues_zh : result.issues) ? (lang === "zh" ? result.issues_zh : result.issues) : [];
  const suggestionsArr = Array.isArray(lang === "zh" ? result.suggestions_zh : result.suggestions) ? (lang === "zh" ? result.suggestions_zh : result.suggestions) : [];
  const scores = (!scoreWithheld && result.scores && typeof result.scores === "object") ? result.scores as Record<string, number> : {};
  const gemObs = result.gemini_observation || {};
  const gemObsBulletsRaw = (lang === "zh" ? gemObs.bullets_zh : gemObs.bullets_en) || [];
  const gemObsVisible =
    Boolean(gemObs.available) ||
    Boolean((lang === "zh" ? gemObs.summary_zh : gemObs.summary_en)?.trim()) ||
    (Array.isArray(gemObsBulletsRaw) && gemObsBulletsRaw.length > 0) ||
    (Array.isArray(gemObs.frame_notes) && gemObs.frame_notes.length > 0) ||
    Boolean(
      gemObs.mode === "observation_only" &&
        gemObs.observed_phase_keyframes &&
        Object.keys(gemObs.observed_phase_keyframes).length > 0,
    );
  const gemObsSummary = lang === "zh" ? (gemObs.summary_zh || "") : (gemObs.summary_en || "");
  const gemObsBullets = (lang === "zh" ? gemObs.bullets_zh : gemObs.bullets_en) || [];
  const gemObsNotes = Array.isArray(gemObs.frame_notes) ? gemObs.frame_notes : [];

  const getPoseForKf = useCallback((kfIdx: number): PoseFrame | null => {
    const kf = lowTrustPreviewOnly ? null : officialKeyframes?.[kfIdx];
    if (!kf) return null;

    // 0. Prefer pose_snapshot embedded in the keyframe (guaranteed same moment as JPEG)
    if (kf.pose_snapshot) {
      const syntheticPose = _snapshotToPoseFrame(kf.pose_snapshot, kf.timestamp, kf.width, kf.height);
      if (syntheticPose) return syntheticPose;
    }

    const poses = result.pose_frames;
    if (!poses?.length) return null;

    // 1. Use phase_keyframes for exact match
    const pi = poseIndexForPhase(result, kf.phase);
    if (pi != null && poses[pi]) return poses[pi];

    // 2. Match by timestamp for accurate skeleton overlay
    if (typeof kf.timestamp === "number") {
      let bestIdx = 0;
      let bestDiff = Infinity;
      for (let i = 0; i < poses.length; i++) {
        const d = Math.abs((poses[i].timestamp ?? 0) - kf.timestamp);
        if (d < bestDiff) { bestDiff = d; bestIdx = i; }
      }
      return poses[bestIdx];
    }

    return poses[Math.min(kfIdx, poses.length - 1)];
  }, [result]);

  return (
    <div className="space-y-4 animate-fade-in pb-8">
      {/* Coach Banner */}
      <div className="flex items-center justify-center gap-2 rounded-full bg-white/[0.04] border border-white/10 px-4 py-2 mx-auto max-w-xs">
        <div className="h-6 w-6 rounded-full bg-gradient-to-br from-brand-purple to-brand-gold flex items-center justify-center text-[10px] font-bold text-white">AI</div>
        <span className="text-xs text-white/60">
          {scoreWithheld
            ? (lang === "zh" ? "分析已完成（正式评分已暂缓）" : "Analysis complete (formal score withheld)")
            : (lang === "zh" ? "助理教练已完成诊断" : "Assistant coach completed diagnosis")}
        </span>
      </div>

      {result.type === "pro" && result.screen_mode ? (
        <div
          className={`rounded-xl border p-4 space-y-3 ${
            result.analysis_trust === "low_trust"
              ? "border-amber-500/35 bg-amber-500/[0.07]"
              : "border-brand-gold/25 bg-brand-gold/[0.06]"
          }`}
        >
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full border border-white/15 bg-black/30 px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-brand-gold">
              {lang === "zh" ? "屏幕模式" : "Screen mode"}
            </span>
            <span className="text-[10px] text-white/45">
              {lang === "zh" ? "翻拍/录屏链路" : "Screen / re-capture pipeline"}
            </span>
            {typeof result.review_round === "number" && result.review_round > 0 ? (
              <span className="text-[10px] text-white/35 font-mono">
                review_round={result.review_round}
              </span>
            ) : null}
          </div>
          <div className="flex flex-wrap gap-2 text-[11px]">
            <span
              className={`rounded-md px-2 py-1 font-semibold ${
                result.analysis_trust === "low_trust" ? "bg-amber-500/20 text-amber-200" : "bg-emerald-500/15 text-emerald-200"
              }`}
            >
              {result.analysis_trust === "low_trust"
                ? lang === "zh"
                  ? "低信任 low_trust"
                  : "Low trust"
                : lang === "zh"
                  ? "高信任 high_trust"
                  : "High trust"}
            </span>
            <span
              className={`rounded-md px-2 py-1 font-medium ${
                result.report_mode === "limited" ? "bg-white/10 text-amber-100/90" : "bg-white/10 text-white/80"
              }`}
            >
              {result.report_mode === "limited"
                ? lang === "zh"
                  ? "报告：受限 limited"
                  : "Report: limited"
                : lang === "zh"
                  ? "报告：正式 formal"
                  : "Report: formal"}
            </span>
          </div>
          {result.keyframe_mismatch_notice && (result.warning || "").trim() ? (
            <p className="text-sm font-semibold text-amber-200/95 leading-snug">{result.warning}</p>
          ) : null}
          {result.screen_keyframe_audit ? (
            <div
              className={`rounded-lg border p-3 space-y-2 ${
                result.screen_keyframe_audit.formal_report_allowed
                  ? "border-emerald-500/25 bg-emerald-500/[0.05]"
                  : "border-amber-500/30 bg-amber-500/[0.08]"
              }`}
            >
              <p className="text-[10px] uppercase tracking-wider text-white/45">
                {lang === "zh" ? "关键帧审核（结构 / 运动 / 视觉）" : "Keyframe audit (ROI / motion / visual)"}
              </p>
              <p className="text-xs text-white/75 leading-snug">
                {lang === "zh"
                  ? result.screen_keyframe_audit.summary_zh || "—"
                  : result.screen_keyframe_audit.summary_en || "—"}
              </p>
              <ul className="grid grid-cols-2 gap-x-2 gap-y-1 text-[10px] text-white/55">
                <li>
                  {lang === "zh" ? "Screen Mode" : "Screen mode"}:{" "}
                  {result.screen_mode
                    ? lang === "zh"
                      ? "是"
                      : "Yes"
                    : lang === "zh"
                      ? "否"
                      : "No"}
                </li>
                <li>
                  {lang === "zh" ? "结构 Gate" : "Structural gate"}:{" "}
                  {result.screen_keyframe_audit.structural_gates_passed
                    ? lang === "zh"
                      ? "通过"
                      : "OK"
                    : lang === "zh"
                      ? "未通过"
                      : "Fail"}
                </li>
                <li>
                  ROI:{" "}
                  {result.screen_keyframe_audit.roi_passed
                    ? lang === "zh"
                      ? "通过"
                      : "OK"
                    : lang === "zh"
                      ? "未通过"
                      : "Fail"}
                </li>
                <li>
                  {lang === "zh" ? "运动曲线" : "Dense motion"}:{" "}
                  {result.screen_keyframe_audit.dense_motion_passed
                    ? lang === "zh"
                      ? "通过"
                      : "OK"
                    : lang === "zh"
                      ? "未通过"
                      : "Fail"}
                </li>
                <li>
                  {lang === "zh" ? "视觉去重" : "Visual dedupe"}:{" "}
                  {result.screen_keyframe_audit.visual_gate_passed
                    ? lang === "zh"
                      ? "通过"
                      : "OK"
                    : lang === "zh"
                      ? "未通过"
                      : "Fail"}
                </li>
                <li>
                  {lang === "zh" ? "核心帧 AI≥90" : "Core AI ≥90"}:{" "}
                  {result.screen_keyframe_audit.all_core_ai_pass_90
                    ? lang === "zh"
                      ? "是"
                      : "Yes"
                    : lang === "zh"
                      ? "否"
                      : "No"}
                </li>
              </ul>
              {(result.screen_keyframe_audit.reason_codes?.length ?? 0) > 0 ? (
                <div className="space-y-1">
                  <p className="text-[9px] text-white/40 uppercase tracking-wider">
                    {lang === "zh" ? "原因代码" : "Reason codes"}
                  </p>
                  <ul className="font-mono text-[9px] text-amber-200/90 break-all space-y-0.5">
                    {(result.screen_keyframe_audit.reason_codes ?? []).slice(0, 14).map((c) => (
                      <li key={c}>{c}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
              {(result.screen_keyframe_audit.duplicate_pairs?.length ?? 0) > 0 ? (
                <div className="space-y-1">
                  <p className="text-[9px] text-white/40 uppercase tracking-wider">
                    {lang === "zh" ? "视觉过于相近的相邻阶段" : "Adjacent phases too similar"}
                  </p>
                  <ul className="font-mono text-[9px] text-white/60">
                    {(result.screen_keyframe_audit.duplicate_pairs ?? []).map((pair, i) => (
                      <li key={i}>
                        {Array.isArray(pair) ? pair.map(String).join(" ↔ ") : String(pair ?? "—")}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          ) : null}
          {result.screen_clean_video_url ? (
            <div className="space-y-1">
              <p className="text-[10px] uppercase tracking-wider text-white/40">
                {lang === "zh" ? "标准化 screen_clean.mp4 预览" : "Standardized screen_clean.mp4 preview"}
              </p>
              <video className="w-full max-h-48 rounded-lg border border-white/10 bg-black" controls playsInline preload="metadata" src={result.screen_clean_video_url} />
            </div>
          ) : null}
          {result.screen_cropped_video_url ? (
            <div className="space-y-1">
              <p className="text-[10px] uppercase tracking-wider text-white/40">
                {lang === "zh" ? "裁剪后屏幕区域预览" : "Cropped screen preview"}
              </p>
              <video
                className="w-full max-h-48 rounded-lg border border-white/10 bg-black"
                controls
                playsInline
                preload="metadata"
                src={result.screen_cropped_video_url}
              />
            </div>
          ) : null}
          {result.core_frame_scores && Object.keys(result.core_frame_scores).length > 0 ? (
            <div className="space-y-2">
              <p className="text-[10px] uppercase tracking-wider text-white/40">
                {lang === "zh" ? "核心关键帧审核（逐张评分 + 评论，≥90 为高信任门槛）" : "Core keyframe audit (per-frame score + comment; 90+ = trust gate)"}
              </p>
              <ul className="grid gap-1.5 sm:grid-cols-2 text-xs">
                {(
                  ["takeaway", "backswing_mid", "top", "early_downswing", "impact", "release"] as const
                ).map((key) => {
                  const row = result.core_frame_scores?.[key];
                  const lab = PROV3_CORE_SCORE_LABELS[key];
                  const tip = PROV3_CORE_STRIP_TOOLTIP[key];
                  const sc = typeof row?.score === "number" ? row.score : "—";
                  const ok = row?.pass_90 === true;
                  const cf =
                    typeof row?.confidence === "number" ? Math.round(row.confidence * 100) : null;
                  return (
                    <li
                      key={key}
                      className={`flex items-center justify-between rounded-lg border px-2 py-1.5 ${
                        ok ? "border-white/10 bg-white/[0.03]" : "border-amber-500/25 bg-amber-500/[0.06]"
                      }`}
                    >
                      <span
                        className="text-white/70 cursor-help underline decoration-dotted decoration-white/25 underline-offset-2"
                        title={lang === "zh" ? tip.zh : tip.en}
                      >
                        {lang === "zh" ? lab.zh : lab.en}
                      </span>
                      <span className="font-mono text-white/90 tabular-nums">
                        {sc}
                        {cf != null ? <span className="text-white/35 text-[10px] ml-1">({cf}%)</span> : null}
                        {!ok && sc !== "—" ? (
                          <span className="ml-1 text-amber-300/90 text-[10px]">&lt;90</span>
                        ) : null}
                      </span>
                    </li>
                  );
                })}
              </ul>
              <ul className="grid gap-1 text-[10px] text-white/55">
                {(["takeaway", "backswing_mid", "top", "early_downswing", "impact", "release"] as const).map((key) => {
                  const row = result.core_frame_scores?.[key];
                  const cmt = lang === "zh" ? row?.comment_zh : row?.comment_en;
                  const rs = row?.reason_codes?.slice(0, 3).join(", ");
                  return (
                    <li key={`cmt-${key}`} className="rounded border border-white/10 bg-white/[0.02] px-2 py-1">
                      <span className="text-white/40">{(lang === "zh" ? PROV3_CORE_SCORE_LABELS[key].zh : PROV3_CORE_SCORE_LABELS[key].en)}: </span>
                      <span>{cmt || "—"}</span>
                      {rs ? <span className="ml-1 font-mono text-amber-200/80">[{rs}]</span> : null}
                    </li>
                  );
                })}
              </ul>
              <p className="text-[9px] text-white/38 leading-snug pt-1">
                {lang === "zh" ? PROV3_PHASE_NAMING.coreFootnote.zh : PROV3_PHASE_NAMING.coreFootnote.en}
              </p>
            </div>
          ) : null}
          {Array.isArray(previewKeyframes) && previewKeyframes.length > 0 ? (
            <div className="space-y-1">
              <p className="text-[10px] uppercase tracking-wider text-white/40">
                {lang === "zh" ? "关键帧回源调试（展示帧来源）" : "Display keyframe source debug"}
              </p>
              <ul className="grid gap-1 text-[10px] text-white/60 font-mono">
                {previewKeyframes.map((k) => (
                  <li key={`dbg-${k.phase}`} className="rounded border border-white/10 bg-white/[0.02] px-2 py-1 break-all">
                    {`${k.phase}: source=${k.display_source_kind ?? "unknown"}, idx=${k.display_source_frame_index ?? -1}, ts=${typeof k.display_source_timestamp === "number" ? k.display_source_timestamp.toFixed(4) : "0.0000"}, ok=${k.display_render_ok === false ? "false" : "true"}${k.display_render_error ? `, err=${k.display_render_error}` : ""}`}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      ) : null}

      {/* Scores */}
      <div
        className={`glass-card p-5 ${
          result.type === "pro" && result.report_mode === "limited" && result.keyframe_mismatch_notice
            ? "ring-1 ring-amber-500/20"
            : ""
        }`}
      >
        {scoreWithheld ? (
          <div className="space-y-2 text-center py-2">
            <p className="text-sm font-medium text-amber-200/95">
              {lang === "zh" ? "暂不可评分" : "Score unavailable"}
            </p>
            <p className="text-xs text-white/55 leading-relaxed px-1">
              {lang === "zh"
                ? "关键帧未通过严格校验，暂无正式 AI 报告与数值评分。"
                : "Keyframes did not pass strict validation; no formal AI report or numeric score."}
            </p>
            {result.report_error_code ? (
              <p className="text-[10px] text-white/35 font-mono">{result.report_error_code}</p>
            ) : null}
          </div>
        ) : (
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs text-white/40 mb-1">{lang === "zh" ? "姿势分数" : "Posture Score"}</p>
              <div className="flex items-baseline gap-1">
                <span className={`text-4xl font-bold ${scoreColor}`}>{postureNum.toFixed(1)}</span>
                <span className="text-sm text-white/30">/10</span>
              </div>
            </div>
            <div className="text-right">
              <p className="text-xs text-white/40 mb-1">{lang === "zh" ? "综合得分" : "Overall"}</p>
              <div className="relative h-14 w-14">
                <svg className="-rotate-90" viewBox="0 0 56 56">
                  <circle cx="28" cy="28" r="24" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="4" />
                  <circle cx="28" cy="28" r="24" fill="none" stroke="#a855f7" strokeWidth="4"
                    strokeDasharray={`${(totalScore ?? 0) * 1.508} 150.8`} strokeLinecap="round" />
                </svg>
                <span className="absolute inset-0 flex items-center justify-center text-sm font-bold text-brand-purple">{totalScore ?? "—"}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 rounded-xl bg-white/5 p-1">
        {([
          { key: "diagnosis" as TabKey, zh: "姿势诊断", en: "Diagnosis" },
          { key: "video" as TabKey, zh: "视频分析", en: "Video" },
          { key: "fullswing" as TabKey, zh: "全挥杆", en: "Full Swing" },
          { key: "compare" as TabKey, zh: "Pro对比", en: "Pro Compare" },
        ]).map(tab => (
          <button key={tab.key} onClick={() => setActiveTab(tab.key)}
            className={`flex-1 rounded-lg py-2 text-center text-sm font-semibold transition-all ${activeTab === tab.key ? "bg-brand-purple/20 text-white border border-brand-purple/30" : "text-white/40 hover:text-white/60 border border-transparent"}`}>
            {tab.key === "compare" && <span className="inline-block h-2.5 w-2.5 rounded bg-gradient-to-br from-brand-purple to-brand-gold mr-1 align-middle" />}
            {lang === "zh" ? tab.zh : tab.en}
          </button>
        ))}
      </div>

      {/* ─── Video Analysis Tab ─── */}
      {activeTab === "video" && (
        <div className="relative space-y-3">
          {originalVideoDownloadUrl ? (
            <button
              type="button"
              title={lang === "zh" ? "下载原视频" : "Download original video"}
              aria-label={lang === "zh" ? "下载原视频" : "Download original video"}
              onClick={() => void onDownloadOriginalVideo()}
              disabled={originalVideoDownloadBusy}
              className="absolute right-1 top-0 z-30 rounded-lg border border-white/[0.08] bg-black/35 p-2 text-white/45 opacity-50 shadow-sm backdrop-blur-sm transition hover:border-white/15 hover:bg-black/50 hover:text-white/85 hover:opacity-95 active:scale-[0.97] disabled:opacity-30"
            >
              {originalVideoDownloadBusy ? (
                <span className="flex h-4 w-4 items-center justify-center text-[10px]" aria-hidden>
                  …
                </span>
              ) : (
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5} aria-hidden>
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3"
                  />
                </svg>
              )}
            </button>
          ) : null}
          {videoSrc ? (
            <VideoAnalysisOverlay
              videoSrc={videoSrc}
              poseFrames={normalizePoseFramesForOverlay(result.pose_frames)}
              lang={lang}
              coachingTips={coachingTipsFromParsed(result, overlayCoachingMode)}
              prediction={result.prediction as { predicted_distance?: number; shot_shape?: string; shot_shape_zh?: string; club_head_speed?: number; club_type?: string; hand?: "R" | "L" | "UNKNOWN" } | undefined}
              sourceFrameCount={result.video_meta?.source_frame_count}
              skeletonStyle="plus"
            />
          ) : videoIdbExhausted ? (
            <div className="glass-card p-6 text-center">
              <p className="text-xs text-white/50 leading-relaxed">
                {lang === "zh"
                  ? "本机未找到该分析的视频缓存。时间线关键帧若仍指向旧服务器的 /pro-v3/media/ 链接，也可能已失效。请重新分析或使用历史中的「重新分析」。"
                  : "No cached video for this analysis on this device. Timeline keyframe URLs under /pro-v3/media/ may also be expired. Re-analyze or use Re-analyze from history."}
              </p>
            </div>
          ) : (
            <div className="glass-card p-8 text-center">
              <div className="h-5 w-5 mx-auto mb-2 animate-spin rounded-full border-2 border-white/20 border-t-white/60" />
              <p className="text-xs text-white/40">
                {lang === "zh" ? "加载原视频..." : "Loading video..."}
              </p>
            </div>
          )}
        </div>
      )}

      {/* ─── Diagnosis Tab ─── */}
      {activeTab === "diagnosis" && (
        <>
          {gemObsVisible && (
            <div className="glass-card p-4 border border-cyan-400/25">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-semibold text-cyan-200">
                  {lang === "zh" ? "Gemini视觉观察（非正式报告）" : "Gemini Visual Observation (Non-formal)"}
                </p>
                <span className="text-[11px] px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-400/20">
                  {gemObs.mode === "authoritative_phase_report"
                    ? (lang === "zh" ? "参考相位标签" : "Phase labels referenced")
                    : (lang === "zh" ? "仅基于可见帧" : "Visible-frames only")}
                </span>
              </div>
              {!gemObs.phase_labels_trusted && (
                <p className="text-xs text-amber-200/90 mb-2">
                  {lang === "zh"
                    ? "当前关键帧/阶段标签不可靠，以下内容仅基于当前可见帧的视觉观察。"
                    : "Current keyframes/phase labels are unreliable; notes below are visual observations from currently visible frames only."}
                </p>
              )}
              {gemObsSummary ? <p className="text-sm text-white/85 leading-relaxed">{gemObsSummary}</p> : null}
              {gemObsBullets.length > 0 && (
                <ul className="mt-2 space-y-1">
                  {gemObsBullets.map((b, i) => <li key={i} className="text-xs text-white/70">• {b}</li>)}
                </ul>
              )}
              {gemObsNotes.length > 0 && (
                <div className="mt-3 grid gap-1">
                  {gemObsNotes.slice(0, 8).map((n, i) => (
                    <p key={i} className="text-[11px] text-white/55">
                      {lang === "zh"
                        ? `第${n.index ?? i + 1}张可见帧：${n.note_zh || ""}`
                        : `Visible frame ${n.index ?? i + 1}: ${n.note_en || ""}`}
                    </p>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Primary Diagnosis */}
          <div className="glass-card overflow-hidden">
            <div className="border-l-4 border-red-400/60 p-5">
              <div className="flex items-start justify-between mb-2">
                <p className="text-xs text-white/40">{lang === "zh" ? "主要诊断" : "Primary Diagnosis"}</p>
                <span className={`text-xs font-semibold ${statusColor}`}>{statusText}</span>
              </div>
              <h3 className="text-lg font-bold text-white mb-3">{lang === "zh" ? diag.title_zh : diag.title_en}</h3>
              <div className="flex items-center gap-2">
                <div className="h-4 w-4 rounded bg-gradient-to-br from-brand-purple to-brand-gold flex items-center justify-center text-[8px] font-bold text-white">AI</div>
                <div className="flex-1 h-1.5 rounded-full bg-white/10 overflow-hidden">
                  <div className="h-full rounded-full bg-gradient-to-r from-brand-purple to-brand-gold transition-all duration-1000" style={{ width: `${diag.ai_confidence}%` }} />
                </div>
                <span className="text-xs text-white/40">{diag.ai_confidence}/100</span>
              </div>
            </div>
          </div>

          {additionalIssues.length > 0 && (
            <button onClick={() => setShowAllIssues(!showAllIssues)} className="w-full text-left px-2 py-1">
              <span className="text-sm text-emerald-400 hover:text-emerald-300 transition">+{additionalIssues.length}{lang === "zh" ? "个问题被发现" : " more issue(s)"} ›</span>
            </button>
          )}
          {showAllIssues && additionalIssues.map((issue, i) => (
            <div key={i} className="glass-card p-4">
              <div className="flex items-start justify-between">
                <p className="text-sm font-semibold text-white">{lang === "zh" ? issue.title_zh : issue.title_en}</p>
                <span className={`text-xs font-medium ${STATUS_COLORS[lang === "zh" ? issue.status_zh : issue.status_en] || "text-amber-400"}`}>{lang === "zh" ? issue.status_zh : issue.status_en}</span>
              </div>
            </div>
          ))}

          {/* Keyframe viewer with canvas skeleton */}
          {lowTrustPreviewOnly && (
            <div className="glass-card border border-amber-400/35 bg-amber-500/10 p-3 text-xs text-amber-200">
              {lang === "zh" ? "低信任，关键帧未通过验证：以下仅为预览图，不作为正式相位关键帧。" : "Low trust: keyframes not validated. The strip below is preview only, not official phase keyframes."}
            </div>
          )}
          {prov3Strict && (prov3KfGate === "checking" || prov3KfGate === "idle") && (
            <div className="glass-card flex flex-col items-center justify-center border border-white/10 bg-black/25 py-14">
              <div className="h-7 w-7 animate-spin rounded-full border-2 border-white/15 border-t-brand-gold/80" />
              <p className="mt-3 max-w-xs px-4 text-center text-[11px] text-white/45">
                {lang === "zh" ? "正在校验真 240 时间线关键帧图片…" : "Verifying true-240 timeline keyframe images…"}
              </p>
            </div>
          )}
          {prov3Strict && prov3KfGate === "fail" && (
            <div className="glass-card border border-red-500/35 bg-red-500/10 p-4 text-sm leading-relaxed text-red-100/95">
              {lang === "zh" ? PROV3_KEYFRAME_MEDIA_FAIL_ZH : PROV3_KEYFRAME_MEDIA_FAIL_EN}
            </div>
          )}
          {displayKeyframes && displayKeyframes.length > 0 && (!prov3Strict || prov3KfGate === "ok") && (
            <div className="glass-card overflow-hidden">
              {(result.keyframes_degraded || result.keyframe_display_mode === "degraded_debug_strip" || result.final_keyframe_gate_pass === false) && (
                <div className="mx-3 mt-3 inline-flex items-center rounded-full border border-amber-400/50 bg-amber-500/15 px-3 py-1 text-[11px] font-medium text-amber-200">
                  {lang === "zh" ? "关键帧降级显示" : "Degraded keyframe display"}
                </div>
              )}
              {isProv3KeyframeViewer(result) ? (
                <KeyframeProv3InteractiveViewer
                  analysisId={result.analysis_id}
                  keyframes={displayKeyframes}
                  stripMeta={result.keyframes_strip}
                  activeIndex={activeKeyframe}
                  onActiveIndexChange={setActiveKeyframe}
                  lang={lang}
                  keyframeDownloadUrlOnly={prov3Strict}
                  overlay={
                    <div className="pointer-events-none absolute inset-0">
                      <SkeletonCanvas
                        poseFrame={getPoseForKf(activeKeyframe)}
                        showSkeleton={showSkeleton}
                        showGuideLines={showGuideLines}
                      />
                    </div>
                  }
                  skeletonRail={
                    <SkeletonToggles
                      rail
                      showSkeleton={showSkeleton}
                      showGuideLines={showGuideLines}
                      onSkel={() => setShowSkeleton((s) => !s)}
                      onGuide={() => setShowGuideLines((g) => !g)}
                      lang={lang}
                    />
                  }
                />
              ) : (
                <div className="relative isolate h-[70vh] min-h-[280px] w-full max-h-[85vh] bg-black">
                  <PlusKeyframePhoto
                    keyframe_image_url={displayKeyframes[safeActiveKeyframe]?.keyframe_image_url}
                    image_base64={displayKeyframes[safeActiveKeyframe]?.image_base64}
                    alt="Swing frame"
                    className="absolute inset-0 h-full w-full object-contain"
                    lang={lang}
                    urlOnly={prov3Strict}
                  />
                  <SkeletonCanvas
                    poseFrame={getPoseForKf(safeActiveKeyframe)}
                    showSkeleton={showSkeleton}
                    showGuideLines={showGuideLines}
                  />
                  <SkeletonToggles
                    showSkeleton={showSkeleton}
                    showGuideLines={showGuideLines}
                    onSkel={() => setShowSkeleton((s) => !s)}
                    onGuide={() => setShowGuideLines((g) => !g)}
                    lang={lang}
                  />
                  {displayKeyframes[safeActiveKeyframe] && plusKeyframeImageUsable(displayKeyframes[safeActiveKeyframe], prov3Strict) && (
                    <button
                      onClick={() =>
                        void saveHighlight(
                          displayKeyframes[safeActiveKeyframe],
                          displayKeyframes[safeActiveKeyframe].label_en,
                          { urlOnly: prov3Strict },
                        )
                      }
                      className="absolute right-3 top-3 rounded-lg border border-white/10 bg-black/40 p-2 text-white/50 backdrop-blur-sm transition hover:text-white"
                      style={{ zIndex: 20 }}
                    >
                      <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
                      </svg>
                    </button>
                  )}
                  <div className="absolute bottom-3 right-3" style={{ zIndex: 20 }}>
                    <span className="rounded-full bg-black/60 px-3 py-1 text-xs font-medium text-white/80 backdrop-blur-sm">
                      {displayKeyframes[safeActiveKeyframe]
                        ? lang === "zh"
                          ? displayKeyframes[safeActiveKeyframe].label_zh
                          : displayKeyframes[safeActiveKeyframe].label_en
                        : ""}
                    </span>
                  </div>
                </div>
              )}
              {displayKeyframes.length === 0 && (
                <div className="p-3 text-center text-xs text-amber-200">
                  {lang === "zh"
                    ? "本次分析低信任，暂无可用正式关键帧。"
                    : "Low-trust analysis: no usable official keyframes."}
                </div>
              )}
              <div className="flex gap-1.5 p-3 overflow-x-auto">
                {displayKeyframes.map((kf, i) => (
                  <button key={i} onClick={() => setActiveKeyframe(i)}
                    className={`flex-shrink-0 w-20 h-16 rounded-lg overflow-hidden border-2 transition ${activeKeyframe === i ? "border-brand-purple" : "border-transparent opacity-60"}`}>
                    <PlusKeyframePhoto
                      keyframe_image_url={kf.keyframe_image_url}
                      image_base64={kf.image_base64}
                      alt={kf.label_en}
                      className="w-full h-full object-cover"
                      placeholderClassName="w-full h-full"
                      lang={lang}
                      urlOnly={prov3Strict}
                    />
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* 3-Second Tip */}
          <div className="glass-card p-5">
            <h4 className="text-sm font-bold text-brand-gold mb-2">{lang === "zh" ? "3秒小贴士" : "3-Second Tip"}</h4>
            <p className="text-sm text-white/70 leading-relaxed">{lang === "zh" ? (result.quick_tip_zh || "") : (result.quick_tip_en || "")}</p>
          </div>

          {/* Problem Description */}
          <div className="glass-card p-5">
            <h4 className="text-sm font-bold text-white mb-2">{lang === "zh" ? "问题说明" : "Problem Description"}</h4>
            <p className="text-sm text-white/60 leading-relaxed">{lang === "zh" ? (result.problem_description_zh || "") : (result.problem_description_en || "")}</p>
            {!showMoreDesc && <button onClick={() => setShowMoreDesc(true)} className="mt-3 w-full rounded-xl border border-white/10 py-2 text-xs text-white/40 hover:text-white/60 transition">{lang === "zh" ? "查看更多" : "View More"}</button>}
            {showMoreDesc && <div className="mt-3 pt-3 border-t border-white/5 text-sm text-white/50 leading-relaxed">{lang === "zh" ? (result.summary_zh || "") : (result.summary || "")}</div>}
          </div>

          {/* Dimension Scores */}
          <div className="glass-card p-5">
            <h4 className="text-sm font-bold text-white mb-3">{lang === "zh" ? "各维度分数" : "Dimension Scores"}</h4>
            {scoreWithheld || Object.keys(scores).length === 0 ? (
              <p className="text-xs text-white/45 text-center py-2">
                {lang === "zh" ? "暂无正式各维度评分。" : "No formal dimension scores for this analysis."}
              </p>
            ) : (
              <div className="grid grid-cols-5 gap-3">
                {Object.entries(scores).map(([key, value]) => {
                  const labels: Record<string, { en: string; zh: string }> = { grip: { en: "Grip", zh: "握杆" }, stance: { en: "Stance", zh: "站姿" }, backswing: { en: "Back", zh: "后摆" }, downswing: { en: "Down", zh: "下杆" }, follow_through: { en: "Follow", zh: "收杆" } };
                  const label = labels[key] || { en: key, zh: key };
                  const color = value >= 80 ? "#a855f7" : value >= 60 ? "#f59e0b" : "#ef4444";
                  return (
                    <div key={key} className="text-center">
                      <div className="relative mx-auto mb-1.5 h-10 w-10">
                        <svg className="-rotate-90" viewBox="0 0 56 56">
                          <circle cx="28" cy="28" r="22" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="4" />
                          <circle cx="28" cy="28" r="22" fill="none" stroke={color} strokeWidth="4" strokeDasharray={`${value * 1.382} 138.2`} strokeLinecap="round" />
                        </svg>
                        <span className="absolute inset-0 flex items-center justify-center text-[11px] font-bold" style={{ color }}>{value}</span>
                      </div>
                      <p className="text-[10px] text-white/50">{lang === "en" ? label.en : label.zh}</p>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Swing Phase Timeline */}
          <div className="glass-card p-5">
            <h4 className="text-sm font-bold text-white mb-4">{lang === "zh" ? "挥杆流程评估" : "Swing Phase Evaluation"}</h4>
            <div className="flex gap-2 overflow-x-auto pb-2">
              {swingPhaseEvals.map((phase, i) => {
                const isError = phase.status === "error";
                const pl = PHASE_LABELS[phase.phase] || { en: phase.phase, zh: phase.phase };
                return (
                  <div key={i} className="flex-shrink-0 text-center" style={{ minWidth: 60 }}>
                    <div className={`mx-auto mb-2 flex h-14 w-14 items-center justify-center rounded-xl border ${isError ? "border-orange-400/40 bg-orange-400/10" : "border-green-400/30 bg-green-400/5"}`}>
                      <PhaseIcon phase={phase.phase} isError={isError} />
                    </div>
                    <p className="text-[10px] font-medium text-white/60 mb-0.5">{lang === "zh" ? pl.zh : pl.en}</p>
                    <span className={`text-[10px] font-bold ${isError ? "text-orange-400" : "text-green-400"}`}>{isError ? (lang === "zh" ? "失误" : "Error") : (lang === "zh" ? "通过" : "Pass")}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Training (Plus-only AI posture videos — Stellar Pro uses text plan in training block only) */}
          {result.type !== "pro" && result.training && Object.keys(result.training).length > 0 && (
            <div className="glass-card p-5">
              <h4 className="text-sm font-bold text-white mb-3">{lang === "zh" ? "训练建议" : "Training Recommendation"}</h4>
              <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
                <p className="text-sm font-semibold text-white mb-1">{lang === "zh" ? result.training.title_zh : result.training.title_en}</p>
                <p className="text-xs text-white/50 mb-3">{lang === "zh" ? result.training.description_zh : result.training.description_en}</p>
                <div className="flex items-center gap-4 text-xs text-white/40 mb-4">
                  <span>{lang === "zh" ? "修正难度: " : "Difficulty: "}<span className={`font-semibold ${diffInfo.color}`}>{lang === "zh" ? diffInfo.zh : diffInfo.en}</span></span>
                  {(result.training?.frequency_percent ?? 0) > 0 && <span>{lang === "zh" ? "发生频率: " : "Frequency: "}<span className="font-semibold text-blue-400">{(result.training?.frequency_percent ?? 0).toFixed(1)}%</span></span>}
                </div>
                <button
                  onClick={() => setShowPosturePractice(p => !p)}
                  className={`w-full rounded-xl py-3 text-sm font-bold text-white transition ${
                    showPosturePractice
                      ? "bg-white/10 border border-white/20 hover:bg-white/15"
                      : "bg-brand-purple hover:bg-brand-purple/80"
                  }`}
                >
                  {showPosturePractice
                    ? (lang === "zh" ? "收起姿势练习" : "Hide Posture Practice")
                    : (lang === "zh" ? "开始姿势练习" : "Start Posture Practice")}
                </button>
              </div>
            </div>
          )}

          {result.type !== "pro" && showPosturePractice && (
            <PosturePracticePanel
              result={result}
              lang={lang}
              backendUrl={backendUrl || process.env.NEXT_PUBLIC_BACKEND_URL || "https://stellar1-backend.onrender.com"}
            />
          )}

          {result.type === "pro" && result.training && (
            <div className="glass-card p-5">
              <h4 className="text-sm font-bold text-white mb-3">{lang === "zh" ? "训练建议" : "Training Recommendation"}</h4>
              <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
                <p className="text-sm font-semibold text-white mb-1">{lang === "zh" ? result.training.title_zh : result.training.title_en}</p>
                <p className="text-xs text-white/50 leading-relaxed">{lang === "zh" ? result.training.description_zh : result.training.description_en}</p>
                <div className="mt-3 flex flex-wrap items-center gap-4 text-xs text-white/40">
                  <span>{lang === "zh" ? "修正难度: " : "Difficulty: "}<span className={`font-semibold ${diffInfo.color}`}>{lang === "zh" ? diffInfo.zh : diffInfo.en}</span></span>
                </div>
              </div>
            </div>
          )}

          {/* Issues & Suggestions */}
          <div className="grid gap-4 md:grid-cols-2">
            <div className="glass-card p-5">
              <h4 className="mb-3 text-sm font-bold text-red-400/80">{lang === "zh" ? "发现的问题" : "Issues Found"}</h4>
              <ul className="space-y-2">{issuesArr.map((issue, i) => <li key={i} className="flex items-start gap-2 text-xs text-white/60"><span className="mt-0.5 text-red-400/60">●</span>{issue}</li>)}</ul>
            </div>
            <div className="glass-card p-5">
              <h4 className="mb-3 text-sm font-bold text-brand-gold/80">{lang === "zh" ? "改进建议" : "Suggestions"}</h4>
              <ul className="space-y-2">{suggestionsArr.map((sug, i) => <li key={i} className="flex items-start gap-2 text-xs text-white/60"><span className="mt-0.5 text-brand-gold/60">◆</span>{sug}</li>)}</ul>
            </div>
          </div>

          {result.type === "pro" && result.training_plan && Object.keys(result.training_plan).length > 0 && (
            <div className="glass-card p-6">
              <h3 className="mb-6 text-xl font-bold text-brand-gold">
                {lang === "en" ? "7-Day Training Plan" : "7天训练计划"}
              </h3>
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {Object.entries(result.training_plan).map(([day, plan]) => {
                  const dayNum = day.replace("day", "");
                  return (
                    <div
                      key={day}
                      className="rounded-xl border border-brand-gold/10 bg-black/30 p-4 transition hover:border-brand-gold/30"
                    >
                      <div className="mb-2 flex items-center justify-between">
                        <span className="rounded-full bg-brand-gold/20 px-3 py-0.5 text-xs font-bold text-brand-gold">
                          Day {dayNum}
                        </span>
                        <span className="text-xs text-white/40">{plan.duration}</span>
                      </div>
                      <h4 className="mb-2 text-sm font-semibold text-white">{plan.focus}</h4>
                      <ul className="space-y-1">
                        {plan.drills.map((drill, i) => (
                          <li key={i} className="flex items-start gap-1 text-xs text-white/60">
                            <span className="text-brand-gold">•</span>
                            {drill}
                          </li>
                        ))}
                      </ul>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Recommended Videos */}
          {result.recommended_videos && result.recommended_videos.length > 0 && (
            <div className="glass-card p-5">
              <h4 className="text-sm font-bold text-white mb-3">{lang === "zh" ? "练习推荐视频" : "Recommended Videos"}</h4>
              <div className="flex gap-3 overflow-x-auto pb-2">
                {result.recommended_videos.map((vid, i) => (
                  <a key={i} href={`https://www.youtube.com/results?search_query=${encodeURIComponent(vid.search_query)}`} target="_blank" rel="noopener noreferrer"
                    className="flex-shrink-0 w-52 rounded-xl border border-white/5 bg-white/[0.02] p-3 hover:border-brand-purple/30 transition group">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="h-6 w-6 rounded-full bg-red-500/20 flex items-center justify-center"><svg className="h-3 w-3 text-red-400" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z" /></svg></div>
                      <span className="text-[10px] text-white/40 truncate">{vid.creator}</span>
                    </div>
                    <p className="text-xs text-white/70 line-clamp-2 group-hover:text-white transition">{vid.title}</p>
                  </a>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {/* ─── Full Swing Tab ─── */}
      {activeTab === "fullswing" && (
        <FullSwingView result={result} lang={lang} prov3Strict={prov3Strict} prov3KfGate={prov3KfGate} />
      )}

      {/* ─── Pro Compare Tab ─── */}
      {activeTab === "compare" && (
        <ProComparison
          userScores={scores}
          userAngles={result.pose_frames?.[(result.pose_frames?.length || 1) >> 1]?.angles ?? { shoulder_rotation: -35, hip_rotation: -22, x_factor: 42, spine_tilt: 8 }}
          lang={lang}
        />
      )}

      {/* Usage Info */}
      {result._plus_usage && !result._plus_usage.is_pro && result._plus_usage.limit !== null && (
        <div className="text-center text-xs text-white/30 py-2">
          {lang === "zh" ? `今日已用 ${result._plus_usage.used}/${result._plus_usage.limit} 次 Plus 分析` : `Used ${result._plus_usage.used}/${result._plus_usage.limit} Plus analyses today`}
          {result._plus_usage.remaining === 0 && <span className="block mt-1 text-amber-400/70">{lang === "zh" ? "今日次数已用完，明天重置" : "Daily limit reached, resets tomorrow"}</span>}
        </div>
      )}
    </div>
  );
}

/* ═══════════════ Phase Icons ═══════════════ */
function PhaseIcon({ phase, isError }: { phase: string; isError: boolean }) {
  const c = isError ? "#fb923c" : "#4ade80";
  const p = { width: 28, height: 28, viewBox: "0 0 24 24", fill: "none", stroke: c, strokeWidth: 1.5 };
  const icons: Record<string, React.ReactElement> = {
    address: <svg {...p}><circle cx="12" cy="5" r="2" fill={c} stroke="none"/><line x1="12" y1="7" x2="12" y2="16"/><line x1="8" y1="11" x2="16" y2="11"/><line x1="10" y1="16" x2="12" y2="21"/><line x1="14" y1="16" x2="12" y2="21"/></svg>,
    takeaway: <svg {...p}><circle cx="12" cy="5" r="2" fill={c} stroke="none"/><line x1="12" y1="7" x2="11" y2="15"/><line x1="8" y1="10" x2="16" y2="12"/><line x1="9" y1="15" x2="11" y2="21"/><line x1="14" y1="15" x2="11" y2="21"/></svg>,
    backswing: <svg {...p}><circle cx="13" cy="5" r="2" fill={c} stroke="none"/><line x1="13" y1="7" x2="12" y2="15"/><line x1="7" y1="9" x2="17" y2="11"/><line x1="10" y1="15" x2="12" y2="21"/><line x1="14" y1="15" x2="12" y2="21"/></svg>,
    top: <svg {...p}><circle cx="14" cy="5" r="2" fill={c} stroke="none"/><line x1="14" y1="7" x2="12" y2="15"/><line x1="6" y1="7" x2="18" y2="10"/><line x1="10" y1="15" x2="12" y2="21"/><line x1="14" y1="15" x2="12" y2="21"/></svg>,
    downswing: <svg {...p}><circle cx="12" cy="5" r="2" fill={c} stroke="none"/><line x1="12" y1="7" x2="11" y2="15"/><line x1="7" y1="12" x2="15" y2="8"/><line x1="9" y1="15" x2="11" y2="21"/><line x1="13" y1="15" x2="11" y2="21"/></svg>,
    impact: <svg {...p}><circle cx="11" cy="5" r="2" fill={c} stroke="none"/><line x1="11" y1="7" x2="11" y2="15"/><line x1="6" y1="13" x2="16" y2="9"/><line x1="9" y1="15" x2="11" y2="21"/><line x1="13" y1="15" x2="11" y2="21"/></svg>,
    follow_through: <svg {...p}><circle cx="10" cy="5" r="2" fill={c} stroke="none"/><line x1="10" y1="7" x2="11" y2="15"/><line x1="5" y1="12" x2="17" y2="7"/><line x1="9" y1="15" x2="11" y2="21"/><line x1="13" y1="15" x2="11" y2="21"/></svg>,
    finish: <svg {...p}><circle cx="10" cy="5" r="2" fill={c} stroke="none"/><line x1="10" y1="7" x2="12" y2="15"/><line x1="5" y1="10" x2="18" y2="5"/><line x1="10" y1="15" x2="12" y2="21"/><line x1="14" y1="15" x2="12" y2="21"/></svg>,
  };
  return icons[phase] || icons.address;
}
