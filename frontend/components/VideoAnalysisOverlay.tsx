"use client";

import { useRef, useEffect, useState, useCallback, useMemo } from "react";
import type { VideoCoachingTips } from "@/lib/video-analysis-coaching";
import {
  drawPlusStyleSkeletonOverlay,
  plusSkeletonScale,
} from "@/lib/plus-skeleton-canvas-draw";

/* ═══════════════ Types ═══════════════ */

export interface OverlayPoseFrame {
  joints: Array<{
    name: string;
    x: number;
    y: number;
    z: number;
    visibility: number;
    normalized: { x: number; y: number };
  }>;
  connections: number[][];
  angles: Record<string, number>;
  frame_size: { width: number; height: number };
  frame_index: number;
  timestamp: number;
  phase_data?: {
    phase_id: string;
    phase_en: string;
    phase_zh: string;
    progress_pct: number;
  } | null;
}

export interface ShotPrediction {
  predicted_distance?: number;
  shot_shape?: string;
  shot_shape_zh?: string;
  club_head_speed?: number;
  club_type?: string;
  hand?: "R" | "L" | "UNKNOWN";
}

interface Props {
  videoSrc: string;
  /** Omit or pass [] for video-only playback (no skeleton / phase HUD). */
  poseFrames?: OverlayPoseFrame[];
  lang: "en" | "zh";
  coachingTips?: VideoCoachingTips | null;
  prediction?: ShotPrediction | null;
  /** Same as backend OpenCV CAP_PROP_FRAME_COUNT — aligns scrubber with pose frame_index */
  sourceFrameCount?: number;
  /** Plus-style gradient skeleton + plumb + meteor arcs (Stellar Pro / Plus result parity). */
  skeletonStyle?: "legacy" | "plus";
}

export type { VideoCoachingTips };

/* ═══════════════ Pro reference (typical tour ranges, educational) ═══════════════ */

const PRO_REF_LINES: Record<
  string,
  { zh: string; en: string }
> = {
  x_factor: { zh: "职业峰值约40–55°", en: "Tour peak ~40–55°" },
  spine_tilt: { zh: "职业常见约15–35°", en: "Tour often ~15–35°" },
  shoulder_rotation: { zh: "因人而异", en: "Highly individual" },
  hip_rotation: { zh: "因人而异", en: "Highly individual" },
  left_elbow: { zh: "职业伸展约165–175°", en: "Tour ext. ~165–175°" },
  right_elbow: { zh: "职业伸展约165–175°", en: "Tour ext. ~165–175°" },
  left_knee: { zh: "职业微屈约150–170°", en: "Tour flex ~150–170°" },
  right_knee: { zh: "职业微屈约150–170°", en: "Tour flex ~150–170°" },
};

const JOINT_COLOR: Record<string, string> = {
  head: "#ef4444",
  left_shoulder: "#a855f7",
  right_shoulder: "#a855f7",
  left_elbow: "#6366f1",
  right_elbow: "#6366f1",
  left_wrist: "#3b82f6",
  right_wrist: "#3b82f6",
  left_hip: "#14b8a6",
  right_hip: "#14b8a6",
  left_knee: "#22c55e",
  right_knee: "#22c55e",
  left_ankle: "#eab308",
  right_ankle: "#eab308",
};

const PHASE_COLORS: Record<string, string> = {
  address: "#94a3b8",
  takeaway: "#60a5fa",
  backswing: "#818cf8",
  top: "#a78bfa",
  downswing: "#f97316",
  impact: "#ef4444",
  follow_through: "#22c55e",
  finish: "#14b8a6",
};

const PLAYBACK_RATES = [0.25, 0.5, 1, 1.5, 2] as const;

const IS_MOBILE =
  typeof navigator !== "undefined" &&
  /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);

/* ═══════════════ Helpers ═══════════════ */

/** Min/max frame_index across poses (decoder ordinal in source file). */
function frameIndexBounds(frames: OverlayPoseFrame[]): {
  minFi: number;
  maxFi: number;
} {
  let minFi = frames[0].frame_index;
  let maxFi = frames[0].frame_index;
  for (let i = 1; i < frames.length; i++) {
    const fi = frames[i].frame_index;
    if (fi < minFi) minFi = fi;
    if (fi > maxFi) maxFi = fi;
  }
  return { minFi, maxFi };
}

/**
 * Pick the pose that matches the current video scrub position.
 * Prefer mapping currentTime/duration → frame_index linearly: browser timeline and
 * OpenCV-reported FPS often disagree, so timestamp = frame_index/fps drifts from
 * HTML5 video.currentTime. Fallback: nearest by backend timestamp.
 */
function findNearestPose(
  frames: OverlayPoseFrame[],
  time: number,
  videoDuration?: number,
  sourceFrameCount?: number,
): OverlayPoseFrame | null {
  if (!frames.length) return null;

  const dur =
    videoDuration != null &&
    Number.isFinite(videoDuration) &&
    videoDuration > 0.05
      ? videoDuration
      : 0;
  const u = dur > 0 ? Math.max(0, Math.min(1, time / dur)) : null;

  if (u != null) {
    let targetFi: number | null = null;
    if (
      sourceFrameCount != null &&
      Number.isFinite(sourceFrameCount) &&
      sourceFrameCount > 1
    ) {
      targetFi = u * (sourceFrameCount - 1);
    } else {
      const { minFi, maxFi } = frameIndexBounds(frames);
      if (maxFi > minFi) {
        targetFi = minFi + u * (maxFi - minFi);
      }
    }
    if (targetFi != null) {
      let best = frames[0];
      let bestD = Math.abs(frames[0].frame_index - targetFi);
      for (let i = 1; i < frames.length; i++) {
        const d = Math.abs(frames[i].frame_index - targetFi);
        if (d < bestD) {
          bestD = d;
          best = frames[i];
        } else if (d === bestD) {
          const db = Math.abs(best.timestamp - time);
          const di = Math.abs(frames[i].timestamp - time);
          if (di < db) best = frames[i];
        }
      }
      return best;
    }
  }

  const byTs = [...frames].sort((a, b) => a.timestamp - b.timestamp);
  let lo = 0,
    hi = byTs.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (byTs[mid].timestamp < time) lo = mid + 1;
    else hi = mid;
  }
  if (
    lo > 0 &&
    Math.abs(byTs[lo - 1].timestamp - time) <
      Math.abs(byTs[lo].timestamp - time)
  ) {
    return byTs[lo - 1];
  }
  return byTs[lo];
}

/** Map a source frame index to HTML5 video seconds (linear in file vs browser duration). */
function mediaSecondsAtFrameIndex(
  frames: OverlayPoseFrame[],
  fiTarget: number,
  videoDuration: number,
  sourceFrameCount?: number,
): number {
  if (!frames.length) return 0;
  if (
    sourceFrameCount != null &&
    sourceFrameCount > 1 &&
    videoDuration > 0.05
  ) {
    const u = fiTarget / (sourceFrameCount - 1);
    return Math.max(0, Math.min(videoDuration, u * videoDuration));
  }
  const { minFi, maxFi } = frameIndexBounds(frames);
  if (maxFi > minFi && videoDuration > 0.05) {
    const u = (fiTarget - minFi) / (maxFi - minFi);
    return Math.max(0, Math.min(videoDuration, u * videoDuration));
  }
  let best = frames[0];
  let bd = Math.abs(frames[0].frame_index - fiTarget);
  for (let i = 1; i < frames.length; i++) {
    const d = Math.abs(frames[i].frame_index - fiTarget);
    if (d < bd) {
      bd = d;
      best = frames[i];
    }
  }
  return best.timestamp;
}

function calcLetterbox(
  fW: number,
  fH: number,
  cW: number,
  cH: number,
) {
  if (!fW || !fH) return { offsetX: 0, offsetY: 0, renderW: cW, renderH: cH };
  const containerAR = cW / cH;
  const frameAR = fW / fH;
  let renderW: number, renderH: number, offsetX: number, offsetY: number;
  if (frameAR >= containerAR) {
    renderW = cW;
    renderH = cW / frameAR;
    offsetX = 0;
    offsetY = (cH - renderH) / 2;
  } else {
    renderH = cH;
    renderW = cH * frameAR;
    offsetX = (cW - renderW) / 2;
    offsetY = 0;
  }
  return { offsetX, offsetY, renderW, renderH };
}

function fillRoundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  ctx.beginPath();
  if (typeof ctx.roundRect === "function") {
    ctx.roundRect(x, y, w, h, r);
  } else {
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }
  ctx.fill();
}

interface SwingTrailData {
  backswing: Array<{ nx: number; ny: number }>;
  downswing: Array<{ nx: number; ny: number }>;
  phase: string;
  bsOpacity: number;
  dsOpacity: number;
}

interface DrawOpts {
  duration: number;
  currentTime: number;
  sweetSpot: { start: number; end: number; center: number } | null;
  compositeSkipClear?: boolean;
  prediction?: ShotPrediction | null;
  impactTime?: number;
  ballOriginNorm?: { nx: number; ny: number } | null;
  swingTrail?: SwingTrailData | null;
  videoFrameSize?: { width: number; height: number } | null;
  skeletonStyle?: "legacy" | "plus";
}

const TRAJECTORY_ANIM_SEC = 2.0;

/** Yardage for trajectory / badge when predicted_distance is missing or zero. */
function trajectoryDisplayYards(pred: ShotPrediction | null | undefined): number {
  if (!pred) return 0;
  const raw = Number(pred.predicted_distance);
  if (Number.isFinite(raw) && raw > 0) return raw;
  const chs = Number(pred.club_head_speed);
  if (Number.isFinite(chs) && chs > 0) {
    return Math.round(Math.min(320, Math.max(150, chs * 4.2)));
  }
  return 0;
}

function bezierPoint(
  t: number,
  p0: number, p1: number, p2: number, p3: number,
): number {
  const u = 1 - t;
  return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3;
}

interface TrajectoryOrigin {
  x: number;
  y: number;
}

function drawTrajectory(
  ctx: CanvasRenderingContext2D,
  cW: number,
  cH: number,
  prediction: ShotPrediction,
  lang: "en" | "zh",
  sh: number,
  progress: number,
  origin?: TrajectoryOrigin | null,
) {
  const dist = trajectoryDisplayYards(prediction);
  const shape = lang === "zh"
    ? (prediction.shot_shape_zh || prediction.shot_shape || "")
    : (prediction.shot_shape || "");
  if (dist <= 0) return;

  const p = Math.max(0, Math.min(1, progress));
  if (p <= 0) return;

  ctx.save();

  const baseX = origin ? origin.x : cW * 0.68;
  const baseY = origin ? origin.y : cH * 0.82;
  const flightH = baseY - cH * 0.04;

  let lateralDrift = 0;
  const shapeLC = (prediction.shot_shape || "").toLowerCase();
  if (shapeLC.includes("draw") || shapeLC.includes("左曲") || shapeLC.includes("hook"))
    lateralDrift = cW * 0.06;
  else if (shapeLC.includes("fade") || shapeLC.includes("右曲") || shapeLC.includes("slice"))
    lateralDrift = -cW * 0.08;

  const endX = baseX + lateralDrift;
  const endY = cH * 0.04;
  const cp1x = baseX + lateralDrift * 0.1;
  const cp1y = baseY - flightH * 0.35;
  const cp2x = baseX + lateralDrift * 0.65;
  const cp2y = baseY - flightH * 0.78;

  const headX = bezierPoint(p, baseX, cp1x, cp2x, endX);
  const headY = bezierPoint(p, baseY, cp1y, cp2y, endY);

  /* ── Comet-tail tracer ── */
  const isLanded = p >= 1;
  const tailLen = isLanded ? 1.0 : 0.40;
  const tailStart = Math.max(0, p - tailLen);
  const SEGMENTS = IS_MOBILE ? 32 : 64;
  const segFrom = Math.floor(tailStart * SEGMENTS);
  const segTo = Math.ceil(p * SEGMENTS);

  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  for (let i = segFrom; i < segTo; i++) {
    const t0 = i / SEGMENTS;
    const t1 = (i + 1) / SEGMENTS;
    if (t1 > p) break;

    const localAlpha = (t1 - tailStart) / (p - tailStart + 0.001);
    const alpha = Math.pow(Math.max(0, Math.min(1, localAlpha)), isLanded ? 0.6 : 1.8);
    const lineW = Math.max(2, (2 + 14 * alpha) * sh);

    const x0 = bezierPoint(t0, baseX, cp1x, cp2x, endX);
    const y0 = bezierPoint(t0, baseY, cp1y, cp2y, endY);
    const x1 = bezierPoint(t1, baseX, cp1x, cp2x, endX);
    const y1 = bezierPoint(t1, baseY, cp1y, cp2y, endY);

    if (!IS_MOBILE) {
      ctx.strokeStyle = `rgba(255, 50, 0, ${(0.06 * alpha).toFixed(3)})`;
      ctx.lineWidth = lineW * 3.5;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }

    ctx.strokeStyle = `rgba(255, 80, 20, ${(0.12 + 0.7 * alpha).toFixed(2)})`;
    ctx.lineWidth = lineW * 1.6;
    if (!IS_MOBILE) {
      ctx.shadowColor = `rgba(255, 60, 0, ${(0.5 * alpha).toFixed(2)})`;
      ctx.shadowBlur = lineW * 5;
    }
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();

    ctx.strokeStyle = `rgba(255, 255, 240, ${(0.25 + 0.75 * alpha).toFixed(2)})`;
    ctx.lineWidth = lineW * 0.45;
    if (!IS_MOBILE) {
      ctx.shadowColor = `rgba(255, 200, 150, ${(0.6 * alpha).toFixed(2)})`;
      ctx.shadowBlur = lineW * 2;
    }
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  }
  ctx.shadowBlur = 0;

  /* ── Tracer Head (animated ball) ── */
  if (!isLanded) {
    const headR = Math.max(6, 8 * sh);
    const headGlow = Math.max(20, 32 * sh);

    // Outer halo
    const grad = ctx.createRadialGradient(headX, headY, headR * 0.3, headX, headY, headR * 2.5);
    grad.addColorStop(0, "rgba(255, 200, 100, 0.7)");
    grad.addColorStop(0.5, "rgba(255, 80, 20, 0.3)");
    grad.addColorStop(1, "rgba(255, 40, 0, 0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(headX, headY, headR * 2.5, 0, Math.PI * 2);
    ctx.fill();

    // Red-orange ball
    ctx.shadowColor = "rgba(255, 80, 20, 0.95)";
    ctx.shadowBlur = headGlow;
    ctx.fillStyle = "rgba(255, 90, 40, 0.95)";
    ctx.beginPath();
    ctx.arc(headX, headY, headR, 0, Math.PI * 2);
    ctx.fill();

    // White-hot center
    ctx.shadowColor = "rgba(255, 255, 200, 0.9)";
    ctx.shadowBlur = headGlow * 0.4;
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(headX, headY, headR * 0.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  /* ── Dynamic Yardage HUD (glassmorphism badge) ── */
  const distEase = 1 - Math.pow(1 - p, 3);
  const currentDist = isLanded ? dist : Math.round(dist * distEase);

  const yardFont = Math.max(16, 24 * sh);
  const labelFont = Math.max(9, 12 * sh);
  const yardText = `${currentDist}`;
  const unitText = lang === "zh" ? "码" : "YDS";

  ctx.font = `800 ${yardFont}px ui-sans-serif, system-ui, -apple-system, sans-serif`;
  const yardW = ctx.measureText(yardText).width;
  ctx.font = `600 ${labelFont}px ui-sans-serif, system-ui, -apple-system, sans-serif`;
  const unitW = ctx.measureText(unitText).width;

  const padX = Math.max(10, 14 * sh);
  const padY = Math.max(7, 10 * sh);
  const gap = 3;
  const badgeContentW = yardW + gap + unitW;
  const badgeW = badgeContentW + padX * 2;
  const badgeH = yardFont + padY * 2;
  const badgeR = Math.max(8, 10 * sh);

  // Position: attached to head when animating, fixed top-right when landed
  let badgeX: number, badgeY: number;
  if (isLanded) {
    badgeX = headX + Math.max(14, 20 * sh);
    badgeY = headY - badgeH / 2;
  } else {
    badgeX = headX + Math.max(14, 20 * sh);
    badgeY = headY - badgeH / 2;
  }

  // Clamp badge within canvas
  if (badgeX + badgeW > cW - 4) badgeX = headX - badgeW - Math.max(14, 20 * sh);
  if (badgeY < 4) badgeY = 4;

  // Pop animation on landing
  let badgeScale = 1;
  if (isLanded && progress < 1.15) {
    const landT = Math.min(1, (progress - 1) / 0.15);
    badgeScale = 1 + 0.12 * Math.sin(landT * Math.PI);
  }

  ctx.save();
  if (badgeScale !== 1) {
    const cx = badgeX + badgeW / 2;
    const cy = badgeY + badgeH / 2;
    ctx.translate(cx, cy);
    ctx.scale(badgeScale, badgeScale);
    ctx.translate(-cx, -cy);
  }

  // Glassmorphism background
  ctx.fillStyle = "rgba(10, 10, 16, 0.72)";
  fillRoundRect(ctx, badgeX, badgeY, badgeW, badgeH, badgeR);

  // Glass border
  ctx.strokeStyle = isLanded ? "rgba(255, 180, 80, 0.35)" : "rgba(255, 255, 255, 0.12)";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  if (typeof ctx.roundRect === "function") {
    ctx.roundRect(badgeX, badgeY, badgeW, badgeH, badgeR);
  } else {
    ctx.rect(badgeX, badgeY, badgeW, badgeH);
  }
  ctx.stroke();

  // Yardage number
  const textCY = badgeY + badgeH / 2;
  const textStartX = badgeX + padX;

  ctx.font = `800 ${yardFont}px ui-sans-serif, system-ui, -apple-system, sans-serif`;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillStyle = isLanded ? "#ffcc66" : "#ffffff";
  ctx.shadowColor = "rgba(0,0,0,0.6)";
  ctx.shadowBlur = 3;
  ctx.fillText(yardText, textStartX, textCY);

  // Unit label
  ctx.font = `600 ${labelFont}px ui-sans-serif, system-ui, -apple-system, sans-serif`;
  ctx.fillStyle = isLanded ? "rgba(255, 200, 100, 0.8)" : "rgba(255, 255, 255, 0.6)";
  ctx.fillText(unitText, textStartX + yardW + gap, textCY + (yardFont - labelFont) * 0.15);
  ctx.shadowBlur = 0;

  ctx.restore();

  /* ── Club head speed badge (shown after landing) ── */
  if (isLanded && prediction.club_head_speed && prediction.club_head_speed > 0) {
    const speedFont = Math.max(9, 11 * sh);
    const speedVal = `${Math.round(prediction.club_head_speed)}`;
    const speedUnit = lang === "zh" ? "mph 杆速" : "mph CHS";

    ctx.font = `700 ${speedFont}px ui-sans-serif, system-ui, -apple-system, sans-serif`;
    const speedTextW = ctx.measureText(speedVal + " " + speedUnit).width;
    const sPadX = Math.max(6, 8 * sh);
    const sPadY = Math.max(4, 5 * sh);
    const sBadgeW = speedTextW + sPadX * 2;
    const sBadgeH = speedFont + sPadY * 2;
    const sBadgeX = badgeX;
    const sBadgeY = badgeY + badgeH + Math.max(4, 6 * sh);

    ctx.fillStyle = "rgba(10, 10, 16, 0.6)";
    fillRoundRect(ctx, sBadgeX, sBadgeY, sBadgeW, sBadgeH, Math.max(4, 6 * sh));
    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (typeof ctx.roundRect === "function") {
      ctx.roundRect(sBadgeX, sBadgeY, sBadgeW, sBadgeH, Math.max(4, 6 * sh));
    } else {
      ctx.rect(sBadgeX, sBadgeY, sBadgeW, sBadgeH);
    }
    ctx.stroke();

    ctx.font = `700 ${speedFont}px ui-sans-serif, system-ui, -apple-system, sans-serif`;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
    ctx.fillText(speedVal, sBadgeX + sPadX, sBadgeY + sBadgeH / 2);
    const svW = ctx.measureText(speedVal).width;
    ctx.font = `500 ${speedFont * 0.85}px ui-sans-serif, system-ui, -apple-system, sans-serif`;
    ctx.fillStyle = "rgba(255, 255, 255, 0.45)";
    ctx.fillText(" " + speedUnit, sBadgeX + sPadX + svW, sBadgeY + sBadgeH / 2);
  }

  /* ── Shot shape tag ── */
  if (isLanded && shape) {
    const speedH = (prediction.club_head_speed && prediction.club_head_speed > 0) ? Math.max(9, 11 * sh) + Math.max(4, 5 * sh) * 2 + Math.max(4, 6 * sh) : 0;
    const shapeFont = Math.max(10, 12 * sh);
    ctx.font = `700 ${shapeFont}px ui-sans-serif, system-ui, -apple-system, sans-serif`;
    const sw = ctx.measureText(shape).width;
    const sX = badgeX + badgeW / 2;
    const sY = badgeY + badgeH + speedH + Math.max(8, 12 * sh);

    ctx.fillStyle = "rgba(10, 10, 16, 0.6)";
    fillRoundRect(ctx, sX - sw / 2 - 8, sY - shapeFont * 0.6, sw + 16, shapeFont + 6, 4);
    ctx.fillStyle = "#ff9800";
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    ctx.fillText(shape, sX, sY);
  }

  ctx.restore();
}

/* ═══════════════ Swing trail helpers ═══════════════ */

const BS_PHASES = new Set(["takeaway", "backswing", "top"]);
const BS_FADE_SEC = 0.6;
const DS_FADE_SEC = 1.0;
const CLUB_EXT = 0.7;

function getClubHead(f: OverlayPoseFrame): { nx: number; ny: number } | null {
  const rw = f.joints.find(j => j.name === "right_wrist" && j.visibility > 0.3);
  const lw = f.joints.find(j => j.name === "left_wrist" && j.visibility > 0.3);
  let wx: number, wy: number;
  if (rw && lw) { wx = (rw.normalized.x + lw.normalized.x) / 2; wy = (rw.normalized.y + lw.normalized.y) / 2; }
  else { const w = rw || lw; if (!w) return null; wx = w.normalized.x; wy = w.normalized.y; }

  const re = f.joints.find(j => j.name === "right_elbow" && j.visibility > 0.3);
  const le = f.joints.find(j => j.name === "left_elbow" && j.visibility > 0.3);
  if (re || le) {
    let ex: number, ey: number;
    if (re && le) { ex = (re.normalized.x + le.normalized.x) / 2; ey = (re.normalized.y + le.normalized.y) / 2; }
    else { const e = (re || le)!; ex = e.normalized.x; ey = e.normalized.y; }
    const dx = wx - ex, dy = wy - ey;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len > 0.008) { wx += (dx / len) * len * CLUB_EXT; wy += (dy / len) * len * CLUB_EXT; }
  }
  return { nx: wx, ny: wy };
}

/** Post-impact arc: club head path from strike through finish (not pre-impact downswing). */
const DS_PHASES = new Set(["impact", "follow_through", "finish"]);

function computeSwingTrail(
  frames: OverlayPoseFrame[],
  currentTime: number,
  videoDuration?: number,
  sourceFrameCount?: number,
): SwingTrailData {
  const backswing: Array<{ nx: number; ny: number }> = [];
  const downswing: Array<{ nx: number; ny: number }> = [];
  let phase = "address";
  let lastBsTime = -1;
  let lastDsTime = -1;
  let lastFinishTime = -1;

  const dur =
    videoDuration != null &&
    Number.isFinite(videoDuration) &&
    videoDuration > 0.05
      ? videoDuration
      : 0;
  const uCut =
    dur > 0 ? Math.max(0, Math.min(1, currentTime / dur)) : null;
  const sorted = [...frames].sort((a, b) => a.frame_index - b.frame_index);
  const { minFi, maxFi } =
    sorted.length > 0 ? frameIndexBounds(sorted) : { minFi: 0, maxFi: 0 };
  const span = maxFi > minFi ? maxFi - minFi : 0;
  const useFileNorm =
    dur > 0 &&
    sourceFrameCount != null &&
    sourceFrameCount > 1;
  const normFi = (f: OverlayPoseFrame) => {
    if (useFileNorm) return f.frame_index / (sourceFrameCount! - 1);
    return span > 0 ? (f.frame_index - minFi) / span : 0;
  };
  const useNormCut = uCut != null && (useFileNorm || span > 0);
  const mediaTime = (f: OverlayPoseFrame) =>
    dur > 0 && useNormCut ? normFi(f) * dur : f.timestamp;

  for (const f of sorted) {
    if (useNormCut) {
      if (normFi(f) > uCut! + 0.0005) break;
    } else if (f.timestamp > currentTime + 0.01) {
      break;
    }
    const pid = f.phase_data?.phase_id || "";
    if (pid) phase = pid;
    if (pid === "finish") lastFinishTime = mediaTime(f);

    const pt = getClubHead(f);
    if (!pt) continue;

    if (BS_PHASES.has(pid)) {
      backswing.push(pt);
      lastBsTime = mediaTime(f);
    } else if (DS_PHASES.has(pid)) {
      downswing.push(pt);
      lastDsTime = mediaTime(f);
    }
  }

  let bsOpacity = 0;
  if (BS_PHASES.has(phase)) {
    bsOpacity = 1;
  } else if (lastBsTime > 0) {
    bsOpacity = Math.max(0, 1 - (currentTime - lastBsTime) / BS_FADE_SEC);
  }

  let dsOpacity = 0;
  if (DS_PHASES.has(phase)) {
    dsOpacity = 1;
  } else if (lastDsTime > 0) {
    const fadeFrom = lastFinishTime > 0 ? lastFinishTime : lastDsTime;
    dsOpacity = Math.max(0, 1 - (currentTime - fadeFrom) / DS_FADE_SEC);
  }

  return { backswing, downswing, phase, bsOpacity, dsOpacity };
}

function catmullRom(pts: Array<[number, number]>, sub: number = 4): Array<[number, number]> {
  if (pts.length < 3) return pts;
  const out: Array<[number, number]> = [pts[0]];
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)];
    const p1 = pts[i];
    const p2 = pts[Math.min(pts.length - 1, i + 1)];
    const p3 = pts[Math.min(pts.length - 1, i + 2)];
    for (let k = 1; k <= sub; k++) {
      const t = k / sub;
      const t2 = t * t, t3 = t2 * t;
      out.push([
        0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3),
        0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3),
      ]);
    }
  }
  return out;
}

function drawGradientTrail(
  ctx: CanvasRenderingContext2D,
  rawPts: Array<[number, number]>,
  colorStops: string[],
  s: number,
  masterAlpha: number,
) {
  if (rawPts.length < 2 || masterAlpha <= 0) return;

  let sampled = rawPts;
  if (rawPts.length > 25) {
    sampled = [];
    for (let i = 0; i < 25; i++) sampled.push(rawPts[Math.round((i / 24) * (rawPts.length - 1))]);
  }
  const pts = catmullRom(sampled, sampled.length < 6 ? 5 : 3);
  const n = pts.length;
  if (n < 2) return;

  const cLerp = (t: number): string => {
    const idx = t * (colorStops.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.min(lo + 1, colorStops.length - 1);
    return colorStops[lo === hi ? lo : Math.round(idx) <= lo ? lo : hi];
  };

  // Layer 1: soft outer glow (skeleton-scale)
  ctx.save();
  for (let i = 0; i < n - 1; i++) {
    const t = i / (n - 1);
    ctx.globalAlpha = masterAlpha * (0.04 + t * 0.08);
    ctx.strokeStyle = cLerp(t);
    ctx.lineWidth = Math.max(2, (5 + t * 5) * s);
    ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(pts[i][0], pts[i][1]); ctx.lineTo(pts[i + 1][0], pts[i + 1][1]); ctx.stroke();
  }
  ctx.restore();

  // Layer 2: main gradient line
  ctx.save();
  for (let i = 0; i < n - 1; i++) {
    const t = i / (n - 1);
    const [x1, y1] = pts[i], [x2, y2] = pts[i + 1];
    const ci = Math.floor(t * (colorStops.length - 1));
    const ci2 = Math.min(ci + 1, colorStops.length - 1);
    const g = ctx.createLinearGradient(x1, y1, x2, y2);
    g.addColorStop(0, colorStops[ci]); g.addColorStop(1, colorStops[ci2]);
    ctx.globalAlpha = masterAlpha * (0.2 + t * 0.65);
    ctx.strokeStyle = g;
    ctx.lineWidth = Math.max(1, (1.5 + t * 1.5) * s);
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }
  ctx.restore();

  // Layer 3: inner white specular on last 25%
  const s0 = Math.floor(n * 0.75);
  ctx.save();
  for (let i = s0; i < n - 1; i++) {
    const t = (i - s0) / Math.max(1, n - 1 - s0);
    ctx.globalAlpha = masterAlpha * t * 0.3;
    ctx.strokeStyle = "rgba(255,255,255,0.6)";
    ctx.lineWidth = Math.max(0.4, 0.8 * s);
    ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(pts[i][0], pts[i][1]); ctx.lineTo(pts[i + 1][0], pts[i + 1][1]); ctx.stroke();
  }
  ctx.restore();

  // Layer 4: meteor head glow
  const [hx, hy] = pts[n - 1];
  ctx.save();
  const hc = colorStops[colorStops.length - 1];
  const glowR = Math.max(3, 6 * s);
  const glow = ctx.createRadialGradient(hx, hy, 0, hx, hy, glowR);
  glow.addColorStop(0, "rgba(255,255,255,0.9)");
  glow.addColorStop(0.25, hc + "bb");
  glow.addColorStop(0.6, hc + "44");
  glow.addColorStop(1, "transparent");
  ctx.globalAlpha = masterAlpha;
  ctx.fillStyle = glow;
  ctx.beginPath(); ctx.arc(hx, hy, glowR, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}

function drawOverlay(
  ctx: CanvasRenderingContext2D,
  cW: number,
  cH: number,
  pose: OverlayPoseFrame | null,
  flags: { skeleton: boolean; angles: boolean; phase: boolean; guideLines?: boolean },
  lang: "en" | "zh",
  opts: DrawOpts,
) {
  if (!opts.compositeSkipClear) ctx.clearRect(0, 0, cW, cH);
  const { duration, currentTime, sweetSpot } = opts;

  const hasPose = !!(pose?.joints?.length);
  // Keep overlay in the same coordinate space as pose output. Only trust browser video
  // dimensions when aspect ratio is effectively identical, otherwise prefer pose frame_size.
  const poseW = hasPose ? (pose!.frame_size?.width || cW) : cW;
  const poseH = hasPose ? (pose!.frame_size?.height || cH) : cH;
  const videoW = opts.videoFrameSize?.width || 0;
  const videoH = opts.videoFrameSize?.height || 0;
  const poseAR = poseW > 0 && poseH > 0 ? poseW / poseH : 0;
  const videoAR = videoW > 0 && videoH > 0 ? videoW / videoH : 0;
  const arClose =
    poseAR > 0 &&
    videoAR > 0 &&
    Math.abs(poseAR - videoAR) / poseAR <= 0.02;
  const fW = arClose ? videoW : poseW;
  const fH = arClose ? videoH : poseH;
  const { offsetX, offsetY, renderW, renderH } = calcLetterbox(fW, fH, cW, cH);

  const px = (nx: number, ny: number): [number, number] => [
    offsetX + nx * renderW,
    offsetY + ny * renderH,
  ];

  const ref = Math.min(renderW, renderH);
  const s = Math.max(0.35, Math.min(2.8, ref / 260));
  /** Compact HUD scale: small, subtle text (metrics / timeline / phase). */
  const sh = Math.min(0.52, Math.max(0.26, ref / 720));
  const joints = pose?.joints ?? [];

  /* Height of top phase HUD (for metrics panel vertical offset) */
  let phaseTopH = 0;

  /* ─── Phase: thin bar + label pill TOP CENTER ─── */
  if (hasPose && flags.phase && pose!.phase_data) {
    const pd = pose!.phase_data!;
    const phaseColor = PHASE_COLORS[pd.phase_id] || "#94a3b8";
    const pct = pd.progress_pct || 0;

    const barH = Math.max(2, 2.5 * sh);
    ctx.fillStyle = "rgba(0,0,0,0.1)";
    ctx.fillRect(0, 0, cW, barH);
    ctx.globalAlpha = 0.5;
    ctx.fillStyle = phaseColor;
    ctx.fillRect(0, 0, (cW * pct) / 100, barH);
    ctx.globalAlpha = 1;

    const label = lang === "zh" ? pd.phase_zh : pd.phase_en;
    const fontSize = Math.max(9, 10 * sh);
    ctx.save();
    ctx.font = `600 ${fontSize}px ui-sans-serif, system-ui, -apple-system, sans-serif`;
    const tw = ctx.measureText(label).width;
    const pillW = tw + 14 * sh;
    const pillH = fontSize + 7 * sh;
    const pillX = (cW - pillW) / 2;
    const pillY = barH + 4 * sh;
    phaseTopH = pillY + pillH + 4 * sh;

    ctx.fillStyle = "rgba(0,0,0,0.22)";
    fillRoundRect(ctx, pillX, pillY, pillW, pillH, 5 * sh);
    ctx.fillStyle = phaseColor;
    ctx.globalAlpha = 0.82;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, cW / 2, pillY + pillH / 2);
    ctx.globalAlpha = 1;
    ctx.restore();
  }

  /* ─── Club type + hand indicator pill ─── */
  if (opts.prediction && (opts.prediction.club_type && opts.prediction.club_type !== "UNKNOWN" || opts.prediction.hand && opts.prediction.hand !== "UNKNOWN")) {
    const clubLabel = (opts.prediction.club_type && opts.prediction.club_type !== "UNKNOWN") ? opts.prediction.club_type : "";
    const handLabel = opts.prediction.hand === "L" ? (lang === "zh" ? "左手" : "L") : (opts.prediction.hand === "R" ? (lang === "zh" ? "右手" : "R") : "");
    const infoText = [clubLabel, handLabel].filter(Boolean).join(" · ");
    if (infoText) {
      const fs2 = Math.max(7, 8 * sh);
      ctx.save();
      ctx.font = `500 ${fs2}px ui-sans-serif, system-ui, sans-serif`;
      const tw2 = ctx.measureText(infoText).width;
      const pw2 = tw2 + 10 * sh;
      const ph2 = fs2 + 5 * sh;
      const px2 = (cW - pw2) / 2;
      const py2 = phaseTopH + 2 * sh;
      ctx.fillStyle = "rgba(0,0,0,0.25)";
      fillRoundRect(ctx, px2, py2, pw2, ph2, 4 * sh);
      ctx.fillStyle = "rgba(255,255,255,0.6)";
      ctx.globalAlpha = 0.7;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(infoText, cW / 2, py2 + ph2 / 2);
      ctx.restore();
      phaseTopH = py2 + ph2 + 3 * sh;
    }
  }

  /* ─── Skeleton + guides: Plus parity (meteor arcs) or legacy (time-based swing trail) ─── */
  const skelPlus = opts.skeletonStyle === "plus";
  const showPlusGuides = flags.guideLines ?? flags.skeleton;
  if (hasPose && skelPlus) {
    const sPlus = plusSkeletonScale(renderW, renderH);
    drawPlusStyleSkeletonOverlay(
      ctx,
      pose!,
      px,
      sPlus,
      offsetY,
      renderH,
      flags.skeleton,
      showPlusGuides,
    );
  } else if (hasPose && flags.skeleton) {
    const lineMain = Math.max(1, 2.5 * s);
    const lineGlow = Math.max(2, 8 * s);

    for (const conn of pose.connections || []) {
      const j1 = joints[conn[0]];
      const j2 = joints[conn[1]];
      if (!j1 || !j2 || j1.visibility < 0.3 || j2.visibility < 0.3) continue;
      const [x1, y1] = px(j1.normalized.x, j1.normalized.y);
      const [x2, y2] = px(j2.normalized.x, j2.normalized.y);

      ctx.save();
      ctx.globalAlpha = 0.07;
      ctx.strokeStyle = "#9f5fff";
      ctx.lineWidth = lineGlow;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();

      ctx.globalAlpha = 0.85;
      const g = ctx.createLinearGradient(x1, y1, x2, y2);
      g.addColorStop(0, JOINT_COLOR[j1.name] || "#b97bff");
      g.addColorStop(1, JOINT_COLOR[j2.name] || "#f5c518");
      ctx.strokeStyle = g;
      ctx.lineWidth = lineMain;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      ctx.restore();
    }

    for (const j of joints) {
      if (j.visibility < 0.3) continue;
      const [x, y] = px(j.normalized.x, j.normalized.y);
      const color = JOINT_COLOR[j.name] || "#a855f7";
      const isKey =
        j.name.includes("shoulder") ||
        j.name.includes("hip") ||
        j.name.includes("wrist");
      const r = isKey ? Math.max(2, 4.5 * s) : Math.max(1.2, 3 * s);

      ctx.save();
      const glow = ctx.createRadialGradient(x, y, 0, x, y, r * 2.5);
      glow.addColorStop(0, color + "33");
      glow.addColorStop(1, "transparent");
      ctx.beginPath();
      ctx.arc(x, y, r * 2.5, 0, Math.PI * 2);
      ctx.fillStyle = glow;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = color + "28";
      ctx.fill();
      ctx.strokeStyle =
        j.visibility >= 0.7 ? color + "dd" : color + "88";
      ctx.lineWidth = Math.max(0.8, 1.6 * s);
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(x, y, Math.max(0.5, r * 0.35), 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,255,255,0.9)";
      ctx.fill();
      ctx.restore();
    }

    const nose = joints.find(j => j.name === "nose");
    const headJ = joints.find(j => j.name === "head");
    const lank = joints.find(j => j.name === "left_ankle");
    const rank = joints.find(j => j.name === "right_ankle");
    const gHead = nose && nose.visibility > 0.3 ? nose : (headJ && headJ.visibility > 0.3 ? headJ : null);

    if (gHead) {
      const [nx, ny] = px(gHead.normalized.x, gHead.normalized.y);
      const footY = lank && lank.visibility > 0.3 ? px(0, lank.normalized.y)[1]
        : rank && rank.visibility > 0.3 ? px(0, rank.normalized.y)[1]
        : offsetY + renderH * 0.92;
      ctx.save(); ctx.globalAlpha = 0.45; ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = Math.max(1, 1.8 * s); ctx.setLineDash([6 * s, 4 * s]);
      ctx.beginPath(); ctx.moveTo(nx, ny - 8 * s); ctx.lineTo(nx, footY); ctx.stroke();
      ctx.restore();
    }

    if (opts.swingTrail) {
      const { backswing, downswing, bsOpacity, dsOpacity } = opts.swingTrail;
      if (backswing.length >= 2 && bsOpacity > 0) {
        drawGradientTrail(ctx, backswing.map(p => px(p.nx, p.ny)), ["#1e3a8a", "#2563eb", "#0ea5e9", "#22d3ee"], s, bsOpacity);
      }
      if (downswing.length >= 2 && dsOpacity > 0) {
        drawGradientTrail(ctx, downswing.map(p => px(p.nx, p.ny)), ["#16a34a", "#22c55e", "#eab308", "#f97316", "#ef4444"], s, dsOpacity);
      }
    }
  }

  /* ─── Angles + Pro ref + metrics panel ─── */
  if (hasPose && flags.angles && pose!.angles) {
    const ang = pose!.angles!;
    const angleItems: { key: string; joint: string; dir: number }[] = [
      { key: "left_elbow", joint: "left_elbow", dir: -1 },
      { key: "right_elbow", joint: "right_elbow", dir: 1 },
      { key: "left_knee", joint: "left_knee", dir: -1 },
      { key: "right_knee", joint: "right_knee", dir: 1 },
    ];

    const fontSize = Math.max(7, 8 * sh);
    const microSize = Math.max(6, 6.5 * sh);
    ctx.save();
    ctx.font = `500 ${fontSize}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";

    for (const ai of angleItems) {
      const val = ang[ai.key];
      if (val == null) continue;
      const j = joints.find((jt) => jt.name === ai.joint);
      if (!j || j.visibility < 0.3) continue;
      const [x, y] = px(j.normalized.x, j.normalized.y);
      const tx = x + ai.dir * Math.max(14, 18 * sh);
      const ty = y - Math.max(2, 4 * sh);

      const text = `${Math.round(val)}°`;
      const tw = ctx.measureText(text).width;
      const proLine = PRO_REF_LINES[ai.key];
      const proText = proLine
        ? lang === "zh"
          ? proLine.zh
          : proLine.en
        : "";

      ctx.font = `400 ${microSize}px ui-sans-serif, system-ui, sans-serif`;
      const pw = proText ? ctx.measureText(proText).width : 0;
      ctx.font = `500 ${fontSize}px ui-sans-serif, system-ui, sans-serif`;
      const boxInnerW = Math.max(tw, pw) + 6 * sh;

      ctx.fillStyle = "rgba(0,0,0,0.12)";
      const boxH = proText ? fontSize + microSize + 7 * sh : fontSize + 4 * sh;
      fillRoundRect(
        ctx,
        tx - boxInnerW / 2,
        ty - boxH / 2,
        boxInnerW,
        boxH,
        2 * sh,
      );
      ctx.fillStyle = "rgba(255,255,255,0.68)";
      ctx.fillText(text, tx, ty - (proText ? microSize * 0.32 : 0));
      if (proText) {
        ctx.font = `400 ${microSize}px ui-sans-serif, system-ui, sans-serif`;
        ctx.fillStyle = "rgba(219,39,119,0.72)";
        ctx.fillText(proText, tx, ty + fontSize * 0.42);
        ctx.font = `500 ${fontSize}px ui-sans-serif, system-ui, sans-serif`;
      }
    }
    ctx.restore();

    const metrics: { key: string; labelZh: string; labelEn: string; value: number | undefined }[] = [
      { key: "x_factor", labelZh: "X因子", labelEn: "X-Factor", value: ang.x_factor },
      { key: "spine_tilt", labelZh: "脊柱", labelEn: "Spine", value: ang.spine_tilt },
      { key: "shoulder_rotation", labelZh: "肩旋转", labelEn: "Shoulder", value: ang.shoulder_rotation },
      { key: "hip_rotation", labelZh: "髋旋转", labelEn: "Hip", value: ang.hip_rotation },
    ];
    const validMetrics = metrics.filter((m) => m.value != null);
    if (validMetrics.length > 0) {
      const mFontSize = Math.max(6.5, 7 * sh);
      const proFont = Math.max(5.5, 6 * sh);
      const panelW = Math.max(82, 96 * sh);
      const lineH = Math.max(16, mFontSize + proFont + 3 * sh);
      const panelH = validMetrics.length * lineH + 6 * sh;
      const panelX = cW - panelW - 4;
      const panelY = phaseTopH > 0 ? phaseTopH + 2 * sh : 6;

      ctx.save();
      ctx.fillStyle = "rgba(0,0,0,0.08)";
      fillRoundRect(ctx, panelX, panelY, panelW, panelH, 4 * sh);

      ctx.textBaseline = "top";

      for (let i = 0; i < validMetrics.length; i++) {
        const m = validMetrics[i];
        const y = panelY + 3 * sh + i * lineH;
        const label = lang === "zh" ? m.labelZh : m.labelEn;
        const pref = PRO_REF_LINES[m.key];

        ctx.font = `500 ${mFontSize}px ui-sans-serif, system-ui, sans-serif`;
        ctx.fillStyle = "rgba(255,255,255,0.55)";
        ctx.textAlign = "left";
        ctx.fillText(label, panelX + 4, y);

        ctx.fillStyle = "rgba(255,255,255,0.78)";
        ctx.textAlign = "right";
        ctx.fillText(`${Math.round(m.value!)}°`, panelX + panelW - 4, y);

        if (pref) {
          ctx.font = `400 ${proFont}px ui-sans-serif, system-ui, sans-serif`;
          ctx.fillStyle = "rgba(219,39,119,0.62)";
          ctx.textAlign = "left";
          const ptxt = lang === "zh" ? pref.zh : pref.en;
          ctx.fillText(ptxt, panelX + 4, y + mFontSize + 0.5);
        }
      }
      ctx.restore();
    }
  }

  /* ─── Trajectory: starts from ball position (pre-computed from impact frame) ─── */
  if (opts.prediction && trajectoryDisplayYards(opts.prediction) > 0 && opts.impactTime != null) {
    const elapsed = currentTime - opts.impactTime;
    if (elapsed > 0) {
      // Trajectory persists after animation completes (PGA broadcast style)
      const progress = Math.min(1, elapsed / TRAJECTORY_ANIM_SEC);
      let ballOrigin: TrajectoryOrigin | null = null;
      if (opts.ballOriginNorm) {
        const [bx, by] = px(opts.ballOriginNorm.nx, opts.ballOriginNorm.ny);
        ballOrigin = { x: bx, y: by };
      }
      drawTrajectory(ctx, cW, cH, opts.prediction, lang, sh, progress, ballOrigin);
    }
  }

  /* ─── Timeline + sweet spot (bottom) ─── */
  const tlH = Math.max(3, 4 * sh);
  const tlY = cH - tlH - Math.max(4, 5 * sh);
  const tlPad = 6;
  const tlW = cW - tlPad * 2;
  const tlX = tlPad;

  if (duration > 0.05) {
    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,0.18)";
    fillRoundRect(ctx, tlX - 2, tlY - 2, tlW + 4, tlH + 4, 2 * sh);

    ctx.fillStyle = "rgba(255,255,255,0.06)";
    ctx.fillRect(tlX, tlY, tlW, tlH);

    if (sweetSpot) {
      const x0 = tlX + (sweetSpot.start / duration) * tlW;
      const x1 = tlX + (Math.min(sweetSpot.end, duration) / duration) * tlW;
      ctx.fillStyle = "rgba(245, 197, 24, 0.22)";
      ctx.fillRect(x0, tlY, Math.max(2, x1 - x0), tlH);

      const cx = tlX + (sweetSpot.center / duration) * tlW;
      ctx.fillStyle = "#dc2626";
      ctx.shadowColor = "rgba(255, 40, 40, 0.85)";
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.moveTo(cx, tlY - 1.5 * sh);
      ctx.lineTo(cx - 3 * sh, tlY + tlH + 1.5 * sh);
      ctx.lineTo(cx + 3 * sh, tlY + tlH + 1.5 * sh);
      ctx.closePath();
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    const playX = tlX + (currentTime / duration) * tlW;
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.fillRect(Math.min(tlX + tlW - 2, Math.max(tlX, playX - 1)), tlY, 2, tlH);

    const sweetLabel = lang === "zh" ? "甜蜜点区" : "Sweet spot";
    const sweetFont = Math.max(7, 7.5 * sh);
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    const lx = tlX;
    const ly = tlY - 1 * sh;
    ctx.font = `500 ${sweetFont}px ui-sans-serif, system-ui, -apple-system, sans-serif`;
    ctx.fillStyle = "rgba(250, 204, 21, 0.65)";
    ctx.fillText(sweetLabel, lx, ly);
    ctx.restore();
  }
}

/* ═══════════════ Component ═══════════════ */

export default function VideoAnalysisOverlay({
  videoSrc,
  poseFrames = [],
  lang,
  coachingTips,
  prediction,
  sourceFrameCount,
  skeletonStyle = "legacy",
}: Props) {
  const rootRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoShellRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(-999);
  const forceRedrawRef = useRef(true);
  const [immersive, setImmersive] = useState(false);
  const immersiveRef = useRef(false);
  const [isFsApi, setIsFsApi] = useState(false);
  const [uiTick, setUiTick] = useState(0);
  const [scrubTime, setScrubTime] = useState<number | null>(null);
  const [videoDuration, setVideoDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [playbackRate, setPlaybackRateState] = useState(1);
  const [showTips, setShowTips] = useState(false);
  const [exportBusy, setExportBusy] = useState(false);
  const exportBusyRef = useRef(false);
  /** Stops <video> from retrying a dead remote URL (e.g. expired Modal /pro-v3/media). */
  const [videoLoadFailed, setVideoLoadFailed] = useState(false);

  useEffect(() => {
    setVideoLoadFailed(false);
  }, [videoSrc]);

  const displayExpanded = immersive || isFsApi;

  const setImmersiveBoth = useCallback((v: boolean) => {
    immersiveRef.current = v;
    setImmersive(v);
    document.body.style.overflow = v ? "hidden" : "";
  }, []);

  const exitExpanded = useCallback(() => {
    setImmersiveBoth(false);
    if (document.fullscreenElement) {
      void document.exitFullscreen?.().catch(() => {});
    }
  }, [setImmersiveBoth]);

  const formatClock = (sec: number) => {
    if (!Number.isFinite(sec) || sec < 0) return "0:00";
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  const [showSkeleton, setShowSkeleton] = useState(true);
  const [showAngles, setShowAngles] = useState(true);
  const [showPhase, setShowPhase] = useState(true);
  /** Plus style only: plumb line + meteor arcs (independent from bone overlay when skeletonStyle is plus). */
  const [showPlusGuideLines, setShowPlusGuideLines] = useState(true);

  const flagsRef = useRef({ skeleton: true, angles: true, phase: true, guideLines: true });
  flagsRef.current = {
    skeleton: showSkeleton,
    angles: showAngles,
    phase: showPhase,
    guideLines: skeletonStyle === "plus" ? showPlusGuideLines : showSkeleton,
  };

  useEffect(() => {
    forceRedrawRef.current = true;
    setUiTick((n) => n + 1);
  }, [showSkeleton, showAngles, showPhase, showPlusGuideLines]);

  const poseRef = useRef(poseFrames);
  poseRef.current = poseFrames;

  const predictionRef = useRef(prediction);
  predictionRef.current = prediction;

  const impactFrameIndex = useMemo(() => {
    if (!poseFrames || poseFrames.length === 0) return null;

    const impactFrames = poseFrames.filter((f) => f.phase_data?.phase_id === "impact");
    if (impactFrames.length > 0) {
      return impactFrames[0].frame_index;
    }

    const firstFollow = poseFrames.find((f) => f.phase_data?.phase_id === "follow_through");
    const lastDownswing = [...poseFrames].reverse().find((f) => f.phase_data?.phase_id === "downswing");
    if (lastDownswing && firstFollow) {
      return Math.round((lastDownswing.frame_index + firstFollow.frame_index) / 2);
    }
    if (firstFollow) return firstFollow.frame_index;
    if (lastDownswing) return lastDownswing.frame_index;

    const wristData: { fi: number; ts: number; y: number }[] = [];
    for (const f of poseFrames) {
      const lw = f.joints?.find((j) => j.name === "left_wrist");
      const rw = f.joints?.find((j) => j.name === "right_wrist");
      if (lw || rw) {
        const y = lw && rw ? (lw.normalized.y + rw.normalized.y) / 2
          : (lw ?? rw)!.normalized.y;
        wristData.push({ fi: f.frame_index, ts: f.timestamp, y });
      }
    }
    if (wristData.length < 5) return null;

    wristData.sort((a, b) => a.fi - b.fi);

    let topIdx = 0;
    let topY = 999;
    const searchEnd = Math.floor(wristData.length * 0.65);
    for (let i = 0; i < searchEnd; i++) {
      if (wristData[i].y < topY) { topY = wristData[i].y; topIdx = i; }
    }

    let maxVel = -Infinity;
    let impactIdx = topIdx + 1;
    for (let i = topIdx + 1; i < wristData.length - 1; i++) {
      const dt = wristData[i + 1].ts - wristData[i].ts;
      if (dt <= 0) continue;
      const vel = (wristData[i + 1].y - wristData[i].y) / dt;
      if (vel > maxVel) {
        maxVel = vel;
        impactIdx = i;
      }
    }

    return wristData[Math.min(impactIdx + 1, wristData.length - 1)]?.fi ?? null;
  }, [poseFrames]);

  /** Impact moment on the same time axis as HTML5 video.currentTime */
  const impactMediaTime = useMemo(() => {
    if (impactFrameIndex == null || !poseFrames.length) return null;
    return mediaSecondsAtFrameIndex(
      poseFrames,
      impactFrameIndex,
      videoDuration,
      sourceFrameCount,
    );
  }, [impactFrameIndex, poseFrames, videoDuration, sourceFrameCount]);

  const sweetSpot = useMemo(() => {
    if (impactMediaTime == null) return null;
    const span = 0.12;
    const cap = videoDuration > 0.05 ? videoDuration : impactMediaTime + span;
    return {
      center: impactMediaTime,
      start: Math.max(0, impactMediaTime - span / 2),
      end: Math.min(cap, impactMediaTime + span / 2),
    };
  }, [impactMediaTime, videoDuration]);

  const impactTimeRef = useRef<number | null>(null);
  impactTimeRef.current = impactMediaTime;

  const ballOriginNorm = useMemo(() => {
    // Try impact frames first, fallback to any frame near impact
    const impactFrames = poseFrames.filter(
      (f) => f.phase_data?.phase_id === "impact" && f.joints?.length,
    );
    // Also consider follow_through frames as fallback
    const candidateFrames = impactFrames.length
      ? impactFrames
      : poseFrames.filter(
          (f) =>
            (f.phase_data?.phase_id === "follow_through" || f.phase_data?.phase_id === "downswing") &&
            f.joints?.length,
        );
    if (!candidateFrames.length) return null;

    // Use the frame closest to peak impact (middle of impact frames)
    const frame = candidateFrames[Math.floor(candidateFrames.length / 2)];
    const js = frame.joints;

    // Ground Y: max Y of ankles (lowest point = ground)
    const la = js.find((j) => j.name === "left_ankle");
    const ra = js.find((j) => j.name === "right_ankle");
    let groundY = 0.85;
    if (la && ra) groundY = Math.max(la.normalized.y, ra.normalized.y);
    else if (la || ra) groundY = (la || ra)!.normalized.y;

    // Ball X: start from the impact-side wrist, then extend AWAY from body center.
    // This prevents the origin from flipping behind the player.
    const lw = js.find((j) => j.name === "left_wrist");
    const rw = js.find((j) => j.name === "right_wrist");
    const lh = js.find((j) => j.name === "left_hip");
    const rh = js.find((j) => j.name === "right_hip");
    const ls = js.find((j) => j.name === "left_shoulder");
    const rs = js.find((j) => j.name === "right_shoulder");

    const bodyCenterX = lh && rh
      ? (lh.normalized.x + rh.normalized.x) / 2
      : ls && rs
        ? (ls.normalized.x + rs.normalized.x) / 2
        : 0.5;

    let wristX = (lw || rw)?.normalized.x ?? bodyCenterX;
    if (lw && rw) {
      // Pick the wrist that is farther from body center (impact/club side).
      wristX = Math.abs(lw.normalized.x - bodyCenterX) >= Math.abs(rw.normalized.x - bodyCenterX)
        ? lw.normalized.x
        : rw.normalized.x;
    }

    // Wrist Y (how high are the hands above ground in normalized units)
    const wristY = lw && rw
      ? (lw.normalized.y + rw.normalized.y) / 2
      : (lw || rw)?.normalized.y ?? groundY * 0.7;
    const handHeightAboveGround = groundY - wristY; // normalized height

    // Direction is always from body center -> wrist side (outside), never behind body.
    const dir = wristX >= bodyCenterX ? 1 : -1;

    // Club-length factor (driver longer, iron medium, wedge shorter).
    // If speed exists, use it as soft proxy of club length.
    const speed = Number(prediction?.club_head_speed || 0);
    const speedFactor = speed > 0
      ? Math.max(0.8, Math.min(1.25, 0.85 + (speed - 30) / 90))
      : 1.0;
    const clubExtX = handHeightAboveGround * 1.0 * speedFactor * dir;
    const fixedOffsetX = 0.08 * dir;

    const ballX = wristX + clubExtX + fixedOffsetX;

    return {
      nx: Math.max(0.05, Math.min(0.95, ballX)),
      ny: Math.min(0.97, groundY),
    };
  }, [poseFrames, prediction?.club_head_speed]);

  const ballOriginNormRef = useRef(ballOriginNorm);
  ballOriginNormRef.current = ballOriginNorm;

  const hasTips =
    coachingTips &&
    (coachingTips.postureZh ||
      coachingTips.postureEn ||
      coachingTips.trainingZh ||
      coachingTips.trainingEn);

  const postureText =
    lang === "zh"
      ? coachingTips?.postureZh || coachingTips?.postureEn
      : coachingTips?.postureEn || coachingTips?.postureZh;
  const trainingText =
    lang === "zh"
      ? coachingTips?.trainingZh || coachingTips?.trainingEn
      : coachingTips?.trainingEn || coachingTips?.trainingZh;

  useEffect(() => {
    const el = videoShellRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => {
      forceRedrawRef.current = true;
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const onFs = () => {
      const fs = !!document.fullscreenElement;
      setIsFsApi(fs);
      if (fs) {
        immersiveRef.current = false;
        setImmersive(false);
        document.body.style.overflow = "";
      }
    };
    document.addEventListener("fullscreenchange", onFs);
    return () => document.removeEventListener("fullscreenchange", onFs);
  }, []);

  useEffect(() => {
    return () => {
      document.body.style.overflow = "";
    };
  }, []);

  useEffect(() => {
    const vid = videoRef.current;
    if (vid) setPlaying(!vid.paused);
  }, [videoSrc]);

  useEffect(() => {
    setPlaybackRateState(1);
  }, [videoSrc]);

  useEffect(() => {
    const vid = videoRef.current;
    if (vid) vid.playbackRate = playbackRate;
  }, [playbackRate, videoSrc]);

  const setPlaybackRate = useCallback((r: number) => {
    setPlaybackRateState(r);
    const vid = videoRef.current;
    if (vid) vid.playbackRate = r;
    setUiTick((n) => n + 1);
  }, []);

  const isMobileRef = useRef(false);
  useEffect(() => {
    isMobileRef.current =
      typeof window !== "undefined" &&
      (/iPhone|iPad|iPod|Android/i.test(navigator.userAgent) ||
        ("ontouchstart" in window && window.innerWidth < 1024));
  }, []);

  const enterExpanded = useCallback(() => {
    if (immersiveRef.current || document.fullscreenElement) return;
    if (isMobileRef.current) {
      setImmersiveBoth(true);
      return;
    }
    const el = rootRef.current;
    if (!el) return;
    void (async () => {
      try {
        if (el.requestFullscreen) {
          await el.requestFullscreen();
        } else {
          const wk = (
            el as HTMLElement & { webkitRequestFullscreen?: () => void }
          ).webkitRequestFullscreen;
          if (typeof wk === "function") wk.call(el);
          else throw new Error("no fullscreen");
        }
      } catch {
        setImmersiveBoth(true);
      }
    })();
  }, [setImmersiveBoth]);

  const downloadOriginal = useCallback(async () => {
    try {
      const res = await fetch(videoSrc);
      const blob = await res.blob();
      const ext =
        blob.type.includes("mp4") ? "mp4" : blob.type.includes("webm") ? "webm" : "bin";
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `stellar-swing-${Date.now()}.${ext}`;
      a.rel = "noopener";
      a.click();
      URL.revokeObjectURL(a.href);
    } catch {
      const a = document.createElement("a");
      a.href = videoSrc;
      a.download = `stellar-swing-${Date.now()}.mp4`;
      a.target = "_blank";
      a.rel = "noopener";
      a.click();
    }
  }, [videoSrc]);

  const recordWithOverlay = useCallback(async () => {
    const v = videoRef.current;
    if (!v || exportBusyRef.current || !poseRef.current.length) return;
    const vw = v.videoWidth;
    const vh = v.videoHeight;
    if (!vw || !vh) {
      window.alert(
        lang === "zh" ? "请等待视频加载完成后再导出" : "Wait for video to load",
      );
      return;
    }
    const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
      ? "video/webm;codecs=vp9"
      : MediaRecorder.isTypeSupported("video/webm")
        ? "video/webm"
        : "";
    if (!mime || typeof MediaRecorder === "undefined") {
      window.alert(
        lang === "zh"
          ? "当前浏览器无法录制带叠加视频，请使用「下载原视频」"
          : "Recording not supported — use Download",
      );
      return;
    }

    exportBusyRef.current = true;
    setExportBusy(true);

    const wasPaused = v.paused;
    const prevTime = v.currentTime;

    try {
      const maxW = 720;
      let cw = vw;
      let ch = vh;
      if (cw > maxW) {
        ch = Math.round((ch * maxW) / cw);
        cw = maxW;
      }

      const exportCanvas = document.createElement("canvas");
      exportCanvas.width = cw;
      exportCanvas.height = ch;
      const xctx = exportCanvas.getContext("2d");
      if (!xctx) return;

      const stream = exportCanvas.captureStream(30);
      const rec = new MediaRecorder(stream, {
        mimeType: mime,
        videoBitsPerSecond: 2_500_000,
      });
      const chunks: Blob[] = [];
      rec.ondataavailable = (e) => {
        if (e.data.size) chunks.push(e.data);
      };

      v.pause();
      v.currentTime = 0;
      await new Promise<void>((resolve) => {
        const onSeeked = () => {
          v.removeEventListener("seeked", onSeeked);
          resolve();
        };
        v.addEventListener("seeked", onSeeked);
      });

      const stopped = new Promise<void>((resolve) => {
        rec.onstop = () => resolve();
      });

      rec.start(250);
      try {
        await v.play();
      } catch {
        window.alert(
          lang === "zh" ? "无法播放视频，无法导出叠加" : "Playback blocked",
        );
        if (rec.state === "recording") rec.stop();
        await stopped;
        return;
      }

      const safeStopRec = () => {
        if (rec.state === "recording") rec.stop();
      };
      const step = () => {
        try {
          xctx.drawImage(v, 0, 0, cw, ch);
          const recFrames = poseRef.current;
          const exportDur = v.duration || 0;
          const pose = findNearestPose(
            recFrames,
            v.currentTime,
            exportDur,
            sourceFrameCount,
          );
          if (pose) {
            const swingTrail = computeSwingTrail(
              recFrames,
              v.currentTime,
              exportDur,
              sourceFrameCount,
            );
            drawOverlay(xctx, cw, ch, pose, flagsRef.current, lang, {
              duration: v.duration || 0,
              currentTime: v.currentTime,
              sweetSpot,
              compositeSkipClear: true,
              prediction: predictionRef.current,
              impactTime: impactTimeRef.current ?? undefined,
              ballOriginNorm: ballOriginNormRef.current,
              swingTrail,
              videoFrameSize:
                (v.videoWidth > 0 && v.videoHeight > 0)
                  ? { width: v.videoWidth, height: v.videoHeight }
                  : null,
              skeletonStyle,
            });
          }
        } catch {
          v.pause();
          safeStopRec();
          return;
        }
        const dur = v.duration || 0;
        if (v.ended || (dur > 0 && v.currentTime >= dur - 0.04)) {
          v.pause();
          safeStopRec();
          return;
        }
        requestAnimationFrame(step);
      };
      requestAnimationFrame(step);

      await stopped;

      const outType = mime.split(";")[0] || "video/webm";
      const blob = new Blob(chunks, { type: outType });
      if (blob.size > 0) {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `stellar-swing-with-overlay-${Date.now()}.webm`;
        a.rel = "noopener";
        a.click();
        URL.revokeObjectURL(a.href);
      } else {
        window.alert(
          lang === "zh"
            ? "导出失败（可能被跨域限制），请用下载原视频"
            : "Export failed (try Download)",
        );
      }
    } catch {
      window.alert(
        lang === "zh" ? "导出出错，请重试或下载原视频" : "Export error",
      );
    } finally {
      v.pause();
      v.currentTime = prevTime;
      if (!wasPaused) void v.play().catch(() => {});
      exportBusyRef.current = false;
      setExportBusy(false);
    }
  }, [lang, sweetSpot, sourceFrameCount, skeletonStyle]);

  const tick = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const container = videoShellRef.current;
    if (!video || !canvas || !container) {
      rafRef.current = requestAnimationFrame(tick);
      return;
    }

    const time = video.currentTime;
    const rawDpr = window.devicePixelRatio || 1;
    const dpr = Math.min(rawDpr, 2);
    const cW = container.offsetWidth;
    const cH = container.offsetHeight;

    const dur = Number.isFinite(video.duration) && video.duration > 0
      ? video.duration
      : 0;

    const sizeChanged =
      canvas.width !== Math.round(cW * dpr) ||
      canvas.height !== Math.round(cH * dpr);
    const timeChanged = Math.abs(time - lastTimeRef.current) > 0.008;
    const forced = forceRedrawRef.current;
    const trajectoryAnimating =
      impactTimeRef.current != null &&
      predictionRef.current &&
      trajectoryDisplayYards(predictionRef.current) > 0 &&
      time >= impactTimeRef.current;

    if (!sizeChanged && !timeChanged && !forced && !trajectoryAnimating) {
      rafRef.current = requestAnimationFrame(tick);
      return;
    }
    forceRedrawRef.current = false;

    if (sizeChanged || forced) {
      canvas.width = Math.round(cW * dpr);
      canvas.height = Math.round(cH * dpr);
      canvas.style.width = `${cW}px`;
      canvas.style.height = `${cH}px`;
    }
    lastTimeRef.current = time;

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      rafRef.current = requestAnimationFrame(tick);
      return;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const frames = poseRef.current;
    const pose =
      frames.length > 0
        ? findNearestPose(frames, time, dur || undefined, sourceFrameCount)
        : null;
    const swingTrail =
      frames.length > 0
        ? computeSwingTrail(frames, time, dur || undefined, sourceFrameCount)
        : null;

    drawOverlay(ctx, cW, cH, pose, flagsRef.current, lang, {
      duration: dur,
      currentTime: time,
      sweetSpot,
      prediction: predictionRef.current,
      impactTime: impactTimeRef.current ?? undefined,
      ballOriginNorm: ballOriginNormRef.current,
      swingTrail,
      videoFrameSize:
        (video.videoWidth > 0 && video.videoHeight > 0)
          ? { width: video.videoWidth, height: video.videoHeight }
          : null,
      skeletonStyle,
    });

    rafRef.current = requestAnimationFrame(tick);
  }, [lang, sweetSpot, sourceFrameCount, skeletonStyle]);

  useEffect(() => {
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [tick]);

  const onVideoMeta = useCallback(() => {
    const v = videoRef.current;
    if (v && Number.isFinite(v.duration) && v.duration > 0) {
      setVideoDuration(v.duration);
    }
    if (v) v.playbackRate = playbackRate;
    forceRedrawRef.current = true;
  }, [playbackRate]);

  const v = videoRef.current;
  const tNow = scrubTime ?? v?.currentTime ?? 0;
  const dur =
    (Number.isFinite(videoDuration) && videoDuration > 0
      ? videoDuration
      : v?.duration) || 0;
  void uiTick;

  return (
    <div
      ref={rootRef}
      className={
        displayExpanded
          ? "fixed inset-0 z-[2147483000] flex h-[100dvh] max-h-[100dvh] flex-col overflow-hidden bg-[#050508] shadow-none"
          : "glass-card flex flex-col overflow-hidden rounded-xl border border-white/10"
      }
    >
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-white/5 px-2 py-1.5">
        <div className="flex min-w-0 flex-col gap-0.5">
          <span className="text-[10px] font-medium text-white/45">
            {lang === "zh" ? "视频技术分析" : "Video Technical Analysis"}
          </span>
          {poseFrames.length === 0 ? (
            <span className="text-[9px] leading-tight text-amber-200/40">
              {lang === "zh"
                ? "暂无骨架帧，可照常播放原视频"
                : "No pose frames — original video only"}
            </span>
          ) : null}
        </div>
        <div className="flex flex-wrap items-center justify-end gap-1">
          <button
            type="button"
            onClick={() => void downloadOriginal()}
            disabled={exportBusy}
            className="rounded-md px-1.5 py-0.5 text-[9px] font-medium bg-white/[0.06] text-white/45 border border-white/10 hover:bg-white/10 disabled:opacity-40"
          >
            {lang === "zh" ? "原视频" : "Video"}
          </button>
          <button
            type="button"
            onClick={() => void recordWithOverlay()}
            disabled={exportBusy}
            className="rounded-md px-1.5 py-0.5 text-[9px] font-medium bg-amber-500/10 text-amber-200/55 border border-amber-500/20 hover:bg-amber-500/15 disabled:opacity-40"
          >
            {lang === "zh" ? "带叠加" : "Overlay"}
          </button>
          {hasTips ? (
            <button
              type="button"
              onClick={() => setShowTips((x) => !x)}
              className="rounded-md px-1.5 py-0.5 text-[9px] font-medium bg-amber-500/12 text-amber-200/55 border border-amber-500/18"
            >
              {lang === "zh" ? "建议" : "Tips"}
            </button>
          ) : null}
          <button
            type="button"
            disabled={poseFrames.length === 0}
            onClick={() => setShowSkeleton((v) => !v)}
            className={`rounded-md px-1.5 py-0.5 text-[9px] font-medium transition disabled:pointer-events-none disabled:opacity-35 ${showSkeleton ? "bg-purple-500/15 text-purple-300/55 border border-purple-500/22" : "bg-white/[0.04] text-white/25 border border-transparent"}`}
          >
            {lang === "zh" ? "骨架" : "Skel"}
          </button>
          {skeletonStyle === "plus" && poseFrames.length > 0 ? (
            <button
              type="button"
              onClick={() => setShowPlusGuideLines((v) => !v)}
              className={`rounded-md px-1.5 py-0.5 text-[9px] font-medium transition ${showPlusGuideLines ? "bg-cyan-500/15 text-cyan-200/55 border border-cyan-500/22" : "bg-white/[0.04] text-white/25 border border-transparent"}`}
            >
              {lang === "zh" ? "辅助线" : "Guides"}
            </button>
          ) : null}
          <button
            type="button"
            disabled={poseFrames.length === 0}
            onClick={() => setShowAngles((v) => !v)}
            className={`rounded-md px-1.5 py-0.5 text-[9px] font-medium transition disabled:pointer-events-none disabled:opacity-35 ${showAngles ? "bg-blue-500/15 text-blue-300/55 border border-blue-500/22" : "bg-white/[0.04] text-white/25 border border-transparent"}`}
          >
            {lang === "zh" ? "角度" : "Angles"}
          </button>
          <button
            type="button"
            disabled={poseFrames.length === 0}
            onClick={() => setShowPhase((v) => !v)}
            className={`rounded-md px-1.5 py-0.5 text-[9px] font-medium transition disabled:pointer-events-none disabled:opacity-35 ${showPhase ? "bg-emerald-500/15 text-emerald-300/55 border border-emerald-500/22" : "bg-white/[0.04] text-white/25 border border-transparent"}`}
          >
            {lang === "zh" ? "阶段" : "Phase"}
          </button>
        </div>
      </div>

      {showTips && hasTips ? (
        <div className="shrink-0 border-b border-white/5 bg-black/25 px-2.5 py-2 space-y-1.5 animate-fade-in">
          <p className="text-[9px] uppercase tracking-wide text-amber-200/35">
            {lang === "zh" ? "改姿势建议" : "Posture fix"}
          </p>
          <p className="text-[10px] text-white/38 leading-snug">
            {postureText || (lang === "zh" ? "暂无" : "—")}
          </p>
          <p className="text-[9px] uppercase tracking-wide text-emerald-200/35 pt-0.5">
            {lang === "zh" ? "训练建议" : "Training"}
          </p>
          <p className="text-[10px] text-white/32 leading-snug">
            {trainingText || (lang === "zh" ? "暂无" : "—")}
          </p>
          <button
            type="button"
            onClick={() => setShowTips(false)}
            className="text-[9px] text-white/28 hover:text-white/45"
          >
            {lang === "zh" ? "收起" : "Close"}
          </button>
        </div>
      ) : null}

      <div
        className={`flex min-h-0 w-full flex-col ${displayExpanded ? "min-h-0 flex-1" : ""}`}
      >
        <div
          ref={videoShellRef}
          className={`relative w-full bg-black isolate ${displayExpanded ? "min-h-0 flex-1" : ""}`}
          style={
            displayExpanded
              ? { minHeight: 0 }
              : {
                  aspectRatio: "9 / 16",
                  maxHeight: "min(92vh, 920px)",
                  minHeight: "min(52vh, 560px)",
                }
          }
        >
          {videoLoadFailed ? (
            <div className="absolute inset-0 z-0 flex flex-col items-center justify-center gap-2 bg-black px-4 text-center">
              <p className="text-xs text-white/55 leading-relaxed">
                {lang === "zh"
                  ? "视频无法加载（链接可能已失效或未迁移到持久存储）。请重新分析或使用历史中的「重新分析」。"
                  : "Video failed to load (URL may be expired or not on durable storage). Re-analyze or use Re-analyze from history."}
              </p>
            </div>
          ) : (
            <video
              ref={videoRef}
              src={videoSrc}
              controls={false}
              playsInline
              preload="auto"
              className="absolute inset-0 z-0 h-full w-full object-contain [touch-action:manipulation]"
              onClick={() => {
                const vid = videoRef.current;
                if (!vid) return;
                if (vid.paused) void vid.play().catch(() => {});
                else vid.pause();
              }}
              onError={() => setVideoLoadFailed(true)}
              onLoadedMetadata={onVideoMeta}
              onDurationChange={onVideoMeta}
              onTimeUpdate={() => setUiTick((n) => n + 1)}
              onPlay={() => {
                setPlaying(true);
                setUiTick((n) => n + 1);
              }}
              onPause={() => {
                setPlaying(false);
                setUiTick((n) => n + 1);
              }}
              onEnded={() => {
                setPlaying(false);
                setUiTick((n) => n + 1);
              }}
            />
          )}
          <canvas
            ref={canvasRef}
            className="absolute inset-0 z-[1] h-full w-full pointer-events-none touch-none"
            aria-hidden
          />
          {exportBusy ? (
            <div className="absolute inset-0 z-[2] flex flex-col items-center justify-center gap-2 bg-black/50 pointer-events-none">
              <div className="h-6 w-6 animate-spin rounded-full border-2 border-white/15 border-t-white/50" />
              <p className="px-4 text-center text-[10px] text-white/45">
                {lang === "zh"
                  ? "正在录制带叠加视频，请稍候…"
                  : "Recording overlay video…"}
              </p>
            </div>
          ) : null}
        </div>

        <div className="flex shrink-0 flex-col gap-1.5 border-t border-white/[0.08] bg-black/90 px-2 py-2">
          <input
            type="range"
            aria-label={lang === "zh" ? "进度" : "Seek"}
            min={0}
            max={Math.max(0.001, dur || 0.001)}
            step={0.02}
            value={Math.min(tNow, dur || 0)}
            disabled={!dur}
            onChange={(e) => {
              const nt = parseFloat(e.target.value);
              setScrubTime(nt);
              const vid = videoRef.current;
              if (vid) vid.currentTime = nt;
            }}
            onMouseDown={() => setScrubTime(videoRef.current?.currentTime ?? 0)}
            onTouchStart={() => setScrubTime(videoRef.current?.currentTime ?? 0)}
            onMouseUp={() => setScrubTime(null)}
            onTouchEnd={() => setScrubTime(null)}
            className="h-1 w-full cursor-pointer accent-white/50"
          />
          <div className="flex items-center justify-center gap-4">
            <button
              type="button"
              onClick={() => {
                const vid = videoRef.current;
                if (!vid) return;
                if (vid.paused) void vid.play().catch(() => {});
                else vid.pause();
                setUiTick((n) => n + 1);
              }}
              className="rounded-full border border-white/20 bg-white/[0.08] px-4 py-2 text-[11px] font-medium text-white/75 active:scale-95"
            >
              {playing
                ? lang === "zh"
                  ? "暂停"
                  : "Pause"
                : lang === "zh"
                  ? "播放"
                  : "Play"}
            </button>
            <span className="text-[10px] tabular-nums text-white/50">
              {formatClock(tNow)} / {formatClock(dur)}
            </span>
          </div>
          <div className="flex flex-wrap items-center justify-center gap-1">
            {PLAYBACK_RATES.map((r) => (
              <button
                key={r}
                type="button"
                onClick={() => setPlaybackRate(r)}
                className={`rounded-md px-2 py-1 text-[10px] font-medium tabular-nums transition ${
                  Math.abs(playbackRate - r) < 0.01
                    ? "border border-amber-400/40 bg-amber-500/20 text-amber-100"
                    : "border border-white/10 bg-white/[0.05] text-white/45 hover:bg-white/10"
                }`}
              >
                {r === 1 ? "1×" : `${r}×`}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap items-center justify-center gap-2 pt-0.5">
            {!displayExpanded ? (
              <button
                type="button"
                onClick={() => void enterExpanded()}
                className="rounded-lg border border-white/15 bg-white/[0.08] px-4 py-2 text-[11px] font-medium text-white/70 hover:bg-white/12"
              >
                {lang === "zh" ? "放大全屏" : "Fullscreen"}
              </button>
            ) : (
              <button
                type="button"
                onClick={exitExpanded}
                className="rounded-lg border border-red-400/35 bg-red-500/15 px-5 py-2 text-[11px] font-semibold text-red-200/90 hover:bg-red-500/25"
              >
                {lang === "zh" ? "退出全屏" : "Exit fullscreen"}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
