"use client";

import { useState } from "react";
import { keyframeImageDataUrl } from "@/lib/image-base64";
import { resolveProv3ProductMediaUrl } from "@/lib/prov3-media-url";

export interface PoseSnapshotJoint {
  name: string;
  nx: number;
  ny: number;
  v: number;
}

export interface PoseSnapshot {
  joints: PoseSnapshotJoint[];
  connections: number[][];
}

interface Keyframe {
  phase: string;
  label_en: string;
  label_zh: string;
  timestamp: number;
  /** Pro v3: true-240 analysis timeline JPG URL (preferred over base64). */
  keyframe_image_url?: string;
  /** Optional: compact D1 rows or R2 merge failures may omit images. */
  image_base64?: string;
  pose_snapshot?: PoseSnapshot | null;
  skeleton_overlay?: string;
}

function KeyframeStripMedia({
  kf,
  uniqueId,
  className,
  enlarged,
  lang,
  showSkeleton,
  proMode,
  urlOnlyTimeline,
}: {
  kf: Keyframe;
  uniqueId: string;
  className?: string;
  enlarged?: boolean;
  lang: "en" | "zh";
  showSkeleton: boolean;
  proMode: boolean;
  /** Pro v3 history: only show timeline JPG URLs — no base64 fallback */
  urlOnlyTimeline?: boolean;
}) {
  const [imgBroken, setImgBroken] = useState(false);
  const showSkel = showSkeleton && !!kf.pose_snapshot?.joints?.length;
  const url = resolveProv3ProductMediaUrl(String(kf.keyframe_image_url ?? "").trim());
  const b64Url = urlOnlyTimeline ? null : keyframeImageDataUrl(kf.image_base64);
  const imgSrc = urlOnlyTimeline ? (url || null) : url || b64Url;
  if (imgSrc && !imgBroken) {
    return (
      <div className={`relative ${enlarged ? "min-h-[65vh] w-full" : ""} ${className ?? ""}`}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          key={imgSrc}
          src={imgSrc}
          alt={kf.phase}
          className={
            enlarged
              ? "w-full h-full object-contain absolute inset-0"
              : "w-full object-contain transition-transform group-hover:scale-105"
          }
          onError={() => setImgBroken(true)}
        />
        {showSkel ? (
          <KeyframeSkeletonSvg snap={kf.pose_snapshot!} proMode={proMode} uniqueId={uniqueId} />
        ) : showSkeleton && !enlarged ? (
          <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-brand-purple/15 via-transparent to-transparent" />
        ) : null}
      </div>
    );
  }
  if (showSkel && !urlOnlyTimeline) {
    return (
      <div
        className={`relative flex items-center justify-center bg-black/50 ${className ?? ""}`}
        style={{ minHeight: enlarged ? "65vh" : 112 }}
      >
        <KeyframeSkeletonSvg snap={kf.pose_snapshot!} proMode={proMode} uniqueId={uniqueId} />
        <span className="pointer-events-none absolute bottom-2 left-2 right-2 text-center text-[9px] text-white/35">
          {lang === "en" ? "Pose only" : "仅骨架"}
        </span>
      </div>
    );
  }
  return (
    <div
      className={`flex flex-col items-center justify-center gap-1 bg-black/40 px-2 text-center ${className ?? ""}`}
      style={{ minHeight: enlarged ? "65vh" : 112 }}
    >
      <span className="text-[10px] text-white/30">
        {urlOnlyTimeline
          ? lang === "en"
            ? "Timeline JPG missing — re-analyze"
            : "缺少真240时间线图片，请重新分析"
          : lang === "en"
            ? "Image unavailable"
            : "关键帧图片缺失"}
      </span>
    </div>
  );
}

const UPPER = new Set([
  "head",
  "left_shoulder",
  "right_shoulder",
  "left_elbow",
  "right_elbow",
  "left_wrist",
  "right_wrist",
]);
const LOWER = new Set([
  "left_hip",
  "right_hip",
  "left_knee",
  "right_knee",
  "left_ankle",
  "right_ankle",
]);

function lineColorForConnection(
  fromName: string,
  toName: string,
): { stroke: string; width: number } {
  const h = fromName === "head" || toName === "head";
  const u = UPPER.has(fromName) && UPPER.has(toName);
  const l = LOWER.has(fromName) && LOWER.has(toName);
  if (h) return { stroke: "#f5c518", width: 1.4 };
  if (u) return { stroke: "#22d3ee", width: 1.2 };
  if (l) return { stroke: "#4ade80", width: 1.2 };
  return { stroke: "#a78bfa", width: 1.0 };
}

function KeyframeSkeletonSvg({
  snap,
  proMode,
  uniqueId,
}: {
  snap: PoseSnapshot;
  proMode: boolean;
  uniqueId: string;
}) {
  const joints = snap.joints || [];
  if (!joints.length) return null;

  const filterId = proMode ? `kf-skel-glow-${uniqueId}` : undefined;

  return (
    <svg
      className="pointer-events-none absolute inset-0 h-full w-full"
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
    >
      {proMode && (
        <defs>
          <filter id={filterId} x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="0.8" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
      )}
      <g filter={filterId}>
        {(snap.connections || []).map(([ia, ib], idx) => {
          const j1 = joints[ia];
          const j2 = joints[ib];
          if (!j1 || !j2 || j1.v < 0.25 || j2.v < 0.25) return null;
          const { stroke, width } = lineColorForConnection(j1.name, j2.name);
          return (
            <line
              key={idx}
              x1={j1.nx * 100}
              y1={j1.ny * 100}
              x2={j2.nx * 100}
              y2={j2.ny * 100}
              stroke={stroke}
              strokeWidth={proMode ? width * 1.35 : width}
              strokeLinecap="round"
              opacity={0.92}
            />
          );
        })}
        {joints.map((j, i) => {
          if (j.v < 0.25) return null;
          const cx = j.nx * 100;
          const cy = j.ny * 100;
          const isHead = j.name === "head";
          const isUpper = UPPER.has(j.name);
          const fill = isHead ? "#f5c518" : isUpper ? "#22d3ee" : "#4ade80";
          const r = isHead ? 2.8 : proMode ? 2.2 : 1.8;
          return (
            <circle
              key={i}
              cx={cx}
              cy={cy}
              r={r}
              fill={fill}
              opacity={0.85}
              stroke="rgba(0,0,0,0.35)"
              strokeWidth={0.35}
            />
          );
        })}
      </g>
    </svg>
  );
}

interface KeyframeStripProps {
  keyframes: Keyframe[];
  lang: "en" | "zh";
  /** Pro: stronger skeleton glow + thicker overlay */
  mode?: "default" | "pro";
  /** Pro v3: URLs only, no base64 thumbnails */
  urlOnlyTimeline?: boolean;
}

export default function KeyframeStrip({
  keyframes,
  lang,
  mode = "default",
  urlOnlyTimeline = false,
}: KeyframeStripProps) {
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const proMode = mode === "pro";

  return (
    <div className="glass-card p-5">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-base font-semibold text-white">
          {lang === "en" ? "Swing Keyframes" : "挥杆关键帧"}
        </h3>
        <div className="flex flex-wrap items-center justify-end gap-2">
          <button
            type="button"
            onClick={() => setShowSkeleton(!showSkeleton)}
            className={`rounded-lg border px-2.5 py-1 text-[10px] transition ${
              showSkeleton
                ? "border-brand-purple/30 bg-brand-purple/10 text-brand-gold/80"
                : "border-white/10 text-white/30 hover:text-white/50"
            }`}
          >
            {lang === "en" ? "Skeleton" : "骨架"}
          </button>
          <span className="text-[9px] text-white/25">
            {lang === "en" ? "Cyan upper · Green lower · Gold head" : "青上身 · 绿下身 · 金头部"}
          </span>
          <span className="text-[10px] text-white/25">
            {keyframes.length} {lang === "en" ? "frames" : "帧"}
          </span>
        </div>
      </div>

      {/* Enlarged selected keyframe */}
      {selectedIdx !== null && keyframes[selectedIdx] && (
        <div className="mb-3 relative bg-black rounded-xl overflow-hidden" style={{ minHeight: "65vh" }}>
          <KeyframeStripMedia
            kf={keyframes[selectedIdx]}
            uniqueId={`enlarged-${selectedIdx}`}
            enlarged
            className="w-full"
            lang={lang}
            showSkeleton={showSkeleton}
            proMode={proMode}
            urlOnlyTimeline={urlOnlyTimeline}
          />
          <div className="absolute bottom-3 left-3 rounded-full bg-black/60 backdrop-blur-sm px-3 py-1">
            <span className="text-xs font-semibold text-brand-gold/90">
              {lang === "en" ? keyframes[selectedIdx].label_en : keyframes[selectedIdx].label_zh}
            </span>
          </div>
          <div className="absolute top-3 right-3 rounded bg-black/50 px-2 py-1 font-mono text-[10px] text-white/50">
            {typeof keyframes[selectedIdx].timestamp === "number" ? keyframes[selectedIdx].timestamp.toFixed(2) : "—"}s
          </div>
        </div>
      )}

      <div className="flex gap-3 overflow-x-auto pb-2">
        {keyframes.map((kf, i) => (
          <div
            key={`${kf.phase}-${i}`}
            className="group flex-shrink-0 cursor-pointer"
            onClick={() => setSelectedIdx(selectedIdx === i ? null : i)}
          >
            <div
              className={`relative mb-2 w-28 overflow-hidden rounded-lg border transition ${
                selectedIdx === i
                  ? "border-brand-gold/40 shadow-[0_0_15px_rgba(245,197,24,0.15)]"
                  : "border-white/8 group-hover:border-brand-purple/30"
              }`}
            >
              <KeyframeStripMedia
                kf={kf}
                uniqueId={`${i}-${kf.phase}`}
                lang={lang}
                showSkeleton={showSkeleton}
                proMode={proMode}
                urlOnlyTimeline={urlOnlyTimeline}
              />

              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-2">
                <span className="text-[10px] font-semibold text-brand-gold/80">
                  {lang === "en" ? kf.label_en : kf.label_zh}
                </span>
              </div>

              <div className="absolute right-1 top-1 rounded bg-black/50 px-1 py-0.5 font-mono text-[8px] text-white/40">
                {typeof kf.timestamp === "number" ? kf.timestamp.toFixed(2) : "—"}s
              </div>
            </div>

            <div className="flex items-center justify-center">
              <div className="h-2 w-px bg-brand-purple/20" />
            </div>
            <div className="flex items-center gap-1">
              <div
                className={`h-1.5 w-1.5 rounded-full transition ${
                  selectedIdx === i ? "bg-brand-gold" : "bg-brand-purple/60"
                }`}
              />
              {i < keyframes.length - 1 && (
                <div className="h-px flex-1 bg-brand-purple/15" />
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
