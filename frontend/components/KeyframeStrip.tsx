"use client";

import { useEffect, useRef, useState } from "react";
import { keyframeImageDataUrl } from "@/lib/image-base64";
import { resolveProv3ProductMediaUrl } from "@/lib/prov3-media-url";
import {
  drawPlusStyleSkeletonOverlay,
  letterboxPoseInContainer,
  plusSkeletonScale,
  type PlusSkelPose,
} from "@/lib/plus-skeleton-canvas-draw";

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

function snapshotToPlusSkelPose(snap: PoseSnapshot): PlusSkelPose {
  return {
    joints: snap.joints.map((j) => ({
      name: j.name,
      normalized: { x: j.nx, y: j.ny },
      visibility: j.v,
    })),
    connections: snap.connections || [],
  };
}

function drawPlusKeyframeSkeleton(
  ctx: CanvasRenderingContext2D,
  cW: number,
  cH: number,
  pose: PlusSkelPose,
  showSkeleton: boolean,
  showGuideLines: boolean,
) {
  ctx.clearRect(0, 0, cW, cH);
  if (!pose?.joints?.length) return;
  const { offsetX, offsetY, renderW, renderH } = letterboxPoseInContainer(
    100,
    100,
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
    pose,
    px,
    s,
    offsetY,
    renderH,
    showSkeleton,
    showGuideLines,
  );
}

function KeyframePlusSkeletonCanvas({
  snap,
  showSkeleton,
}: {
  snap: PoseSnapshot;
  showSkeleton: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement as HTMLElement;
    if (!parent) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cW = parent.offsetWidth;
    const cH = parent.offsetHeight;
    if (!cW || !cH) return;
    canvas.width = cW * dpr;
    canvas.height = cH * dpr;
    canvas.style.width = `${cW}px`;
    canvas.style.height = `${cH}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const pose = snapshotToPlusSkelPose(snap);
    drawPlusKeyframeSkeleton(ctx, cW, cH, pose, showSkeleton, showSkeleton);
  }, [snap, showSkeleton]);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute inset-0 h-full w-full"
      style={{ zIndex: 5 }}
    />
  );
}

function KeyframeStripMedia({
  kf,
  uniqueId,
  className,
  enlarged,
  lightboxFill,
  lang,
  showSkeleton,
  proMode,
  urlOnlyTimeline,
  plusStyleKeyframeSkeleton,
}: {
  kf: Keyframe;
  uniqueId: string;
  className?: string;
  enlarged?: boolean;
  /** Fills 80vh lightbox viewer (centered object-contain) */
  lightboxFill?: boolean;
  lang: "en" | "zh";
  showSkeleton: boolean;
  proMode: boolean;
  /** Pro v3 history: only show timeline JPG URLs — no base64 fallback */
  urlOnlyTimeline?: boolean;
  /** Pro v3: draw Plus gradient / meteor skeleton on keyframe thumbs (not legacy Pro SVG). */
  plusStyleKeyframeSkeleton?: boolean;
}) {
  const [imgBroken, setImgBroken] = useState(false);
  const showSkel = showSkeleton && !!kf.pose_snapshot?.joints?.length;
  const url = resolveProv3ProductMediaUrl(String(kf.keyframe_image_url ?? "").trim());
  const b64Url = urlOnlyTimeline ? null : keyframeImageDataUrl(kf.image_base64);
  const imgSrc = urlOnlyTimeline ? (url || null) : url || b64Url;
  const bigBox =
    enlarged && lightboxFill
      ? "relative flex h-full min-h-0 w-full items-center justify-center"
      : enlarged
        ? "relative min-h-[65vh] w-full"
        : "";
  if (imgSrc && !imgBroken) {
    return (
      <div className={`${bigBox} ${className ?? ""}`}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          key={imgSrc}
          src={imgSrc}
          alt={kf.phase}
          className={
            enlarged && lightboxFill
              ? "max-h-full max-w-full object-contain select-none"
              : enlarged
                ? "absolute inset-0 h-full w-full object-contain"
                : "w-full object-contain transition-transform group-hover:scale-105"
          }
          onError={() => setImgBroken(true)}
          draggable={false}
        />
        {showSkel ? (
          plusStyleKeyframeSkeleton ? (
            <KeyframePlusSkeletonCanvas snap={kf.pose_snapshot!} showSkeleton={showSkeleton} />
          ) : (
            <KeyframeSkeletonSvg snap={kf.pose_snapshot!} proMode={proMode} uniqueId={uniqueId} />
          )
        ) : showSkeleton && !enlarged ? (
          <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-brand-purple/15 via-transparent to-transparent" />
        ) : null}
      </div>
    );
  }
  if (showSkel && !urlOnlyTimeline) {
    return (
      <div
        className={`relative flex items-center justify-center overflow-hidden bg-black/50 ${className ?? ""}`}
        style={{
          minHeight: enlarged && lightboxFill ? undefined : enlarged ? "65vh" : 112,
          height: enlarged && lightboxFill ? "100%" : undefined,
        }}
      >
        {plusStyleKeyframeSkeleton ? (
          <KeyframePlusSkeletonCanvas snap={kf.pose_snapshot!} showSkeleton={showSkeleton} />
        ) : (
          <KeyframeSkeletonSvg snap={kf.pose_snapshot!} proMode={proMode} uniqueId={uniqueId} />
        )}
        <span className="pointer-events-none absolute bottom-2 left-2 right-2 text-center text-[9px] text-white/35">
          {lang === "en" ? "Pose only" : "仅骨架"}
        </span>
      </div>
    );
  }
  return (
    <div
      className={`flex flex-col items-center justify-center gap-1 bg-black/40 px-2 text-center ${className ?? ""}`}
      style={{
        minHeight: enlarged && lightboxFill ? undefined : enlarged ? "65vh" : 112,
        height: enlarged && lightboxFill ? "100%" : undefined,
      }}
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
  /** Pro v3 pipeline: Plus-style gradient skeleton on thumbs (same kernel as video overlay). */
  plusStyleKeyframeSkeleton?: boolean;
}

const LB_MIN_SCALE = 1;
const LB_MAX_SCALE = 4;
const LB_SWIPE_PX = 52;

function KeyframeLightbox({
  open,
  index,
  keyframes,
  onClose,
  onIndexChange,
  lang,
  showSkeleton,
  proMode,
  urlOnlyTimeline,
  plusStyleKeyframeSkeleton,
}: {
  open: boolean;
  index: number;
  keyframes: Keyframe[];
  onClose: () => void;
  onIndexChange: (i: number) => void;
  lang: "en" | "zh";
  showSkeleton: boolean;
  proMode: boolean;
  urlOnlyTimeline?: boolean;
  plusStyleKeyframeSkeleton?: boolean;
}) {
  const [scale, setScale] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const scaleRef = useRef(1);
  const panRef = useRef({ x: 0, y: 0 });
  const pinchLastD = useRef(0);
  const swipe0 = useRef({ x: 0, y: 0 });
  const panDrag = useRef({
    active: false,
    startX: 0,
    startY: 0,
    ox: 0,
    oy: 0,
  });

  useEffect(() => {
    scaleRef.current = scale;
  }, [scale]);
  useEffect(() => {
    panRef.current = pan;
  }, [pan]);

  useEffect(() => {
    if (!open) return;
    setScale(1);
    setPan({ x: 0, y: 0 });
    scaleRef.current = 1;
    panRef.current = { x: 0, y: 0 };
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = "";
    };
  }, [open, index]);

  const go = (dir: -1 | 1) => {
    const n = keyframes.length;
    if (n === 0) return;
    onIndexChange((index + dir + n * 8) % n);
  };

  const onTouchStart = (e: React.TouchEvent) => {
    if (e.touches.length === 2) {
      const a = e.touches[0];
      const b = e.touches[1];
      pinchLastD.current = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
      panDrag.current.active = false;
    } else if (e.touches.length === 1) {
      const t = e.touches[0];
      swipe0.current = { x: t.clientX, y: t.clientY };
      const z = scaleRef.current;
      panDrag.current = {
        active: z > 1.02,
        startX: t.clientX,
        startY: t.clientY,
        ox: panRef.current.x,
        oy: panRef.current.y,
      };
    }
  };

  const onTouchMove = (e: React.TouchEvent) => {
    if (e.touches.length === 2) {
      e.preventDefault();
      const a = e.touches[0];
      const b = e.touches[1];
      const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
      if (pinchLastD.current > 1) {
        const r = d / pinchLastD.current;
        pinchLastD.current = d;
        setScale((s) => Math.min(LB_MAX_SCALE, Math.max(LB_MIN_SCALE, s * r)));
      }
    } else if (e.touches.length === 1 && panDrag.current.active) {
      e.preventDefault();
      const t = e.touches[0];
      setPan({
        x: panDrag.current.ox + (t.clientX - panDrag.current.startX),
        y: panDrag.current.oy + (t.clientY - panDrag.current.startY),
      });
    }
  };

  const onTouchEnd = (e: React.TouchEvent) => {
    if (e.touches.length === 1) {
      pinchLastD.current = 0;
      const t = e.touches[0];
      swipe0.current = { x: t.clientX, y: t.clientY };
      const z = scaleRef.current;
      panDrag.current = {
        active: z > 1.02,
        startX: t.clientX,
        startY: t.clientY,
        ox: panRef.current.x,
        oy: panRef.current.y,
      };
      return;
    }
    if (e.touches.length > 0) return;
    pinchLastD.current = 0;
    panDrag.current.active = false;
    if (scaleRef.current <= 1.02 && e.changedTouches.length === 1) {
      const t = e.changedTouches[0];
      const dx = t.clientX - swipe0.current.x;
      const dy = t.clientY - swipe0.current.y;
      if (Math.abs(dx) > LB_SWIPE_PX && Math.abs(dx) > Math.abs(dy) * 1.12) {
        go(dx < 0 ? 1 : -1);
      }
    }
  };

  const onWheel = (e: React.WheelEvent) => {
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    const f = 1 - Math.sign(e.deltaY) * 0.08;
    setScale((s) => Math.min(LB_MAX_SCALE, Math.max(LB_MIN_SCALE, s * f)));
  };

  if (!open || index < 0 || index >= keyframes.length) return null;
  const kf = keyframes[index];

  return (
    <div
      className="fixed inset-0 z-[220] flex flex-col items-center justify-center bg-black/88 p-3 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label={lang === "en" ? "Keyframe preview" : "关键帧预览"}
      onClick={onClose}
    >
      <button
        type="button"
        className="absolute right-3 top-3 z-10 rounded-full border border-white/20 bg-black/60 px-3 py-1.5 text-xs text-white/80 hover:bg-white/10"
        onClick={(ev) => {
          ev.stopPropagation();
          onClose();
        }}
      >
        {lang === "en" ? "Close" : "关闭"}
      </button>
      <button
        type="button"
        className="absolute left-2 top-1/2 z-10 -translate-y-1/2 rounded-full border border-white/15 bg-black/50 p-2 text-white/70 hover:bg-white/10 md:left-4"
        aria-label={lang === "en" ? "Previous" : "上一张"}
        onClick={(ev) => {
          ev.stopPropagation();
          go(-1);
        }}
      >
        ‹
      </button>
      <button
        type="button"
        className="absolute right-2 top-1/2 z-10 -translate-y-1/2 rounded-full border border-white/15 bg-black/50 p-2 text-white/70 hover:bg-white/10 md:right-4"
        aria-label={lang === "en" ? "Next" : "下一张"}
        onClick={(ev) => {
          ev.stopPropagation();
          go(1);
        }}
      >
        ›
      </button>

      <div
        className="relative flex w-[80vw] max-w-[min(80vw,960px)] flex-col overflow-hidden rounded-2xl border border-white/10 bg-black/90 shadow-2xl"
        style={{ height: "min(80dvh, 80vh)", touchAction: "none" }}
        onClick={(ev) => ev.stopPropagation()}
        onWheel={onWheel}
        onTouchStart={onTouchStart}
        onTouchMove={onTouchMove}
        onTouchEnd={onTouchEnd}
      >
        <div className="relative min-h-0 flex-1 overflow-hidden">
          <div
            className="flex h-full w-full items-center justify-center"
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale})`,
              transformOrigin: "center center",
            }}
          >
            <KeyframeStripMedia
              kf={kf}
              uniqueId={`lb-${index}`}
              enlarged
              lightboxFill
              lang={lang}
              showSkeleton={showSkeleton}
              proMode={proMode}
              urlOnlyTimeline={urlOnlyTimeline}
              plusStyleKeyframeSkeleton={plusStyleKeyframeSkeleton}
            />
          </div>
        </div>
        <div className="flex shrink-0 items-center justify-between gap-2 border-t border-white/10 bg-black/60 px-3 py-2">
          <span className="text-xs font-semibold text-brand-gold/90">
            {lang === "en" ? kf.label_en : kf.label_zh}
          </span>
          <span className="font-mono text-[10px] text-white/45">
            {index + 1}/{keyframes.length} ·{" "}
            {typeof kf.timestamp === "number" ? kf.timestamp.toFixed(2) : "—"}s
          </span>
        </div>
      </div>
      <p className="mt-2 max-w-[85vw] text-center text-[10px] text-white/35">
        {lang === "en"
          ? "Swipe to change · Pinch to zoom · Drag when zoomed · Tap outside to close"
          : "左右滑动切换 · 双指捏合放大 · 放大后可拖动 · 点击外侧关闭"}
      </p>
    </div>
  );
}

export default function KeyframeStrip({
  keyframes,
  lang,
  mode = "default",
  urlOnlyTimeline = false,
  plusStyleKeyframeSkeleton = false,
}: KeyframeStripProps) {
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const proMode = mode === "pro";

  return (
    <>
      <KeyframeLightbox
        open={lightboxOpen && selectedIdx !== null}
        index={selectedIdx ?? 0}
        keyframes={keyframes}
        onClose={() => {
          setLightboxOpen(false);
          setSelectedIdx(null);
        }}
        onIndexChange={setSelectedIdx}
        lang={lang}
        showSkeleton={showSkeleton}
        proMode={proMode}
        urlOnlyTimeline={urlOnlyTimeline}
        plusStyleKeyframeSkeleton={plusStyleKeyframeSkeleton}
      />
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
            {plusStyleKeyframeSkeleton
              ? lang === "en"
                ? "Plus gradient skeleton · plumb · swing arcs"
                : "Plus 渐变骨架 · 铅垂线 · 挥杆弧"
              : lang === "en"
                ? "Cyan upper · Green lower · Gold head"
                : "青上身 · 绿下身 · 金头部"}
          </span>
          <span className="text-[10px] text-white/25">
            {keyframes.length} {lang === "en" ? "frames" : "帧"}
          </span>
        </div>
      </div>

      <div className="flex gap-3 overflow-x-auto pb-2">
        {keyframes.map((kf, i) => (
          <div
            key={`${kf.phase}-${i}`}
            className="group flex-shrink-0 cursor-pointer"
            onClick={() => {
              setSelectedIdx(i);
              setLightboxOpen(true);
            }}
          >
            <div
              className={`relative mb-2 w-28 overflow-hidden rounded-lg border transition ${
                lightboxOpen && selectedIdx === i
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
                plusStyleKeyframeSkeleton={plusStyleKeyframeSkeleton}
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
                  lightboxOpen && selectedIdx === i ? "bg-brand-gold" : "bg-brand-purple/60"
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
    </>
  );
}
