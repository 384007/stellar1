"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PROV3_KEYFRAME_MEDIA_FAIL_EN, PROV3_KEYFRAME_MEDIA_FAIL_ZH } from "@/lib/prov3-keyframe-media";
import { resolveProv3ProductMediaUrl } from "@/lib/prov3-media-url";
import {
  getFrameState,
  loadProv3KfStore,
  saveProv3KfStore,
  setFrameState,
  type Prov3KfFrameState,
  type Prov3KfStore,
} from "@/lib/keyframe-prov3-storage";
import { keyframeImageDataUrl } from "@/lib/image-base64";
import { downloadHrefAsFile } from "@/lib/download-href-as-file";

type Tool = "pan" | "draw" | "ruler";

/** iOS 标记风格画笔色条（横向滚动） */
const STROKE_COLORS = [
  "#ffffff",
  "#000000",
  "#8e8e93",
  "#ff3b30",
  "#ff9500",
  "#ffcc00",
  "#34c759",
  "#5ac8fa",
  "#007aff",
  "#5856d6",
  "#af52de",
  "#ff2d55",
  "#a2845e",
] as const;

function imageContainRect(cw: number, ch: number, nw: number, nh: number) {
  if (!nw || !nh) return { x: 0, y: 0, w: cw, h: ch };
  const s = Math.min(cw / nw, ch / nh);
  const w = nw * s;
  const h = nh * s;
  const x = (cw - w) / 2;
  const y = (ch - h) / 2;
  return { x, y, w, h };
}

/** Container-local CSS px → normalized UV in original bitmap [0,1]² */
function pointerToUv(
  clientX: number,
  clientY: number,
  containerRect: DOMRect,
  nw: number,
  nh: number,
  tx: number,
  ty: number,
  scale: number,
  rotQ: number,
): { u: number; v: number } | null {
  const cw = containerRect.width;
  const ch = containerRect.height;
  const cx = clientX - containerRect.left;
  const cy = clientY - containerRect.top;
  let vx = cx - cw / 2 - tx;
  let vy = cy - ch / 2 - ty;
  if (scale <= 0.05) return null;
  vx /= scale;
  vy /= scale;
  const rad = (-rotQ * Math.PI) / 2;
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  const rx = vx * c - vy * s;
  const ry = vx * s + vy * c;
  const { x: ix, y: iy, w: iw, h: ih } = imageContainRect(cw, ch, nw, nh);
  const px = cw / 2 + rx;
  const py = ch / 2 + ry;
  const u = (px - ix) / iw;
  const v = (py - iy) / ih;
  if (!Number.isFinite(u) || !Number.isFinite(v)) return null;
  if (u < -0.02 || u > 1.02 || v < -0.02 || v > 1.02) return null;
  return { u: Math.min(1, Math.max(0, u)), v: Math.min(1, Math.max(0, v)) };
}

function uvToCanvas(
  u: number,
  v: number,
  cw: number,
  ch: number,
  nw: number,
  nh: number,
  tx: number,
  ty: number,
  scale: number,
  rotQ: number,
): { x: number; y: number } {
  const { x: ix, y: iy, w: iw, h: ih } = imageContainRect(cw, ch, nw, nh);
  const px = ix + u * iw;
  const py = iy + v * ih;
  let vx = px - cw / 2;
  let vy = py - ch / 2;
  const rad = (rotQ * Math.PI) / 2;
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  const rx = vx * c - vy * s;
  const ry = vx * s + vy * c;
  return {
    x: cw / 2 + rx * scale + tx,
    y: ch / 2 + ry * scale + ty,
  };
}

function rulerLengthPx(a: [number, number], b: [number, number], nw: number, nh: number): number {
  const dx = (b[0] - a[0]) * nw;
  const dy = (b[1] - a[1]) * nh;
  return Math.hypot(dx, dy);
}

export interface KeyframeLike {
  label_en: string;
  label_zh: string;
  keyframe_image_url?: string;
  image_base64?: string;
}

interface Props {
  analysisId: string;
  keyframes: KeyframeLike[];
  /** 与历史 / API 同步的条带元数据（父组件传入；不在此组件内展示文案） */
  stripMeta?: {
    timeline?: string;
    analysis_fps?: number;
    thumbnails_from_analysis_video?: boolean;
  };
  activeIndex: number;
  onActiveIndexChange: (i: number) => void;
  lang: "en" | "zh";
  /** Skeleton / guides canvas — pointer-events none */
  overlay?: React.ReactNode;
  /** 骨架 + 辅助线竖条（置于左侧，工具栏紧随其后） */
  skeletonRail?: React.ReactNode;
  /** 与 Pro v3 一致：仅允许 URL 关键帧，禁止用内嵌 base64 下载 */
  keyframeDownloadUrlOnly?: boolean;
}

export default function KeyframeProv3InteractiveViewer({
  analysisId,
  keyframes,
  stripMeta: _stripMeta,
  activeIndex,
  onActiveIndexChange,
  lang,
  overlay,
  skeletonRail,
  keyframeDownloadUrlOnly = false,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [store, setStore] = useState<Prov3KfStore>(() => loadProv3KfStore(analysisId));
  const [tool, setTool] = useState<Tool>("pan");
  const [strokeColor, setStrokeColor] = useState<string>(STROKE_COLORS[0]);
  const [tx, setTx] = useState(0);
  const [ty, setTy] = useState(0);
  const [scale, setScale] = useState(1);
  const [imgBroken, setImgBroken] = useState(false);

  const frame = keyframes[activeIndex];
  const dataUrl = frame
    ? resolveProv3ProductMediaUrl(String(frame.keyframe_image_url || "").trim()) || null
    : null;

  useEffect(() => {
    setStore(loadProv3KfStore(analysisId));
  }, [analysisId]);

  useEffect(() => {
    setImgBroken(false);
  }, [activeIndex, frame?.keyframe_image_url]);

  const frameState: Prov3KfFrameState = useMemo(
    () => getFrameState(store, activeIndex),
    [store, activeIndex],
  );

  const updateFrame = useCallback(
    (patch: Partial<Prov3KfFrameState> | ((prev: Prov3KfFrameState) => Partial<Prov3KfFrameState>)) => {
      setStore((prevStore) => {
        const fs = getFrameState(prevStore, activeIndex);
        const p = typeof patch === "function" ? patch(fs) : patch;
        const next = setFrameState(prevStore, activeIndex, p);
        saveProv3KfStore(analysisId, next);
        return next;
      });
    },
    [analysisId, activeIndex],
  );

  /* Reset view when switching frame — keep per-frame rotation/lines from store */
  useEffect(() => {
    setTx(0);
    setTy(0);
    setScale(1);
    rulerRef.current.a = null;
    drawRef.current = { active: false, pts: [] };
  }, [activeIndex]);

  const nw = imgRef.current?.naturalWidth ?? 0;
  const nh = imgRef.current?.naturalHeight ?? 0;

  const redrawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    const img = imgRef.current;
    if (!canvas || !container || !img || !dataUrl) return;
    const cw = container.offsetWidth;
    const ch = container.offsetHeight;
    if (!cw || !ch) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = cw * dpr;
    canvas.height = ch * dpr;
    canvas.style.width = `${cw}px`;
    canvas.style.height = `${ch}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);

    const inw = img.naturalWidth || cw;
    const inh = img.naturalHeight || ch;
    const rotQ = frameState.rotQ % 4;
    const fs = frameState;

    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    for (const line of fs.lines) {
      if (line.points.length < 2) continue;
      ctx.strokeStyle = line.color;
      ctx.lineWidth = 2.5;
      ctx.globalAlpha = 0.92;
      ctx.beginPath();
      const p0 = uvToCanvas(line.points[0][0], line.points[0][1], cw, ch, inw, inh, tx, ty, scale, rotQ);
      ctx.moveTo(p0.x, p0.y);
      for (let i = 1; i < line.points.length; i++) {
        const p = uvToCanvas(line.points[i][0], line.points[i][1], cw, ch, inw, inh, tx, ty, scale, rotQ);
        ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();
    }

    if (fs.ruler) {
      const { a, b, color } = fs.ruler;
      const pa = uvToCanvas(a[0], a[1], cw, ch, inw, inh, tx, ty, scale, rotQ);
      const pb = uvToCanvas(b[0], b[1], cw, ch, inw, inh, tx, ty, scale, rotQ);
      ctx.globalAlpha = 0.95;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
      ctx.setLineDash([]);
      const midX = (pa.x + pb.x) / 2;
      const midY = (pa.y + pb.y) / 2;
      const len = rulerLengthPx(a, b, inw, inh);
      ctx.font = "600 11px ui-sans-serif,system-ui,sans-serif";
      const label = `${Math.round(len)} px`;
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      ctx.fillRect(midX - tw / 2 - 4, midY - 16, tw + 8, 18);
      ctx.fillStyle = color;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, midX, midY - 7);
    }
    ctx.globalAlpha = 1;
  }, [dataUrl, frameState, tx, ty, scale]);

  useEffect(() => {
    redrawCanvas();
  }, [redrawCanvas]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => redrawCanvas());
    ro.observe(el);
    return () => ro.disconnect();
  }, [redrawCanvas]);

  /* Draw / ruler interaction */
  const drawRef = useRef<{ active: boolean; pts: [number, number][] }>({ active: false, pts: [] });
  const rulerRef = useRef<{ a: [number, number] | null }>({ a: null });

  const onPointerDown = (e: React.PointerEvent) => {
    if (!containerRef.current || !dataUrl) return;
    const rect = containerRef.current.getBoundingClientRect();
    const inw = imgRef.current?.naturalWidth ?? 0;
    const inh = imgRef.current?.naturalHeight ?? 0;
    if (!inw || !inh) return;

    if (tool === "draw") {
      (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
      const uv = pointerToUv(e.clientX, e.clientY, rect, inw, inh, tx, ty, scale, frameState.rotQ % 4);
      if (!uv) return;
      drawRef.current = { active: true, pts: [[uv.u, uv.v]] };
    } else if (tool === "ruler") {
      const uv = pointerToUv(e.clientX, e.clientY, rect, inw, inh, tx, ty, scale, frameState.rotQ % 4);
      if (!uv) return;
      const p: [number, number] = [uv.u, uv.v];
      if (!rulerRef.current.a) {
        rulerRef.current.a = p;
        updateFrame({ ruler: { color: strokeColor, a: p, b: p } });
      } else {
        updateFrame({ ruler: { color: strokeColor, a: rulerRef.current.a, b: p } });
        rulerRef.current.a = null;
      }
    }
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (tool !== "draw" || !drawRef.current.active || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const inw = imgRef.current?.naturalWidth ?? 0;
    const inh = imgRef.current?.naturalHeight ?? 0;
    const uv = pointerToUv(e.clientX, e.clientY, rect, inw, inh, tx, ty, scale, frameState.rotQ % 4);
    if (!uv) return;
    const last = drawRef.current.pts[drawRef.current.pts.length - 1];
    if (last && Math.hypot(uv.u - last[0], uv.v - last[1]) < 0.002) return;
    drawRef.current.pts.push([uv.u, uv.v]);
    redrawCanvas();
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx || drawRef.current.pts.length < 2) return;
    const cw = containerRef.current.offsetWidth;
    const ch = containerRef.current.offsetHeight;
    const rotQ = frameState.rotQ % 4;
    const pts = drawRef.current.pts;
    ctx.save();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2.5;
    ctx.globalAlpha = 0.9;
    ctx.beginPath();
    const p0 = uvToCanvas(pts[0][0], pts[0][1], cw, ch, inw, inh, tx, ty, scale, rotQ);
    ctx.moveTo(p0.x, p0.y);
    for (let i = 1; i < pts.length; i++) {
      const p = uvToCanvas(pts[i][0], pts[i][1], cw, ch, inw, inh, tx, ty, scale, rotQ);
      ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();
    ctx.restore();
  };

  const onPointerUp = () => {
    if (tool === "draw" && drawRef.current.active && drawRef.current.pts.length >= 2) {
      const pts = drawRef.current.pts;
      const col = strokeColor;
      updateFrame((fs) => ({ lines: [...fs.lines, { color: col, points: pts }] }));
    }
    drawRef.current = { active: false, pts: [] };
    redrawCanvas();
  };

  /* Pan */
  const panRef = useRef<{ id: number | null; lx: number; ly: number }>({ id: null, lx: 0, ly: 0 });
  const onStagePointerDown = (e: React.PointerEvent) => {
    if (tool !== "pan") return;
    panRef.current = { id: e.pointerId, lx: e.clientX, ly: e.clientY };
    (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
  };
  const onStagePointerMove = (e: React.PointerEvent) => {
    if (tool !== "pan" || panRef.current.id !== e.pointerId) return;
    const dx = e.clientX - panRef.current.lx;
    const dy = e.clientY - panRef.current.ly;
    panRef.current.lx = e.clientX;
    panRef.current.ly = e.clientY;
    setTx((t) => t + dx);
    setTy((t) => t + dy);
  };

  /* Pinch zoom */
  const pinchRef = useRef<{ d0: number; s0: number } | null>(null);
  const onTouchStart = (e: React.TouchEvent) => {
    if (e.touches.length === 2) {
      const [a, b] = [e.touches[0], e.touches[1]];
      const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
      pinchRef.current = { d0: Math.max(d, 1), s0: scale };
    }
  };
  const onTouchEnd = () => {
    pinchRef.current = null;
  };

  /* Swipe change frame */
  const swipeRef = useRef<{ x: number; y: number; t: number } | null>(null);
  const onSwipeTouchStart = (e: React.TouchEvent) => {
    if (e.touches.length !== 1) return;
    swipeRef.current = { x: e.touches[0].clientX, y: e.touches[0].clientY, t: Date.now() };
  };
  const onSwipeTouchEnd = (e: React.TouchEvent) => {
    const s = swipeRef.current;
    swipeRef.current = null;
    if (!s || scale > 1.08) return;
    const t = e.changedTouches[0];
    const dx = t.clientX - s.x;
    const dy = t.clientY - s.y;
    if (Math.abs(dx) < 48 || Math.abs(dx) < Math.abs(dy) * 1.2) return;
    if (dx < 0 && activeIndex < keyframes.length - 1) onActiveIndexChange(activeIndex + 1);
    if (dx > 0 && activeIndex > 0) onActiveIndexChange(activeIndex - 1);
  };

  const wheelHandler = (e: React.WheelEvent) => {
    if (!containerRef.current) return;
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.08 : 0.08;
    setScale((sc) => Math.min(6, Math.max(0.5, sc + delta)));
  };

  /* Non-passive pinch so preventDefault works on mobile — must run before any conditional return */
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const move = (e: TouchEvent) => {
      if (e.touches.length !== 2 || !pinchRef.current) return;
      e.preventDefault();
      const [a, b] = [e.touches[0], e.touches[1]];
      const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
      const next = Math.min(6, Math.max(0.6, pinchRef.current.s0 * (d / pinchRef.current.d0)));
      setScale(next);
    };
    el.addEventListener("touchmove", move, { passive: false });
    return () => el.removeEventListener("touchmove", move);
  }, []);

  const rotateCw = () =>
    updateFrame((fs) => ({
      rotQ: (fs.rotQ + 1) % 4,
    }));
  const clearMarkup = () => updateFrame({ lines: [], ruler: null });
  const undoStroke = () =>
    updateFrame((fs) =>
      fs.lines.length === 0 ? {} : { lines: fs.lines.slice(0, -1) },
    );

  const [keyframeDownloadBusy, setKeyframeDownloadBusy] = useState(false);

  const keyframeDownloadHref = frame
    ? resolveProv3ProductMediaUrl(String(frame.keyframe_image_url || "").trim()) ||
      (!keyframeDownloadUrlOnly ? keyframeImageDataUrl(frame.image_base64) : null)
    : null;

  const onDownloadKeyframe = useCallback(async () => {
    if (!keyframeDownloadHref || keyframeDownloadBusy) return;
    setKeyframeDownloadBusy(true);
    try {
      const ext = keyframeDownloadHref.startsWith("data:image/png") ? "png" : "jpg";
      await downloadHrefAsFile(
        keyframeDownloadHref,
        `stellar_${analysisId}_kf${activeIndex + 1}.${ext}`,
        true,
      );
    } finally {
      setKeyframeDownloadBusy(false);
    }
  }, [activeIndex, analysisId, keyframeDownloadBusy, keyframeDownloadHref]);

  const t = lang === "zh";

  if (!frame) {
    return (
      <div className="relative isolate h-[70vh] min-h-[280px] w-full max-h-[85vh] bg-black">
        <div className="absolute inset-0 flex items-center justify-center text-xs text-amber-200">
          {t ? "本次分析低信任，暂无可用关键帧。" : "Low-trust analysis: no usable keyframes."}
        </div>
      </div>
    );
  }

  return (
    <div
      className="relative isolate h-[70vh] min-h-[280px] w-full max-h-[85vh] bg-black select-none"
      style={{ touchAction: "none" }}
    >
      <div
        ref={containerRef}
        className="absolute inset-0 overflow-hidden"
        onWheel={wheelHandler}
        onTouchStart={(e) => {
          onTouchStart(e);
          onSwipeTouchStart(e);
        }}
        onTouchEnd={(ev) => {
          onTouchEnd();
          onSwipeTouchEnd(ev);
        }}
        onTouchCancel={() => {
          onTouchEnd();
        }}
      >
        <div
          role="presentation"
          className="absolute inset-0 flex items-center justify-center"
          style={{
            transform: `translate(${tx}px, ${ty}px) scale(${scale}) rotate(${((frameState.rotQ % 4) * 90)}deg)`,
            transformOrigin: "center center",
          }}
          onPointerDown={onStagePointerDown}
          onPointerMove={onStagePointerMove}
          onPointerUp={(e) => {
            if (panRef.current.id === e.pointerId) panRef.current.id = null;
          }}
        >
          {dataUrl && !imgBroken ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              key={dataUrl}
              ref={imgRef}
              src={dataUrl}
              alt={lang === "zh" ? frame.label_zh : frame.label_en}
              className="h-auto w-full max-h-full max-w-full object-contain"
              draggable={false}
              decoding="async"
              onLoad={redrawCanvas}
              onError={() => setImgBroken(true)}
            />
          ) : (
            <div className="flex h-48 max-w-md items-center justify-center px-4 text-center text-xs leading-relaxed text-red-200/90">
              {t ? PROV3_KEYFRAME_MEDIA_FAIL_ZH : PROV3_KEYFRAME_MEDIA_FAIL_EN}
            </div>
          )}
        </div>

        <canvas
          ref={canvasRef}
          className="absolute inset-0 z-[8]"
          style={{
            pointerEvents: tool === "draw" || tool === "ruler" ? "auto" : "none",
            touchAction: "none",
          }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerLeave={onPointerUp}
        />

        <div className="pointer-events-none absolute inset-0 z-[10]">{overlay}</div>

        {keyframeDownloadHref ? (
          <div className="pointer-events-auto absolute right-2 top-2 z-[16]">
            <button
              type="button"
              title={t ? "下载当前关键帧图" : "Download current keyframe image"}
              aria-label={t ? "下载当前关键帧图" : "Download current keyframe image"}
              onClick={() => void onDownloadKeyframe()}
              disabled={keyframeDownloadBusy}
              className="rounded-lg border border-white/[0.08] bg-black/30 p-2 text-white/45 opacity-50 backdrop-blur-sm transition hover:border-white/15 hover:bg-black/45 hover:text-white/85 hover:opacity-95 active:scale-[0.97] disabled:opacity-30"
            >
              {keyframeDownloadBusy ? (
                <span className="flex h-4 w-4 items-center justify-center text-[10px] tabular-nums" aria-hidden>
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
          </div>
        ) : null}

        {skeletonRail != null ? (
          <>
            <div className="pointer-events-auto absolute left-2 top-11 z-[17] flex flex-col items-center gap-1.5 opacity-[0.38] transition-opacity duration-200 hover:opacity-[0.92] [@media(hover:none)]:opacity-[0.55]">
              {skeletonRail}
              <div className="h-px w-7 shrink-0 bg-gradient-to-r from-transparent via-white/12 to-transparent" aria-hidden />
              <div className="flex flex-col gap-0.5 rounded-[10px] border border-white/[0.07] bg-black/28 p-1 shadow-[0_6px_28px_rgba(0,0,0,0.45)] backdrop-blur-md">
                <IconTool
                  active={tool === "pan"}
                  label={t ? "移动" : "Move"}
                  onClick={() => setTool("pan")}
                  rail
                >
                  <IconHand />
                </IconTool>
                <IconTool
                  active={tool === "draw"}
                  label={t ? "画笔" : "Brush"}
                  onClick={() => setTool("draw")}
                  rail
                >
                  <IconBrush />
                </IconTool>
                <IconTool
                  active={tool === "ruler"}
                  label={t ? "测量" : "Measure"}
                  onClick={() => setTool("ruler")}
                  rail
                >
                  <IconRuler />
                </IconTool>
                <div className="my-0.5 h-px w-6 shrink-0 self-center bg-white/[0.08]" aria-hidden />
                <IconTool label={t ? "旋转" : "Rotate"} onClick={rotateCw} rail>
                  <IconRotate />
                </IconTool>
                <IconTool
                  label={t ? "适合画面" : "Fit view"}
                  onClick={() => {
                    setTx(0);
                    setTy(0);
                    setScale(1);
                  }}
                  rail
                >
                  <IconFit />
                </IconTool>
                <IconTool label={t ? "撤销笔画" : "Undo stroke"} onClick={undoStroke} rail>
                  <IconUndo />
                </IconTool>
                <IconTool label={t ? "清除标注" : "Clear marks"} onClick={clearMarkup} rail>
                  <IconClear />
                </IconTool>
              </div>
            </div>
            <div className="pointer-events-auto absolute bottom-[4.25rem] left-12 right-3 z-[16] flex items-center justify-start px-1">
              <StrokeColorPicker
                colors={STROKE_COLORS}
                value={strokeColor}
                onChange={setStrokeColor}
                ariaLabel={t ? "画笔颜色" : "Stroke color"}
              />
            </div>
          </>
        ) : (
          <div className="pointer-events-auto absolute bottom-3 left-1/2 z-[15] flex max-w-[calc(100%-1rem)] -translate-x-1/2 flex-row flex-wrap items-center justify-center gap-1.5">
            <StrokeColorPicker
              colors={STROKE_COLORS}
              value={strokeColor}
              onChange={setStrokeColor}
              ariaLabel={t ? "画笔颜色" : "Stroke color"}
            />
            <div className="flex max-w-full items-center gap-0.5 rounded-[12px] border border-white/[0.09] bg-[#2b2b2b]/90 px-1 py-1 opacity-80 shadow-lg backdrop-blur-xl">
              <IconTool active={tool === "pan"} label={t ? "移动" : "Move"} onClick={() => setTool("pan")}>
                <IconHand />
              </IconTool>
              <IconTool active={tool === "draw"} label={t ? "画笔" : "Brush"} onClick={() => setTool("draw")}>
                <IconBrush />
              </IconTool>
              <IconTool active={tool === "ruler"} label={t ? "测量" : "Measure"} onClick={() => setTool("ruler")}>
                <IconRuler />
              </IconTool>
              <div className="mx-0.5 h-7 w-px shrink-0 bg-white/[0.08]" aria-hidden />
              <IconTool label={t ? "旋转" : "Rotate"} onClick={rotateCw}>
                <IconRotate />
              </IconTool>
              <IconTool
                label={t ? "适合画面" : "Fit view"}
                onClick={() => {
                  setTx(0);
                  setTy(0);
                  setScale(1);
                }}
              >
                <IconFit />
              </IconTool>
              <IconTool label={t ? "撤销" : "Undo"} onClick={undoStroke}>
                <IconUndo />
              </IconTool>
              <IconTool label={t ? "清除" : "Clear"} onClick={clearMarkup}>
                <IconClear />
              </IconTool>
            </div>
          </div>
        )}

        <div className="pointer-events-none absolute bottom-3 left-0 right-0 z-[12] flex flex-col items-center gap-0.5 text-center">
          <span className="rounded-md bg-black/35 px-2.5 py-0.5 text-[10px] font-medium tracking-wide text-white/50 backdrop-blur-sm">
            {lang === "zh" ? frame.label_zh : frame.label_en}
          </span>
          <span className="text-[9px] tabular-nums text-white/25">{activeIndex + 1} / {keyframes.length}</span>
        </div>
      </div>
    </div>
  );
}

/** 调色板按钮；点击后在同位置叠层淡入色板（无常驻色点） */
function StrokeColorPicker({
  colors,
  value,
  onChange,
  ariaLabel,
}: {
  colors: readonly string[];
  value: string;
  onChange: (c: string) => void;
  ariaLabel: string;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onDoc(e: MouseEvent | TouchEvent) {
      const el = rootRef.current;
      const target = e.target;
      if (!el || !(target instanceof Node) || el.contains(target)) return;
      setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("touchstart", onDoc, { passive: true });
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("touchstart", onDoc);
    };
  }, [open]);

  return (
    <div ref={rootRef} className="relative z-[30] flex h-9 w-9 shrink-0 items-center justify-center">
      <button
        type="button"
        title={ariaLabel}
        aria-label={ariaLabel}
        aria-expanded={open}
        aria-haspopup="listbox"
        onClick={() => setOpen((v) => !v)}
        className={`absolute inset-0 z-10 flex items-center justify-center rounded-xl border border-white/[0.11] bg-gradient-to-b from-white/[0.08] to-white/[0.02] text-white/42 shadow-[inset_0_1px_0_rgba(255,255,255,0.07),0_4px_18px_rgba(0,0,0,0.25)] backdrop-blur-md transition-all duration-200 ease-out hover:border-white/[0.18] hover:from-white/[0.11] hover:text-white/72 active:scale-[0.96] ${
          open ? "pointer-events-none scale-[0.92] opacity-0" : "opacity-100"
        }`}
      >
        <IconPalette />
      </button>
      {open ? (
        <div
          role="listbox"
          aria-label={ariaLabel}
          className="prov3-stroke-color-popover-animate absolute left-1/2 top-1/2 z-20 flex max-w-[min(calc(100vw-4rem),268px)] items-center gap-1 overflow-x-auto rounded-2xl border border-white/[0.14] bg-gradient-to-br from-white/[0.12] via-black/45 to-black/[0.62] px-2.5 py-1.5 shadow-[0_14px_44px_rgba(0,0,0,0.55),inset_0_1px_0_rgba(255,255,255,0.1)] ring-1 ring-white/[0.05] backdrop-blur-xl [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
        >
          {colors.map((c) => (
            <button
              key={c}
              type="button"
              role="option"
              aria-selected={value === c}
              title={c}
              className={`h-4 w-4 shrink-0 rounded-full border border-black/30 shadow-[0_1px_4px_rgba(0,0,0,0.4)] transition-[transform,opacity,box-shadow] duration-150 ease-out hover:scale-110 hover:opacity-100 active:scale-95 ${
                value === c
                  ? "opacity-100 ring-2 ring-white/50 ring-offset-1 ring-offset-black/50 scale-105"
                  : "opacity-[0.72] hover:ring-1 hover:ring-white/25"
              }`}
              style={{ backgroundColor: c }}
              onClick={() => {
                onChange(c);
                setOpen(false);
              }}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function IconTool({
  active,
  label,
  onClick,
  children,
  rail,
}: {
  active?: boolean;
  label: string;
  onClick: () => void;
  children: React.ReactNode;
  /** 左侧竖条：更淡，与骨架/辅助线一致 */
  rail?: boolean;
}) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      onClick={onClick}
      className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-md transition ${
        rail
          ? active
            ? "bg-[#5eb3ff]/14 text-[#a8d8ff] shadow-[inset_0_0_0_1px_rgba(94,179,255,0.22)]"
            : "text-white/30 hover:bg-white/[0.05] hover:text-white/65"
          : active
            ? "bg-[#5eb3ff]/20 text-[#7ec8ff] shadow-[inset_0_0_0_1px_rgba(94,179,255,0.35)]"
            : "text-white/45 hover:bg-white/[0.08] hover:text-white/85"
      }`}
    >
      {children}
    </button>
  );
}

function IconHand() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 9v10M5 9l4-3v10M9 6l4 2v8M13 8l4-1v7M17 7v8a3 3 0 0 1-3 3h-6a3 3 0 0 1-3-3V9" />
    </svg>
  );
}

function IconBrush() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" d="m16.862 4.487 1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L6.832 19.82a4.5 4.5 0 0 1-1.897 1.13l-2.685.8.8-2.685a4.5 4.5 0 0 1 1.13-1.897L16.863 4.487Zm0 0L19.5 7.125" />
    </svg>
  );
}

function IconRuler() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 19.5 15-15" />
      <path strokeLinecap="round" d="M6 17.5V16M8 15.5V14M10 13.5V12M12 11.5V10M14 9.5V8M16 7.5V6" />
    </svg>
  );
}

function IconRotate() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 11.667 0l1.194-1.194m-4.892-4.892 3.182-3.182a8.25 8.25 0 0 0-11.667 0L2.985 9.348" />
    </svg>
  );
}

function IconFit() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15" />
    </svg>
  );
}

function IconUndo() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 15 3 9l6-6" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M3 9h10.5a5.5 5.5 0 0 1 0 11H12" />
    </svg>
  );
}

function IconClear() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" d="m19 7-.867 12.142A2 2 0 0 1 16.138 21H7.862a2 2 0 0 1-1.995-1.858L5 7m5 4v6m4-6v6m1-10V5a1 1 0 0 0-1-1h-4a1 1 0 0 0-1 1v2M4 7h16" />
    </svg>
  );
}

function IconPalette() {
  return (
    <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M9.53 16.122a3 3 0 0 0-5.78 1.128 2.25 2.25 0 0 1-2.4 2.245 4.5 4.5 0 0 0 8.4-2.245c0-.399-.078-.78-.22-1.128zm0 0a15.998 15.998 0 0 0 3.388-1.62m-5.043-.025a15.994 15.994 0 0 1 1.622-3.395m3.42 3.42a15.995 15.995 0 0 0 4.764-4.648l3.876-5.814a1.151 1.151 0 0 0-1.597-1.597L14.146 6.32a15.996 15.996 0 0 0-4.649 4.763m3.42 3.42a6.776 6.776 0 0 0-3.42-3.42"
      />
    </svg>
  );
}
