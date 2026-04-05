"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { keyframeImageDataUrl } from "@/lib/image-base64";
import {
  getFrameState,
  loadProv3KfStore,
  saveProv3KfStore,
  setFrameState,
  type Prov3KfFrameState,
  type Prov3KfStore,
} from "@/lib/keyframe-prov3-storage";

type Tool = "pan" | "draw" | "ruler";

const STROKE_COLORS = ["#ffffff", "#f5c518", "#ef4444", "#22d3ee", "#4ade80", "#a855f7"];

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
  image_base64?: string;
}

interface Props {
  analysisId: string;
  keyframes: KeyframeLike[];
  activeIndex: number;
  onActiveIndexChange: (i: number) => void;
  lang: "en" | "zh";
  /** Skeleton / guides — pointer-events none */
  overlay?: React.ReactNode;
  /** e.g. download highlight */
  topRightActions?: React.ReactNode;
}

export default function KeyframeProv3InteractiveViewer({
  analysisId,
  keyframes,
  activeIndex,
  onActiveIndexChange,
  lang,
  overlay,
  topRightActions,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [store, setStore] = useState<Prov3KfStore>(() => loadProv3KfStore(analysisId));
  const [tool, setTool] = useState<Tool>("pan");
  const [strokeColor, setStrokeColor] = useState(STROKE_COLORS[0]);
  const [tx, setTx] = useState(0);
  const [ty, setTy] = useState(0);
  const [scale, setScale] = useState(1);
  const [imgBroken, setImgBroken] = useState(false);

  const frame = keyframes[activeIndex];
  const dataUrl = frame ? keyframeImageDataUrl(frame.image_base64) : null;

  useEffect(() => {
    setStore(loadProv3KfStore(analysisId));
  }, [analysisId]);

  useEffect(() => {
    setImgBroken(false);
  }, [activeIndex, frame?.image_base64]);

  const persist = useCallback(
    (next: Prov3KfStore) => {
      setStore(next);
      saveProv3KfStore(analysisId, next);
    },
    [analysisId],
  );

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

  const t = lang === "zh";

  if (!frame) return null;

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
            <div className="flex h-48 items-center justify-center text-xs text-white/35">
              {t ? "无图" : "No image"}
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

        {topRightActions ? (
          <div className="absolute right-2 top-2 z-[14] flex flex-col gap-1" style={{ pointerEvents: "auto" }}>
            {topRightActions}
          </div>
        ) : null}

        {/* Faded toolbar — bottom, compact */}
        <div
          className="absolute bottom-3 left-1/2 z-[15] flex max-w-[98vw] -translate-x-1/2 flex-wrap items-center justify-center gap-1 rounded-full border border-white/10 bg-black/30 px-2 py-1.5 backdrop-blur-md transition-opacity duration-300 hover:bg-black/55"
          style={{ opacity: 0.38, pointerEvents: "auto" }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLDivElement).style.opacity = "0.92";
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLDivElement).style.opacity = "0.38";
          }}
          onTouchStart={(e) => {
            (e.currentTarget as HTMLDivElement).style.opacity = "0.92";
          }}
        >
          <ToolBtn active={tool === "pan"} label={t ? "拖" : "Pan"} onClick={() => setTool("pan")} />
          <ToolBtn active={tool === "draw"} label={t ? "画" : "Draw"} onClick={() => setTool("draw")} />
          <ToolBtn active={tool === "ruler"} label={t ? "尺" : "Ruler"} onClick={() => setTool("ruler")} />
          <span className="mx-0.5 h-4 w-px bg-white/15" />
          {STROKE_COLORS.map((c) => (
            <button
              key={c}
              type="button"
              title={c}
              className={`h-5 w-5 rounded-full border-2 transition ${
                strokeColor === c ? "border-white scale-110" : "border-transparent opacity-70"
              }`}
              style={{ backgroundColor: c }}
              onClick={() => setStrokeColor(c)}
            />
          ))}
          <span className="mx-0.5 h-4 w-px bg-white/15" />
          <ToolBtn label="↻90°" title={t ? "顺时针旋转" : "Rotate 90° CW"} onClick={rotateCw} />
          <ToolBtn
            label={t ? "复位视" : "View"}
            title={t ? "重置缩放与平移" : "Reset zoom & pan"}
            onClick={() => {
              setTx(0);
              setTy(0);
              setScale(1);
            }}
          />
          <ToolBtn label={t ? "撤线" : "Undo"} onClick={undoStroke} />
          <ToolBtn label={t ? "清标" : "Clear"} onClick={clearMarkup} />
        </div>

        <div className="pointer-events-none absolute bottom-14 left-3 right-3 z-[12] flex flex-col items-center gap-0.5 text-center">
          <span className="rounded-full bg-black/35 px-2 py-0.5 text-[10px] font-medium text-white/55">
            {lang === "zh" ? frame.label_zh : frame.label_en}
          </span>
          <span className="text-[9px] text-white/40">
            {t ? `${activeIndex + 1}/${keyframes.length} · 双指缩放 · 缩小时左右滑换帧` : `${activeIndex + 1} / ${keyframes.length} · Pinch zoom · swipe L/R when zoomed out`}
          </span>
        </div>
      </div>
    </div>
  );
}

function ToolBtn({
  active,
  label,
  title,
  onClick,
}: {
  active?: boolean;
  label: string;
  title?: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      className={`rounded-lg px-2 py-0.5 text-[10px] font-semibold transition ${
        active ? "bg-brand-gold/25 text-brand-gold" : "text-white/55 hover:bg-white/10 hover:text-white/80"
      }`}
    >
      {label}
    </button>
  );
}
