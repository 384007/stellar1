"use client";

import { useRef, useEffect, useState, useCallback } from "react";

interface Joint {
  name: string;
  x: number;
  y: number;
  z: number;
  visibility: number;
  normalized: { x: number; y: number };
}

interface SkeletonFrame {
  joints: Joint[];
  connections: number[][];
  angles: Record<string, number>;
  frame_size: { width: number; height: number };
  frame_index?: number;
  timestamp?: number;
  image_base64?: string;
}

interface Skeleton3DViewerProps {
  frames: SkeletonFrame[];
  lang: "en" | "zh";
}

const JOINT_NAMES = [
  "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_hip", "right_hip",
  "left_knee", "right_knee", "left_ankle", "right_ankle",
];

const CONNECTIONS: [number, number][] = [
  [1, 2], [1, 3], [3, 5], [2, 4], [4, 6],
  [1, 7], [2, 8], [7, 8],
  [7, 9], [9, 11], [8, 10], [10, 12],
  [0, 1], [0, 2],
];

const LIMB_WIDTHS: Record<string, number> = {
  "1-2": 6, "7-8": 5,
  "1-7": 4, "2-8": 4,
  "1-3": 4, "3-5": 3, "2-4": 4, "4-6": 3,
  "7-9": 5, "9-11": 4, "8-10": 5, "10-12": 4,
  "0-1": 3, "0-2": 3,
};

function getLimbColor(i1: number, i2: number): string {
  const n1 = JOINT_NAMES[i1], n2 = JOINT_NAMES[i2];
  if (n1?.includes("shoulder") || n2?.includes("shoulder")) return "#a78bfa";
  if (n1?.includes("elbow") || n1?.includes("wrist") || n2?.includes("wrist")) return "#818cf8";
  if (n1?.includes("hip")) return "#f87171";
  if (n1?.includes("knee") || n2?.includes("ankle")) return "#fb923c";
  return "#c4b5fd";
}

interface SwingPhase {
  id: string;
  en: string;
  zh: string;
  icon: string;
  pct: [number, number];
  tipZh: string;
  tipEn: string;
}

const PHASES: SwingPhase[] = [
  { id: "address", en: "Address", zh: "准备", icon: "🏌️", pct: [0, 10],
    tipZh: "双脚肩宽，重心居中，脊柱前倾约30°", tipEn: "Feet shoulder-width, centered weight, ~30° spine tilt" },
  { id: "takeaway", en: "Takeaway", zh: "起杆", icon: "↗️", pct: [10, 25],
    tipZh: "一体化启动，保持三角形", tipEn: "One-piece takeaway, keep the triangle" },
  { id: "backswing", en: "Backswing", zh: "后摆", icon: "🔄", pct: [25, 45],
    tipZh: "肩转90°，右膝保持稳定", tipEn: "90° shoulder turn, stable trail knee" },
  { id: "top", en: "Top", zh: "顶点", icon: "⬆️", pct: [45, 55],
    tipZh: "充分蓄力！杆身平行，手腕完全上翘", tipEn: "Full coil! Club parallel, wrists fully hinged" },
  { id: "down", en: "Down", zh: "下杆", icon: "⬇️", pct: [55, 70],
    tipZh: "髋先启动 → 躯干 → 肩 → 手臂", tipEn: "Hips lead → torso → shoulders → arms" },
  { id: "impact", en: "Impact", zh: "触球", icon: "💥", pct: [70, 80],
    tipZh: "手在球前方，杆面方正，向下压缩", tipEn: "Hands ahead, square face, descending blow" },
  { id: "follow", en: "Finish", zh: "收杆", icon: "🎯", pct: [80, 100],
    tipZh: "完全释放，皮带扣朝目标", tipEn: "Full release, belt buckle to target" },
];

function getPhase(idx: number, total: number): number {
  const pct = total <= 1 ? 0 : (idx / (total - 1)) * 100;
  for (let i = PHASES.length - 1; i >= 0; i--) {
    if (pct >= PHASES[i].pct[0]) return i;
  }
  return 0;
}

export default function Skeleton3DViewer({ frames, lang }: Skeleton3DViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgCacheRef = useRef<Map<number, HTMLImageElement>>(new Map());
  const [currentFrame, setCurrentFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(250);
  const [overlayOpacity, setOverlayOpacity] = useState(0.7);
  const [showAngles, setShowAngles] = useState(true);
  const [viewMode, setViewMode] = useState<"pose" | "flow">("pose");
  const playRef = useRef<ReturnType<typeof setInterval>>(undefined);

  const frame = frames[currentFrame];
  const hasImages = frames.some(f => f.image_base64);
  const activePhase = getPhase(currentFrame, frames.length);
  const phase = PHASES[activePhase];

  // Preload images
  useEffect(() => {
    const cache = imgCacheRef.current;
    frames.forEach((f, i) => {
      if (f.image_base64 && !cache.has(i)) {
        const img = new Image();
        img.src = `data:image/jpeg;base64,${f.image_base64}`;
        cache.set(i, img);
      }
    });
  }, [frames]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frame) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width, H = rect.height;

    // Background
    ctx.fillStyle = "#0a0a12";
    ctx.fillRect(0, 0, W, H);

    // Draw the real person image
    const img = imgCacheRef.current.get(currentFrame);
    if (img && img.complete && img.naturalWidth > 0) {
      const imgAspect = img.naturalWidth / img.naturalHeight;
      const canvasAspect = W / H;
      let dw: number, dh: number, dx: number, dy: number;
      if (imgAspect > canvasAspect) {
        dh = H; dw = H * imgAspect; dx = (W - dw) / 2; dy = 0;
      } else {
        dw = W; dh = W / imgAspect; dx = 0; dy = (H - dh) / 2;
      }
      ctx.globalAlpha = 1;
      ctx.drawImage(img, dx, dy, dw, dh);

      // Dark overlay so skeleton is visible
      ctx.globalAlpha = 1 - overlayOpacity;
      ctx.fillStyle = "#0a0a12";
      ctx.fillRect(0, 0, W, H);
      ctx.globalAlpha = 1;
    }

    // Compute joint pixel positions based on normalized coords
    const joints = frame.joints;
    const pts = joints.map(j => {
      const nx = j.normalized?.x ?? j.x / (frame.frame_size?.width || 1);
      const ny = j.normalized?.y ?? j.y / (frame.frame_size?.height || 1);
      return { px: nx * W, py: ny * H, vis: j.visibility };
    });

    // Motion trail (previous frames ghosted)
    const trailIndices = [currentFrame - 2, currentFrame - 1].filter(i => i >= 0);
    for (let ti = 0; ti < trailIndices.length; ti++) {
      const tf = frames[trailIndices[ti]];
      if (!tf) continue;
      const tPts = tf.joints.map(j => {
        const nx = j.normalized?.x ?? j.x / (tf.frame_size?.width || 1);
        const ny = j.normalized?.y ?? j.y / (tf.frame_size?.height || 1);
        return { px: nx * W, py: ny * H, vis: j.visibility };
      });
      const alpha = 0.08 + ti * 0.06;
      ctx.globalAlpha = alpha;
      for (const [i1, i2] of CONNECTIONS) {
        if (i1 >= tPts.length || i2 >= tPts.length) continue;
        if (tPts[i1].vis < 0.3 || tPts[i2].vis < 0.3) continue;
        ctx.beginPath();
        ctx.moveTo(tPts[i1].px, tPts[i1].py);
        ctx.lineTo(tPts[i2].px, tPts[i2].py);
        ctx.strokeStyle = "#a78bfa";
        ctx.lineWidth = 2;
        ctx.lineCap = "round";
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;

    // Limbs with glow
    for (const [i1, i2] of CONNECTIONS) {
      if (i1 >= pts.length || i2 >= pts.length) continue;
      const p1 = pts[i1], p2 = pts[i2];
      if (p1.vis < 0.3 || p2.vis < 0.3) continue;
      const key = `${Math.min(i1, i2)}-${Math.max(i1, i2)}`;
      const w = LIMB_WIDTHS[key] || 3;
      const color = getLimbColor(i1, i2);

      // Glow
      ctx.globalAlpha = 0.25;
      ctx.beginPath(); ctx.moveTo(p1.px, p1.py); ctx.lineTo(p2.px, p2.py);
      ctx.strokeStyle = color; ctx.lineWidth = w + 8; ctx.lineCap = "round"; ctx.stroke();

      // Main line
      ctx.globalAlpha = 0.85;
      ctx.beginPath(); ctx.moveTo(p1.px, p1.py); ctx.lineTo(p2.px, p2.py);
      ctx.strokeStyle = color; ctx.lineWidth = w; ctx.lineCap = "round"; ctx.stroke();

      // Highlight
      ctx.globalAlpha = 0.3;
      const dx = p2.px - p1.px, dy = p2.py - p1.py;
      const len = Math.hypot(dx, dy);
      if (len > 0) {
        const nx = -dy / len, ny = dx / len;
        ctx.beginPath();
        ctx.moveTo(p1.px + nx * w * 0.3, p1.py + ny * w * 0.3);
        ctx.lineTo(p2.px + nx * w * 0.3, p2.py + ny * w * 0.3);
        ctx.strokeStyle = "#ffffff"; ctx.lineWidth = w * 0.3; ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;

    // Joints
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      if (p.vis < 0.3) continue;
      const name = JOINT_NAMES[i] || "";
      const isHead = name === "head";
      const r = isHead ? 10 : (name.includes("wrist") || name.includes("ankle") ? 4 : 6);

      // Outer glow
      ctx.globalAlpha = 0.3;
      const glow = ctx.createRadialGradient(p.px, p.py, 0, p.px, p.py, r * 3);
      glow.addColorStop(0, isHead ? "rgba(245,197,24,0.5)" : "rgba(167,139,250,0.4)");
      glow.addColorStop(1, "transparent");
      ctx.beginPath(); ctx.arc(p.px, p.py, r * 3, 0, Math.PI * 2);
      ctx.fillStyle = glow; ctx.fill();

      // Joint sphere
      ctx.globalAlpha = 0.9;
      const jg = ctx.createRadialGradient(p.px - r * 0.3, p.py - r * 0.3, 0, p.px, p.py, r);
      if (isHead) {
        jg.addColorStop(0, "#ffe88a"); jg.addColorStop(1, "#d4af37");
      } else if (name.includes("shoulder") || name.includes("elbow") || name.includes("wrist")) {
        jg.addColorStop(0, "#c4b5fd"); jg.addColorStop(1, "#7c3aed");
      } else {
        jg.addColorStop(0, "#fca5a5"); jg.addColorStop(1, "#dc2626");
      }
      ctx.beginPath(); ctx.arc(p.px, p.py, r, 0, Math.PI * 2);
      ctx.fillStyle = jg; ctx.fill();

      // Specular
      ctx.globalAlpha = 0.5;
      ctx.beginPath(); ctx.arc(p.px - r * 0.25, p.py - r * 0.25, r * 0.35, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,255,255,0.7)"; ctx.fill();
    }
    ctx.globalAlpha = 1;

    // Angle labels
    if (showAngles && frame.angles) {
      const labels: [string, string, number, string][] = [
        ["L.Elbow", "左肘", 3, "left_elbow"],
        ["R.Elbow", "右肘", 4, "right_elbow"],
        ["L.Knee", "左膝", 9, "left_knee"],
        ["R.Knee", "右膝", 10, "right_knee"],
      ];
      ctx.font = "bold 10px system-ui, sans-serif";
      for (const [en, zh, jIdx, key] of labels) {
        const val = frame.angles[key];
        if (val === undefined || jIdx >= pts.length || pts[jIdx].vis < 0.3) continue;
        const p = pts[jIdx];
        const text = `${lang === "zh" ? zh : en} ${val}°`;
        const tw = ctx.measureText(text).width;
        ctx.globalAlpha = 0.7;
        ctx.fillStyle = "rgba(0,0,0,0.65)";
        ctx.beginPath(); ctx.roundRect(p.px + 8, p.py - 14, tw + 10, 18, 4); ctx.fill();
        ctx.globalAlpha = 0.95;
        ctx.fillStyle = "#fbbf24";
        ctx.fillText(text, p.px + 13, p.py);
      }
    }

    // Bottom-left HUD
    if (frame.angles) {
      const xf = frame.angles.x_factor?.toFixed(1) ?? "—";
      const sp = frame.angles.spine_tilt?.toFixed(1) ?? "—";
      ctx.globalAlpha = 0.8;
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      ctx.beginPath(); ctx.roundRect(8, H - 56, 150, 48, 8); ctx.fill();
      ctx.font = "bold 11px system-ui, sans-serif";
      ctx.globalAlpha = 0.9;
      ctx.fillStyle = "#a78bfa";
      ctx.fillText(`X-Factor: ${xf}°`, 16, H - 36);
      ctx.fillStyle = "#fbbf24";
      ctx.fillText(`${lang === "zh" ? "脊柱倾斜" : "Spine"}: ${sp}°`, 16, H - 18);
    }

    // Top-right phase badge
    ctx.globalAlpha = 0.85;
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    const phaseText = `${phase.icon} ${lang === "zh" ? phase.zh : phase.en}`;
    const ptw = ctx.measureText(phaseText).width;
    ctx.beginPath(); ctx.roundRect(W - ptw - 28, 8, ptw + 20, 28, 6); ctx.fill();
    ctx.font = "bold 12px system-ui, sans-serif";
    ctx.fillStyle = "#fbbf24";
    ctx.globalAlpha = 0.95;
    ctx.fillText(phaseText, W - ptw - 18, 27);

    // Timestamp
    if (frame.timestamp !== undefined) {
      ctx.globalAlpha = 0.5;
      ctx.font = "10px monospace";
      ctx.fillStyle = "#fff";
      ctx.fillText(`${frame.timestamp.toFixed(2)}s`, 12, 20);
    }

    ctx.globalAlpha = 1;
  }, [frame, frames, currentFrame, overlayOpacity, showAngles, lang, phase]);

  useEffect(() => {
    const id = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(id);
  }, [draw]);

  useEffect(() => {
    if (playing) {
      playRef.current = setInterval(() => {
        setCurrentFrame(f => {
          if (f + 1 >= frames.length) { setPlaying(false); return f; }
          return f + 1;
        });
      }, playSpeed);
    }
    return () => { if (playRef.current) clearInterval(playRef.current); };
  }, [playing, frames.length, playSpeed]);

  function jumpToPhase(i: number) {
    const p = PHASES[i];
    const target = Math.round((p.pct[0] + p.pct[1]) / 2 / 100 * (frames.length - 1));
    setCurrentFrame(Math.min(target, frames.length - 1));
    setPlaying(false);
  }

  function playSwingFlow() {
    setCurrentFrame(0);
    setPlaySpeed(350);
    setPlaying(true);
  }

  if (!frame) return null;

  return (
    <div className="glass-card overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-white/5 px-4 py-3">
        <div className="flex rounded-lg border border-white/10 bg-white/[0.03] p-0.5">
          <button onClick={() => setViewMode("pose")}
            className={`rounded-md px-3 py-1 text-[11px] font-semibold transition ${viewMode === "pose" ? "bg-brand-purple/30 text-white" : "text-white/40"}`}>
            {lang === "zh" ? "3D 姿态" : "3D Pose"}
          </button>
          <button onClick={() => setViewMode("flow")}
            className={`rounded-md px-3 py-1 text-[11px] font-semibold transition ${viewMode === "flow" ? "bg-brand-gold/30 text-brand-gold" : "text-white/40"}`}>
            Swing Flow
          </button>
        </div>
        <div className="flex items-center gap-1.5">
          {hasImages && (
            <div className="flex items-center gap-1.5 rounded-lg border border-white/10 px-2 py-1">
              <span className="text-[9px] text-white/30">{lang === "zh" ? "人像" : "Person"}</span>
              <input type="range" min={20} max={100} value={overlayOpacity * 100}
                onChange={e => setOverlayOpacity(Number(e.target.value) / 100)}
                className="w-14 accent-brand-purple h-1" />
            </div>
          )}
          <button onClick={() => setShowAngles(!showAngles)}
            className={`rounded-lg border px-2 py-1 text-[10px] transition ${showAngles ? "border-brand-gold/30 bg-brand-gold/10 text-brand-gold" : "border-white/10 text-white/30"}`}>
            {lang === "zh" ? "角度" : "Angles"}
          </button>
        </div>
      </div>

      {/* Canvas */}
      <div className="relative">
        <canvas ref={canvasRef}
          className="h-80 w-full sm:h-[440px]"
        />
        {!hasImages && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <p className="rounded-lg bg-black/50 px-4 py-2 text-xs text-white/30">
              {lang === "zh" ? "视频帧加载中..." : "Loading video frames..."}
            </p>
          </div>
        )}
      </div>

      {/* Swing Flow phase navigation */}
      {viewMode === "flow" && (
        <div className="border-t border-white/5">
          <div className="relative px-2 py-3 overflow-x-auto scrollbar-hide">
            <div className="flex gap-0.5 min-w-max">
              {PHASES.map((p, i) => {
                const isActive = i === activePhase;
                const isPast = i < activePhase;
                return (
                  <button key={p.id} onClick={() => jumpToPhase(i)}
                    className={`relative flex flex-col items-center px-2.5 py-1.5 rounded-lg transition-all ${
                      isActive ? "bg-brand-gold/15 ring-1 ring-brand-gold/30" : isPast ? "opacity-50" : "opacity-35 hover:opacity-70"
                    }`}>
                    <span className="text-lg leading-none">{p.icon}</span>
                    <span className={`mt-1 text-[9px] font-bold whitespace-nowrap ${isActive ? "text-brand-gold" : "text-white/50"}`}>
                      {lang === "zh" ? p.zh : p.en}
                    </span>
                    {isActive && <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 h-0.5 w-5 rounded-full bg-brand-gold" />}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="mx-3 mb-3 rounded-xl border border-brand-gold/10 bg-brand-gold/[0.04] px-4 py-3">
            <div className="flex items-start gap-3">
              <span className="text-2xl flex-shrink-0">{phase.icon}</span>
              <div>
                <h4 className="text-sm font-bold text-brand-gold">{lang === "zh" ? phase.zh : phase.en}</h4>
                <p className="mt-1 text-xs text-white/50 leading-relaxed">
                  {lang === "zh" ? phase.tipZh : phase.tipEn}
                </p>
              </div>
            </div>
          </div>

          <div className="flex justify-center pb-3">
            <button onClick={playSwingFlow}
              className="flex items-center gap-2 rounded-full border border-brand-gold/20 bg-brand-gold/10 px-5 py-2 text-xs font-semibold text-brand-gold transition hover:bg-brand-gold/20">
              <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24"><polygon points="5,3 19,12 5,21" /></svg>
              {lang === "zh" ? "播放完整挥杆" : "Play Swing Flow"}
            </button>
          </div>
        </div>
      )}

      {/* Playback bar */}
      {frames.length > 1 && (
        <div className="flex items-center gap-2 border-t border-white/5 px-4 py-2.5">
          <button onClick={() => { setPlaying(!playing); if (!playing) setPlaySpeed(250); }}
            className="flex h-8 w-8 items-center justify-center rounded-full bg-brand-purple/20 text-brand-purple transition hover:bg-brand-purple/30">
            {playing ? (
              <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="4" width="4" height="16" /><rect x="14" y="4" width="4" height="16" />
              </svg>
            ) : (
              <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24">
                <polygon points="5,3 19,12 5,21" />
              </svg>
            )}
          </button>
          <button onClick={() => setCurrentFrame(f => Math.max(0, f - 1))} disabled={currentFrame === 0}
            className="flex h-6 w-6 items-center justify-center rounded bg-white/5 text-white/40 text-xs hover:bg-white/10 disabled:opacity-20">◀</button>
          <button onClick={() => setCurrentFrame(f => Math.min(frames.length - 1, f + 1))} disabled={currentFrame >= frames.length - 1}
            className="flex h-6 w-6 items-center justify-center rounded bg-white/5 text-white/40 text-xs hover:bg-white/10 disabled:opacity-20">▶</button>

          {/* Frame scrubber with phase markers */}
          <div className="relative flex-1 h-6 flex items-center">
            <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-1.5 rounded-full bg-white/10 overflow-hidden">
              <div className="h-full rounded-full bg-gradient-to-r from-brand-purple to-brand-gold transition-all"
                style={{ width: `${(currentFrame / Math.max(frames.length - 1, 1)) * 100}%` }} />
            </div>
            {PHASES.map((p, i) => {
              const left = (p.pct[0] + p.pct[1]) / 2;
              return (
                <div key={i} className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 pointer-events-none"
                  style={{ left: `${left}%` }}>
                  <div className={`h-2.5 w-0.5 rounded-full ${i === activePhase ? "bg-brand-gold" : "bg-white/15"}`} />
                </div>
              );
            })}
            <input type="range" min={0} max={frames.length - 1} value={currentFrame}
              onChange={e => { setCurrentFrame(Number(e.target.value)); setPlaying(false); }}
              className="absolute inset-0 w-full opacity-0 cursor-pointer" />
          </div>

          <span className="min-w-[3rem] text-right text-[10px] text-white/40">{currentFrame + 1}/{frames.length}</span>

          <select value={playSpeed} onChange={e => setPlaySpeed(Number(e.target.value))}
            className="rounded bg-white/5 border border-white/10 px-1.5 py-0.5 text-[10px] text-white/40 outline-none">
            <option value={100}>2x</option>
            <option value={180}>1.5x</option>
            <option value={250}>1x</option>
            <option value={400}>0.6x</option>
            <option value={600}>0.4x</option>
          </select>
        </div>
      )}

      {/* Legend */}
      <div className="flex items-center gap-3 border-t border-white/5 px-4 py-2 text-[10px] text-white/40">
        <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-[#d4af37]" />{lang === "zh" ? "头部" : "Head"}</span>
        <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-[#a78bfa]" />{lang === "zh" ? "上肢" : "Upper"}</span>
        <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-[#f87171]" />{lang === "zh" ? "下肢" : "Lower"}</span>
        <span className="ml-auto text-[9px] text-white/20">{lang === "zh" ? "真人姿态重建" : "Real Pose Reconstruction"}</span>
      </div>
    </div>
  );
}
