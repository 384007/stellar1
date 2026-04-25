"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type ToggleState = {
  skeleton: boolean;
  club: boolean;
  shaft: boolean;
  impact: boolean;
  hud: boolean;
  ball: boolean;
};

type Point2D = { frame_index: number; timestamp: number; nx: number; ny: number; confidence: number; source: string };
type BodyJoint = { name: string; nx: number; ny: number; visibility: number };
type BodyFrame = { frame_index: number; timestamp: number; joints: BodyJoint[]; confidence: number; source: string };
type Joint3D = { name: string; x: number; y: number; z: number; visibility: number };
type Pose3DFrame = { frame_index: number; timestamp: number; joints: Joint3D[] };
type Path3DPoint = { frame_index: number; timestamp: number; x: number; y: number; z: number; confidence: number; source: string };
type ShotTracerResponse = {
  video?: { fps?: number };
  phases?: { impact_t?: number };
  paths?: {
    club_head_2d?: Point2D[];
    ball_flight_2d?: Point2D[];
    body_2d?: BodyFrame[];
    skeleton_3d?: Pose3DFrame[];
    club_head_3d?: Path3DPoint[];
    ball_flight_3d?: Path3DPoint[];
  };
  metrics?: {
    estimated_launch_angle_deg?: number;
    estimated_carry_yards?: number;
    estimated_apex_yards?: number;
    estimated_lateral_curve_yards?: number;
    confidence?: number;
    estimated_club_head_speed_mph?: number;
    estimated_ball_speed_mph?: number;
    tempo_ratio?: string;
    scores?: { overall_score?: number };
  };
};

export default function ShotTracerPage() {
  const [mainFile, setMainFile] = useState<File | null>(null);
  const [frontFile, setFrontFile] = useState<File | null>(null);
  const [sideFile, setSideFile] = useState<File | null>(null);
  const [mode, setMode] = useState("single_video");
  const [calibration, setCalibration] = useState("{}");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<ShotTracerResponse | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [rotY, setRotY] = useState(15);
  const [autoRotate, setAutoRotate] = useState(true);
  const [toggles, setToggles] = useState<ToggleState>({ skeleton: true, club: true, shaft: true, impact: true, hud: true, ball: true });

  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    if (!autoRotate) return;
    const tick = () => {
      setRotY((v) => (v + 0.5) % 360);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, [autoRotate]);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [videoTime, setVideoTime] = useState(0);

  const clubPath = result?.paths?.club_head_2d || [];
  const ballPath = result?.paths?.ball_flight_2d || [];
  const bodyPath = result?.paths?.body_2d || [];
  const impactT = result?.phases?.impact_t || 0;

  const visibleBallPath = useMemo<Point2D[]>(() => {
    if (!ballPath.length) return [];
    return ballPath.filter((p) => p.timestamp <= videoTime + 0.02);
  }, [ballPath, videoTime]);

  async function onAnalyze() {
    if (!mainFile) {
      setError("请先上传主视频");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const fd = new FormData();
      fd.append("file", mainFile);
      if (frontFile) fd.append("front_view", frontFile);
      if (sideFile) fd.append("side_view", sideFile);
      if (calibration.trim()) fd.append("calibration_json", calibration);
      fd.append("mode", mode);

      const res = await fetch("/api/shot-tracer/reconstruct", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "分析失败");
      setResult(data);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto max-w-7xl px-4 py-8 text-white">
      <h1 className="text-2xl font-bold">Stellar Shot Tracer 3D Reconstruction</h1>
      <p className="mt-1 text-sm text-white/60">真实视频分析 + 可替换适配器 + video-based estimate（非雷达）。</p>

      <section className="mt-6 grid gap-3 rounded-xl border border-white/10 bg-black/30 p-4 md:grid-cols-2">
        <label className="text-sm">主视频<input type="file" accept="video/*" className="mt-1 block w-full" onChange={(e) => { const f = e.target.files?.[0] || null; setMainFile(f); if (videoUrl) URL.revokeObjectURL(videoUrl); setVideoUrl(f ? URL.createObjectURL(f) : null); }} /></label>
        <label className="text-sm">正面视频（可选）<input type="file" accept="video/*" className="mt-1 block w-full" onChange={(e) => setFrontFile(e.target.files?.[0] || null)} /></label>
        <label className="text-sm">侧面视频（可选）<input type="file" accept="video/*" className="mt-1 block w-full" onChange={(e) => setSideFile(e.target.files?.[0] || null)} /></label>
        <label className="text-sm">模式
          <select value={mode} onChange={(e) => setMode(e.target.value)} className="mt-1 w-full rounded bg-black/40 p-2">
            <option value="single_video">single_video</option><option value="dual_camera">dual_camera</option><option value="high_speed">high_speed</option><option value="trellis_asset">trellis_asset</option><option value="postshot_scene">postshot_scene</option>
          </select>
        </label>
        <label className="text-sm md:col-span-2">Calibration JSON（可选）
          <textarea value={calibration} onChange={(e) => setCalibration(e.target.value)} className="mt-1 h-20 w-full rounded bg-black/40 p-2 font-mono text-xs" />
        </label>
        <button onClick={onAnalyze} disabled={loading} className="rounded-lg bg-amber-500 px-4 py-2 font-semibold text-black disabled:opacity-50">{loading ? "重建中..." : "开始重建"}</button>
        {error && <p className="text-sm text-red-300">{error}</p>}
      </section>

      <section className="mt-6 grid gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-white/10 bg-black/30 p-3">
          <div className="mb-2 flex flex-wrap gap-2 text-xs">
            {Object.entries(toggles).map(([k, v]) => (
              <label key={k} className="rounded border border-white/20 px-2 py-1"><input type="checkbox" checked={v} onChange={() => setToggles((s) => ({ ...s, [k]: !s[k as keyof ToggleState] }))} className="mr-1" />{k}</label>
            ))}
          </div>
          <div className="relative aspect-video overflow-hidden rounded-lg bg-black">
            {videoUrl ? <video ref={videoRef} src={videoUrl} controls className="h-full w-full object-contain" onTimeUpdate={(e) => setVideoTime((e.target as HTMLVideoElement).currentTime)} /> : <div className="grid h-full place-items-center text-sm text-white/40">上传视频后预览</div>}
            <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox="0 0 1 1" preserveAspectRatio="none">
              <defs>
                <linearGradient id="clubGrad" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stopColor="#fde047"/><stop offset="55%" stopColor="#fb923c"/><stop offset="100%" stopColor="#ef4444"/></linearGradient>
                <linearGradient id="ballGrad" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stopColor="#fef9c3"/><stop offset="100%" stopColor="#facc15"/></linearGradient>
                <filter id="glow"><feGaussianBlur stdDeviation="0.006" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
              </defs>
              {toggles.club && clubPath.length > 1 && <polyline points={clubPath.map((p) => `${p.nx},${p.ny}`).join(" ")} stroke="url(#clubGrad)" strokeWidth="0.006" fill="none" filter="url(#glow)" strokeLinecap="round" />}
              {toggles.ball && visibleBallPath.length > 1 && <polyline points={visibleBallPath.map((p) => `${p.nx},${p.ny}`).join(" ")} stroke="url(#ballGrad)" strokeWidth="0.004" fill="none" filter="url(#glow)" />}
              {toggles.impact && <circle cx={clubPath.find((p) => Math.abs(p.timestamp - impactT) < 0.04)?.nx || 0} cy={clubPath.find((p) => Math.abs(p.timestamp - impactT) < 0.04)?.ny || 0} r="0.012" fill="#fef08a" filter="url(#glow)" />}
              {toggles.skeleton && bodyPath[Math.floor(videoTime * (result?.video?.fps || 30))]?.joints?.map((j) => <circle key={j.name} cx={j.nx} cy={j.ny} r="0.004" fill="rgba(56,189,248,0.9)" />)}
            </svg>
          </div>
          {toggles.hud && result?.metrics && (
            <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
              <div>Impact: {result.phases?.impact_t?.toFixed?.(3)}s</div><div>Launch: {result.metrics?.estimated_launch_angle_deg}° (estimated)</div>
              <div>Carry: {result.metrics?.estimated_carry_yards} yd</div><div>Apex: {result.metrics?.estimated_apex_yards} yd</div>
              <div>Curve: {result.metrics?.estimated_lateral_curve_yards} yd</div><div>Confidence: {result.metrics?.confidence}</div>
            </div>
          )}
        </div>

        <div className="rounded-xl border border-white/10 bg-black/30 p-3">
          <div className="mb-2 flex items-center justify-between"><h3 className="font-semibold">3D Skeleton / Path Viewer (SVG Projection)</h3><label className="text-xs"><input type="checkbox" checked={autoRotate} onChange={(e) => setAutoRotate(e.target.checked)} className="mr-1"/>自动旋转</label></div>
          <input type="range" min={0} max={360} value={rotY} onChange={(e) => setRotY(Number(e.target.value))} className="w-full" />
          <svg className="mt-2 h-[340px] w-full rounded bg-slate-950" viewBox="-1 -1 2 2">
            {result?.paths?.skeleton_3d?.[0]?.joints?.map((j) => {
              const a = (rotY / 180) * Math.PI;
              const x = j.x * Math.cos(a) + j.z * Math.sin(a);
              const z = -j.x * Math.sin(a) + j.z * Math.cos(a);
              const y = -j.y;
              const depthScale = 0.8 + Math.max(-0.3, Math.min(0.3, z));
              return <circle key={j.name} cx={x} cy={y} r={0.012 * depthScale} fill="rgba(56,189,248,0.8)" />;
            })}
            <polyline points={(result?.paths?.club_head_3d || []).map((p) => {
              const a = (rotY / 180) * Math.PI;
              const x = p.x * Math.cos(a) + p.z * Math.sin(a);
              const y = -p.y;
              return `${x},${y}`;
            }).join(" ")} fill="none" stroke="#f59e0b" strokeWidth={0.01} />
            <polyline points={(result?.paths?.ball_flight_3d || []).map((p) => {
              const a = (rotY / 180) * Math.PI;
              const x = p.x * Math.cos(a) + p.z * Math.sin(a);
              const y = -p.y;
              return `${x},${y}`;
            }).join(" ")} fill="none" stroke="#fef08a" strokeWidth={0.01} />
          </svg>
        </div>
      </section>

      {result?.metrics && (
        <section className="mt-6 grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
          {[
            ["Club Head Speed", `${result.metrics.estimated_club_head_speed_mph} mph`],
            ["Ball Speed", `${result.metrics.estimated_ball_speed_mph} mph`],
            ["Launch Angle", `${result.metrics.estimated_launch_angle_deg}°`],
            ["Carry", `${result.metrics.estimated_carry_yards} yd`],
            ["Apex", `${result.metrics.estimated_apex_yards} yd`],
            ["Curve", `${result.metrics.estimated_lateral_curve_yards} yd`],
            ["Tempo", result.metrics.tempo_ratio],
            ["Swing Score", `${result.metrics?.scores?.overall_score}`],
            ["Confidence", `${Math.round((result.metrics.confidence || 0) * 100)}%`],
          ].map(([k, v]) => <div key={k} className="rounded border border-white/10 bg-black/30 p-3"><div className="text-xs text-white/50">{k}</div><div className="mt-1 text-lg font-semibold text-amber-300">{v}</div></div>)}
        </section>
      )}
    </main>
  );
}
