"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { buildShotTracerTips, type ShotTracerUiResult } from "@/lib/shot-tracer-ui";

type Lang = "en" | "zh";

type Props = {
  lang: Lang;
  videoUrl: string | null;
  result: ShotTracerUiResult | null;
  busy: boolean;
  statusText: string;
  onRun: () => void;
  runDisabled: boolean;
  error: string;
};

export default function AiShotTracerPanel({ lang, videoUrl, result, busy, statusText, onRun, runDisabled, error }: Props) {
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [showClubPath, setShowClubPath] = useState(true);
  const [showBallPath, setShowBallPath] = useState(true);
  const [showImpact, setShowImpact] = useState(true);
  const [showHud, setShowHud] = useState(true);
  const [show3D, setShow3D] = useState(true);
  const [autoRotate, setAutoRotate] = useState(true);
  const [rotY, setRotY] = useState(20);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [videoTime, setVideoTime] = useState(0);

  const fps = result?.video?.fps || 30;
  const currentBodyFrame = result?.paths?.body_2d?.[Math.max(0, Math.floor(videoTime * fps))];
  const impactT = result?.phases?.impact_t || 0;

  const visibleBall = useMemo(() => {
    const full = result?.paths?.ball_flight_2d || [];
    return full.filter((p) => p.timestamp <= videoTime + 0.02);
  }, [result?.paths?.ball_flight_2d, videoTime]);

  const tips = useMemo(() => buildShotTracerTips(result?.metrics, lang), [result?.metrics, lang]);

  useEffect(() => {
    if (!autoRotate || !show3D) return;
    let raf = 0;
    const tick = () => {
      setRotY((v) => (v + 0.45) % 360);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [autoRotate, show3D]);

  return (
    <div className="space-y-4">
      <div className="glass-card p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-semibold text-white">AI Shot Tracer</h3>
            <p className="text-[11px] text-white/35">{lang === "zh" ? "Video-based Estimate" : "Video-based Estimate"}</p>
          </div>
          <button
            type="button"
            onClick={onRun}
            disabled={runDisabled || busy}
            className="rounded-lg border border-brand-gold/30 bg-brand-gold/15 px-4 py-2 text-xs font-semibold text-brand-gold transition hover:bg-brand-gold/25 disabled:opacity-50"
          >
            {busy ? (lang === "zh" ? "AI 重建中..." : "Reconstructing...") : (lang === "zh" ? "开始 AI 重建" : "Start AI Reconstruction")}
          </button>
        </div>
        {statusText && <p className="mb-2 text-xs text-white/45">{statusText}</p>}
        {error && <p className="mb-2 text-xs text-red-300">{error}</p>}

        <div className="mb-3 flex flex-wrap gap-2 text-[11px] text-white/60">
          {[
            [showSkeleton, setShowSkeleton, lang === "zh" ? "身体骨架" : "Body Skeleton"],
            [showClubPath, setShowClubPath, lang === "zh" ? "杆头轨迹" : "Club Path"],
            [showBallPath, setShowBallPath, lang === "zh" ? "球路轨迹" : "Ball Flight"],
            [showImpact, setShowImpact, lang === "zh" ? "击球瞬间" : "Impact Moment"],
            [showHud, setShowHud, lang === "zh" ? "数据面板" : "Data HUD"],
            [show3D, setShow3D, "3D Swing Reconstruction"],
          ].map(([v, set, label]) => (
            <label key={String(label)} className="rounded border border-white/10 px-2 py-1">
              <input type="checkbox" checked={Boolean(v)} onChange={() => (set as (x: boolean) => void)(!Boolean(v))} className="mr-1" />
              {label as string}
            </label>
          ))}
        </div>

        <div className="relative aspect-video overflow-hidden rounded-xl border border-white/10 bg-black">
          {videoUrl ? (
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              className="h-full w-full object-contain"
              onTimeUpdate={(e) => setVideoTime((e.target as HTMLVideoElement).currentTime)}
            />
          ) : (
            <div className="grid h-full place-items-center text-sm text-white/40">{lang === "zh" ? "请先上传视频" : "Upload video first"}</div>
          )}

          <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox="0 0 1 1" preserveAspectRatio="none">
            <defs>
              <linearGradient id="clubPathGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#fde047" /><stop offset="50%" stopColor="#fb923c" /><stop offset="100%" stopColor="#ef4444" />
              </linearGradient>
              <linearGradient id="ballPathGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#fef9c3" /><stop offset="100%" stopColor="#facc15" />
              </linearGradient>
              <filter id="shotGlow"><feGaussianBlur stdDeviation="0.006" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
            </defs>

            {showClubPath && (result?.paths?.club_head_2d || []).length > 1 && (
              <polyline points={(result?.paths?.club_head_2d || []).map((p) => `${p.nx},${p.ny}`).join(" ")} stroke="url(#clubPathGrad)" strokeWidth="0.006" fill="none" filter="url(#shotGlow)" strokeLinecap="round" />
            )}
            {showBallPath && visibleBall.length > 1 && (
              <polyline points={visibleBall.map((p) => `${p.nx},${p.ny}`).join(" ")} stroke="url(#ballPathGrad)" strokeWidth="0.004" fill="none" filter="url(#shotGlow)" />
            )}
            {showImpact && (
              <circle
                cx={(result?.paths?.club_head_2d || []).find((p) => Math.abs(p.timestamp - impactT) < 0.04)?.nx || 0}
                cy={(result?.paths?.club_head_2d || []).find((p) => Math.abs(p.timestamp - impactT) < 0.04)?.ny || 0}
                r="0.012"
                fill="#fff59d"
                filter="url(#shotGlow)"
              />
            )}
            {showSkeleton && currentBodyFrame?.joints?.map((j) => (
              <circle key={j.name} cx={j.nx} cy={j.ny} r="0.004" fill="rgba(56,189,248,0.95)" />
            ))}
          </svg>
        </div>

        {showHud && result?.metrics && (
          <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-white/70 md:grid-cols-3">
            <div>Impact: {result.phases?.impact_t?.toFixed(3)}s</div>
            <div>Launch: {result.metrics.estimated_launch_angle_deg}° (Estimated)</div>
            <div>Carry: {result.metrics.estimated_carry_yards} yd (Estimated)</div>
            <div>Apex: {result.metrics.estimated_apex_yards} yd (Estimated)</div>
            <div>Curve: {result.metrics.estimated_lateral_curve_yards} yd (Estimated)</div>
            <div>Confidence: {Math.round((result.metrics.confidence || 0) * 100)}%</div>
          </div>
        )}
      </div>

      {show3D && (
        <div className="glass-card p-5">
          <div className="mb-3 flex items-center justify-between gap-3">
            <h4 className="text-sm font-semibold text-white">3D Swing Reconstruction</h4>
            <label className="text-xs text-white/60"><input type="checkbox" checked={autoRotate} onChange={(e) => setAutoRotate(e.target.checked)} className="mr-1" />{lang === "zh" ? "自动旋转" : "Auto rotate"}</label>
          </div>
          <input type="range" min={0} max={360} value={rotY} onChange={(e) => setRotY(Number(e.target.value))} className="w-full" />
          <svg className="mt-2 h-[320px] w-full rounded-lg bg-slate-950" viewBox="-1 -1 2 2">
            {(result?.paths?.skeleton_3d?.[0]?.joints || []).map((j) => {
              const a = (rotY / 180) * Math.PI;
              const x = j.x * Math.cos(a) + j.z * Math.sin(a);
              const z = -j.x * Math.sin(a) + j.z * Math.cos(a);
              const y = -j.y;
              const size = 0.011 * (0.8 + Math.max(-0.3, Math.min(0.3, z)));
              return <circle key={j.name} cx={x} cy={y} r={size} fill="rgba(56,189,248,0.85)" />;
            })}
            <polyline points={(result?.paths?.club_head_3d || []).map((p) => {
              const a = (rotY / 180) * Math.PI;
              return `${p.x * Math.cos(a) + p.z * Math.sin(a)},${-p.y}`;
            }).join(" ")} fill="none" stroke="#f59e0b" strokeWidth={0.01} />
            <polyline points={(result?.paths?.ball_flight_3d || []).map((p) => {
              const a = (rotY / 180) * Math.PI;
              return `${p.x * Math.cos(a) + p.z * Math.sin(a)},${-p.y}`;
            }).join(" ")} fill="none" stroke="#fde047" strokeWidth={0.01} />
          </svg>
        </div>
      )}

      {result?.metrics && (
        <div className="grid grid-cols-2 gap-2 md:grid-cols-3 lg:grid-cols-5">
          {[
            ["Club Speed", `${result.metrics.estimated_club_head_speed_mph} mph (Estimated)`],
            ["Ball Speed", `${result.metrics.estimated_ball_speed_mph} mph (Estimated)`],
            ["Launch Angle", `${result.metrics.estimated_launch_angle_deg}° (Estimated)`],
            ["Carry", `${result.metrics.estimated_carry_yards} yd (Estimated)`],
            ["Apex", `${result.metrics.estimated_apex_yards} yd (Estimated)`],
            ["Curve", `${result.metrics.estimated_lateral_curve_yards} yd (Estimated)`],
            ["Tempo", result.metrics.tempo_ratio],
            ["Swing Score", `${result.metrics.scores?.overall_score ?? "--"}`],
            ["Confidence", `${Math.round((result.metrics.confidence || 0) * 100)}%`],
          ].map(([k, v]) => (
            <div key={String(k)} className="glass-card p-3">
              <p className="text-[11px] text-white/45">{k as string}</p>
              <p className="mt-1 text-sm font-semibold text-brand-gold">{v as string}</p>
            </div>
          ))}
        </div>
      )}

      <div className="glass-card p-5">
        <h4 className="text-sm font-semibold text-white">{lang === "zh" ? "训练建议" : "Training Suggestions"}</h4>
        <ul className="mt-2 space-y-2 text-xs text-white/65">
          {tips.map((tip) => <li key={tip}>• {tip}</li>)}
        </ul>
      </div>
    </div>
  );
}
