"use client";

import { useState, useEffect } from "react";
import KeyframeStrip from "@/components/KeyframeStrip";
import HUDOverlay from "@/components/HUDOverlay";
import Skeleton3DViewer from "@/components/Skeleton3DViewer";
import ProComparison from "@/components/ProComparison";
import SimAnimation from "@/components/SimAnimation";
import PlusResultView, { type PlusAnalysisResult } from "@/components/PlusResultView";
import Prov3PlusVideoRenderer from "@/components/prov3/Prov3PlusVideoRenderer";
import VideoAnalysisOverlay from "@/components/VideoAnalysisOverlay";
import { coachingTipsFromParsed } from "@/lib/video-analysis-coaching";
import { normalizePoseFramesForOverlay } from "@/lib/analysis-pose-storage";
import { normalizeProv3MediaInRaw } from "@/lib/prov3-media-url";
import {
  isProv3StrictMediaPolicyResult,
  prov3DisplayKeyframeRows,
  type Prov3ResultLike,
} from "@/lib/prov3-keyframe-media";

interface SharedRecord {
  id: string;
  type: string;
  total_score: number;
  result_json: string;
  created_at: string;
  has_video: boolean;
  result_stale?: boolean;
  result_partial?: boolean;
  image_missing?: boolean;
  result_source?: string;
}

interface ParsedResult {
  scores?: Record<string, number>;
  total_score?: number;
  issues?: string[];
  issues_zh?: string[];
  suggestions?: string[];
  suggestions_zh?: string[];
  summary?: string;
  summary_zh?: string;
  keyframes?: Array<{
    phase: string;
    label_en: string;
    label_zh: string;
    timestamp: number;
    image_base64?: string;
    keyframe_image_url?: string;
    visual_diff_from_prev?: number;
    phase_validation_passed?: boolean;
  }>;
  preview_keyframes?: ParsedResult["keyframes"];
  official_phase_keyframes?: ParsedResult["keyframes"];
  analysis_id?: string;
  final_status?: string;
  analysis_trust?: string;
  trust_level?: string;
  low_trust_preview_only?: boolean;
  video_meta?: { source_frame_count?: number };
  skeleton_data?: {
    frames: Array<Record<string, unknown>>;
    total_frames: number;
  };
  pose_frames?: Array<{
    joints: Array<{ name: string; x: number; y: number; z: number; visibility: number; normalized: { x: number; y: number } }>;
    connections: number[][];
    angles: Record<string, number>;
    frame_size: { width: number; height: number };
    frame_index: number;
    timestamp: number;
    image_base64?: string;
  }>;
  prediction?: {
    predicted_distance: number;
    lateral_offset: number;
    shot_shape: string;
    shot_shape_zh: string;
    club_head_speed: number;
    ball_speed: number;
    launch_angle: number;
    spin_rate: number;
    smash_factor: number;
    trajectory: Array<{ t: number; x: number; y: number; lateral: number }>;
    club_type?: string;
    club_group?: string;
    hand?: "R" | "L" | "UNKNOWN";
    hand_confidence?: number;
    baseline_distance?: number;
    technique_multiplier?: number;
    strike_multiplier?: number;
    speed_multiplier?: number;
    distance_confidence?: number;
    distance_debug?: Record<string, unknown>;
  };
  training_plan?: Record<string, { focus: string; drills: string[]; duration: string }>;
  primary_diagnosis?: {
    title_zh?: string;
    title_en?: string;
    status_zh?: string;
    status_en?: string;
    ai_confidence?: number;
  };
  analysis_reliability?: {
    level?: "high" | "medium" | "low";
    capped_confidence?: number;
    reasons?: string[];
  };
  phase_source?: string;
  keyframe_warning?: string;
  result_partial?: boolean;
  analysis_mode?: string;
  /** e.g. ``prov3`` — Stellar Pro v3 product pipeline */
  pipeline?: string;
  phase_pipeline_applied?: boolean;
  phase_evaluations_reliable?: boolean;
  phase_evaluations_warning?: string | null;
  phase_warning_zh?: string;
  phase_warning_en?: string;
  phase_boundary?: {
    phase_strip_is_monotonic_fallback_only?: boolean;
    phase_vision_complete_strip?: boolean;
    phase_keyframe_extraction_label?: string;
    keyframe_strip_frame_count?: number;
    ai_vision_frame_count?: number;
    expected_phase_vision_frames?: number;
    analysis_route_tier?: string;
    plus_grade_phase_evaluations?: boolean;
    gemini_uniform_map_vs_strip?: {
      gemini_uniform_thumbnail_map_applies?: boolean;
      gemini_map_aligned_with_final_strip?: boolean | null;
      aligned?: boolean | null;
      strip_divergence_reason?: string | null;
    };
    phase_strip_technically_sound?: boolean;
  };
  /** Passed through from packed Plus API for history/share replay */
  gemini_observation?: Record<string, unknown>;
}

interface CurveFrame {
  angles: Record<string, number>;
  timestamp: number;
}

function buildCurveFrames(parsed: ParsedResult, totalScoreFallback = 0): CurveFrame[] {
  if (parsed.pose_frames && parsed.pose_frames.length > 1) {
    return parsed.pose_frames.map((f, i) => ({
      angles: f.angles || {},
      timestamp: typeof f.timestamp === "number" ? f.timestamp : i * 33,
    }));
  }
  if (parsed.skeleton_data?.frames && parsed.skeleton_data.frames.length > 1) {
    const frames = parsed.skeleton_data.frames
      .map((f, i) => {
        const frame = f as Record<string, unknown>;
        const stats = (frame.stats as Record<string, unknown> | undefined) || frame;
        const toNum = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : undefined);
        const angles: Record<string, number> = {};
        const xf = toNum(stats.x_factor); if (xf !== undefined) angles.x_factor = xf;
        const sh = toNum(stats.shoulder_rotation); if (sh !== undefined) angles.shoulder_rotation = sh;
        const hp = toNum(stats.hip_rotation); if (hp !== undefined) angles.hip_rotation = hp;
        const sp = toNum(stats.spine_tilt); if (sp !== undefined) angles.spine_tilt = sp;
        return { angles, timestamp: i * 33 };
      })
      .filter((f) => Object.keys(f.angles).length > 0);
    if (frames.length > 1) return frames;
  }
  const s = parsed.scores || {};
  const stance = s.stance || totalScoreFallback || 60;
  const backswing = s.backswing || totalScoreFallback || 60;
  const downswing = s.downswing || totalScoreFallback || 60;
  const follow = s.follow_through || totalScoreFallback || 60;
  return [
    { timestamp: 0, angles: { x_factor: 8, shoulder_rotation: -4, hip_rotation: -2, spine_tilt: 11 } },
    { timestamp: 1, angles: { x_factor: backswing * 0.36, shoulder_rotation: -backswing * 0.35, hip_rotation: -stance * 0.16, spine_tilt: 10.5 } },
    { timestamp: 2, angles: { x_factor: backswing * 0.5, shoulder_rotation: -backswing * 0.52, hip_rotation: -stance * 0.28, spine_tilt: 10 } },
    { timestamp: 3, angles: { x_factor: downswing * 0.3, shoulder_rotation: -downswing * 0.24, hip_rotation: -downswing * 0.4, spine_tilt: 9 } },
    { timestamp: 4, angles: { x_factor: follow * 0.12, shoulder_rotation: follow * 0.22, hip_rotation: follow * 0.35, spine_tilt: 8 } },
    { timestamp: 5, angles: { x_factor: 5, shoulder_rotation: follow * 0.35, hip_rotation: follow * 0.42, spine_tilt: 7 } },
  ];
}

function ProTrainingCurve({ frames, keyframes, lang }: {
  frames: CurveFrame[];
  keyframes?: ParsedResult["keyframes"];
  lang: "en" | "zh";
}) {
  if (frames.length < 2) return null;
  const VW = 1000; const VH = 220;
  const pad = { top: 16, right: 16, bottom: 36, left: 40 };
  const chartW = VW - pad.left - pad.right;
  const chartH = VH - pad.top - pad.bottom;
  const n = frames.length;

  const toSeries = (key: string) =>
    frames.map((f, i) => ({
      x: pad.left + (chartW * i) / Math.max(1, n - 1),
      y: typeof f.angles?.[key] === "number" ? (f.angles[key] as number) : 0,
    }));

  const xFactor = toSeries("x_factor");
  const shoulder = toSeries("shoulder_rotation");
  const hip = toSeries("hip_rotation");
  const spine = toSeries("spine_tilt");

  const allVals = [...xFactor, ...shoulder, ...hip, ...spine].map((p) => p.y);
  const minV = Math.min(...allVals, 0);
  const maxV = Math.max(...allVals, 1);
  const range = maxV - minV || 1;
  const toY = (v: number) => pad.top + chartH - ((v - minV) / range) * chartH;

  const smoothPath = (series: Array<{ x: number; y: number }>) => {
    if (series.length < 2) return "";
    let d = `M ${series[0].x.toFixed(1)} ${toY(series[0].y).toFixed(1)}`;
    for (let i = 0; i < series.length - 1; i++) {
      const p0 = series[i]; const p1 = series[i + 1];
      const cx = ((p0.x + p1.x) / 2).toFixed(1);
      const cy = ((toY(p0.y) + toY(p1.y)) / 2).toFixed(1);
      d += ` Q ${p0.x.toFixed(1)} ${toY(p0.y).toFixed(1)} ${cx} ${cy}`;
    }
    const last = series[series.length - 1];
    d += ` L ${last.x.toFixed(1)} ${toY(last.y).toFixed(1)}`;
    return d;
  };

  const phaseDefs = [
    { keys: ["address", "setup"], zh: "准备", en: "Setup" },
    { keys: ["backswing", "takeaway"], zh: "上杆", en: "Back" },
    { keys: ["top"], zh: "顶点", en: "Top" },
    { keys: ["downswing"], zh: "下杆", en: "Down" },
    { keys: ["impact"], zh: "击球", en: "Impact" },
    { keys: ["follow_through", "finish"], zh: "收杆", en: "Finish" },
  ];

  const phaseMarkers = phaseDefs.map((phase, i) => {
    let idx = Math.round((i / Math.max(1, phaseDefs.length - 1)) * (n - 1));
    if (keyframes && keyframes.length > 0) {
      const matched = keyframes.find((k) => phase.keys.includes((k.phase || "").toLowerCase()));
      if (matched) {
        const ts = matched.timestamp ?? 0;
        let best = 0; let bestDiff = Infinity;
        for (let j = 0; j < n; j++) {
          const d = Math.abs((frames[j].timestamp ?? 0) - ts);
          if (d < bestDiff) { bestDiff = d; best = j; }
        }
        idx = best;
      }
    }
    const clamped = Math.max(0, Math.min(n - 1, idx));
    const x = pad.left + (chartW * clamped) / Math.max(1, n - 1);
    return { x, idx: clamped, label: lang === "zh" ? phase.zh : phase.en };
  });

  return (
    <div className="mb-4 rounded-xl border border-white/5 bg-black/20 p-4">
      <div className="mb-2 flex items-center justify-between">
        <h4 className="text-sm font-semibold text-white">
          {lang === "zh" ? "训练曲线图（挥杆过程）" : "Training Curves (Swing Timeline)"}
        </h4>
        <div className="flex items-center gap-3 text-[10px]">
          <span className="text-brand-gold">● X-Factor</span>
          <span className="text-purple-400">● Shoulder</span>
          <span className="text-cyan-400">● Hip</span>
          <span className="text-green-400">● Spine</span>
        </div>
      </div>
      <svg viewBox={`0 0 ${VW} ${VH}`} style={{ width: "100%", height: "auto", display: "block" }} preserveAspectRatio="none">
        {[0, 1, 2, 3, 4].map((i) => {
          const val = minV + (range * i) / 4;
          const y = pad.top + chartH - (chartH * i) / 4;
          return (
            <g key={i}>
              <line x1={pad.left} y1={y} x2={VW - pad.right} y2={y} stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
              <text x={pad.left - 5} y={y + 4} fill="rgba(255,255,255,0.35)" fontSize="18" textAnchor="end">{val.toFixed(0)}</text>
            </g>
          );
        })}
        <path d={smoothPath(xFactor)}  stroke="#d4af37" strokeWidth="3" fill="none" strokeLinecap="round" />
        <path d={smoothPath(shoulder)} stroke="#a855f7" strokeWidth="3" fill="none" strokeLinecap="round" />
        <path d={smoothPath(hip)}      stroke="#06b6d4" strokeWidth="3" fill="none" strokeLinecap="round" />
        <path d={smoothPath(spine)}    stroke="#22c55e" strokeWidth="3" fill="none" strokeLinecap="round" />
        {phaseMarkers.map((m, i) => {
          const yDot = toY(xFactor[m.idx]?.y ?? 0);
          const labelY = pad.top + chartH + (i % 2 === 0 ? 20 : 32);
          return (
            <g key={i}>
              <line x1={m.x} y1={pad.top} x2={m.x} y2={pad.top + chartH} stroke="rgba(245,197,24,0.2)" strokeWidth="1.5" strokeDasharray="6 5" />
              <circle cx={m.x} cy={yDot} r="5" fill="#f5c518" stroke="rgba(255,255,255,0.7)" strokeWidth="1.5" />
              <text x={m.x} y={labelY} fill="rgba(255,255,255,0.55)" fontSize="17" textAnchor="middle">{m.label}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function parseResult(json: string): ParsedResult {
  try {
    const r = JSON.parse(json) as Record<string, unknown>;
    if (!r || typeof r !== "object") return {};
    normalizeProv3MediaInRaw(r);
    return r as ParsedResult;
  } catch {
    return {};
  }
}

function getUserAngles(parsed: ParsedResult) {
  if (!parsed.pose_frames?.length) return {};
  const mid = parsed.pose_frames[Math.floor(parsed.pose_frames.length / 2)];
  return mid?.angles || {};
}

function formatDate(dateStr: string, lang: "en" | "zh"): string {
  try {
    const d = new Date(dateStr);
    if (lang === "zh") return `${d.getFullYear()}年${d.getMonth() + 1}月${d.getDate()}日`;
    return d.toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
  } catch { return dateStr; }
}

export default function SharePageClient({ token }: { token: string }) {
  const [record, setRecord] = useState<SharedRecord | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lang, setLang] = useState<"en" | "zh">("zh");
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!token) return;
    fetch(`/api/share/${token}`)
      .then((res) => {
        if (!res.ok) { throw new Error(res.status === 404 ? "link_invalid" : "fetch_failed"); }
        return res.json();
      })
      .then((data) => { setRecord(data as SharedRecord); setLoading(false); })
      .catch((err: Error) => { setError(err.message); setLoading(false); });
  }, [token]);

  const shareUrl = typeof window !== "undefined" ? window.location.href : "";

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(shareUrl);
    } catch {
      const el = document.createElement("textarea");
      el.value = shareUrl; document.body.appendChild(el); el.select();
      document.execCommand("copy"); document.body.removeChild(el);
    }
    setCopied(true); setTimeout(() => setCopied(false), 2000);
  };

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-brand-dark">
        <div className="flex flex-col items-center gap-4">
          <div className="h-10 w-10 animate-spin rounded-full border-2 border-white/10 border-t-brand-purple" />
          <p className="text-sm text-white/40">{lang === "zh" ? "加载报告中..." : "Loading report..."}</p>
        </div>
      </div>
    );
  }

  if (error || !record) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-6 bg-brand-dark px-6 text-center">
        <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-white/5">
          <svg className="h-10 w-10 text-white/20" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z" />
          </svg>
        </div>
        <div>
          <h2 className="text-lg font-semibold text-white/70">{lang === "zh" ? "分享链接无效" : "Invalid Share Link"}</h2>
          <p className="mt-1 text-sm text-white/35">{lang === "zh" ? "该链接可能已失效或不存在" : "This link may have expired or does not exist"}</p>
        </div>
        <a href="/" className="rounded-xl bg-brand-purple/80 px-6 py-2.5 text-sm font-medium text-white transition hover:bg-brand-purple">
          {lang === "zh" ? "前往 Stellar" : "Go to Stellar"}
        </a>
      </div>
    );
  }

  const parsed = parseResult(record.result_json);
  const shareIsProv3Product = isProv3StrictMediaPolicyResult(parsed as Prov3ResultLike);
  const shareStripKeyframes = shareIsProv3Product
    ? prov3DisplayKeyframeRows(parsed as Prov3ResultLike)
    : (parsed.keyframes ?? []);
  const shareOverlayPoses = normalizePoseFramesForOverlay(parsed.pose_frames);
  const curveFrames = buildCurveFrames(parsed, record.total_score);

  // Determine reliability level
  const reliability = parsed.analysis_reliability?.level;
  const aiConfidence = parsed.primary_diagnosis?.ai_confidence;
  const isLowReliability = reliability === "low" || (typeof aiConfidence === "number" && aiConfidence < 50);

  const scoreKeys = ["grip", "stance", "backswing", "downswing", "follow_through"];
  const scoreLabels: Record<string, { en: string; zh: string }> = {
    grip: { en: "Grip", zh: "握杆" },
    stance: { en: "Stance", zh: "站姿" },
    backswing: { en: "Backswing", zh: "后摆" },
    downswing: { en: "Downswing", zh: "下杆" },
    follow_through: { en: "Follow", zh: "收杆" },
  };
  const typeColor = record.type === "pro" ? "#d4af37" : record.type === "plus" ? "#a855f7" : "#7c3aed";
  const videoSrc = record.has_video ? `/api/share/video/${token}` : null;

  return (
    <div className="min-h-screen bg-brand-dark">
      {/* Top bar */}
      <div className="sticky top-0 z-20 border-b border-white/5 bg-brand-dark/90 backdrop-blur-sm">
        <div className="mx-auto flex max-w-2xl items-center justify-between px-4 py-3">
          <a href="/" className="flex items-center gap-2">
            <span className="font-rajdhani text-lg font-bold tracking-widest text-white">
              STELLAR<span className="text-brand-purple">.</span>
            </span>
          </a>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setLang(lang === "zh" ? "en" : "zh")}
              className="rounded-lg border border-white/10 px-2.5 py-1 text-xs text-white/40 transition hover:text-white/60"
            >
              {lang === "zh" ? "EN" : "中文"}
            </button>
            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 rounded-lg bg-brand-purple/20 px-3 py-1.5 text-xs font-medium text-brand-purple transition hover:bg-brand-purple/30"
            >
              <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M7.217 10.907a2.25 2.25 0 1 0 0 2.186m0-2.186c.18.324.283.696.283 1.093s-.103.77-.283 1.093m0-2.186 9.566-5.314m-9.566 7.5 9.566 5.314m0 0a2.25 2.25 0 1 0 3.935 2.186 2.25 2.25 0 0 0-3.935-2.186zm0-12.814a2.25 2.25 0 1 0 3.933-2.185 2.25 2.25 0 0 0-3.933 2.185z" />
              </svg>
              {copied ? (lang === "zh" ? "✓ 已复制" : "✓ Copied") : (lang === "zh" ? "复制链接" : "Copy Link")}
            </button>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-2xl space-y-4 px-4 py-6">
        {/* Header card */}
        <div className="glass-card p-5">
          <div className="flex items-center gap-4">
            <div
              className="flex h-16 w-16 flex-shrink-0 items-center justify-center rounded-2xl"
              style={{ background: `conic-gradient(${typeColor} ${record.total_score}%, rgba(255,255,255,0.05) ${record.total_score}%)` }}
            >
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-brand-dark">
                <span className="text-xl font-bold" style={{ color: typeColor }}>{record.total_score}</span>
              </div>
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <span className="rounded px-2 py-0.5 text-xs font-bold" style={{ background: typeColor + "22", color: typeColor }}>
                  {record.type.toUpperCase()}
                </span>
                <span className="text-xs text-white/30">{formatDate(record.created_at, lang)}</span>
              </div>
              <p className="mt-1.5 text-sm font-semibold text-white">
                {lang === "zh" ? "高尔夫挥杆分析报告" : "Golf Swing Analysis Report"}
              </p>
              <p className="text-xs text-white/40">{lang === "zh" ? "由 Stellar AI 生成" : "Generated by Stellar AI"}</p>
            </div>
          </div>

          {parsed.scores && (
            <div className="mt-4 grid grid-cols-5 gap-2">
              {scoreKeys.map((key) => (
                <div key={key} className="text-center">
                  <p className="text-base font-bold" style={{ color: typeColor }}>
                    {(parsed.scores as Record<string, number>)[key] || 0}
                  </p>
                  <div className="mx-auto mt-1 h-1 w-full rounded-full bg-white/5">
                    <div className="h-full rounded-full" style={{ width: `${(parsed.scores as Record<string, number>)[key] || 0}%`, background: typeColor, opacity: 0.7 }} />
                  </div>
                  <p className="mt-1 text-[10px] text-white/40">
                    {lang === "zh" ? scoreLabels[key]?.zh : scoreLabels[key]?.en}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Video — with pose overlay when pose_frames available (same as history page) */}
        {/* Stale result warning — shown when stored result has broken keyframes from before the fix */}
        {(record.result_partial || record.image_missing || (parsed.keyframe_warning && String(parsed.keyframe_warning).trim())) && (
          <div className="rounded-xl border border-white/10 bg-white/[0.03] px-4 py-2.5">
            <p className="text-[10px] font-medium text-white/50">
              {lang === "zh" ? "分享数据提示" : "Share payload notice"}
            </p>
            <ul className="mt-1 space-y-1 text-[11px] text-white/40">
              {record.image_missing ? (
                <li>
                  {lang === "zh"
                    ? "关键帧图片未包含在此条分享数据中；下方会显示骨架或占位。"
                    : "Keyframe images are missing from this shared record; skeleton or placeholders are shown below."}
                </li>
              ) : null}
              {record.result_partial && !record.image_missing ? (
                <li>
                  {lang === "zh"
                    ? "此条为精简或部分同步的结果，若云端有完整副本，系统会优先使用。"
                    : "This record may be a compact or partially synced result; a fuller copy is preferred when available."}
                </li>
              ) : null}
              {parsed.keyframe_warning && String(parsed.keyframe_warning).trim() ? (
                <li className="text-white/45">{parsed.keyframe_warning}</li>
              ) : null}
            </ul>
          </div>
        )}

        {/* Phase pipeline trust — backend fields; never pretend fallback / partial vision is Plus-validated */}
        {(() => {
          const pb = parsed.phase_boundary;
          const imgOnly = parsed.analysis_mode === "image_only";
          const litePose = record.type === "lite" && parsed.analysis_mode === "pose_lite";
          const plusPro = record.type === "plus" || record.type === "pro";
          const notTrusted = plusPro && parsed.phase_evaluations_reliable !== true;
          const monotonic = pb?.phase_strip_is_monotonic_fallback_only === true;
          const warnIncomplete = String(parsed.phase_evaluations_warning || "").includes("incomplete_phase_vision_strip");
          const incomplete = pb?.phase_vision_complete_strip === false || warnIncomplete;
          const geminiDiv =
            pb?.gemini_uniform_map_vs_strip?.aligned === false ||
            pb?.gemini_uniform_map_vs_strip?.gemini_map_aligned_with_final_strip === false;
          if (!imgOnly && !litePose && !notTrusted && !monotonic && !incomplete && !geminiDiv) return null;
          const linesZh: string[] = [];
          const linesEn: string[] = [];
          if (imgOnly) {
            linesZh.push(parsed.phase_warning_zh || "无姿态阶段管线：纯图像分析，关键帧未通过阶段校验。");
            linesEn.push(parsed.phase_warning_en || "No pose phase pipeline: image-only analysis; not phase-validated keyframes.");
          }
          if (litePose) {
            linesZh.push("Lite 分析：与 Plus 不同级；不提供 Plus 式逐阶段 AI 评估，阶段条仅供辅助参考。");
            linesEn.push("Lite analysis: not equivalent to Plus; no Plus-grade per-phase AI evaluations — keyframe strip is supporting context only.");
          }
          if (monotonic) {
            linesZh.push("单调兜底 ≠ 语义顶点/击球：仅按时间单调选取 pose，不代表语义级顶点或击球检测成功。");
            linesEn.push("Monotonic fallback ≠ semantic top/impact: frames are time-ordered pose picks only, not event-validated top or impact.");
          }
          if (incomplete) {
            linesZh.push(
              `阶段条不完整（关键帧 ${pb?.keyframe_strip_frame_count ?? "?"} / 模型图 ${pb?.ai_vision_frame_count ?? "?"}，期望 ${pb?.expected_phase_vision_frames ?? 8}）。逐阶段结论已降级。`,
            );
            linesEn.push(
              `Incomplete phase strip (keyframes ${pb?.keyframe_strip_frame_count ?? "?"} / model images ${pb?.ai_vision_frame_count ?? "?"}, expected ${pb?.expected_phase_vision_frames ?? 8}). Per-phase claims are downgraded.`,
            );
          }
          if (geminiDiv) {
            linesZh.push("均匀采样缩略图的 Gemini 相位映射与最终 8 张阶段图不一致；已关闭高可信逐阶段视觉结论。");
            linesEn.push("Gemini uniform-thumbnail phase map diverged from the final 8 phase images; high-trust per-phase vision is off.");
          }
          if (notTrusted && !monotonic && !incomplete && !geminiDiv && !imgOnly && !litePose) {
            linesZh.push("阶段视觉未通过服务器可信性门槛（语义/校验/来源）；逐阶段评价仅供参考。");
            linesEn.push("Phase vision did not pass server trust gates; treat per-phase notes as non-authoritative.");
          }
          if (linesZh.length === 0) return null;
          return (
            <div className="rounded-xl border border-amber-500/35 bg-amber-500/10 px-4 py-3">
              <p className="text-[10px] font-semibold uppercase tracking-wide text-amber-200/90">
                {lang === "zh" ? "阶段关键帧可信度" : "Phase keyframe trust"}
              </p>
              <ul className="mt-2 space-y-1.5 text-[11px] text-amber-100/85">
                {(lang === "zh" ? linesZh : linesEn).map((t, i) => (
                  <li key={i}>{t}</li>
                ))}
              </ul>
              {parsed.phase_evaluations_warning ? (
                <p className="mt-2 font-mono text-[10px] text-amber-200/50">{String(parsed.phase_evaluations_warning)}</p>
              ) : null}
            </div>
          );
        })()}

        {record?.result_stale && (
          <div className="rounded-xl border border-orange-500/30 bg-orange-500/10 px-4 py-3">
            <div className="flex items-start gap-2">
              <svg className="mt-0.5 h-4 w-4 flex-shrink-0 text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99" />
              </svg>
              <div>
                <p className="text-xs font-medium text-orange-300">
                  {lang === "zh" ? "关键帧数据已过期" : "Keyframe Data Outdated"}
                </p>
                <p className="mt-0.5 text-[11px] text-orange-300/60">
                  {lang === "zh"
                    ? "此结果在系统升级前生成，关键帧可能重复。请重新上传视频以获取准确分析。"
                    : "This result was generated before a system upgrade. Keyframes may be duplicated. Please re-upload the video for accurate analysis."}
                </p>
              </div>
            </div>
          </div>
        )}

        {isLowReliability && (
          <div className="rounded-xl border border-yellow-500/30 bg-yellow-500/10 px-4 py-3">
            <div className="flex items-start gap-2">
              <svg className="mt-0.5 h-4 w-4 flex-shrink-0 text-yellow-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
              </svg>
              <div>
                <p className="text-xs font-medium text-yellow-300">
                  {lang === "zh" ? "分析可信度较低" : "Low Analysis Reliability"}
                </p>
                <p className="mt-0.5 text-[11px] text-yellow-300/60">
                  {lang === "zh"
                    ? `本次分析置信度为 ${aiConfidence ?? "?"}%，结果仅供参考。建议重新录制更清晰的视频后再次分析。`
                    : `Analysis confidence is ${aiConfidence ?? "?"}%. Results are for reference only. Consider re-recording a clearer video.`}
                </p>
                {parsed.analysis_reliability?.reasons && parsed.analysis_reliability.reasons.length > 0 && (
                  <p className="mt-1 text-[10px] text-yellow-300/40">
                    {parsed.analysis_reliability.reasons.map(r => r.replace(/_/g, " ")).join(" · ")}
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Video — with pose overlay when pose_frames available (same as history page) */}
        {videoSrc && (
          shareOverlayPoses.length > 0 ? (
            <div className="mb-4">
              {shareIsProv3Product ? (
                <Prov3PlusVideoRenderer videoSrc={videoSrc} result={parsed} lang={lang} />
              ) : (
                <VideoAnalysisOverlay
                  videoSrc={videoSrc}
                  poseFrames={shareOverlayPoses}
                  lang={lang}
                  coachingTips={coachingTipsFromParsed(parsed, record.type)}
                  prediction={parsed.prediction as { predicted_distance?: number; shot_shape?: string; shot_shape_zh?: string; club_head_speed?: number; club_type?: string; hand?: "R" | "L" | "UNKNOWN" } | undefined}
                  sourceFrameCount={parsed.video_meta?.source_frame_count}
                  skeletonStyle={record.type === "pro" || record.type === "plus" ? "plus" : "legacy"}
                />
              )}
            </div>
          ) : (
            <div className="overflow-hidden rounded-xl border border-white/10 bg-black/30">
              <video
                className="h-full max-h-[420px] w-full bg-black object-contain"
                src={videoSrc}
                controls
                playsInline
                preload="metadata"
              />
            </div>
          )
        )}

        {/* Plus — full PlusResultView */}
        {record.type === "plus" ? (
          <PlusResultView result={parsed as PlusAnalysisResult} lang={lang} externalVideoSrc={videoSrc} />
        ) : (
          <>
            {/* Keyframe strip */}
            {shareStripKeyframes.length > 0 && (
              <KeyframeStrip
                keyframes={shareStripKeyframes as NonNullable<ParsedResult["keyframes"]>}
                lang={lang}
                urlOnlyTimeline={shareIsProv3Product}
                plusStyleKeyframeSkeleton={shareIsProv3Product}
              />
            )}

            {/* Skeleton HUD — show impact frame (most informative) or mid-point */}
            {parsed.skeleton_data?.frames && parsed.skeleton_data.frames.length > 0 && (() => {
              const frames = parsed.skeleton_data.frames;
              // Find impact frame by phase_id, or use ~60% through the sequence
              const impactIdx = frames.findIndex((f: Record<string, unknown>) => {
                const pd = f.phase_data as Record<string, unknown> | undefined;
                return pd?.phase_id === "impact";
              });
              const bestIdx = impactIdx >= 0 ? impactIdx : Math.min(Math.floor(frames.length * 0.6), frames.length - 1);
              return (
                <div className="rounded-xl border border-white/5 bg-black/20 p-4">
                  <h4 className="mb-3 text-sm font-semibold text-white">
                    {lang === "zh" ? "骨架 HUD 回放" : "Skeleton HUD Replay"}
                  </h4>
                  <HUDOverlay
                    hudData={frames[bestIdx] as Record<string, unknown>}
                    showExtended={record.type === "pro"}
                    mode={record.type === "pro" ? "pro" : "lite"}
                    lang={lang}
                  />
                </div>
              );
            })()}

            {/* 3D skeleton */}
            {parsed.pose_frames && parsed.pose_frames.length > 0 && (
              <Skeleton3DViewer frames={parsed.pose_frames} lang={lang} />
            )}

            {/* Training curve */}
            <ProTrainingCurve
              frames={curveFrames}
              keyframes={shareStripKeyframes as ParsedResult["keyframes"]}
              lang={lang}
            />

            {/* Summary */}
            {(parsed.summary_zh || parsed.summary) && (
              <div className="rounded-lg bg-white/[0.02] p-4">
                <p className="text-xs leading-relaxed text-white/50">
                  {lang === "zh" ? parsed.summary_zh : parsed.summary}
                </p>
              </div>
            )}

            {/* Issues */}
            {((lang === "zh" ? parsed.issues_zh : parsed.issues) || []).length > 0 && (
              <div className="glass-card p-4">
                <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-red-400/70">
                  {lang === "zh" ? "问题" : "Issues"}
                </p>
                <ul className="space-y-1">
                  {((lang === "zh" ? parsed.issues_zh : parsed.issues) || []).map((issue, i) => (
                    <li key={i} className="flex items-start gap-1.5 text-xs text-white/50">
                      <span className="mt-0.5 text-red-400/50">●</span>{issue}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Suggestions */}
            {((lang === "zh" ? parsed.suggestions_zh : parsed.suggestions) || []).length > 0 && (
              <div className="glass-card p-4">
                <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-brand-gold/70">
                  {lang === "zh" ? "建议" : "Suggestions"}
                </p>
                <ul className="space-y-1">
                  {((lang === "zh" ? parsed.suggestions_zh : parsed.suggestions) || []).map((sug, i) => (
                    <li key={i} className="flex items-start gap-1.5 text-xs text-white/50">
                      <span className="mt-0.5 text-brand-gold/50">◆</span>{sug}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Sim animation */}
            {parsed.prediction && parsed.prediction.predicted_distance > 0 && (
              <SimAnimation prediction={parsed.prediction} lang={lang} isPro={record.type === "pro"} />
            )}

            {/* Pro comparison */}
            {parsed.scores && record.type === "pro" && (
              <ProComparison userScores={parsed.scores} userAngles={getUserAngles(parsed)} lang={lang} />
            )}

            {/* Training plan */}
            {parsed.training_plan && (
              <div className="rounded-xl border border-brand-gold/15 bg-black/20 p-4">
                <h4 className="mb-3 text-sm font-semibold text-brand-gold">
                  {lang === "zh" ? "训练计划" : "Training Plan"}
                </h4>
                <div className="grid gap-3 sm:grid-cols-2">
                  {Object.entries(parsed.training_plan).map(([day, plan]) => (
                    <div key={day} className="rounded-xl border border-brand-gold/10 bg-black/30 p-3">
                      <div className="mb-2 flex items-center justify-between">
                        <span className="rounded-full bg-brand-gold/20 px-2 py-0.5 text-[10px] font-semibold text-brand-gold">{day.toUpperCase()}</span>
                        <span className="text-[10px] text-white/40">{plan.duration}</span>
                      </div>
                      <p className="mb-1 text-xs font-semibold text-white">{plan.focus}</p>
                      <ul className="space-y-1">
                        {plan.drills.slice(0, 3).map((drill, i) => (
                          <li key={i} className="text-[11px] text-white/45">- {drill}</li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {/* CTA footer */}
        <div className="glass-card p-5 text-center">
          <p className="text-sm font-semibold text-white">
            {lang === "zh" ? "想分析你自己的挥杆？" : "Want to analyze your own swing?"}
          </p>
          <p className="mt-1 text-xs text-white/40">
            {lang === "zh" ? "Stellar AI — 专业高尔夫挥杆分析平台" : "Stellar AI — Professional Golf Swing Analysis"}
          </p>
          <a href="/" className="mt-4 inline-block rounded-xl bg-brand-purple px-6 py-2.5 text-sm font-semibold text-white transition hover:bg-brand-purple/80">
            {lang === "zh" ? "免费开始分析" : "Start Free Analysis"}
          </a>
        </div>
      </div>
    </div>
  );
}
