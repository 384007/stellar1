"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import UploadZone from "@/components/UploadZone";
import AnalysisWaiting from "@/components/AnalysisWaiting";
import ScreenModeCapture from "@/components/ScreenModeCapture";
import { preloadPoseModel } from "@/lib/mediapipe-assets";
import { fetchWithRetry, makeFormData } from "@/lib/fetch-retry";
import { readLiteAnalyzeResult } from "@/lib/read-lite-analyze-response";
import {
  appendLiteAnalyzeFileToFormData,
  isVideoFile,
  uploadVideoForAnalysis,
} from "@/lib/upload-video";
import type {
  LabMetrics,
  LabIssue,
  LabDrill,
  LabResult,
  LabPrediction,
  LabHistoryItem,
  LabQuotaResponse,
  FieldsVisibility,
  LabTrendPoint,
} from "@/lib/lab-types";
import {
  consumeReanalyzeFromHistoryPayload,
  fetchLabVideoBlobForReanalyze,
  reanalyzeHistoryFilename,
} from "@/lib/reanalyze-from-history";
import { devWarn } from "@/lib/dev-only-log";

type Stage = "upload" | "processing" | "results";
type Lang = "en" | "zh";
type InputMode = "upload" | "capture" | "screen";
type ResultTab = "metrics" | "trajectory" | "issues" | "drills" | "report" | "compare" | "trend";

interface LabResponse {
  job_id: string;
  status: string;
  tier: string;
  report_tier: string;
  result: LabResult;
  quota: LabQuotaResponse;
}

// ── Reusable sub-components ──

function MetricCard({
  label, labelZh, value, unit, confidence, source, lang, locked,
}: {
  label: string; labelZh: string; value: number | null; unit: string;
  confidence?: number; source?: string; lang: Lang; locked?: boolean;
}) {
  if (locked) {
    return (
      <div className="rounded-xl border border-white/5 bg-black/30 p-3 text-center relative overflow-hidden">
        <p className="text-[10px] text-white/40">{lang === "zh" ? labelZh : label}</p>
        <p className="mt-1 text-xl font-bold text-white/10">—</p>
        <p className="text-[10px] text-white/20">{unit}</p>
        <div className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-[2px]">
          <span className="rounded-full bg-brand-gold/10 border border-brand-gold/20 px-2 py-0.5 text-[9px] text-brand-gold">PRO</span>
        </div>
      </div>
    );
  }
  return (
    <div className="rounded-xl border border-white/5 bg-black/30 p-3 text-center">
      <p className="text-[10px] text-white/40">{lang === "zh" ? labelZh : label}</p>
      <p className="mt-1 text-xl font-bold text-brand-gold">
        {value != null ? (typeof value === "number" ? (Number.isInteger(value) ? value : value.toFixed(1)) : value) : "—"}
      </p>
      <p className="text-[10px] text-white/30">{unit}</p>
      {source && <p className="mt-0.5 text-[8px] text-white/15 italic">{source}</p>}
      {confidence != null && confidence > 0 && (
        <div className="mt-1 flex items-center justify-center gap-1">
          <div className="h-1 w-8 rounded-full bg-white/10 overflow-hidden">
            <div
              className={`h-full rounded-full ${confidence >= 0.7 ? "bg-green-500" : confidence >= 0.4 ? "bg-yellow-500" : "bg-red-400"}`}
              style={{ width: `${confidence * 100}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function ProLockCard({ lang, title, titleZh, description, descriptionZh }: {
  lang: Lang; title: string; titleZh: string; description: string; descriptionZh: string;
}) {
  return (
    <div className="rounded-xl border border-brand-gold/10 bg-gradient-to-br from-brand-gold/5 to-transparent p-5 text-center">
      <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-brand-gold/10 border border-brand-gold/20">
        <svg className="h-5 w-5 text-brand-gold/60" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 1 0-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 0 0 2.25-2.25v-6.75a2.25 2.25 0 0 0-2.25-2.25H6.75a2.25 2.25 0 0 0-2.25 2.25v6.75a2.25 2.25 0 0 0 2.25 2.25Z" />
        </svg>
      </div>
      <h4 className="text-sm font-semibold text-white/70">{lang === "zh" ? titleZh : title}</h4>
      <p className="mt-1 text-xs text-white/35">{lang === "zh" ? descriptionZh : description}</p>
      <a href="/pro-login" className="mt-3 inline-block rounded-lg bg-brand-gold/15 border border-brand-gold/25 px-4 py-1.5 text-xs font-semibold text-brand-gold transition hover:bg-brand-gold/25">
        {lang === "zh" ? "解锁 Pro" : "Unlock Pro"}
      </a>
    </div>
  );
}

function ShotTracerPlaceholder({ lang, locked }: { lang: Lang; locked: boolean }) {
  return (
    <div className="glass-card p-5 relative overflow-hidden">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-base font-semibold text-white">{lang === "zh" ? "弹道轨迹" : "Shot Tracer"}</h3>
        <span className="text-[9px] text-white/20 italic">{lang === "zh" ? "视频估算" : "video estimate"}</span>
      </div>
      <div className="relative h-40 rounded-xl bg-gradient-to-t from-green-900/20 via-transparent to-blue-900/10 border border-white/5 flex items-center justify-center">
        <div className="absolute inset-0 overflow-hidden rounded-xl">
          <svg className="w-full h-full" viewBox="0 0 400 160" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path
              d="M 40 140 Q 120 20, 200 30 Q 280 40, 360 120"
              stroke="rgba(234,179,8,0.3)"
              strokeWidth="2"
              strokeDasharray="6 4"
              fill="none"
            />
            <circle cx="40" cy="140" r="4" fill="rgba(234,179,8,0.5)" />
            <circle cx="360" cy="120" r="3" fill="rgba(234,179,8,0.3)" />
          </svg>
        </div>
        {locked && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/50 backdrop-blur-[3px] rounded-xl">
            <span className="rounded-full bg-brand-gold/10 border border-brand-gold/20 px-3 py-1 text-[10px] font-bold text-brand-gold mb-2">PRO</span>
            <p className="text-[10px] text-white/40">{lang === "zh" ? "完整弹道轨迹与叠加层" : "Full trajectory overlay"}</p>
          </div>
        )}
        {!locked && (
          <p className="text-xs text-white/30 z-10">{lang === "zh" ? "基础弹道预览（基于 AI 估算）" : "Basic trajectory (AI estimated)"}</p>
        )}
      </div>
    </div>
  );
}

function SwingTimeline({ metrics, lang, locked }: { metrics: LabMetrics; lang: Lang; locked: boolean }) {
  const backswing = metrics.backswing_time_sec;
  const downswing = metrics.downswing_time_sec;
  const total = (backswing || 0) + (downswing || 0);
  const phases = [
    { key: "address", label: "Address", labelZh: "站位", pct: 5, color: "bg-blue-500/40" },
    { key: "backswing", label: "Backswing", labelZh: "后摆", pct: backswing && total ? (backswing / total) * 70 : 35, color: "bg-purple-500/50" },
    { key: "top", label: "Top", labelZh: "顶点", pct: 5, color: "bg-yellow-500/50" },
    { key: "downswing", label: "Downswing", labelZh: "下杆", pct: downswing && total ? (downswing / total) * 70 : 35, color: "bg-red-500/50" },
    { key: "impact", label: "Impact", labelZh: "击球", pct: 5, color: "bg-brand-gold/60" },
    { key: "finish", label: "Finish", labelZh: "收杆", pct: 10, color: "bg-green-500/40" },
  ];

  return (
    <div className="glass-card p-5 relative overflow-hidden">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-base font-semibold text-white">{lang === "zh" ? "挥杆时间轴" : "Swing Timeline"}</h3>
        {backswing && downswing && (
          <span className="text-[10px] text-white/30">
            {backswing.toFixed(2)}s / {downswing.toFixed(2)}s
          </span>
        )}
      </div>
      <div className="flex h-6 rounded-lg overflow-hidden border border-white/5">
        {phases.map((p) => (
          <div key={p.key} className={`${p.color} flex items-center justify-center transition-all`} style={{ width: `${p.pct}%` }}>
            <span className="text-[7px] text-white/60 truncate px-0.5">{lang === "zh" ? p.labelZh : p.label}</span>
          </div>
        ))}
      </div>
      {locked && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-[2px] rounded-xl">
          <div className="text-center">
            <span className="rounded-full bg-brand-gold/10 border border-brand-gold/20 px-3 py-1 text-[10px] font-bold text-brand-gold">PRO</span>
            <p className="mt-1 text-[10px] text-white/40">{lang === "zh" ? "展开详细时间轴" : "Expand detailed timeline"}</p>
          </div>
        </div>
      )}
    </div>
  );
}

function TrendChart({ points, lang }: { points: LabTrendPoint[]; lang: Lang }) {
  if (!points.length) {
    return (
      <div className="glass-card p-5 text-center">
        <p className="text-sm text-white/40">{lang === "zh" ? "暂无趋势数据" : "No trend data yet"}</p>
      </div>
    );
  }

  const validSpeeds = points.filter(p => p.ball_speed_mph != null).map(p => p.ball_speed_mph!);
  const maxSpeed = Math.max(...validSpeeds, 1);
  const minSpeed = Math.min(...validSpeeds, 0);
  const range = maxSpeed - minSpeed || 1;

  return (
    <div className="glass-card p-5">
      <h4 className="mb-3 text-sm font-semibold text-white">{lang === "zh" ? "球速趋势" : "Ball Speed Trend"}</h4>
      <div className="relative h-32">
        <svg className="w-full h-full" viewBox={`0 0 ${points.length * 40} 120`} preserveAspectRatio="none">
          {points.map((p, i) => {
            if (p.ball_speed_mph == null) return null;
            const x = i * 40 + 20;
            const y = 110 - ((p.ball_speed_mph - minSpeed) / range) * 100;
            return (
              <g key={p.job_id}>
                <circle cx={x} cy={y} r={3} fill="rgba(234,179,8,0.8)" />
                {i > 0 && points[i - 1].ball_speed_mph != null && (
                  <line
                    x1={(i - 1) * 40 + 20}
                    y1={110 - ((points[i - 1].ball_speed_mph! - minSpeed) / range) * 100}
                    x2={x}
                    y2={y}
                    stroke="rgba(234,179,8,0.4)"
                    strokeWidth={1.5}
                  />
                )}
                <text x={x} y={y - 8} textAnchor="middle" fontSize="8" fill="rgba(255,255,255,0.4)">
                  {p.ball_speed_mph.toFixed(0)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <div className="mt-2 flex justify-between text-[9px] text-white/25">
        <span>{points[0]?.date ? new Date(points[0].date).toLocaleDateString(lang === "zh" ? "zh-CN" : "en-US", { month: "short", day: "numeric" }) : ""}</span>
        <span>{points[points.length - 1]?.date ? new Date(points[points.length - 1].date).toLocaleDateString(lang === "zh" ? "zh-CN" : "en-US", { month: "short", day: "numeric" }) : ""}</span>
      </div>
    </div>
  );
}

// ── Main Page ──

export default function ShotLabPage() {
  const router = useRouter();
  const [stage, setStage] = useState<Stage>("upload");
  const [lang, setLang] = useState<Lang>("zh");
  const [authChecked, setAuthChecked] = useState(false);
  const [username, setUsername] = useState("");
  const [isPro, setIsPro] = useState(false);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState(0);
  const [labResponse, setLabResponse] = useState<LabResponse | null>(null);
  const [history, setHistory] = useState<LabHistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [quotaInfo, setQuotaInfo] = useState<LabQuotaResponse | null>(null);
  const [inputMode, setInputMode] = useState<InputMode>("upload");
  const [liveCapture, setLiveCapture] = useState(false);
  const [screenRecording, setScreenRecording] = useState(false);
  const screenRecRef = useRef<MediaRecorder | null>(null);
  const screenStreamRef = useRef<MediaStream | null>(null);
  const screenChunksRef = useRef<Blob[]>([]);
  const [screenRecTime, setScreenRecTime] = useState(0);
  const screenTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [historyOpenId, setHistoryOpenId] = useState<string | null>(null);
  const historyLoadingRef = useRef(false);
  const [resultTab, setResultTab] = useState<ResultTab>("metrics");
  const [trendData, setTrendData] = useState<LabTrendPoint[]>([]);
  const [trendLoading, setTrendLoading] = useState(false);
  const [compareSelectMode, setCompareSelectMode] = useState(false);
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [compareResult, setCompareResult] = useState<{ shots: Record<string, unknown>[]; diff: Record<string, { a: number | null; b: number | null; delta: number | null }> } | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [exportLoading, setExportLoading] = useState(false);
  /** 防止双击 / 重入导致多次 ``/api/lab`` 任务。 */
  const labAnalysisInFlightRef = useRef(false);

  useEffect(() => {
    const token = localStorage.getItem("stellar_token");
    if (!token) { router.push("/login"); return; }
    const userStr = localStorage.getItem("stellar_user");
    if (userStr) {
      try {
        const u = JSON.parse(userStr);
        setUsername(u.username || u.email || "");
        setIsPro(!!u.is_pro);
      } catch { /* ignore */ }
    }
    const storedLang = localStorage.getItem("stellar_lang");
    if (storedLang === "en" || storedLang === "zh") setLang(storedLang);
    setAuthChecked(true);
    preloadPoseModel();
  }, [router]);

  const loadHistory = useCallback(async () => {
    const token = localStorage.getItem("stellar_token");
    if (!token) return;
    setHistoryLoading(true);
    try {
      const res = await fetch("/api/lab/history", { headers: { Authorization: `Bearer ${token}` } });
      if (res.ok) {
        const data = await res.json();
        setHistory(data.items || []);
        setQuotaInfo(data.quota || null);
      }
    } catch { /* ignore */ }
    setHistoryLoading(false);
  }, []);

  useEffect(() => {
    if (authChecked) void loadHistory();
  }, [authChecked, loadHistory]);

  const openHistoryRecord = useCallback(
    async (item: LabHistoryItem) => {
      if (historyLoadingRef.current) return;
      const token = localStorage.getItem("stellar_token");
      if (!token) { router.push("/login"); return; }
      historyLoadingRef.current = true;
      setHistoryOpenId(item.id);
      setError("");
      try {
        const res = await fetch(`/api/lab/${encodeURIComponent(item.id)}`, { headers: { Authorization: `Bearer ${token}` } });
        let data: Record<string, unknown> = {};
        try { data = await res.json(); } catch { /* non-JSON */ }
        if (!res.ok) {
          setError((data.detail as string) || (lang === "zh" ? "无法加载该记录" : "Could not load this analysis"));
          return;
        }
        if (data.status !== "completed" || !data.result) {
          setError(
            item.status === "failed"
              ? (lang === "zh" ? "该次分析失败，无结果可复盘" : "This run failed; there is no report to open.")
              : (lang === "zh" ? "分析尚未完成，请稍后再试" : "This analysis is not finished yet."),
          );
          return;
        }
        setLabResponse({
          job_id: data.job_id as string,
          status: data.status as string,
          tier: data.tier as string,
          report_tier: (data.report_tier as string) || "free",
          result: data.result as LabResult,
          quota: data.quota as LabQuotaResponse,
        });
        if (data.quota && typeof data.quota === "object") setQuotaInfo(data.quota as LabQuotaResponse);
        setStage("results");
        setResultTab("metrics");
        window.scrollTo({ top: 0, behavior: "smooth" });
      } catch {
        setError(lang === "zh" ? "网络错误，请重试" : "Network error. Try again.");
      } finally {
        historyLoadingRef.current = false;
        setHistoryOpenId(null);
      }
    },
    [lang, router],
  );

  const handleVideoCapture = useCallback((videoBlob: Blob) => {
    setLiveCapture(false);
    handleUpload(new File([videoBlob], "swing-capture.webm", { type: videoBlob.type }));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function startScreenRecording() {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      screenStreamRef.current = stream;
      screenChunksRef.current = [];
      const mime = MediaRecorder.isTypeSupported("video/webm") ? "video/webm" : "video/mp4";
      const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 3_000_000 });
      rec.ondataavailable = (e) => { if (e.data.size > 0) screenChunksRef.current.push(e.data); };
      rec.onstop = () => {
        stream.getTracks().forEach(t => t.stop());
        if (screenTimerRef.current) { clearInterval(screenTimerRef.current); screenTimerRef.current = null; }
        setScreenRecording(false);
        const blob = new Blob(screenChunksRef.current, { type: mime });
        if (blob.size > 0) handleUpload(new File([blob], "screen-capture.webm", { type: mime }));
      };
      rec.start(1000);
      screenRecRef.current = rec;
      setScreenRecording(true);
      setScreenRecTime(0);
      screenTimerRef.current = setInterval(() => {
        setScreenRecTime(t => {
          if (t >= 30) { rec.stop(); return t; }
          return t + 1;
        });
      }, 1000);
    } catch { /* user cancelled */ }
  }

  function stopScreenRecording() {
    const rec = screenRecRef.current;
    if (rec && rec.state !== "inactive") rec.stop();
    if (screenTimerRef.current) { clearInterval(screenTimerRef.current); screenTimerRef.current = null; }
  }

  const fetchUnifiedPrediction = useCallback(async (
    file: File,
    headers: Record<string, string>,
    signal: AbortSignal,
  ): Promise<LabPrediction | null> => {
    try {
      const rid = crypto.randomUUID();
      const headersWithIdem = { ...headers, "X-Stellar-Idempotency-Key": rid };
      const fd = new FormData();
      appendLiteAnalyzeFileToFormData(fd, file, file.name, rid);
      const ctrl = new AbortController();
      const timeout = setTimeout(() => ctrl.abort(), 90_000);
      signal.addEventListener("abort", () => ctrl.abort());
      const res = await fetchWithRetry("/api/lite/analyze-proxy", {
        method: "POST",
        headers: headersWithIdem,
        body: fd,
        signal: ctrl.signal,
        retries: 0,
      });
      clearTimeout(timeout);
      if (!res.ok) return null;
      const data = (await readLiteAnalyzeResult(res)) as {
        prediction?: LabPrediction & { predicted_distance?: number };
      };
      const pred = data.prediction;
      if (!pred || typeof pred.predicted_distance !== "number") return null;
      return pred as LabPrediction;
    } catch {
      return null;
    }
  }, []);

  const submitLabAnalysisFile = useCallback(async (file: File) => {
    if (labAnalysisInFlightRef.current) {
      devWarn("[shot-lab] lab analysis already in flight, ignoring duplicate trigger");
      return;
    }
    labAnalysisInFlightRef.current = true;
    setStage("processing");
    setError("");
    setProgress(0);

    const video = isVideoFile(file, file.name);
    let phaseTarget = video ? 8 : 88;

    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 99) return prev;
        const next = prev + (phaseTarget - prev) * 0.08 + Math.random() * 0.5;
        return Math.min(next, Math.max(prev, phaseTarget));
      });
    }, 800);

    try {
      const token = localStorage.getItem("stellar_token");
      const headers: Record<string, string> = {};
      if (token) headers["Authorization"] = `Bearer ${token}`;

      const jobId = `lab-${crypto.randomUUID()}`;
      const controller = new AbortController();
      const abortTimer = setTimeout(() => controller.abort(), 300_000);

      const unifiedPromise = fetchUnifiedPrediction(file, headers, controller.signal);
      let res: Response;
      try {
        if (video) {
          const up = await uploadVideoForAnalysis(
            file, file.name, headers,
            (pct) => { phaseTarget = pct; },
            controller.signal,
          );
          phaseTarget = 95;
          const unifiedPrediction = await unifiedPromise;
          const fd = new FormData();
          fd.append("upload_token", up.upload_token);
          fd.append("mime_type", up.mime_type);
          fd.append("file", file, file.name);
          fd.append("job_id", jobId);
          if (unifiedPrediction) fd.append("unified_prediction", JSON.stringify(unifiedPrediction));
          res = await fetch("/api/lab", { method: "POST", headers, body: fd, signal: controller.signal });
        } else {
          const unifiedPrediction = await unifiedPromise;
          const fd = makeFormData(file, file.name);
          fd.append("job_id", jobId);
          if (unifiedPrediction) fd.append("unified_prediction", JSON.stringify(unifiedPrediction));
          res = await fetchWithRetry("/api/lab", { method: "POST", headers, body: fd, signal: controller.signal, retries: 2 });
        }
      } finally {
        clearTimeout(abortTimer);
      }

      clearInterval(progressInterval);

      if (!res.ok) {
        let data: Record<string, unknown> = {};
        try { data = await res.json(); } catch { /* HTML error page */ }

        if (data.error === "QUOTA_EXCEEDED") {
          setError(lang === "zh"
            ? "今日免费分析次数已用完。明天再来，或升级 Pro 继续练习。"
            : "You've used today's included analyses. Come back tomorrow—or continue with Pro.");
          setStage("upload");
          if (data.quota) setQuotaInfo(data.quota as LabQuotaResponse);
          return;
        }
        if (res.status === 503 || res.status === 524) {
          throw new Error(lang === "zh" ? "服务繁忙，请稍后重试。" : "Service busy, please try again shortly.");
        }
        if (res.status === 502) {
          throw new Error((data.detail as string) || (lang === "zh" ? "AI 服务暂时不可用，请稍后重试。" : "AI service temporarily unavailable."));
        }
        throw new Error((data.detail as string) || `HTTP ${res.status}`);
      }

      setProgress(100);
      const data: LabResponse = await res.json();
      setLabResponse(data);
      setQuotaInfo(data.quota);
      setStage("results");
      setResultTab("metrics");
      void loadHistory();
    } catch (err) {
      clearInterval(progressInterval);
      setProgress(0);
      if (err instanceof DOMException && err.name === "AbortError") {
        setError(lang === "zh" ? "请求超时，请稍后重试。" : "Request timed out, please try again.");
      } else {
        setError(err instanceof Error ? err.message : "分析失败，请重试");
      }
      setStage("upload");
    } finally {
      labAnalysisInFlightRef.current = false;
    }
  }, [fetchUnifiedPrediction, lang, loadHistory]);

  const handleUpload = useCallback((file: File) => {
    void submitLabAnalysisFile(file);
  }, [submitLabAnalysisFile]);

  useEffect(() => {
    if (!authChecked) return;
    const p = consumeReanalyzeFromHistoryPayload();
    if (!p || p.page !== "shot-lab") return;
    void (async () => {
      if (labAnalysisInFlightRef.current) {
        devWarn("[shot-lab] reanalyze skipped while lab analysis already in flight");
        return;
      }
      const blob = await fetchLabVideoBlobForReanalyze(p.analysisId);
      if (!blob || blob.size === 0) {
        setError(
          lang === "zh"
            ? "无法加载该 Shot Lab 的备份媒体（旧记录可能未存档）。请重新上传后再分析。"
            : "Could not load backed-up media for this Shot Lab job. Upload again, or use a newer completed session.",
        );
        return;
      }
      const fname = reanalyzeHistoryFilename(blob);
      const file = new File([blob], fname, { type: blob.type || "application/octet-stream" });
      await submitLabAnalysisFile(file);
    })();
    // 仅登录就绪时消费一次；勿依赖 lang/submitLabAnalysisFile，避免重复跑。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authChecked]);

  async function loadTrend() {
    const token = localStorage.getItem("stellar_token");
    if (!token || trendLoading) return;
    setTrendLoading(true);
    try {
      const res = await fetch("/api/lab/trend?days=90", { headers: { Authorization: `Bearer ${token}` } });
      if (res.ok) {
        const data = await res.json();
        setTrendData(data.points || []);
      }
    } catch { /* ignore */ }
    setTrendLoading(false);
  }

  async function handleCompare() {
    if (compareIds.length !== 2) return;
    const token = localStorage.getItem("stellar_token");
    if (!token) return;
    setCompareLoading(true);
    try {
      const res = await fetch(`/api/lab/compare?a=${encodeURIComponent(compareIds[0])}&b=${encodeURIComponent(compareIds[1])}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        setCompareResult(data);
        setResultTab("compare");
        setCompareSelectMode(false);
      } else {
        const data = await res.json().catch(() => ({}));
        setError((data as Record<string, string>).detail || (lang === "zh" ? "对比失败" : "Compare failed"));
      }
    } catch {
      setError(lang === "zh" ? "网络错误" : "Network error");
    }
    setCompareLoading(false);
  }

  async function handleExport() {
    if (!labResponse?.job_id) return;
    const token = localStorage.getItem("stellar_token");
    if (!token) return;
    setExportLoading(true);
    try {
      const res = await fetch("/api/lab/export", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: labResponse.job_id }),
      });
      if (res.ok) {
        const data = await res.json();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `shot-lab-${labResponse.job_id.slice(0, 8)}.json`;
        a.click();
        URL.revokeObjectURL(url);
      } else {
        const data = await res.json().catch(() => ({}));
        if ((data as Record<string, string>).error === "PRO_REQUIRED") {
          setError(lang === "zh" ? "导出为 Pro 专属功能" : "Export requires Pro");
        } else {
          setError((data as Record<string, string>).detail || (lang === "zh" ? "导出失败" : "Export failed"));
        }
      }
    } catch {
      setError(lang === "zh" ? "导出失败" : "Export failed");
    }
    setExportLoading(false);
  }

  function handleSignOut() {
    localStorage.removeItem("stellar_token");
    localStorage.removeItem("stellar_user");
    router.push("/login");
  }

  if (!authChecked) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-white/40 text-sm">加载中...</div>
      </div>
    );
  }

  if (liveCapture) {
    return (
      <ScreenModeCapture
        lang={lang}
        onCapture={(img) => {
          setLiveCapture(false);
          const byteChars = atob(img);
          const byteArray = new Uint8Array(byteChars.length);
          for (let i = 0; i < byteChars.length; i++) byteArray[i] = byteChars.charCodeAt(i);
          handleUpload(new File([byteArray], "swing-capture.jpg", { type: "image/jpeg" }));
        }}
        onVideoCapture={handleVideoCapture}
        onExit={() => setLiveCapture(false)}
      />
    );
  }

  const result = labResponse?.result;
  const tier = labResponse?.tier || (isPro ? "pro" : "free");
  const fv = result?.fields_visibility;

  return (
    <div className="min-h-screen pb-24">
      {/* Nav */}
      <nav className="sticky top-0 z-50 border-b border-white/5 bg-brand-dark/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
          <a href="/" className="flex items-center gap-2">
            <img src="/logo.svg" alt="Stellar" className="h-8 w-8" />
            <span className="text-xl font-bold text-brand-gold">STELLAR</span>
          </a>
          <div className="flex items-center gap-2">
            {username && (
              <a href="/history"
                className="flex items-center gap-1 rounded-lg border border-white/10 px-2.5 py-1 text-xs text-white/60 transition hover:border-brand-gold/30 hover:text-brand-gold">
                <span className="font-medium text-brand-gold">{username}</span>
              </a>
            )}
            <button onClick={() => { const next = lang === "en" ? "zh" : "en"; setLang(next); localStorage.setItem("stellar_lang", next); }}
              className="rounded-lg border border-white/10 px-3 py-1 text-xs text-white/60 transition hover:text-white">
              {lang === "en" ? "中文" : "EN"}
            </button>
            <button onClick={handleSignOut}
              className="rounded-lg border border-white/10 px-3 py-1 text-xs text-white/40 transition hover:text-white/70">
              {lang === "zh" ? "退出" : "Sign Out"}
            </button>
          </div>
        </div>
      </nav>

      <div className="mx-auto max-w-4xl px-4 py-6">
        {/* Header */}
        <div className="mb-6 text-center">
          <h1 className="text-3xl font-bold text-white tracking-wide">Shot Lab</h1>
          <p className="mt-1 text-sm text-white/50">击球实验室</p>
          <p className="mt-2 text-xs text-white/30">
            {isPro
              ? (lang === "zh" ? "完整数据、长期历史与深度训练计划，为认真练习者而设。" : "Full data, long-term history, and deeper training plans—built for serious practice.")
              : (lang === "zh" ? "用手机完成每一杆的专业级洞察。" : "Professional-grade insight from every swing—using only your phone.")}
          </p>
          {isPro && (
            <span className="mt-2 inline-block rounded-full bg-brand-gold/20 border border-brand-gold/30 px-3 py-0.5 text-[10px] font-bold text-brand-gold tracking-wider">
              PRO
            </span>
          )}
        </div>

        {/* Quota indicator */}
        {quotaInfo && !isPro && (
          <div className="mx-auto mb-4 flex max-w-xl items-center justify-center gap-2 rounded-xl border border-white/5 bg-white/[0.02] px-4 py-2">
            <span className="text-xs text-white/40">{lang === "zh" ? "今日剩余" : "Remaining today"}:</span>
            <span className={`text-sm font-bold ${quotaInfo.remaining > 0 ? "text-brand-gold" : "text-red-400"}`}>
              {quotaInfo.remaining}/{quotaInfo.limit}
            </span>
            <div className="flex gap-0.5">
              {Array.from({ length: quotaInfo.limit || 3 }).map((_, i) => (
                <div key={i} className={`h-2 w-5 rounded-full ${i < quotaInfo.used ? "bg-brand-gold/60" : "bg-white/10"}`} />
              ))}
            </div>
          </div>
        )}

        {/* ═══════ Upload Stage ═══════ */}
        {stage === "upload" && (
          <div className="animate-fade-in">
            {error && (
              <div className="mx-auto mb-6 max-w-xl rounded-xl bg-red-500/10 border border-red-500/20 p-4 text-sm text-red-400">
                <p className="break-words whitespace-pre-wrap">{error}</p>
                {error.includes("次数") || error.includes("analyses") ? (
                  <a href="/pro-login"
                    className="mt-3 inline-block rounded-lg bg-brand-gold/20 border border-brand-gold/30 px-4 py-1.5 text-xs font-semibold text-brand-gold transition hover:bg-brand-gold/30">
                    {lang === "zh" ? "升级 Pro" : "Upgrade to Pro"}
                  </a>
                ) : (
                  <button onClick={() => setError("")} className="mt-3 text-xs text-white/50 underline hover:text-white/70">
                    {lang === "zh" ? "关闭" : "Dismiss"}
                  </button>
                )}
              </div>
            )}

            {/* Input Mode Tabs */}
            <div className="mx-auto mb-6 flex max-w-xl rounded-xl border border-white/10 bg-white/[0.02] p-1 overflow-hidden">
              {(["upload", "capture", "screen"] as InputMode[]).map((mode) => (
                <button key={mode} onClick={() => setInputMode(mode)}
                  className={`flex-1 rounded-lg py-2.5 text-xs font-semibold transition-all ${
                    inputMode === mode
                      ? "bg-red-500/20 text-white border border-red-500/30"
                      : "text-white/40 hover:text-white/60 border border-transparent"
                  }`}>
                  {mode === "upload" ? (lang === "zh" ? "上传视频" : "Upload") :
                   mode === "capture" ? (lang === "zh" ? "实拍模式" : "Camera") :
                   (lang === "zh" ? "屏幕模式" : "Screen")}
                </button>
              ))}
            </div>

            {inputMode === "upload" && <UploadZone onUploadComplete={handleUpload} lang={lang} />}

            {inputMode === "capture" && (
              <div className="mx-auto max-w-xl">
                <div className="glass-card overflow-hidden">
                  <div className="relative bg-gradient-to-b from-red-500/10 to-transparent p-8 text-center">
                    <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-2xl bg-red-500/10 border border-red-500/20">
                      <svg className="h-10 w-10 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25h-9A2.25 2.25 0 0 0 2.25 7.5v9a2.25 2.25 0 0 0 2.25 2.25Z" />
                      </svg>
                    </div>
                    <h3 className="mb-2 text-lg font-semibold text-white">{lang === "zh" ? "实拍模式" : "Live Capture"}</h3>
                    <p className="mb-2 text-sm text-white/40">{lang === "zh" ? "摄像头录制挥杆，AI 骨架实时引导" : "Record swing with camera, AI skeleton overlay"}</p>
                    <p className="mb-6 text-xs text-white/25">{lang === "zh" ? "录制完成后自动进入 Shot Lab 分析" : "Automatically analyzed by Shot Lab after recording"}</p>
                    <button onClick={() => setLiveCapture(true)} className="mx-auto rounded-xl bg-red-500/20 border border-red-500/30 px-6 py-3 text-sm font-semibold text-red-300 transition hover:bg-red-500/30">
                      {lang === "zh" ? "打开摄像头" : "Open Camera"}
                    </button>
                  </div>
                </div>
              </div>
            )}

            {inputMode === "screen" && (
              <div className="mx-auto max-w-xl">
                <div className="glass-card overflow-hidden">
                  <div className="relative bg-gradient-to-b from-red-500/5 to-transparent p-8 text-center">
                    <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-2xl bg-red-500/10 border border-red-500/20">
                      <span className="text-4xl">📺</span>
                    </div>
                    <h3 className="mb-2 text-lg font-semibold text-white">{lang === "zh" ? "屏幕模式" : "Screen Mode"}</h3>
                    <p className="mb-6 text-sm text-white/40">
                      {lang === "zh" ? "录制电脑屏幕上的高尔夫视频，或用摄像头对着手机/电视拍摄" : "Record golf video from screen, or point camera at phone/TV"}
                    </p>
                    {screenRecording ? (
                      <div className="flex flex-col items-center gap-4">
                        <div className="flex items-center gap-2 text-red-400">
                          <span className="h-2 w-2 rounded-full bg-red-500 animate-pulse" />
                          <span className="font-mono text-sm">
                            {Math.floor(screenRecTime / 60).toString().padStart(2, "0")}:{(screenRecTime % 60).toString().padStart(2, "0")} / 00:30
                          </span>
                        </div>
                        <div className="h-1 w-40 rounded-full bg-white/10 overflow-hidden">
                          <div className="h-full rounded-full bg-red-500 transition-all duration-1000" style={{ width: `${(screenRecTime / 30) * 100}%` }} />
                        </div>
                        <button onClick={stopScreenRecording}
                          className="rounded-xl bg-red-500/20 border border-red-500/30 px-6 py-3 text-sm font-semibold text-red-300 transition hover:bg-red-500/30">
                          {lang === "zh" ? "⏹ 停止录制并分析" : "⏹ Stop & Analyze"}
                        </button>
                      </div>
                    ) : (
                      <div className="flex flex-col sm:flex-row gap-3 justify-center">
                        <button onClick={startScreenRecording}
                          className="rounded-xl bg-red-500/20 border border-red-500/30 px-6 py-3 text-sm font-semibold text-red-300 transition hover:bg-red-500/30">
                          {lang === "zh" ? "📺 录制屏幕" : "📺 Record Screen"}
                        </button>
                        <button onClick={() => setLiveCapture(true)}
                          className="rounded-xl border border-white/20 bg-white/5 px-6 py-3 text-sm text-white/70 transition hover:bg-white/10">
                          {lang === "zh" ? "📷 对屏拍摄" : "📷 Point Camera"}
                        </button>
                      </div>
                    )}
                    <p className="mt-4 text-[10px] text-white/20">
                      {lang === "zh" ? "提示：屏幕录制适合电脑端；对屏拍摄适合用手机对着电视/电脑" : "Screen recording for desktop; Camera for pointing at TV/monitor"}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Classic analyze link */}
            <div className="mx-auto mt-4 max-w-xl">
              <a href="/analyze" className="flex items-center justify-between rounded-xl border border-white/5 bg-white/[0.02] p-3 transition hover:border-red-500/20 group">
                <div className="flex items-center gap-3">
                  <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-red-500/10 text-sm">🏌️</span>
                  <div>
                    <span className="block text-sm font-medium text-white/60 group-hover:text-white/80 transition">{lang === "zh" ? "经典挥杆分析" : "Classic Swing Analysis"}</span>
                    <span className="block text-[10px] text-white/25">{lang === "zh" ? "五维评分 · AI诊断" : "5-Dimension Score · AI Diagnosis"}</span>
                  </div>
                </div>
                <svg className="h-4 w-4 text-white/20 group-hover:text-white/40 transition" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="m8.25 4.5 7.5 7.5-7.5 7.5" />
                </svg>
              </a>
            </div>

            {/* History Section */}
            {history.length > 0 && (
              <div className="mx-auto mt-8 max-w-xl">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-semibold text-white/60">{lang === "zh" ? "最近分析" : "Recent Analyses"}</h3>
                  {isPro && history.length >= 2 && (
                    <button
                      onClick={() => { setCompareSelectMode(!compareSelectMode); setCompareIds([]); }}
                      className="text-[10px] font-medium text-brand-gold/70 hover:text-brand-gold transition">
                      {compareSelectMode ? (lang === "zh" ? "取消对比" : "Cancel") : (lang === "zh" ? "挥杆对比" : "Compare")}
                    </button>
                  )}
                </div>

                {compareSelectMode && (
                  <div className="mb-3 rounded-xl border border-brand-gold/15 bg-brand-gold/5 p-3">
                    <p className="text-xs text-white/50 mb-2">
                      {lang === "zh" ? `请选择两条记录进行对比 (${compareIds.length}/2)` : `Select two records to compare (${compareIds.length}/2)`}
                    </p>
                    {compareIds.length === 2 && (
                      <button onClick={handleCompare} disabled={compareLoading}
                        className="rounded-lg bg-brand-gold/20 border border-brand-gold/30 px-4 py-1.5 text-xs font-semibold text-brand-gold transition hover:bg-brand-gold/30 disabled:opacity-50">
                        {compareLoading ? (lang === "zh" ? "对比中..." : "Comparing...") : (lang === "zh" ? "开始对比" : "Compare Now")}
                      </button>
                    )}
                  </div>
                )}

                <div className="space-y-2">
                  {history.slice(0, isPro ? 10 : 5).map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => {
                        if (compareSelectMode) {
                          setCompareIds(prev => {
                            if (prev.includes(item.id)) return prev.filter(id => id !== item.id);
                            if (prev.length >= 2) return prev;
                            return [...prev, item.id];
                          });
                        } else {
                          void openHistoryRecord(item);
                        }
                      }}
                      disabled={!compareSelectMode && historyOpenId !== null}
                      className={`glass-card flex w-full cursor-pointer items-center justify-between gap-2 rounded-xl border p-3 text-left transition hover:border-white/10 hover:bg-white/[0.03] disabled:cursor-wait disabled:opacity-60 ${
                        compareSelectMode && compareIds.includes(item.id) ? "border-brand-gold/40 bg-brand-gold/10" : "border-transparent"
                      }`}
                    >
                      <div className="flex min-w-0 flex-1 items-center gap-3">
                        {compareSelectMode ? (
                          <div className={`shrink-0 h-4 w-4 rounded border ${compareIds.includes(item.id) ? "bg-brand-gold/60 border-brand-gold" : "border-white/20"} flex items-center justify-center`}>
                            {compareIds.includes(item.id) && <span className="text-[8px] text-white">✓</span>}
                          </div>
                        ) : (
                          <div className={`shrink-0 h-2 w-2 rounded-full ${item.status === "completed" ? "bg-green-500" : item.status === "failed" ? "bg-red-500" : "bg-yellow-500"}`} />
                        )}
                        <div className="min-w-0">
                          <p className="truncate text-xs text-white/60">{item.summary_snippet || item.id.slice(0, 16) + "…"}</p>
                          <p className="text-[10px] text-white/30">
                            {new Date(item.created_at).toLocaleString(lang === "zh" ? "zh-CN" : "en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                          </p>
                        </div>
                      </div>
                      <div className="flex shrink-0 flex-col items-end gap-1">
                        <span className={`rounded-full px-2 py-0.5 text-[9px] font-bold ${item.tier === "pro" ? "bg-brand-gold/20 text-brand-gold" : "bg-white/5 text-white/30"}`}>
                          {item.tier === "pro" ? "PRO" : "FREE"}
                        </span>
                        {!compareSelectMode && (
                          <span className="text-[9px] font-medium text-brand-gold/80">
                            {historyOpenId === item.id ? (lang === "zh" ? "加载中…" : "Loading…") : (lang === "zh" ? "复盘" : "Review")}
                          </span>
                        )}
                      </div>
                    </button>
                  ))}
                </div>
                {historyLoading && (
                  <p className="mt-2 text-center text-xs text-white/20">{lang === "zh" ? "加载中..." : "Loading..."}</p>
                )}
                {!isPro && history.length >= 5 && (
                  <div className="mt-3 rounded-xl border border-brand-gold/10 bg-brand-gold/5 p-3 text-center">
                    <p className="text-xs text-white/40">
                      {lang === "zh" ? "更早的记录已归档至 Pro 历史库。" : "Older sessions are kept in your Pro history library."}
                    </p>
                    <a href="/pro-login" className="mt-1 inline-block text-xs font-semibold text-brand-gold hover:underline">
                      {lang === "zh" ? "解锁 Pro" : "Unlock Pro"}
                    </a>
                  </div>
                )}
                {!isPro && history.length >= 2 && (
                  <div className="mt-3 rounded-xl border border-brand-gold/10 bg-brand-gold/5 p-3 text-center">
                    <p className="text-xs text-white/40">
                      {lang === "zh" ? "并排对比与差异分析为 Pro 功能，用于追踪技术演变。" : "Side-by-side comparison is a Pro feature for tracking technique evolution."}
                    </p>
                    <a href="/pro-login" className="mt-1 inline-block text-xs font-semibold text-brand-gold hover:underline">
                      {lang === "zh" ? "解锁 Pro" : "Unlock Pro"}
                    </a>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ═══════ Processing Stage ═══════ */}
        {stage === "processing" && (
          <div className="mx-auto max-w-xl animate-fade-in">
            <AnalysisWaiting progress={progress} lang={lang} mode="lab" />
            <p className="mt-3 text-center text-[10px] text-white/15 italic">
              {lang === "zh" ? "所有指标均为视频估算值" : "All metrics are video-based estimates"}
            </p>
          </div>
        )}

        {/* ═══════ Results Stage ═══════ */}
        {stage === "results" && result && (
          <div className="space-y-5 animate-fade-in">
            {error && (
              <div className="mx-auto max-w-xl rounded-xl border border-amber-500/25 bg-amber-500/10 p-4 text-sm text-amber-200/90">
                <p className="break-words whitespace-pre-wrap">{error}</p>
                <button type="button" onClick={() => setError("")} className="mt-2 text-xs text-amber-300/80 underline hover:text-amber-200">
                  {lang === "zh" ? "关闭" : "Dismiss"}
                </button>
              </div>
            )}

            {/* Result Tabs */}
            <div className="flex gap-1 overflow-x-auto rounded-xl border border-white/5 bg-white/[0.02] p-1 no-scrollbar">
              {([
                { key: "metrics", label: "Metrics", labelZh: "指标" },
                { key: "trajectory", label: "Trajectory", labelZh: "弹道" },
                { key: "issues", label: "Issues", labelZh: "问题" },
                { key: "drills", label: "Drills", labelZh: "训练" },
                { key: "report", label: "Report", labelZh: "报告" },
                ...(isPro ? [
                  { key: "compare", label: "Compare", labelZh: "对比" },
                  { key: "trend", label: "Trend", labelZh: "趋势" },
                ] : []),
              ] as { key: ResultTab; label: string; labelZh: string }[]).map((tab) => (
                <button
                  key={tab.key}
                  onClick={() => {
                    setResultTab(tab.key);
                    if (tab.key === "trend" && trendData.length === 0) loadTrend();
                  }}
                  className={`whitespace-nowrap rounded-lg px-3 py-2 text-xs font-medium transition-all ${
                    resultTab === tab.key
                      ? "bg-brand-gold/15 text-brand-gold border border-brand-gold/25"
                      : "text-white/40 hover:text-white/60 border border-transparent"
                  }`}
                >
                  {lang === "zh" ? tab.labelZh : tab.label}
                </button>
              ))}
            </div>

            {/* What AI sees */}
            {result.what_i_see_zh && (
              <div className="glass-card p-4">
                <p className="text-xs text-white/40 mb-1">{lang === "zh" ? "AI 看到的内容" : "AI detected"}</p>
                <p className="text-sm text-white/70">{lang === "zh" ? result.what_i_see_zh : result.what_i_see}</p>
                {!result.is_golf_swing && (
                  <div className="mt-2 rounded-lg bg-yellow-500/10 border border-yellow-500/20 p-2 text-xs text-yellow-400">
                    {lang === "zh" ? "未检测到明确的高尔夫挥杆动作，分析结果可能不准确" : "No clear golf swing detected, results may be inaccurate"}
                  </div>
                )}
              </div>
            )}

            {/* ── Metrics Tab ── */}
            {resultTab === "metrics" && (
              <>
                <div className="glass-card p-5">
                  <div className="mb-4 flex items-center justify-between">
                    <h3 className="text-base font-semibold text-white">{lang === "zh" ? "击球指标" : "Shot Metrics"}</h3>
                    <span className="text-[9px] text-white/20 italic">{lang === "zh" ? "视频估算" : "video estimates"}</span>
                  </div>
                  {result.prediction && (
                    <div className="mb-3 rounded-xl border border-brand-gold/20 bg-brand-gold/5 px-3 py-2 text-[11px] text-white/60">
                      {(lang === "zh" ? "统一引擎" : "Unified engine")}:{" "}
                      {`${lang === "zh" ? "球杆" : "Club"} ${result.prediction.club_type || result.club_type || "UNKNOWN"} · ${
                        lang === "zh" ? "惯用手" : "Hand"
                      } ${result.prediction.hand || result.hand || "UNKNOWN"}`}
                    </div>
                  )}
                  <div className="grid grid-cols-3 gap-3">
                    <MetricCard label="Ball Speed" labelZh="球速" value={result.prediction?.ball_speed ?? result.metrics.ball_speed_mph} unit="mph" confidence={result.metrics.ball_speed_confidence} source={result.prediction ? "backend unified" : "estimated"} lang={lang} />
                    <MetricCard label="Launch Angle" labelZh="起飞角度" value={result.prediction?.launch_angle ?? result.metrics.launch_angle_deg} unit="°" confidence={result.metrics.launch_angle_confidence} source={result.prediction ? "backend unified" : "estimated"} lang={lang} />
                    <MetricCard label="Direction" labelZh="方向" value={result.metrics.launch_direction_deg} unit="°" confidence={result.metrics.launch_direction_confidence} source="estimated" lang={lang} />
                    <MetricCard label="Tempo" labelZh="节奏" value={result.metrics.tempo_ratio} unit="ratio" confidence={result.metrics.tempo_confidence} source="measured-like" lang={lang} />
                    <MetricCard label="Carry" labelZh="飞行距离" value={result.prediction?.predicted_distance ?? result.metrics.carry_distance_yards} unit={lang === "zh" ? "码" : "yards"} confidence={result.prediction?.distance_confidence ?? result.metrics.carry_distance_confidence} source={result.prediction ? "backend unified" : "estimated"} lang={lang} />
                    <MetricCard label="Contact" labelZh="触球质量" value={result.metrics.contact_quality_score} unit="/100" confidence={result.metrics.contact_quality_confidence} source="estimated" lang={lang} />
                  </div>
                  <div className="mt-3 grid grid-cols-2 gap-3">
                    <MetricCard label="Backswing" labelZh="后摆时间" value={result.metrics.backswing_time_sec} unit="sec" lang={lang} locked={fv?.backswing_time === "locked"} />
                    <MetricCard label="Downswing" labelZh="下杆时间" value={result.metrics.downswing_time_sec} unit="sec" lang={lang} locked={fv?.downswing_time === "locked"} />
                  </div>
                </div>

                <SwingTimeline metrics={result.metrics} lang={lang} locked={fv?.backswing_time === "locked"} />

                {/* Pro action bar */}
                <div className="flex flex-wrap gap-2 justify-center">
                  {isPro && (
                    <button onClick={handleExport} disabled={exportLoading}
                      className="rounded-lg border border-white/10 bg-white/5 px-4 py-2 text-xs text-white/60 transition hover:bg-white/10 disabled:opacity-40">
                      {exportLoading ? (lang === "zh" ? "导出中…" : "Exporting…") : (lang === "zh" ? "导出报告" : "Export Report")}
                    </button>
                  )}
                  {!isPro && (
                    <div className="text-center">
                      <a href="/pro-login" className="text-[10px] text-brand-gold/60 hover:text-brand-gold">
                        {lang === "zh" ? "升级 Pro：置信度分解 · 导出报告 · 趋势曲线" : "Upgrade: confidence breakdown, export, trends"}
                      </a>
                    </div>
                  )}
                </div>
              </>
            )}

            {/* ── Trajectory Tab ── */}
            {resultTab === "trajectory" && (
              <ShotTracerPlaceholder lang={lang} locked={fv?.trajectory_full === "locked"} />
            )}

            {/* ── Issues Tab ── */}
            {resultTab === "issues" && (
              <div className="glass-card p-5">
                <h3 className="mb-3 text-base font-semibold text-red-400/80">
                  {lang === "zh" ? "动作问题" : "Swing Issues"}
                  {result.issues_total != null && result.issues_total > result.issues.length && (
                    <span className="ml-2 text-[10px] font-normal text-white/30">
                      {lang === "zh" ? `显示 ${result.issues.length}/${result.issues_total}` : `${result.issues.length} of ${result.issues_total}`}
                    </span>
                  )}
                </h3>
                <div className="space-y-3">
                  {result.issues.map((issue: LabIssue, i: number) => (
                    <div key={issue.id || i} className="rounded-xl border border-white/5 bg-black/20 p-3">
                      <div className="flex items-start gap-2">
                        <span className={`mt-0.5 text-xs ${issue.severity === "high" ? "text-red-400" : issue.severity === "medium" ? "text-yellow-400" : "text-blue-400"}`}>●</span>
                        <div className="flex-1">
                          <p className="text-sm font-medium text-white/80">{lang === "zh" ? issue.title_zh : issue.title}</p>
                          <p className="mt-0.5 text-xs text-white/45">{lang === "zh" ? issue.description_zh : issue.description}</p>
                          {/* Inline drill per issue */}
                          {(issue.drill || issue.drill_zh) && (
                            <div className="mt-2 rounded-lg bg-brand-gold/5 border border-brand-gold/10 p-2">
                              <p className="text-[10px] text-brand-gold/60 font-medium mb-0.5">{lang === "zh" ? "训练建议" : "Drill"}</p>
                              <p className="text-xs text-white/50">{lang === "zh" ? issue.drill_zh : issue.drill}</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {result.issues_total != null && result.issues_total > result.issues.length && (
                  <div className="mt-3 text-center">
                    <a href="/pro-login" className="text-xs font-medium text-brand-gold hover:underline">
                      {lang === "zh"
                        ? `还有 ${result.issues_total - result.issues.length} 项问题 · 升级 Pro 查看`
                        : `+${result.issues_total - result.issues.length} more issues · Unlock with Pro`}
                    </a>
                  </div>
                )}
              </div>
            )}

            {/* ── Drills Tab ── */}
            {resultTab === "drills" && (
              <div className="glass-card p-5">
                <h3 className="mb-3 text-base font-semibold text-brand-gold/80">{lang === "zh" ? "训练建议" : "Drill Recommendations"}</h3>
                <div className="space-y-3">
                  {result.drills.map((drill: LabDrill, i: number) => (
                    <div key={i} className="rounded-xl border border-white/5 bg-black/20 p-3">
                      <p className="text-sm font-medium text-white/80">
                        <span className="mr-2 text-brand-gold/60">◆</span>
                        {lang === "zh" ? drill.title_zh : drill.title}
                      </p>
                      <p className="mt-1 pl-5 text-xs text-white/45">{lang === "zh" ? drill.description_zh : drill.description}</p>
                    </div>
                  ))}
                </div>
                {result.drills_total != null && result.drills_total > result.drills.length && (
                  <div className="mt-3 text-center">
                    <a href="/pro-login" className="text-xs font-medium text-brand-gold hover:underline">
                      {lang === "zh"
                        ? `还有 ${result.drills_total - result.drills.length} 项训练建议 · 升级 Pro`
                        : `+${result.drills_total - result.drills.length} more drills · Unlock with Pro`}
                    </a>
                  </div>
                )}
                {!isPro && (
                  <ProLockCard
                    lang={lang}
                    title="Full Drill Library"
                    titleZh="完整训练库"
                    description="Access the complete drill library with multi-session training sequences."
                    descriptionZh="解锁完整训练库，含多组训练序列与进阶计划。"
                  />
                )}
              </div>
            )}

            {/* ── Report Tab ── */}
            {resultTab === "report" && (
              <div className="glass-card p-5">
                <h3 className="mb-2 text-base font-semibold text-white">{lang === "zh" ? "AI 分析报告" : "AI Analysis Report"}</h3>

                {/* Summary - always visible */}
                <div className="mb-4">
                  <h4 className="text-xs font-medium text-white/40 mb-1">{lang === "zh" ? "分析总结" : "Summary"}</h4>
                  <p className="text-sm leading-relaxed text-white/60">{lang === "zh" ? result.summary_zh : result.summary}</p>
                </div>

                {/* Full report */}
                {tier === "pro" && (result.full_report_zh || result.full_report) && (
                  <div className="border-t border-white/5 pt-4">
                    <h4 className="mb-2 text-sm font-semibold text-brand-gold">{lang === "zh" ? "完整报告" : "Full Report"}</h4>
                    <p className="text-sm leading-relaxed text-white/50 whitespace-pre-line">
                      {lang === "zh" ? result.full_report_zh : result.full_report}
                    </p>
                  </div>
                )}

                {/* Free: preview then lock */}
                {tier !== "pro" && (
                  <div className="border-t border-white/5 pt-4 relative">
                    <h4 className="mb-2 text-sm font-semibold text-white/40">{lang === "zh" ? "完整报告预览" : "Full Report Preview"}</h4>
                    {(result.full_report_zh || result.full_report) ? (
                      <div className="relative">
                        <p className="text-sm leading-relaxed text-white/40 whitespace-pre-line">
                          {lang === "zh" ? result.full_report_zh : result.full_report}
                        </p>
                        {result.full_report_preview && (
                          <div className="absolute bottom-0 left-0 right-0 h-24 bg-gradient-to-t from-brand-dark via-brand-dark/80 to-transparent flex items-end justify-center pb-3">
                            <a href="/pro-login" className="rounded-lg bg-brand-gold/15 border border-brand-gold/25 px-5 py-2 text-xs font-semibold text-brand-gold transition hover:bg-brand-gold/25">
                              {lang === "zh" ? "解锁完整报告" : "Unlock Full Report"}
                            </a>
                          </div>
                        )}
                      </div>
                    ) : (
                      <ProLockCard
                        lang={lang}
                        title="Full AI Report"
                        titleZh="完整 AI 报告"
                        description="Detailed structured report covering setup through follow-through."
                        descriptionZh="此模块包含完整动作诊断与训练序列。升级 Pro 查看全部。"
                      />
                    )}
                  </div>
                )}
              </div>
            )}

            {/* ── Compare Tab (Pro) ── */}
            {resultTab === "compare" && (
              <div>
                {!isPro ? (
                  <ProLockCard
                    lang={lang}
                    title="Swing Comparison"
                    titleZh="挥杆对比"
                    description="Side-by-side comparison and difference analysis for tracking technique evolution."
                    descriptionZh="并排对比与差异分析为 Pro 功能，用于追踪技术演变。"
                  />
                ) : compareResult ? (
                  <div className="glass-card p-5">
                    <h3 className="mb-4 text-base font-semibold text-white">{lang === "zh" ? "挥杆对比" : "Swing Comparison"}</h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="border-b border-white/10">
                            <th className="py-2 pr-3 text-left text-white/40">{lang === "zh" ? "指标" : "Metric"}</th>
                            <th className="py-2 px-3 text-right text-white/40">A</th>
                            <th className="py-2 px-3 text-right text-white/40">B</th>
                            <th className="py-2 pl-3 text-right text-white/40">{lang === "zh" ? "差异" : "Delta"}</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(compareResult.diff).map(([key, v]) => (
                            <tr key={key} className="border-b border-white/5">
                              <td className="py-2 pr-3 text-white/60">{key.replace(/_/g, " ")}</td>
                              <td className="py-2 px-3 text-right text-white/50">{v.a != null ? v.a.toFixed(1) : "—"}</td>
                              <td className="py-2 px-3 text-right text-white/50">{v.b != null ? v.b.toFixed(1) : "—"}</td>
                              <td className={`py-2 pl-3 text-right font-medium ${v.delta != null ? (v.delta > 0 ? "text-green-400" : v.delta < 0 ? "text-red-400" : "text-white/30") : "text-white/20"}`}>
                                {v.delta != null ? (v.delta > 0 ? "+" : "") + v.delta.toFixed(1) : "—"}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <button onClick={() => { setCompareResult(null); setCompareSelectMode(true); setCompareIds([]); setResultTab("metrics"); }}
                      className="mt-4 text-xs text-white/40 hover:text-white/60 transition">
                      {lang === "zh" ? "重新选择" : "New comparison"}
                    </button>
                  </div>
                ) : (
                  <div className="glass-card p-5 text-center">
                    <p className="text-sm text-white/40 mb-3">{lang === "zh" ? "请从历史记录中选择两条进行对比" : "Select two records from history to compare"}</p>
                    <button onClick={() => { setCompareSelectMode(true); setStage("upload"); }}
                      className="rounded-lg bg-brand-gold/15 border border-brand-gold/25 px-4 py-2 text-xs font-semibold text-brand-gold transition hover:bg-brand-gold/25">
                      {lang === "zh" ? "选择记录" : "Select Records"}
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* ── Trend Tab (Pro) ── */}
            {resultTab === "trend" && (
              <div>
                {!isPro ? (
                  <ProLockCard
                    lang={lang}
                    title="Trend Analytics"
                    titleZh="趋势分析"
                    description="Track your progress over time with visual trend charts."
                    descriptionZh="通过可视化趋势图追踪你的长期进步。"
                  />
                ) : trendLoading ? (
                  <div className="glass-card p-5 text-center">
                    <p className="text-sm text-white/40">{lang === "zh" ? "加载趋势数据…" : "Loading trend data…"}</p>
                  </div>
                ) : (
                  <TrendChart points={trendData} lang={lang} />
                )}
              </div>
            )}

            {/* ── Upgrade Pro Module (inline, not popup) ── */}
            {!isPro && (
              <div className="glass-card p-5 bg-gradient-to-br from-brand-gold/[0.04] to-transparent border-brand-gold/10">
                <div className="flex items-center gap-3 mb-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-gold/10 border border-brand-gold/20">
                    <span className="text-brand-gold text-lg font-bold">P</span>
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-white">{lang === "zh" ? "解锁 Shot Lab Pro" : "Unlock Shot Lab Pro"}</h3>
                    <p className="text-[10px] text-white/40">{lang === "zh" ? "完整报告与长期进步曲线" : "Full reports and long-term progress curves"}</p>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2 mb-3">
                  {[
                    { zh: "完整动作诊断", en: "Full diagnosis" },
                    { zh: "长期历史保存", en: "Long-term history" },
                    { zh: "挥杆对比分析", en: "Swing comparison" },
                    { zh: "趋势进步曲线", en: "Progress trends" },
                    { zh: "完整训练库", en: "Full drill library" },
                    { zh: "导出与分享", en: "Export & share" },
                  ].map((item, i) => (
                    <div key={i} className="flex items-center gap-1.5">
                      <span className="text-brand-gold/60 text-[10px]">✓</span>
                      <span className="text-[10px] text-white/50">{lang === "zh" ? item.zh : item.en}</span>
                    </div>
                  ))}
                </div>
                <a href="/pro-login" className="block w-full rounded-lg bg-brand-gold/20 border border-brand-gold/30 py-2.5 text-center text-sm font-semibold text-brand-gold transition hover:bg-brand-gold/30">
                  {lang === "zh" ? "升级 Pro" : "Upgrade to Pro"}
                </a>
              </div>
            )}

            {/* History in results */}
            {history.length > 0 && (
              <div className="mx-auto max-w-xl">
                <h3 className="mb-2 text-sm font-semibold text-white/50">
                  {lang === "zh" ? "最近分析 · 点选复盘" : "Recent analyses · tap to review"}
                </h3>
                <div className="space-y-2">
                  {history.slice(0, 5).map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => void openHistoryRecord(item)}
                      disabled={historyOpenId !== null}
                      className={`glass-card flex w-full cursor-pointer items-center justify-between gap-2 rounded-xl border p-3 text-left transition hover:border-white/10 hover:bg-white/[0.03] disabled:cursor-wait disabled:opacity-60 ${
                        labResponse?.job_id === item.id ? "border-brand-gold/30 bg-brand-gold/5" : "border-transparent"
                      }`}
                    >
                      <div className="flex min-w-0 flex-1 items-center gap-3">
                        <div className={`shrink-0 h-2 w-2 rounded-full ${item.status === "completed" ? "bg-green-500" : item.status === "failed" ? "bg-red-500" : "bg-yellow-500"}`} />
                        <div className="min-w-0">
                          <p className="truncate text-xs text-white/60">{item.summary_snippet || item.id.slice(0, 16) + "…"}</p>
                          <p className="text-[10px] text-white/30">
                            {new Date(item.created_at).toLocaleString(lang === "zh" ? "zh-CN" : "en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                          </p>
                        </div>
                      </div>
                      <span className="shrink-0 text-[9px] font-medium text-brand-gold/80">
                        {historyOpenId === item.id
                          ? (lang === "zh" ? "加载中…" : "Loading…")
                          : labResponse?.job_id === item.id
                            ? (lang === "zh" ? "当前" : "Current")
                            : (lang === "zh" ? "复盘" : "Open")}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Analyze Again */}
            <div className="text-center pt-2 pb-4">
              <button
                onClick={() => {
                  setStage("upload");
                  setLabResponse(null);
                  setError("");
                  setCompareResult(null);
                  setCompareIds([]);
                  setCompareSelectMode(false);
                  loadHistory();
                }}
                className="btn-primary"
              >
                {lang === "zh" ? "再次分析" : "Analyze Again"}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Bottom bar */}
      {stage === "upload" && (
        <div className="fixed bottom-0 left-0 right-0 z-40 border-t border-white/10 bg-brand-dark/85 backdrop-blur-xl">
          <div className="mx-auto flex max-w-7xl items-center justify-center gap-3 px-4 py-3">
            <a href="/analyze" className="rounded-lg border border-brand-purple/30 bg-brand-purple/15 px-4 py-2 text-sm font-semibold text-white transition hover:bg-brand-purple/25">
              {lang === "zh" ? "经典分析" : "Classic Analysis"}
            </a>
            {!isPro && (
              <a href="/pro-login" className="btn-pro rounded-lg px-4 py-2 text-sm font-semibold">
                {lang === "zh" ? "升级 Pro" : "Upgrade to Pro"}
              </a>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
