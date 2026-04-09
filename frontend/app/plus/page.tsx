"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import UploadZone from "@/components/UploadZone";
import AnalysisWaiting from "@/components/AnalysisWaiting";
import ScreenModeCapture from "@/components/ScreenModeCapture";
import PlusResultView, { type PlusAnalysisResult } from "@/components/PlusResultView";
import { devWarn } from "@/lib/dev-only-log";
import { preloadPoseModel } from "@/lib/mediapipe-assets";
import { saveAnalysisVideo } from "@/lib/video-store";
import { makeFormData } from "@/lib/fetch-retry";
import {
  slimAnalysisResultForHistoryTransport,
  slimAnalysisResultForServerHistory,
} from "@/lib/strip-result";
import { normalizedTotalScoreForStorage } from "@/lib/safe-analysis-score";
import { patchLocalHistoryVideoR2Key } from "@/lib/history-sync-record";
import { pruneLocalStellarHistoryRecords } from "@/lib/pro-history-retention";
import {
  consumeReanalyzeFromHistoryPayload,
  fetchVideoBlobForHistoryReanalyze,
  reanalyzeHistoryFilename,
  reanalyzePayloadProv3ScreenMode,
} from "@/lib/reanalyze-from-history";

interface ClubDetection { club_type: string; club_group: string; confidence: number }

type Stage = "upload" | "processing" | "results";
type InputMode = "upload" | "capture" | "screen";

function isVideoBlobForOverlay(blob: Blob, filename: string): boolean {
  if (blob.type.startsWith("video/")) return true;
  return /\.(mp4|mov|webm|m4v|avi)$/i.test(filename);
}

interface PlusUsageInfo {
  used: number;
  remaining: number;
  limit: number | null;
  is_pro: boolean;
}

export default function PlusPage() {
  const router = useRouter();
  const [stage, setStage] = useState<Stage>("upload");
  const [result, setResult] = useState<PlusAnalysisResult | null>(null);
  /** In-memory object URL for the clip just analyzed — avoids "video lost" if IndexedDB save lags or fails. */
  const [sessionVideoSrc, setSessionVideoSrc] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState(0);
  const [lang, setLang] = useState<"en" | "zh">("zh");
  const [inputMode, setInputMode] = useState<InputMode>("upload");
  const [liveCapture, setLiveCapture] = useState(false);
  const [username, setUsername] = useState("");
  const [authChecked, setAuthChecked] = useState(false);
  const [usage, setUsage] = useState<PlusUsageInfo | null>(null);
  const [usageLoading, setUsageLoading] = useState(true);

  const [screenRecording, setScreenRecording] = useState(false);
  const screenRecRef = useRef<MediaRecorder | null>(null);
  const screenStreamRef = useRef<MediaStream | null>(null);
  const screenChunksRef = useRef<Blob[]>([]);
  const [screenRecTime, setScreenRecTime] = useState(0);
  const screenTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [processingClub, setProcessingClub] = useState<ClubDetection | null>(null);
  const processingClubRef = useRef<ClubDetection | null>(null);
  const [showClubPicker, setShowClubPicker] = useState(false);

  const CLUB_GROUPS = [
    { id: "WOOD", label_zh: "木杆", label_en: "Wood", clubs: ["1W", "3W", "5W"] },
    { id: "IRON", label_zh: "铁杆", label_en: "Iron", clubs: ["3I", "4I", "5I", "6I", "7I", "8I", "9I"] },
    { id: "WEDGE", label_zh: "挖起杆", label_en: "Wedge", clubs: ["PW", "AW", "SW", "LW"] },
    { id: "PUTTER", label_zh: "推杆", label_en: "Putter", clubs: ["PT"] },
  ] as const;

  const CLUB_DISPLAY: Record<string, string> = {
    "1W": "1号木", "3W": "3号木", "5W": "5号木",
    "3I": "3号铁", "4I": "4号铁", "5I": "5号铁", "6I": "6号铁",
    "7I": "7号铁", "8I": "8号铁", "9I": "9号铁",
    "PW": "劈起杆", "AW": "间隙杆", "SW": "沙坑杆", "LW": "高抛杆",
    "PT": "推杆",
  };

  useEffect(() => {
    const token = localStorage.getItem("stellar_token");
    if (!token) { router.push("/login"); return; }
    const userStr = localStorage.getItem("stellar_user");
    if (userStr) {
      try { const u = JSON.parse(userStr); setUsername(u.username || u.email || ""); } catch { /* */ }
    }
    setAuthChecked(true);
    fetch("/api/plus/usage", { headers: { Authorization: `Bearer ${token}` } })
      .then(r => r.json()).then(data => setUsage(data))
      .catch(() => setUsage({ used: 0, remaining: 3, limit: 3, is_pro: false }))
      .finally(() => setUsageLoading(false));

    preloadPoseModel();
  }, [router]);

  useEffect(() => {
    if (!authChecked) return;
    const p = consumeReanalyzeFromHistoryPayload();
    if (!p || p.page !== "plus") return;
    void (async () => {
      try {
        const blob = await fetchVideoBlobForHistoryReanalyze(
          p.analysisId,
          p.videoUrl,
          p.analysisVideoUrl,
        );
        if (!blob || blob.size === 0) {
          setError(
            lang === "zh"
              ? "无法加载该记录原视频。请确认本机已缓存或已登录且云端仍保存视频。"
              : "Could not load the original video for this record.",
          );
          return;
        }
        if (reanalyzePayloadProv3ScreenMode(p)) setInputMode("screen");
        await processBlob(blob, reanalyzeHistoryFilename(blob));
      } catch (e) {
        devWarn("[plus] reanalyze pipeline error:", e);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authChecked, lang]);

  // Avoid [sessionVideoSrc] effect cleanup + revokeObjectURL: Strict Mode can revoke the URL
  // while the video tab still needs it. Revoke only in setSessionVideoSrc updaters / reset flows.

  function minimalPlusHistoryPayload(data: PlusAnalysisResult): Record<string, unknown> {
    return {
      analysis_id: data.analysis_id,
      type: "plus",
      total_score: data.total_score,
      posture_score: data.posture_score,
      primary_diagnosis: data.primary_diagnosis,
      quick_tip_zh: data.quick_tip_zh,
      quick_tip_en: data.quick_tip_en,
    };
  }

  function saveToLocalHistory(data: PlusAnalysisResult) {
    const key = "stellar_history_local";
    const id = (data.analysis_id || "").trim() || `local-${Date.now()}`;
    const writeEntry = (resultPayload: unknown) => {
      const existing = JSON.parse(localStorage.getItem(key) || "[]");
      const entry = {
        id,
        type: "plus",
        total_score: normalizedTotalScoreForStorage(data.total_score),
        result_json: JSON.stringify(resultPayload),
        created_at: new Date().toISOString(),
        _local: true,
      };
      const filtered = existing.filter((r: { id: string }) => r.id !== entry.id);
      filtered.unshift(entry);
      localStorage.setItem(key, JSON.stringify(filtered.slice(0, 200)));
    };
    try {
      writeEntry(slimAnalysisResultForHistoryTransport(data));
      pruneLocalStellarHistoryRecords();
    } catch (e1) {
      try {
        writeEntry(minimalPlusHistoryPayload(data));
        pruneLocalStellarHistoryRecords();
      } catch (e2) {
        devWarn("[plus] local history save failed (quota or storage):", e2, e1);
      }
    }
  }

  function markLocalRecordSynced(id: string) {
    try {
      const key = "stellar_history_local";
      const existing = JSON.parse(localStorage.getItem(key) || "[]");
      const updated = existing.map((r: Record<string, unknown>) =>
        r.id === id ? { ...r, _synced: true } : r
      );
      localStorage.setItem(key, JSON.stringify(updated));
    } catch { /* ignore */ }
  }

  async function saveAnalysisToHistory(data: PlusAnalysisResult, blob?: Blob, filename?: string) {
    saveToLocalHistory(data);
    const token = localStorage.getItem("stellar_token");
    if (!token || token.startsWith("local-")) return;
    try {
      let videoR2Key = "";
      if (blob && blob.size > 0) {
        const form = new FormData();
        form.append("analysis_id", data.analysis_id);
        form.append("file", blob, filename || `${data.analysis_id}.mp4`);
        const uploadRes = await fetch("/api/history/upload-video", {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` },
          body: form,
        });
        if (uploadRes.ok) {
          const uploadData = await uploadRes.json().catch(() => ({}));
          videoR2Key = typeof uploadData.video_r2_key === "string" ? uploadData.video_r2_key : "";
          if (videoR2Key) patchLocalHistoryVideoR2Key(data.analysis_id, videoR2Key);
        }
      }

      const res = await fetch("/api/history", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({
          analysis_id: data.analysis_id,
          type: "plus",
          total_score: normalizedTotalScoreForStorage(data.total_score),
          result: slimAnalysisResultForServerHistory(data),
          video_r2_key: videoR2Key,
        }),
      });
      if (res.ok) {
        const saved = await res.json().catch(() => ({}));
        if (saved.success !== false) {
          markLocalRecordSynced(data.analysis_id);
        } else {
          devWarn("[history] plus save rejected:", saved.detail || saved);
        }
      } else {
        const errText = await res.text().catch(() => "");
        devWarn("[history] plus save failed:", res.status, errText.slice(0, 400));
      }
    } catch (e) {
      devWarn("[history] plus save error:", e);
    }
  }

  function extractFrameFromBlob(blob: Blob): Promise<Blob | null> {
    return new Promise((resolve) => {
      const isVideo = blob.type.startsWith("video/") ||
        /\.(mp4|mov|webm|avi|m4v)$/i.test((blob as File).name || "");
      if (!isVideo && blob.type.startsWith("image/")) { resolve(blob); return; }
      const video = document.createElement("video");
      video.muted = true; video.playsInline = true; video.preload = "auto";
      const url = URL.createObjectURL(blob);
      video.src = url;
      let resolved = false;
      const cleanup = () => { try { URL.revokeObjectURL(url); } catch { /* */ } };
      const done = (b: Blob | null) => { if (resolved) return; resolved = true; cleanup(); resolve(b); };
      const timer = setTimeout(() => done(null), 10000);
      const captureFrame = () => {
        clearTimeout(timer);
        try {
          const w = video.videoWidth || 640, h = video.videoHeight || 480;
          const canvas = document.createElement("canvas");
          const scale = Math.min(640 / w, 1);
          canvas.width = Math.round(w * scale); canvas.height = Math.round(h * scale);
          const ctx = canvas.getContext("2d");
          if (!ctx) { done(null); return; }
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          canvas.toBlob((b) => done(b), "image/jpeg", 0.8);
        } catch { done(null); }
      };
      video.onseeked = captureFrame;
      video.onloadeddata = () => {
        if (video.duration && isFinite(video.duration) && video.duration > 0.5) video.currentTime = video.duration * 0.4;
        else captureFrame();
      };
      video.onerror = () => { clearTimeout(timer); done(null); };
      video.load();
    });
  }

  async function detectClubFromBlob(blob: Blob) {
    try {
      const frameBlob = await extractFrameFromBlob(blob);
      if (!frameBlob) return;
      const fd = new FormData();
      fd.append("frame", frameBlob, "frame.jpg");
      const token = localStorage.getItem("stellar_token");
      const headers: Record<string, string> = {};
      if (token) headers["Authorization"] = `Bearer ${token}`;
      const res = await fetch("/api/club-detect", { method: "POST", headers, body: fd });
      if (!res.ok) return;
      const club = await res.json();
      const data: ClubDetection = {
        club_type: club.club_type || "UNKNOWN",
        club_group: club.club_group || "IRON",
        confidence: club.confidence || 0,
      };
      setProcessingClub(data);
      processingClubRef.current = data;
    } catch { /* non-fatal */ }
  }

  function handleProcessingClubChange(clubType: string) {
    setShowClubPicker(false);
    const groupMap: Record<string, string> = {};
    for (const g of CLUB_GROUPS) for (const c of g.clubs) groupMap[c] = g.id;
    const newClub: ClubDetection = { club_type: clubType, club_group: groupMap[clubType] || "IRON", confidence: 1.0 };
    setProcessingClub(newClub);
    processingClubRef.current = newClub;
  }

  async function processBlob(blob: Blob, filename: string) {
    setSessionVideoSrc((prev) => {
      if (prev) try { URL.revokeObjectURL(prev); } catch { /* */ }
      return null;
    });
    setStage("processing"); setError(""); setProgress(0);
    setProcessingClub(null); processingClubRef.current = null;
    const t0 = Date.now();
    const progressInterval = setInterval(() => {
      const elapsed = (Date.now() - t0) / 1000;
      setProgress(prev => {
        if (prev >= 99) return prev;
        const target = elapsed < 10
          ? Math.min(55, elapsed * 5.5)
          : elapsed < 60
          ? 55 + (elapsed - 10) * 0.6
          : 85 + Math.min(13.5, (elapsed - 60) * 0.05);
        const next = prev + (target - prev) * 0.12 + Math.random() * 1.5;
        return Math.min(next, 98.5);
      });
    }, 800);

    detectClubFromBlob(blob);
    const clubFallbackTimer = setTimeout(() => {
      if (!processingClubRef.current) {
        const fb: ClubDetection = { club_type: "UNKNOWN", club_group: "IRON", confidence: 0 };
        setProcessingClub(fb); processingClubRef.current = fb;
      }
    }, 6000);

    try {
      const token = localStorage.getItem("stellar_token");
      const authHeaders: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};

      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 360_000);
      const res = await fetch("/api/plus", {
        method: "POST",
        headers: authHeaders,
        body: makeFormData(blob, filename),
        signal: controller.signal,
      });
      clearTimeout(timer);

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        if (errData.limit_reached) {
          throw new Error(lang === "zh"
            ? `今日 Plus 分析次数已达上限（${errData.limit}次/天）。升级 Pro 解锁无限次使用。`
            : `Daily Plus limit reached (${errData.limit}/day). Upgrade to Pro.`);
        }
        throw new Error(errData.detail || `Plus 分析失败 [${res.status}]`);
      }

      setProgress(97);
      const data: PlusAnalysisResult = await res.json();
      clearInterval(progressInterval); clearTimeout(clubFallbackTimer);
      setProgress(100);
      if (data._plus_usage) setUsage(data._plus_usage);
      setResult(data);
      if (isVideoBlobForOverlay(blob, filename) && blob.size > 0) {
        setSessionVideoSrc(URL.createObjectURL(blob));
      }
      setStage("results");
      void saveAnalysisVideo(data.analysis_id, blob, filename).catch(() => {});
      try {
        await saveAnalysisToHistory(data, blob, filename);
      } catch (e) {
        devWarn("[plus] history save failed:", e);
      }
    } catch (err: unknown) {
      clearInterval(progressInterval); clearTimeout(clubFallbackTimer); setProgress(0);
      if (err instanceof DOMException && err.name === "AbortError") {
        setError(lang === "zh" ? "Plus 分析超时，请压缩视频后重试" : "Plus analysis timed out");
      } else {
        const msg = err instanceof Error ? err.message : "Plus 分析失败";
        setError(msg);
      }
      setStage("upload");
    }
  }

  const handleUploadComplete = useCallback((file: File) => { processBlob(file, file.name); }, [lang]); // eslint-disable-line react-hooks/exhaustive-deps
  const handleVideoCapture = useCallback((videoBlob: Blob) => { setLiveCapture(false); processBlob(videoBlob, "swing-capture.webm"); }, [lang]); // eslint-disable-line react-hooks/exhaustive-deps
  const handleScreenCapture = useCallback((imageBase64: string) => {
    setLiveCapture(false);
    const byteChars = atob(imageBase64);
    const byteArray = new Uint8Array(byteChars.length);
    for (let i = 0; i < byteChars.length; i++) byteArray[i] = byteChars.charCodeAt(i);
    processBlob(new Blob([byteArray], { type: "image/jpeg" }), "swing-capture.jpg");
  }, [lang]); // eslint-disable-line react-hooks/exhaustive-deps

  async function startScreenRecording() {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      screenStreamRef.current = stream; screenChunksRef.current = [];
      const mime = MediaRecorder.isTypeSupported("video/webm") ? "video/webm" : "video/mp4";
      const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 3000000 });
      rec.ondataavailable = e => { if (e.data.size > 0) screenChunksRef.current.push(e.data); };
      rec.onstop = () => {
        stream.getTracks().forEach(t => t.stop());
        if (screenTimerRef.current) clearInterval(screenTimerRef.current);
        setScreenRecording(false);
        const blob = new Blob(screenChunksRef.current, { type: mime });
        if (blob.size > 0) processBlob(blob, "screen-capture.webm");
      };
      rec.start(200); screenRecRef.current = rec; setScreenRecording(true); setScreenRecTime(0);
      screenTimerRef.current = setInterval(() => { setScreenRecTime(prev => { if (prev >= 30) { stopScreenRecording(); return prev; } return prev + 1; }); }, 1000);
      stream.getVideoTracks()[0].onended = () => { if (rec.state !== "inactive") rec.stop(); };
    } catch { setLiveCapture(true); }
  }

  function stopScreenRecording() {
    const rec = screenRecRef.current;
    if (rec && rec.state === "recording") {
      try { rec.requestData(); } catch { /* ignore if no data pending */ }
    }
    if (rec?.state !== "inactive") rec?.stop();
    if (screenTimerRef.current) { clearInterval(screenTimerRef.current); screenTimerRef.current = null; }
    setScreenRecording(false);
  }

  function handleSignOut() { localStorage.removeItem("stellar_token"); localStorage.removeItem("stellar_user"); router.push("/login"); }

  if (!authChecked) return <div className="flex min-h-screen items-center justify-center"><div className="text-white/40 text-sm">加载中...</div></div>;

  if (screenRecording) {
    return (
      <div className="fixed inset-0 z-[9999] flex flex-col items-center justify-center bg-black/90">
        <div className="mb-6 h-4 w-4 rounded-full bg-red-500 animate-pulse" />
        <h2 className="mb-2 text-xl font-bold text-white">{lang === "zh" ? "正在录制屏幕..." : "Recording Screen..."}</h2>
        <p className="mb-6 text-sm text-white/40">{lang === "zh" ? "播放高尔夫挥杆视频，录制完成后自动分析" : "Play the golf swing video, analysis starts after recording"}</p>
        <div className="mb-4 text-3xl font-bold text-brand-gold">
          {Math.floor(screenRecTime / 60).toString().padStart(2, "0")}:{(screenRecTime % 60).toString().padStart(2, "0")}
          <span className="text-sm text-white/30 ml-2">/ 00:30</span>
        </div>
        <div className="mb-8 w-64 h-1 rounded-full bg-white/10 overflow-hidden">
          <div className="h-full rounded-full bg-red-500 transition-all duration-1000" style={{ width: `${(screenRecTime / 30) * 100}%` }} />
        </div>
        <button onClick={stopScreenRecording} className="btn-primary px-8 py-3 text-sm">{lang === "zh" ? "停止录制并分析" : "Stop & Analyze"}</button>
      </div>
    );
  }

  if (liveCapture) {
    return <ScreenModeCapture onCapture={handleScreenCapture} onVideoCapture={handleVideoCapture} onExit={() => setLiveCapture(false)} lang={lang} />;
  }

  const isLimitReached = usage && !usage.is_pro && usage.limit !== null && usage.remaining <= 0;

  return (
    <div className="min-h-screen">
      <nav className="sticky top-0 z-50 border-b border-white/5 bg-brand-dark/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
          <a href="/" className="flex items-center gap-2">
            <img src="/logo.svg" alt="Stellar" className="h-8 w-8" />
            <span className="text-xl font-bold text-brand-gold">STELLAR</span>
          </a>
          <div className="flex items-center gap-2">
            <span className="rounded-lg bg-gradient-to-r from-brand-purple to-brand-gold px-2.5 py-0.5 text-[10px] font-bold text-white">PLUS</span>
            {username && (
              <a href="/history" className="flex items-center gap-1 rounded-lg border border-white/10 px-2.5 py-1 text-xs text-white/60 transition hover:border-brand-gold/30 hover:text-brand-gold">
                <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A17.933 17.933 0 0 1 12 21.75c-2.676 0-5.216-.584-7.499-1.632Z" /></svg>
                <span className="font-medium text-brand-gold">{username}</span>
              </a>
            )}
            <button onClick={() => setLang(lang === "en" ? "zh" : "en")} className="rounded-lg border border-white/10 px-3 py-1 text-xs text-white/60 transition hover:text-white">{lang === "en" ? "中文" : "EN"}</button>
            <button onClick={handleSignOut} className="rounded-lg border border-white/10 px-3 py-1 text-xs text-white/40 transition hover:text-white/70">退出</button>
          </div>
        </div>
      </nav>

      <div className="mx-auto max-w-2xl px-4 py-6">
        {stage === "upload" && (
          <div className="animate-fade-in">
            <div className="text-center mb-6">
              <div className="inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-brand-purple/20 to-brand-gold/20 border border-brand-purple/30 px-4 py-1.5 mb-3">
                <span className="h-2 w-2 rounded-full bg-brand-gold animate-pulse" />
                <span className="text-xs font-semibold text-white/70">{lang === "zh" ? "高级姿势诊断" : "Advanced Posture Diagnosis"}</span>
              </div>
              <h1 className="text-3xl font-bold text-white mb-2">{lang === "zh" ? "Plus 分析" : "Plus Analysis"}</h1>
              <p className="text-sm text-white/40">{lang === "zh" ? "深度挥杆诊断 · 8阶段评估 · Pro对比 · 个性化训练" : "Deep diagnosis · 8-phase eval · Pro compare · Training plan"}</p>
            </div>

            {!usageLoading && usage && !usage.is_pro && usage.limit !== null && (
              <div className="mx-auto mb-4 max-w-sm">
                <div className="flex items-center justify-between text-xs text-white/40 mb-1">
                  <span>{lang === "zh" ? "今日剩余" : "Today remaining"}</span>
                  <span>{usage.remaining}/{usage.limit}</span>
                </div>
                <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
                  <div className="h-full rounded-full bg-gradient-to-r from-brand-purple to-brand-gold transition-all" style={{ width: `${((usage.limit - usage.remaining) / usage.limit) * 100}%` }} />
                </div>
              </div>
            )}
            {usage?.is_pro && (
              <div className="mx-auto mb-4 max-w-sm text-center">
                <span className="inline-flex items-center gap-1 text-xs text-brand-gold">
                  <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" /></svg>
                  {lang === "zh" ? "Pro 会员 · 无限次 Plus 分析" : "Pro member · Unlimited Plus"}
                </span>
              </div>
            )}

            {error && (
              <div className="mx-auto mb-6 max-w-lg rounded-xl bg-red-500/10 border border-red-500/20 p-4 text-sm text-red-400">
                <p className="font-semibold mb-1">{lang === "zh" ? "分析出错" : "Analysis Error"}</p>
                <p className="text-red-400/80 break-words whitespace-pre-wrap">{error}</p>
                <button onClick={() => setError("")} className="mt-3 text-xs text-white/50 underline hover:text-white/70">{lang === "zh" ? "关闭" : "Close"}</button>
              </div>
            )}

            {isLimitReached ? (
              <div className="mx-auto max-w-lg glass-card p-8 text-center">
                <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-amber-500/10 border border-amber-500/20">
                  <svg className="h-8 w-8 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 1 0-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 0 0 2.25-2.25v-6.75a2.25 2.25 0 0 0-2.25-2.25H6.75a2.25 2.25 0 0 0-2.25 2.25v6.75a2.25 2.25 0 0 0 2.25 2.25Z" /></svg>
                </div>
                <h3 className="text-lg font-bold text-white mb-2">{lang === "zh" ? "今日次数已用完" : "Daily Limit Reached"}</h3>
                <p className="text-sm text-white/40 mb-4">{lang === "zh" ? `免费用户每天可使用 ${usage?.limit || 3} 次 Plus 分析。升级 Pro 获得无限次。` : `Free users get ${usage?.limit || 3} Plus analyses per day. Upgrade to Pro for unlimited.`}</p>
                <div className="flex gap-3 justify-center">
                  <a href="/pro-login" className="btn-pro text-sm">{lang === "zh" ? "升级 Pro" : "Upgrade to Pro"}</a>
                  <a href="/analyze" className="rounded-xl border border-white/20 bg-white/5 px-6 py-3 text-sm text-white/70 hover:bg-white/10 transition">{lang === "zh" ? "使用普通分析" : "Use Standard"}</a>
                </div>
              </div>
            ) : (
              <>
                {/* Input Mode Tabs */}
                <div className="mx-auto mb-6 flex max-w-xl rounded-xl border border-white/10 bg-white/[0.02] p-1 overflow-hidden">
                  {(["upload", "capture", "screen"] as const).map(mode => (
                    <button key={mode} onClick={() => setInputMode(mode)}
                      className={`flex-1 rounded-lg py-2.5 text-xs font-semibold transition-all ${inputMode === mode ? "bg-brand-purple/20 text-white border border-brand-purple/30" : "text-white/40 hover:text-white/60 border border-transparent"}`}>
                      {mode === "upload" ? (lang === "zh" ? "上传视频" : "Upload") : mode === "capture" ? (lang === "zh" ? "实拍模式" : "Camera") : (lang === "zh" ? "屏幕模式" : "Screen")}
                    </button>
                  ))}
                </div>

                {inputMode === "upload" && <UploadZone onUploadComplete={handleUploadComplete} lang={lang} isPro={false} />}

                {inputMode === "capture" && (
                  <div className="mx-auto max-w-xl">
                    <div className="glass-card overflow-hidden">
                      <div className="relative bg-gradient-to-b from-brand-purple/10 to-transparent p-8 text-center">
                        <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-2xl bg-brand-purple/10 border border-brand-purple/20">
                          <svg className="h-10 w-10 text-brand-gold" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25h-9A2.25 2.25 0 0 0 2.25 7.5v9a2.25 2.25 0 0 0 2.25 2.25Z" /></svg>
                        </div>
                        <h3 className="mb-2 text-lg font-semibold text-white">{lang === "zh" ? "实拍模式" : "Live Capture"}</h3>
                        <p className="mb-6 text-sm text-white/40">{lang === "zh" ? "使用摄像头录制挥杆，AI骨架实时引导" : "Record swing with camera, AI skeleton overlay"}</p>
                        <button onClick={() => setLiveCapture(true)} className="btn-primary mx-auto">{lang === "zh" ? "打开摄像头" : "Open Camera"}</button>
                      </div>
                    </div>
                  </div>
                )}

                {inputMode === "screen" && (
                  <div className="mx-auto max-w-xl">
                    <div className="glass-card overflow-hidden">
                      <div className="relative bg-gradient-to-b from-brand-gold/5 to-transparent p-8 text-center">
                        <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-2xl bg-brand-gold/10 border border-brand-gold/20">
                          <span className="text-4xl">📺</span>
                        </div>
                        <h3 className="mb-2 text-lg font-semibold text-white">{lang === "zh" ? "屏幕模式" : "Screen Mode"}</h3>
                        <p className="mb-6 text-sm text-white/40">{lang === "zh" ? "录制电脑屏幕上的高尔夫视频，或用摄像头对着手机/电视拍摄" : "Record golf video from screen, or point camera at phone/TV"}</p>
                        <div className="flex flex-col sm:flex-row gap-3 justify-center">
                          <button onClick={startScreenRecording} className="btn-primary px-6 py-3 text-sm">📺 {lang === "zh" ? "录制屏幕" : "Record Screen"}</button>
                          <button onClick={() => setLiveCapture(true)} className="rounded-xl border border-white/20 bg-white/5 px-6 py-3 text-sm text-white/70 transition hover:bg-white/10">📷 {lang === "zh" ? "对屏拍摄" : "Point Camera"}</button>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}

            <div className="mt-8 grid grid-cols-2 gap-3 mx-auto max-w-lg">
              {[{ zh: "姿势评分 /10", en: "Posture Score /10", icon: "📊" }, { zh: "8阶段评估", en: "8-Phase Eval", icon: "📋" }, { zh: "Pro对比", en: "Pro Compare", icon: "🏌️" }, { zh: "精彩瞬间保存", en: "Save Highlights", icon: "📸" }].map((feat, i) => (
                <div key={i} className="rounded-xl border border-white/5 bg-white/[0.02] p-3 text-center">
                  <span className="text-lg">{feat.icon}</span>
                  <p className="mt-1 text-[11px] text-white/50">{lang === "zh" ? feat.zh : feat.en}</p>
                </div>
              ))}
            </div>
            <div className="mt-6 text-center">
              <a href="/analyze" className="text-xs text-white/30 hover:text-white/50 transition">← {lang === "zh" ? "返回普通分析" : "Back to Standard Analysis"}</a>
            </div>
          </div>
        )}

        {stage === "processing" && (
          <div className="relative">
            <AnalysisWaiting progress={progress} lang={lang} mode="pro" />
            {processingClub && (
              <div className="fixed bottom-6 left-4 right-4 z-50 animate-fade-in">
                <div className="mx-auto max-w-md rounded-2xl border border-brand-gold/30 bg-brand-dark/95 backdrop-blur-xl p-4 shadow-2xl">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-gold/10 text-xl">🏌️</div>
                      <div>
                        <p className="text-sm font-semibold text-white">
                          {processingClub.club_type !== "UNKNOWN"
                            ? (lang === "zh" ? `检测到：${CLUB_DISPLAY[processingClub.club_type] || processingClub.club_type}` : `Detected: ${processingClub.club_type}`)
                            : (lang === "zh" ? "未能识别球杆" : "Club not identified")}
                        </p>
                        {processingClub.club_type !== "UNKNOWN" && (
                          <p className="text-[10px] text-white/30">
                            {lang === "zh" ? "AI 置信度" : "Confidence"}: {Math.round(processingClub.confidence * 100)}%
                            {processingClub.confidence < 0.7 && <span className="ml-1 text-yellow-400">{lang === "zh" ? "· 建议确认" : "· verify"}</span>}
                          </p>
                        )}
                      </div>
                    </div>
                    <button onClick={() => setShowClubPicker(true)}
                      className="rounded-lg border border-brand-gold/30 bg-brand-gold/10 px-3 py-1.5 text-xs font-medium text-brand-gold transition hover:bg-brand-gold/20">
                      {processingClub.club_type !== "UNKNOWN" ? (lang === "zh" ? "修改 ▾" : "Change ▾") : (lang === "zh" ? "选择 ▾" : "Select ▾")}
                    </button>
                  </div>
                </div>
              </div>
            )}
            {showClubPicker && (
              <div className="fixed inset-0 z-[100] flex items-end sm:items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setShowClubPicker(false)}>
                <div className="w-full max-w-md rounded-t-2xl sm:rounded-2xl bg-brand-dark border border-white/10 p-5 animate-fade-in" onClick={(e) => e.stopPropagation()}>
                  <div className="mb-4 flex items-center justify-between">
                    <h3 className="text-base font-bold text-white">{lang === "zh" ? "选择球杆" : "Select Club"}</h3>
                    <button onClick={() => setShowClubPicker(false)} className="text-white/30 hover:text-white/60 text-lg">&times;</button>
                  </div>
                  <div className="space-y-3">
                    {CLUB_GROUPS.map((group) => (
                      <div key={group.id}>
                        <p className="mb-1.5 text-[10px] font-semibold text-white/40 uppercase tracking-wider">{lang === "zh" ? group.label_zh : group.label_en}</p>
                        <div className="flex flex-wrap gap-1.5">
                          {group.clubs.map((club) => (
                            <button key={club} onClick={() => handleProcessingClubChange(club)}
                              className={`rounded-lg border px-3 py-2 text-xs font-medium transition-all ${
                                processingClub?.club_type === club
                                  ? "border-brand-gold/50 bg-brand-gold/20 text-brand-gold"
                                  : "border-white/10 bg-white/5 text-white/60 hover:border-brand-gold/30 hover:text-white"
                              }`}>{club}</button>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {stage === "results" && result && (
          <>
            <PlusResultView result={result} lang={lang} externalVideoSrc={sessionVideoSrc} />
            <div className="text-center py-6">
              <button
                onClick={() => {
                  setSessionVideoSrc((prev) => {
                    if (prev) try { URL.revokeObjectURL(prev); } catch { /* */ }
                    return null;
                  });
                  setStage("upload");
                  setResult(null);
                  setError("");
                }}
                className="btn-primary"
              >
                {lang === "zh" ? "再次分析" : "Analyze Again"}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
