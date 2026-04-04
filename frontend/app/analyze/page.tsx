"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import UploadZone from "@/components/UploadZone";
import HUDOverlay from "@/components/HUDOverlay";
import KeyframeStrip from "@/components/KeyframeStrip";
import SimAnimation from "@/components/SimAnimation";
import ProComparison from "@/components/ProComparison";
import Skeleton3DViewer from "@/components/Skeleton3DViewer";
import ScreenModeCapture from "@/components/ScreenModeCapture";
import { preloadPoseModel } from "@/lib/mediapipe-assets";
import AnalysisWaiting from "@/components/AnalysisWaiting";
import { saveAnalysisVideo } from "@/lib/video-store";
import { fetchWithRetry, makeFormData } from "@/lib/fetch-retry";
import { isVideoFile, uploadVideoToGemini } from "@/lib/upload-video";
import { stripResultForStorage } from "@/lib/strip-result";
import { normalizedTotalScoreForStorage } from "@/lib/safe-analysis-score";
import { patchLocalHistoryVideoR2Key } from "@/lib/history-sync-record";
import { expandStellarProForUi } from "@/lib/stellar-pro-result";
import { pruneLocalStellarHistoryRecords } from "@/lib/pro-history-retention";
import {
  DEFAULT_PRO_V2_MODAL_URL,
  normalizeProV2UrlListsFromPrecheck,
} from "@/lib/pro-v2-endpoints";
import { runProV2AnalyzeMultipart, yieldUiBeforeHeavyParse } from "@/lib/pro-v2-analyze-client";
import {
  consumeReanalyzeFromHistoryPayload,
  fetchVideoBlobForHistoryReanalyze,
  reanalyzeHistoryFilename,
} from "@/lib/reanalyze-from-history";

interface AnalysisResult {
  analysis_id: string;
  type: string;
  scores: Record<string, number>;
  total_score: number;
  issues: string[];
  issues_zh: string[];
  suggestions: string[];
  suggestions_zh: string[];
  summary: string;
  summary_zh: string;
  what_i_see?: string;
  what_i_see_zh?: string;
  is_golf_swing?: boolean;
  keyframes: Array<{
    phase: string;
    label_en: string;
    label_zh: string;
    timestamp: number;
    image_base64: string;
  }>;
  skeleton_data: {
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
  prediction: {
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
    club_detection_confidence?: number;
    blur_speed?: number;
    trajectory_speed?: number;
    fused_speed?: number;
    fusion_weights?: { blur: number; trajectory: number; formula: number };
    speed_confidence?: string;
    error_estimate_pct?: number;
    blur_confidence?: string;
    trajectory_confidence?: string;
    trajectory_tracked_frames?: number;
    hand?: "R" | "L" | "UNKNOWN";
    hand_confidence?: number;
    baseline_distance?: number;
    technique_multiplier?: number;
    strike_multiplier?: number;
    speed_multiplier?: number;
    distance_confidence?: number;
    distance_debug?: Record<string, unknown>;
  };
}

interface ClubDetection { club_type: string; club_group: string; confidence: number; hand?: "R" | "L" }
type Stage = "upload" | "processing" | "results";
type InputMode = "upload" | "capture" | "screen";
type AnalysisMode = "lite" | "pro";

export default function AnalyzePage() {
  const router = useRouter();
  const [stage, setStage] = useState<Stage>("upload");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState(0);
  const [inputMode, setInputMode] = useState<InputMode>("upload");
  const [liveCapture, setLiveCapture] = useState(false);
  const [showExtendedHUD, setShowExtendedHUD] = useState(false);
  const [lang, setLang] = useState<"en" | "zh">("zh");
  const [activeTab, setActiveTab] = useState<"analysis" | "3d" | "comparison">("analysis");
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>("lite");
  const [username, setUsername] = useState("");
  const [authChecked, setAuthChecked] = useState(false);
  const [screenRecording, setScreenRecording] = useState(false);
  const screenRecRef = useRef<MediaRecorder | null>(null);
  const screenStreamRef = useRef<MediaStream | null>(null);
  const screenChunksRef = useRef<Blob[]>([]);
  const [screenRecTime, setScreenRecTime] = useState(0);
  const screenTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [showClubPicker, setShowClubPicker] = useState(false);
  const backendBaseRef = useRef<string>(process.env.NEXT_PUBLIC_BACKEND_URL || "https://stellar1-backend.onrender.com");
  const lastBlobRef = useRef<{ blob: Blob; filename: string } | null>(null);
  /** Pro v2 screen preprocess: set before opening 实拍 when source is 对屏拍摄 (screen tab). */
  const screenCaptureForProV2Ref = useRef(false);
  const [processingClub, setProcessingClub] = useState<ClubDetection | null>(null);
  const processingClubRef = useRef<ClubDetection | null>(null);
  const [detectedHand, setDetectedHand] = useState<"R" | "L" | null>(null);
  const [handConfirmed, setHandConfirmed] = useState(false);
  const [showHandPopup, setShowHandPopup] = useState(false);
  const handRef = useRef<"R" | "L">("R");
  const [processingProScreenMode, setProcessingProScreenMode] = useState(false);
  /** 防止重复提交 Pro v2 / 分析流程。 */
  const analysisInFlightRef = useRef(false);

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
    if (!token) {
      router.push("/login");
      return;
    }
    const userStr = localStorage.getItem("stellar_user");
    if (userStr) {
      try {
        const u = JSON.parse(userStr);
        setUsername(u.username || u.email || "");
      } catch { /* ignore */ }
    }
    setAuthChecked(true);
    preloadPoseModel();
  }, [router]);

  useEffect(() => {
    if (!authChecked) return;
    const p = consumeReanalyzeFromHistoryPayload();
    if (!p || p.page !== "analyze") return;
    void (async () => {
      const blob = await fetchVideoBlobForHistoryReanalyze(p.analysisId, p.videoUrl);
      if (!blob || blob.size === 0) {
        setError(
          lang === "zh"
            ? "无法加载该记录原视频。请确认本机已缓存或已登录且云端仍保存视频。"
            : "Could not load the original video for this record.",
        );
        return;
      }
      const mode: AnalysisMode = p.analysisMode === "pro" ? "pro" : "lite";
      const screenTag = Boolean(p.proV2ScreenMode);
      if (screenTag) setInputMode("screen");
      processBlob(blob, reanalyzeHistoryFilename(blob), screenTag, mode);
    })();
    // processBlob 为稳定闭包即可；仅依赖登录就绪与语言（错误文案）
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authChecked, lang]);

  function resolveProV2ScreenMode(
    filename: string,
    explicit: boolean | undefined,
  ): boolean {
    if (explicit === true) return true;
    if (explicit === false) return false;
    if (inputMode === "screen") return true;
    return /(?:^|\/)(screen-capture\.webm|pro-screen\.webm)$/i.test(filename);
  }

  async function sendFileForAnalysis(
    file: File | Blob,
    filename: string,
    proV2ScreenMode?: boolean,
    modeOverride?: AnalysisMode,
  ): Promise<AnalysisResult> {
    const isPro = (modeOverride ?? analysisMode) === "pro";
    if (!isPro) {
      console.warn("[stellar v2] 当前为 LITE 模式：请求走 /api/analyze（Cloudflare Edge），不会进入 Modal 日志。");
    }

    if (isPro) {
      // Pro mode: precheck (Edge) → always try Modal first (all regions, including CN).
      // Render runs only when NEXT_PUBLIC_PRO_V2_RENDER_FALLBACK=true; cnNetworkHint never reorders hosts.
      // Never proxy through the CF Worker — its 30 s wall-clock limit kills long analyses.
      const token = localStorage.getItem("stellar_token");
      const authHeaders: Record<string, string> = {};
      if (token && token.includes(".")) authHeaders["Authorization"] = `Bearer ${token}`;

      // ① Precheck — Modal + Render URL lists; try order is Modal → Render (Render opt-in via env, same for CN)
      const defaultBackend =
        process.env.NEXT_PUBLIC_BACKEND_URL || "https://stellar1-backend.onrender.com";
      let proNetworkHint = "";
      let modalUrls: string[] = [DEFAULT_PRO_V2_MODAL_URL];
      let backendUrls: string[] = [defaultBackend];
      try {
        const pc = await fetch("/api/pro/precheck", {
          method: "POST",
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        });
        if (pc.ok) {
          const data = await pc.json();
          const lists = normalizeProV2UrlListsFromPrecheck(data);
          modalUrls = lists.modalUrls;
          backendUrls = lists.backendUrls.length ? lists.backendUrls : [defaultBackend];
          if (data.network_hint === "cn") proNetworkHint = "cn";
        }
      } catch { /* precheck failed — proceed with defaults */ }
      backendBaseRef.current = backendUrls[0] || defaultBackend;

      const mb = (file.size / 1024 / 1024).toFixed(1);
      const cnPro = proNetworkHint === "cn";
      const screenMode = resolveProV2ScreenMode(filename, proV2ScreenMode);
      const { response: res, route: proServedBy } = await runProV2AnalyzeMultipart(
        file as Blob,
        filename,
        authHeaders,
        {
          modalUrls,
          backendUrls,
          cnNetworkHint: cnPro,
          screenMode,
          modalTimeoutMs: cnPro ? 45_000 : 120_000,
          renderTimeoutMs: 360_000,
          logPrefix: `[stellar v2] ${mb}MB`,
        },
      );

      const runtimeHeader = res.headers.get("x-stellar-runtime") || "unknown";
      console.log(`[stellar v2] Pro response: ${res.status} (served_by=${proServedBy}, runtime=${runtimeHeader})`);
      if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try { detail = (await res.json()).detail || detail; } catch { /* ignore */ }
        throw new Error(`Pro分析失败 [${res.status}]: ${detail}`);
      }
      await yieldUiBeforeHeavyParse();
      const rawText = await res.text();
      let rawPro: Record<string, unknown>;
      try {
        rawPro = JSON.parse(rawText) as Record<string, unknown>;
      } catch {
        throw new Error("Pro 分析返回数据无法解析，请重试");
      }
      return expandStellarProForUi(rawPro) as AnalysisResult;
    }

    // Lite: /api/analyze on Edge. Video: upload-video → file_uri (fast path) + same File in FormData so Edge can re-upload if URI is stale (403).
    const headers: Record<string, string> = {};
    const token = localStorage.getItem("stellar_token");
    if (token && token.includes(".")) headers["Authorization"] = `Bearer ${token}`;

    let res: Response;
    try {
      if (isVideoFile(file as File, filename)) {
        const uploadCtrl = new AbortController();
        const uploadTimer = setTimeout(() => uploadCtrl.abort(), 360_000);
        try {
          const up = await uploadVideoToGemini(
            file as File,
            filename,
            headers,
            undefined,
            uploadCtrl.signal,
          );
          const fd = new FormData();
          fd.append("file_uri", up.file_uri);
          fd.append("mime_type", up.mime_type);
          fd.append("file", file as File, filename);
          if (typeof up.gemini_key_index === "number") {
            fd.append("gemini_key_index", String(up.gemini_key_index));
          }
          const analyzeCtrl = new AbortController();
          const analyzeTimer = setTimeout(() => analyzeCtrl.abort(), 180_000);
          try {
            res = await fetch("/api/analyze", {
              method: "POST",
              headers,
              body: fd,
              signal: analyzeCtrl.signal,
            });
          } finally {
            clearTimeout(analyzeTimer);
          }
        } finally {
          clearTimeout(uploadTimer);
        }
      } else {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 120_000);
        try {
          res = await fetch("/api/analyze", {
            method: "POST",
            headers,
            body: makeFormData(file as Blob, filename),
            signal: controller.signal,
          });
        } finally {
          clearTimeout(timer);
        }
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        throw new Error("分析超时，请稍后重试");
      }
      throw new Error(`网络错误：${e instanceof Error ? e.message : "无法连接服务器"}`);
    }

    console.log(`[stellar v2] Lite Edge response: ${res.status}`);

    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const d = await res.json();
        detail = d.detail || detail;
      } catch {
        try { detail = (await res.text()).substring(0, 200) || detail; } catch { /* ignore */ }
      }
      throw new Error(`分析失败 [${res.status}]: ${detail}`);
    }
    return res.json();
  }

  const RECALCULATE_TIMEOUT_MS = 10_000;

  async function recalculatePredictionFromBackend(
    data: AnalysisResult,
    overrides?: { club_type?: string; club_group?: string; hand?: "R" | "L" | "UNKNOWN"; hand_confidence?: number; preferred_ball_speed?: number },
  ): Promise<AnalysisResult["prediction"] | null> {
    try {
      const backendUrl = backendBaseRef.current || process.env.NEXT_PUBLIC_BACKEND_URL || "https://stellar1-backend.onrender.com";
      const poseFrames = data.pose_frames || [];
      if (poseFrames.length === 0) return null;
      const mid = poseFrames[Math.floor(poseFrames.length / 2)];
      const allAngles = poseFrames.map((f) => f.angles || {});
      const swingDuration = poseFrames.length >= 2
        ? Math.max(0.3, (poseFrames[poseFrames.length - 1].timestamp || 0) - (poseFrames[0].timestamp || 0))
        : 1.2;
      const token = localStorage.getItem("stellar_token");
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (token && token.includes(".")) headers.Authorization = `Bearer ${token}`;
      const body = {
        pose_data: { angles: mid?.angles || {} },
        all_frame_angles: allAngles,
        swing_duration: swingDuration,
        club_type: overrides?.club_type ?? data.prediction.club_type,
        club_group: overrides?.club_group ?? data.prediction.club_group,
        hand: overrides?.hand ?? data.prediction.hand ?? "UNKNOWN",
        hand_confidence: overrides?.hand_confidence ?? data.prediction.hand_confidence ?? 0,
        preferred_ball_speed: overrides?.preferred_ball_speed ?? data.prediction.fused_speed ?? undefined,
      };
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), RECALCULATE_TIMEOUT_MS);
      let res: Response;
      try {
        res = await fetch(`${backendUrl}/analyze/recalculate`, {
          method: "POST",
          headers,
          body: JSON.stringify(body),
          signal: ctrl.signal,
        });
      } catch (e) {
        if ((e as Error)?.name === "AbortError") {
          console.warn(`[analyze] /analyze/recalculate aborted after ${RECALCULATE_TIMEOUT_MS}ms`);
        } else {
          console.warn("[analyze] /analyze/recalculate fetch error:", e);
        }
        return null;
      } finally {
        clearTimeout(t);
      }
      if (!res.ok) {
        console.warn("[analyze] /analyze/recalculate HTTP", res.status);
        return null;
      }
      const payload = await res.json();
      return payload?.prediction ?? null;
    } catch (e) {
      console.warn("[analyze] recalculatePredictionFromBackend:", e);
      return null;
    }
  }

  function saveToLocalHistory(data: AnalysisResult, mode: string) {
    try {
      const key = "stellar_history_local";
      const existing = JSON.parse(localStorage.getItem(key) || "[]");
      const entry = {
        id: data.analysis_id || `local-${Date.now()}`,
        type: data.type || mode,
        total_score: normalizedTotalScoreForStorage(data.total_score),
        result_json: JSON.stringify(stripResultForStorage(data)),
        created_at: new Date().toISOString(),
        _local: true,
      };
      const filtered = existing.filter((r: { id: string }) => r.id !== entry.id);
      filtered.unshift(entry);
      localStorage.setItem(key, JSON.stringify(filtered.slice(0, 200)));
      pruneLocalStellarHistoryRecords();
    } catch { /* ignore quota errors */ }
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

  async function saveAnalysisToHistory(
    data: AnalysisResult,
    blob?: Blob,
    filename?: string,
    modeForSave?: AnalysisMode,
  ) {
    const token = localStorage.getItem("stellar_token");
    const mode = modeForSave ?? analysisMode;

    saveToLocalHistory(data, mode);

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
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          analysis_id: data.analysis_id,
          type: data.type || mode,
          total_score: normalizedTotalScoreForStorage(data.total_score),
          result: data,
          video_r2_key: videoR2Key,
        }),
      });
      if (res.ok) {
        const saved = await res.json().catch(() => ({}));
        if (saved.success !== false) {
          markLocalRecordSynced(data.analysis_id);
        }
      } else {
        console.warn("[history] server save failed:", res.status);
      }
    } catch (e) {
      console.warn("[history] server save error:", e);
    }
  }

  async function processBlob(
    blob: Blob,
    filename: string,
    proV2ScreenMode?: boolean,
    analysisModeOverride?: AnalysisMode,
  ) {
    if (analysisInFlightRef.current) {
      console.warn("[analyze] analyze already in flight, ignoring duplicate trigger");
      return;
    }
    analysisInFlightRef.current = true;
    try {
    const modeForRun = analysisModeOverride ?? analysisMode;
    if (analysisModeOverride) setAnalysisMode(analysisModeOverride);
    setProcessingProScreenMode(
      modeForRun === "pro" && resolveProV2ScreenMode(filename, proV2ScreenMode),
    );
    setStage("processing");
    setError("");
    setProgress(0);
    setProcessingClub(null);
    processingClubRef.current = null;
    setDetectedHand(null);
    setHandConfirmed(false);
    setShowHandPopup(false);
    lastBlobRef.current = { blob, filename };
    const handWasConfirmed = false;

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
        const fallback: ClubDetection = { club_type: "UNKNOWN", club_group: "IRON", confidence: 0 };
        setProcessingClub(fallback);
        processingClubRef.current = fallback;
      }
    }, 6000);

    try {
      const data = await sendFileForAnalysis(blob, filename, proV2ScreenMode, modeForRun);
      clearInterval(progressInterval);
      clearTimeout(clubFallbackTimer);

      const userClub = processingClubRef.current as ClubDetection | null;
      if (userClub && data.prediction) {
        if (userClub.club_type !== "UNKNOWN") {
          data.prediction.club_type = userClub.club_type;
          data.prediction.club_group = userClub.club_group;
          data.prediction.club_detection_confidence = userClub.confidence;
        }
      }

      if (data.prediction?.hand && data.prediction.hand !== "UNKNOWN") {
        setDetectedHand(data.prediction.hand);
        handRef.current = data.prediction.hand;
      }

      const analysisId = data.analysis_id;
      setProgress(100);
      setResult(data);
      setStage("results");

      const poseOk = (data.pose_frames?.length ?? 0) > 0;
      const clubKnown =
        Boolean(data.prediction?.club_type && data.prediction.club_type !== "UNKNOWN");
      const shouldBackgroundRecalc =
        poseOk && data.prediction && (clubKnown || handWasConfirmed);

      if (shouldBackgroundRecalc) {
        void (async () => {
          try {
            const recomputed = await recalculatePredictionFromBackend(data, {
              club_type: data.prediction.club_type,
              club_group: data.prediction.club_group,
              hand: handWasConfirmed ? handRef.current : (data.prediction.hand ?? "UNKNOWN"),
              hand_confidence: handWasConfirmed ? 1.0 : (data.prediction.hand_confidence ?? 0),
              preferred_ball_speed: data.prediction.fused_speed,
            });
            if (recomputed) {
              setResult((prev) => {
                if (!prev || prev.analysis_id !== analysisId) return prev;
                return { ...prev, prediction: { ...prev.prediction, ...recomputed } };
              });
            }
          } catch (e) {
            console.warn("[analyze] background recalculate failed:", e);
          }
        })();
      }

      void saveAnalysisVideo(data.analysis_id, blob, filename).catch(() => {});
      try {
        await saveAnalysisToHistory(data, blob, filename, modeForRun);
      } catch (e) {
        console.warn("[analyze] history save failed:", e);
      }
    } catch (err: unknown) {
      clearInterval(progressInterval);
      clearTimeout(clubFallbackTimer);
      setProgress(0);
      setProcessingProScreenMode(false);
      setError(err instanceof Error ? err.message : "分析失败，请重试");
      setStage("upload");
    }
    } finally {
      analysisInFlightRef.current = false;
    }
  }

  function extractFrameFromBlob(blob: Blob): Promise<Blob | null> {
    return new Promise((resolve) => {
      const isVideo = blob.type.startsWith("video/") ||
        /\.(mp4|mov|webm|avi|m4v)$/i.test((blob as File).name || "");
      if (!isVideo && blob.type.startsWith("image/")) {
        resolve(blob);
        return;
      }
      const video = document.createElement("video");
      video.muted = true;
      video.playsInline = true;
      video.preload = "auto";
      video.crossOrigin = "anonymous";
      const url = URL.createObjectURL(blob);
      video.src = url;
      let resolved = false;
      const cleanup = () => { try { URL.revokeObjectURL(url); } catch { /* */ } };
      const done = (b: Blob | null) => { if (resolved) return; resolved = true; cleanup(); resolve(b); };
      const timer = setTimeout(() => { console.warn("[club-detect] frame extraction timeout"); done(null); }, 10000);

      const captureFrame = () => {
        clearTimeout(timer);
        try {
          const w = video.videoWidth || 640;
          const h = video.videoHeight || 480;
          const canvas = document.createElement("canvas");
          const scale = Math.min(640 / w, 1);
          canvas.width = Math.round(w * scale);
          canvas.height = Math.round(h * scale);
          const ctx = canvas.getContext("2d");
          if (!ctx) { done(null); return; }
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          canvas.toBlob((b) => done(b), "image/jpeg", 0.8);
        } catch { done(null); }
      };

      video.onseeked = captureFrame;
      video.onloadeddata = () => {
        if (video.duration && isFinite(video.duration) && video.duration > 0.5) {
          video.currentTime = video.duration * 0.4;
        } else {
          captureFrame();
        }
      };
      video.onerror = () => { console.warn("[club-detect] video load error"); clearTimeout(timer); done(null); };
      video.load();
    });
  }

  function extractFrameAtPercent(blob: Blob, pct: number): Promise<Blob | null> {
    return new Promise((resolve) => {
      const isVideo = blob.type.startsWith("video/") || /\.(mp4|mov|webm|avi|m4v)$/i.test((blob as File).name || "");
      if (!isVideo && blob.type.startsWith("image/")) { resolve(blob); return; }
      const video = document.createElement("video");
      video.muted = true; video.playsInline = true; video.preload = "auto"; video.crossOrigin = "anonymous";
      const url = URL.createObjectURL(blob);
      video.src = url;
      let resolved = false;
      const cleanup = () => { try { URL.revokeObjectURL(url); } catch { /* */ } };
      const done = (b: Blob | null) => { if (resolved) return; resolved = true; cleanup(); resolve(b); };
      const timer = setTimeout(() => done(null), 8000);
      const captureFrame = () => {
        clearTimeout(timer);
        try {
          const w = video.videoWidth || 640;
          const h = video.videoHeight || 480;
          const canvas = document.createElement("canvas");
          const scale = Math.min(640 / w, 1);
          canvas.width = Math.round(w * scale);
          canvas.height = Math.round(h * scale);
          const ctx = canvas.getContext("2d");
          if (!ctx) { done(null); return; }
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          canvas.toBlob((b) => done(b), "image/jpeg", 0.8);
        } catch { done(null); }
      };
      video.onseeked = captureFrame;
      video.onloadeddata = () => {
        if (video.duration && isFinite(video.duration) && video.duration > 0.5) {
          video.currentTime = video.duration * pct;
        } else captureFrame();
      };
      video.onerror = () => { clearTimeout(timer); done(null); };
      video.load();
    });
  }

  async function detectClubFromBlob(blob: Blob) {
    try {
      const token = localStorage.getItem("stellar_token");
      const headers: Record<string, string> = {};
      if (token) headers["Authorization"] = `Bearer ${token}`;

      const framePcts = [0.25, 0.4, 0.6];
      const frameBlobs = await Promise.all(framePcts.map(p => extractFrameAtPercent(blob, p)));
      const validFrames = frameBlobs.filter((b): b is Blob => b != null && b.size > 0);
      if (validFrames.length === 0) {
        const single = await extractFrameFromBlob(blob);
        if (single) validFrames.push(single);
      }
      if (validFrames.length === 0) { console.warn("[club-detect] no frames extracted"); return; }

      const results = await Promise.all(validFrames.map(async (frameBlob) => {
        try {
          const fd = new FormData();
          fd.append("frame", frameBlob, "frame.jpg");
          const res = await fetch("/api/club-detect", { method: "POST", headers, body: fd });
          if (!res.ok) return null;
          return await res.json();
        } catch { return null; }
      }));

      const valid = results.filter((r): r is { club_type: string; club_group: string; confidence: number; hand?: string } =>
        r != null && r.club_type && r.club_type !== "UNKNOWN"
      );

      if (valid.length === 0) {
        const first = results.find(r => r != null);
        const hand = (first?.hand === "L" ? "L" : "R") as "R" | "L";
        const fallback: ClubDetection = { club_type: "UNKNOWN", club_group: "IRON", confidence: 0, hand };
        setProcessingClub(fallback);
        processingClubRef.current = fallback;
        return;
      }

      const votes: Record<string, { count: number; totalConf: number; group: string }> = {};
      for (const r of valid) {
        if (!votes[r.club_type]) votes[r.club_type] = { count: 0, totalConf: 0, group: r.club_group };
        votes[r.club_type].count++;
        votes[r.club_type].totalConf += r.confidence;
      }
      const sorted = Object.entries(votes).sort((a, b) => b[1].count - a[1].count || b[1].totalConf - a[1].totalConf);
      const winner = sorted[0];
      const avgConf = winner[1].totalConf / winner[1].count;

      const handVotes = { R: 0, L: 0 };
      for (const r of results.filter(Boolean)) {
        const h = (r as { hand?: string }).hand === "L" ? "L" : "R";
        handVotes[h]++;
      }
      const hand: "R" | "L" = handVotes.L > handVotes.R ? "L" : "R";

      const data: ClubDetection = {
        club_type: winner[0],
        club_group: winner[1].group,
        confidence: Math.round(avgConf * 100) / 100,
        hand,
      };
      console.log("[club-detect] multi-frame result:", data, "votes:", votes);
      setProcessingClub(data);
      processingClubRef.current = data;
      if (!handConfirmed) { setDetectedHand(hand); handRef.current = hand; setShowHandPopup(true); }
    } catch (e) {
      console.warn("[club-detect] error:", e);
    }
  }

  function handleProcessingClubChange(clubType: string) {
    setShowClubPicker(false);
    const groupMap: Record<string, string> = {};
    for (const g of CLUB_GROUPS) for (const c of g.clubs) groupMap[c] = g.id;
    const newClub = { club_type: clubType, club_group: groupMap[clubType] || "IRON", confidence: 1.0 };
    setProcessingClub(newClub);
    processingClubRef.current = newClub;
  }

  async function handleClubOverride(clubType: string) {
    setShowClubPicker(false);
    if (!result) return;

    const groupMap: Record<string, string> = {};
    for (const g of CLUB_GROUPS) for (const c of g.clubs) groupMap[c] = g.id;
    const newGroup = groupMap[clubType] || "IRON";

    const next = {
      ...result,
      prediction: {
        ...result.prediction,
        club_type: clubType,
        club_group: newGroup,
        club_detection_confidence: 1.0,
      },
    };
    const recomputed = await recalculatePredictionFromBackend(next, {
      club_type: clubType,
      club_group: newGroup,
      hand: next.prediction.hand ?? "UNKNOWN",
      hand_confidence: next.prediction.hand_confidence ?? 0,
      preferred_ball_speed: next.prediction.fused_speed,
    });
    setResult({
      ...next,
      prediction: recomputed ? { ...next.prediction, ...recomputed } : next.prediction,
    });
  }

  async function handleHandConfirm() {
    setHandConfirmed(true);
    setShowHandPopup(false);
    if (!result) return;
    const selectedHand = (detectedHand || handRef.current || "UNKNOWN") as "R" | "L" | "UNKNOWN";
    const next = {
      ...result,
      prediction: {
        ...result.prediction,
        hand: selectedHand,
        hand_confidence: 1.0,
      },
    };
    const recomputed = await recalculatePredictionFromBackend(next, {
      hand: selectedHand,
      hand_confidence: 1.0,
      club_type: next.prediction.club_type,
      club_group: next.prediction.club_group,
      preferred_ball_speed: next.prediction.fused_speed,
    });
    setResult({
      ...next,
      prediction: recomputed ? { ...next.prediction, ...recomputed } : next.prediction,
    });
  }

  const handleUploadComplete = useCallback(
    (file: File) => {
      screenCaptureForProV2Ref.current = false;
      processBlob(file, file.name, false);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [lang, analysisMode]
  );

  const handleVideoCapture = useCallback(
    (videoBlob: Blob) => {
      setLiveCapture(false);
      processBlob(videoBlob, "swing-capture.webm", screenCaptureForProV2Ref.current);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [lang, analysisMode]
  );

  const handleScreenCapture = useCallback(
    (imageBase64: string) => {
      setLiveCapture(false);
      const byteChars = atob(imageBase64);
      const byteArray = new Uint8Array(byteChars.length);
      for (let i = 0; i < byteChars.length; i++) byteArray[i] = byteChars.charCodeAt(i);
      processBlob(
        new Blob([byteArray], { type: "image/jpeg" }),
        "swing-capture.jpg",
        screenCaptureForProV2Ref.current,
      );
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [lang, analysisMode]
  );

  async function startScreenRecording() {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      screenStreamRef.current = stream;
      screenChunksRef.current = [];

      const mime = MediaRecorder.isTypeSupported("video/webm") ? "video/webm" : "video/mp4";
      const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 3000000 });

      rec.ondataavailable = (e) => { if (e.data.size > 0) screenChunksRef.current.push(e.data); };
      rec.onstop = () => {
        stream.getTracks().forEach(t => t.stop());
        if (screenTimerRef.current) clearInterval(screenTimerRef.current);
        setScreenRecording(false);
        const blob = new Blob(screenChunksRef.current, { type: mime });
        if (blob.size > 0) processBlob(blob, "screen-capture.webm", true);
      };

      rec.start(200);
      screenRecRef.current = rec;
      setScreenRecording(true);
      setScreenRecTime(0);

      screenTimerRef.current = setInterval(() => {
        setScreenRecTime(prev => {
          if (prev >= 30) { stopScreenRecording(); return prev; }
          return prev + 1;
        });
      }, 1000);

      stream.getVideoTracks()[0].onended = () => {
        if (rec.state !== "inactive") rec.stop();
      };
    } catch {
      screenCaptureForProV2Ref.current = true;
      setLiveCapture(true);
    }
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

  if (screenRecording) {
    return (
      <div className="fixed inset-0 z-[9999] flex flex-col items-center justify-center bg-black/90">
        <div className="mb-6 h-4 w-4 rounded-full bg-red-500 animate-pulse" />
        <h2 className="mb-2 text-xl font-bold text-white">
          {lang === "zh" ? "正在录制屏幕..." : "Recording Screen..."}
        </h2>
        <p className="mb-6 text-sm text-white/40">
          {lang === "zh" ? "播放高尔夫挥杆视频，录制完成后自动分析" : "Play the golf swing video, analysis starts after recording"}
        </p>
        <div className="mb-4 text-3xl font-bold text-brand-gold">
          {Math.floor(screenRecTime / 60).toString().padStart(2, "0")}:{(screenRecTime % 60).toString().padStart(2, "0")}
          <span className="text-sm text-white/30 ml-2">/ 00:30</span>
        </div>
        <div className="mb-8 w-64 h-1 rounded-full bg-white/10 overflow-hidden">
          <div className="h-full rounded-full bg-red-500 transition-all duration-1000" style={{ width: `${(screenRecTime / 30) * 100}%` }} />
        </div>
        <button onClick={stopScreenRecording} className="btn-primary px-8 py-3 text-sm">
          {lang === "zh" ? "停止录制并分析" : "Stop & Analyze"}
        </button>
      </div>
    );
  }

  if (liveCapture) {
    return (
      <ScreenModeCapture
        onCapture={handleScreenCapture}
        onVideoCapture={handleVideoCapture}
        onExit={() => setLiveCapture(false)}
        lang={lang}
      />
    );
  }

  return (
    <div className="min-h-screen">
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
                <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A17.933 17.933 0 0 1 12 21.75c-2.676 0-5.216-.584-7.499-1.632Z" />
                </svg>
                <span className="font-medium text-brand-gold">{username}</span>
              </a>
            )}
            <button onClick={() => setLang(lang === "en" ? "zh" : "en")}
              className="rounded-lg border border-white/10 px-3 py-1 text-xs text-white/60 transition hover:text-white">
              {lang === "en" ? "中文" : "EN"}
            </button>
            <button onClick={handleSignOut}
              className="rounded-lg border border-white/10 px-3 py-1 text-xs text-white/40 transition hover:text-white/70">
              退出
            </button>
          </div>
        </div>
      </nav>

      <div className="mx-auto max-w-6xl px-4 py-6">
        {stage === "upload" && (
          <div className="animate-fade-in">
            <h1 className="mb-2 text-center text-3xl font-bold text-white">
              {lang === "en" ? "Swing Analysis" : "挥杆分析"}
            </h1>
            <p className="mb-6 text-center text-sm text-white/40">
              {lang === "en" ? "Upload, capture, or record from screen" : "上传视频、实拍、或屏幕录制"}
              <span className="ml-2 text-[9px] text-white/15">v2.2</span>
            </p>

            {error && (
              <div className="mx-auto mb-6 max-w-lg rounded-xl bg-red-500/10 border border-red-500/20 p-4 text-sm text-red-400">
                <p className="font-semibold mb-1">分析出错 (v2.2)</p>
                <p className="text-red-400/80 break-words whitespace-pre-wrap">{error}</p>
                <button
                  onClick={() => setError("")}
                  className="mt-3 text-xs text-white/50 underline hover:text-white/70"
                >
                  关闭
                </button>
              </div>
            )}

            {/* Analysis Mode Selector */}
            <div className="mx-auto mb-4 max-w-xl">
              <div className="flex rounded-xl border border-white/10 bg-white/[0.02] p-1 overflow-hidden">
                <button onClick={() => setAnalysisMode("lite")}
                  className={`flex-1 rounded-lg py-3 text-sm font-semibold transition-all ${
                    analysisMode === "lite"
                      ? "bg-brand-purple/20 text-white border border-brand-purple/30"
                      : "text-white/40 hover:text-white/60 border border-transparent"
                  }`}>
                  <span className="block">{lang === "zh" ? "普通分析" : "Standard"}</span>
                  <span className="block text-[10px] font-normal text-white/30 mt-0.5">Stellar AI</span>
                </button>
                <button onClick={() => setAnalysisMode("pro")}
                  className={`flex-1 rounded-lg py-3 text-sm font-semibold transition-all ${
                    analysisMode === "pro"
                      ? "bg-brand-gold/20 text-brand-gold border border-brand-gold/30"
                      : "text-white/40 hover:text-white/60 border border-transparent"
                  }`}>
                  <span className="block">{lang === "zh" ? "Pro 深度分析" : "Pro Analysis"}</span>
                  <span className="block text-[10px] font-normal text-white/30 mt-0.5">
                    {lang === "zh" ? "骨架 + AI + 弹道" : "Skeleton + AI + Trajectory"}
                  </span>
                </button>
              </div>
            </div>

            {/* Plus Analysis Link */}
            <div className="mx-auto mb-4 max-w-xl">
              <a
                href="/plus"
                className="flex items-center justify-between rounded-xl border border-purple-500/20 bg-gradient-to-r from-brand-purple/5 to-brand-gold/5 p-3 transition hover:border-purple-500/40 group"
              >
                <div className="flex items-center gap-3">
                  <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-brand-purple to-brand-gold text-[10px] font-bold text-white">
                    P+
                  </span>
                  <div>
                    <span className="block text-sm font-semibold text-white group-hover:text-brand-gold transition">
                      {lang === "zh" ? "Plus 高级诊断" : "Plus Advanced Diagnosis"}
                    </span>
                    <span className="block text-[10px] text-white/30">
                      {lang === "zh" ? "姿势评分 · 8阶段评估 · 训练建议" : "Posture score · 8-phase eval · Training plan"}
                    </span>
                  </div>
                </div>
                <svg className="h-4 w-4 text-white/30 group-hover:text-brand-gold transition" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="m8.25 4.5 7.5 7.5-7.5 7.5" />
                </svg>
              </a>
            </div>

            {/* Input Mode Tabs: Upload / Capture / Screen */}
            <div className="mx-auto mb-6 flex max-w-xl rounded-xl border border-white/10 bg-white/[0.02] p-1 overflow-hidden">
              <button onClick={() => setInputMode("upload")}
                className={`flex-1 rounded-lg py-2.5 text-xs font-semibold transition-all ${
                  inputMode === "upload"
                    ? "bg-brand-purple/20 text-white border border-brand-purple/30"
                    : "text-white/40 hover:text-white/60 border border-transparent"
                }`}>
                {lang === "zh" ? "上传视频" : "Upload"}
              </button>
              <button onClick={() => setInputMode("capture")}
                className={`flex-1 rounded-lg py-2.5 text-xs font-semibold transition-all ${
                  inputMode === "capture"
                    ? "bg-brand-purple/20 text-white border border-brand-purple/30"
                    : "text-white/40 hover:text-white/60 border border-transparent"
                }`}>
                {lang === "zh" ? "实拍模式" : "Camera"}
              </button>
              <button onClick={() => setInputMode("screen")}
                className={`flex-1 rounded-lg py-2.5 text-xs font-semibold transition-all ${
                  inputMode === "screen"
                    ? "bg-brand-purple/20 text-white border border-brand-purple/30"
                    : "text-white/40 hover:text-white/60 border border-transparent"
                }`}>
                {lang === "zh" ? "屏幕模式" : "Screen"}
              </button>
            </div>

            {inputMode === "upload" && (
              <UploadZone onUploadComplete={handleUploadComplete} lang={lang} />
            )}

            {inputMode === "capture" && (
              <div className="mx-auto max-w-xl">
                <div className="glass-card overflow-hidden">
                  <div className="relative bg-gradient-to-b from-brand-purple/10 to-transparent p-8 text-center">
                    <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-2xl bg-brand-purple/10 border border-brand-purple/20">
                      <svg className="h-10 w-10 text-brand-gold" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25h-9A2.25 2.25 0 0 0 2.25 7.5v9a2.25 2.25 0 0 0 2.25 2.25Z" />
                      </svg>
                    </div>
                    <h3 className="mb-2 text-lg font-semibold text-white">
                      {lang === "zh" ? "实拍模式" : "Live Capture"}
                    </h3>
                    <p className="mb-6 text-sm text-white/40">
                      {lang === "zh" ? "使用摄像头录制挥杆，AI骨架实时引导" : "Record swing with camera, AI skeleton overlay"}
                    </p>
                    <button
                      onClick={() => {
                        screenCaptureForProV2Ref.current = false;
                        setLiveCapture(true);
                      }}
                      className="btn-primary mx-auto"
                    >
                      {lang === "zh" ? "打开摄像头" : "Open Camera"}
                    </button>
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
                    <h3 className="mb-2 text-lg font-semibold text-white">
                      {lang === "zh" ? "屏幕模式" : "Screen Mode"}
                    </h3>
                    <p className="mb-6 text-sm text-white/40">
                      {lang === "zh"
                        ? "录制电脑屏幕上的高尔夫视频，或用摄像头对着手机/电视拍摄"
                        : "Record golf video from screen, or point camera at phone/TV"}
                    </p>
                    <div className="flex flex-col sm:flex-row gap-3 justify-center">
                      <button onClick={startScreenRecording}
                        className="btn-primary px-6 py-3 text-sm">
                        {lang === "zh" ? "📺 录制屏幕" : "📺 Record Screen"}
                      </button>
                      <button
                        onClick={() => {
                          screenCaptureForProV2Ref.current = true;
                          setLiveCapture(true);
                        }}
                        className="rounded-xl border border-white/20 bg-white/5 px-6 py-3 text-sm text-white/70 transition hover:bg-white/10"
                      >
                        {lang === "zh" ? "📷 对屏拍摄" : "📷 Point Camera"}
                      </button>
                    </div>
                    <p className="mt-4 text-[10px] text-white/20">
                      {lang === "zh"
                        ? "提示：屏幕录制适合电脑端；对屏拍摄适合用手机对着电视/电脑"
                        : "Screen recording for desktop; Camera for pointing at TV/monitor"}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {stage === "processing" && (
          <div className="relative">
            <AnalysisWaiting
              progress={progress}
              lang={lang}
              mode={analysisMode}
              proV2ScreenMode={processingProScreenMode}
            />

            {/* Handedness confirmation popup — only when club was detected */}
            {showHandPopup && detectedHand && !handConfirmed && processingClub?.club_type !== "UNKNOWN" && (
              <div className="fixed inset-0 z-[110] flex items-center justify-center bg-black/60 backdrop-blur-sm animate-fade-in">
                <div className="mx-4 w-full max-w-sm rounded-2xl border border-brand-gold/30 bg-brand-dark/95 backdrop-blur-xl p-6 shadow-2xl">
                  <p className="mb-1 text-center text-lg font-bold text-white">
                    {lang === "zh" ? "检测到打球方式" : "Detected Swing Hand"}
                  </p>
                  <p className="mb-5 text-center text-sm text-white/50">
                    {lang === "zh" ? "请确认以提高分析精度" : "Please confirm for better accuracy"}
                  </p>
                  <div className="flex gap-3 mb-4">
                    <button
                      onClick={() => { setDetectedHand("R"); handRef.current = "R"; }}
                      className={`flex-1 rounded-xl border-2 py-4 text-center font-bold transition-all ${detectedHand === "R"
                        ? "border-brand-gold bg-brand-gold/15 text-brand-gold"
                        : "border-white/10 bg-white/5 text-white/40 hover:border-white/30"}`}
                    >
                      <span className="block text-2xl mb-1">🫱</span>
                      {lang === "zh" ? "右手打球" : "Right-handed"}
                    </button>
                    <button
                      onClick={() => { setDetectedHand("L"); handRef.current = "L"; }}
                      className={`flex-1 rounded-xl border-2 py-4 text-center font-bold transition-all ${detectedHand === "L"
                        ? "border-brand-gold bg-brand-gold/15 text-brand-gold"
                        : "border-white/10 bg-white/5 text-white/40 hover:border-white/30"}`}
                    >
                      <span className="block text-2xl mb-1">🫲</span>
                      {lang === "zh" ? "左手打球" : "Left-handed"}
                    </button>
                  </div>
                  <button
                    onClick={() => { void handleHandConfirm(); }}
                    className="w-full rounded-xl bg-brand-gold py-3 text-sm font-bold text-black transition hover:bg-brand-gold/90"
                  >
                    {lang === "zh" ? "确认" : "Confirm"}
                  </button>
                </div>
              </div>
            )}

            {/* Club detection banner — only when club was actually detected */}
            {processingClub && processingClub.club_type !== "UNKNOWN" && (
              <div className="fixed bottom-6 left-4 right-4 z-50 animate-fade-in">
                <div className="mx-auto max-w-md rounded-2xl border border-brand-gold/30 bg-brand-dark/95 backdrop-blur-xl p-4 shadow-2xl">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-gold/10 text-xl">🏌️</div>
                      <div>
                        <p className="text-sm font-semibold text-white">
                          {processingClub.club_type !== "UNKNOWN"
                            ? (lang === "zh"
                              ? `检测到：${CLUB_DISPLAY[processingClub.club_type] || processingClub.club_type}`
                              : `Detected: ${processingClub.club_type}`)
                            : (lang === "zh" ? "未能识别球杆" : "Club not identified")}
                          {detectedHand && (
                            <span className="ml-2 text-brand-gold/80">
                              · {detectedHand === "R" ? (lang === "zh" ? "右手" : "R") : (lang === "zh" ? "左手" : "L")}
                            </span>
                          )}
                        </p>
                        {processingClub.club_type !== "UNKNOWN" && (
                          <p className="text-[10px] text-white/30">
                            {lang === "zh" ? "AI 置信度" : "Confidence"}: {Math.round(processingClub.confidence * 100)}%
                            {processingClub.confidence < 0.7 && (
                              <span className="ml-1 text-yellow-400">{lang === "zh" ? "· 建议确认" : "· verify"}</span>
                            )}
                          </p>
                        )}
                      </div>
                    </div>
                    <button
                      onClick={() => setShowClubPicker(true)}
                      className="rounded-lg border border-brand-gold/30 bg-brand-gold/10 px-3 py-1.5 text-xs font-medium text-brand-gold transition hover:bg-brand-gold/20"
                    >
                      {processingClub.club_type !== "UNKNOWN"
                        ? (lang === "zh" ? "修改 ▾" : "Change ▾")
                        : (lang === "zh" ? "选择 ▾" : "Select ▾")}
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Club Picker overlay during processing */}
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
                        <p className="mb-1.5 text-[10px] font-semibold text-white/40 uppercase tracking-wider">
                          {lang === "zh" ? group.label_zh : group.label_en}
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                          {group.clubs.map((club) => (
                            <button
                              key={club}
                              onClick={() => handleProcessingClubChange(club)}
                              className={`rounded-lg border px-3 py-2 text-xs font-medium transition-all ${
                                processingClub?.club_type === club
                                  ? "border-brand-gold/50 bg-brand-gold/20 text-brand-gold"
                                  : "border-white/10 bg-white/5 text-white/60 hover:border-brand-gold/30 hover:text-white"
                              }`}
                            >
                              {club}
                            </button>
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
          <div className="space-y-6 animate-fade-in">
            {/* What AI sees */}
            {result.what_i_see_zh && (
              <div className="glass-card p-4">
                <p className="text-xs text-white/40 mb-1">{lang === "zh" ? "AI 看到的内容：" : "AI detected:"}</p>
                <p className="text-sm text-white/70">
                  {lang === "zh" ? result.what_i_see_zh : result.what_i_see}
                </p>
              </div>
            )}

            {/* Score Overview */}
            <div className="glass-card p-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-white">
                    {lang === "en" ? "Analysis Results" : "分析结果"}
                  </h2>
                  <span className={`mt-1 inline-block rounded px-2 py-0.5 text-[10px] font-bold ${
                    analysisMode === "pro"
                      ? "bg-brand-gold/20 text-brand-gold"
                      : "bg-brand-purple/20 text-brand-purple"
                  }`}>
                    {analysisMode === "pro" ? "PRO" : "LITE"}
                  </span>
                </div>
                <div className="relative flex h-20 w-20 items-center justify-center rounded-full"
                  style={{ background: `conic-gradient(${analysisMode === "pro" ? "#d4af37" : "#7c3aed"} ${result.total_score}%, rgba(255,255,255,0.1) ${result.total_score}%)` } as React.CSSProperties}>
                  <div className="flex h-16 w-16 items-center justify-center rounded-full bg-brand-dark">
                    <span className={`text-xl font-bold ${analysisMode === "pro" ? "text-brand-gold" : "text-brand-purple"}`}>{result.total_score}</span>
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-5 gap-3">
                {Object.entries(result.scores).map(([key, value]) => {
                  const labels: Record<string, { en: string; zh: string }> = {
                    grip: { en: "Grip", zh: "握杆" },
                    stance: { en: "Stance", zh: "站姿" },
                    backswing: { en: "Backswing", zh: "后摆" },
                    downswing: { en: "Downswing", zh: "下杆" },
                    follow_through: { en: "Follow", zh: "收杆" },
                  };
                  const label = labels[key] || { en: key, zh: key };
                  const proColor = value >= 80 ? "#d4af37" : value >= 60 ? "#f59e0b" : "#ff5252";
                  const liteColor = value >= 80 ? "#7c3aed" : value >= 60 ? "#a78bfa" : "#ff5252";
                  const color = analysisMode === "pro" ? proColor : liteColor;
                  return (
                    <div key={key} className="text-center">
                      <div className="relative mx-auto mb-2 h-12 w-12">
                        <svg className="-rotate-90" viewBox="0 0 56 56">
                          <circle cx="28" cy="28" r="24" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="4" />
                          <circle cx="28" cy="28" r="24" fill="none" stroke={color} strokeWidth="4"
                            strokeDasharray={`${value * 1.508} 150.8`} strokeLinecap="round" />
                        </svg>
                        <span className="absolute inset-0 flex items-center justify-center text-xs font-bold" style={{ color }}>{value}</span>
                      </div>
                      <p className="text-[10px] font-medium text-white/60">{lang === "en" ? label.en : label.zh}</p>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Club Detection Banner + Picker — only when a club was detected */}
            {result.prediction?.club_type && result.prediction.club_type !== "UNKNOWN" && (<>
            <div className="glass-card flex items-center justify-between p-4">
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-gold/10 text-lg">
                  🏌️
                </div>
                <div>
                  <p className="text-sm font-semibold text-white">
                    {lang === "zh"
                      ? `检测到：${CLUB_DISPLAY[result.prediction.club_type] || result.prediction.club_type}`
                      : `Detected: ${result.prediction.club_type}`}
                    {result.prediction.hand && result.prediction.hand !== "UNKNOWN" && (
                      <span className="ml-2 text-brand-gold/80">
                        · {result.prediction.hand === "R" ? (lang === "zh" ? "右手" : "R") : (lang === "zh" ? "左手" : "L")}
                      </span>
                    )}
                  </p>
                  {result.prediction.club_detection_confidence != null && (
                    <p className="text-[10px] text-white/30">
                      {lang === "zh" ? "AI 置信度" : "AI confidence"}: {Math.round(result.prediction.club_detection_confidence * 100)}%
                      {result.prediction.club_detection_confidence < 0.7 && (
                        <span className="ml-1 text-yellow-400">{lang === "zh" ? "· 建议手动确认" : "· please verify"}</span>
                      )}
                    </p>
                  )}
                </div>
              </div>
              <button
                onClick={() => setShowClubPicker(true)}
                className="rounded-lg border border-brand-gold/30 bg-brand-gold/10 px-3 py-1.5 text-xs font-medium text-brand-gold transition hover:bg-brand-gold/20"
              >
                {lang === "zh" ? "修改球杆 ▾" : "Change ▾"}
              </button>
            </div>

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
                        <p className="mb-1.5 text-[10px] font-semibold text-white/40 uppercase tracking-wider">
                          {lang === "zh" ? group.label_zh : group.label_en}
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                          {group.clubs.map((club) => (
                            <button
                              key={club}
                              onClick={() => handleClubOverride(club)}
                              className={`rounded-lg border px-3 py-2 text-xs font-medium transition-all ${
                                result?.prediction.club_type === club
                                  ? "border-brand-gold/50 bg-brand-gold/20 text-brand-gold"
                                  : "border-white/10 bg-white/5 text-white/60 hover:border-brand-gold/30 hover:text-white"
                              }`}
                            >
                              {club}
                            </button>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
            </>)}

            {/* Speed Data Card — only when prediction has data */}
            {result.prediction && (result.prediction.ball_speed > 0 || result.prediction.fused_speed) && (
              <div className="glass-card p-5">
                <h3 className="mb-4 text-base font-semibold text-white">{lang === "zh" ? "发球数据" : "Shot Data"}</h3>
                <div className="grid grid-cols-3 gap-3 mb-4">
                  <div className="rounded-xl border border-white/5 bg-black/30 p-3 text-center" style={{ transition: "all 0.4s ease" }}>
                    <p className="text-[10px] text-white/40">{lang === "zh" ? "球速" : "Ball Speed"}</p>
                    <p className="mt-1 text-xl font-bold text-brand-gold" style={{ transition: "all 0.4s ease" }}>
                      {result.prediction.fused_speed || result.prediction.ball_speed}
                    </p>
                    <p className="text-[10px] text-white/30">mph</p>
                  </div>
                  <div className="rounded-xl border border-white/5 bg-black/30 p-3 text-center" style={{ transition: "all 0.4s ease" }}>
                    <p className="text-[10px] text-white/40">{lang === "zh" ? "杆头速度" : "Club Speed"}</p>
                    <p className="mt-1 text-xl font-bold text-white" style={{ transition: "all 0.4s ease" }}>
                      {result.prediction.club_head_speed}
                    </p>
                    <p className="text-[10px] text-white/30">mph</p>
                  </div>
                  <div className="rounded-xl border border-white/5 bg-black/30 p-3 text-center" style={{ transition: "all 0.4s ease" }}>
                    <p className="text-[10px] text-white/40">{lang === "zh" ? "预测距离" : "Est. Distance"}</p>
                    <p className="mt-1 text-xl font-bold text-white" style={{ transition: "all 0.4s ease" }}>
                      {result.prediction.predicted_distance}
                    </p>
                    <p className="text-[10px] text-white/30">{lang === "zh" ? "码" : "yards"}</p>
                  </div>
                </div>
                <div className="space-y-2 rounded-xl border border-white/5 bg-black/20 p-3">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-white/40">{lang === "zh" ? "数据来源" : "Source"}</span>
                    <span className="text-white/60">
                      {result.prediction.fused_speed
                        ? "Steller Vision Engine"
                        : analysisMode === "pro" ? "Steller AI Pro" : "Steller AI Lite"}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-white/40">{lang === "zh" ? "置信度" : "Confidence"}</span>
                    <div className="flex items-center gap-2">
                      <div className="flex h-2 w-20 overflow-hidden rounded-full bg-white/10">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            (result.prediction.speed_confidence || "low") === "high"
                              ? "bg-green-500"
                              : (result.prediction.speed_confidence || "low") === "medium"
                              ? "bg-yellow-500"
                              : "bg-red-400"
                          }`}
                          style={{
                            width: (result.prediction.speed_confidence || "low") === "high" ? "90%" : (result.prediction.speed_confidence || "low") === "medium" ? "60%" : "30%",
                          }}
                        />
                      </div>
                      <span className="text-white/60">
                        {(result.prediction.speed_confidence || "low") === "high"
                          ? (lang === "zh" ? "高" : "High")
                          : (result.prediction.speed_confidence || "low") === "medium"
                          ? (lang === "zh" ? "中等" : "Medium")
                          : (lang === "zh" ? "低" : "Low")}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-white/40">{lang === "zh" ? "误差估计" : "Error Est."}</span>
                    <span className="text-white/60">±{result.prediction.error_estimate_pct ?? (analysisMode === "pro" ? 8 : 15)}%</span>
                  </div>
                </div>
              </div>
            )}

            {/* Summary */}
            <div className="glass-card p-5">
              <h3 className="mb-2 text-base font-semibold text-white">{lang === "en" ? "Summary" : "分析总结"}</h3>
              <p className="text-sm leading-relaxed text-white/60">
                {lang === "en" ? result.summary : result.summary_zh}
              </p>
            </div>

            {/* Tabs */}
            <div className="flex gap-1 rounded-xl bg-white/5 p-1 overflow-hidden">
              {(["analysis", "3d", "comparison"] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`flex-1 rounded-lg py-2.5 text-sm font-semibold transition-all ${
                    activeTab === tab
                      ? analysisMode === "pro"
                        ? "bg-brand-gold text-black"
                        : "bg-brand-purple text-white"
                      : "text-white/40 hover:text-white/60"
                  }`}
                >
                  {tab === "analysis"
                    ? lang === "en" ? "Details" : "详细分析"
                    : tab === "3d"
                    ? "3D"
                    : lang === "en" ? "Pro Compare" : "职业对比"}
                </button>
              ))}
            </div>

            {activeTab === "analysis" && (
              <>
                {result.keyframes && result.keyframes.length > 0 && (
                  <KeyframeStrip keyframes={result.keyframes} lang={lang} />
                )}

                {result.skeleton_data && result.skeleton_data.frames.length > 0 && (
                  <div className="glass-card p-5">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-base font-semibold text-white">{lang === "en" ? "Skeleton HUD" : "骨架 HUD"}</h3>
                      <button onClick={() => setShowExtendedHUD(!showExtendedHUD)}
                        className="rounded-lg border border-brand-purple/20 bg-brand-purple/5 px-3 py-1 text-xs text-white/50">
                        {showExtendedHUD ? (lang === "zh" ? "收起" : "Less") : (lang === "zh" ? "查看全部" : "All")}
                      </button>
                    </div>
                    <HUDOverlay hudData={result.skeleton_data.frames[0] as Record<string, unknown>} showExtended={showExtendedHUD} mode={analysisMode === "pro" ? "pro" : "lite"} lang={lang} />
                  </div>
                )}

                <div className="grid gap-4 md:grid-cols-2">
                  <div className="glass-card p-5">
                    <h3 className="mb-3 text-base font-semibold text-red-400/80">{lang === "en" ? "Issues" : "发现的问题"}</h3>
                    <ul className="space-y-2.5">
                      {(lang === "en" ? result.issues : result.issues_zh).map((issue, i) => (
                        <li key={i} className="flex items-start gap-2 text-sm text-white/60">
                          <span className="mt-0.5 text-red-400/60 text-xs">●</span>{issue}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div className="glass-card p-5">
                    <h3 className="mb-3 text-base font-semibold text-brand-gold/80">{lang === "en" ? "Suggestions" : "改进建议"}</h3>
                    <ul className="space-y-2.5">
                      {(lang === "en" ? result.suggestions : result.suggestions_zh).map((sug, i) => (
                        <li key={i} className="flex items-start gap-2 text-sm text-white/60">
                          <span className="mt-0.5 text-brand-gold/60 text-xs">◆</span>{sug}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                {result.prediction && result.prediction.predicted_distance > 0 && (
                  <SimAnimation
                    prediction={result.prediction}
                    lang={lang}
                  />
                )}
              </>
            )}

            {activeTab === "3d" && result.pose_frames && result.pose_frames.length > 0 && (
              <div className="space-y-6">
                <Skeleton3DViewer
                  frames={result.pose_frames}
                  lang={lang}
                />

                <div className="glass-card p-5">
                  <h3 className="mb-4 text-sm font-semibold text-white">
                    {lang === "zh" ? "3D 运动数据" : "3D Motion Data"}
                  </h3>
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                    {Object.entries(result.pose_frames[0]?.angles || {}).map(([key, val]) => {
                      const labels: Record<string, { en: string; zh: string }> = {
                        left_elbow: { en: "L.Elbow", zh: "左肘" },
                        right_elbow: { en: "R.Elbow", zh: "右肘" },
                        left_knee: { en: "L.Knee", zh: "左膝" },
                        right_knee: { en: "R.Knee", zh: "右膝" },
                        left_shoulder: { en: "L.Shoulder", zh: "左肩" },
                        right_shoulder: { en: "R.Shoulder", zh: "右肩" },
                        shoulder_rotation: { en: "Shoulder Rot.", zh: "肩旋转" },
                        hip_rotation: { en: "Hip Rot.", zh: "髋旋转" },
                        x_factor: { en: "X-Factor", zh: "X因子" },
                        spine_tilt: { en: "Spine Tilt", zh: "脊柱倾斜" },
                      };
                      const l = labels[key] || { en: key, zh: key };
                      return (
                        <div key={key} className="rounded-xl border border-white/5 bg-black/30 p-3 text-center">
                          <p className="text-[10px] text-white/40">{lang === "zh" ? l.zh : l.en}</p>
                          <p className="mt-1 text-lg font-bold text-brand-purple">{typeof val === "number" ? val.toFixed(1) : val}°</p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {activeTab === "3d" && (!result.pose_frames || result.pose_frames.length === 0) && (
              <div className="glass-card p-8 text-center">
                <p className="text-white/40 text-sm">
                  {lang === "zh" ? "未检测到骨架数据，请上传包含人物的高尔夫挥杆视频" : "No skeleton data detected. Please upload a golf swing video with a visible person."}
                </p>
              </div>
            )}

            {activeTab === "comparison" && (
              <ProComparison
                userScores={result.scores}
                userAngles={result.pose_frames?.[0]?.angles
                  ?? { shoulder_rotation: -35.2, hip_rotation: -22.1, x_factor: 42.5, spine_tilt: 8.3 }}
                lang={lang}
              />
            )}

            <div className="text-center pb-4">
              <button onClick={() => { setProcessingProScreenMode(false); setStage("upload"); setResult(null); setError(""); setActiveTab("analysis"); }}
                className="btn-primary">
                {lang === "en" ? "Analyze Again" : "再次分析"}
              </button>
            </div>
            <p className="text-center text-[9px] text-white/10 pb-2">v2.2</p>
          </div>
        )}
      </div>
    </div>
  );
}
