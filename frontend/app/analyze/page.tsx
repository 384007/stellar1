"use client";

import { useState, useCallback, useEffect, useRef, useMemo } from "react";
import { useRouter } from "next/navigation";
import UploadZone from "@/components/UploadZone";
import HUDOverlay from "@/components/HUDOverlay";
import KeyframeStrip from "@/components/KeyframeStrip";
import Prov3PlusVideoRenderer from "@/components/prov3/Prov3PlusVideoRenderer";
import Prov3MotionEvidenceReport from "@/components/prov3/Prov3MotionEvidenceReport";
import FrontErrorBoundary from "@/components/debug/FrontErrorBoundary";
import type { PlusAnalysisResult } from "@/components/PlusResultView";
import SimAnimation from "@/components/SimAnimation";
import ProComparison from "@/components/ProComparison";
import Skeleton3DViewer from "@/components/Skeleton3DViewer";
import ScreenModeCapture from "@/components/ScreenModeCapture";
import { preloadPoseModel } from "@/lib/mediapipe-assets";
import AnalysisWaiting from "@/components/AnalysisWaiting";
import ClubHandSummaryBar from "@/components/ClubHandSummaryBar";
import { saveAnalysisVideo } from "@/lib/video-store";
import { makeFormData } from "@/lib/fetch-retry";
import { isVideoFile } from "@/lib/upload-video";
import { stripResultForStorage } from "@/lib/strip-result";
import { normalizedTotalScoreForStorage } from "@/lib/safe-analysis-score";
import { patchLocalHistoryVideoR2Key } from "@/lib/history-sync-record";
import { expandStellarProForUi, stellarProTrustIsLow } from "@/lib/stellar-pro-result";
import { displayKeyframesForResult } from "@/lib/analysis-display-keyframes";
import { isProv3StrictMediaPolicyResult, type Prov3ResultLike } from "@/lib/prov3-keyframe-media";
import { resolveProv3ProductMediaUrl } from "@/lib/prov3-media-url";
import { clientLikelyMainlandChinaUser } from "@/lib/client-region-hint";
import { devLog, devWarn } from "@/lib/dev-only-log";
import { LITE_ANALYZE_FETCH_TIMEOUT_MS } from "@/lib/lite-analyze-timeout";
import { readLiteAnalyzeResult } from "@/lib/read-lite-analyze-response";
import { pruneLocalStellarHistoryRecords } from "@/lib/pro-history-retention";
import {
  requestProv3AnalyzeCancel,
  runProv3AnalyzeMultipart,
  yieldUiBeforeHeavyParse,
} from "@/lib/pro-v3-api";
import {
  isProv3ScreenWebmForMp4Upload,
  prov3ScreenRecordingToMp4File,
} from "@/lib/prov3-screen-recording-to-mp4";
import {
  consumeReanalyzeFromHistoryPayload,
  fetchVideoBlobForHistoryReanalyze,
  reanalyzeHistoryFilename,
  reanalyzePayloadProv3ScreenMode,
} from "@/lib/reanalyze-from-history";

/** Upstream may return ``detail`` as string, object, or validation error array. */
function stringifyFastApiDetail(detail: unknown): string {
  if (detail == null) return "";
  if (typeof detail === "string") return detail.trim();
  if (Array.isArray(detail)) {
    const parts = detail.map((item) => {
      if (item && typeof item === "object" && "msg" in item) {
        const m = (item as { msg?: unknown }).msg;
        return typeof m === "string" ? m : String(m ?? "");
      }
      return typeof item === "string" ? item : "";
    });
    return parts.filter(Boolean).join(" ").trim();
  }
  if (typeof detail === "object" && detail !== null && "msg" in detail) {
    const m = (detail as { msg?: unknown }).msg;
    return typeof m === "string" ? m.trim() : String(m ?? "").trim();
  }
  return "";
}

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
    image_base64?: string;
    keyframe_image_url?: string;
  }>;
  /** Pro v3 低信任：条图在 ``preview_keyframes``，顶层 ``keyframes`` 可能被清空 */
  preview_keyframes?: AnalysisResult["keyframes"];
  official_phase_keyframes?: AnalysisResult["keyframes"];
  pipeline?: string;
  /** OpenCV / 时间线 scrubber 与 pose frame_index 对齐（与 ``Prov3PlusVideoRenderer`` 一致） */
  video_meta?: { source_frame_count?: number; fps?: number; duration_s?: number };
  /** 与 ``/pro`` PlusResultView / ``displayKeyframesForResult`` 对齐，用于低高信任选条 */
  final_status?: string;
  analysis_trust?: string;
  trust_level?: string;
  low_trust_preview_only?: boolean;
  analysis_video_url?: string;
  playback_video_url?: string;
  video_url?: string;
  original_video_url?: string;
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

function proTimelineVideoUrlForAnalyze(r: AnalysisResult): string | null {
  const raw = String(
    r.analysis_video_url || r.playback_video_url || r.video_url || r.original_video_url || "",
  ).trim();
  const u = resolveProv3ProductMediaUrl(raw);
  return /^https?:\/\//i.test(u) ? u : null;
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
  /** 默认 Lite：避免未切换 Tab 就上传时仍走 Pro 预检；与 ref 同步防竞态。 */
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>("lite");
  const analysisModeRef = useRef<AnalysisMode>("lite");
  const [username, setUsername] = useState("");
  const [authChecked, setAuthChecked] = useState(false);
  const [screenRecording, setScreenRecording] = useState(false);
  const screenRecRef = useRef<MediaRecorder | null>(null);
  const screenStreamRef = useRef<MediaStream | null>(null);
  const screenChunksRef = useRef<Blob[]>([]);
  const [screenRecTime, setScreenRecTime] = useState(0);
  const screenTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [showClubPicker, setShowClubPicker] = useState(false);
  const stripKeyframesForResult = useMemo(
    () =>
      (result
        ? (displayKeyframesForResult(result as unknown as Prov3ResultLike) as AnalysisResult["keyframes"])
        : []),
    [result],
  );
  const proVideoTimelineUrl = useMemo(
    () => (result ? proTimelineVideoUrlForAnalyze(result) : null),
    [result],
  );
  const analyzePageIsProv3Product = useMemo(
    () => Boolean(result && isProv3StrictMediaPolicyResult(result as Prov3ResultLike)),
    [result],
  );
  const lastBlobRef = useRef<{ blob: Blob; filename: string } | null>(null);
  /** Pro v3 对屏路径：在打开实拍前标记来源为拍屏 tab。 */
  const screenCaptureForProv3Ref = useRef(false);
  const [processingClub, setProcessingClub] = useState<ClubDetection | null>(null);
  const processingClubRef = useRef<ClubDetection | null>(null);
  const [detectedHand, setDetectedHand] = useState<"R" | "L" | null>(null);
  const [handConfirmed, setHandConfirmed] = useState(false);
  const [showHandPopup, setShowHandPopup] = useState(false);
  const handRef = useRef<"R" | "L">("R");
  /** User confirmed L/R during processing before ``result`` exists — apply to returned prediction. */
  const handLockedDuringProcessingRef = useRef(false);
  const [processingProScreenMode, setProcessingProScreenMode] = useState(false);
  /** 防止重复提交 Pro / Lite 分析流程。 */
  const analysisInFlightRef = useRef(false);
  /** Lite：单次分析固定 idempotency key（整段 processBlob 不变）。 */
  const liteIdempotencyKeyRef = useRef<string | null>(null);
  /** Lite：同一次分析最多发 1 次 POST /analyze/lite（禁止隐式重试）。 */
  const liteAnalyzeHttpIssuedRef = useRef(false);
  const proAnalyzeAbortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    analysisModeRef.current = analysisMode;
  }, [analysisMode]);

  const stopProAnalysis = useCallback(async () => {
    const token = localStorage.getItem("stellar_token");
    const authHeaders: Record<string, string> = {};
    if (token && token.includes(".")) authHeaders["Authorization"] = `Bearer ${token}`;
    await requestProv3AnalyzeCancel(authHeaders);
    proAnalyzeAbortRef.current?.abort();
  }, []);

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

  const litePredictionView = useMemo(() => {
    const pred = result?.prediction;
    const clubType = pred?.club_type || "UNKNOWN";
    const hand = pred?.hand || "UNKNOWN";
    const handConfidence = typeof pred?.hand_confidence === "number" ? pred.hand_confidence : 0;
    const lowHandConfidence = hand !== "R" && hand !== "L" ? true : handConfidence < 0.6;
    return { clubType, hand, handConfidence, lowHandConfidence };
  }, [result]);


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
      try {
        if (analysisInFlightRef.current) {
          devWarn("[analyze] reanalyze skipped while analysis already in flight");
          return;
        }
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
        const mode: AnalysisMode = p.analysisMode === "pro" ? "pro" : "lite";
        const screenTag = reanalyzePayloadProv3ScreenMode(p);
        if (screenTag) setInputMode("screen");
        await processBlob(blob, reanalyzeHistoryFilename(blob), screenTag, mode);
      } catch (e) {
        devWarn("[analyze] reanalyze pipeline error:", e);
      }
    })();
    // 仅登录就绪时消费一次；勿依赖 lang，避免切换语言重复跑 effect。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authChecked]);

  function resolveProv3ScreenMode(
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
    prov3ScreenMode?: boolean,
    modeOverride?: AnalysisMode,
    proAbortSignal?: AbortSignal,
    liteIdempotencyKey?: string | null,
  ): Promise<AnalysisResult> {
    const isPro = (modeOverride ?? analysisModeRef.current) === "pro";

    if (isPro) {
      // Pro mode: precheck (Edge) → same-origin upload + ``/api/prov3/analyze/start`` → poll job from R2
      // (no long browser→Modal ``POST /pro-v3/analyze``; Modal runs the job in the background).
      const token = localStorage.getItem("stellar_token");
      const authHeaders: Record<string, string> = {};
      if (token && token.includes(".")) authHeaders["Authorization"] = `Bearer ${token}`;

      // Pro v3：网络提示与 Lite 同源（CF / 客户端语言回退）；Modal/Render 基址仅在 Edge 解析。
      let cnPro = false;
      try {
        const nh = await fetch("/api/lite/network-hint", { method: "GET" });
        if (nh.ok) {
          const j = (await nh.json()) as { network_hint?: string; lite_geo?: string };
          if (j.network_hint === "cn" || j.lite_geo === "cn") cnPro = true;
          else if (j.lite_geo === "unknown" && clientLikelyMainlandChinaUser()) cnPro = true;
        } else if (clientLikelyMainlandChinaUser()) {
          cnPro = true;
        }
      } catch {
        if (clientLikelyMainlandChinaUser()) cnPro = true;
      }

      const mb = (file.size / 1024 / 1024).toFixed(1);
      const screenMode = resolveProv3ScreenMode(filename, prov3ScreenMode);
      const { raw: rawPro, route: proServedBy } = await runProv3AnalyzeMultipart(
        file as Blob,
        filename,
        authHeaders,
        {
          cnNetworkHint: cnPro,
          unboundedJobPoll: cnPro || clientLikelyMainlandChinaUser(),
          screenMode,
          modalTimeoutMs: 600_000,
          renderTimeoutMs: 600_000,
          logPrefix: `[stellar-pro] ${mb}MB`,
          abortSignal: proAbortSignal,
          userCancelledMessage: lang === "zh" ? "分析已停止" : "Analysis stopped",
        },
      );

      devLog(`[stellar-pro] Pro job completed (route=${proServedBy})`);
      await yieldUiBeforeHeavyParse();
      return expandStellarProForUi(rawPro) as AnalysisResult;
    }

    // Lite: always same-origin proxy → Edge resolves Modal/Lite upstream (no browser-direct analyze host).
    const rid = (liteIdempotencyKey || "").trim();
    if (!rid) {
      throw new Error(
        lang === "zh" ? "分析请求标识缺失，请重试" : "Missing analysis request id. Please try again.",
      );
    }
    if (liteAnalyzeHttpIssuedRef.current) {
      throw new Error(
        lang === "zh" ? "分析已在进行中" : "Analysis already in progress.",
      );
    }

    const headers: Record<string, string> = { "X-Stellar-Idempotency-Key": rid };
    const token = localStorage.getItem("stellar_token");
    if (token && token.includes(".")) headers.Authorization = `Bearer ${token}`;

    const liteAnalyzeUrl = "/api/lite/analyze-proxy";
    let res: Response;
    liteAnalyzeHttpIssuedRef.current = true;
    try {
      if (isVideoFile(file as File, filename)) {
        const fd = new FormData();
        fd.append("file", file as File, filename);
        fd.append("request_id", rid);
        const analyzeCtrl = new AbortController();
        const analyzeTimer = setTimeout(() => analyzeCtrl.abort(), LITE_ANALYZE_FETCH_TIMEOUT_MS);
        try {
          res = await fetch(liteAnalyzeUrl, {
            method: "POST",
            headers,
            body: fd,
            signal: analyzeCtrl.signal,
          });
        } finally {
          clearTimeout(analyzeTimer);
        }
      } else {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), LITE_ANALYZE_FETCH_TIMEOUT_MS);
        try {
          const fd = makeFormData(file as Blob, filename);
          fd.append("request_id", rid);
          res = await fetch(liteAnalyzeUrl, {
            method: "POST",
            headers,
            body: fd,
            signal: controller.signal,
          });
        } finally {
          clearTimeout(timer);
        }
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        throw new Error(
          lang === "zh"
            ? "分析等待超时。若网络较慢，服务端可能仍在处理，请稍后从历史记录查看是否已完成。"
            : "This request timed out waiting for a response. The server may still be processing—check History shortly or try again.",
        );
      }
      throw new Error(
        lang === "zh" ? "当前无法完成分析，请稍后重试" : "Analysis could not be completed. Please try again.",
      );
    } finally {
      liteAnalyzeHttpIssuedRef.current = false;
    }

    const resCt = res.headers.get("content-type") || "";
    if (res.ok && resCt.includes("text/event-stream")) {
      try {
        return (await readLiteAnalyzeResult(res)) as unknown as AnalysisResult;
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        throw new Error(
          msg.trim() ||
            (lang === "zh" ? "分析失败，请重试" : "Analysis failed. Please try again."),
        );
      }
    }

    if (!res.ok) {
      if (res.status === 524) {
        throw new Error(
          lang === "zh"
            ? "分析用时较长，连接已中断。请缩短视频或稍后重试。"
            : "The connection timed out while analysis was running. Try a shorter video or try again later.",
        );
      }
      if (res.status === 522 || res.status === 523) {
        throw new Error(
          lang === "zh"
            ? "暂时无法连接到分析服务，请稍后重试或检查网络。"
            : "Could not reach the analysis service. Retry or check your network.",
        );
      }
      let errJson: unknown;
      try {
        errJson = await res.json();
      } catch {
        errJson = null;
      }
      const code =
        errJson && typeof errJson === "object" && errJson !== null && "code" in errJson
          ? String((errJson as { code?: string }).code || "")
          : "";
      const detailRaw =
        errJson && typeof errJson === "object" && errJson !== null && "detail" in errJson
          ? (errJson as { detail?: unknown }).detail
          : undefined;
      let detailStr = stringifyFastApiDetail(detailRaw);
      if (!detailStr && res.status >= 400) {
        const st = (res.statusText || "").trim();
        detailStr =
          lang === "zh"
            ? st
              ? `服务返回异常（${res.status}）。请稍后重试。`
              : `服务返回异常（${res.status}）。请稍后重试。`
            : st
              ? `Something went wrong (${res.status}). Please try again.`
              : `Something went wrong (${res.status}). Please try again.`;
      }

      if (res.status === 409 && code === "LITE_ANALYZE_ALREADY_RUNNING") {
        throw new Error(
          lang === "zh" ? "已有分析正在进行，请稍候再试" : "An analysis is already running. Please wait.",
        );
      }
      if (res.status === 400 && code === "LITE_IDEMPOTENCY_KEY_REQUIRED") {
        throw new Error(
          lang === "zh" ? "缺少请求标识，请刷新页面后重试" : "Missing request id. Refresh and try again.",
        );
      }
      if (res.status === 400 && code === "LITE_IDEMPOTENCY_MISMATCH") {
        throw new Error(
          lang === "zh" ? "请求标识不一致，请刷新后重试" : "Request id mismatch. Refresh and try again.",
        );
      }
      if (detailStr) {
        throw new Error(detailStr);
      }
      throw new Error(lang === "zh" ? "分析失败，请重试" : "Analysis failed. Please try again.");
    }
    return (await readLiteAnalyzeResult(res)) as unknown as AnalysisResult;
  }

  const RECALCULATE_TIMEOUT_MS = 10_000;

  async function recalculatePredictionFromBackend(
    data: AnalysisResult,
    overrides?: { club_type?: string; club_group?: string; hand?: "R" | "L" | "UNKNOWN"; hand_confidence?: number; preferred_ball_speed?: number },
  ): Promise<AnalysisResult["prediction"] | null> {
    try {
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
        res = await fetch("/api/analyze/recalculate", {
          method: "POST",
          headers,
          body: JSON.stringify(body),
          signal: ctrl.signal,
        });
      } catch (e) {
        if ((e as Error)?.name !== "AbortError") {
          devWarn("[analyze] recalculate request failed");
        }
        return null;
      } finally {
        clearTimeout(t);
      }
      if (!res.ok) {
        return null;
      }
      const payload = await res.json();
      return payload?.prediction ?? null;
    } catch {
      devWarn("[analyze] recalculate failed");
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
        devWarn("[history] server save failed:", res.status);
      }
    } catch (e) {
      devWarn("[history] server save error:", e);
    }
  }

  async function processBlob(
    blob: Blob,
    filename: string,
    prov3ScreenMode?: boolean,
    analysisModeOverride?: AnalysisMode,
  ) {
    if (analysisInFlightRef.current) {
      devWarn("[analyze] analyze already in flight, ignoring duplicate trigger");
      return;
    }
    analysisInFlightRef.current = true;
    try {
    const modeForRun = analysisModeOverride ?? analysisModeRef.current;
    if (modeForRun === "lite") {
      liteIdempotencyKeyRef.current =
        typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
          ? crypto.randomUUID()
          : `${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
      liteAnalyzeHttpIssuedRef.current = false;
    } else {
      liteIdempotencyKeyRef.current = null;
    }
    if (analysisModeOverride) setAnalysisMode(analysisModeOverride);
    setProcessingProScreenMode(
      modeForRun === "pro" && resolveProv3ScreenMode(filename, prov3ScreenMode),
    );
    setStage("processing");
    setError("");
    setProgress(0);
    setProcessingClub(null);
    processingClubRef.current = null;
    setDetectedHand(null);
    setHandConfirmed(false);
    handLockedDuringProcessingRef.current = false;
    setShowHandPopup(false);

    let sendBlob: Blob = blob;
    let sendFilename = filename;
    if (
      modeForRun === "pro" &&
      isProv3ScreenWebmForMp4Upload(blob, filename, prov3ScreenMode, resolveProv3ScreenMode)
    ) {
      try {
        const mp4File = await prov3ScreenRecordingToMp4File(blob);
        sendBlob = mp4File;
        sendFilename = mp4File.name;
      } catch (e) {
        devWarn("[analyze] screen WebM→MP4 failed, uploading original container", e);
      }
    }
    lastBlobRef.current = { blob: sendBlob, filename: sendFilename };
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

    const proAbort = modeForRun === "pro" ? new AbortController() : null;
    proAnalyzeAbortRef.current = proAbort;

    try {
      const data = await sendFileForAnalysis(
        sendBlob,
        sendFilename,
        prov3ScreenMode,
        modeForRun,
        proAbort?.signal,
        modeForRun === "lite" ? liteIdempotencyKeyRef.current : null,
      );
      clearInterval(progressInterval);

      /** 主分析 prediction（杆型由服务端与主流程同次推理合并）。 */
      if (data.prediction) {
        const p = data.prediction;
        const cd: ClubDetection = {
          club_type: typeof p.club_type === "string" && p.club_type ? p.club_type : "UNKNOWN",
          club_group: typeof p.club_group === "string" && p.club_group ? p.club_group : "IRON",
          confidence: typeof p.club_detection_confidence === "number" ? p.club_detection_confidence : 0,
          hand: p.hand === "L" || p.hand === "R" ? p.hand : undefined,
        };
        setProcessingClub(cd);
        processingClubRef.current = cd;
        if (p.hand === "R" || p.hand === "L") {
          setDetectedHand(p.hand);
          handRef.current = p.hand;
          setShowHandPopup(Boolean((p.hand_confidence ?? 0) < 0.6));
        } else {
          setDetectedHand("R");
          setShowHandPopup(true);
        }
      } else {
        const fallbackClub: ClubDetection = {
          club_type: "UNKNOWN",
          club_group: "IRON",
          confidence: 0,
          hand: undefined,
        };
        setProcessingClub(fallbackClub);
        processingClubRef.current = fallbackClub;
        setDetectedHand("R");
        setShowHandPopup(true);
      }

      if (
        handLockedDuringProcessingRef.current &&
        data.prediction &&
        (handRef.current === "R" || handRef.current === "L")
      ) {
        data.prediction.hand = handRef.current;
        data.prediction.hand_confidence = 1.0;
      }

      const analysisId = data.analysis_id;
      setProgress(100);
      setResult(data);
      setStage("results");

      const poseOk = (data.pose_frames?.length ?? 0) > 0;
      const clubKnown =
        Boolean(data.prediction?.club_type && data.prediction.club_type !== "UNKNOWN");
      const shouldBackgroundRecalc =
        modeForRun === "pro" &&
        poseOk &&
        data.prediction &&
        (clubKnown || handWasConfirmed);

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
          } catch {
            devWarn("[analyze] background recalculate failed");
          }
        })();
      }

      void saveAnalysisVideo(data.analysis_id, sendBlob, sendFilename).catch(() => {});
      try {
        await saveAnalysisToHistory(data, sendBlob, sendFilename, modeForRun);
      } catch (e) {
        devWarn("[analyze] history save failed:", e);
      }
    } catch (err: unknown) {
      clearInterval(progressInterval);
      setProgress(0);
      setProcessingProScreenMode(false);
      const msg = err instanceof Error ? err.message : "";
      if (
        msg === "分析已停止" ||
        msg === "Analysis stopped" ||
        msg.includes("分析已取消")
      ) {
        setError("");
      } else {
        setError(msg || "分析失败，请重试");
      }
      setStage("upload");
    }
    } finally {
      analysisInFlightRef.current = false;
      proAnalyzeAbortRef.current = null;
    }
  }

  function handleProcessingClubChange(clubType: string) {
    setShowClubPicker(false);
    const groupMap: Record<string, string> = {};
    for (const g of CLUB_GROUPS) for (const c of g.clubs) groupMap[c] = g.id;
    const prev = processingClubRef.current;
    const newClub: ClubDetection = {
      club_type: clubType,
      club_group: groupMap[clubType] || "IRON",
      confidence: 1.0,
      hand: prev?.hand && (prev.hand === "L" || prev.hand === "R") ? prev.hand : handRef.current,
    };
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
    if (!result) {
      handLockedDuringProcessingRef.current = true;
      return;
    }
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
      screenCaptureForProv3Ref.current = false;
      processBlob(file, file.name, false);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [lang, analysisMode]
  );

  const handleVideoCapture = useCallback(
    (videoBlob: Blob) => {
      setLiveCapture(false);
      processBlob(videoBlob, "swing-capture.webm", screenCaptureForProv3Ref.current);
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
        screenCaptureForProv3Ref.current,
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
      screenCaptureForProv3Ref.current = true;
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
                <button
                  type="button"
                  onClick={() => {
                    analysisModeRef.current = "lite";
                    setAnalysisMode("lite");
                  }}
                  className={`flex-1 rounded-lg py-3 text-sm font-semibold transition-all ${
                    analysisMode === "lite"
                      ? "bg-brand-purple/20 text-white border border-brand-purple/30"
                      : "text-white/40 hover:text-white/60 border border-transparent"
                  }`}>
                  <span className="block">{lang === "zh" ? "普通分析" : "Standard"}</span>
                  <span className="block text-[10px] font-normal text-white/30 mt-0.5">Stellar AI</span>
                </button>
                <button
                  type="button"
                  onClick={() => {
                    analysisModeRef.current = "pro";
                    setAnalysisMode("pro");
                  }}
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
                        screenCaptureForProv3Ref.current = false;
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
                          screenCaptureForProv3Ref.current = true;
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
              prov3ScreenMode={processingProScreenMode}
              onCancel={analysisMode === "pro" ? stopProAnalysis : undefined}
            />

            {/* Lite：主分析 ``/analyze/lite`` 内带杆型/左右手；此处仅展示进度条，结果页再显示具体识别 */}
            {analysisMode === "lite" && (
              <div className="fixed bottom-6 left-4 right-4 z-50 animate-fade-in space-y-2">
                <ClubHandSummaryBar
                  lang={lang}
                  clubType={processingClub?.club_type}
                  clubConfidence={processingClub?.confidence}
                  hand={(processingClub?.hand as "R" | "L" | "UNKNOWN" | undefined) ?? detectedHand ?? "UNKNOWN"}
                  pending={true}
                />
                <div className="flex justify-center gap-2">
                  <button
                    type="button"
                    onClick={() => setShowClubPicker(true)}
                    className="rounded-xl border border-brand-purple/35 bg-brand-purple/15 px-4 py-2 text-xs font-medium text-brand-purple transition hover:bg-brand-purple/25"
                  >
                    {lang === "zh" ? "修改球杆" : "Change club"}
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowHandPopup(true)}
                    className="rounded-xl border border-brand-purple/35 bg-brand-purple/15 px-4 py-2 text-xs font-medium text-brand-purple transition hover:bg-brand-purple/25"
                  >
                    {lang === "zh" ? "确认左右手" : "Confirm hand"}
                  </button>
                </div>
              </div>
            )}

            {/* Pro only: live club banner during processing */}
            {analysisMode === "pro" && processingClub && processingClub.club_type !== "UNKNOWN" && (
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

            {/* Pro / Lite: club picker during processing */}
            {(analysisMode === "pro" || analysisMode === "lite") && showClubPicker && (
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

        {(stage === "processing" || stage === "results") &&
          (analysisMode === "pro" || analysisMode === "lite") &&
          showHandPopup &&
          detectedHand &&
          !handConfirmed && (
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
                  type="button"
                  onClick={() => { setDetectedHand("R"); handRef.current = "R"; }}
                  className={`flex-1 rounded-xl border-2 py-4 text-center font-bold transition-all ${detectedHand === "R"
                    ? "border-brand-gold bg-brand-gold/15 text-brand-gold"
                    : "border-white/10 bg-white/5 text-white/40 hover:border-white/30"}`}
                >
                  <span className="block text-2xl mb-1">🫱</span>
                  {lang === "zh" ? "右手打球" : "Right-handed"}
                </button>
                <button
                  type="button"
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
                type="button"
                onClick={() => { void handleHandConfirm(); }}
                className="w-full rounded-xl bg-brand-gold py-3 text-sm font-bold text-black transition hover:bg-brand-gold/90"
              >
                {lang === "zh" ? "确认" : "Confirm"}
              </button>
            </div>
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

            {/* Lite: always show club + hand row when prediction exists (incl. unknown club + known hand) */}
            {analysisMode === "lite" && result.prediction && (
              <div className="space-y-2">
                <ClubHandSummaryBar
                  lang={lang}
                  clubType={litePredictionView.clubType}
                  clubConfidence={result.prediction.club_detection_confidence}
                  hand={litePredictionView.hand as "R" | "L" | "UNKNOWN"}
                  pending={false}
                />
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setShowClubPicker(true)}
                    className="w-full rounded-xl border border-brand-purple/35 bg-brand-purple/10 py-2.5 text-xs font-medium text-brand-purple transition hover:bg-brand-purple/20"
                  >
                    {lang === "zh" ? "修改球杆" : "Change club"}
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowHandPopup(true)}
                    className="w-full rounded-xl border border-brand-purple/35 bg-brand-purple/10 py-2.5 text-xs font-medium text-brand-purple transition hover:bg-brand-purple/20"
                  >
                    {lang === "zh" ? "确认左右手" : "Confirm hand"}
                  </button>
                </div>
                {litePredictionView.lowHandConfidence && (
                  <p className="text-xs text-yellow-300/90">
                    {lang === "zh"
                      ? "左右手识别置信度较低，请确认。"
                      : "Handedness confidence is low. Please confirm."}
                  </p>
                )}
              </div>
            )}

            {/* Pro: gold banner when club detected */}
            {analysisMode === "pro" &&
              result.prediction?.club_type &&
              result.prediction.club_type !== "UNKNOWN" && (
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
                          {lang === "zh" ? "AI 置信度" : "AI confidence"}:{" "}
                          {Math.round(result.prediction.club_detection_confidence * 100)}%
                          {result.prediction.club_detection_confidence < 0.7 && (
                            <span className="ml-1 text-yellow-400">
                              {lang === "zh" ? "· 建议手动确认" : "· please verify"}
                            </span>
                          )}
                        </p>
                      )}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setShowClubPicker(true)}
                    className="rounded-lg border border-brand-gold/30 bg-brand-gold/10 px-3 py-1.5 text-xs font-medium text-brand-gold transition hover:bg-brand-gold/20"
                  >
                    {lang === "zh" ? "修改球杆 ▾" : "Change ▾"}
                  </button>
                </div>
              )}

            {/* Results-stage club picker (Pro + Lite) */}
            {showClubPicker && result && (
              <div
                className="fixed inset-0 z-[100] flex items-end justify-center bg-black/60 backdrop-blur-sm sm:items-center"
                onClick={() => setShowClubPicker(false)}
              >
                <div
                  className="w-full max-w-md animate-fade-in rounded-t-2xl border border-white/10 bg-brand-dark p-5 sm:rounded-2xl"
                  onClick={(e) => e.stopPropagation()}
                >
                  <div className="mb-4 flex items-center justify-between">
                    <h3 className="text-base font-bold text-white">{lang === "zh" ? "选择球杆" : "Select Club"}</h3>
                    <button
                      type="button"
                      onClick={() => setShowClubPicker(false)}
                      className="text-lg text-white/30 hover:text-white/60"
                    >
                      &times;
                    </button>
                  </div>
                  <div className="space-y-3">
                    {CLUB_GROUPS.map((group) => (
                      <div key={group.id}>
                        <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-white/40">
                          {lang === "zh" ? group.label_zh : group.label_en}
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                          {group.clubs.map((club) => (
                            <button
                              key={club}
                              type="button"
                              onClick={() => handleClubOverride(club)}
                              className={`rounded-lg border px-3 py-2 text-xs font-medium transition-all ${
                                result.prediction.club_type === club
                                  ? analysisMode === "pro"
                                    ? "border-brand-gold/50 bg-brand-gold/20 text-brand-gold"
                                    : "border-brand-purple/50 bg-brand-purple/20 text-brand-purple"
                                  : "border-white/10 bg-white/5 text-white/60 hover:border-white/20 hover:text-white"
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

            {analysisMode === "pro" && isProv3StrictMediaPolicyResult(result as Prov3ResultLike) ? (
              stellarProTrustIsLow(result) ? (
                <div className="glass-card border border-amber-400/35 bg-amber-500/10 p-3 text-xs leading-relaxed text-amber-200">
                  {lang === "zh"
                    ? "低信任：关键帧未通过正式验证。下方条带为预览图，不作为正式相位关键帧；评分与建议仅供参考。"
                    : "Low trust: keyframes are not formally validated. The strip below is preview only, not official phase keyframes; scores and tips are for reference."}
                </div>
              ) : (
                <div className="glass-card border border-emerald-500/25 bg-emerald-500/[0.07] p-3 text-xs leading-relaxed text-emerald-100/90">
                  {lang === "zh"
                    ? "高信任：真 240 时间线关键帧与报告已通过验证，可与视频时间线对照使用。"
                    : "High trust: true-240 timeline keyframes and report passed validation—use with the timeline video."}
                </div>
              )
            ) : null}

            {analysisMode === "pro" && proVideoTimelineUrl ? (
              <div className="glass-card overflow-hidden p-0">
                <div className="border-b border-white/10 px-4 py-2.5">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-white/45">
                    {lang === "zh" ? "时间线视频" : "Timeline video"}
                  </h3>
                  <p className="mt-0.5 text-[10px] leading-snug text-white/35">
                    {lang === "zh"
                      ? "真 240 分析用 H.264；若原片 .mov 在浏览器中无法播放，请优先看此处。"
                      : "True-240 H.264 timeline. If the original .mov will not play in-browser, use this."}
                  </p>
                </div>
                {analyzePageIsProv3Product ? (
                  <div className="w-full bg-black px-1 pb-2 pt-1">
                    <FrontErrorBoundary
                      label="analyze-video-timeline"
                      details={{
                        hasVideoSrc: Boolean(String(proVideoTimelineUrl ?? "").trim()),
                        poseFramesCount: result.pose_frames?.length ?? 0,
                        hasPrediction: result.prediction != null,
                        sourceFrameCount: result.video_meta?.source_frame_count ?? null,
                        recordType: "analyze",
                        analysisId: result.analysis_id,
                        hasOfficialKeyframes:
                          Array.isArray(result.official_phase_keyframes) &&
                          result.official_phase_keyframes.length > 0,
                        hasPreviewKeyframes:
                          Array.isArray(result.preview_keyframes) && result.preview_keyframes.length > 0,
                        activeTab,
                        stripKeyframeCount: stripKeyframesForResult.length,
                      }}
                    >
                      <Prov3PlusVideoRenderer
                        videoSrc={proVideoTimelineUrl}
                        result={result}
                        lang={lang}
                      />
                    </FrontErrorBoundary>
                  </div>
                ) : (
                  <video
                    className="w-full max-h-[min(56vh,520px)] bg-black"
                    controls
                    playsInline
                    preload="metadata"
                    src={proVideoTimelineUrl}
                  />
                )}
              </div>
            ) : null}

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
                {analyzePageIsProv3Product && result ? (
                  <Prov3MotionEvidenceReport
                    result={result as unknown as PlusAnalysisResult}
                    lang={lang}
                  />
                ) : null}

                {stripKeyframesForResult.length > 0 ? (
                  analyzePageIsProv3Product ? (
                    <FrontErrorBoundary
                      label="analyze-keyframe-strip"
                      details={{
                        hasVideoSrc: Boolean(String(proVideoTimelineUrl ?? "").trim()),
                        poseFramesCount: result.pose_frames?.length ?? 0,
                        hasPrediction: result.prediction != null,
                        sourceFrameCount: result.video_meta?.source_frame_count ?? null,
                        recordType: "analyze",
                        analysisId: result.analysis_id,
                        hasOfficialKeyframes:
                          Array.isArray(result.official_phase_keyframes) &&
                          result.official_phase_keyframes.length > 0,
                        hasPreviewKeyframes:
                          Array.isArray(result.preview_keyframes) && result.preview_keyframes.length > 0,
                        activeTab,
                        stripKeyframeCount: stripKeyframesForResult.length,
                      }}
                    >
                      <KeyframeStrip
                        keyframes={stripKeyframesForResult}
                        lang={lang}
                        mode={analysisMode === "pro" ? "pro" : "default"}
                        urlOnlyTimeline={analysisMode === "pro" || analyzePageIsProv3Product}
                        plusStyleKeyframeSkeleton={analyzePageIsProv3Product}
                      />
                    </FrontErrorBoundary>
                  ) : (
                    <KeyframeStrip
                      keyframes={stripKeyframesForResult}
                      lang={lang}
                      mode={analysisMode === "pro" ? "pro" : "default"}
                      urlOnlyTimeline={analysisMode === "pro" || analyzePageIsProv3Product}
                      plusStyleKeyframeSkeleton={analyzePageIsProv3Product}
                    />
                  )
                ) : null}

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
