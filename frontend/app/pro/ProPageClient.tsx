"use client";

import { useState, useCallback, useEffect, useRef, type Dispatch, type SetStateAction } from "react";
import { useRouter } from "next/navigation";
import UploadZone from "@/components/UploadZone";
import PlusResultView from "@/components/PlusResultView";
import FrontErrorBoundary from "@/components/debug/FrontErrorBoundary";
import ScreenModeCapture from "@/components/ScreenModeCapture";
import { devError, devInfo, devWarn } from "@/lib/dev-only-log";
import { preloadPoseModel } from "@/lib/mediapipe-assets";
import AnalysisWaiting from "@/components/AnalysisWaiting";
import type { PoseSnapshot } from "@/components/KeyframeStrip";
import { getAnalysisVideoBlob, saveAnalysisVideo } from "@/lib/video-store";
import { clientLikelyMainlandChinaUser } from "@/lib/client-region-hint";
import {
  PRO_V3_EDGE_PRECHECK_PATH,
  requestProv3AnalyzeCancel,
  runProv3AnalyzeMultipart,
  yieldUiBeforeHeavyParse,
} from "@/lib/pro-v3-api";
import {
  slimAnalysisResultForHistoryTransport,
  slimAnalysisResultForServerHistory,
} from "@/lib/strip-result";
import { normalizedTotalScoreForStorage } from "@/lib/safe-analysis-score";
import { patchLocalHistoryVideoR2Key } from "@/lib/history-sync-record";
import { expandStellarProForUi, proExpandedToPlusViewModel } from "@/lib/stellar-pro-result";
import { resolveProv3ProductMediaUrl } from "@/lib/prov3-media-url";
import {
  isProv3ScreenWebmForMp4Upload,
  prov3ScreenRecordingToMp4File,
} from "@/lib/prov3-screen-recording-to-mp4";

/** Survives ``router.replace`` remount to /pro/[id] before localStorage / IndexedDB sync. */
function proSessionResultKey(analysisId: string): string {
  return `stellar_pro_session_result_v1_${analysisId}`;
}
import { pruneLocalStellarHistoryRecords } from "@/lib/pro-history-retention";
import {
  fetchVideoBlobForHistoryReanalyze,
  reanalyzeHistoryFilename,
  reanalyzePayloadProv3ScreenMode,
  reconcileProPageReanalyzeSession,
} from "@/lib/reanalyze-from-history";
import { loadProAnalysisById } from "@/lib/load-pro-analysis-by-id";

/**
 * IndexedDB video can land slightly after ``router.replace`` / session restore; PlusResultView skips IDB when
 * ``externalVideoSrc`` is set, so we attach blob here (same backoff as PlusResultView).
 */
function scheduleProVideoFromIndexedDB(
  analysisId: string,
  setSrc: Dispatch<SetStateAction<string | null>>,
  isCancelled: () => boolean,
) {
  const gapsMs = [0, 150, 400, 1000, 2000];
  void (async () => {
    for (let i = 0; i < gapsMs.length; i++) {
      if (i > 0) await new Promise((r) => setTimeout(r, gapsMs[i] - gapsMs[i - 1]));
      if (isCancelled()) return;
      const blob = await getAnalysisVideoBlob(analysisId).catch(() => null);
      if (blob && blob.size > 0) {
        if (isCancelled()) return;
        setSrc((prev) => {
          if (prev?.startsWith("blob:")) {
            try {
              URL.revokeObjectURL(prev);
            } catch {
              /* ignore */
            }
          }
          return URL.createObjectURL(blob);
        });
        return;
      }
    }
  })();
}

function isVideoBlobForOverlay(blob: Blob, filename: string): boolean {
  const t = (blob.type || "").toLowerCase();
  if (t.startsWith("video/")) return true;
  if (/\.(mp4|mov|webm|m4v|avi|mkv|3gp|qt)$/i.test(filename)) return true;
  if (t === "application/octet-stream" && /\.(mp4|mov|webm|m4v|avi|mkv)$/i.test(filename)) return true;
  return false;
}

interface ProAnalysisResult {
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
  advanced_metrics: Record<string, unknown>;
  training_plan: Record<string, { focus: string; drills: string[]; duration: string }>;
  keyframes: Array<{
    phase: string;
    label_en: string;
    label_zh: string;
    timestamp: number;
    image_base64: string;
    pose_snapshot?: PoseSnapshot | null;
  }>;
  skeleton_data: {
    frames: Array<Record<string, unknown>>;
    total_frames: number;
  };
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
    hand?: "R" | "L" | "UNKNOWN";
    hand_confidence?: number;
    baseline_distance?: number;
    technique_multiplier?: number;
    strike_multiplier?: number;
    speed_multiplier?: number;
    distance_confidence?: number;
    distance_debug?: Record<string, unknown>;
  };
  trajectory: Array<{ frame_index: number; timestamp: number; x: number; y: number; speed: number }>;
  keyframe_validation?: Record<string, unknown>;
  video_meta?: {
    fps?: number;
    total_pose_frames?: number;
    duration_s?: number;
    source_frame_count?: number;
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
}

type Stage = "upload" | "processing" | "rendering" | "results";
type InputMode = "upload" | "capture" | "screen";

/** Client implementation; mounted from ``app/pro/page.tsx`` (no id) or ``app/pro/[analysisId]/page.tsx``. */
export default function ProPageClient({ deepLinkAnalysisId }: { deepLinkAnalysisId?: string }) {
  const router = useRouter();
  const deepId = (deepLinkAnalysisId || "").trim();
  const [stage, setStage] = useState<Stage>(deepId ? "processing" : "upload");
  const [result, setResult] = useState<ProAnalysisResult | null>(null);
  const [resultRenderError, setResultRenderError] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [progress, setProgress] = useState(deepId ? 15 : 0);
  const [liveCapture, setLiveCapture] = useState(false);
  const [inputMode, setInputMode] = useState<InputMode>("upload");
  const [lang, setLang] = useState<"en" | "zh">("zh");
  const [proVideoSrc, setProVideoSrc] = useState<string | null>(null);
  const [username, setUsername] = useState("");
  const [screenRecording, setScreenRecording] = useState(false);
  const [screenRecTime, setScreenRecTime] = useState(0);
  /** Shown on AnalysisWaiting while Pro v3 runs with `screen_mode=true`. */
  const [processingProScreenMode, setProcessingProScreenMode] = useState(false);
  /** Extra line under Pro waiting UI (upload / job / reconnecting hints). */
  const [proWaitSubline, setProWaitSubline] = useState("");

  const screenStreamRef = useRef<MediaStream | null>(null);
  const screenRecRef = useRef<MediaRecorder | null>(null);
  const screenChunksRef = useRef<Blob[]>([]);
  const screenTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  /** Pro v3 对屏预处理：拍屏 tab 进入、尚未打开实拍时为 true。 */
  const screenCaptureForProv3Ref = useRef(false);
  /** 对屏模式分析完成后先打开姿势诊断（报告文案）；上传主路径仍以视频分析为先。 */
  const prov3ScreenOpenDiagnosisTabRef = useRef(false);

  const cnNetworkHintRef = useRef(false);
  /** 防止双击/历史再次分析竞态导致重复 POST Modal。 */
  const analysisInFlightRef = useRef(false);
  const proAnalyzeAbortRef = useRef<AbortController | null>(null);
  const deepLinkStartedForRef = useRef<string | null>(null);
  /** 本页仅允许完成一次 Pro（单次分析任务）；用户主动「停止」不计入，可重试；刷新页面后重置。 */
  const [proAnalyzeLocked, setProAnalyzeLocked] = useState(false);

  const stopProAnalysis = useCallback(async () => {
    const token = localStorage.getItem("stellar_token");
    const authHeaders: Record<string, string> = {};
    if (token && token.includes(".")) authHeaders["Authorization"] = `Bearer ${token}`;
    await requestProv3AnalyzeCancel(authHeaders);
    proAnalyzeAbortRef.current?.abort();
  }, []);

  useEffect(() => {
    const token = localStorage.getItem("stellar_token");
    if (!token || token.startsWith("local-")) {
      window.location.href = "/pro-login";
      return;
    }

    const user = localStorage.getItem("stellar_user");
    if (user) {
      try {
        const parsed = JSON.parse(user);
        setUsername(parsed.username || parsed.email || "");
      } catch { /* ignore */ }
    }

    // Verify pro status server-side and get backend URLs
    fetch(PRO_V3_EDGE_PRECHECK_PATH, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(r => {
        if (r.status === 401 || r.status === 403) {
          window.location.href = "/pro-login";
          return null;
        }
        return r.json();
      })
      .then(data => {
        if (!data) return;
        if (!data.allowed || !data.is_pro) {
          window.location.href = "/pro-login";
          return;
        }
        cnNetworkHintRef.current = data.network_hint === "cn";
      })
      .catch(() => {});

    preloadPoseModel();
    pruneLocalStellarHistoryRecords();
  }, []);

  useEffect(() => {
    if (!deepId) return;
    const token = localStorage.getItem("stellar_token");
    if (!token || token.startsWith("local-")) return;
    if (deepLinkStartedForRef.current === deepId) return;
    deepLinkStartedForRef.current = deepId;

    let cancelled = false;
    void (async () => {
      setStage("processing");
      setProgress(25);
      setError("");
      try {
        let sessionRaw: string | null = null;
        try {
          sessionRaw = sessionStorage.getItem(proSessionResultKey(deepId));
        } catch {
          sessionRaw = null;
        }
        if (sessionRaw) {
          try {
            const parsed = JSON.parse(sessionRaw) as Record<string, unknown>;
            const data = expandStellarProForUi(parsed) as ProAnalysisResult;
            if (!cancelled && !analysisInFlightRef.current) {
              setResult(data);
              setProgress(100);
              setStage("results");
              setResultRenderError(null);
              const d = data as unknown as Record<string, unknown>;
              const vu = resolveProv3ProductMediaUrl(
                String(
                  d.analysis_video_url ||
                    d.playback_video_url ||
                    d.video_url ||
                    d.original_video_url ||
                    "",
                ).trim(),
              );
              if (vu.startsWith("http")) {
                setProVideoSrc(vu);
              } else {
                setProVideoSrc(null);
              }
              scheduleProVideoFromIndexedDB(deepId, setProVideoSrc, () => cancelled);
              return;
            }
            // Session had payload but we skipped UI (e.g. analyze still marked in-flight, or Strict Mode
            // cancelled the first run). Fall through to history/API so the results page is not blank.
          } catch (e) {
            devWarn("[pro] session result restore failed", e);
          }
        }

        const loaded = await loadProAnalysisById(deepId, token);
        if (cancelled) return;
        if (analysisInFlightRef.current) {
          return;
        }
        if (!loaded) {
          setError(
            lang === "zh"
              ? "找不到该分析记录。请确认已登录且记录在本机或云端仍存在。"
              : "Analysis not found. Sign in and ensure the record exists locally or on the server.",
          );
          setStage("upload");
          setProgress(0);
          deepLinkStartedForRef.current = null;
          return;
        }
        const data = expandStellarProForUi(loaded.raw) as ProAnalysisResult;
        setResult(data);
        setProgress(100);
        setStage("results");
        if (loaded.videoBlob && loaded.videoBlob.size > 0) {
          setProVideoSrc(URL.createObjectURL(loaded.videoBlob));
        } else {
          const vu = resolveProv3ProductMediaUrl(
            String(
              loaded.raw.analysis_video_url ||
                loaded.raw.playback_video_url ||
                loaded.raw.video_url ||
                loaded.raw.original_video_url ||
                "",
            ).trim(),
          );
          if (vu.startsWith("http")) setProVideoSrc(vu);
          else setProVideoSrc(null);
          scheduleProVideoFromIndexedDB(deepId, setProVideoSrc, () => cancelled);
        }
      } catch {
        if (!cancelled) {
          setError(lang === "zh" ? "加载记录失败" : "Failed to load analysis");
          setStage("upload");
          setProgress(0);
          deepLinkStartedForRef.current = null;
        }
      }
    })();
    return () => {
      cancelled = true;
      // React Strict Mode / remount: allow the next mount to run restore again (otherwise the guard
      // below blocks forever and session restore never applies).
      if (deepLinkStartedForRef.current === deepId) {
        deepLinkStartedForRef.current = null;
      }
    };
    // 仅 deepId 变化时重载记录；勿依赖 lang，避免重复拉取同一条目。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deepId]);

  useEffect(() => {
    const explicitReanalyzeNav =
      typeof window !== "undefined" &&
      new URLSearchParams(window.location.search).get("reanalyze") === "1";

    const stripReanalyzeQuery = () => {
      if (typeof window === "undefined") return;
      const u = new URL(window.location.href);
      if (!u.searchParams.has("reanalyze")) return;
      u.searchParams.delete("reanalyze");
      const qs = u.searchParams.toString();
      window.history.replaceState(null, "", `${u.pathname}${qs ? `?${qs}` : ""}${u.hash}`);
    };

    // Deep link to a saved analysis: never run history reanalyze; drop stale pro-only queue.
    if (deepId) {
      reconcileProPageReanalyzeSession(false);
      if (explicitReanalyzeNav) stripReanalyzeQuery();
      return;
    }

    const p = reconcileProPageReanalyzeSession(explicitReanalyzeNav);
    if (explicitReanalyzeNav) stripReanalyzeQuery();
    if (!p) return;

    void (async () => {
      try {
        if (analysisInFlightRef.current) {
          devWarn("[pro] reanalyze skipped while Pro analysis already in flight");
          return;
        }
        // History queues both URLs; timeline first is implemented inside fetchVideoBlobForHistoryReanalyze.
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
        const sm = reanalyzePayloadProv3ScreenMode(p);
        if (sm) setInputMode("screen");
        await processBlob(blob, reanalyzeHistoryFilename(blob), sm);
      } catch (e) {
        devWarn("[pro] reanalyze pipeline error:", e);
      }
    })();
    // 勿依赖 lang：避免切换语言时重复 reconcile / 异步链。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deepId]);

  // Do not revoke proVideoSrc in a [proVideoSrc] effect cleanup: React Strict Mode runs that
  // cleanup while the URL is still needed, breaking the <video> src. Revoke only when replacing
  // or clearing via setProVideoSrc updaters / "Analyze Another".

  function minimalProHistoryPayload(data: ProAnalysisResult): Record<string, unknown> {
    return {
      analysis_id: data.analysis_id,
      type: "pro",
      total_score: data.total_score,
      scores: data.scores ?? {},
      summary: data.summary ?? "",
      summary_zh: data.summary_zh ?? "",
      issues: Array.isArray(data.issues) ? data.issues.slice(0, 12) : [],
      issues_zh: Array.isArray(data.issues_zh) ? data.issues_zh.slice(0, 12) : [],
      suggestions: Array.isArray(data.suggestions) ? data.suggestions.slice(0, 8) : [],
      suggestions_zh: Array.isArray(data.suggestions_zh) ? data.suggestions_zh.slice(0, 8) : [],
    };
  }

  function saveToLocalHistory(data: ProAnalysisResult) {
    const key = "stellar_history_local";
    const id = (data.analysis_id || "").trim() || `local-${Date.now()}`;
    const writeEntry = (resultPayload: unknown) => {
      const existing = JSON.parse(localStorage.getItem(key) || "[]");
      const entry = {
        id,
        type: "pro",
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
        writeEntry(minimalProHistoryPayload(data));
        pruneLocalStellarHistoryRecords();
      } catch (e2) {
        devWarn("[pro] local history save failed (quota or storage):", e2, e1);
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

  async function saveAnalysisToHistory(data: ProAnalysisResult, blob?: Blob, filename?: string) {
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

      let resultForApi: Record<string, unknown>;
      try {
        resultForApi = slimAnalysisResultForServerHistory(data) as Record<string, unknown>;
        JSON.stringify(resultForApi);
      } catch (e) {
        devWarn("[pro] history API payload slim failed, using minimal fields:", e);
        resultForApi = minimalProHistoryPayload(data);
      }

      const res = await fetch("/api/history", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          analysis_id: data.analysis_id,
          type: "pro",
          total_score: normalizedTotalScoreForStorage(data.total_score),
          result: resultForApi,
          video_r2_key: videoR2Key,
        }),
      });
      if (res.ok) {
        const saved = await res.json().catch(() => ({}));
        if (saved.success !== false) {
          markLocalRecordSynced(data.analysis_id);
        } else {
          devWarn("[history] pro save rejected:", saved.detail || saved);
        }
      } else {
        const errText = await res.text().catch(() => "");
        devWarn("[history] pro save failed:", res.status, errText.slice(0, 400));
      }
    } catch (e) {
      devWarn("[history] pro save error:", e);
    }
  }

  function resolveProv3ScreenMode(filename: string, explicit: boolean | undefined): boolean {
    if (explicit === true) return true;
    if (explicit === false) return false;
    if (inputMode === "screen") return true;
    return /(?:^|\/)(screen-capture\.webm|pro-screen\.webm)$/i.test(filename);
  }

  async function processBlob(blob: Blob, filename: string, prov3ScreenMode?: boolean) {
    if (proAnalyzeLocked) {
      setError(
        lang === "zh"
          ? "本页仅可提交一次 Pro 分析。请刷新页面后再试。"
          : "Only one Pro analysis per page visit. Refresh to analyze again.",
      );
      return;
    }
    if (analysisInFlightRef.current) {
      devWarn("[pro] analyze already in flight, ignoring duplicate trigger");
      return;
    }
    analysisInFlightRef.current = true;
    try {
    prov3ScreenOpenDiagnosisTabRef.current = resolveProv3ScreenMode(filename, prov3ScreenMode);
    setProcessingProScreenMode(resolveProv3ScreenMode(filename, prov3ScreenMode));
    setStage("processing");
    setProWaitSubline("");
    setResult(null);
    setResultRenderError(null);
    setError("");
    setProgress(0);
    setProVideoSrc((prev) => {
      if (prev) try { URL.revokeObjectURL(prev); } catch { /* */ }
      return null;
    });
    // Edge-orchestrated Pro v3: same-origin upload + job poll (no long browser→Modal analyze POST).
    setProgress(42);

    const effectiveScreenMode = resolveProv3ScreenMode(filename, prov3ScreenMode);
    let uploadBlob: Blob = blob;
    let uploadFilename = filename;
    if (isProv3ScreenWebmForMp4Upload(blob, filename, prov3ScreenMode, resolveProv3ScreenMode)) {
      setProWaitSubline(
        lang === "zh"
          ? "正在将屏幕录制转为 MP4，随后走与上传文件相同的提交流程…"
          : "Converting screen recording to MP4, then uploading like a normal file…",
      );
      try {
        const mp4File = await prov3ScreenRecordingToMp4File(blob);
        uploadBlob = mp4File;
        uploadFilename = mp4File.name;
      } catch (e) {
        devWarn("[pro] screen WebM→MP4 failed, using original container", e);
      }
      setProWaitSubline("");
    }

    const proAbort = new AbortController();
    proAnalyzeAbortRef.current = proAbort;

    try {
      const token = localStorage.getItem("stellar_token");
      const authHeaders: Record<string, string> = {};
      if (token && token.includes(".")) authHeaders["Authorization"] = `Bearer ${token}`;

      let raw: Record<string, unknown>;
      try {
        const cn = cnNetworkHintRef.current;
        const out = await runProv3AnalyzeMultipart(uploadBlob, uploadFilename, authHeaders, {
          cnNetworkHint: cn,
          unboundedJobPoll: cn || clientLikelyMainlandChinaUser(),
          screenMode: effectiveScreenMode,
          // Polling-only budget (after job_id). Long Modal runs + margin; upload no longer consumes this window.
          modalTimeoutMs: 600_000,
          renderTimeoutMs: 600_000,
          logPrefix: "[pro]",
          abortSignal: proAbort.signal,
          userCancelledMessage: lang === "zh" ? "分析已停止" : "Analysis stopped",
          onJobPhase: (phase) => {
            if (phase === "reconnecting") {
              setProWaitSubline(
                lang === "zh"
                  ? "网络不稳定，正在重新连接并查询进度… 分析可能仍在云端进行，也可稍后在「历史记录」查看。"
                  : "Reconnecting to check progress… The server may still be running; try History if this persists.",
              );
            } else if (phase === "polling") {
              setProWaitSubline(
                lang === "zh"
                  ? "分析在云端进行中，请保持页面…"
                  : "Server-side analysis in progress; keep this page open…",
              );
            } else if (phase === "uploading") {
              setProWaitSubline(
                lang === "zh" ? "正在上传视频到安全存储…" : "Uploading video to secure storage…",
              );
            } else {
              setProWaitSubline(
                lang === "zh" ? "正在提交分析任务（单次）…" : "Submitting analysis job (single request)…",
              );
            }
          },
        });
        raw = out.raw;
      } catch (e) {
        const msg = e instanceof Error ? e.message : "";
        if (msg === "分析已停止" || msg === "Analysis stopped") {
          setError("");
          setProWaitSubline("");
          setProcessingProScreenMode(false);
          setStage("upload");
          return;
        }
        setProAnalyzeLocked(true);
        setError(
          e instanceof Error
            ? e.message
            : lang === "zh"
              ? "分析失败"
              : "Analysis failed",
        );
        setProWaitSubline("");
        setProcessingProScreenMode(false);
        setStage("upload");
        return;
      }

      devInfo("[pro] Pro v3 edge-job completed");
      setProgress(96);
      setProWaitSubline("");
      setStage("rendering");
      await yieldUiBeforeHeavyParse();
      devInfo("[pro] result payload keys:", Object.keys(raw).length);
      let data: ProAnalysisResult;
      try {
        data = expandStellarProForUi(raw) as ProAnalysisResult;
        const finalStatus = String((raw as { final_status?: unknown }).final_status ?? "");
        const trustLevel = String(
          (raw as { analysis_trust?: unknown; trust_level?: unknown }).analysis_trust ??
            (raw as { trust_level?: unknown }).trust_level ??
            "",
        );
        const displayCount = Array.isArray((data as { keyframes?: unknown[] }).keyframes)
          ? (data as { keyframes?: unknown[] }).keyframes?.length ?? 0
          : 0;
        const officialCount = Array.isArray((raw as { official_phase_keyframes?: unknown[] }).official_phase_keyframes)
          ? (raw as { official_phase_keyframes?: unknown[] }).official_phase_keyframes?.length ?? 0
          : 0;
        const previewCount = Array.isArray((raw as { preview_keyframes?: unknown[] }).preview_keyframes)
          ? (raw as { preview_keyframes?: unknown[] }).preview_keyframes?.length ?? 0
          : 0;
        devInfo("[pro] parsed final_status/trust:", { finalStatus, trustLevel });
        devInfo("[pro] display/official/preview keyframes:", {
          displayCount,
          officialCount,
          previewCount,
        });
      } catch (parseErr) {
        devError("[pro] expand result failed:", parseErr, {
          analysis_id: String((raw as { analysis_id?: unknown }).analysis_id ?? ""),
          final_status: String((raw as { final_status?: unknown }).final_status ?? ""),
          trust_level: String(
            (raw as { analysis_trust?: unknown; trust_level?: unknown }).analysis_trust ??
              (raw as { trust_level?: unknown }).trust_level ??
              "",
          ),
          keyframes_len: Array.isArray((raw as { keyframes?: unknown[] }).keyframes)
            ? (raw as { keyframes?: unknown[] }).keyframes?.length ?? 0
            : 0,
        });
        setResult(null);
        setResultRenderError(
          lang === "zh"
            ? "分析结果解析失败，请重试或缩短视频"
            : "Could not parse analysis result. Try again or use a shorter clip.",
        );
        setProgress(100);
        setStage("results");
        return;
      }
      setProgress(100);
      setResult(data);
      setResultRenderError(null);
      setStage("results");
      setProAnalyzeLocked(true);
      if (typeof window !== "undefined" && data.analysis_id) {
        try {
          sessionStorage.setItem(proSessionResultKey(data.analysis_id), JSON.stringify(data));
        } catch (e) {
          devWarn("[pro] session snapshot before /pro/[id] navigation failed", e);
        }
      }
      if (isVideoBlobForOverlay(uploadBlob, uploadFilename) && uploadBlob.size > 0) {
        setProVideoSrc(URL.createObjectURL(uploadBlob));
      }
      if (typeof window !== "undefined" && data.analysis_id) {
        try {
          router.replace(`/pro/${encodeURIComponent(data.analysis_id)}`);
        } catch {
          try {
            window.history.replaceState(null, "", `/pro/${encodeURIComponent(data.analysis_id)}`);
          } catch {
            /* ignore */
          }
        }
      }
      // 大 payload：先切结果页再写 localStorage / 同步，避免移动端主线程长时间卡在 JSON.stringify 前看不到 UI
      window.setTimeout(() => {
        try {
          saveToLocalHistory(data);
        } catch (e) {
          devWarn("[pro] deferred local history:", e);
        }
        void saveAnalysisVideo(data.analysis_id, uploadBlob, uploadFilename).catch(() => {});
        void saveAnalysisToHistory(data, uploadBlob, uploadFilename).catch((e) => {
          devWarn("[pro] history save failed:", e);
        });
      }, 0);
    } catch (err: unknown) {
      setProcessingProScreenMode(false);
      setProWaitSubline("");
      const msg = err instanceof Error ? err.message : "";
      if (
        msg === "分析已停止" ||
        msg === "Analysis stopped" ||
        msg.includes("分析已取消")
      ) {
        setError("");
      } else {
        setProAnalyzeLocked(true);
        setError(err instanceof Error ? err.message : "Analysis failed");
      }
      setStage("upload");
    }
    } finally {
      analysisInFlightRef.current = false;
      proAnalyzeAbortRef.current = null;
    }
  }

  const handleUploadComplete = useCallback(
    (file: File) => {
      screenCaptureForProv3Ref.current = false;
      processBlob(file, file.name, false);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [lang]
  );

  const handleVideoCapture = useCallback(
    (videoBlob: Blob) => {
      setLiveCapture(false);
      processBlob(videoBlob, "pro-capture.webm", screenCaptureForProv3Ref.current);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [lang]
  );

  const handleScreenCapture = useCallback(
    (imageBase64: string) => {
      setLiveCapture(false);
      const byteChars = atob(imageBase64);
      const byteArray = new Uint8Array(byteChars.length);
      for (let i = 0; i < byteChars.length; i++) byteArray[i] = byteChars.charCodeAt(i);
      processBlob(
        new Blob([byteArray], { type: "image/jpeg" }),
        "pro-capture.jpg",
        screenCaptureForProv3Ref.current,
      );
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [lang]
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
        if (blob.size > 0) processBlob(blob, "pro-screen.webm", true);
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
        <button onClick={stopScreenRecording} className="rounded-xl bg-brand-gold px-8 py-3 text-sm font-semibold text-black transition hover:bg-brand-gold/80">
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
      {/* Pro Header */}
      <nav className="sticky top-0 z-50 border-b border-brand-gold/20 bg-black/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
          <a href="/" className="flex items-center gap-2">
            <img src="/logo.svg" alt="Stellar" className="h-8 w-8" />
            <span className="text-xl font-bold text-brand-gold">STELLAR</span>
            <span className="rounded-full border border-brand-gold/30 bg-brand-gold/10 px-2 py-0.5 text-[10px] font-bold tracking-widest text-brand-gold">
              PRO
            </span>
          </a>
          <div className="flex items-center gap-3">
            {username && (
              <a href="/history"
                className="flex items-center gap-1 rounded-lg border border-brand-gold/20 px-2.5 py-1 text-xs text-brand-gold/60 transition hover:text-brand-gold">
                <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A17.933 17.933 0 0 1 12 21.75c-2.676 0-5.216-.584-7.499-1.632Z" />
                </svg>
                <span className="font-medium text-brand-gold">{username}</span>
              </a>
            )}
            <button
              onClick={() => setLang(lang === "en" ? "zh" : "en")}
              className="rounded-lg border border-brand-gold/20 px-3 py-1 text-xs text-brand-gold/60 transition hover:text-brand-gold"
            >
              {lang === "en" ? "中文" : "EN"}
            </button>
          </div>
        </div>
      </nav>

      <div className="mx-auto max-w-7xl px-6 py-8">
        {stage === "upload" && (
          <div className="animate-fade-in">
            <div className="mb-2 flex items-center justify-center gap-2">
              <h1 className="text-center text-3xl font-bold text-white">
                {lang === "en" ? "Pro Swing Analysis" : "Pro专业挥杆分析"}
              </h1>
              <span className="rounded-full bg-brand-gold/20 px-2 py-0.5 text-xs text-brand-gold">
                PRO
              </span>
            </div>
            <p className="mb-6 text-center text-white/50">
              {lang === "en"
                ? "Upload, capture, or record from screen — full 3D + pro comparison + training plan"
                : "上传视频、实拍、或屏幕录制 — 完整3D + 职业对比 + 训练计划"}
            </p>

            {error && (
              <div className="mx-auto mb-6 max-w-lg rounded-xl bg-red-500/10 border border-red-500/20 p-4 text-sm text-red-400">
                <p className="font-semibold mb-1">{lang === "zh" ? "分析出错" : "Error"}</p>
                <p className="text-red-400/80 break-words whitespace-pre-line">{error}</p>
                <button type="button" onClick={() => setError("")} className="mt-3 text-xs text-white/50 underline hover:text-white/70">
                  {lang === "zh" ? "关闭" : "Close"}
                </button>
              </div>
            )}

            {/* Input Mode Tabs */}
            <div className="mx-auto mb-6 flex max-w-xl rounded-xl border border-brand-gold/10 bg-white/[0.02] p-1 overflow-hidden">
              <button onClick={() => setInputMode("upload")}
                className={`flex-1 rounded-lg py-2.5 text-xs font-semibold transition-all ${
                  inputMode === "upload"
                    ? "bg-brand-gold/20 text-brand-gold border border-brand-gold/30"
                    : "text-white/40 hover:text-white/60 border border-transparent"
                }`}>
                {lang === "zh" ? "上传视频" : "Upload"}
              </button>
              <button onClick={() => setInputMode("capture")}
                className={`flex-1 rounded-lg py-2.5 text-xs font-semibold transition-all ${
                  inputMode === "capture"
                    ? "bg-brand-gold/20 text-brand-gold border border-brand-gold/30"
                    : "text-white/40 hover:text-white/60 border border-transparent"
                }`}>
                {lang === "zh" ? "实拍模式" : "Camera"}
              </button>
              <button onClick={() => setInputMode("screen")}
                className={`flex-1 rounded-lg py-2.5 text-xs font-semibold transition-all ${
                  inputMode === "screen"
                    ? "bg-brand-gold/20 text-brand-gold border border-brand-gold/30"
                    : "text-white/40 hover:text-white/60 border border-transparent"
                }`}>
                {lang === "zh" ? "屏幕模式" : "Screen"}
              </button>
            </div>

            {inputMode === "upload" && (
              <>
                {proAnalyzeLocked && (
                  <p className="mb-3 text-center text-xs text-brand-gold/80">
                    {lang === "zh"
                      ? "本页已使用过 Pro 分析。刷新页面后可再次上传。"
                      : "This page already used its Pro analysis. Refresh to upload again."}
                  </p>
                )}
                <UploadZone
                  onUploadComplete={handleUploadComplete}
                  lang={lang}
                  isPro
                  disabled={proAnalyzeLocked}
                />
              </>
            )}

            {inputMode === "capture" && (
              <div className="mx-auto max-w-xl">
                <div className="glass-card overflow-hidden">
                  <div className="relative bg-gradient-to-b from-brand-gold/10 to-transparent p-8 text-center">
                    <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-2xl bg-brand-gold/10 border border-brand-gold/20">
                      <svg className="h-10 w-10 text-brand-gold" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25h-9A2.25 2.25 0 0 0 2.25 7.5v9a2.25 2.25 0 0 0 2.25 2.25Z" />
                      </svg>
                    </div>
                    <h3 className="mb-2 text-lg font-semibold text-white">
                      {lang === "zh" ? "实拍模式" : "Live Capture"}
                    </h3>
                    <p className="mb-2 text-sm text-white/40">
                      {lang === "zh" ? "使用摄像头录制挥杆，AI骨架实时引导" : "Record swing with camera, AI skeleton overlay"}
                    </p>
                    <p className="mb-6 text-[10px] text-brand-gold/50">
                      {lang === "zh" ? "包含3D骨架 · 职业对比 · 训练计划" : "Includes 3D skeleton · Pro compare · Training plan"}
                    </p>
                    <button
                      onClick={() => {
                        screenCaptureForProv3Ref.current = false;
                        setLiveCapture(true);
                      }}
                      className="rounded-xl bg-brand-gold px-6 py-3 text-sm font-semibold text-black transition hover:bg-brand-gold/80"
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
                      <svg className="h-10 w-10 text-brand-gold" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 17.25v1.007a3 3 0 0 1-.879 2.122L7.5 21h9l-.621-.621A3 3 0 0 1 15 18.257V17.25m6-12V15a2.25 2.25 0 0 1-2.25 2.25H5.25A2.25 2.25 0 0 1 3 15V5.25A2.25 2.25 0 0 1 5.25 3h13.5A2.25 2.25 0 0 1 21 5.25Z" />
                      </svg>
                    </div>
                    <h3 className="mb-2 text-lg font-semibold text-white">
                      {lang === "zh" ? "屏幕模式" : "Screen Mode"}
                    </h3>
                    <p className="mb-2 text-sm text-white/40">
                      {lang === "zh"
                        ? "录制电脑屏幕上的高尔夫视频，或用摄像头对着手机/电视拍摄"
                        : "Record golf video from screen, or point camera at phone/TV"}
                    </p>
                    <p className="mb-6 text-[10px] text-brand-gold/50">
                      {lang === "zh" ? "包含3D骨架 · 职业对比 · 训练计划" : "Includes 3D skeleton · Pro compare · Training plan"}
                    </p>
                    <div className="flex flex-col sm:flex-row gap-3 justify-center">
                      <button onClick={startScreenRecording}
                        className="rounded-xl bg-brand-gold px-6 py-3 text-sm font-semibold text-black transition hover:bg-brand-gold/80">
                        {lang === "zh" ? "录制屏幕" : "Record Screen"}
                      </button>
                      <button
                        onClick={() => {
                          screenCaptureForProv3Ref.current = true;
                          setLiveCapture(true);
                        }}
                        className="rounded-xl border border-brand-gold/20 bg-brand-gold/5 px-6 py-3 text-sm text-brand-gold/70 transition hover:bg-brand-gold/10"
                      >
                        {lang === "zh" ? "对屏拍摄" : "Point Camera"}
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
          <AnalysisWaiting
            progress={progress}
            lang={lang}
            mode="pro"
            prov3ScreenMode={processingProScreenMode}
            statusSubline={proWaitSubline}
            onCancel={stopProAnalysis}
          />
        )}
        {stage === "rendering" && (
          <div className="mx-auto max-w-xl rounded-xl border border-brand-gold/20 bg-white/[0.03] p-6 text-center">
            <div className="mx-auto h-8 w-8 animate-spin rounded-full border-2 border-white/15 border-t-brand-gold/70" />
            <p className="mt-4 text-sm text-white/70">
              {lang === "zh" ? "分析已完成，正在展示结果…" : "Analysis finished. Preparing your results…"}
            </p>
          </div>
        )}

        {stage === "results" && (
          <>
            <div className="mb-4 flex justify-center">
              <span className="rounded-full border border-brand-gold/40 bg-brand-gold/10 px-3 py-1 text-[11px] font-bold tracking-wider text-brand-gold">
                STELLAR PRO
              </span>
            </div>
            {result ? (
              <FrontErrorBoundary
                label="ProPageClient PlusResultView (prov3 result)"
                details={{
                  hasVideoSrc: Boolean(String(proVideoSrc ?? "").trim()),
                  poseFramesCount: result.pose_frames?.length ?? 0,
                  hasPrediction: result.prediction != null,
                  sourceFrameCount: result.video_meta?.source_frame_count ?? null,
                  recordType: "pro",
                  analysisId: result.analysis_id,
                  hasOfficialKeyframes:
                    Array.isArray(
                      (result as { official_phase_keyframes?: unknown[] }).official_phase_keyframes,
                    ) &&
                    ((result as { official_phase_keyframes?: unknown[] }).official_phase_keyframes
                      ?.length ?? 0) > 0,
                  hasPreviewKeyframes:
                    Array.isArray((result as { preview_keyframes?: unknown[] }).preview_keyframes) &&
                    ((result as { preview_keyframes?: unknown[] }).preview_keyframes?.length ?? 0) > 0,
                }}
              >
                <PlusResultView
                  key={result.analysis_id}
                  result={proExpandedToPlusViewModel(result as unknown as Record<string, unknown>)}
                  lang={lang}
                  externalVideoSrc={proVideoSrc}
                  coachingMode="pro"
                  initialActiveTab={prov3ScreenOpenDiagnosisTabRef.current ? "diagnosis" : "video"}
                />
              </FrontErrorBoundary>
            ) : (
              <div className="mx-auto mb-6 max-w-xl rounded-xl border border-amber-400/30 bg-amber-500/10 p-4 text-sm text-amber-200">
                {resultRenderError ||
                  (lang === "zh"
                    ? "本次分析低信任，暂无可用正式关键帧。"
                    : "Low-trust analysis: no usable official keyframes.")}
              </div>
            )}
            <div className="text-center py-6 space-y-2">
              <p className="text-xs text-white/40">
                {lang === "zh"
                  ? "每页仅一次 Pro 分析；要看新视频请刷新本页或重新打开 /pro。"
                  : "One Pro analysis per page load; refresh or reopen /pro for a new video."}
              </p>
              <button
                type="button"
                onClick={() => {
                  window.location.reload();
                }}
                className="btn-pro"
              >
                {lang === "en" ? "Refresh page to analyze again" : "刷新页面后再次分析"}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
