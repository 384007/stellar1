"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import KeyframeStrip from "@/components/KeyframeStrip";
import HUDOverlay from "@/components/HUDOverlay";
import Skeleton3DViewer from "@/components/Skeleton3DViewer";
import ProComparison from "@/components/ProComparison";
import SimAnimation from "@/components/SimAnimation";
import PlusResultView, { type PlusAnalysisResult } from "@/components/PlusResultView";
import VideoAnalysisOverlay from "@/components/VideoAnalysisOverlay";
import { coachingTipsFromParsed } from "@/lib/video-analysis-coaching";
import {
  getAnalysisVideoBlob,
  saveAnalysisVideo,
  getAnalysisDetail,
  saveAnalysisDetail,
  deleteAnalysisDetail,
  deleteAnalysisVideo,
} from "@/lib/video-store";
import ShareButton from "@/components/ShareButton";
import { normalizePoseFramesForOverlay } from "@/lib/analysis-pose-storage";
import { rawBase64ImagePayload } from "@/lib/image-base64";
import {
  DEFAULT_HISTORY_RETENTION_DAYS,
  pruneLocalStellarHistoryRecords,
} from "@/lib/pro-history-retention";
import {
  blobToDataUrl,
  buildOfflineAnalysisHtml,
  buildOfflineLabHtml,
  downloadOfflineHtml,
} from "@/lib/offline-analysis-html";
import {
  queueReanalyzeFromHistory,
  type ReanalyzeFromHistoryPayload,
} from "@/lib/reanalyze-from-history";
import { expandStellarProForUi, proExpandedToPlusViewModel } from "@/lib/stellar-pro-result";

function isFiniteAnalysisScore(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

function recordHasFiniteTotalScore(r: AnalysisRecord): r is AnalysisRecord & { total_score: number } {
  return isFiniteAnalysisScore(r.total_score);
}

/** Pro v3：历史 `result_json` 是否标记为屏幕模式（再次分析时回传 `screen_mode`）。 */
function prov3ScreenModeFromHistoryRecord(rec: AnalysisRecord): boolean {
  if (rec.type === "lab") return false;
  try {
    const p = JSON.parse(rec.result_json || "{}") as { screen_mode?: unknown };
    return p.screen_mode === true || p.screen_mode === "true";
  } catch {
    return false;
  }
}

/** D1 list rows are compacted (~90k cap); keyframe JPEGs often live only in R2. Detect stripped/poisoned cache. */
function plusKeyframesMissingImages(parsed: ParsedResult): boolean {
  const kfs = parsed.keyframes;
  if (!Array.isArray(kfs) || kfs.length === 0) return true;
  const withImg = kfs.filter((k) => {
    const b = (k as { image_base64?: string }).image_base64;
    if (typeof b !== "string") return false;
    return rawBase64ImagePayload(b).length > 400;
  }).length;
  return withImg < Math.min(6, kfs.length);
}

/** Prefer merged `result_json` with real keyframe bitmaps over longer text-only local rows (localStorage strips JPEGs). */
function keyframeImagePayloadScore(json: string | null | undefined): number {
  if (!json) return 0;
  try {
    const p = JSON.parse(json) as { keyframes?: unknown[] };
    const kfs = p.keyframes;
    if (!Array.isArray(kfs)) return 0;
    let score = 0;
    for (const k of kfs) {
      if (!k || typeof k !== "object" || Array.isArray(k)) continue;
      const b = (k as { image_base64?: string }).image_base64;
      if (typeof b !== "string") continue;
      score += rawBase64ImagePayload(b).length;
    }
    return score;
  } catch {
    return 0;
  }
}

interface AnalysisRecord {
  id: string;
  type: string;
  video_url?: string;
  video_r2_key?: string;
  result_r2_key?: string;
  total_score: number | null;
  result_json: string;
  created_at: string;
}

interface TrendPoint {
  id: string;
  type: string;
  total_score: number | null;
  created_at: string;
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
    image_base64: string;
  }>;
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
  /** Pro / PlusResultView：屏幕模式分析 */
  screen_mode?: boolean;
}

interface UserInfo {
  username: string;
  email: string;
  is_pro: boolean;
  joined: string;
}

type Tab = "profile" | "history";

function safeExportFileId(id: string) {
  return (id || "record").replace(/[^a-zA-Z0-9._-]+/g, "_").slice(0, 64);
}

interface CurveFrame {
  angles: Record<string, number>;
  timestamp: number;
}

/** SVG-based training curve — no canvas, no useEffect, no layout timing issues */
function ProTrainingCurve({
  frames,
  keyframes,
  lang,
}: {
  frames: CurveFrame[];
  keyframes?: ParsedResult["keyframes"];
  lang: "en" | "zh";
}) {
  if (frames.length < 2) return null;

  // Fixed viewBox coordinate space — SVG scales naturally to 100% width
  const VW = 1000;
  const VH = 220;
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
      const p0 = series[i];
      const p1 = series[i + 1];
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
      const matched = keyframes.find((k) =>
        phase.keys.includes((k.phase || "").toLowerCase())
      );
      if (matched) {
        const ts = matched.timestamp ?? 0;
        let best = 0;
        let bestDiff = Infinity;
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

      <svg
        viewBox={`0 0 ${VW} ${VH}`}
        style={{ width: "100%", height: "auto", display: "block" }}
        preserveAspectRatio="none"
      >
        {/* Grid lines + Y labels */}
        {[0, 1, 2, 3, 4].map((i) => {
          const val = minV + (range * i) / 4;
          const y = pad.top + chartH - (chartH * i) / 4;
          return (
            <g key={i}>
              <line
                x1={pad.left} y1={y} x2={VW - pad.right} y2={y}
                stroke="rgba(255,255,255,0.08)" strokeWidth="1"
              />
              <text
                x={pad.left - 5} y={y + 4}
                fill="rgba(255,255,255,0.35)" fontSize="18" textAnchor="end"
              >
                {val.toFixed(0)}
              </text>
            </g>
          );
        })}

        {/* Series paths */}
        <path d={smoothPath(xFactor)}  stroke="#d4af37" strokeWidth="3" fill="none" strokeLinecap="round" />
        <path d={smoothPath(shoulder)} stroke="#a855f7" strokeWidth="3" fill="none" strokeLinecap="round" />
        <path d={smoothPath(hip)}      stroke="#06b6d4" strokeWidth="3" fill="none" strokeLinecap="round" />
        <path d={smoothPath(spine)}    stroke="#22c55e" strokeWidth="3" fill="none" strokeLinecap="round" />

        {/* Phase markers */}
        {phaseMarkers.map((m, i) => {
          const yDot = toY(xFactor[m.idx]?.y ?? 0);
          const labelY = pad.top + chartH + (i % 2 === 0 ? 20 : 32);
          return (
            <g key={i}>
              <line
                x1={m.x} y1={pad.top} x2={m.x} y2={pad.top + chartH}
                stroke="rgba(245,197,24,0.2)" strokeWidth="1.5" strokeDasharray="6 5"
              />
              <circle cx={m.x} cy={yDot} r="5" fill="#f5c518" stroke="rgba(255,255,255,0.7)" strokeWidth="1.5" />
              <text
                x={m.x} y={labelY}
                fill="rgba(255,255,255,0.55)" fontSize="17" textAnchor="middle"
              >
                {m.label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export default function HistoryPage() {
  const router = useRouter();
  const [records, setRecords] = useState<AnalysisRecord[]>([]);
  const [trend, setTrend] = useState<TrendPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [recordVideos, setRecordVideos] = useState<Record<string, string>>({});
  const [videoLoading, setVideoLoading] = useState<Record<string, boolean>>({});
  const [videoProgress, setVideoProgress] = useState<Record<string, number>>({});
  const [detailLoading, setDetailLoading] = useState<Record<string, boolean>>({});
  const [recordDetails, setRecordDetails] = useState<Record<string, ParsedResult>>({});
  const [lang, setLang] = useState<"en" | "zh">("zh");
  const [tab, setTab] = useState<Tab>("history");
  const [user, setUser] = useState<UserInfo | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [exportingId, setExportingId] = useState<string | null>(null);
  /** Always call latest fetchHistory from visibility/pageshow (avoids stale closures). */
  const fetchHistoryRef = useRef<(token: string) => Promise<void>>(async () => undefined);

  useEffect(() => {
    const token = localStorage.getItem("stellar_token");
    if (!token) {
      router.push("/login");
      return;
    }

    // Restore cached user profile from localStorage (instant)
    const stored = localStorage.getItem("stellar_user");
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        setUser({
          username: parsed.username || (token.startsWith("local-") ? "本地用户" : ""),
          email: parsed.email || "",
          is_pro: !!parsed.is_pro,
          joined: parsed.created_at || new Date().toISOString(),
        });
      } catch { /* ignore */ }
    }

    // Show local records immediately so the page is never stuck on "加载中"
    loadLocalRecords();

    if (token.startsWith("local-")) return;

    // JWT users: fetch fresh profile + server history in background
    fetch("/api/user", { headers: { Authorization: `Bearer ${token}` } })
      .then(r => {
        if (r.status === 401) {
          localStorage.removeItem("stellar_token");
          router.push("/login");
          return null;
        }
        return r.ok ? r.json() : null;
      })
      .then(data => {
        if (data?.user) {
          setUser({
            username: data.user.username || "",
            email: data.user.email || "",
            is_pro: !!data.user.is_pro,
            joined: data.user.created_at || new Date().toISOString(),
          });
          localStorage.setItem("stellar_user", JSON.stringify({
            user_id: data.user.id,
            email: data.user.email,
            username: data.user.username,
            is_pro: data.user.is_pro,
            is_guest: false,
          }));
        }
      })
      .catch(() => { /* local profile already loaded */ });
    fetchHistory(token);
  }, [router]); // eslint-disable-line react-hooks/exhaustive-deps

  function readLocalRecords(): AnalysisRecord[] {
    try {
      const raw = localStorage.getItem("stellar_history_local");
      if (!raw) return [];
      const parsed = JSON.parse(raw) as unknown;
      if (!Array.isArray(parsed)) return [];
      return parsed as AnalysisRecord[];
    } catch {
      return [];
    }
  }

  function loadLocalRecords() {
    pruneLocalStellarHistoryRecords();
    const localRecords = readLocalRecords();
    const sorted = [...localRecords].sort((a, b) =>
      new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    );
    setRecords(sorted);
    setTrend(sorted.map((r) => ({
      id: r.id, type: r.type, total_score: r.total_score, created_at: r.created_at,
    })));
    setLoading(false);
  }

  /**
   * Merge server records with local records.
   *
   * For records that exist in both, keep the RICHER result_json (longer
   * string = more keyframes/skeleton data).  Server's compact version
   * (from compactResultForStorage) often has fewer keyframes — using it
   * would make photos disappear after the server fetch replaces local data.
   */
  function mergeWithLocal(serverRecords: AnalysisRecord[]): AnalysisRecord[] {
    const localRecords = readLocalRecords();
    const localMap = new Map(localRecords.map(r => [r.id, r]));
    const serverIds = new Set<string>();

    const merged = serverRecords.map(sr => {
      serverIds.add(sr.id);
      const local = localMap.get(sr.id);
      if (!local) return sr;
      const out = { ...sr };
      const sj = sr.result_json || "";
      const lj = local.result_json || "";
      const serverImg = keyframeImagePayloadScore(sj);
      const localImg = keyframeImagePayloadScore(lj);
      if (localImg > serverImg) {
        out.result_json = local.result_json;
      } else if (serverImg > localImg) {
        out.result_json = sr.result_json;
      } else if (lj.length > sj.length) {
        out.result_json = local.result_json;
      }
      // Keep server video when set; otherwise use local R2 key / https URL (fixes "video missing" after partial sync).
      if (!(sr.video_r2_key || "").trim() && (local.video_r2_key || "").trim()) {
        out.video_r2_key = local.video_r2_key;
      }
      const lu = (local.video_url || "").trim();
      if (!(sr.video_url || "").trim() && lu && /^https?:\/\//i.test(lu) && !lu.toLowerCase().startsWith("blob:")) {
        out.video_url = local.video_url;
      }
      return out;
    });

    const localOnly = localRecords.filter(r => !serverIds.has(r.id));
    return [...merged, ...localOnly].sort((a, b) =>
      new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    );
  }

  async function syncLocalRecords(token: string): Promise<number> {
    try {
      pruneLocalStellarHistoryRecords();
      const raw = localStorage.getItem("stellar_history_local");
      if (!raw) return 0;
      const parsed = JSON.parse(raw) as unknown;
      if (!Array.isArray(parsed)) return 0;
      const localRecords = parsed as Array<Record<string, unknown>>;
      const unsynced = localRecords.filter((r) => !r._synced);
      if (unsynced.length === 0) return 0;

      const res = await fetch("/api/history", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ records: unsynced }),
      });

      if (res.ok) {
        const data = await res.json();
        const synced = typeof data.synced === "number" ? data.synced : 0;
        const attempted = unsynced.length;
        // 仅当本批全部写入成功才标记已同步，否则下次进页会重试（避免假同步、多端列表不一致）
        if (synced > 0 && synced === attempted) {
          const unsyncedIds = new Set(
            unsynced.map((r) => String(r.id ?? (r as { analysis_id?: string }).analysis_id ?? "")).filter(Boolean),
          );
          const updated = localRecords.map((r) => {
            const rid = String(r.id ?? (r as { analysis_id?: string }).analysis_id ?? "");
            if (unsyncedIds.has(rid)) return { ...r, _synced: true };
            return r;
          });
          localStorage.setItem("stellar_history_local", JSON.stringify(updated));
        }
        return synced;
      }
    } catch {
      // sync failed silently — will retry next page load
    }
    return 0;
  }

  function removeFromLocalStorageHistory(analysisId: string) {
    try {
      const raw = localStorage.getItem("stellar_history_local");
      if (!raw) return;
      const parsed = JSON.parse(raw) as unknown;
      if (!Array.isArray(parsed)) return;
      const arr = parsed as Array<{ id: string }>;
      localStorage.setItem(
        "stellar_history_local",
        JSON.stringify(arr.filter((r) => r.id !== analysisId)),
      );
    } catch {
      /* ignore */
    }
  }

  async function deleteHistoryRecord(rec: AnalysisRecord) {
    if (rec.type === "lab") return;
    const msg =
      lang === "zh"
        ? "确定删除这条分析？云端视频与报告将一并删除，且不可恢复。"
        : "Delete this analysis? Video and report will be removed from the cloud. This cannot be undone.";
    if (!window.confirm(msg)) return;

    setDeletingId(rec.id);
    try {
      const token = localStorage.getItem("stellar_token");
      if (token && !token.startsWith("local-")) {
        const res = await fetch(`/api/history/${encodeURIComponent(rec.id)}`, {
          method: "DELETE",
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok && res.status !== 404) {
          const j = (await res.json().catch(() => ({}))) as { detail?: string };
          window.alert(
            lang === "zh"
              ? `删除失败：${j.detail || res.status}`
              : `Delete failed: ${j.detail || res.status}`,
          );
          return;
        }
      }

      removeFromLocalStorageHistory(rec.id);
      await deleteAnalysisVideo(rec.id).catch(() => {});
      await deleteAnalysisDetail(rec.id).catch(() => {});

      const vsrc = recordVideos[rec.id];
      if (vsrc?.startsWith("blob:")) {
        try {
          URL.revokeObjectURL(vsrc);
        } catch {
          /* ignore */
        }
      }

      setRecordVideos((prev) => {
        const next = { ...prev };
        delete next[rec.id];
        return next;
      });
      setRecordDetails((prev) => {
        const next = { ...prev };
        delete next[rec.id];
        return next;
      });
      setExpandedId((e) => (e === rec.id ? null : e));

      setRecords((prev) => {
        const next = prev.filter((r) => r.id !== rec.id);
        const nonLab = next.filter((r) => r.type !== "lab");
        setTrend(
          [...nonLab]
            .sort(
              (a, b) =>
                new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
            )
            .slice(-30)
            .map((r) => ({
              id: r.id,
              type: r.type,
              total_score: r.total_score,
              created_at: r.created_at,
            })),
        );
        return next;
      });
    } finally {
      setDeletingId(null);
    }
  }

  async function downloadRecordExport(rec: AnalysisRecord) {
    if (rec.type === "lab") {
      const html = buildOfflineLabHtml({
        lang,
        id: rec.id,
        created_at: rec.created_at,
      });
      downloadOfflineHtml(`stellar-lab-${safeExportFileId(rec.id)}.html`, html);
      return;
    }

    setExportingId(rec.id);
    try {
      /** 仅使用本机已有数据，导出后打开 HTML 不依赖网络 / R2 / D1 */
      const candidates: Record<string, unknown>[] = [];
      const det = recordDetails[rec.id];
      if (det) candidates.push(det as unknown as Record<string, unknown>);
      const idbJson = await getAnalysisDetail(rec.id).catch(() => null);
      if (idbJson) {
        try {
          candidates.push(JSON.parse(idbJson) as Record<string, unknown>);
        } catch {
          /* ignore */
        }
      }
      candidates.push(parseResult(rec.result_json) as unknown as Record<string, unknown>);

      let best: Record<string, unknown> = {};
      let bestLen = 0;
      for (const c of candidates) {
        const len = JSON.stringify(c).length;
        if (len > bestLen) {
          best = c;
          bestLen = len;
        }
      }

      let videoDataUrl: string | null = null;
      const vBlob = await getAnalysisVideoBlob(rec.id).catch(() => null);
      if (vBlob && vBlob.size > 0) {
        videoDataUrl = await blobToDataUrl(vBlob);
      }

      const html = buildOfflineAnalysisHtml({
        lang,
        record: {
          id: rec.id,
          type: rec.type,
          created_at: rec.created_at,
          total_score: rec.total_score,
        },
        result: best,
        videoDataUrl,
      });
      downloadOfflineHtml(`stellar-report-${safeExportFileId(rec.id)}.html`, html);
    } finally {
      setExportingId(null);
    }
  }

  function goReanalyzeFromRecord(rec: AnalysisRecord) {
    if (rec.type === "lab") {
      queueReanalyzeFromHistory({ analysisId: rec.id, page: "shot-lab" });
      router.push("/shot-lab");
      return;
    }
    const page: ReanalyzeFromHistoryPayload["page"] =
      rec.type === "plus" ? "plus" : rec.type === "pro" ? "pro" : "analyze";
    const analysisMode: "lite" | "pro" | undefined = page === "analyze" ? "lite" : undefined;
    const vu = (rec.video_url || "").trim();
    queueReanalyzeFromHistory({
      analysisId: rec.id,
      page,
      analysisMode,
      videoUrl: vu && /^https?:\/\//i.test(vu) ? vu : undefined,
      prov3ScreenMode: prov3ScreenModeFromHistoryRecord(rec),
    });
    router.push(page === "plus" ? "/plus" : page === "pro" ? "/pro" : "/analyze");
  }

  async function fetchHistory(token: string) {
    // 必须先等本地未同步条写入 D1，再拉列表，否则手机刚分析完、电脑立刻打开会竞态丢记录
    try {
      await syncLocalRecords(token);
    } catch {
      /* 同步失败仍拉云端已有数据 */
    }

    try {
      const res = await fetch("/api/history?limit=100", {
        headers: { Authorization: `Bearer ${token}` },
        cache: "no-store",
      });
      if (!res.ok) {
        if (res.status === 401) {
          localStorage.removeItem("stellar_token");
          router.push("/login");
          return;
        }
        throw new Error("Failed to fetch");
      }
      const data = await res.json();
      const serverRecords: AnalysisRecord[] = data.analyses || [];

      const labRecords: AnalysisRecord[] = (data.lab_records || []).map(
        (r: { id: string; created_at: string }) => ({
          id: r.id,
          type: "lab",
          video_url: "",
          total_score: 0,
          result_json: "",
          created_at: r.created_at,
        })
      );

      const allServer = [...serverRecords, ...labRecords];
      const merged = mergeWithLocal(allServer);
      setRecords(merged);
      setTrend(data.trend || merged.filter(r => r.type !== "lab").map((r) => ({
        id: r.id, type: r.type, total_score: r.total_score, created_at: r.created_at,
      })));
    } catch {
      // Server unavailable — local records already visible
    }
  }

  fetchHistoryRef.current = fetchHistory;

  useEffect(() => {
    if (typeof window === "undefined") return undefined;

    const refreshFromServer = () => {
      const t = localStorage.getItem("stellar_token");
      if (!t || t.startsWith("local-")) return;
      void fetchHistoryRef.current(t);
    };

    const onVisibility = () => {
      if (document.visibilityState === "visible") refreshFromServer();
    };

    const onPageShow = (e: Event) => {
      if ((e as PageTransitionEvent).persisted) refreshFromServer();
    };

    document.addEventListener("visibilitychange", onVisibility);
    window.addEventListener("pageshow", onPageShow);

    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("pageshow", onPageShow);
    };
  }, []);

  function renderScoreTrendSVG() {
    if (trend.length < 2) return null;
    const trendFinite = trend.filter((t) => isFiniteAnalysisScore(t.total_score));
    if (trendFinite.length < 2) return null;
    const VW = 1000;
    const VH = 220;
    const pad = { top: 20, right: 20, bottom: 35, left: 44 };
    const chartW = VW - pad.left - pad.right;
    const chartH = VH - pad.top - pad.bottom;

    const allScores = trendFinite.map((t) => t.total_score as number);
    const minS = Math.max(0, Math.min(...allScores) - 10);
    const maxS = Math.min(100, Math.max(...allScores) + 10);
    const rangeS = maxS - minS || 1;

    const litePoints: { x: number; y: number }[] = [];
    const proPoints: { x: number; y: number }[] = [];
    const plusPoints: { x: number; y: number }[] = [];

    trendFinite.forEach((t, i) => {
      const s = t.total_score as number;
      const x = pad.left + (chartW * i) / Math.max(1, trendFinite.length - 1);
      const y = pad.top + chartH - ((s - minS) / rangeS) * chartH;
      if (t.type === "pro") proPoints.push({ x, y });
      else if (t.type === "plus") plusPoints.push({ x, y });
      else litePoints.push({ x, y });
    });

    const makeLine = (pts: { x: number; y: number }[]) => {
      if (pts.length < 2) return "";
      let d = `M ${pts[0].x.toFixed(1)} ${pts[0].y.toFixed(1)}`;
      for (let i = 0; i < pts.length - 1; i++) {
        const p0 = pts[i], p1 = pts[i + 1];
        const cx = ((p0.x + p1.x) / 2).toFixed(1);
        const cy = ((p0.y + p1.y) / 2).toFixed(1);
        d += ` Q ${p0.x.toFixed(1)} ${p0.y.toFixed(1)} ${cx} ${cy}`;
      }
      const last = pts[pts.length - 1];
      d += ` L ${last.x.toFixed(1)} ${last.y.toFixed(1)}`;
      return d;
    };

    const makeFill = (pts: { x: number; y: number }[]) => {
      if (pts.length < 2) return "";
      const line = makeLine(pts);
      const bottom = pad.top + chartH;
      return `M ${pts[0].x.toFixed(1)} ${bottom} ${line.slice(1)} L ${pts[pts.length - 1].x.toFixed(1)} ${bottom} Z`;
    };

    const step = Math.max(1, Math.floor(trend.length / 5));
    const xLabels: { x: number; label: string }[] = [];
    for (let i = 0; i < trend.length; i += step) {
      const x = pad.left + (chartW * i) / Math.max(1, trend.length - 1);
      const d = new Date(trend[i].created_at);
      xLabels.push({ x, label: `${d.getMonth() + 1}/${d.getDate()}` });
    }

    const seriesDefs = [
      { pts: litePoints, stroke: "#7c3aed", fill: "rgba(124,58,237,0.15)", id: "lite" },
      { pts: proPoints, stroke: "#d4af37", fill: "rgba(212,175,55,0.15)", id: "pro" },
      { pts: plusPoints, stroke: "#a855f7", fill: "rgba(168,85,247,0.12)", id: "plus" },
    ];

    return (
      <svg viewBox={`0 0 ${VW} ${VH}`} style={{ width: "100%", height: "auto", display: "block" }} preserveAspectRatio="none">
        <defs>
          {seriesDefs.map((s) => (
            <linearGradient key={s.id} id={`trend-grad-${s.id}`} x1="0" y1={pad.top} x2="0" y2={pad.top + chartH} gradientUnits="userSpaceOnUse">
              <stop offset="0%" stopColor={s.fill} />
              <stop offset="100%" stopColor="transparent" />
            </linearGradient>
          ))}
        </defs>

        {/* Grid + Y labels */}
        {[0, 1, 2, 3, 4].map((i) => {
          const val = Math.round(minS + (rangeS * i) / 4);
          const y = pad.top + chartH - (chartH * i) / 4;
          return (
            <g key={i}>
              <line x1={pad.left} y1={y} x2={VW - pad.right} y2={y} stroke="rgba(255,255,255,0.06)" strokeWidth="1" />
              <text x={pad.left - 6} y={y + 5} fill="rgba(255,255,255,0.3)" fontSize="18" textAnchor="end">{val}</text>
            </g>
          );
        })}

        {/* X labels */}
        {xLabels.map((xl, i) => (
          <text key={i} x={xl.x} y={VH - 4} fill="rgba(255,255,255,0.3)" fontSize="17" textAnchor="middle">{xl.label}</text>
        ))}

        {/* Series fills + lines + dots */}
        {seriesDefs.map((s) => (
          <g key={s.id}>
            {s.pts.length >= 2 && <path d={makeFill(s.pts)} fill={`url(#trend-grad-${s.id})`} />}
            {s.pts.length >= 2 && <path d={makeLine(s.pts)} stroke={s.stroke} strokeWidth="3" fill="none" strokeLinecap="round" strokeLinejoin="round" />}
            {s.pts.map((p, i) => <circle key={i} cx={p.x} cy={p.y} r="5" fill={s.stroke} />)}
          </g>
        ))}
      </svg>
    );
  }

  useEffect(() => {
    return () => {
      Object.values(recordVideos).forEach((url) => {
        if (url.startsWith("blob:")) URL.revokeObjectURL(url);
      });
    };
  }, [recordVideos]);

  function parseResult(json: string | null | undefined): ParsedResult {
    try {
      if (!json) return {};
      const r = JSON.parse(json);
      return (r && typeof r === "object") ? r : {};
    } catch {
      return {};
    }
  }

  async function ensureRecordVideoLoaded(record: AnalysisRecord) {
    if (recordVideos[record.id]) return;

    if (record.video_url && /^https?:\/\//.test(record.video_url)) {
      setRecordVideos((prev) => ({ ...prev, [record.id]: record.video_url as string }));
      return;
    }

    setVideoLoading((prev) => ({ ...prev, [record.id]: true }));
    setVideoProgress((prev) => ({ ...prev, [record.id]: 0 }));

    // 1. Try local IndexedDB cache (instant)
    try {
      const blob = await getAnalysisVideoBlob(record.id);
      if (blob && blob.size > 0) {
        const url = URL.createObjectURL(blob);
        setRecordVideos((prev) => ({ ...prev, [record.id]: url }));
        setVideoLoading((prev) => ({ ...prev, [record.id]: false }));
        return;
      }
    } catch { /* ignore */ }

    // 2. Download via XHR with progress, then cache to IndexedDB
    const token = localStorage.getItem("stellar_token");
    if (token && !token.startsWith("local-")) {
      try {
        const blob = await new Promise<Blob>((resolve, reject) => {
          const xhr = new XMLHttpRequest();
          xhr.open("GET", `/api/history/video/${encodeURIComponent(record.id)}?token=${encodeURIComponent(token)}`, true);
          xhr.responseType = "blob";
          xhr.timeout = 120_000;
          xhr.onprogress = (e) => {
            if (e.lengthComputable) {
              setVideoProgress((prev) => ({ ...prev, [record.id]: Math.round((e.loaded / e.total) * 100) }));
            }
          };
          xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300 && xhr.response) {
              resolve(xhr.response as Blob);
            } else {
              reject(new Error(`HTTP ${xhr.status}`));
            }
          };
          xhr.onerror = () => reject(new Error("network"));
          xhr.ontimeout = () => reject(new Error("timeout"));
          xhr.send();
        });

        if (blob.size > 0) {
          // Cache to IndexedDB for next time
          try { await saveAnalysisVideo(record.id, blob, `${record.id}.mp4`); } catch { /* non-fatal */ }
          const url = URL.createObjectURL(blob);
          setRecordVideos((prev) => ({ ...prev, [record.id]: url }));
          setVideoLoading((prev) => ({ ...prev, [record.id]: false }));
          return;
        }
      } catch { /* download failed */ }
    }

    setVideoLoading((prev) => ({ ...prev, [record.id]: false }));
    setVideoProgress((prev) => ({ ...prev, [record.id]: -1 }));
  }

  async function ensureRecordDetailLoaded(record: AnalysisRecord) {
    const recordId = record.id;

    /** In-memory detail from a prior expand must not block reload when keyframes lack real bitmaps. */
    if (recordDetails[recordId]) {
      if (!plusKeyframesMissingImages(recordDetails[recordId])) {
        return;
      }
      setRecordDetails((prev) => {
        const next = { ...prev };
        delete next[recordId];
        return next;
      });
    }

    const token = localStorage.getItem("stellar_token");
    if (!token || token.startsWith("local-")) return;

    setDetailLoading((prev) => ({ ...prev, [recordId]: true }));

    // 1. IndexedDB — same validation as R2 path: compact rows often omit JPEGs; stale cache must not win over D1/R2.
    try {
      const cached = await getAnalysisDetail(recordId);
      if (cached) {
        const parsed = parseResult(cached);
        if (!plusKeyframesMissingImages(parsed)) {
          setRecordDetails((prev) => ({ ...prev, [recordId]: parsed }));
          setDetailLoading((prev) => ({ ...prev, [recordId]: false }));
          return;
        }
        await deleteAnalysisDetail(recordId).catch(() => {});
      }
    } catch { /* ignore */ }

    // 2. Fetch from API (GET merges result_r2_key → full JSON when R2 object exists)
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const ac = new AbortController();
        const timer = setTimeout(() => ac.abort(), 30_000);
        const res = await fetch(`/api/history/${encodeURIComponent(recordId)}`, {
          headers: { Authorization: `Bearer ${token}` },
          signal: ac.signal,
        });
        clearTimeout(timer);
        if (!res.ok) { setDetailLoading((prev) => ({ ...prev, [recordId]: false })); return; }
        const data = await res.json();
        const rawJson = typeof data.result_json === "string" ? data.result_json : "{}";
        const parsed = parseResult(rawJson);
        setRecordDetails((prev) => ({ ...prev, [recordId]: parsed }));
        try { await saveAnalysisDetail(recordId, rawJson); } catch { /* non-fatal */ }
        setDetailLoading((prev) => ({ ...prev, [recordId]: false }));
        return;
      } catch {
        if (attempt === 0) await new Promise((r) => setTimeout(r, 2000));
      }
    }
    setDetailLoading((prev) => ({ ...prev, [recordId]: false }));
  }

  function getUserAngles(parsed: ParsedResult): Record<string, number> | undefined {
    const midIndex = parsed.pose_frames ? Math.floor(parsed.pose_frames.length / 2) : -1;
    const mid = midIndex >= 0 ? parsed.pose_frames?.[midIndex] : undefined;
    return mid?.angles ?? parsed.pose_frames?.[0]?.angles;
  }

  function buildCurveFrames(parsed: ParsedResult, totalScoreFallback: number | null | undefined = 0): CurveFrame[] {
    const scoreFb =
      typeof totalScoreFallback === "number" && Number.isFinite(totalScoreFallback) ? totalScoreFallback : 0;
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
          const xFactor = toNum(stats.x_factor);
          const shoulder = toNum(stats.shoulder_rotation);
          const hip = toNum(stats.hip_rotation);
          const spine = toNum(stats.spine_tilt);
          if (xFactor !== undefined) angles.x_factor = xFactor;
          if (shoulder !== undefined) angles.shoulder_rotation = shoulder;
          if (hip !== undefined) angles.hip_rotation = hip;
          if (spine !== undefined) angles.spine_tilt = spine;
          return { angles, timestamp: i * 33 };
        })
        .filter((f) => Object.keys(f.angles).length > 0);
      if (frames.length > 1) return frames;
    }

    // Synthetic curve from scores or total_score — always produces data
    const s = parsed.scores || {};
    const stance = s.stance || scoreFb || 60;
    const backswing = s.backswing || scoreFb || 60;
    const downswing = s.downswing || scoreFb || 60;
    const follow = s.follow_through || scoreFb || 60;
    return [
      { timestamp: 0, angles: { x_factor: 8, shoulder_rotation: -4, hip_rotation: -2, spine_tilt: 11 } },
      { timestamp: 1, angles: { x_factor: backswing * 0.36, shoulder_rotation: -backswing * 0.35, hip_rotation: -stance * 0.16, spine_tilt: 10.5 } },
      { timestamp: 2, angles: { x_factor: backswing * 0.5, shoulder_rotation: -backswing * 0.52, hip_rotation: -stance * 0.28, spine_tilt: 10 } },
      { timestamp: 3, angles: { x_factor: downswing * 0.3, shoulder_rotation: -downswing * 0.24, hip_rotation: -downswing * 0.4, spine_tilt: 9 } },
      { timestamp: 4, angles: { x_factor: downswing * 0.46, shoulder_rotation: downswing * 0.12, hip_rotation: -downswing * 0.52, spine_tilt: 8 } },
      { timestamp: 5, angles: { x_factor: follow * 0.24, shoulder_rotation: follow * 0.38, hip_rotation: -follow * 0.42, spine_tilt: 12.5 } },
    ];
  }

  function formatDate(dateStr: string): string {
    const d = new Date(dateStr);
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  }

  const scoredRecords = records.filter((r) => r.type !== "lab");
  const numericScored = scoredRecords.filter(recordHasFiniteTotalScore);
  const bestScore = numericScored.length > 0 ? Math.max(...numericScored.map((r) => r.total_score)) : 0;
  const avgScore =
    numericScored.length > 0
      ? Math.round(numericScored.reduce((s, r) => s + r.total_score, 0) / numericScored.length)
      : 0;
  const latestRecord = scoredRecords[0];
  const latestScore = latestRecord && isFiniteAnalysisScore(latestRecord.total_score) ? latestRecord.total_score : 0;

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-white/40 text-sm">{lang === "zh" ? "加载中..." : "Loading..."}</div>
      </div>
    );
  }

  const totalSessions = records.length;
  const proSessions = records.filter((r) => r.type === "pro").length;
  const plusSessions = records.filter((r) => r.type === "plus").length;
  const labSessions = records.filter((r) => r.type === "lab").length;
  const liteSessions = totalSessions - proSessions - plusSessions - labSessions;

  return (
    <div className="min-h-screen">
      <nav className="sticky top-0 z-50 border-b border-white/5 bg-brand-dark/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
          <a href="/" className="flex items-center gap-2">
            <img src="/logo.svg" alt="Stellar" className="h-8 w-8" />
            <span className="text-xl font-bold text-brand-gold">STELLAR</span>
          </a>
          <div className="flex items-center gap-2">
            <a href="/analyze"
              className="rounded-lg border border-white/10 px-3 py-1 text-xs text-white/60 transition hover:text-white">
              {lang === "zh" ? "开始分析" : "Analyze"}
            </a>
            <button onClick={() => setLang(lang === "en" ? "zh" : "en")}
              className="rounded-lg border border-white/10 px-3 py-1 text-xs text-white/60 transition hover:text-white">
              {lang === "en" ? "中文" : "EN"}
            </button>
          </div>
        </div>
      </nav>

      <div className="mx-auto max-w-4xl px-4 py-8">
        {/* User Profile Header */}
        {user && (
          <div className="glass-card mb-6 p-6">
            <div className="flex items-center gap-4">
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-gradient-to-br from-brand-purple to-brand-gold text-2xl font-bold text-white">
                {(user.username || user.email || "U").charAt(0).toUpperCase()}
              </div>
              <div className="flex-1">
                <h1 className="text-xl font-bold text-white">{user.username || user.email}</h1>
                <div className="mt-1 flex flex-wrap items-center gap-3 text-xs text-white/40">
                  {user.email && user.username && (
                    <span>{user.email}</span>
                  )}
                  <span className={`rounded px-1.5 py-0.5 font-bold ${
                    user.is_pro
                      ? "bg-brand-gold/20 text-brand-gold"
                      : "bg-brand-purple/20 text-brand-purple"
                  }`}>
                    {user.is_pro ? "PRO" : "FREE"}
                  </span>
                </div>
              </div>
            </div>
            <div className="mt-5 grid grid-cols-4 gap-3 border-t border-white/5 pt-5">
              <div className="text-center">
                <p className="text-lg font-bold text-white">{totalSessions}</p>
                <p className="text-[10px] text-white/40">{lang === "zh" ? "总分析" : "Total"}</p>
              </div>
              <div className="text-center">
                <p className="text-lg font-bold text-brand-gold">{bestScore}</p>
                <p className="text-[10px] text-white/40">{lang === "zh" ? "最佳" : "Best"}</p>
              </div>
              <div className="text-center">
                <p className="text-lg font-bold text-brand-purple">{avgScore}</p>
                <p className="text-[10px] text-white/40">{lang === "zh" ? "平均" : "Avg"}</p>
              </div>
              <div className="text-center">
                <p className="text-lg font-bold text-white/70">{proSessions}</p>
                <p className="text-[10px] text-white/40">{lang === "zh" ? "Pro分析" : "Pro"}</p>
              </div>
              {plusSessions > 0 && (
                <div className="text-center">
                  <p className="text-lg font-bold bg-gradient-to-r from-brand-purple to-brand-gold bg-clip-text text-transparent">{plusSessions}</p>
                  <p className="text-[10px] text-white/40">Plus</p>
                </div>
              )}
              {labSessions > 0 && (
                <div className="text-center">
                  <p className="text-lg font-bold text-cyan-400">{labSessions}</p>
                  <p className="text-[10px] text-white/40">Lab</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tabs */}
        <div className="mb-6 flex gap-1 rounded-lg bg-white/[0.03] p-1">
          {([
            { key: "profile" as Tab, zh: "训练概览", en: "Overview" },
            { key: "history" as Tab, zh: "分析记录", en: "Records" },
          ]).map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`flex-1 rounded-md px-4 py-2 text-sm font-medium transition ${
                tab === t.key
                  ? "bg-white/10 text-white shadow"
                  : "text-white/40 hover:text-white/60"
              }`}
            >
              {lang === "zh" ? t.zh : t.en}
            </button>
          ))}
        </div>

        <div
          role="status"
          className="mb-5 rounded-xl border border-amber-500/30 bg-amber-500/[0.07] px-4 py-3 text-[13px] leading-relaxed text-amber-100/90"
        >
          {lang === "zh" ? (
            <>
              <strong className="text-amber-200">数据保留 {DEFAULT_HISTORY_RETENTION_DAYS} 天</strong>
              （云端 D1 与 R2）：Lite、Pro、Plus、Shot Lab 记录在创建满 {DEFAULT_HISTORY_RETENTION_DAYS} 天后会自动删除，含视频与报告 JSON；分享链接随之失效。
              本机列表也会按同一期限清理。使用<strong className="text-amber-200">同一账号</strong>登录手机与电脑即可查看同步记录。
            </>
          ) : (
            <>
              <strong className="text-amber-200">{DEFAULT_HISTORY_RETENTION_DAYS}-day cloud retention</strong>{" "}
              (D1 and R2): Lite, Pro, Plus, and Shot Lab rows are removed after {DEFAULT_HISTORY_RETENTION_DAYS} days, including video and report JSON; share links expire.
              On-device lists follow the same window. Use the <strong className="text-amber-200">same account</strong> on phone and desktop to keep history in sync.
            </>
          )}
        </div>

        {tab === "profile" && (
          <>
            {/* Stats Cards */}
            {records.length > 0 && (
              <>
                <div className="mb-4 grid grid-cols-3 gap-3">
                  <div className="glass-card p-4 text-center">
                    <p className="text-[10px] text-white/40 uppercase tracking-wider">
                      {lang === "zh" ? "最近得分" : "Latest"}
                    </p>
                    <p className="mt-1 text-2xl font-bold text-brand-purple">{latestScore}</p>
                  </div>
                  <div className="glass-card p-4 text-center">
                    <p className="text-[10px] text-white/40 uppercase tracking-wider">
                      {lang === "zh" ? "最佳得分" : "Best"}
                    </p>
                    <p className="mt-1 text-2xl font-bold text-brand-gold">{bestScore}</p>
                  </div>
                  <div className="glass-card p-4 text-center">
                    <p className="text-[10px] text-white/40 uppercase tracking-wider">
                      {lang === "zh" ? "平均得分" : "Average"}
                    </p>
                    <p className="mt-1 text-2xl font-bold text-white/70">{avgScore}</p>
                  </div>
                </div>

                {/* Trend Chart */}
                {trend.length >= 2 && (
                  <div className="glass-card mb-6 p-5">
                    <div className="mb-3 flex items-center justify-between">
                      <h3 className="text-sm font-semibold text-white">
                        {lang === "zh" ? "分数趋势" : "Score Trend"}
                      </h3>
                      <div className="flex items-center gap-3 text-[10px]">
                        <span className="flex items-center gap-1">
                          <span className="inline-block h-2 w-2 rounded-full bg-brand-purple" />
                          <span className="text-white/40">Lite</span>
                        </span>
                        <span className="flex items-center gap-1">
                          <span className="inline-block h-2 w-2 rounded-full bg-brand-gold" />
                          <span className="text-white/40">Pro</span>
                        </span>
                        {plusSessions > 0 && (
                          <span className="flex items-center gap-1">
                            <span className="inline-block h-2 w-2 rounded-full bg-purple-500" />
                            <span className="text-white/40">Plus</span>
                          </span>
                        )}
                      </div>
                    </div>
                    {renderScoreTrendSVG()}
                  </div>
                )}

                {/* Latest Swing Training Curve (always visible in Overview) */}
                {(() => {
                  const latest = records.find((r) => r.type !== "plus");
                  if (!latest) return null;
                  const latestParsed = parseResult(latest.result_json);
                  const latestCurve = buildCurveFrames(latestParsed, latest.total_score);
                  return (
                    <div className="mb-6">
                      <ProTrainingCurve frames={latestCurve} keyframes={latestParsed.keyframes} lang={lang} />
                    </div>
                  );
                })()}

                {/* Training Summary */}
                <div className="glass-card p-5">
                  <h3 className="mb-3 text-sm font-semibold text-white">
                    {lang === "zh" ? "训练统计" : "Training Stats"}
                  </h3>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-white/50">{lang === "zh" ? "Lite 分析次数" : "Lite Sessions"}</span>
                      <span className="font-medium text-brand-purple">{liteSessions}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-white/50">{lang === "zh" ? "Pro 分析次数" : "Pro Sessions"}</span>
                      <span className="font-medium text-brand-gold">{proSessions}</span>
                    </div>
                    {plusSessions > 0 && (
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-white/50">{lang === "zh" ? "Plus 分析次数" : "Plus Sessions"}</span>
                        <span className="font-medium text-purple-400">{plusSessions}</span>
                      </div>
                    )}
                    {labSessions > 0 && (
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-white/50">{lang === "zh" ? "Shot Lab 次数" : "Shot Lab Sessions"}</span>
                        <span className="font-medium text-cyan-400">{labSessions}</span>
                      </div>
                    )}
                    {records.length > 0 && (
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-white/50">{lang === "zh" ? "最近分析" : "Last Analysis"}</span>
                        <span className="font-medium text-white/60">{formatDate(records[0].created_at)}</span>
                      </div>
                    )}
                    {bestScore > 0 && (
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-white/50">{lang === "zh" ? "进步幅度" : "Improvement"}</span>
                        <span className={`font-medium ${
                          latestScore >= avgScore ? "text-green-400" : "text-red-400"
                        }`}>
                          {latestScore >= avgScore ? "+" : ""}{latestScore - avgScore} vs avg
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </>
            )}

            {records.length === 0 && (
              <div className="glass-card p-12 text-center">
                <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-2xl bg-white/5">
                  <svg className="h-10 w-10 text-white/20" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
                  </svg>
                </div>
                <h3 className="mb-2 text-lg font-semibold text-white/60">
                  {lang === "zh" ? "暂无训练记录" : "No training records yet"}
                </h3>
                <p className="mb-6 text-sm text-white/30">
                  {lang === "zh"
                    ? "上传挥杆视频开始你的第一次分析"
                    : "Upload a swing video to start your first analysis"}
                </p>
                <a href="/analyze" className="btn-primary inline-block">
                  {lang === "zh" ? "开始分析" : "Start Analysis"}
                </a>
              </div>
            )}
          </>
        )}

        {tab === "history" && records.length === 0 && (
          <div className="glass-card p-12 text-center">
            <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-2xl bg-white/5">
              <svg className="h-10 w-10 text-white/20" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
              </svg>
            </div>
            <h3 className="mb-2 text-lg font-semibold text-white/60">
              {lang === "zh" ? "暂无分析记录" : "No analysis records yet"}
            </h3>
            <p className="mb-6 text-sm text-white/30">
              {lang === "zh"
                ? "上传挥杆视频开始你的第一次分析"
                : "Upload a swing video to start your first analysis"}
            </p>
            <a href="/analyze" className="btn-primary inline-block">
              {lang === "zh" ? "开始分析" : "Start Analysis"}
            </a>
          </div>
        )}

        {tab === "history" && records.length > 0 && (
          <>
            <p className="mb-4 text-sm text-white/40">
              {lang === "zh"
                ? `共 ${records.length} 条分析记录`
                : `${records.length} analysis records`}
            </p>

            {/* Record List */}
            <div className="space-y-3">
              {records.map((rec) => {
                if (!rec || !rec.id) return null;
                const scores: Record<string, number> = (() => {
                  try { return (JSON.parse(rec.result_json || "{}") as ParsedResult).scores || {}; }
                  catch { return {}; }
                })();
                const isExpanded = expandedId === rec.id;
                const scoreKeys = ["grip", "stance", "backswing", "downswing", "follow_through"];
                const labels: Record<string, { en: string; zh: string }> = {
                  grip: { en: "Grip", zh: "握杆" },
                  stance: { en: "Stance", zh: "站姿" },
                  backswing: { en: "Backswing", zh: "后摆" },
                  downswing: { en: "Downswing", zh: "下杆" },
                  follow_through: { en: "Follow", zh: "收杆" },
                };

                // Shot Lab records: show a minimal card with a link to the Shot Lab page
                if (rec.type === "lab") {
                  return (
                    <div key={rec.id} className="glass-card overflow-hidden">
                      <div className="flex w-full items-center gap-4 p-4">
                        <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-xl bg-cyan-500/10">
                          <svg className="h-6 w-6 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 0 1-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 0 1 4.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0 1 12 15a9.065 9.065 0 0 1-6.23-.693L5 14.5m14.8.8 1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0 1 12 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
                          </svg>
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="rounded px-1.5 py-0.5 text-[10px] font-bold bg-cyan-500/20 text-cyan-400">
                              SHOT LAB
                            </span>
                            <span className="text-xs text-white/30">{formatDate(rec.created_at)}</span>
                          </div>
                          <p className="mt-1 text-xs text-white/40">
                            {lang === "zh" ? "完整报告请在 Shot Lab 页面查看" : "View full report in Shot Lab"}
                          </p>
                        </div>
                        <div className="flex flex-shrink-0 flex-col items-stretch gap-2 sm:flex-row sm:items-center">
                          <button
                            type="button"
                            onClick={() => void downloadRecordExport(rec)}
                            className="rounded-lg border border-white/15 bg-white/[0.06] px-3 py-2 text-xs font-medium text-white/75 transition hover:bg-white/10 touch-manipulation"
                          >
                            {lang === "zh" ? "离线说明" : "Offline note"}
                          </button>
                          <a
                            href="/shot-lab"
                            className="rounded-lg border border-cyan-500/30 bg-cyan-500/10 px-3 py-2 text-center text-xs font-medium text-cyan-400 transition hover:bg-cyan-500/20"
                          >
                            {lang === "zh" ? "复盘" : "Review"}
                          </a>
                          <button
                            type="button"
                            onClick={() => goReanalyzeFromRecord(rec)}
                            className="rounded-lg border border-violet-500/35 bg-violet-500/10 px-3 py-2 text-xs font-medium text-violet-200/90 transition hover:bg-violet-500/15 touch-manipulation"
                          >
                            {lang === "zh" ? "重新分析" : "Re-analyze"}
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                }

                return (
                  <div key={rec.id} className="glass-card overflow-hidden">
                    <div className="flex items-stretch">
                      <button
                        type="button"
                        onClick={() => {
                          if (isExpanded) {
                            setExpandedId(null);
                            return;
                          }
                          setExpandedId(rec.id);
                          ensureRecordVideoLoaded(rec);
                          ensureRecordDetailLoaded(rec);
                        }}
                        className="flex min-w-0 flex-1 items-center gap-4 p-4 text-left transition hover:bg-white/[0.02]"
                      >
                        <div
                          className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-xl"
                          style={{
                            background: `conic-gradient(${rec.type === "pro" ? "#d4af37" : rec.type === "plus" ? "#a855f7" : "#7c3aed"} ${isFiniteAnalysisScore(rec.total_score) ? rec.total_score : 0}%, rgba(255,255,255,0.05) ${isFiniteAnalysisScore(rec.total_score) ? rec.total_score : 0}%)`,
                          }}
                        >
                          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-dark">
                            <span className="text-sm font-bold" style={{ color: rec.type === "pro" ? "#d4af37" : rec.type === "plus" ? "#a855f7" : "#7c3aed" }}>
                              {isFiniteAnalysisScore(rec.total_score) ? rec.total_score : "—"}
                            </span>
                          </div>
                        </div>

                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-2">
                            <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold ${
                              rec.type === "pro"
                                ? "bg-brand-gold/20 text-brand-gold"
                                : rec.type === "plus"
                                ? "bg-purple-500/20 text-purple-400"
                                : "bg-brand-purple/20 text-brand-purple"
                            }`}>
                              {(rec.type || "lite").toUpperCase()}
                            </span>
                            <span className="text-xs text-white/30">{formatDate(rec.created_at)}</span>
                          </div>
                          <div className="mt-1.5 flex gap-1">
                            {scoreKeys.map((key) => (
                              <div key={key} className="flex-1">
                                <div className="h-1 rounded-full bg-white/5">
                                  <div
                                    className="h-full rounded-full"
                                    style={{
                                      width: `${scores[key] || 0}%`,
                                      background: rec.type === "pro" ? "#d4af37" : rec.type === "plus" ? "#a855f7" : "#7c3aed",
                                      opacity: 0.7,
                                    }}
                                  />
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        <svg
                          className={`h-4 w-4 flex-shrink-0 text-white/20 transition-transform ${isExpanded ? "rotate-180" : ""}`}
                          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
                        </svg>
                      </button>
                      <button
                        type="button"
                        disabled={deletingId === rec.id}
                        onClick={() => void deleteHistoryRecord(rec)}
                        className="flex min-h-[52px] min-w-[52px] flex-shrink-0 touch-manipulation items-center justify-center border-l border-white/5 px-2 text-red-400/85 transition hover:bg-red-500/10 hover:text-red-300 disabled:opacity-40"
                        aria-label={lang === "zh" ? "删除记录" : "Delete record"}
                        title={lang === "zh" ? "删除" : "Delete"}
                      >
                        {deletingId === rec.id ? (
                          <span className="inline-block h-5 w-5 animate-spin rounded-full border-2 border-red-400/25 border-t-red-400" />
                        ) : (
                          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
                          </svg>
                        )}
                      </button>
                    </div>

                    {isExpanded && (() => {
                      const listParsed = parseResult(rec.result_json);
                      const listHasPoses = (listParsed.pose_frames?.length ?? 0) > 0;
                      const hasR2Full = !!(rec.result_r2_key && String(rec.result_r2_key).trim());
                      const loadingDetail = detailLoading[rec.id] === true;
                      const detailParsed = recordDetails[rec.id];
                      const awaitingKeyframePayload =
                        plusKeyframesMissingImages(listParsed) &&
                        loadingDetail &&
                        !detailParsed;
                      const waitingSkeletonDetail =
                        hasR2Full &&
                        !listHasPoses &&
                        !detailParsed &&
                        loadingDetail;
                      const parsed = detailParsed || listParsed;
                      const overlayPoses = normalizePoseFramesForOverlay(parsed.pose_frames);
                      const curveFrames = buildCurveFrames(parsed, rec.total_score);
                      const isLoadingData = !recordVideos[rec.id] && videoLoading[rec.id] !== false
                        || detailLoading[rec.id];
                      return (
                      <div className="border-t border-white/5 p-4 animate-fade-in">
                        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                          <button
                            type="button"
                            disabled={exportingId === rec.id}
                            onClick={() => void downloadRecordExport(rec)}
                            className="inline-flex min-h-[44px] touch-manipulation items-center gap-2 rounded-lg border border-emerald-500/35 bg-emerald-500/10 px-4 py-2 text-xs font-medium text-emerald-200/90 transition hover:bg-emerald-500/15 disabled:opacity-40"
                          >
                            {exportingId === rec.id ? (
                              <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-emerald-400/30 border-t-emerald-300" />
                            ) : (
                              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
                              </svg>
                            )}
                            {lang === "zh" ? "下载离线报告（单页 HTML）" : "Download offline report (HTML)"}
                          </button>
                          <ShareButton
                            analysisId={rec.id}
                            score={rec.total_score}
                            type={rec.type}
                            lang={lang}
                          />
                          <button
                            type="button"
                            onClick={() => goReanalyzeFromRecord(rec)}
                            className="inline-flex min-h-[44px] touch-manipulation items-center gap-2 rounded-lg border border-violet-500/35 bg-violet-500/10 px-4 py-2 text-xs font-medium text-violet-200/90 transition hover:bg-violet-500/15"
                          >
                            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 0 13.803 3.7m0 0V19.5m0-4.992v-.01" />
                            </svg>
                            {lang === "zh" ? "重新分析" : "Re-analyze"}
                          </button>
                        </div>

                        {isLoadingData && !recordVideos[rec.id] && (
                          <div className="mb-3 text-center text-xs text-white/40">
                            {lang === "zh" ? "首次加载可能需要几秒钟，加载后会缓存到本地" : "First load may take a few seconds, data will be cached locally"}
                          </div>
                        )}

                        {waitingSkeletonDetail && (
                          <div className="mb-3 rounded-lg border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-center text-[11px] text-amber-200/55">
                            {lang === "zh"
                              ? "正在从云端拉取完整分析数据（含视频叠加骨架）…"
                              : "Loading full analysis (pose overlay) from cloud…"}
                          </div>
                        )}

                        {recordVideos[rec.id] ? (
                          waitingSkeletonDetail ? (
                            <div className="mb-4 flex flex-col items-center justify-center rounded-xl border border-white/10 bg-black/40 py-12">
                              <div className="h-6 w-6 animate-spin rounded-full border-2 border-white/15 border-t-white/50" />
                              <p className="mt-3 px-4 text-center text-xs text-white/45">
                                {lang === "zh" ? "加载骨架叠加数据…" : "Loading pose overlay data…"}
                              </p>
                            </div>
                          ) : overlayPoses.length > 0 ? (
                            <div className="mb-4">
                              <VideoAnalysisOverlay
                                videoSrc={recordVideos[rec.id]}
                                poseFrames={overlayPoses}
                                lang={lang}
                                coachingTips={coachingTipsFromParsed(parsed, rec.type)}
                                prediction={parsed.prediction as { predicted_distance?: number; shot_shape?: string; shot_shape_zh?: string; club_head_speed?: number; club_type?: string; hand?: "R" | "L" | "UNKNOWN" } | undefined}
                                sourceFrameCount={
                                  (parsed as { video_meta?: { source_frame_count?: number } })
                                    .video_meta?.source_frame_count
                                }
                              />
                            </div>
                          ) : (
                            <div className="mb-4 overflow-hidden rounded-xl border border-white/10 bg-black/30">
                              <video
                                className="h-full max-h-[420px] w-full bg-black object-contain"
                                src={recordVideos[rec.id]}
                                controls
                                playsInline
                                preload="metadata"
                              />
                            </div>
                          )
                        ) : videoLoading[rec.id] !== false ? (
                          <div className="mb-4 rounded-xl border border-white/10 bg-white/[0.02] p-4">
                            <div className="flex items-center justify-center">
                              <div className="h-5 w-5 animate-spin rounded-full border-2 border-white/20 border-t-white/60" />
                              <span className="ml-2 text-xs text-white/35">
                                {lang === "zh"
                                  ? `加载视频${(videoProgress[rec.id] ?? 0) > 0 ? ` ${videoProgress[rec.id]}%` : "..."}`
                                  : `Loading video${(videoProgress[rec.id] ?? 0) > 0 ? ` ${videoProgress[rec.id]}%` : "..."}`}
                              </span>
                            </div>
                            {(videoProgress[rec.id] ?? 0) > 0 && (
                              <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
                                <div
                                  className="h-full rounded-full bg-violet-500/70 transition-all duration-300"
                                  style={{ width: `${videoProgress[rec.id]}%` }}
                                />
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="mb-4 rounded-xl border border-white/10 bg-white/[0.02] p-3 text-xs text-white/35">
                            {lang === "zh"
                              ? "该记录未找到原视频。新分析会自动保存视频用于回放。"
                              : "Original video is not available for this record."}
                          </div>
                        )}

                        {rec.type === "plus" ? (
                          awaitingKeyframePayload ? (
                            <div className="mb-4 flex flex-col items-center justify-center rounded-xl border border-white/10 bg-black/30 py-16">
                              <div className="h-8 w-8 animate-spin rounded-full border-2 border-white/15 border-t-violet-400/80" />
                              <p className="mt-4 px-6 text-center text-xs text-white/45">
                                {lang === "zh"
                                  ? "正在加载完整关键帧与报告（来自云端）…"
                                  : "Loading full keyframes and report from cloud…"}
                              </p>
                            </div>
                          ) : (
                            <PlusResultView result={parsed as PlusAnalysisResult} lang={lang} />
                          )
                        ) : rec.type === "pro" ? (
                          awaitingKeyframePayload ? (
                            <div className="mb-4 flex flex-col items-center justify-center rounded-xl border border-white/10 bg-black/30 py-16">
                              <div className="h-8 w-8 animate-spin rounded-full border-2 border-white/15 border-t-brand-gold/80" />
                              <p className="mt-4 px-6 text-center text-xs text-white/45">
                                {lang === "zh"
                                  ? "正在加载完整关键帧与报告（来自云端）…"
                                  : "Loading full keyframes and report from cloud…"}
                              </p>
                            </div>
                          ) : (
                            <PlusResultView
                              result={proExpandedToPlusViewModel(
                                expandStellarProForUi(parsed as Record<string, unknown>),
                              )}
                              lang={lang}
                              externalVideoSrc={recordVideos[rec.id] || undefined}
                            />
                          )
                        ) : (
                          <>
                            {/* Score details */}
                            <div className="mb-4 grid grid-cols-5 gap-2">
                              {scoreKeys.map((key) => (
                                <div key={key} className="text-center">
                                  <p className="text-lg font-bold" style={{ color: rec.type === "pro" ? "#d4af37" : "#7c3aed" }}>
                                    {scores[key] || 0}
                                  </p>
                                  <p className="text-[10px] text-white/40">
                                    {lang === "zh" ? labels[key]?.zh : labels[key]?.en}
                                  </p>
                                </div>
                              ))}
                            </div>

                            {parsed.keyframes && parsed.keyframes.length > 0 && (
                              <div className="mb-4">
                                {awaitingKeyframePayload ? (
                                  <div className="flex flex-col items-center justify-center rounded-xl border border-white/10 bg-black/25 py-10">
                                    <div className="h-6 w-6 animate-spin rounded-full border-2 border-white/15 border-t-brand-purple/70" />
                                    <p className="mt-3 text-center text-[11px] text-white/40">
                                      {lang === "zh" ? "加载关键帧图片…" : "Loading keyframe images…"}
                                    </p>
                                  </div>
                                ) : (
                                  <KeyframeStrip keyframes={parsed.keyframes} lang={lang} />
                                )}
                              </div>
                            )}

                            {parsed.skeleton_data?.frames?.[0] && (
                              <div className="mb-4 rounded-xl border border-white/5 bg-black/20 p-4">
                                <h4 className="mb-3 text-sm font-semibold text-white">
                                  {lang === "zh" ? "骨架 HUD 回放" : "Skeleton HUD Replay"}
                                </h4>
                                <HUDOverlay
                                  hudData={parsed.skeleton_data.frames[0]}
                                  showExtended={rec.type === "pro"}
                                  mode={rec.type === "pro" ? "pro" : "lite"}
                                  lang={lang}
                                />
                              </div>
                            )}

                            {parsed.pose_frames && parsed.pose_frames.length > 0 && (
                              <div className="mb-4">
                                <Skeleton3DViewer frames={parsed.pose_frames} lang={lang} />
                              </div>
                            )}

                            <ProTrainingCurve frames={curveFrames} keyframes={parsed.keyframes} lang={lang} />

                            {/* Summary */}
                            {(parsed.summary_zh || parsed.summary) && (
                              <div className="mb-3 rounded-lg bg-white/[0.02] p-3">
                                <p className="text-xs leading-relaxed text-white/50">
                                  {lang === "zh" ? parsed.summary_zh : parsed.summary}
                                </p>
                              </div>
                            )}

                            {/* Issues */}
                            {((lang === "zh" ? parsed.issues_zh : parsed.issues) || []).length > 0 && (
                              <div className="mb-2">
                                <p className="mb-1 text-[10px] font-semibold text-red-400/70 uppercase tracking-wider">
                                  {lang === "zh" ? "问题" : "Issues"}
                                </p>
                                <ul className="space-y-0.5">
                                  {((lang === "zh" ? parsed.issues_zh : parsed.issues) || []).slice(0, 5).map((issue, i) => (
                                    <li key={i} className="flex items-start gap-1.5 text-xs text-white/40">
                                      <span className="mt-0.5 text-red-400/40">●</span>{issue}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {/* Suggestions */}
                            {((lang === "zh" ? parsed.suggestions_zh : parsed.suggestions) || []).length > 0 && (
                              <div className="mb-4">
                                <p className="mb-1 text-[10px] font-semibold text-brand-gold/70 uppercase tracking-wider">
                                  {lang === "zh" ? "建议" : "Suggestions"}
                                </p>
                                <ul className="space-y-0.5">
                                  {((lang === "zh" ? parsed.suggestions_zh : parsed.suggestions) || []).slice(0, 5).map((sug, i) => (
                                    <li key={i} className="flex items-start gap-1.5 text-xs text-white/40">
                                      <span className="mt-0.5 text-brand-gold/40">◆</span>{sug}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {parsed.prediction && parsed.prediction.predicted_distance > 0 && (
                              <div className="mb-4">
                                <SimAnimation
                                  prediction={parsed.prediction}
                                  lang={lang}
                                  isPro={rec.type === "pro"}
                                />
                              </div>
                            )}

                            {parsed.scores && (
                              <div className="mb-4">
                                <ProComparison
                                  userScores={parsed.scores}
                                  userAngles={getUserAngles(parsed)}
                                  lang={lang}
                                />
                              </div>
                            )}

                            {parsed.training_plan && (
                              <div className="rounded-xl border border-brand-gold/15 bg-black/20 p-4">
                                <h4 className="mb-3 text-sm font-semibold text-brand-gold">
                                  {lang === "zh" ? "训练计划" : "Training Plan"}
                                </h4>
                                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                                  {Object.entries(parsed.training_plan).map(([day, plan]) => (
                                    <div key={day} className="rounded-xl border border-brand-gold/10 bg-black/30 p-3">
                                      <div className="mb-2 flex items-center justify-between">
                                        <span className="rounded-full bg-brand-gold/20 px-2 py-0.5 text-[10px] font-semibold text-brand-gold">
                                          {day.toUpperCase()}
                                        </span>
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
                      </div>
                      );
                    })()}
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
