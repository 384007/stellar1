"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { LandmarkSmoother } from "@/lib/pose-filters";
import { CaptureQualityAssessor, type QualityReport } from "@/lib/capture-quality";
import {
  getMediaPipeAllowForeignFallback,
  getMediaPipeWasmBases,
  getPoseModelUrls,
  resolveMediaPipeSelfHostBase,
} from "@/lib/mediapipe-assets";

function getMediaPipeLoadTimeouts(selfBase: string | null) {
  const selfOnly = !!selfBase && !getMediaPipeAllowForeignFallback();
  if (selfOnly) {
    return {
      bundle: 20_000,
      wasm: 25_000,
      liteGpu: 20_000,
      liteCpu: 25_000,
      fullGpu: 40_000,
    };
  }
  return {
    bundle: 12_000,
    wasm: 12_000,
    liteGpu: 15_000,
    liteCpu: 20_000,
    fullGpu: 30_000,
  };
}

interface ScreenModeCaptureProps {
  onCapture: (imageBase64: string) => void;
  onVideoCapture?: (videoBlob: Blob) => void;
  onExit: () => void;
  lang?: "en" | "zh";
}

type CaptureMode = "photo" | "video";

interface KeyJoint {
  idx: number;
  name: string;
  nameZh: string;
  color: string;
  proAngle: number;
  category: "core" | "extended";
}

/**
 * BlazePose 33 头部索引 0–10：鼻、双眼内外眼角、双耳、嘴角；与双肩衔接。
 * 此前实拍只画躯干四肢，头部未连线，故单独补全。
 */
const HEAD_POSE_CONNECTIONS: [number, number][] = [
  [0, 1],
  [0, 4],
  [1, 2],
  [2, 3],
  [4, 5],
  [5, 6],
  [3, 7],
  [6, 8],
  [7, 9],
  [8, 10],
  [9, 10],
  [9, 11],
  [10, 12],
];

const KEY_JOINTS: KeyJoint[] = [
  { idx: 0, name: "Nose", nameZh: "鼻尖", color: "#38bdf8", proAngle: 0, category: "core" },
  { idx: 11, name: "L.Shoulder", nameZh: "左肩", color: "#9f5fff", proAngle: 90,  category: "core" },
  { idx: 12, name: "R.Shoulder", nameZh: "右肩", color: "#9f5fff", proAngle: 85,  category: "core" },
  { idx: 23, name: "L.Hip",      nameZh: "左髋", color: "#7c3aed", proAngle: 148, category: "core" },
  { idx: 24, name: "R.Hip",      nameZh: "右髋", color: "#7c3aed", proAngle: 148, category: "core" },
  { idx: 13, name: "L.Elbow",    nameZh: "左肘", color: "#f5c518", proAngle: 170, category: "extended" },
  { idx: 14, name: "R.Elbow",    nameZh: "右肘", color: "#f5c518", proAngle: 148, category: "extended" },
  { idx: 25, name: "L.Knee",     nameZh: "左膝", color: "#00c853", proAngle: 163, category: "extended" },
  { idx: 26, name: "R.Knee",     nameZh: "右膝", color: "#00c853", proAngle: 163, category: "extended" },
];

const BODY_POSE_CONNECTIONS: [number, number][] = [
  [11, 12], [11, 13], [12, 14], [13, 15], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28],
];

const CONNECTIONS: [number, number][] = [...HEAD_POSE_CONNECTIONS, ...BODY_POSE_CONNECTIONS];

const ANGLE_DEFS: { joint: number; a: number; b: number }[] = [
  { joint: 11, a: 13, b: 23 }, { joint: 12, a: 14, b: 24 },
  { joint: 13, a: 11, b: 15 }, { joint: 14, a: 12, b: 16 },
  { joint: 23, a: 11, b: 25 }, { joint: 24, a: 12, b: 26 },
  { joint: 25, a: 23, b: 27 }, { joint: 26, a: 24, b: 28 },
];

/**
 * Guide skeleton in SCREEN-normalised coordinates (0–1).
 * Head raised to y=0.07 so the figure fills top-to-bottom with correct proportions.
 */
const GUIDE_JOINTS = [
  { x: 0.50, y: 0.07, r: 8  }, // 0 head
  { x: 0.50, y: 0.15, r: 4  }, // 1 neck
  { x: 0.40, y: 0.21, r: 5  }, // 2 L-shoulder
  { x: 0.60, y: 0.21, r: 5  }, // 3 R-shoulder
  { x: 0.32, y: 0.34, r: 4  }, // 4 L-elbow
  { x: 0.68, y: 0.34, r: 4  }, // 5 R-elbow
  { x: 0.27, y: 0.44, r: 4  }, // 6 L-wrist
  { x: 0.73, y: 0.44, r: 4  }, // 7 R-wrist
  { x: 0.43, y: 0.50, r: 5  }, // 8 L-hip
  { x: 0.57, y: 0.50, r: 5  }, // 9 R-hip
  { x: 0.41, y: 0.67, r: 4  }, // 10 L-knee
  { x: 0.59, y: 0.67, r: 4  }, // 11 R-knee
  { x: 0.39, y: 0.87, r: 3  }, // 12 L-ankle
  { x: 0.61, y: 0.87, r: 3  }, // 13 R-ankle
];

const GUIDE_CONNS = [
  [0,1],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7],
  [1,8],[1,9],[8,9],[8,10],[9,11],[10,12],[11,13],
];

function calcAngle(ax:number,ay:number,bx:number,by:number,cx:number,cy:number):number{
  const v1x=ax-bx,v1y=ay-by,v2x=cx-bx,v2y=cy-by;
  const dot=v1x*v2x+v1y*v2y;
  const m1=Math.sqrt(v1x*v1x+v1y*v1y),m2=Math.sqrt(v2x*v2x+v2y*v2y);
  if(m1===0||m2===0)return 0;
  return Math.round((Math.acos(Math.max(-1,Math.min(1,dot/(m1*m2))))*180)/Math.PI);
}

/**
 * Load MediaPipe vision bundle — dual approach for browser + PWA.
 *
 * Method A (browser): fetch /mp/vision_bundle.mjs text → Blob URL → import()
 *   Fastest, ESM-native. May fail in PWA due to CSP blocking new Function.
 *
 * Method B (PWA fallback): <script src="/mp/vision_bundle_global.js">
 *   Non-ESM build that sets window.__mediapipe_vision. No eval, no CSP issues.
 *
 * Both files are same-origin static assets built from node_modules.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _visionMod: any = null;

async function loadVisionBundle(): Promise<unknown> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  if (_visionMod) return _visionMod;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  if ((window as any).__mediapipe_vision) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    _visionMod = (window as any).__mediapipe_vision;
    return _visionMod;
  }

  // ── Method A: ESM import via Blob URL (browser, fast) ──
  try {
    const res = await fetch("/mp/vision_bundle.mjs");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const js = await res.text();
    if (js.length < 1000) throw new Error(`too small (${js.length}b)`);
    const blob = new Blob([js], { type: "text/javascript" });
    const blobUrl = URL.createObjectURL(blob);
    try {
      // eslint-disable-next-line @typescript-eslint/no-implied-eval
      _visionMod = await (new Function("u", "return import(u)"))(blobUrl);
    } finally {
      URL.revokeObjectURL(blobUrl);
    }
    if (_visionMod?.PoseLandmarker) return _visionMod;
  } catch { /* Method A failed (CSP / PWA) — fall through to B */ }

  // ── Method B: <script> tag (PWA-safe, no eval) ──
  return new Promise((resolve, reject) => {
    if (document.querySelector('script[data-mp-bundle]')) {
      const check = setInterval(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        if ((window as any).__mediapipe_vision) {
          clearInterval(check);
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          _visionMod = (window as any).__mediapipe_vision;
          resolve(_visionMod);
        }
      }, 100);
      setTimeout(() => { clearInterval(check); reject(new Error("bundle wait timeout")); }, 30_000);
      return;
    }

    const script = document.createElement("script");
    script.src = "/mp/vision_bundle_global.js";
    script.setAttribute("data-mp-bundle", "1");
    script.onload = () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      _visionMod = (window as any).__mediapipe_vision;
      if (_visionMod) resolve(_visionMod);
      else reject(new Error("script loaded but __mediapipe_vision not set"));
    };
    script.onerror = () => reject(new Error("vision_bundle_global.js load failed"));
    document.head.appendChild(script);
  });
}

function formatErr(err: unknown): string {
  if (err instanceof Error) return err.message || err.name;
  if (typeof err === "string") return err;
  if (err instanceof Event) return `Event:${err.type || "error"}`;
  try { return JSON.stringify(err); } catch { return "unknown error"; }
}

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timeout`)), ms);
    promise.then((v) => { clearTimeout(timer); resolve(v); })
           .catch((e) => { clearTimeout(timer); reject(e); });
  });
}

function withCountdown<T>(
  promise: Promise<T>,
  ms: number,
  label: string,
  onStage?: (stage: string) => void,
): Promise<T> {
  let elapsed = 0;
  const tick = setInterval(() => { elapsed++; onStage?.(`${label} ${elapsed}s`); }, 1000);
  return withTimeout(promise, ms, label).finally(() => clearInterval(tick));
}

const MP_CACHE = "mediapipe-0.10.33";

function xhrDownload(
  url: string,
  timeoutMs: number,
  onProgress?: (pct: number) => void,
): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("GET", url, true);
    xhr.responseType = "arraybuffer";
    xhr.timeout = timeoutMs;
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        onProgress?.(100);
        resolve(new Uint8Array(xhr.response as ArrayBuffer));
      } else {
        reject(new Error(`HTTP ${xhr.status}`));
      }
    };
    xhr.onerror = () => reject(new Error("network error"));
    xhr.ontimeout = () => reject(new Error("timeout"));
    xhr.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.min(99, Math.round((e.loaded / e.total) * 100)));
      }
    };
    xhr.send();
  });
}

/**
 * Race multiple URLs in parallel. First successful download wins, others abort.
 * Progress shows the highest percentage from any in-flight download.
 */
function raceDownloads(
  urls: string[],
  timeoutMs: number,
  onProgress?: (pct: number) => void,
): Promise<Uint8Array> {
  if (urls.length === 0) return Promise.reject(new Error("no urls"));
  return new Promise((resolve, reject) => {
    let done = false;
    let fails = 0;
    let maxPct = 0;
    const xhrs: XMLHttpRequest[] = [];

    for (const url of urls) {
      const xhr = new XMLHttpRequest();
      xhrs.push(xhr);
      xhr.open("GET", url, true);
      xhr.responseType = "arraybuffer";
      xhr.timeout = timeoutMs;
      xhr.onload = () => {
        if (done) return;
        if (xhr.status >= 200 && xhr.status < 300) {
          done = true;
          xhrs.forEach((x) => { if (x !== xhr && x.readyState < 4) try { x.abort(); } catch {} });
          onProgress?.(100);
          resolve(new Uint8Array(xhr.response as ArrayBuffer));
        } else {
          fails++;
          if (fails >= urls.length) reject(new Error(`all ${urls.length} sources HTTP error`));
        }
      };
      const fail = () => { if (!done && ++fails >= urls.length) reject(new Error("all sources failed")); };
      xhr.onerror = fail;
      xhr.ontimeout = fail;
      xhr.onprogress = (e) => {
        if (!done && e.lengthComputable) {
          const pct = Math.min(99, Math.round((e.loaded / e.total) * 100));
          if (pct > maxPct) { maxPct = pct; onProgress?.(pct); }
        }
      };
      xhr.send();
    }
  });
}

/**
 * Download a file from the fastest available source, then cache it under
 * the same-origin key so MediaPipe's internal fetches hit the SW cache.
 *
 * 1. Check SW cache → instant if cached from previous visit
 * 2. Race all URLs in parallel → fastest source wins
 * 3. Store result in cache under same-origin key
 */
async function downloadAndCache(
  cacheKey: string,
  urls: string[],
  timeoutMs: number,
  onProgress?: (pct: number) => void,
): Promise<Uint8Array> {
  try {
    const cache = await caches.open(MP_CACHE);
    const hit = await cache.match(cacheKey);
    if (hit) {
      onProgress?.(100);
      return new Uint8Array(await hit.arrayBuffer());
    }
  } catch { /* cache API unavailable */ }

  const buffer = await raceDownloads(urls, timeoutMs, onProgress);

  try {
    const cache = await caches.open(MP_CACHE);
    const ab = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) as ArrayBuffer;
    await cache.put(cacheKey, new Response(ab, {
      headers: { "Content-Type": "application/octet-stream" },
    }));
  } catch { /* cache write failed, non-fatal */ }

  return buffer;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _cachedLandmarker: any = null;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function loadMediaPipe(
  onProgress?: (p: number) => void,
  onStage?: (stage: string) => void,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
): Promise<any> {
  if (_cachedLandmarker) {
    onProgress?.(100);
    onStage?.("ready: cached");
    return _cachedLandmarker;
  }

  const selfBase = await resolveMediaPipeSelfHostBase();
  const failReasons: string[] = [];
  const t = getMediaPipeLoadTimeouts(selfBase);
  const wasmBases = getMediaPipeWasmBases(selfBase);
  const { full: fullModelUrls, lite: liteModelUrls } = getPoseModelUrls(selfBase);

  // ── Step 1: Vision bundle (Method A: ESM import → Method B: <script> tag) ──
  onStage?.("加载视觉引擎");
  onProgress?.(10);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let mod: any;
  try {
    mod = await withTimeout(loadVisionBundle(), t.bundle, "vision bundle") as /* eslint-disable-line @typescript-eslint/no-explicit-any */ any;
  } catch (err) {
    throw new Error(`vision bundle: ${formatErr(err)}`);
  }
  if (!mod?.PoseLandmarker) throw new Error("vision bundle loaded but PoseLandmarker missing");

  // ── Detect WASM variant (SIMD vs nosimd) ──
  const simd = (() => { try {
    return WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,10,1,8,0,65,0,253,15,253,98,11]));
  } catch { return false; } })();
  const wasmFile = simd ? "vision_wasm_internal" : "vision_wasm_nosimd_internal";

  // Build URL lists for parallel racing
  const wasmBinaryUrls = [
    `/mp/wasm/${wasmFile}.wasm`,
    `/api/mp-ext/wasm/${wasmFile}.wasm`,
  ];
  const wasmJsUrls = [`/mp/wasm/${wasmFile}.js`, `/api/mp-ext/wasm/${wasmFile}.js`];
  if (selfBase) {
    wasmBinaryUrls.splice(1, 0, `${selfBase}/wasm/${wasmFile}.wasm`);
    wasmJsUrls.splice(1, 0, `${selfBase}/wasm/${wasmFile}.js`);
  }

  // ── Step 2: Download WASM binary (~11MB) — race all sources in parallel ──
  onStage?.("下载引擎组件");
  onProgress?.(15);
  try {
    await downloadAndCache(`/mp/wasm/${wasmFile}.wasm`, wasmBinaryUrls, 60_000, (pct) => {
      onStage?.(`下载引擎组件 ${pct}%`);
      onProgress?.(15 + Math.round(pct * 0.2));
    });
  } catch (err) {
    failReasons.push(`wasm-dl: ${formatErr(err)}`);
  }
  // JS wrapper is small, race quietly
  try {
    await downloadAndCache(`/mp/wasm/${wasmFile}.js`, wasmJsUrls, 15_000);
  } catch {}
  onProgress?.(38);

  // ── Step 3: FilesetResolver (WASM files cached → should be instant) ──
  onStage?.("加载视觉组件");
  onProgress?.(40);
  let fileset: unknown;
  for (const wasmBase of wasmBases) {
    try {
      fileset = await withTimeout(
        mod.FilesetResolver.forVisionTasks(wasmBase), 30_000, "vision wasm"
      );
      break;
    } catch (err) {
      failReasons.push(`wasm(${wasmBase.slice(-40)}): ${formatErr(err)}`);
    }
  }
  if (!fileset) throw new Error(failReasons.join(" | "));
  onProgress?.(50);

  const baseOpts = { runningMode: "VIDEO", numPoses: 1,
    minPoseDetectionConfidence: 0.35, minPosePresenceConfidence: 0.35, minTrackingConfidence: 0.4 };

  // ── Step 4: Download lite model (~4.4MB) — race all sources in parallel ──
  onStage?.("下载骨架模型");
  onProgress?.(55);
  let liteBuffer: Uint8Array | null = null;
  try {
    liteBuffer = await downloadAndCache(
      "/mp/models/pose_landmarker_lite.task",
      liteModelUrls,
      60_000,
      (pct) => {
        onStage?.(`下载骨架模型 ${pct}%`);
        onProgress?.(55 + Math.round(pct * 0.2));
      },
    );
  } catch (err) {
    failReasons.push(`model-dl: ${formatErr(err)}`);
  }

  if (liteBuffer) {
    // ── Step 5a: Init lite + GPU (all files cached/buffered → no network) ──
    onStage?.("初始化骨架引擎 (GPU)");
    onProgress?.(78);
    try {
      const lm = await withCountdown(
        mod.PoseLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetBuffer: liteBuffer, delegate: "GPU" }, ...baseOpts,
        }), 45_000, "初始化骨架引擎 GPU", onStage);
      onProgress?.(100);
      onStage?.("ready: lite GPU");
      _cachedLandmarker = lm;
      return lm;
    } catch (err) {
      failReasons.push(`init-GPU: ${formatErr(err)}`);
    }

    // ── Step 5b: Init lite + CPU (same buffer, no re-download) ──
    onStage?.("初始化骨架引擎 (CPU)");
    onProgress?.(85);
    try {
      const lm = await withCountdown(
        mod.PoseLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetBuffer: liteBuffer, delegate: "CPU" }, ...baseOpts,
        }), 60_000, "初始化骨架引擎 CPU", onStage);
      onProgress?.(100);
      onStage?.("ready: lite CPU");
      _cachedLandmarker = lm;
      return lm;
    } catch (err) {
      failReasons.push(`init-CPU: ${formatErr(err)}`);
    }
  }

  throw new Error(failReasons.join(" | "));
}

interface JointCard {
  nameZh: string; name: string; color: string;
  angle: number; proAngle: number; category: "core" | "extended";
}

const Q_COLORS = {
  excellent: { border: "rgba(34,197,94,0.3)",  bg: "rgba(34,197,94,0.12)",  dot: "#22c55e", text: "#4ade80" },
  good:      { border: "rgba(234,179,8,0.3)",   bg: "rgba(234,179,8,0.12)",  dot: "#eab308", text: "#facc15" },
  adjust:    { border: "rgba(239,68,68,0.3)",   bg: "rgba(239,68,68,0.12)",  dot: "#ef4444", text: "#f87171" },
};
const DIM_COLORS = {
  good: "#22c55e", fair: "#eab308", poor: "#ef4444",
};
const CARD_BORDER = {
  none: "rgba(255,255,255,0.05)",
  good: "rgba(34,197,94,0.25)",
  fair: "rgba(234,179,8,0.25)",
  poor: "rgba(239,68,68,0.25)",
};
const CARD_DOT = {
  none: "#444", good: "#22c55e", fair: "#eab308", poor: "#ef4444",
};

export default function ScreenModeCapture({ onCapture, onVideoCapture, onExit, lang = "zh" }: ScreenModeCaptureProps) {
  const videoRef      = useRef<HTMLVideoElement>(null);
  const canvasRef     = useRef<HTMLCanvasElement>(null);
  const overlayRef    = useRef<HTMLCanvasElement>(null);
  const streamRef     = useRef<MediaStream | null>(null);
  const recorderRef   = useRef<MediaRecorder | null>(null);
  const chunksRef     = useRef<Blob[]>([]);
  const wakeLockRef   = useRef<WakeLockSentinel | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const lmRef         = useRef<any>(null);
  const animRef       = useRef<number>(0);
  const lastTRef      = useRef(-1);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const trailsRef     = useRef<any[][]>([]);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const lastLmRef     = useRef<any[] | null>(null);
  const lastCardUpdateRef  = useRef(0);
  const pendingCardsRef    = useRef<JointCard[] | null>(null);
  const pendingCountRef    = useRef<number | null>(null);
  const pendingQualityRef  = useRef<QualityReport | null | undefined>(undefined);
  const lastQualityTimeRef = useRef(0);
  const smootherRef   = useRef(new LandmarkSmoother());
  const qualityRef    = useRef(new CaptureQualityAssessor());
  const brightCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const brightRef     = useRef(128);

  // Refs that mirror state/props so the RAF loop never needs to restart when they change
  const isRecordingRef   = useRef(false);
  const recordingTimeRef = useRef(0);
  const poseStatusRef    = useRef<"loading" | "ready" | "failed">("loading");
  const loadProgressRef  = useRef(0);
  const langRef          = useRef(lang);

  const [cards, setCards] = useState<JointCard[]>(
    KEY_JOINTS.map(kj=>({nameZh:kj.nameZh,name:kj.name,color:kj.color,angle:0,proAngle:kj.proAngle,category:kj.category}))
  );
  const [cameraActive,  setCameraActive]  = useState(false);
  const [error,         setError]         = useState("");
  const [captureMode,   setCaptureMode]   = useState<CaptureMode>("video");
  const [isRecording,   setIsRecording]   = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [poseStatus,    setPoseStatus]    = useState<"loading"|"ready"|"failed">("loading");
  const [loadProgress,  setLoadProgress]  = useState(0);
  const [detectedCount, setDetectedCount] = useState(0);
  const [showAll,       setShowAll]       = useState(false);
  const [qualityReport, setQualityReport] = useState<QualityReport | null>(null);
  const [poseRetry,     setPoseRetry]     = useState(0);
  const [poseStage,     setPoseStage]     = useState("");
  const [poseErrorDetail, setPoseErrorDetail] = useState("");
  const detectErrCountRef = useRef(0);
  const missRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Keep render-loop refs in sync with state/props
  useEffect(() => { isRecordingRef.current = isRecording; }, [isRecording]);
  useEffect(() => { recordingTimeRef.current = recordingTime; }, [recordingTime]);
  useEffect(() => { poseStatusRef.current = poseStatus; }, [poseStatus]);
  useEffect(() => { loadProgressRef.current = loadProgress; }, [loadProgress]);
  useEffect(() => { langRef.current = lang; }, [lang]);
  // ── Canvas resize ── set buffer size (DPR-scaled) AND explicit CSS size
  useEffect(() => {
    function resize() {
      if (!overlayRef.current) return;
      const dpr = window.devicePixelRatio || 1;
      const w = window.innerWidth;
      const h = window.innerHeight;
      overlayRef.current.width  = w * dpr;
      overlayRef.current.height = h * dpr;
      overlayRef.current.style.width  = `${w}px`;
      overlayRef.current.style.height = `${h}px`;
    }
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);

  // ── Load MediaPipe ──
  useEffect(() => {
    let cancelled = false;
    if (typeof window !== "undefined" && (window as unknown as { __STELLAR_E2E_SKIP_MEDIAPIPE__?: boolean }).__STELLAR_E2E_SKIP_MEDIAPIPE__) {
      setLoadProgress(100);
      setPoseStatus("ready");
      setPoseStage("ready: e2e skip");
      setPoseErrorDetail("");
      return () => { cancelled = true; };
    }
    setPoseStatus("loading");
    setLoadProgress(0);
    setPoseStage("starting");
    setPoseErrorDetail("");
    detectErrCountRef.current = 0;
    loadMediaPipe(
      (p) => { if (!cancelled) setLoadProgress(p); },
      (stage) => { if (!cancelled) setPoseStage(stage); },
    )
      .then((l) => {
        if (!cancelled) {
          lmRef.current = l;
          setPoseStatus("ready");
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setPoseStatus("failed");
          setPoseErrorDetail(formatErr(err));
        }
      });
    return () => { cancelled = true; };
  }, [poseRetry]);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        try {
          await videoRef.current.play();
        } catch {
          /* Autoplay / headless: stream is still valid for MediaRecorder */
        }
      }
      setCameraActive(true);
    } catch {
      setError(lang === "zh" ? "无法访问摄像头，请允许权限后重试" : "Camera access denied.");
    }
  }, [lang]);

  useEffect(() => {
    startCamera();
    try { navigator.wakeLock?.request("screen").then(wl => { wakeLockRef.current = wl; }); } catch {}
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onExit(); };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      streamRef.current?.getTracks().forEach(t => t.stop());
      wakeLockRef.current?.release();
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [startCamera, onExit]);

  // ── Render loop ──
  useEffect(() => {
    if (!cameraActive || !overlayRef.current) return;
    const ctx = overlayRef.current.getContext("2d");
    if (!ctx) return;

    function frame() {
      if (!ctx || !overlayRef.current) return;
      const W = overlayRef.current.width, H = overlayRef.current.height;
      ctx.clearRect(0, 0, W, H);
      const t = Date.now() / 1000;

      /**
       * Convert MediaPipe normalised landmark coords (0–1, relative to the
       * VIDEO frame) to canvas pixel coords, accounting for the CSS
       * `object-cover` transform applied to the <video> element.
       *
       * Without this correction the skeleton is drawn at completely wrong
       * positions — on a portrait phone with a landscape (1280×720) camera
       * feed the horizontal offset alone can exceed half the screen width.
       */
      const video = videoRef.current;
      const vW = (video && video.videoWidth  > 0) ? video.videoWidth  : W;
      const vH = (video && video.videoHeight > 0) ? video.videoHeight : H;
      // object-cover: scale the video so it completely fills the canvas
      const covScale = Math.max(W / vW, H / vH);
      const covW = vW * covScale;   // scaled video width  in canvas pixels
      const covH = vH * covScale;   // scaled video height in canvas pixels
      const covOX = (W - covW) / 2; // horizontal offset  (negative = left crop)
      const covOY = (H - covH) / 2; // vertical offset    (negative = top  crop)

      function lmX(nx: number) { return nx * covW + covOX; }
      function lmY(ny: number) { return ny * covH + covOY; }

      // ── Pose detection ──
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let freshLm: any[] | null = null;
      const det = lmRef.current;
      if (det && video && video.readyState >= 2) {
        const now = performance.now();
        if (now > lastTRef.current + 33) {
          try {
            const res = det.detectForVideo(video, now);
            if (res?.landmarks?.[0]) {
              freshLm = smootherRef.current.smooth(res.landmarks[0], now);
              lastLmRef.current = freshLm;
              missRef.current = 0;
            } else {
              missRef.current = (missRef.current || 0) + 1;
              if (missRef.current > 6) lastLmRef.current = null;
            }
          } catch (err) {
            if (detectErrCountRef.current < 3) {
              detectErrCountRef.current += 1;
              setPoseErrorDetail(`detectForVideo: ${formatErr(err)}`);
            }
          }
          lastTRef.current = now;
        }
      }

      const lm = freshLm || lastLmRef.current;

      if (lm) {
        drawVignette(ctx, W, H, lm, lmX, lmY);
        if (freshLm) {
          trailsRef.current.push(lm.map((p: {x:number;y:number}) => ({x:p.x,y:p.y})));
          if (trailsRef.current.length > 10) trailsRef.current.shift();
        }
        drawSkeleton(ctx, W, H, t, lm, lmX, lmY);

        const qNow = Date.now();
        if (qNow - lastQualityTimeRef.current > 400) {
          lastQualityTimeRef.current = qNow;
          measureBrightness();
          pendingQualityRef.current = qualityRef.current.assess(lm, brightRef.current);
        }
      } else {
        trailsRef.current = [];
        smootherRef.current.reset();
        missRef.current = 0;
        qualityRef.current.reset();
        pendingCountRef.current   = 0;
        pendingCardsRef.current   = KEY_JOINTS.map(kj=>({nameZh:kj.nameZh,name:kj.name,color:kj.color,angle:0,proAngle:kj.proAngle,category:kj.category}));
        pendingQualityRef.current = null;
        drawGuide(ctx, W, H, t);
      }

      const now2 = Date.now();
      if (now2 - lastCardUpdateRef.current > 150) {
        lastCardUpdateRef.current = now2;
        if (pendingCardsRef.current)  { setCards(pendingCardsRef.current); pendingCardsRef.current = null; }
        if (pendingCountRef.current !== null) { setDetectedCount(pendingCountRef.current); pendingCountRef.current = null; }
        if (pendingQualityRef.current !== undefined) { setQualityReport(pendingQualityRef.current ?? null); pendingQualityRef.current = undefined; }
      }

      drawCompositionGuides(ctx, W, H);
      drawCorners(ctx, W, H);
      drawTopHUD(ctx, W, H, t);
      animRef.current = requestAnimationFrame(frame);
    }

    // ── Vignette ──
    function drawVignette(
      ctx: CanvasRenderingContext2D, W: number, H: number,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      lm: any[],
      lmX: (nx: number) => number,
      lmY: (ny: number) => number,
    ) {
      const vis = lm.filter((p:{visibility?:number}) => (p.visibility ?? 1) >= 0.3);
      if (vis.length < 4) return;
      const xs = vis.map((p:{x:number}) => lmX(p.x));
      const ys = vis.map((p:{y:number}) => lmY(p.y));
      const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
      const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
      const rBody = Math.max(Math.max(...xs)-Math.min(...xs), Math.max(...ys)-Math.min(...ys)) / 2;
      const r = rBody + Math.min(W, H) * 0.12;
      const g = ctx.createRadialGradient(cx, cy, r * 0.4, cx, cy, r * 2.4);
      g.addColorStop(0, "transparent");
      g.addColorStop(0.55, "rgba(0,0,10,0.10)");
      g.addColorStop(1, "rgba(0,0,10,0.45)");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, W, H);
    }

    // ── Skeleton ──
    function drawSkeleton(
      ctx: CanvasRenderingContext2D, W: number, H: number, t: number,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      lm: any[],
      lmX: (nx: number) => number,
      lmY: (ny: number) => number,
    ) {
      void W; void H;
      let visible = 0;
      const angleMap: Record<number, number> = {};
      const keySet = new Set(KEY_JOINTS.map(j=>j.idx));

      for (const ad of ANGLE_DEFS) {
        if (!keySet.has(ad.joint)) continue;
        const jt=lm[ad.joint], a=lm[ad.a], b=lm[ad.b];
        if (jt&&a&&b&&jt.visibility>=0.2&&a.visibility>=0.2&&b.visibility>=0.2) {
          angleMap[ad.joint]=calcAngle(a.x,a.y,jt.x,jt.y,b.x,b.y);
        }
      }

      // Motion trails
      for (let ti = 0; ti < trailsRef.current.length - 1; ti++) {
        const trail = trailsRef.current[ti];
        const alpha = 0.025 + (ti / trailsRef.current.length) * 0.07;
        ctx.globalAlpha = alpha;
        for (const [fi, toI] of CONNECTIONS) {
          const from = trail[fi], to = trail[toI];
          if (!from || !to) continue;
          ctx.beginPath(); ctx.moveTo(lmX(from.x), lmY(from.y)); ctx.lineTo(lmX(to.x), lmY(to.y));
          ctx.strokeStyle = "rgba(159,95,255,0.7)"; ctx.lineWidth = 2; ctx.lineCap = "round"; ctx.stroke();
        }
      }

      // Connections
      ctx.globalAlpha = 1;
      for (const [fi, toI] of CONNECTIONS) {
        const from = lm[fi], to = lm[toI];
        if (!from || !to || (from.visibility ?? 1) < 0.15 || (to.visibility ?? 1) < 0.15) continue;
        const x1=lmX(from.x), y1=lmY(from.y), x2=lmX(to.x), y2=lmY(to.y);
        const d = Math.hypot(x2-x1, y2-y1);

        ctx.globalAlpha = 0.08;
        ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2);
        ctx.strokeStyle = "rgba(159,95,255,1)"; ctx.lineWidth = 22; ctx.lineCap = "round"; ctx.stroke();

        ctx.globalAlpha = 0.14;
        ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2);
        ctx.strokeStyle = "rgba(159,95,255,1)"; ctx.lineWidth = 12; ctx.lineCap = "round"; ctx.stroke();

        ctx.globalAlpha = 0.85;
        const g = ctx.createLinearGradient(x1,y1,x2,y2);
        g.addColorStop(0, "rgba(185,115,255,1)"); g.addColorStop(1, "rgba(245,197,24,0.9)");
        ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2);
        ctx.strokeStyle = g; ctx.lineWidth = 3.5; ctx.lineCap = "round"; ctx.stroke();

        ctx.globalAlpha = 1;
        for (let p = 0; p < Math.floor(d / 24); p++) {
          const prog = ((t * 0.9 + p * 0.25) % 1);
          const px = x1+(x2-x1)*prog, py = y1+(y2-y1)*prog;
          const sa = Math.sin(prog*Math.PI)*0.6;
          ctx.beginPath(); ctx.arc(px, py, 1.5+Math.sin(t*5+p)*0.5, 0, Math.PI*2);
          ctx.fillStyle = `rgba(245,197,24,${sa})`; ctx.fill();
        }

        if (d > 60) {
          ctx.globalAlpha = 0.13;
          const mid = 0.5+Math.sin(t*6)*0.12;
          const mx=x1+(x2-x1)*mid, my=y1+(y2-y1)*mid;
          const nx=-(y2-y1)/d, ny=(x2-x1)/d;
          const offset=Math.sin(t*9)*9;
          ctx.beginPath(); ctx.moveTo(x1,y1);
          ctx.quadraticCurveTo(mx+nx*offset, my+ny*offset, x2,y2);
          ctx.strokeStyle="rgba(100,210,255,0.8)"; ctx.lineWidth=1; ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }

      // Joints
      const newCards: JointCard[] = [];
      for (const kj of KEY_JOINTS) {
        const pt = lm[kj.idx];
        const detected = pt && (pt.visibility ?? 1) >= 0.15;

        if (detected) {
          visible++;
          const x = lmX(pt.x), y = lmY(pt.y);
          const breathe = 1 + 0.18*Math.sin(t*Math.PI*1.8+kj.idx*0.6);
          const r = 9 * breathe;
          const highConf = pt.visibility >= 0.7;

          ctx.globalAlpha = 0.12;
          ctx.beginPath(); ctx.arc(x, y, r*3.5, 0, Math.PI*2);
          ctx.strokeStyle = kj.color; ctx.lineWidth = 0.8; ctx.stroke();

          ctx.globalAlpha = 0.07;
          ctx.beginPath(); ctx.arc(x, y, r*5, 0, Math.PI*2);
          ctx.strokeStyle = kj.color; ctx.lineWidth = 0.5; ctx.stroke();

          ctx.globalAlpha = 1;
          const glow = ctx.createRadialGradient(x, y, 0, x, y, r*5.5);
          glow.addColorStop(0, kj.color + "55");
          glow.addColorStop(0.3, kj.color + "20");
          glow.addColorStop(1, "transparent");
          ctx.beginPath(); ctx.arc(x, y, r*5.5, 0, Math.PI*2);
          ctx.fillStyle = glow; ctx.fill();

          ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI*2);
          ctx.fillStyle = kj.color + "35"; ctx.fill();

          ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI*2);
          ctx.strokeStyle = highConf ? kj.color + "ee" : kj.color + "99";
          ctx.lineWidth = 2.8; ctx.stroke();

          ctx.beginPath(); ctx.arc(x, y, r*0.3, 0, Math.PI*2);
          ctx.fillStyle = "rgba(255,255,255,0.95)"; ctx.fill();

          const arcColor = highConf ? "rgba(0,230,100," : "rgba(245,197,24,";
          const arcStart = t*3+kj.idx;
          ctx.beginPath(); ctx.arc(x, y, r*1.6, arcStart, arcStart+1.5);
          ctx.strokeStyle = arcColor+"0.55)"; ctx.lineWidth = 2; ctx.stroke();
        }

        const angle = detected ? (angleMap[kj.idx] ?? 0) : 0;
        newCards.push({nameZh:kj.nameZh,name:kj.name,color:kj.color,angle,proAngle:kj.proAngle,category:kj.category});
      }

      pendingCountRef.current  = visible;
      pendingCardsRef.current  = newCards;
    }

    function drawGuide(ctx: CanvasRenderingContext2D, W: number, H: number, t: number) {
      const breathe = Math.sin(t*1.5)*0.003, sway = Math.sin(t*0.8)*0.005;
      ctx.globalAlpha = 0.22+Math.sin(t*0.6)*0.05;
      for (const [fi,ti] of GUIDE_CONNS) {
        const from=GUIDE_JOINTS[fi], to=GUIDE_JOINTS[ti];
        const x1=(from.x+sway)*W, y1=(from.y+breathe)*H;
        const x2=(to.x+sway)*W,   y2=(to.y+breathe)*H;
        const g=ctx.createLinearGradient(x1,y1,x2,y2);
        g.addColorStop(0,"rgba(124,58,237,0.7)"); g.addColorStop(1,"rgba(245,197,24,0.6)");
        ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2);
        ctx.strokeStyle=g; ctx.lineWidth=2; ctx.lineCap="round"; ctx.stroke();
      }
      for (const j of GUIDE_JOINTS) {
        const x=(j.x+sway)*W, y=(j.y+breathe)*H;
        const glow=ctx.createRadialGradient(x,y,0,x,y,j.r*3);
        glow.addColorStop(0,"rgba(124,58,237,0.5)"); glow.addColorStop(1,"transparent");
        ctx.beginPath(); ctx.arc(x,y,j.r*3,0,Math.PI*2); ctx.fillStyle=glow; ctx.fill();
        ctx.beginPath(); ctx.arc(x,y,j.r,0,Math.PI*2);
        ctx.fillStyle="rgba(159,95,255,0.7)"; ctx.fill();
      }
      ctx.globalAlpha=1;

      // Alignment hint text
      const ps = poseStatusRef.current;
      const msg = ps==="loading"
        ? (langRef.current==="zh" ? `AI 骨架加载中 ${loadProgressRef.current}%` : `Loading AI ${loadProgressRef.current}%`)
        : ps==="failed"
        ? (langRef.current==="zh" ? "请站到画面中央" : "Step into frame")
        : (langRef.current==="zh" ? "全身入框 · 头顶对齐上方" : "Full body in frame · head near top");
      ctx.globalAlpha=0.9;
      ctx.font="bold 15px system-ui, sans-serif"; ctx.textAlign="center";
      const tw=ctx.measureText(msg).width;
      ctx.fillStyle="rgba(0,0,0,0.55)";
      ctx.beginPath(); ctx.roundRect(W/2-tw/2-14, H*0.91-20, tw+28, 32, 8); ctx.fill();
      ctx.fillStyle="#a78bfa"; ctx.fillText(msg, W/2, H*0.91);
      ctx.textAlign="left"; ctx.globalAlpha=1;
    }

    function drawCompositionGuides(ctx: CanvasRenderingContext2D, W: number, H: number) {
      ctx.globalAlpha = 0.06;
      ctx.strokeStyle = "rgba(159,95,255,0.6)";
      ctx.lineWidth = 0.8;
      ctx.setLineDash([6, 10]);
      for (let i = 1; i <= 2; i++) {
        ctx.beginPath(); ctx.moveTo(W*i/3, 0); ctx.lineTo(W*i/3, H); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, H*i/3); ctx.lineTo(W, H*i/3); ctx.stroke();
      }
      ctx.setLineDash([]); ctx.globalAlpha=1;
    }

    function drawCorners(ctx: CanvasRenderingContext2D, W: number, H: number) {
      const rec = isRecordingRef.current;
      const color = rec ? "rgba(255,50,50,0.8)" : "rgba(0,220,100,0.6)";
      ctx.strokeStyle=color; ctx.lineWidth=2.5; ctx.lineCap="round";
      for (const [x,y,dx1,dy1,dx2,dy2] of [
        [20,20,38,0,0,38],[W-20,20,-38,0,0,38],
        [20,H-20,38,0,0,-38],[W-20,H-20,-38,0,0,-38],
      ] as number[][]) {
        ctx.beginPath(); ctx.moveTo(x+dx1,y+dy1); ctx.lineTo(x,y); ctx.lineTo(x+dx2,y+dy2); ctx.stroke();
      }
      if (rec) {
        ctx.globalAlpha = 0.15+Math.sin(Date.now()/200)*0.1;
        ctx.strokeStyle="rgba(255,50,50,0.7)"; ctx.lineWidth=3;
        ctx.strokeRect(6,6,W-12,H-12);
        ctx.globalAlpha=1;
      }
    }

    function drawTopHUD(ctx: CanvasRenderingContext2D, W: number, H: number, t: number) {
      void H;
      const rec = isRecordingRef.current;
      const recTime = recordingTimeRef.current;
      const ps = poseStatusRef.current;
      const lp = loadProgressRef.current;
      ctx.globalAlpha=1;

      if (rec) {
        ctx.font="bold 13px monospace";
        ctx.fillStyle = Math.floor(t*2)%2===0 ? "rgba(255,60,60,0.95)" : "rgba(255,60,60,0.25)";
        ctx.fillText("● REC", 16, 28);
        ctx.fillStyle="rgba(255,255,255,0.7)"; ctx.fillText(fmtTime(recTime), 72, 28);
      } else {
        ctx.font="bold 11px monospace";
        ctx.fillStyle="rgba(0,220,100,0.8)"; ctx.fillText("STELLAR AI", 16, 26);
        ctx.font="10px monospace"; ctx.fillStyle="rgba(255,255,255,0.4)";
        ctx.fillText(langRef.current==="zh" ? "实拍模式" : "LIVE", 16, 40);
      }

      ctx.font="bold 10px monospace";
      if (ps==="ready") {
        ctx.fillStyle="rgba(0,220,100,0.7)"; ctx.fillText("AI ✓", W-52, 26);
      } else if (ps==="loading") {
        ctx.fillStyle="rgba(245,197,24,0.7)";
        ctx.fillText(`${lp}%`, W-42, 26);
        ctx.fillStyle="rgba(255,255,255,0.1)";
        ctx.beginPath(); ctx.roundRect(W-80, 32, 60, 4, 2); ctx.fill();
        ctx.fillStyle="rgba(245,197,24,0.6)";
        ctx.beginPath(); ctx.roundRect(W-80, 32, 60*(lp/100), 4, 2); ctx.fill();
      }
    }

    function measureBrightness() {
      if (!videoRef.current) return;
      if (!brightCanvasRef.current) {
        brightCanvasRef.current = document.createElement("canvas");
        brightCanvasRef.current.width=16; brightCanvasRef.current.height=12;
      }
      const bctx = brightCanvasRef.current.getContext("2d", { willReadFrequently: true });
      if (!bctx) return;
      try {
        bctx.drawImage(videoRef.current, 0, 0, 16, 12);
        const id = bctx.getImageData(0, 0, 16, 12);
        let sum = 0;
        for (let i = 0; i < id.data.length; i+=4) sum+=(id.data[i]+id.data[i+1]+id.data[i+2])/3;
        brightRef.current = sum / (id.data.length/4);
      } catch { /* CORS/security */ }
    }

    frame();
    return () => { cancelAnimationFrame(animRef.current); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraActive]);

  function fmtTime(s: number) {
    return `${Math.floor(s/60).toString().padStart(2,"0")}:${(s%60).toString().padStart(2,"0")}`;
  }

  function captureFrame() {
    if (!videoRef.current || !canvasRef.current) return;
    const v=videoRef.current, c=canvasRef.current;
    c.width=v.videoWidth; c.height=v.videoHeight;
    const ctx=c.getContext("2d"); if (!ctx) return;
    ctx.drawImage(v,0,0);
    cleanup(); onCapture(c.toDataURL("image/jpeg",0.85).split(",")[1]);
  }

  function startRecording() {
    if (!streamRef.current) return;
    chunksRef.current=[];
    const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9"
      : MediaRecorder.isTypeSupported("video/webm") ? "video/webm" : "video/mp4";
    const rec = new MediaRecorder(streamRef.current, {mimeType:mime, videoBitsPerSecond:4000000});
    rec.ondataavailable = (e) => { if (e.data.size>0) chunksRef.current.push(e.data); };
    rec.onstop = () => { cleanup(); onVideoCapture ? onVideoCapture(new Blob(chunksRef.current,{type:mime})) : captureFrame(); };
    rec.start(100); recorderRef.current=rec;
    setIsRecording(true); setRecordingTime(0);
    timerRef.current=setInterval(()=>{
      setRecordingTime(prev=>{ if(prev>=15){stopRecording();return prev;} return prev+1; });
    }, 1000);
  }

  function stopRecording() {
    const rec = recorderRef.current;
    if (rec && rec.state === "recording") {
      try { rec.requestData(); } catch { /* flush */ }
    }
    if (rec?.state !== "inactive") rec?.stop();
    setIsRecording(false);
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current=null; }
  }

  function cleanup() {
    streamRef.current?.getTracks().forEach(t=>t.stop());
    wakeLockRef.current?.release();
    if (document.fullscreenElement) document.exitFullscreen().catch(()=>{});
  }

  const visibleCards = showAll ? cards : cards.filter(c=>c.category==="core");
  const qr = qualityReport;
  const qKey = qr?.overall ?? "adjust";
  const qC = Q_COLORS[qKey as keyof typeof Q_COLORS] ?? Q_COLORS.adjust;

  return (
    <div className="fixed inset-0 z-[9999] bg-black">
      {error ? (
        <div className="flex h-full items-center justify-center">
          <div className="text-center px-8">
            <div className="mb-4 text-4xl">📷</div>
            <p className="mb-4 text-red-400">{error}</p>
            <button onClick={onExit} className="btn-primary">{lang==="zh"?"返回":"Go Back"}</button>
          </div>
        </div>
      ) : (
        <>
          {/*
            video: object-cover fills the screen, maintaining aspect ratio.
            The overlay canvas is the same physical size.
            Landmark coordinates are re-mapped via the covScale/covOX/covOY
            transform computed in the render loop so the skeleton tracks the
            visible body correctly regardless of device orientation or camera
            aspect ratio.
          */}
          <video ref={videoRef} className="h-full w-full object-cover" playsInline muted autoPlay style={{ transform: "translateZ(0)", willChange: "transform" }} />
          <canvas ref={overlayRef} className="absolute inset-0 h-full w-full pointer-events-none" />
          <canvas ref={canvasRef} className="hidden" />

          {/* Mode tabs */}
          <div className="absolute top-4 left-1/2 -translate-x-1/2 flex rounded-full border border-white/10 bg-black/60 backdrop-blur-md overflow-hidden">
            <button onClick={()=>setCaptureMode("video")}
              className={`px-5 py-2 text-xs font-semibold transition-all ${captureMode==="video"?"bg-brand-purple/40 text-white":"text-white/40"}`}>
              {lang==="zh"?"实拍":"Record"}
            </button>
            <button onClick={()=>setCaptureMode("photo")}
              className={`px-5 py-2 text-xs font-semibold transition-all ${captureMode==="photo"?"bg-brand-purple/40 text-white":"text-white/40"}`}>
              {lang==="zh"?"拍照":"Photo"}
            </button>
          </div>

          {/* AI failed banner — retry button */}
          {poseStatus === "failed" && (
            <div className="absolute top-16 left-1/2 -translate-x-1/2 rounded-xl px-4 py-2 backdrop-blur-md"
              style={{ border:"1px solid rgba(239,68,68,0.3)", background:"rgba(239,68,68,0.12)" }}>
              <div className="flex items-center gap-3">
                <span className="text-[11px] text-red-400">{lang==="zh"?"AI骨架加载失败":"AI skeleton failed"}</span>
                <button
                  onClick={() => { lmRef.current = null; setPoseRetry(n => n + 1); }}
                  className="rounded-full bg-red-500/20 px-3 py-0.5 text-[11px] font-semibold text-red-300 hover:bg-red-500/40 transition">
                  {lang==="zh"?"重试":"Retry"}
                </button>
              </div>
              <div className="mt-1 max-w-[78vw] text-[10px] leading-4 text-red-200/90">
                {poseErrorDetail || poseStage}
              </div>
            </div>
          )}

          {(poseStatus === "loading" || (poseStatus === "ready" && poseErrorDetail)) && (
            <div className="absolute left-1/2 top-[4.9rem] z-30 -translate-x-1/2 rounded-md border border-white/10 bg-black/60 px-2 py-1 text-[10px] text-white/70">
              {poseStatus === "loading"
                ? `${lang==="zh" ? "加载阶段" : "Stage"}: ${poseStage || "starting"}`
                : `${lang==="zh" ? "运行提示" : "Runtime"}: ${poseErrorDetail}`}
            </div>
          )}

          {/* Quality badge */}
          {qr && !isRecording && (
            <div className="absolute top-14 left-1/2 -translate-x-1/2 flex items-center gap-2 rounded-full px-3 py-1.5 backdrop-blur-md"
              style={{ border:`1px solid ${qC.border}`, background: qC.bg }}>
              <span className="h-2 w-2 rounded-full" style={{ backgroundColor: qC.dot, boxShadow:`0 0 6px ${qC.dot}` }} />
              <span className="text-[11px] font-semibold" style={{ color: qC.text }}>
                {lang==="zh" ? qr.summaryZh : qr.summaryEn}
              </span>
              <div className="flex gap-1 ml-1">
                {qr.dims.map(d=>(
                  <span key={d.id}
                    title={lang==="zh" ? d.msgZh : d.msgEn}
                    className="h-1.5 w-1.5 rounded-full"
                    style={{ backgroundColor: DIM_COLORS[d.status as keyof typeof DIM_COLORS] ?? "#555" }} />
                ))}
              </div>
            </div>
          )}

          {/* Detected count badge */}
          {detectedCount > 0 && (
            <div className="absolute left-1/2 -translate-x-1/2 rounded-full bg-black/50 px-3 py-0.5 text-[10px] backdrop-blur-md"
              style={{ top: qr && !isRecording ? "5.5rem" : "3.5rem",
                       color: detectedCount>=6 ? "#4ade80" : "#facc15",
                       border: `1px solid ${detectedCount>=6 ? "rgba(74,222,128,0.2)" : "rgba(250,204,21,0.2)"}` }}>
              {lang==="zh" ? `已识别 ${detectedCount}/8 部位` : `${detectedCount}/8 joints`}
            </div>
          )}

          {/* Exit */}
          <button onClick={()=>{ if(isRecording) stopRecording(); onExit(); }}
            className="absolute right-4 top-4 flex h-10 w-10 items-center justify-center rounded-full border border-white/20 bg-black/50 text-white backdrop-blur-md text-sm">
            ✕
          </button>

          {/* Joint cards — right side */}
          <div className="absolute right-3 top-28 flex flex-col gap-1.5" style={{maxWidth:130}}>
            {visibleCards.map((card)=>{
              const isDetected = card.angle > 0;
              const diff = isDetected ? Math.abs(card.angle-card.proAngle) : -1;
              const st = !isDetected?"none":diff<10?"good":diff<25?"fair":"poor";
              return (
                <div key={card.nameZh} className="rounded-lg bg-black/55 backdrop-blur-md px-2.5 py-1.5 transition-all"
                  style={{ border:`1px solid ${CARD_BORDER[st as keyof typeof CARD_BORDER]}`, opacity: isDetected?0.85:0.4 }}>
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-1.5">
                      <span className="h-1.5 w-1.5 rounded-full" style={{backgroundColor: isDetected?card.color:"#555"}} />
                      <span className="text-[10px] text-white/60">{card.nameZh}</span>
                    </div>
                    <span className="h-1.5 w-1.5 rounded-full" style={{backgroundColor: CARD_DOT[st as keyof typeof CARD_DOT]}} />
                  </div>
                  <div className="mt-0.5 flex items-baseline justify-between">
                    <span className="text-xs font-bold" style={{color: isDetected?"#ffd85e":"#555"}}>
                      {isDetected?`${card.angle}°`:"—"}
                    </span>
                    <span className="text-[9px] text-white/30">参考 {card.proAngle}°</span>
                  </div>
                </div>
              );
            })}
            <button onClick={()=>setShowAll(!showAll)}
              className="rounded-lg border border-white/10 bg-black/40 backdrop-blur-md px-2 py-1 text-[9px] text-white/40 transition hover:text-white/70"
              style={{opacity:0.65}}>
              {showAll?(lang==="zh"?"收起":"Less"):(lang==="zh"?"全部 8 项":"All 8")}
            </button>
          </div>

          {/* Bottom controls */}
          <div className="absolute bottom-0 left-0 right-0 pb-8 pt-4">
            {!isRecording && (
              <p className="mb-4 text-center text-[10px] text-white/30">
                {captureMode==="video"
                  ?(lang==="zh"?"录制完整挥杆 · 全身入框 · 距离约3米 · 最长15秒":"Record swing · full body · ~3m · max 15s")
                  :(lang==="zh"?"拍摄站姿照片进行快速分析":"Capture stance for quick analysis")}
              </p>
            )}
            <div className="flex items-center justify-center">
              {captureMode==="video" ? (
                <button
                  type="button"
                  data-testid="e2e-capture-record"
                  onClick={isRecording?stopRecording:startRecording} disabled={!cameraActive}
                  className="group relative flex h-20 w-20 items-center justify-center rounded-full transition disabled:opacity-50">
                  <div className={`absolute inset-0 rounded-full border-4 transition-colors ${isRecording?"border-red-500":"border-white/60"}`} />
                  <div className={`transition-all ${isRecording?"h-7 w-7 rounded-md bg-red-500":"h-14 w-14 rounded-full bg-red-500 group-hover:scale-90"}`} />
                  {isRecording && <div className="absolute inset-0 rounded-full border-2 border-red-500 animate-ping opacity-30" />}
                </button>
              ) : (
                <button onClick={captureFrame} disabled={!cameraActive}
                  className="group flex h-20 w-20 items-center justify-center rounded-full border-4 border-white/60 disabled:opacity-50">
                  <div className="h-14 w-14 rounded-full bg-white/90 transition group-hover:scale-90" />
                </button>
              )}
            </div>
            {isRecording && (
              <div className="mx-auto mt-4 max-w-xs">
                <div className="h-1 w-full rounded-full bg-white/10 overflow-hidden">
                  <div className="h-full rounded-full bg-red-500 transition-all duration-1000" style={{width:`${(recordingTime/15)*100}%`}} />
                </div>
                <p className="mt-1 text-center text-[10px] text-white/40">{fmtTime(recordingTime)} / 00:15</p>
              </div>
            )}
          </div>

          {/* Bottom-left brand */}
          <div className="absolute left-4 bottom-4 rounded-full border border-white/5 bg-black/40 px-3 py-1 text-[10px] font-mono text-brand-gold/50 backdrop-blur-md">
            {cameraActive?(lang==="zh"?"STELLAR AI · 就绪":"STELLAR AI · READY"):(lang==="zh"?"初始化...":"INIT...")}
          </div>
        </>
      )}
    </div>
  );
}
