/**
 * 三帧（25% / 40% / 60%）杆型预检，**单次** ``POST /api/club-detect`` → Edge 转发
 * ``/analyze/club-detect-batch``，Modal 侧一次 HTTP、进程内三次 vision（非三次独立 run）。
 */

import { devWarn } from "@/lib/dev-only-log";
import { isVideoFile } from "@/lib/upload-video";

export type ClubDetectBatchResult = {
  club_type: string;
  club_group: string;
  confidence: number;
  hand: "R" | "L";
};

function extractFrameFromBlob(blob: Blob): Promise<Blob | null> {
  return new Promise((resolve) => {
    const isVideo = isVideoFile(blob as File, (blob as File).name || "");
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
    const cleanup = () => {
      try {
        URL.revokeObjectURL(url);
      } catch {
        /* */
      }
    };
    const done = (b: Blob | null) => {
      if (resolved) return;
      resolved = true;
      cleanup();
      resolve(b);
    };
    const timer = setTimeout(() => {
      devWarn("[club-detect-batch] frame extraction timeout");
      done(null);
    }, 10000);

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
        if (!ctx) {
          done(null);
          return;
        }
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((b) => done(b), "image/jpeg", 0.8);
      } catch {
        done(null);
      }
    };

    video.onseeked = captureFrame;
    video.onloadeddata = () => {
      if (video.duration && isFinite(video.duration) && video.duration > 0.5) {
        video.currentTime = video.duration * 0.4;
      } else {
        captureFrame();
      }
    };
    video.onerror = () => {
      devWarn("[club-detect-batch] video load error");
      clearTimeout(timer);
      done(null);
    };
    video.load();
  });
}

function extractFrameAtPercent(blob: Blob, pct: number): Promise<Blob | null> {
  return new Promise((resolve) => {
    const isVideo = isVideoFile(blob as File, (blob as File).name || "");
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
    const cleanup = () => {
      try {
        URL.revokeObjectURL(url);
      } catch {
        /* */
      }
    };
    const done = (b: Blob | null) => {
      if (resolved) return;
      resolved = true;
      cleanup();
      resolve(b);
    };
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
        if (!ctx) {
          done(null);
          return;
        }
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((b) => done(b), "image/jpeg", 0.8);
      } catch {
        done(null);
      }
    };
    video.onseeked = captureFrame;
    video.onloadeddata = () => {
      if (video.duration && isFinite(video.duration) && video.duration > 0.5) {
        video.currentTime = video.duration * pct;
      } else {
        captureFrame();
      }
    };
    video.onerror = () => {
      clearTimeout(timer);
      done(null);
    };
    video.load();
  });
}

/**
 * 抽三帧并 **一次** 请求 Edge；服务端在同一次 Modal 调用内合并结果。
 */
export async function fetchClubDetectThreeFrames(
  blob: Blob,
  authHeaders: Record<string, string>,
): Promise<ClubDetectBatchResult | null> {
  const framePcts = [0.25, 0.4, 0.6];
  const frameBlobs = await Promise.all(framePcts.map((p) => extractFrameAtPercent(blob, p)));
  const valid = frameBlobs.filter((b): b is Blob => b != null && b.size > 0);
  if (valid.length === 0) {
    const single = await extractFrameFromBlob(blob);
    if (single && single.size > 0) {
      valid.push(single, single, single);
    }
  }
  if (valid.length === 0) {
    devWarn("[club-detect-batch] no frames extracted");
    return null;
  }
  while (valid.length < 3) {
    valid.push(valid[valid.length - 1]!);
  }
  const three = valid.slice(0, 3);

  const fd = new FormData();
  three.forEach((b, i) => {
    fd.append(`frame_${i}`, b, `frame_${i}.jpg`);
  });

  try {
    const res = await fetch("/api/club-detect", { method: "POST", headers: authHeaders, body: fd });
    if (!res.ok) return null;
    const json = (await res.json()) as {
      club_type?: string;
      club_group?: string;
      confidence?: number;
      hand?: string;
    };
    if (!json || typeof json !== "object") return null;
    return {
      club_type: String(json.club_type || "UNKNOWN"),
      club_group: String(json.club_group || "IRON"),
      confidence: Number(json.confidence) || 0,
      hand: json.hand === "L" ? "L" : "R",
    };
  } catch (e) {
    devWarn("[club-detect-batch] fetch error:", e);
    return null;
  }
}
