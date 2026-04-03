/**
 * Client-side video compression for mobile uploads.
 *
 * Reduces resolution to 720p and bitrate to ~2.5 Mbps using
 * Canvas.captureStream + MediaRecorder.
 *
 * Browser support:
 *   Android Chrome: video/webm;codecs=vp8  ✔
 *   Desktop Chrome: video/webm;codecs=vp8  ✔
 *   Desktop Firefox: video/webm             ✔
 *   iOS Safari 14.6+: partial (may fall back to original)
 *
 * Falls back gracefully to the original file if unsupported.
 */

const COMPRESS_THRESHOLD = 15 * 1024 * 1024; // 15 MB
const TARGET_MAX_DIM = 720;
const TARGET_BITRATE = 2_500_000; // 2.5 Mbps

function pickMime(): string | null {
  if (typeof MediaRecorder === "undefined") return null;
  for (const m of [
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4",
  ]) {
    if (MediaRecorder.isTypeSupported(m)) return m;
  }
  return null;
}

export async function compressVideoForUpload(
  file: File,
  onProgress?: (pct: number, stage: string) => void,
): Promise<File> {
  if (file.size < COMPRESS_THRESHOLD) return file;

  const mime = pickMime();
  if (!mime) return file;

  return new Promise<File>((resolve) => {
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";

    const url = URL.createObjectURL(file);
    video.src = url;
    const cleanup = () => URL.revokeObjectURL(url);

    const timeout = setTimeout(() => { cleanup(); resolve(file); }, 30_000);

    video.onerror = () => { clearTimeout(timeout); cleanup(); resolve(file); };

    video.onloadedmetadata = () => {
      const { videoWidth: vw, videoHeight: vh, duration } = video;
      if (!vw || !vh || !duration || duration < 0.5) {
        clearTimeout(timeout); cleanup(); resolve(file); return;
      }

      if (Math.max(vw, vh) <= TARGET_MAX_DIM && file.size < COMPRESS_THRESHOLD * 1.5) {
        clearTimeout(timeout); cleanup(); resolve(file); return;
      }

      let w = vw, h = vh;
      if (Math.max(w, h) > TARGET_MAX_DIM) {
        const scale = TARGET_MAX_DIM / Math.max(w, h);
        w = Math.round(w * scale);
        h = Math.round(h * scale);
      }
      w = w % 2 === 0 ? w : w + 1;
      h = h % 2 === 0 ? h : h + 1;

      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) { clearTimeout(timeout); cleanup(); resolve(file); return; }

      let stream: MediaStream;
      try {
        stream = canvas.captureStream(30);
      } catch {
        clearTimeout(timeout); cleanup(); resolve(file); return;
      }

      let recorder: MediaRecorder;
      try {
        recorder = new MediaRecorder(stream, {
          mimeType: mime,
          videoBitsPerSecond: TARGET_BITRATE,
        });
      } catch {
        clearTimeout(timeout); cleanup(); resolve(file); return;
      }

      const chunks: Blob[] = [];
      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };

      recorder.onstop = () => {
        clearTimeout(timeout); cleanup();
        const blob = new Blob(chunks, { type: mime });
        if (blob.size > 0 && blob.size < file.size * 0.85) {
          const ext = mime.includes("mp4") ? "mp4" : "webm";
          resolve(new File([blob], file.name.replace(/\.\w+$/, `.${ext}`), { type: mime }));
        } else {
          resolve(file);
        }
      };

      recorder.onerror = () => {
        clearTimeout(timeout); cleanup(); resolve(file);
      };

      recorder.start(100);
      onProgress?.(5, "压缩中");

      video.playbackRate = 3.0;
      video.play().catch(() => { clearTimeout(timeout); cleanup(); resolve(file); });

      const drawLoop = () => {
        if (video.paused || video.ended) {
          try { recorder.stop(); } catch { /* already stopped */ }
          return;
        }
        ctx.drawImage(video, 0, 0, w, h);
        if (duration > 0) {
          onProgress?.(5 + Math.round((video.currentTime / duration) * 85), "压缩中");
        }
        requestAnimationFrame(drawLoop);
      };
      drawLoop();
    };
  });
}
