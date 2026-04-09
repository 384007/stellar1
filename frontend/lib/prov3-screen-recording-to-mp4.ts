/**
 * Pro v3 screen capture: WebM (MediaRecorder) → MP4 before R2 upload + analyze/start,
 * so the pipeline matches a normal MP4 file upload (container + ext) while keeping screen_mode on the API.
 */

const OUT_NAME = "pro-screen.mp4";

function ffmpegCoreOrigin(): string {
  if (typeof window !== "undefined" && window.location?.origin) {
    return `${window.location.origin}/api/ffmpeg-core`;
  }
  return "/api/ffmpeg-core";
}

/** Avoid static `@ffmpeg/ffmpeg` imports so Edge/server bundles never pull UMD code that touches `document`. */
type FfmpegLike = {
  load: (opts: { coreURL: string; wasmURL: string }) => Promise<boolean>;
  writeFile: (name: string, data: Uint8Array) => Promise<void>;
  exec: (args: string[]) => Promise<void>;
  readFile: (name: string) => Promise<Uint8Array | string>;
  deleteFile: (name: string) => Promise<void>;
};

let ffmpegLoadPromise: Promise<FfmpegLike> | null = null;

async function getFFmpeg(): Promise<FfmpegLike> {
  if (ffmpegLoadPromise) return ffmpegLoadPromise;
  ffmpegLoadPromise = (async () => {
    const { FFmpeg } = await import("@ffmpeg/ffmpeg");
    const { toBlobURL } = await import("@ffmpeg/util");
    const ffmpeg = new FFmpeg();
    const base = ffmpegCoreOrigin();
    await ffmpeg.load({
      coreURL: await toBlobURL(`${base}/ffmpeg-core.js`, "text/javascript"),
      wasmURL: await toBlobURL(`${base}/ffmpeg-core.wasm`, "application/wasm"),
    });
    return ffmpeg as unknown as FfmpegLike;
  })();
  return ffmpegLoadPromise;
}

function blobLooksLikeMp4(blob: Blob): boolean {
  const t = (blob.type || "").toLowerCase();
  if (t.includes("mp4") || t.includes("mpeg4")) return true;
  return false;
}

/**
 * Whether this blob should be transcoded for prov3 screen upload (WebM screen capture, not already MP4).
 */
export function isProv3ScreenWebmForMp4Upload(
  blob: Blob,
  filename: string,
  explicitScreen: boolean | undefined,
  resolveScreenMode: (fn: string, ex?: boolean) => boolean,
): boolean {
  if (!resolveScreenMode(filename, explicitScreen)) return false;
  if (blobLooksLikeMp4(blob)) return false;
  const fn = filename.toLowerCase();
  const t = (blob.type || "").toLowerCase();
  if (fn.endsWith(".mp4") || fn.endsWith(".m4v")) return false;
  return fn.endsWith(".webm") || t.includes("webm") || t.includes("matroska") || t === "video/x-matroska";
}

/**
 * Transcode screen WebM → H.264 MP4 (no audio). Falls back to mpeg4 if x264 is unavailable in the wasm build.
 */
export async function prov3ScreenRecordingToMp4File(blob: Blob): Promise<File> {
  if (blobLooksLikeMp4(blob)) {
    return new File([blob], OUT_NAME, { type: "video/mp4" });
  }

  const ffmpeg = await getFFmpeg();
  const inName = "screen_in.webm";
  const u8 = new Uint8Array(await blob.arrayBuffer());
  await ffmpeg.writeFile(inName, u8);

  const tryExec = async (args: string[]) => {
    await ffmpeg.deleteFile(OUT_NAME).catch(() => {});
    await ffmpeg.exec(args);
  };

  try {
    await tryExec([
      "-i",
      inName,
      "-c:v",
      "libx264",
      "-preset",
      "veryfast",
      "-crf",
      "23",
      "-pix_fmt",
      "yuv420p",
      "-movflags",
      "+faststart",
      "-an",
      OUT_NAME,
    ]);
  } catch {
    await tryExec(["-i", inName, "-c:v", "mpeg4", "-q:v", "5", "-an", OUT_NAME]);
  }

  const data = await ffmpeg.readFile(OUT_NAME);
  await ffmpeg.deleteFile(inName).catch(() => {});
  await ffmpeg.deleteFile(OUT_NAME).catch(() => {});

  if (!(data instanceof Uint8Array)) {
    throw new Error("prov3_screen_mp4_unexpected_read");
  }
  const outBytes = data;
  if (!outBytes.byteLength) {
    throw new Error("prov3_screen_mp4_empty");
  }
  const copy = new Uint8Array(outBytes.length);
  copy.set(outBytes);
  return new File([copy], OUT_NAME, { type: "video/mp4" });
}
