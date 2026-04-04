import { slimPoseFramesForCloudRow, subsamplePoseFramesEven } from "./analysis-pose-storage";

/**
 * Strip heavy binary data from an analysis result before storing it in
 * localStorage.
 *
 * Keeps:  all keyframes (images ~25 KB each × 8 = ~200 KB — users need to
 *         see all swing phases on mobile), pose_frames without per-frame
 *         images (joints+angles ≈ 2 KB × 60 = ~120 KB — needed for skeleton
 *         overlay, guides, and 3D viewer), skeleton_data.frames (angle stats,
 *         ~30 KB — needed for HUD replay).
 *
 * Drops:  per-frame image_base64 inside pose_frames (bulk of the size),
 *         trajectory arrays, swing_phases (reconstructible).
 *
 * Typical stripped size: ~400 KB per record.  localStorage budget (5–10 MB)
 * comfortably holds 12–25 records.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function stripResultForStorage(result: any): any {
  const out = { ...result };

  // Keep ALL keyframes with images + pose_snapshot (needed for history replay)
  if (Array.isArray(out.keyframes)) {
    out.keyframes = out.keyframes.map(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (kf: any) => {
        const { image_base64, pose_snapshot, ...meta } = kf;
        return {
          ...meta,
          pose_snapshot: pose_snapshot && typeof pose_snapshot === "object" ? pose_snapshot : undefined,
          image_base64: typeof image_base64 === "string" ? image_base64 : undefined,
        };
      },
    );
  }

  // Keep pose_frames but strip the heavy per-frame image_base64
  // (joints, connections, angles, frame_size are lightweight and essential
  // for skeleton overlay, guide lines, and 3D viewer)
  if (Array.isArray(out.pose_frames)) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    out.pose_frames = out.pose_frames.map((f: any) => {
      const { image_base64, world_landmarks, segmentation_mask, ...rest } = f;
      return rest;
    });
  }

  // Keep skeleton_data.frames (small angle/stat data, needed for HUD replay)
  // — only drop if missing or not an object

  // Drop large trajectory arrays (reconstructible from pose_frames)
  delete out.trajectory;

  // Drop per-frame phase labels (reconstructible)
  delete out.swing_phases;

  return out;
}

/**
 * Extra strip for **localStorage list rows only** (Pro/Plus payloads can be several MB with keyframes).
 * Drops keyframe bitmaps and keyframe pose snapshots; keeps phase/timestamp/labels for sorting and cards.
 * Full replay/detail should come from cloud after sync or from the in-session result.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function stripHeavyMediaForLocalHistory(result: any): any {
  const out = { ...result };
  if (Array.isArray(out.keyframes)) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    out.keyframes = out.keyframes.map((kf: any) => {
      const { image_base64: _img, pose_snapshot: _ps, ...meta } = kf;
      return { ...meta };
    });
  }
  return out;
}

const TRANSPORT_POSE_FRAMES_MAX = 18;
const TRANSPORT_SKELETON_FRAMES_MAX = 24;

/**
 * Single pipeline for **localStorage history rows** (and legacy tiny API bodies):
 * drops keyframe bitmaps for quota. For authenticated cloud history POST, prefer
 * `slimAnalysisResultForServerHistory` so keyframes survive in R2/full JSON.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function slimAnalysisResultForHistoryTransport(result: any): any {
  let out: any = stripResultForStorage(stripHeavyMediaForLocalHistory(result));

  if (out.prediction && typeof out.prediction === "object" && !Array.isArray(out.prediction)) {
    const p = { ...(out.prediction as Record<string, unknown>) };
    delete p.trajectory;
    out.prediction = p;
  }

  if (Array.isArray(out.pose_frames) && out.pose_frames.length > 0) {
    const slim = slimPoseFramesForCloudRow(out.pose_frames);
    out.pose_frames = subsamplePoseFramesEven(slim, TRANSPORT_POSE_FRAMES_MAX);
  }

  if (out.skeleton_data && typeof out.skeleton_data === "object" && !Array.isArray(out.skeleton_data)) {
    const sk = out.skeleton_data as { frames?: unknown[]; total_frames?: number };
    const frames = Array.isArray(sk.frames) ? sk.frames : [];
    if (frames.length > TRANSPORT_SKELETON_FRAMES_MAX) {
      out.skeleton_data = { ...sk, frames: frames.slice(0, TRANSPORT_SKELETON_FRAMES_MAX) };
    }
  }

  delete out.trajectory;
  return out;
}

/**
 * For **POST /api/history** (cloud sync): same pose/skeleton slimming as
 * `slimAnalysisResultForHistoryTransport`, but **keeps keyframe `image_base64`**
 * so Pro/Plus history can show the strip after reload. Payload may exceed D1 row
 * cap; the API stores the full JSON in R2 and keeps a compact row in D1.
 *
 * localStorage rows should still use `slimAnalysisResultForHistoryTransport`.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function slimAnalysisResultForServerHistory(result: any): any {
  let out: any = stripResultForStorage(result);

  if (out.prediction && typeof out.prediction === "object" && !Array.isArray(out.prediction)) {
    const p = { ...(out.prediction as Record<string, unknown>) };
    delete p.trajectory;
    out.prediction = p;
  }

  if (Array.isArray(out.pose_frames) && out.pose_frames.length > 0) {
    const slim = slimPoseFramesForCloudRow(out.pose_frames);
    out.pose_frames = subsamplePoseFramesEven(slim, TRANSPORT_POSE_FRAMES_MAX);
  }

  if (out.skeleton_data && typeof out.skeleton_data === "object" && !Array.isArray(out.skeleton_data)) {
    const sk = out.skeleton_data as { frames?: unknown[]; total_frames?: number };
    const frames = Array.isArray(sk.frames) ? sk.frames : [];
    if (frames.length > TRANSPORT_SKELETON_FRAMES_MAX) {
      out.skeleton_data = { ...sk, frames: frames.slice(0, TRANSPORT_SKELETON_FRAMES_MAX) };
    }
  }

  delete out.trajectory;
  return out;
}
