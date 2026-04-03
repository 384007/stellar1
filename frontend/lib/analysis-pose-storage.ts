import type { OverlayPoseFrame } from "@/components/VideoAnalysisOverlay";

/**
 * Slim pose frames for D1 `result_json` rows (overlay + 3D) without per-frame images
 * or duplicate joint blobs.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function slimJoint(j: any): Record<string, unknown> | null {
  if (!j || typeof j !== "object") return null;
  const n = j.normalized;
  const norm =
    n && typeof n === "object" && typeof n.x === "number" && typeof n.y === "number"
      ? { x: n.x, y: n.y }
      : undefined;
  const o: Record<string, unknown> = { name: j.name, visibility: j.visibility };
  if (typeof j.x === "number") o.x = j.x;
  if (typeof j.y === "number") o.y = j.y;
  if (typeof j.z === "number") o.z = j.z;
  if (norm) o.normalized = norm;
  if (typeof o.name !== "string") return null;
  return o;
}

/** Returns slimmed frames array for JSON storage (D1 compact rows). */
export function slimPoseFramesForCloudRow(raw: unknown): unknown[] {
  if (!Array.isArray(raw) || raw.length === 0) return [];
  const out: unknown[] = [];
  for (const f of raw) {
    if (!f || typeof f !== "object" || Array.isArray(f)) continue;
    const fr = f as Record<string, unknown>;
    const jointsRaw =
      (Array.isArray(fr.joints) && fr.joints.length > 0
        ? fr.joints
        : Array.isArray(fr.analysis_joints) && fr.analysis_joints.length > 0
          ? fr.analysis_joints
          : Array.isArray(fr.render_joints)
            ? fr.render_joints
            : null) ?? [];
    const joints = (jointsRaw as unknown[])
      .map((j) => slimJoint(j))
      .filter((x): x is Record<string, unknown> => x != null);
    if (joints.length === 0) continue;
    out.push({
      frame_index: fr.frame_index,
      timestamp: fr.timestamp,
      joints,
      connections: fr.connections,
      angles: fr.angles,
      frame_size: fr.frame_size,
      phase_data: fr.phase_data ?? null,
    });
  }
  return out;
}

/** Evenly subsample pose frames to fit JSON size caps (last resort). */
export function subsamplePoseFramesEven(arr: unknown[], maxFrames: number): unknown[] {
  if (arr.length <= maxFrames) return arr;
  const out: unknown[] = [];
  const step = (arr.length - 1) / Math.max(1, maxFrames - 1);
  for (let i = 0; i < maxFrames; i++) {
    const idx = Math.round(i * step);
    out.push(arr[Math.min(idx, arr.length - 1)]);
  }
  return out;
}

/**
 * Normalize API / storage quirks so VideoAnalysisOverlay always gets `joints`
 * (some payloads use analysis_joints only).
 */
export function normalizePoseFramesForOverlay(raw: unknown): OverlayPoseFrame[] {
  if (!Array.isArray(raw) || raw.length === 0) return [];
  const out: OverlayPoseFrame[] = [];
  for (const f of raw) {
    if (!f || typeof f !== "object" || Array.isArray(f)) continue;
    const fr = f as Record<string, unknown>;
    const joints =
      (Array.isArray(fr.joints) && fr.joints.length > 0
        ? fr.joints
        : Array.isArray(fr.analysis_joints) && fr.analysis_joints.length > 0
          ? fr.analysis_joints
          : Array.isArray(fr.render_joints)
            ? fr.render_joints
            : []) as OverlayPoseFrame["joints"];
    if (!joints.length) continue;
    out.push({
      joints,
      connections: (fr.connections as number[][]) ?? [],
      angles: (fr.angles as Record<string, number>) ?? {},
      frame_size: (fr.frame_size as OverlayPoseFrame["frame_size"]) ?? {
        width: 1,
        height: 1,
      },
      frame_index: typeof fr.frame_index === "number" ? fr.frame_index : 0,
      timestamp: typeof fr.timestamp === "number" ? fr.timestamp : 0,
      phase_data: (fr.phase_data as OverlayPoseFrame["phase_data"]) ?? null,
    });
  }
  return out;
}
