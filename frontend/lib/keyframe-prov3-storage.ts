/**
 * Client-side persistence for Pro v3 keyframe markup (per analysis, per frame index).
 * Syncs when reopening the same analysis from /pro or /history (same analysis_id).
 */

export const KF_PROV3_STORAGE_KEY = "stellar-prov3-kf-ui";

export type Prov3KfLine = { color: string; points: [number, number][] };
export type Prov3KfRuler = { color: string; a: [number, number]; b: [number, number] };

export type Prov3KfFrameState = {
  rotQ: number;
  lines: Prov3KfLine[];
  ruler: Prov3KfRuler | null;
};

export type Prov3KfStore = {
  v: 1;
  byFrame: Record<string, Prov3KfFrameState>;
};

function emptyFrame(): Prov3KfFrameState {
  return { rotQ: 0, lines: [], ruler: null };
}

export function loadProv3KfStore(analysisId: string): Prov3KfStore {
  if (typeof window === "undefined") return { v: 1, byFrame: {} };
  try {
    const raw = window.localStorage.getItem(`${KF_PROV3_STORAGE_KEY}:${analysisId}`);
    if (!raw) return { v: 1, byFrame: {} };
    const p = JSON.parse(raw) as Prov3KfStore;
    if (!p || p.v !== 1 || typeof p.byFrame !== "object") return { v: 1, byFrame: {} };
    return p;
  } catch {
    return { v: 1, byFrame: {} };
  }
}

export function saveProv3KfStore(analysisId: string, store: Prov3KfStore): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(`${KF_PROV3_STORAGE_KEY}:${analysisId}`, JSON.stringify(store));
  } catch {
    /* quota / private mode */
  }
}

export function getFrameState(store: Prov3KfStore, frameIndex: number): Prov3KfFrameState {
  return store.byFrame[String(frameIndex)] ?? emptyFrame();
}

export function setFrameState(
  store: Prov3KfStore,
  frameIndex: number,
  patch: Partial<Prov3KfFrameState>,
): Prov3KfStore {
  const key = String(frameIndex);
  const prev = store.byFrame[key] ?? emptyFrame();
  const next: Prov3KfFrameState = {
    rotQ: patch.rotQ ?? prev.rotQ,
    lines: patch.lines ?? prev.lines,
    ruler: patch.ruler !== undefined ? patch.ruler : prev.ruler,
  };
  return {
    v: 1,
    byFrame: { ...store.byFrame, [key]: next },
  };
}
