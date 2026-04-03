/**
 * One Euro Filter for real-time pose landmark smoothing.
 * Adapts cutoff frequency based on signal speed — smooth when still,
 * responsive during fast swing motion.
 * Reference: https://gery.casiez.net/1euro/
 */

class LowPassFilter {
  private s: number | null = null;

  filter(value: number, a: number): number {
    this.s = this.s === null ? value : a * value + (1 - a) * this.s;
    return this.s;
  }

  last(): number { return this.s ?? 0; }
  hasValue(): boolean { return this.s !== null; }
  reset(): void { this.s = null; }
}

function computeAlpha(te: number, cutoff: number): number {
  const tau = 1.0 / (2 * Math.PI * cutoff);
  return 1.0 / (1.0 + tau / te);
}

export interface OneEuroConfig {
  minCutoff: number;
  beta: number;
  dCutoff: number;
}

const GOLF_DEFAULTS: OneEuroConfig = {
  minCutoff: 1.7,
  beta: 0.4,
  dCutoff: 1.0,
};

export class OneEuroFilter {
  private cfg: OneEuroConfig;
  private xf = new LowPassFilter();
  private dxf = new LowPassFilter();
  private lastT: number | null = null;

  constructor(cfg?: Partial<OneEuroConfig>) {
    this.cfg = { ...GOLF_DEFAULTS, ...cfg };
  }

  filter(value: number, ts: number): number {
    if (this.lastT === null || !this.xf.hasValue()) {
      this.lastT = ts;
      return this.xf.filter(value, 1.0);
    }
    const te = Math.max(ts - this.lastT, 1e-6);
    this.lastT = ts;
    const dx = (value - this.xf.last()) / te;
    const edx = this.dxf.filter(dx, computeAlpha(te, this.cfg.dCutoff));
    const cutoff = this.cfg.minCutoff + this.cfg.beta * Math.abs(edx);
    return this.xf.filter(value, computeAlpha(te, cutoff));
  }

  reset(): void {
    this.xf.reset();
    this.dxf.reset();
    this.lastT = null;
  }
}

export interface LandmarkPoint {
  x: number;
  y: number;
  z?: number;
  visibility?: number;
  [key: string]: unknown;
}

/** Include BlazePose head (0–10) for stable face/head overlay on live capture. */
const SMOOTHED = new Set([
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
  11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28,
]);

export class LandmarkSmoother {
  private filters = new Map<string, OneEuroFilter>();
  private cfg: OneEuroConfig;

  constructor(cfg?: Partial<OneEuroConfig>) {
    this.cfg = { ...GOLF_DEFAULTS, ...cfg };
  }

  private get(key: string): OneEuroFilter {
    let f = this.filters.get(key);
    if (!f) { f = new OneEuroFilter(this.cfg); this.filters.set(key, f); }
    return f;
  }

  smooth(landmarks: LandmarkPoint[], timestampMs: number): LandmarkPoint[] {
    const ts = timestampMs / 1000;
    return landmarks.map((lm, i) => {
      if (!SMOOTHED.has(i) || (lm.visibility ?? 1) < 0.1) return lm;
      return {
        ...lm,
        x: this.get(`${i}x`).filter(lm.x, ts),
        y: this.get(`${i}y`).filter(lm.y, ts),
        z: lm.z !== undefined ? this.get(`${i}z`).filter(lm.z, ts) : lm.z,
      };
    });
  }

  reset(): void { this.filters.clear(); }
}
