/**
 * Real-time capture quality assessment for golf swing recording.
 * Evaluates framing, distance, angle, stability, and lighting.
 */

export interface QualityDimension {
  id: string;
  score: number;
  status: "good" | "fair" | "poor";
  msgZh: string;
  msgEn: string;
  icon: string;
}

export type OverallStatus = "excellent" | "good" | "adjust";

export interface QualityReport {
  dims: QualityDimension[];
  overall: OverallStatus;
  score: number;
  summaryZh: string;
  summaryEn: string;
}

interface Lm { x: number; y: number; visibility?: number }

const B = {
  head: 0, lSh: 11, rSh: 12, lEl: 13, rEl: 14, lWr: 15, rWr: 16,
  lHip: 23, rHip: 24, lKn: 25, rKn: 26, lAn: 27, rAn: 28,
};
const VIS_THRESH = 0.3;

export class CaptureQualityAssessor {
  private buf: Array<{ x: number; y: number }[]> = [];

  assess(landmarks: Lm[], brightness = 128): QualityReport {
    const dims: QualityDimension[] = [
      this.bodyInFrame(landmarks),
      this.distance(landmarks),
      this.angle(landmarks),
      this.stability(landmarks),
      this.light(brightness),
    ];
    const score = Math.round(dims.reduce((s, d) => s + d.score, 0) / dims.length);
    const overall: OverallStatus = score >= 75 ? "excellent" : score >= 50 ? "good" : "adjust";
    const sm: Record<OverallStatus, [string, string]> = {
      excellent: ["取景优秀，可以录制", "Excellent, ready to record"],
      good: ["取景可用，建议微调", "Good, minor adjustments"],
      adjust: ["请调整位置或角度", "Adjust position or angle"],
    };
    return { dims, overall, score, summaryZh: sm[overall][0], summaryEn: sm[overall][1] };
  }

  reset(): void { this.buf = []; }

  private v(lm: Lm | undefined): boolean {
    return !!lm && (lm.visibility ?? 1) >= VIS_THRESH;
  }

  private bodyInFrame(lms: Lm[]): QualityDimension {
    const keys = [B.head, B.lSh, B.rSh, B.lHip, B.rHip, B.lKn, B.rKn, B.lAn, B.rAn];
    const vis = keys.filter(k => this.v(lms[k])).length;
    const r = vis / keys.length;
    const score = Math.round(r * 100);
    if (score >= 80) return { id: "body", score, status: "good", msgZh: "全身入框", msgEn: "Full body visible", icon: "👤" };
    if (score >= 55) return { id: "body", score, status: "fair", msgZh: "部分肢体未入框", msgEn: "Some limbs missing", icon: "👤" };
    return { id: "body", score, status: "poor", msgZh: "请后退让全身入框", msgEn: "Step back for full body", icon: "👤" };
  }

  private distance(lms: Lm[]): QualityDimension {
    const hd = lms[B.head], la = lms[B.lAn], ra = lms[B.rAn];
    if (!this.v(hd) || (!this.v(la) && !this.v(ra)))
      return { id: "dist", score: 30, status: "poor", msgZh: "无法判断距离", msgEn: "Cannot assess distance", icon: "📏" };
    const ankle = this.v(la) ? la : ra;
    const span = Math.abs(ankle!.y - hd!.y);
    if (span > 0.85) return { id: "dist", score: 35, status: "poor", msgZh: "太近，请后退", msgEn: "Too close, step back", icon: "📏" };
    if (span < 0.35) return { id: "dist", score: 40, status: "poor", msgZh: "太远，请靠近", msgEn: "Too far, come closer", icon: "📏" };
    if (span >= 0.50 && span <= 0.80) return { id: "dist", score: 95, status: "good", msgZh: "距离合适", msgEn: "Good distance", icon: "📏" };
    return { id: "dist", score: 65, status: "fair", msgZh: "距离可用", msgEn: "Acceptable distance", icon: "📏" };
  }

  private angle(lms: Lm[]): QualityDimension {
    const ls = lms[B.lSh], rs = lms[B.rSh], lh = lms[B.lHip], rh = lms[B.rHip];
    if (!this.v(ls) || !this.v(rs) || !this.v(lh) || !this.v(rh))
      return { id: "angle", score: 50, status: "fair", msgZh: "无法判断角度", msgEn: "Cannot assess angle", icon: "📐" };
    const sw = Math.abs(rs!.x - ls!.x);
    const hw = Math.abs(rh!.x - lh!.x);
    const bh = Math.abs((ls!.y + rs!.y) / 2 - (lh!.y + rh!.y) / 2);
    const ratio = (sw + hw) / 2 / Math.max(bh, 0.01);
    if (ratio < 0.7) return { id: "angle", score: 90, status: "good", msgZh: "侧面机位良好", msgEn: "Good side angle", icon: "📐" };
    if (ratio < 1.0) return { id: "angle", score: 70, status: "fair", msgZh: "稍偏正面", msgEn: "Slightly frontal", icon: "📐" };
    return { id: "angle", score: 40, status: "poor", msgZh: "建议调整到侧面", msgEn: "Move to side view", icon: "📐" };
  }

  private stability(lms: Lm[]): QualityDimension {
    const keys = [B.lSh, B.rSh, B.lHip, B.rHip];
    const pts = keys.filter(k => this.v(lms[k])).map(k => ({ x: lms[k].x, y: lms[k].y }));
    if (pts.length === 0)
      return { id: "stab", score: 20, status: "poor", msgZh: "检测不稳定", msgEn: "Unstable detection", icon: "📊" };
    this.buf.push(pts);
    if (this.buf.length > 10) this.buf.shift();
    if (this.buf.length < 3)
      return { id: "stab", score: 60, status: "fair", msgZh: "正在稳定...", msgEn: "Stabilizing...", icon: "📊" };
    let tv = 0;
    for (let pi = 0; pi < pts.length; pi++) {
      const xs = this.buf.map(b => b[pi]?.x ?? 0);
      const ys = this.buf.map(b => b[pi]?.y ?? 0);
      tv += this.vr(xs) + this.vr(ys);
    }
    const av = tv / (pts.length * 2);
    const score = Math.round(Math.max(0, Math.min(100, 100 - av * 50000)));
    if (score >= 75) return { id: "stab", score, status: "good", msgZh: "识别稳定", msgEn: "Stable detection", icon: "📊" };
    if (score >= 45) return { id: "stab", score, status: "fair", msgZh: "识别中等", msgEn: "Moderate stability", icon: "📊" };
    return { id: "stab", score, status: "poor", msgZh: "检测不稳定", msgEn: "Unstable detection", icon: "📊" };
  }

  private light(b: number): QualityDimension {
    if (b >= 70 && b <= 220) return { id: "light", score: 90, status: "good", msgZh: "光线良好", msgEn: "Good lighting", icon: "💡" };
    if (b >= 40) return { id: "light", score: 60, status: "fair", msgZh: "光线偏暗", msgEn: "Low light", icon: "💡" };
    return { id: "light", score: 30, status: "poor", msgZh: "光线不足", msgEn: "Poor lighting", icon: "💡" };
  }

  private vr(a: number[]): number {
    if (a.length < 2) return 0;
    const m = a.reduce((s, v) => s + v, 0) / a.length;
    return a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length;
  }
}
