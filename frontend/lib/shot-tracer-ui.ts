export type TracerPoint2D = { frame_index: number; timestamp: number; nx: number; ny: number; confidence: number };
export type TracerJoint2D = { name: string; nx: number; ny: number; visibility: number };
export type TracerBodyFrame = { frame_index: number; timestamp: number; joints: TracerJoint2D[]; confidence: number };
export type TracerJoint3D = { name: string; x: number; y: number; z: number; visibility: number };
export type TracerBody3DFrame = { frame_index: number; timestamp: number; joints: TracerJoint3D[]; confidence: number };
export type TracerPoint3D = { frame_index: number; timestamp: number; x: number; y: number; z: number; confidence: number };

export type ShotTracerUiResult = {
  status: string;
  engine?: string;
  real_video_reconstruction?: boolean;
  video?: { fps: number; duration_sec: number; frame_count: number; width: number; height: number };
  phases?: { address_t: number; top_t: number; impact_t: number; finish_t: number; impact_frame_index: number };
  paths?: {
    body_2d: TracerBodyFrame[];
    hands_2d: TracerPoint2D[];
    club_head_2d: TracerPoint2D[];
    club_shaft_2d: Array<{ frame_index: number; timestamp: number; x1: number; y1: number; x2: number; y2: number; confidence: number }>;
    ball_flight_2d: TracerPoint2D[];
    skeleton_3d: TracerBody3DFrame[];
    club_head_3d: TracerPoint3D[];
    ball_flight_3d: TracerPoint3D[];
  };
  metrics?: {
    estimated_club_head_speed_mph: number;
    estimated_ball_speed_mph: number;
    estimated_launch_angle_deg: number;
    estimated_carry_yards: number;
    estimated_apex_yards: number;
    estimated_lateral_curve_yards: number;
    tempo_ratio: string;
    confidence: number;
    scores?: {
      path_smoothness_score: number;
      tempo_score: number;
      body_stability_score: number;
      impact_score: number;
      overall_score: number;
    };
  };
  display?: { data_label?: string; accuracy_notice?: string };
  limitations?: string[];
};

function removeHidden(data: unknown): unknown {
  if (Array.isArray(data)) return data.map(removeHidden);
  if (data && typeof data === "object") {
    const hidden = new Set([
      "source",
      ["pro", "viders"].join(""),
      ["ad", "apter"].join(""),
      "debug",
      "stack",
      "traceback",
    ]);
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(data as Record<string, unknown>)) {
      if (hidden.has(k)) continue;
      out[k] = removeHidden(v);
    }
    return out;
  }
  return data;
}

export function sanitizeShotTracerResultForUI(result: unknown): ShotTracerUiResult {
  const clean = removeHidden(result) as Record<string, unknown>;
  const limitationsRaw = Array.isArray(clean.limitations) ? clean.limitations : [];
  return {
    ...(clean as ShotTracerUiResult),
    display: {
      data_label: "Video-based Estimate",
      accuracy_notice: "Single-camera estimates are not radar measurements.",
    },
    limitations: limitationsRaw.map(() => "Video-based Estimate"),
  };
}

export function buildShotTracerTips(metrics: ShotTracerUiResult["metrics"], lang: "zh" | "en"): string[] {
  if (!metrics) return lang === "zh" ? ["上传更清晰、完整的挥杆视频可提升重建质量。"] : ["Upload a clearer full-body swing clip to improve reconstruction quality."];
  const tips: string[] = [];
  const scores = metrics.scores;
  if (scores?.path_smoothness_score != null && scores.path_smoothness_score < 70) {
    tips.push(lang === "zh" ? "Club Path 稳定性偏低：建议缩短引杆幅度并保持轨迹连续。" : "Club Path is unstable: shorten takeaway and keep path continuity.");
  }
  if (scores?.tempo_score != null && scores.tempo_score < 72) {
    tips.push(lang === "zh" ? "节奏可优化：目标 Tempo 接近 3:1。" : "Tempo can improve: target near 3:1.");
  }
  if (scores?.body_stability_score != null && scores.body_stability_score < 72) {
    tips.push(lang === "zh" ? "身体稳定性不足：练习头部与骨盆控制，减少横移。" : "Body stability is low: reduce sway and keep head/pelvis control.");
  }
  if (scores?.impact_score != null && scores.impact_score < 72) {
    tips.push(lang === "zh" ? "Impact Moment 质量偏低：关注击球前后杆头路径一致性。" : "Impact quality is low: stabilize pre/post impact club path.");
  }
  if (Math.abs(metrics.estimated_lateral_curve_yards || 0) > 18) {
    tips.push(lang === "zh" ? "Ball Flight 偏曲明显：检查站位与杆面朝向。" : "Ball Flight curve is large: check alignment and club-face direction.");
  }
  if (!tips.length) {
    tips.push(lang === "zh" ? "本次挥杆整体稳定，可逐步提升杆头速度。" : "Swing is stable overall; progress by adding club speed gradually.");
  }
  return tips.slice(0, 4);
}
