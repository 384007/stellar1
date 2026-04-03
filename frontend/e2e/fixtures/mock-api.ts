/**
 * JSON bodies for route mocking — keep in sync with AnalysisResult / PlusAnalysisResult.
 */

export const mockClubDetect = {
  club_type: "7I",
  club_group: "IRON",
  confidence: 0.82,
};

export const mockAnalyzeLite = {
  analysis_id: "e2e-mock-lite",
  type: "lite",
  scores: { grip: 55, stance: 58, backswing: 52, downswing: 60, follow_through: 54 },
  total_score: 56,
  issues: ["e2e"],
  issues_zh: ["e2e"],
  suggestions: ["e2e"],
  suggestions_zh: ["e2e"],
  summary: "E2E mock summary",
  summary_zh: "E2E 模拟分析",
  what_i_see: "mock",
  what_i_see_zh: "模拟",
  is_golf_swing: true,
  keyframes: [],
  skeleton_data: { frames: [], total_frames: 0 },
  prediction: {
    predicted_distance: 150,
    lateral_offset: 5,
    shot_shape: "draw",
    shot_shape_zh: "小左曲",
    club_head_speed: 88,
    ball_speed: 118,
    launch_angle: 12,
    spin_rate: 2400,
    smash_factor: 1.34,
    trajectory: [],
  },
};

export const mockAnalyzePro = {
  ...mockAnalyzeLite,
  analysis_id: "e2e-mock-pro",
  type: "pro",
  pose_frames: [],
};

const eightPhases = [
  "address",
  "takeaway",
  "backswing",
  "top",
  "downswing",
  "impact",
  "follow_through",
  "finish",
] as const;

export const mockPlusResult = {
  analysis_id: "e2e-mock-plus",
  type: "plus" as const,
  posture_score: 6.5,
  primary_diagnosis: {
    title_zh: "E2E 主诊断",
    title_en: "E2E Primary",
    status_zh: "再接再厉",
    status_en: "Keep trying",
    ai_confidence: 0.7,
  },
  additional_issues: [],
  quick_tip_zh: "E2E 提示",
  quick_tip_en: "E2E tip",
  problem_description_zh: "E2E 描述",
  problem_description_en: "E2E description",
  swing_phase_evaluations: eightPhases.map((phase) => ({
    phase,
    status: "Good",
    note_zh: "模拟",
    note_en: "mock",
  })),
  training: {
    title_zh: "训练",
    title_en: "Training",
    description_zh: "描述",
    description_en: "desc",
    difficulty: "normal",
    frequency_percent: 50,
  },
  recommended_videos: [],
  scores: { grip: 55, stance: 58, backswing: 52, downswing: 60, follow_through: 54 },
  total_score: 56,
  issues: ["e2e"],
  issues_zh: ["e2e"],
  suggestions: ["e2e"],
  suggestions_zh: ["e2e"],
  summary: "E2E Plus summary",
  summary_zh: "E2E Plus 总结",
  keyframes: [],
  skeleton_data: { frames: [], total_frames: 0 },
  prediction: {},
  _plus_usage: { used: 1, remaining: 2, limit: 3, is_pro: false },
};
