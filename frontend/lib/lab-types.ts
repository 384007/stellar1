/**
 * Shared Shot Lab types used by API routes, lib, and UI.
 * Single source of truth for the lab JSON contract.
 */

export interface LabMetrics {
  ball_speed_mph: number | null;
  ball_speed_confidence: number;
  launch_angle_deg: number | null;
  launch_angle_confidence: number;
  launch_direction_deg: number | null;
  launch_direction_confidence: number;
  backswing_time_sec: number | null;
  downswing_time_sec: number | null;
  tempo_ratio: number | null;
  tempo_confidence: number;
  carry_distance_yards: number | null;
  carry_distance_confidence: number;
  contact_quality_score: number | null;
  contact_quality_confidence: number;
}

export interface LabPrediction {
  predicted_distance: number;
  lateral_offset: number;
  shot_shape: string;
  shot_shape_zh: string;
  club_head_speed: number;
  ball_speed: number;
  launch_angle: number;
  spin_rate: number;
  smash_factor: number;
  trajectory?: Array<{ t: number; x: number; y: number; lateral: number }>;
  club_type?: string;
  club_group?: string;
  hand?: "R" | "L" | "UNKNOWN";
  hand_confidence?: number;
  baseline_distance?: number;
  technique_multiplier?: number;
  strike_multiplier?: number;
  speed_multiplier?: number;
  distance_confidence?: number;
  distance_debug?: Record<string, unknown>;
}

export interface LabIssue {
  id: string;
  title: string;
  title_zh: string;
  description: string;
  description_zh: string;
  severity: "high" | "medium" | "low";
  drill: string;
  drill_zh: string;
}

export interface LabDrill {
  title: string;
  title_zh: string;
  description: string;
  description_zh: string;
}

export interface LabResult {
  is_golf_swing: boolean;
  what_i_see: string;
  what_i_see_zh: string;
  metrics: LabMetrics;
  issues: LabIssue[];
  issues_total?: number;
  drills: LabDrill[];
  drills_total?: number;
  summary: string;
  summary_zh: string;
  full_report?: string;
  full_report_zh?: string;
  full_report_preview?: boolean;
  report_tier?: "free" | "pro";
  fields_visibility?: FieldsVisibility;
  prediction?: LabPrediction;
  club_type?: string;
  club_group?: string;
  hand?: "R" | "L" | "UNKNOWN";
}

export interface FieldsVisibility {
  backswing_time: "visible" | "locked";
  downswing_time: "visible" | "locked";
  full_report: "visible" | "locked" | "preview";
  full_issues: "visible" | "locked";
  full_drills: "visible" | "locked";
  compare: "visible" | "locked";
  trend: "visible" | "locked";
  export: "visible" | "locked";
  trajectory_full: "visible" | "locked";
}

export interface LabQuotaResponse {
  used: number;
  limit: number | null;
  remaining: number;
  is_pro: boolean;
}

export interface LabJobResponse {
  job_id: string;
  status: string;
  tier: "free" | "pro";
  report_tier: "free" | "pro";
  result?: LabResult;
  quota: LabQuotaResponse;
  created_at?: string;
  updated_at?: string;
}

export interface LabHistoryItem {
  id: string;
  status: string;
  tier: string;
  created_at: string;
  updated_at?: string;
  input_type?: string;
  club_type?: string;
  summary_snippet?: string;
}

export interface LabHistoryResponse {
  items: LabHistoryItem[];
  total: number;
  retention: {
    policy: "limited" | "long_term";
    max_items: number;
    max_days?: number;
  };
  quota: LabQuotaResponse;
  tier: "free" | "pro";
}

export interface LabCompareResponse {
  shots: LabJobResponse[];
  diff: Record<string, { a: number | null; b: number | null; delta: number | null }>;
}

export interface LabTrendPoint {
  job_id: string;
  date: string;
  ball_speed_mph: number | null;
  launch_angle_deg: number | null;
  tempo_ratio: number | null;
  carry_distance_yards: number | null;
  contact_quality_score: number | null;
}

export interface LabTrendResponse {
  points: LabTrendPoint[];
  period_days: number;
  total_sessions: number;
}

export type LabErrorCode =
  | "UNAUTHORIZED"
  | "FORBIDDEN"
  | "FEATURE_DISABLED"
  | "QUOTA_EXCEEDED"
  | "PRO_REQUIRED"
  | "BAD_REQUEST"
  | "NOT_FOUND"
  | "ANALYSIS_FAILED"
  | "DB_UNAVAILABLE"
  | "INTERNAL";
