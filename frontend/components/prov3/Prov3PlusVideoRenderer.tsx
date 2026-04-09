"use client";

import VideoAnalysisOverlay from "@/components/VideoAnalysisOverlay";
import { normalizePoseFramesForOverlay } from "@/lib/analysis-pose-storage";
import { coachingTipsFromParsed } from "@/lib/video-analysis-coaching";

type Props = {
  videoSrc: string;
  result: {
    pose_frames?: unknown[];
    prediction?: Record<string, unknown>;
    video_meta?: { source_frame_count?: number };
    issues?: string[];
    issues_zh?: string[];
    suggestions?: string[];
    suggestions_zh?: string[];
    summary?: string;
    summary_zh?: string;
    training_plan?: Record<string, { focus: string; drills: string[] }>;
  };
  lang: "zh" | "en";
};

function isValidSourceFrameCount(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v) && v > 0;
}

/**
 * Prov3 时间线视频：与 PlusResultView 内默认 VideoAnalysisOverlay 同源能力
 *（plus 骨架、辅助线/弧线、scrub、轨迹、码数 HUD、全屏、toggle），仅封装给分析页等非 PlusResultView 宿主。
 *
 * 缺资产时仅在本组件内做前端诊断提示，不回退裸 video，不改变分析/trust 逻辑。
 */
export default function Prov3PlusVideoRenderer({
  videoSrc,
  result,
  lang,
}: Props) {
  const poses = normalizePoseFramesForOverlay(result.pose_frames);
  const src = String(videoSrc ?? "").trim();
  const zh = lang === "zh";

  if (!src) {
    return (
      <div className="flex min-h-[min(52vh,560px)] items-center justify-center rounded-xl border border-amber-500/30 bg-amber-500/[0.07] px-4 text-center text-xs leading-relaxed text-amber-100/90">
        {zh
          ? "缺少视频地址，无法显示挥杆叠加层。"
          : "No video address available; swing overlay cannot be shown."}
      </div>
    );
  }

  if (poses.length === 0) {
    return (
      <div className="flex min-h-[min(52vh,560px)] flex-col items-center justify-center gap-2 rounded-xl border border-amber-500/30 bg-amber-500/[0.07] px-4 py-10 text-center">
        <p className="text-xs font-medium leading-relaxed text-amber-100/95">
          {zh
            ? "缺少姿态数据，无法显示骨架、轨迹与码数叠加。"
            : "Pose data is missing; skeleton, trajectory, and yardage overlay cannot be shown."}
        </p>
        <p className="text-[11px] leading-relaxed text-amber-200/55">
          {zh
            ? "这是可视化数据缺失，不是页面故障。请确认完整结果已加载或重新分析。"
            : "Missing visualization payload, not a UI failure. Ensure the full result is loaded or re-analyze."}
        </p>
      </div>
    );
  }

  const predictionMissing = result.prediction == null;
  const sfc = result.video_meta?.source_frame_count;
  const sourceFrameCountMissing = !isValidSourceFrameCount(sfc);

  return (
    <div className="flex flex-col overflow-hidden rounded-xl">
      {predictionMissing || sourceFrameCountMissing ? (
        <div className="shrink-0 space-y-1.5 border border-amber-500/25 border-b-0 bg-amber-500/10 px-3 py-2.5 text-[11px] leading-snug text-amber-100/90">
          {predictionMissing ? (
            <p>
              {zh
                ? "本次结果缺少弹道预测数据，码数与轨迹提示可能不完整。"
                : "Ball-flight prediction data is missing; yardage and trajectory hints may be incomplete."}
            </p>
          ) : null}
          {sourceFrameCountMissing ? (
            <p>
              {zh
                ? "缺少视频帧计数信息，时间线与姿态对齐可能不完整。"
                : "Video frame count metadata is missing; timeline and pose alignment may be incomplete."}
            </p>
          ) : null}
        </div>
      ) : null}
      <VideoAnalysisOverlay
        videoSrc={src}
        poseFrames={poses}
        lang={lang}
        coachingTips={coachingTipsFromParsed(result, "pro")}
        prediction={
          result.prediction as
            | {
                predicted_distance?: number;
                shot_shape?: string;
                shot_shape_zh?: string;
                club_head_speed?: number;
                club_type?: string;
                hand?: "R" | "L" | "UNKNOWN";
              }
            | undefined
        }
        sourceFrameCount={isValidSourceFrameCount(sfc) ? sfc : undefined}
        skeletonStyle="plus"
      />
    </div>
  );
}
