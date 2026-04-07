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

/**
 * Prov3 时间线视频：与 PlusResultView 内默认 VideoAnalysisOverlay 同源能力
 *（plus 骨架、辅助线/弧线、scrub、轨迹、码数 HUD、全屏、toggle），仅封装给分析页等非 PlusResultView 宿主。
 */
export default function Prov3PlusVideoRenderer({
  videoSrc,
  result,
  lang,
}: Props) {
  const poses = normalizePoseFramesForOverlay(result.pose_frames);
  const src = String(videoSrc ?? "").trim();

  if (!src) {
    return (
      <div className="flex min-h-[min(52vh,560px)] items-center justify-center rounded-xl border border-white/10 bg-black/40 px-4 text-center text-xs text-white/40">
        {lang === "zh" ? "暂无可用视频地址" : "No video URL"}
      </div>
    );
  }

  if (poses.length === 0) {
    return (
      <div className="flex min-h-[min(52vh,560px)] flex-col items-center justify-center gap-2 rounded-xl border border-white/10 bg-black/40 px-4 py-10">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-white/15 border-t-white/55" />
        <p className="text-center text-xs text-white/45">
          {lang === "zh"
            ? "正在等待骨架帧数据，以启用完整视频叠加（轨迹、码数、Plus 骨架与辅助线）…"
            : "Waiting for pose frames for full overlay (trajectory, yardage, Plus skeleton & guides)…"}
        </p>
      </div>
    );
  }

  return (
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
      sourceFrameCount={result.video_meta?.source_frame_count}
      skeletonStyle="plus"
    />
  );
}
