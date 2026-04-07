"use client";

import VideoAnalysisOverlay from "@/components/VideoAnalysisOverlay";
import { normalizePoseFramesForOverlay } from "@/lib/analysis-pose-storage";

type Props = {
  videoSrc: string;
  result: {
    pose_frames?: unknown[];
    prediction?: Record<string, unknown>;
    video_meta?: { source_frame_count?: number };
  };
  lang: "zh" | "en";
};

export default function Prov3PlusVideoRenderer({
  videoSrc,
  result,
  lang,
}: Props) {
  return (
    <VideoAnalysisOverlay
      videoSrc={videoSrc}
      poseFrames={normalizePoseFramesForOverlay(result.pose_frames)}
      lang={lang}
      prediction={result.prediction as never}
      sourceFrameCount={result.video_meta?.source_frame_count}
      skeletonStyle="plus"
    />
  );
}
