"use client";

import type { PlusAnalysisResult } from "@/components/PlusResultView";
import { stellarProTrustIsLow } from "@/lib/stellar-pro-result";
import {
  isProv3StrictMediaPolicyResult,
  prov3DisplayKeyframeRows,
  type Prov3ResultLike,
} from "@/lib/prov3-keyframe-media";
import { buildProv3MotionEvidenceReport } from "@/lib/prov3-motion-evidence-report";

type Props = {
  result: PlusAnalysisResult;
  lang: "zh" | "en";
};

export default function Prov3MotionEvidenceReport({ result, lang }: Props) {
  if (!isProv3StrictMediaPolicyResult(result as Prov3ResultLike)) return null;

  const lowTrust = stellarProTrustIsLow(result as Prov3ResultLike);
  const keyRows = prov3DisplayKeyframeRows(result as Prov3ResultLike) as PlusAnalysisResult["keyframes"];
  const sections = buildProv3MotionEvidenceReport({
    pose_frames: result.pose_frames,
    video_meta: result.video_meta,
    prediction: result.prediction,
    swing_phase_evaluations: result.swing_phase_evaluations,
    keyframes_strip: result.keyframes_strip,
    phase_keyframes: result.phase_keyframes,
    keyframes_display: keyRows.map((k) => ({
      phase: k.phase,
      label_zh: k.label_zh,
      label_en: k.label_en,
      timestamp: k.timestamp,
      frame_index: k.frame_index,
      display_source_frame_index: k.display_source_frame_index,
      analysis_timestamp: k.analysis_timestamp,
    })),
    core_frame_scores: result.core_frame_scores ?? null,
    gemini_frame_notes: result.gemini_observation?.frame_notes ?? null,
    issues: result.issues,
    issues_zh: result.issues_zh,
    summary: result.summary,
    summary_zh: result.summary_zh,
  });

  if (sections.length === 0) return null;

  const t = lang === "zh";

  return (
    <div className="glass-card overflow-hidden border border-brand-gold/20 bg-brand-gold/[0.04]">
      <div className="border-b border-white/10 bg-black/20 px-4 py-3">
        <p className="text-xs font-semibold uppercase tracking-wider text-brand-gold/90">
          {t ? "骨架与视频 · 依据报告" : "Skeleton & video · evidence report"}
        </p>
        <p className="mt-1 text-[10px] leading-relaxed text-white/45">
          {t
            ? "以下内容根据本结果的逐帧姿态、时间线元数据与视频融合字段自动生成，与「视频分析」叠加、关键帧条、3D 视图同源；非新增推理接口。"
            : "Generated from this result’s pose frames, timeline meta, and video-fusion fields—same sources as the Video overlay, keyframe strip, and 3D view; no extra API."}
        </p>
      </div>
      {lowTrust ? (
        <div className="border-b border-amber-500/25 bg-amber-500/10 px-4 py-2.5 text-[11px] leading-relaxed text-amber-100/95">
          {t
            ? "低信任模式：本报告仍基于同一套骨架序列与时间线生成，便于你对照练习；正式相位/评分以产品规则为准。"
            : "Low-trust mode: this report still uses the same pose sequence and timeline for practice reference; official phases/scoring follow product rules."}
        </div>
      ) : (
        <div className="border-b border-emerald-500/20 bg-emerald-500/[0.06] px-4 py-2.5 text-[11px] leading-relaxed text-emerald-100/90">
          {t
            ? "高信任：报告与已通过校验的真 240 时间线、姿态输出一致。"
            : "High trust: aligned with validated true-240 timeline and pose outputs."}
        </div>
      )}
      <div className="space-y-0 divide-y divide-white/5">
        {sections.map((sec) => (
          <div key={sec.id} className="px-4 py-3">
            <p className="mb-2 text-[11px] font-semibold text-white/70">
              {t ? sec.title_zh : sec.title_en}
            </p>
            <ul className="list-disc space-y-1.5 pl-4 text-xs leading-relaxed text-white/60">
              {(t ? sec.bullets_zh : sec.bullets_en).map((b, i) => (
                <li key={i}>{b}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}
