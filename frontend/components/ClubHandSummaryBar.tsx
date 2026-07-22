"use client";

import { clubTypeLabel, handShortLabel } from "@/lib/club-display-labels";

export type ClubHandSummaryBarProps = {
  lang: "en" | "zh";
  /** UNKNOWN or missing => treat as not detected */
  clubType?: string;
  clubConfidence?: number;
  hand?: "R" | "L" | "UNKNOWN";
  /** True while client/server work is still in flight */
  pending?: boolean;
  className?: string;
};

/**
 * Compact club + handedness line for Lite mobile flow: analyzing, history while loading, share.
 */
export default function ClubHandSummaryBar({
  lang,
  clubType,
  clubConfidence,
  hand,
  pending = false,
  className = "",
}: ClubHandSummaryBarProps) {
  const clubKnown = Boolean(clubType && clubType !== "UNKNOWN");
  const handKnown = hand === "R" || hand === "L";
  const showPending = pending && !clubKnown && !handKnown;

  if (showPending) {
    return (
      <div
        className={`rounded-xl border border-brand-purple/25 bg-brand-purple/10 px-3 py-2 text-[11px] text-brand-purple/95 ${className}`}
      >
        <span className="font-medium">
          {lang === "zh" ? "正在识别球杆与左右手…" : "Detecting club and handedness…"}
        </span>
      </div>
    );
  }

  const clubLine = clubKnown
    ? lang === "zh"
      ? `球杆：${clubTypeLabel(clubType, lang)}`
      : `Club: ${clubType ?? ""}`
    : lang === "zh"
      ? "球杆：未识别（待确认）"
      : "Club: not detected (needs confirmation)";

  const handLine = handKnown
    ? handShortLabel(hand, lang)
    : lang === "zh"
      ? "待确认"
      : "pending confirmation";

  const conf =
    clubKnown && typeof clubConfidence === "number" && clubConfidence > 0
      ? Math.round(clubConfidence * 100)
      : null;

  return (
    <div
      className={`rounded-xl border border-brand-purple/25 bg-brand-purple/10 px-3 py-2 text-[11px] text-white/90 ${className}`}
    >
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span>{clubLine}</span>
        <span className="text-white/35">·</span>
        <span>
          {lang === "zh" ? "左右手：" : "Hand: "}
          <span className="font-medium text-brand-purple/90">{handLine}</span>
        </span>
        {conf != null && (
          <span className="text-white/45">
            ({lang === "zh" ? "置信度" : "conf"} {conf}%)
          </span>
        )}
        {pending && (clubKnown || handKnown) ? (
          <span className="text-brand-purple/70">
            {lang === "zh" ? "· 分析进行中" : "· analysis in progress"}
          </span>
        ) : null}
      </div>
    </div>
  );
}
