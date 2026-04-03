/**
 * Short coaching copy for VideoAnalysisOverlay (tap-to-show).
 */

export interface VideoCoachingTips {
  postureZh: string;
  postureEn: string;
  trainingZh: string;
  trainingEn: string;
}

const MAX_LEN = 120;

function clip(s: string, max = MAX_LEN): string {
  const t = (s || "").trim().replace(/\s+/g, " ");
  if (t.length <= max) return t;
  return `${t.slice(0, max - 1)}…`;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function coachingTipsFromParsed(parsed: any, type: string): VideoCoachingTips {
  if (type === "plus") {
    const postureZh = clip(parsed?.quick_tip_zh || parsed?.problem_description_zh || "");
    const postureEn = clip(parsed?.quick_tip_en || parsed?.problem_description_en || "");
    const tr = parsed?.training;
    const trainingZh = tr
      ? clip(`${tr.title_zh || ""}：${tr.description_zh || ""}`)
      : "";
    const trainingEn = tr
      ? clip(`${tr.title_en || ""}: ${tr.description_en || ""}`)
      : "";
    return { postureZh, postureEn, trainingZh, trainingEn };
  }

  const issZh = Array.isArray(parsed?.issues_zh) ? parsed.issues_zh[0] : "";
  const issEn = Array.isArray(parsed?.issues) ? parsed.issues[0] : "";
  const sugZh = Array.isArray(parsed?.suggestions_zh) ? parsed.suggestions_zh[0] : "";
  const sugEn = Array.isArray(parsed?.suggestions) ? parsed.suggestions[0] : "";
  const postureZh = clip(sugZh || issZh || (parsed?.summary_zh as string) || "");
  const postureEn = clip(sugEn || issEn || (parsed?.summary as string) || "");

  const plan = parsed?.training_plan as Record<string, { focus: string; drills: string[] }> | undefined;
  let trainingZh = "";
  let trainingEn = "";
  if (plan && typeof plan === "object") {
    const first = Object.values(plan)[0];
    if (first?.focus) {
      const d = first.drills?.[0] || "";
      trainingZh = clip(`${first.focus}${d ? ` · ${d}` : ""}`);
      trainingEn = trainingZh;
    }
  }
  if (!trainingZh && !trainingEn) {
    trainingZh = clip(sugZh || "");
    trainingEn = clip(sugEn || "");
  }

  return { postureZh, postureEn, trainingZh, trainingEn };
}
