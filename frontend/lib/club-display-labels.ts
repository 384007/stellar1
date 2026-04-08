/** Shared club type → short label for UI (analyze / history / share). */

export const CLUB_LABEL_ZH: Record<string, string> = {
  "1W": "1号木",
  "3W": "3号木",
  "5W": "5号木",
  "3I": "3铁",
  "4I": "4铁",
  "5I": "5铁",
  "6I": "6铁",
  "7I": "7铁",
  "8I": "8铁",
  "9I": "9铁",
  PW: "劈起杆",
  AW: "A杆",
  SW: "沙坑杆",
  LW: "L杆",
  PT: "推杆",
};

export function clubTypeLabel(clubType: string | undefined, lang: "en" | "zh"): string {
  if (!clubType || clubType === "UNKNOWN") return "";
  if (lang === "zh") return CLUB_LABEL_ZH[clubType] || clubType;
  return clubType;
}

export function handShortLabel(hand: "R" | "L" | "UNKNOWN" | undefined, lang: "en" | "zh"): string {
  if (hand === "L") return lang === "zh" ? "左手" : "LH";
  if (hand === "R") return lang === "zh" ? "右手" : "RH";
  return "";
}
