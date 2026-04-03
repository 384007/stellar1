"use client";

import { useState } from "react";

interface ProComparisonProps {
  userScores: Record<string, number>;
  userAngles?: Record<string, number>;
  lang: "en" | "zh";
}

interface PlayerData {
  name: string;
  nameZh: string;
  era: string;
  image: string;
  scores: Record<string, number>;
  angles: Record<string, number>;
  metrics: Record<string, string>;
  phases: Record<string, Record<string, number>>;
}

const AMATEUR_AVG: PlayerData = {
  name: "Amateur Avg",
  nameZh: "业余平均",
  era: "Baseline",
  image: "👤",
  scores: { grip: 62, stance: 60, backswing: 55, downswing: 52, follow_through: 50 },
  angles: {
    shoulder_rotation: -30.0,
    hip_rotation: -18.0,
    x_factor: 35.0,
    spine_tilt: 12.5,
    left_knee: 160.0,
    right_elbow: 105.0,
  },
  metrics: {
    club_head_speed: "85 mph",
    swing_tempo: "3.8:1",
    avg_drive: "210 yards",
    accuracy: "48%",
  },
  phases: {
    setup:    { shoulder_rotation: -2, hip_rotation: -1, spine_tilt: 10, x_factor: 5 },
    backswing:{ shoulder_rotation: -25, hip_rotation: -12, spine_tilt: 12, x_factor: 28 },
    top:      { shoulder_rotation: -30, hip_rotation: -18, spine_tilt: 14, x_factor: 35 },
    downswing:{ shoulder_rotation: -15, hip_rotation: -25, spine_tilt: 11, x_factor: 22 },
    impact:   { shoulder_rotation: 5, hip_rotation: -35, spine_tilt: 8, x_factor: 40 },
    finish:   { shoulder_rotation: 30, hip_rotation: -42, spine_tilt: 15, x_factor: 18 },
  },
};

const PRO_PLAYERS: Record<string, PlayerData> = {
  tiger_woods: {
    name: "Tiger Woods",
    nameZh: "泰格·伍兹",
    era: "2000 Peak",
    image: "🏌️",
    scores: { grip: 98, stance: 97, backswing: 99, downswing: 99, follow_through: 98 },
    angles: {
      shoulder_rotation: -42.5,
      hip_rotation: -28.3,
      x_factor: 55.2,
      spine_tilt: 6.8,
      left_knee: 145.0,
      right_elbow: 85.0,
    },
    metrics: {
      club_head_speed: "125 mph",
      swing_tempo: "3.2:1",
      avg_drive: "298 yards",
      accuracy: "73%",
    },
    phases: {
      setup:    { shoulder_rotation: 0, hip_rotation: 0, spine_tilt: 6, x_factor: 2 },
      backswing:{ shoulder_rotation: -35, hip_rotation: -18, spine_tilt: 7, x_factor: 38 },
      top:      { shoulder_rotation: -42, hip_rotation: -28, spine_tilt: 7, x_factor: 55 },
      downswing:{ shoulder_rotation: -20, hip_rotation: -38, spine_tilt: 5, x_factor: 30 },
      impact:   { shoulder_rotation: 8, hip_rotation: -45, spine_tilt: 4, x_factor: 53 },
      finish:   { shoulder_rotation: 42, hip_rotation: -52, spine_tilt: 10, x_factor: 12 },
    },
  },
  rory_mcilroy: {
    name: "Rory McIlroy",
    nameZh: "罗里·麦克罗伊",
    era: "Current",
    image: "🏌️‍♂️",
    scores: { grip: 96, stance: 95, backswing: 98, downswing: 99, follow_through: 97 },
    angles: {
      shoulder_rotation: -45.0,
      hip_rotation: -30.0,
      x_factor: 58.0,
      spine_tilt: 5.5,
      left_knee: 148.0,
      right_elbow: 82.0,
    },
    metrics: {
      club_head_speed: "122 mph",
      swing_tempo: "3.0:1",
      avg_drive: "314 yards",
      accuracy: "65%",
    },
    phases: {
      setup:    { shoulder_rotation: 0, hip_rotation: 0, spine_tilt: 5, x_factor: 2 },
      backswing:{ shoulder_rotation: -38, hip_rotation: -20, spine_tilt: 6, x_factor: 42 },
      top:      { shoulder_rotation: -45, hip_rotation: -30, spine_tilt: 6, x_factor: 58 },
      downswing:{ shoulder_rotation: -22, hip_rotation: -40, spine_tilt: 4, x_factor: 32 },
      impact:   { shoulder_rotation: 10, hip_rotation: -48, spine_tilt: 3, x_factor: 58 },
      finish:   { shoulder_rotation: 45, hip_rotation: -55, spine_tilt: 12, x_factor: 10 },
    },
  },
  shin_ji_ae: {
    name: "Shin Ji-ae",
    nameZh: "申智爱",
    era: "Peak",
    image: "🏌️‍♀️",
    scores: { grip: 97, stance: 96, backswing: 95, downswing: 96, follow_through: 97 },
    angles: {
      shoulder_rotation: -38.0,
      hip_rotation: -25.0,
      x_factor: 48.0,
      spine_tilt: 4.2,
      left_knee: 152.0,
      right_elbow: 90.0,
    },
    metrics: {
      club_head_speed: "94 mph",
      swing_tempo: "3.3:1",
      avg_drive: "255 yards",
      accuracy: "78%",
    },
    phases: {
      setup:    { shoulder_rotation: 0, hip_rotation: 0, spine_tilt: 4, x_factor: 2 },
      backswing:{ shoulder_rotation: -30, hip_rotation: -16, spine_tilt: 5, x_factor: 35 },
      top:      { shoulder_rotation: -38, hip_rotation: -25, spine_tilt: 4, x_factor: 48 },
      downswing:{ shoulder_rotation: -18, hip_rotation: -34, spine_tilt: 3, x_factor: 28 },
      impact:   { shoulder_rotation: 6, hip_rotation: -42, spine_tilt: 3, x_factor: 48 },
      finish:   { shoulder_rotation: 38, hip_rotation: -48, spine_tilt: 8, x_factor: 10 },
    },
  },
  dustin_johnson: {
    name: "Dustin Johnson",
    nameZh: "达斯汀·约翰逊",
    era: "2020 Peak",
    image: "🏌️",
    scores: { grip: 95, stance: 96, backswing: 97, downswing: 98, follow_through: 96 },
    angles: {
      shoulder_rotation: -40.0,
      hip_rotation: -26.0,
      x_factor: 52.0,
      spine_tilt: 8.0,
      left_knee: 142.0,
      right_elbow: 88.0,
    },
    metrics: {
      club_head_speed: "124 mph",
      swing_tempo: "2.9:1",
      avg_drive: "313 yards",
      accuracy: "60%",
    },
    phases: {
      setup:    { shoulder_rotation: 0, hip_rotation: 0, spine_tilt: 7, x_factor: 2 },
      backswing:{ shoulder_rotation: -32, hip_rotation: -17, spine_tilt: 8, x_factor: 36 },
      top:      { shoulder_rotation: -40, hip_rotation: -26, spine_tilt: 8, x_factor: 52 },
      downswing:{ shoulder_rotation: -19, hip_rotation: -36, spine_tilt: 6, x_factor: 28 },
      impact:   { shoulder_rotation: 9, hip_rotation: -44, spine_tilt: 5, x_factor: 53 },
      finish:   { shoulder_rotation: 40, hip_rotation: -50, spine_tilt: 12, x_factor: 11 },
    },
  },
  collin_morikawa: {
    name: "Collin Morikawa",
    nameZh: "科林·森川",
    era: "Current",
    image: "🏌️‍♂️",
    scores: { grip: 97, stance: 95, backswing: 96, downswing: 97, follow_through: 96 },
    angles: {
      shoulder_rotation: -41.0,
      hip_rotation: -27.0,
      x_factor: 50.0,
      spine_tilt: 5.8,
      left_knee: 150.0,
      right_elbow: 86.0,
    },
    metrics: {
      club_head_speed: "116 mph",
      swing_tempo: "3.1:1",
      avg_drive: "295 yards",
      accuracy: "75%",
    },
    phases: {
      setup:    { shoulder_rotation: 0, hip_rotation: 0, spine_tilt: 5, x_factor: 2 },
      backswing:{ shoulder_rotation: -34, hip_rotation: -18, spine_tilt: 6, x_factor: 37 },
      top:      { shoulder_rotation: -41, hip_rotation: -27, spine_tilt: 6, x_factor: 50 },
      downswing:{ shoulder_rotation: -20, hip_rotation: -37, spine_tilt: 4, x_factor: 30 },
      impact:   { shoulder_rotation: 7, hip_rotation: -44, spine_tilt: 4, x_factor: 51 },
      finish:   { shoulder_rotation: 40, hip_rotation: -50, spine_tilt: 10, x_factor: 10 },
    },
  },
};

type PlayerKey = keyof typeof PRO_PLAYERS;

const SCORE_KEYS = ["grip", "stance", "backswing", "downswing", "follow_through"];
const SCORE_LABELS: Record<string, { en: string; zh: string }> = {
  grip: { en: "Grip", zh: "握杆" },
  stance: { en: "Stance", zh: "站姿" },
  backswing: { en: "Backswing", zh: "后摆" },
  downswing: { en: "Downswing", zh: "下杆" },
  follow_through: { en: "Follow Through", zh: "收杆" },
};

const ANGLE_KEYS = ["shoulder_rotation", "hip_rotation", "x_factor", "spine_tilt"];
const ANGLE_LABELS: Record<string, { en: string; zh: string; unit: string }> = {
  shoulder_rotation: { en: "Shoulder Rotation", zh: "肩部旋转", unit: "°" },
  hip_rotation:      { en: "Hip Rotation",      zh: "髋部旋转", unit: "°" },
  x_factor:          { en: "X-Factor",           zh: "X因子",    unit: "°" },
  spine_tilt:        { en: "Spine Tilt",         zh: "脊柱倾斜", unit: "°" },
};

const PHASE_LABELS: Record<string, { en: string; zh: string }> = {
  setup:     { en: "Setup",       zh: "准备" },
  backswing: { en: "Backswing",   zh: "后摆" },
  top:       { en: "Top",         zh: "顶点" },
  downswing: { en: "Downswing",   zh: "下杆" },
  impact:    { en: "Impact",      zh: "击球" },
  finish:    { en: "Finish",      zh: "收杆" },
};

function TriBar({ userVal, proVal, amateurVal, max, lang }: {
  userVal: number; proVal: number; amateurVal: number; max: number; lang: "en" | "zh";
}) {
  const pct = (v: number) => Math.min(100, Math.max(0, (Math.abs(v) / max) * 100));
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <span className="w-8 text-right text-[10px] text-brand-purple">{lang === "zh" ? "你" : "You"}</span>
        <div className="relative h-2 flex-1 rounded-full bg-white/5">
          <div className="h-full rounded-full bg-brand-purple/70" style={{ width: `${pct(userVal)}%` }} />
        </div>
        <span className="w-8 text-[10px] text-brand-purple">{userVal}</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="w-8 text-right text-[10px] text-brand-gold">Pro</span>
        <div className="relative h-2 flex-1 rounded-full bg-white/5">
          <div className="h-full rounded-full bg-brand-gold/70" style={{ width: `${pct(proVal)}%` }} />
        </div>
        <span className="w-8 text-[10px] text-brand-gold">{proVal}</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="w-8 text-right text-[10px] text-white/40">{lang === "zh" ? "均" : "Avg"}</span>
        <div className="relative h-2 flex-1 rounded-full bg-white/5">
          <div className="h-full rounded-full bg-white/20" style={{ width: `${pct(amateurVal)}%` }} />
        </div>
        <span className="w-8 text-[10px] text-white/40">{amateurVal}</span>
      </div>
    </div>
  );
}

export default function ProComparison({
  userScores,
  userAngles,
  lang,
}: ProComparisonProps) {
  const [selectedPlayer, setSelectedPlayer] = useState<PlayerKey>("tiger_woods");
  const [phaseAngle, setPhaseAngle] = useState("x_factor");
  const player = PRO_PLAYERS[selectedPlayer];

  const userTotal = SCORE_KEYS.reduce((s, k) => s + (userScores[k] || 0), 0) / SCORE_KEYS.length;
  const proTotal = SCORE_KEYS.reduce((s, k) => s + (player.scores[k] || 0), 0) / SCORE_KEYS.length;
  const amTotal = SCORE_KEYS.reduce((s, k) => s + (AMATEUR_AVG.scores[k] || 0), 0) / SCORE_KEYS.length;

  return (
    <div className="space-y-6">
      {/* Player Selector — scrollable on mobile */}
      <div className="glass-card p-4">
        <label className="mb-3 block text-sm font-semibold text-white/70">
          {lang === "en" ? "Compare with Pro Player" : "选择职业球员对比"}
        </label>
        <div className="flex gap-2 overflow-x-auto pb-1" style={{ scrollbarWidth: "none" }}>
          {(Object.entries(PRO_PLAYERS) as [PlayerKey, PlayerData][]).map(
            ([key, p]) => (
              <button
                key={key}
                onClick={() => setSelectedPlayer(key)}
                className={`flex-shrink-0 rounded-xl px-3 py-2.5 text-center transition ${
                  selectedPlayer === key
                    ? "border border-brand-gold/50 bg-brand-gold/10"
                    : "border border-white/5 bg-white/[0.03] hover:border-white/20"
                }`}
              >
                <div className="text-xl">{p.image}</div>
                <div className="mt-0.5 text-[11px] font-semibold text-white whitespace-nowrap">
                  {lang === "en" ? p.name : p.nameZh}
                </div>
                <div className="text-[9px] text-white/35">{p.era}</div>
              </button>
            )
          )}
        </div>
      </div>

      {/* ── Overall Score Summary ── */}
      <div className="glass-card p-5">
        <h3 className="mb-4 text-sm font-semibold text-white">
          {lang === "en" ? "Overall Rating" : "综合评分"}
        </h3>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-2xl font-bold text-brand-purple">{Math.round(userTotal)}</p>
            <p className="text-[10px] text-white/40">{lang === "zh" ? "你" : "You"}</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-brand-gold">{Math.round(proTotal)}</p>
            <p className="text-[10px] text-white/40">{lang === "zh" ? player.nameZh : player.name}</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-white/50">{Math.round(amTotal)}</p>
            <p className="text-[10px] text-white/40">{lang === "zh" ? "业余平均" : "Amateur Avg"}</p>
          </div>
        </div>
      </div>

      {/* ── Three-Way Score Comparison ── */}
      <div className="glass-card p-5">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-white">
            {lang === "en" ? "Score Comparison" : "评分三方对比"}
          </h3>
          <div className="flex items-center gap-3 text-[10px]">
            <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full bg-brand-purple" />{lang === "zh" ? "你" : "You"}</span>
            <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full bg-brand-gold" />Pro</span>
            <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full bg-white/30" />{lang === "zh" ? "业余" : "Avg"}</span>
          </div>
        </div>

        <div className="space-y-5">
          {SCORE_KEYS.map((key) => {
            const label = SCORE_LABELS[key];
            return (
              <div key={key}>
                <p className="mb-1.5 text-xs text-white/60">{lang === "en" ? label.en : label.zh}</p>
                <TriBar
                  userVal={userScores[key] || 0}
                  proVal={player.scores[key] || 0}
                  amateurVal={AMATEUR_AVG.scores[key] || 0}
                  max={100}
                  lang={lang}
                />
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Biomechanics Three-Way (only when angles available) ── */}
      {userAngles && (
        <div className="glass-card p-5">
          <h3 className="mb-4 text-sm font-semibold text-white">
            {lang === "en" ? "Biomechanics Comparison" : "生物力学三方对比"}
          </h3>

          <div className="grid gap-3 sm:grid-cols-2">
            {ANGLE_KEYS.map((key) => {
              const label = ANGLE_LABELS[key];
              const uv = userAngles[key] ?? 0;
              const pv = player.angles[key] ?? 0;
              const av = AMATEUR_AVG.angles[key] ?? 0;
              const diffPro = Math.abs(uv - pv);
              const diffColor = diffPro < 5 ? "text-green-400" : diffPro < 15 ? "text-brand-gold" : "text-red-400";

              return (
                <div key={key} className="rounded-xl border border-white/5 bg-black/30 p-4">
                  <p className="mb-2 text-xs text-white/50">{lang === "en" ? label.en : label.zh}</p>
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div>
                      <p className="text-lg font-bold text-brand-purple">{uv.toFixed(1)}{label.unit}</p>
                      <p className="text-[9px] text-white/35">{lang === "zh" ? "你" : "You"}</p>
                    </div>
                    <div>
                      <p className="text-lg font-bold text-brand-gold">{pv.toFixed(1)}{label.unit}</p>
                      <p className="text-[9px] text-white/35">Pro</p>
                    </div>
                    <div>
                      <p className="text-lg font-bold text-white/45">{av.toFixed(1)}{label.unit}</p>
                      <p className="text-[9px] text-white/35">{lang === "zh" ? "业余" : "Avg"}</p>
                    </div>
                  </div>
                  <p className={`mt-2 text-center text-[10px] ${diffColor}`}>
                    {lang === "zh" ? "与Pro差距" : "Gap to Pro"}: Δ {diffPro.toFixed(1)}{label.unit}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Swing Phase Breakdown (Pro only — needs angles) ── */}
      {userAngles && (
        <div className="glass-card p-5">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-white">
              {lang === "en" ? "Swing Phase Breakdown" : "挥杆阶段分解"}
            </h3>
            <select
              value={phaseAngle}
              onChange={(e) => setPhaseAngle(e.target.value)}
              className="rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-[11px] text-white/70 outline-none"
            >
              {ANGLE_KEYS.map((k) => (
                <option key={k} value={k} className="bg-brand-dark text-white">
                  {lang === "zh" ? ANGLE_LABELS[k].zh : ANGLE_LABELS[k].en}
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-3">
            {Object.entries(PHASE_LABELS).map(([phaseKey, phaseLabel]) => {
              const proPhaseVal = player.phases[phaseKey]?.[phaseAngle] ?? 0;
              const amPhaseVal = AMATEUR_AVG.phases[phaseKey]?.[phaseAngle] ?? 0;

              return (
                <div key={phaseKey}>
                  <div className="mb-1 flex items-center justify-between">
                    <span className="text-xs text-white/60">{lang === "en" ? phaseLabel.en : phaseLabel.zh}</span>
                    <span className="text-[10px] text-white/30">
                      Pro {proPhaseVal}{ANGLE_LABELS[phaseAngle].unit} · {lang === "zh" ? "业余" : "Avg"} {amPhaseVal}{ANGLE_LABELS[phaseAngle].unit}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="relative h-3 flex-1 rounded-full bg-white/5">
                      <div className="absolute h-full rounded-full bg-brand-gold/60" style={{ width: `${Math.min(100, Math.abs(proPhaseVal) * 1.3)}%` }} />
                      <div className="absolute h-full rounded-full bg-white/10" style={{ width: `${Math.min(100, Math.abs(amPhaseVal) * 1.3)}%` }} />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
          <p className="mt-3 text-[10px] text-white/25">
            {lang === "zh"
              ? "金色 = 职业球员 · 灰色 = 业余平均"
              : "Gold = Pro · Gray = Amateur Avg"}
          </p>
        </div>
      )}

      {/* ── Pro Key Metrics ── */}
      <div className="glass-card p-5">
        <h3 className="mb-4 text-sm font-semibold text-white">
          {lang === "en"
            ? `${player.name} vs Amateur Avg`
            : `${player.nameZh} vs 业余平均`}
        </h3>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {Object.entries(player.metrics).map(([key, value]) => {
            const metricLabels: Record<string, { en: string; zh: string }> = {
              club_head_speed: { en: "Club Speed", zh: "杆头速度" },
              swing_tempo: { en: "Tempo", zh: "挥杆节奏" },
              avg_drive: { en: "Avg Drive", zh: "平均开球" },
              accuracy: { en: "Accuracy", zh: "准确率" },
            };
            const ml = metricLabels[key] || { en: key, zh: key };
            const amVal = AMATEUR_AVG.metrics[key] || "-";
            return (
              <div key={key} className="rounded-xl border border-white/5 bg-black/30 p-3 text-center">
                <p className="text-[10px] text-white/40">{lang === "en" ? ml.en : ml.zh}</p>
                <p className="mt-1 text-base font-bold text-brand-gold">{value}</p>
                <p className="mt-0.5 text-[10px] text-white/30">{lang === "zh" ? "业余" : "Avg"}: {amVal}</p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
