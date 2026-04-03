export interface ProPlayerData {
  id: string;
  name: string;
  nameZh: string;
  era: string;
  nationality: string;
  scores: {
    grip: number;
    stance: number;
    backswing: number;
    downswing: number;
    follow_through: number;
  };
  angles: {
    shoulder_rotation: number;
    hip_rotation: number;
    x_factor: number;
    spine_tilt: number;
    left_knee: number;
    right_knee: number;
    left_elbow: number;
    right_elbow: number;
  };
  metrics: {
    club_head_speed_mph: number;
    ball_speed_mph: number;
    swing_tempo: string;
    avg_drive_yards: number;
    accuracy_pct: number;
    launch_angle: number;
    spin_rate: number;
    smash_factor: number;
  };
  signature_move: string;
  signature_move_zh: string;
}

export const PRO_PLAYERS: ProPlayerData[] = [
  {
    id: "tiger_woods_2000",
    name: "Tiger Woods",
    nameZh: "泰格·伍兹",
    era: "2000 Peak Season",
    nationality: "USA",
    scores: { grip: 98, stance: 97, backswing: 99, downswing: 99, follow_through: 98 },
    angles: {
      shoulder_rotation: -42.5,
      hip_rotation: -28.3,
      x_factor: 55.2,
      spine_tilt: 6.8,
      left_knee: 145.0,
      right_knee: 155.0,
      left_elbow: 170.0,
      right_elbow: 85.0,
    },
    metrics: {
      club_head_speed_mph: 125,
      ball_speed_mph: 180,
      swing_tempo: "3.2:1",
      avg_drive_yards: 298,
      accuracy_pct: 73,
      launch_angle: 11.2,
      spin_rate: 2680,
      smash_factor: 1.49,
    },
    signature_move: "Stinger low iron shot with controlled ball flight",
    signature_move_zh: "低弹道铁杆击球，球路控制精准",
  },
  {
    id: "rory_mcilroy",
    name: "Rory McIlroy",
    nameZh: "罗里·麦克罗伊",
    era: "Current (2024-2026)",
    nationality: "Northern Ireland",
    scores: { grip: 96, stance: 95, backswing: 98, downswing: 99, follow_through: 97 },
    angles: {
      shoulder_rotation: -45.0,
      hip_rotation: -30.0,
      x_factor: 58.0,
      spine_tilt: 5.5,
      left_knee: 148.0,
      right_knee: 158.0,
      left_elbow: 175.0,
      right_elbow: 82.0,
    },
    metrics: {
      club_head_speed_mph: 122,
      ball_speed_mph: 184,
      swing_tempo: "3.0:1",
      avg_drive_yards: 314,
      accuracy_pct: 65,
      launch_angle: 10.8,
      spin_rate: 2450,
      smash_factor: 1.51,
    },
    signature_move: "Explosive hip rotation with maximum X-factor separation",
    signature_move_zh: "爆发性髋部旋转，X因子分离最大化",
  },
  {
    id: "shin_ji_ae",
    name: "Shin Ji-ae",
    nameZh: "申智爱",
    era: "Peak Season",
    nationality: "South Korea",
    scores: { grip: 97, stance: 96, backswing: 95, downswing: 96, follow_through: 97 },
    angles: {
      shoulder_rotation: -38.0,
      hip_rotation: -25.0,
      x_factor: 48.0,
      spine_tilt: 4.2,
      left_knee: 152.0,
      right_knee: 160.0,
      left_elbow: 172.0,
      right_elbow: 90.0,
    },
    metrics: {
      club_head_speed_mph: 94,
      ball_speed_mph: 140,
      swing_tempo: "3.3:1",
      avg_drive_yards: 255,
      accuracy_pct: 78,
      launch_angle: 13.5,
      spin_rate: 2800,
      smash_factor: 1.48,
    },
    signature_move: "Textbook tempo with exceptional accuracy and consistency",
    signature_move_zh: "教科书级节奏，精准度和稳定性出众",
  },
];

export function getPlayerById(id: string): ProPlayerData | undefined {
  return PRO_PLAYERS.find((p) => p.id === id);
}

export function compareWithPro(
  userScores: Record<string, number>,
  userAngles: Record<string, number>,
  proId: string
): {
  scoreDiffs: Record<string, number>;
  angleDiffs: Record<string, number>;
  overallMatch: number;
} {
  const pro = getPlayerById(proId);
  if (!pro) {
    return { scoreDiffs: {}, angleDiffs: {}, overallMatch: 0 };
  }

  const scoreDiffs: Record<string, number> = {};
  let totalDiff = 0;
  let count = 0;

  for (const [key, proVal] of Object.entries(pro.scores)) {
    const userVal = userScores[key] || 0;
    scoreDiffs[key] = userVal - proVal;
    totalDiff += Math.abs(userVal - proVal);
    count++;
  }

  const angleDiffs: Record<string, number> = {};
  for (const [key, proVal] of Object.entries(pro.angles)) {
    const userVal = userAngles[key] || 0;
    angleDiffs[key] = userVal - proVal;
  }

  const avgDiff = count > 0 ? totalDiff / count : 0;
  const overallMatch = Math.max(0, 100 - avgDiff);

  return { scoreDiffs, angleDiffs, overallMatch: Math.round(overallMatch) };
}
