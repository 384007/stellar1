"use client";

import { useState, useEffect, useRef } from "react";

interface AnalysisWaitingProps {
  progress: number;
  lang: "en" | "zh";
  mode: "lite" | "pro" | "lab";
}

type TechPoint = {
  titleZh: string;
  tipZh: string;
  titleEn: string;
  tipEn: string;
};

type Persona = {
  nameZh: string;
  nameEn: string;
  skin: string;
  hair: string;
  outfit: string;
  accent: string;
  trim: string;
  focusZh: string;
  focusEn: string;
  variant: number;
  headwear: "cap" | "visor" | "none";
  facialHair?: boolean;
  longHair?: boolean;
  tips: TechPoint[];
};

const TIGER_TIPS: TechPoint[] = [
  { titleZh: "髋先启动", tipZh: "先髋后肩，顺序正确才有爆发。", titleEn: "Hips Lead", tipEn: "Start with hips, then torso and shoulders for power." },
  { titleZh: "手腕滞后", tipZh: "保持 lag 到最后，释放更猛。", titleEn: "Hold Lag", tipEn: "Keep lag deeper into downswing before release." },
  { titleZh: "触球压缩", tipZh: "手在球前，杆身前倾，压缩击球。", titleEn: "Compress Impact", tipEn: "Hands ahead with forward shaft lean at impact." },
  { titleZh: "结束平衡", tipZh: "全力击球后仍能稳住收杆。", titleEn: "Balanced Finish", tipEn: "Even with speed, finish in full balance." },
];

const RORY_TIPS: TechPoint[] = [
  { titleZh: "2:1节奏", tipZh: "上杆两拍，下杆一拍，速度自然出来。", titleEn: "2:1 Tempo", tipEn: "Two-count backswing, one-count downswing rhythm." },
  { titleZh: "一体起杆", tipZh: "肩、手、杆同速启动，稳定平面。", titleEn: "One-Piece Start", tipEn: "Move shoulder, hands and club together in takeaway." },
  { titleZh: "柔中带快", tipZh: "放松手臂，速度来自身体旋转。", titleEn: "Relaxed Speed", tipEn: "Keep arms relaxed; speed comes from body turn." },
  { titleZh: "连贯收杆", tipZh: "不要急停，完整送杆到平衡。", titleEn: "Flow Through", tipEn: "No abrupt stop; swing through into balanced finish." },
];

const NELLY_TIPS: TechPoint[] = [
  { titleZh: "站姿稳定", tipZh: "脚底压力均匀，重心始终可控。", titleEn: "Stable Setup", tipEn: "Keep pressure balanced and centered at setup." },
  { titleZh: "头部平稳", tipZh: "挥杆过程头部不过度上下移动。", titleEn: "Steady Head", tipEn: "Minimize vertical head movement through swing." },
  { titleZh: "顺畅转体", tipZh: "肩髋同步转动，不抢节奏。", titleEn: "Smooth Rotation", tipEn: "Rotate shoulders and hips smoothly without rushing." },
  { titleZh: "优雅收杆", tipZh: "收杆停住3秒，验证动作稳定。", titleEn: "Hold Finish", tipEn: "Hold your finish for 3 seconds to confirm balance." },
];

const LYDIA_TIPS: TechPoint[] = [
  { titleZh: "杆面对准", tipZh: "起杆到触球都要保持杆面管理。", titleEn: "Face Control", tipEn: "Manage clubface from takeaway to impact." },
  { titleZh: "路径稳定", tipZh: "上杆下杆尽量在同一平面。", titleEn: "Path Stability", tipEn: "Keep backswing and downswing on matching planes." },
  { titleZh: "距离触感", tipZh: "用节奏而不是蛮力控制距离。", titleEn: "Distance Feel", tipEn: "Control distance with rhythm, not brute force." },
  { titleZh: "预挥例行", tipZh: "固定 pre-shot routine，稳定命中率。", titleEn: "Routine First", tipEn: "A repeatable pre-shot routine improves consistency." },
];

const CONTROL_TIPS: TechPoint[] = [
  { titleZh: "杆面稳定", tipZh: "杆面稳定比盲目提速更重要。", titleEn: "Stable Face", tipEn: "Face stability matters more than chasing speed." },
  { titleZh: "短杆节奏", tipZh: "距离控制来自节奏和收杆长度。", titleEn: "Distance Tempo", tipEn: "Distance control comes from tempo and finish length." },
  { titleZh: "中心击球", tipZh: "先保证甜蜜点击中，再加速。", titleEn: "Centered Contact", tipEn: "Center the strike before adding speed." },
  { titleZh: "路径重现", tipZh: "每次都重复同一路径，误差才会缩小。", titleEn: "Repeatable Path", tipEn: "Repeat the same path to reduce misses." },
];

const PERSONAS: Persona[] = [
  {
    nameZh: "泰格·伍兹", nameEn: "Tiger Woods", skin: "#8a5a3d", hair: "#1f1f1f", outfit: "#b91c1c", accent: "#d4af37", trim: "#111827",
    focusZh: "下杆序列与爆发", focusEn: "Downswing Sequence & Power", variant: 0, headwear: "cap", tips: TIGER_TIPS,
  },
  {
    nameZh: "罗里·麦克罗伊", nameEn: "Rory McIlroy", skin: "#f4c7a8", hair: "#3d2b1f", outfit: "#1d4ed8", accent: "#7c3aed", trim: "#93c5fd",
    focusZh: "节奏与流畅速度", focusEn: "Tempo & Flowing Speed", variant: 1, headwear: "cap", tips: RORY_TIPS,
  },
  {
    nameZh: "内莉·科达", nameEn: "Nelly Korda", skin: "#f7d7bf", hair: "#2d1b00", outfit: "#5b2dbf", accent: "#ec4899", trim: "#fbcfe8",
    focusZh: "平衡与节奏控制", focusEn: "Balance & Rhythm Control", variant: 2, headwear: "visor", longHair: true, tips: NELLY_TIPS,
  },
  {
    nameZh: "高宝璟", nameEn: "Lydia Ko", skin: "#f3c9ad", hair: "#4a2800", outfit: "#0369a1", accent: "#7c3aed", trim: "#bae6fd",
    focusZh: "精准控制与杆面", focusEn: "Control & Clubface Precision", variant: 3, headwear: "visor", longHair: true, tips: LYDIA_TIPS,
  },
  {
    nameZh: "斯科蒂·舍夫勒", nameEn: "Scottie Scheffler", skin: "#efc2a4", hair: "#5b4638", outfit: "#0f766e", accent: "#a855f7", trim: "#99f6e4",
    focusZh: "重心转移与支撑", focusEn: "Pressure Shift & Stability", variant: 4, headwear: "cap", tips: CONTROL_TIPS,
  },
  {
    nameZh: "乔丹·斯皮思", nameEn: "Jordan Spieth", skin: "#f1c6a6", hair: "#4a2e1a", outfit: "#2563eb", accent: "#f59e0b", trim: "#bfdbfe",
    focusZh: "手感与球路管理", focusEn: "Feel & Shot Control", variant: 5, headwear: "cap", tips: CONTROL_TIPS,
  },
  {
    nameZh: "琼·拉姆", nameEn: "Jon Rahm", skin: "#d9ad8e", hair: "#2d241f", outfit: "#7c2d12", accent: "#d4af37", trim: "#fdba74",
    focusZh: "短而强的爆发动作", focusEn: "Compact Power Move", variant: 6, headwear: "none", facialHair: true, tips: TIGER_TIPS,
  },
  {
    nameZh: "殷若宁", nameEn: "Ruoning Yin", skin: "#f4ceb4", hair: "#1f2937", outfit: "#059669", accent: "#f59e0b", trim: "#86efac",
    focusZh: "平衡出杆与节奏", focusEn: "Balanced Delivery & Tempo", variant: 7, headwear: "visor", longHair: true, tips: NELLY_TIPS,
  },
  {
    nameZh: "布鲁克·亨德森", nameEn: "Brooke Henderson", skin: "#f8d8c0", hair: "#42210b", outfit: "#db2777", accent: "#a855f7", trim: "#f9a8d4",
    focusZh: "大范围转体与延展", focusEn: "Wide Turn & Extension", variant: 8, headwear: "cap", longHair: true, tips: RORY_TIPS,
  },
  {
    nameZh: "科林·森川", nameEn: "Collin Morikawa", skin: "#deb597", hair: "#2f2a25", outfit: "#312e81", accent: "#d4af37", trim: "#c4b5fd",
    focusZh: "铁杆控制与击球质量", focusEn: "Iron Control & Strike Quality", variant: 9, headwear: "cap", tips: LYDIA_TIPS,
  },
  {
    nameZh: "赞德·谢奥菲勒", nameEn: "Xander Schauffele", skin: "#d6ac8e", hair: "#2a2a2a", outfit: "#111827", accent: "#22c55e", trim: "#9ca3af",
    focusZh: "安静上杆与稳定击球", focusEn: "Quiet Takeaway & Stable Strike", variant: 10, headwear: "cap", tips: CONTROL_TIPS,
  },
  {
    nameZh: "贾斯汀·托马斯", nameEn: "Justin Thomas", skin: "#efc3a0", hair: "#3a2417", outfit: "#dc2626", accent: "#fb7185", trim: "#fecaca",
    focusZh: "节奏变化与快速释放", focusEn: "Tempo Change & Release", variant: 11, headwear: "cap", tips: RORY_TIPS,
  },
];

function shuffleIndices(len: number): number[] {
  const arr = Array.from({ length: len }, (_, i) => i);
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function GhibliGolfer({ mode, persona }: { mode: "lite" | "pro" | "lab"; persona: Persona }) {
  const accent = mode === "pro" || mode === "lab" ? "#d4af37" : persona.accent;
  const armPoses = [-8, 4, -2, 8, -5, 6, -10, 2, -4, 7, -1, 5];
  const bodyTilts = [-2, 1, -1, 2, -3, 1, -2, 0, 2, -1, 1, 3];
  const clubOffsets = [-2, 1, 0, 3, -3, 2, -4, 1, 0, 2, -1, 4];
  const shoulderWidths = [20, 18, 16, 17, 21, 18, 22, 16, 17, 19, 20, 18];
  const torsoWidths = [16, 15, 13, 14, 17, 15, 18, 13, 14, 16, 17, 15];
  const headScales = [1.0, 0.96, 1.02, 0.98, 0.97, 0.96, 1.01, 1.02, 1.0, 0.97, 0.98, 0.96];
  const jawShifts = [0, -1, 1, 0, -1, 1, -2, 1, 0, -1, 1, 0];
  const armPose = `rotate(${armPoses[persona.variant % armPoses.length]}deg)`;
  const bodyTilt = bodyTilts[persona.variant % bodyTilts.length];
  const clubOffset = clubOffsets[persona.variant % clubOffsets.length];
  const sleeveY = 128 + (persona.variant % 3);
  const eyebrowTilt = persona.variant % 2 === 0 ? -1.5 : 1.5;
  const shoulderHalf = shoulderWidths[persona.variant % shoulderWidths.length];
  const torsoHalf = torsoWidths[persona.variant % torsoWidths.length];
  const headScale = headScales[persona.variant % headScales.length];
  const jawShift = jawShifts[persona.variant % jawShifts.length];
  return (
    <svg viewBox="0 0 200 260" className="w-36 h-48 sm:w-44 sm:h-60" aria-hidden="true">
      <g className="animate-drift">
        <circle cx="30" cy="80" r="1.5" fill={accent} opacity="0.3" />
        <circle cx="170" cy="60" r="1" fill={accent} opacity="0.2" />
        <circle cx="50" cy="140" r="1.2" fill="#fff" opacity="0.15" />
        <circle cx="160" cy="120" r="1.8" fill={accent} opacity="0.2" />
      </g>
      <g>
        <ellipse cx="100" cy="240" rx="70" ry="8" fill="#1a4d2e" opacity="0.5" />
        {[40, 55, 70, 85, 100, 115, 130, 145, 160].map((x, i) => (
          <path key={i} d={`M${x} 238 Q${x + 2} ${224 - (i % 3) * 4} ${x + 4} 238`}
            stroke="#2d8f5e" strokeWidth="1.5" fill="none" opacity={0.3 + (i % 3) * 0.1}
            className={i % 2 === 0 ? "animate-sway" : "animate-sway-reverse"} />
        ))}
      </g>
      <g className="animate-float">
        {/* Polo + pants */}
        <g transform={`rotate(${bodyTilt} 100 160)`}>
          <path
            d={`M${100 - shoulderHalf} 122 Q${100 - shoulderHalf - 4} 145 ${100 - torsoHalf} 170 L${100 + torsoHalf} 170 Q${100 + shoulderHalf + 4} 145 ${100 + shoulderHalf} 122 Z`}
            fill={persona.outfit}
          />
          <path d="M88 122 L100 134 L112 122" fill="#fff" opacity="0.85" />
          <path d={`M${100 - torsoHalf} 134 L${100 + torsoHalf} 134`} stroke={persona.trim} strokeWidth="2" opacity="0.55" />
          <path d={`M${100 - torsoHalf + 4} 150 L${100 + torsoHalf - 4} 150`} stroke={persona.trim} strokeWidth="1.5" opacity="0.35" />
          <path d={`M${100 - torsoHalf} 170 L${100 + torsoHalf} 170`} stroke="#000" strokeWidth="1.5" opacity="0.18" />
          <rect x="92" y="168" width="16" height="4" rx="1.5" fill="#111827" opacity="0.8" />
          <rect x="99" y="168" width="2.5" height="4" rx="1" fill={accent} opacity="0.8" />
          <path d="M84 170 L80 214 L95 214 L98 172 Z" fill="#1f2937" />
          <path d="M116 170 L102 172 L105 214 L120 214 Z" fill="#111827" />
          <path d={`M${100 - shoulderHalf} ${sleeveY} Q${100 - shoulderHalf - 4} 135 ${100 - shoulderHalf + 1} 142`} stroke={persona.trim} strokeWidth="3" opacity="0.55" />
          <path d={`M${100 + shoulderHalf} ${sleeveY} Q${100 + shoulderHalf + 4} 135 ${100 + shoulderHalf - 1} 142`} stroke={persona.trim} strokeWidth="3" opacity="0.55" />
        </g>
        <line x1="88" y1="210" x2="85" y2="238" stroke={persona.skin} strokeWidth="5" strokeLinecap="round" />
        <line x1="112" y1="210" x2="115" y2="238" stroke={persona.skin} strokeWidth="5" strokeLinecap="round" />
        <ellipse cx="83" cy="240" rx="7" ry="3" fill={accent} opacity="0.8" />
        <ellipse cx="117" cy="240" rx="7" ry="3" fill={accent} opacity="0.8" />
        <g className="animate-swing-arm" style={{ transformOrigin: "100px 140px", transform: armPose }}>
          <line x1="85" y1="125" x2="65" y2="155" stroke={persona.skin} strokeWidth="4.5" strokeLinecap="round" />
          <line x1="65" y1="155" x2="55" y2="180" stroke={persona.skin} strokeWidth="4" strokeLinecap="round" />
          <path d="M55 178 Q58 182 54 186" stroke="#f8fafc" strokeWidth="2.5" strokeLinecap="round" opacity="0.9" />
          <line x1="55" y1="178" x2={45 + clubOffset} y2="235" stroke="#888" strokeWidth="2" strokeLinecap="round" />
          <path d={`M${42 + clubOffset} 233 Q${38 + clubOffset} 238 ${45 + clubOffset} 240 Q${48 + clubOffset} 238 ${46 + clubOffset} 234 Z`} fill="#aaa" />
        </g>
        <line x1="115" y1="125" x2="130" y2="148" stroke={persona.skin} strokeWidth="4.5" strokeLinecap="round" />
        <line x1="130" y1="148" x2="60" y2="176" stroke={persona.skin} strokeWidth="4" strokeLinecap="round" />
        <g transform={`translate(100 98) scale(${headScale}) translate(-100 -98)`}>
          <path d={`M79 94 Q75 64 100 57 Q125 64 121 94 Q118 115 ${100 + jawShift} 120 Q82 115 79 94 Z`} fill={persona.skin} />
          <path d="M76 95 Q72 60 100 55 Q128 60 124 95" fill={persona.hair} />
        </g>
        {persona.longHair && (
          <>
            <path d="M74 96 Q66 112 70 136 Q72 140 75 137 Q72 118 78 102 Z" fill={persona.hair} className="animate-hair-reverse" />
            <path d="M126 96 Q134 112 130 136 Q128 140 125 137 Q128 118 122 102 Z" fill={persona.hair} className="animate-hair" />
            <path d="M110 62 Q125 55 135 70 Q140 85 130 105 Q125 108 128 95 Q132 78 120 68 Z"
              fill={persona.hair} className="animate-hair" />
          </>
        )}
        {persona.headwear === "cap" && (
          <>
            <path d="M76 86 Q100 82 124 86 Q122 80 100 78 Q78 80 76 86 Z" fill={accent} opacity="0.9" />
            <path d="M74 86 Q61 85 55 90" stroke={accent} strokeWidth="2.4" fill="none" opacity="0.75" />
            <path d="M84 82 L116 82" stroke="#fff" strokeWidth="1.2" opacity="0.28" />
          </>
        )}
        {persona.headwear === "visor" && (
          <>
            <path d="M79 84 Q100 80 121 84 Q119 78 100 77 Q81 78 79 84 Z" fill={accent} opacity="0.88" />
            <path d="M78 85 Q64 83 57 88" stroke={accent} strokeWidth="2.6" fill="none" opacity="0.85" />
            <path d="M86 81 L114 81" stroke="#fff" strokeWidth="1" opacity="0.35" />
          </>
        )}
        <ellipse cx="91" cy="98" rx="4" ry="5" fill="#2d1b4e" />
        <ellipse cx="109" cy="98" rx="4" ry="5" fill="#2d1b4e" />
        <circle cx="92" cy="96" r="1.5" fill="#fff" />
        <circle cx="110" cy="96" r="1.5" fill="#fff" />
        <path d={`M86 92 Q91 ${90 + eyebrowTilt} 96 92`} stroke={persona.hair} strokeWidth="1.2" fill="none" opacity="0.8" />
        <path d={`M104 92 Q109 ${90 - eyebrowTilt} 114 92`} stroke={persona.hair} strokeWidth="1.2" fill="none" opacity="0.8" />
        {persona.facialHair && (
          <>
            <path d="M93 109 Q100 113 107 109" fill="none" stroke="#5b4638" strokeWidth="1.4" strokeLinecap="round" />
            <path d="M96 105 Q100 107 104 105" fill="none" stroke="#5b4638" strokeWidth="1" strokeLinecap="round" opacity="0.9" />
          </>
        )}
        <path d={`M98 101 Q100 104 ${102 + jawShift * 0.4} 101`} stroke="#b98973" strokeWidth="0.9" fill="none" opacity="0.7" />
        <path d="M96 108 Q100 111 104 108" fill="none" stroke="#84594d" strokeWidth="1.2" strokeLinecap="round" />
      </g>
      <g className="animate-sparkle-slow">
        <circle cx="35" cy="100" r="2" fill={accent} opacity="0.4" />
        <circle cx="165" cy="90" r="1.5" fill={accent} opacity="0.3" />
      </g>
    </svg>
  );
}

export default function AnalysisWaiting({ progress, lang, mode }: AnalysisWaitingProps) {
  const [currentTip, setCurrentTip] = useState(0);
  const [currentPersona, setCurrentPersona] = useState(0);
  const [fadeState, setFadeState] = useState<"in" | "out">("in");
  const [elapsed, setElapsed] = useState(0);
  const tipQueueRef = useRef<number[]>([]);
  const personaQueueRef = useRef<number[]>([]);

  useEffect(() => {
    const t = setInterval(() => setElapsed(e => e + 1), 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const personaStart = Math.floor(Math.random() * PERSONAS.length);
    setCurrentPersona(personaStart);
    const personaTipsLen = PERSONAS[personaStart].tips.length;
    const tipStart = Math.floor(Math.random() * personaTipsLen);
    setCurrentTip(tipStart);
    tipQueueRef.current = shuffleIndices(personaTipsLen).filter((i) => i !== tipStart);
    personaQueueRef.current = shuffleIndices(PERSONAS.length).filter((i) => i !== personaStart);
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setFadeState("out");
      setTimeout(() => {
        setCurrentPersona((prevPersona) => {
          if (personaQueueRef.current.length === 0) {
            personaQueueRef.current = shuffleIndices(PERSONAS.length).filter((i) => i !== prevPersona);
          }
          const nextPersona = personaQueueRef.current.shift() ?? prevPersona;
          const tipsLen = PERSONAS[nextPersona].tips.length;
          const nextTip = Math.floor(Math.random() * tipsLen);
          setCurrentTip(nextTip);
          tipQueueRef.current = shuffleIndices(tipsLen).filter((i) => i !== nextTip);
          return nextPersona;
        });
        setFadeState("in");
      }, 400);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const accent = mode === "pro" ? "#d4af37" : mode === "lab" ? "#d4af37" : "#7c3aed";
  const accentClass = mode === "pro" || mode === "lab" ? "text-brand-gold" : "text-brand-purple";
  const accentBg = mode === "pro" || mode === "lab" ? "bg-brand-gold" : "bg-brand-purple";
  const persona = PERSONAS[currentPersona % PERSONAS.length];
  const currentTips = persona.tips;
  const currentPoint = currentTips[currentTip % currentTips.length];
  const nextPointIdx =
    tipQueueRef.current.length > 0
      ? tipQueueRef.current[0]
      : (currentTip + 1) % currentTips.length;
  const nextPoint = currentTips[nextPointIdx];

  const stageLabels = mode === "pro"
    ? (lang === "zh"
      ? [
        { max: 15, text: "提取高清帧..." },
        { max: 30, text: "完整骨架映射..." },
        { max: 50, text: "计算生物力学..." },
        { max: 75, text: "AI 深度分析中..." },
        { max: 100, text: "生成Pro报告..." },
      ]
      : [
        { max: 15, text: "Extracting HD frames..." },
        { max: 30, text: "Full skeleton mapping..." },
        { max: 50, text: "Computing biomechanics..." },
        { max: 75, text: "AI deep analysis..." },
        { max: 100, text: "Generating Pro report..." },
      ])
    : mode === "lab"
    ? (lang === "zh"
      ? [
        { max: 15, text: "上传视频至分析引擎..." },
        { max: 30, text: "检测击球时刻与事件..." },
        { max: 50, text: "提取轨迹与起飞参数..." },
        { max: 70, text: "Shot Lab AI 深度分析..." },
        { max: 88, text: "评估动作问题与节奏..." },
        { max: 100, text: "生成 Shot Lab 报告..." },
      ]
      : [
        { max: 15, text: "Uploading to analysis engine..." },
        { max: 30, text: "Detecting impact event..." },
        { max: 50, text: "Extracting launch & trajectory..." },
        { max: 70, text: "Shot Lab AI deep analysis..." },
        { max: 88, text: "Evaluating issues & tempo..." },
        { max: 100, text: "Generating Shot Lab report..." },
      ])
    : (lang === "zh"
      ? [
        { max: 25, text: "上传文件..." },
        { max: 50, text: "AI 分析中..." },
        { max: 80, text: "生成报告..." },
        { max: 100, text: "即将完成..." },
      ]
      : [
        { max: 25, text: "Uploading..." },
        { max: 50, text: "AI analyzing..." },
        { max: 80, text: "Generating report..." },
        { max: 100, text: "Almost done..." },
      ]);

  const stageText = stageLabels.find((s) => progress < s.max)?.text || stageLabels[stageLabels.length - 1].text;

  return (
    <div className="flex flex-col items-center py-8 animate-fade-in">
      {/* Top spinner */}
      <div className="mb-4 flex flex-col items-center">
        <div className="relative h-20 w-20">
          <svg className="h-20 w-20 -rotate-90" viewBox="0 0 96 96">
            <circle cx="48" cy="48" r="42" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="6" />
            <circle cx="48" cy="48" r="42" fill="none" stroke={accent} strokeWidth="6"
              strokeDasharray={`${progress * 2.64} 263.9`} strokeLinecap="round"
              className="transition-all duration-700" />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className={`text-base font-bold ${accentClass}`}>{Math.round(progress)}%</span>
          </div>
        </div>
      </div>

      {/* Status */}
      <h2 className="mb-1 text-base font-semibold text-white">
        {mode === "pro"
          ? (lang === "zh" ? "Pro 深度分析中" : "Pro Analysis in Progress")
          : mode === "lab"
          ? (lang === "zh" ? "Shot Lab 分析中" : "Shot Lab Analyzing...")
          : (lang === "zh" ? "智能分析中" : "Analyzing...")}
      </h2>
      <p className="mb-1 text-xs text-white/40">{stageText}</p>
      <div className="mb-4 flex items-center gap-2">
        <span className="text-[10px] text-white/25 font-mono tabular-nums">
          {Math.floor(elapsed / 60).toString().padStart(2, "0")}:{(elapsed % 60).toString().padStart(2, "0")}
        </span>
        {elapsed > 90 && (
          <span className="text-[10px] text-amber-400/70 animate-pulse">
            {lang === "zh" ? "· 仍在工作中，请耐心等待" : "· Still working, please wait"}
          </span>
        )}
      </div>

      {/* Middle character carousel */}
      <div className="mb-4 w-full max-w-md">
        <div className="glass-card overflow-hidden px-4 py-3">
          <div
            className={`flex flex-col items-center transition-all duration-500 ${
              fadeState === "in" ? "opacity-100 translate-y-0" : "opacity-0 translate-y-3"
            }`}
          >
            <GhibliGolfer mode={mode} persona={persona} />
            <p className="mt-1 text-[11px] text-white/35">
              {lang === "zh" ? persona.nameZh : persona.nameEn}
            </p>
            <p className="text-[10px] text-white/25">
              {lang === "zh" ? persona.focusZh : persona.focusEn}
            </p>
          </div>
        </div>
      </div>

      {/* Bottom scrolling tips synced with character */}
      <div className="w-full max-w-md">
        <div className="glass-card overflow-hidden">
          {/* Header */}
          <div className="flex items-center gap-2 border-b border-white/5 px-4 py-2.5">
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke={accent} strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 18v-5.25m0 0a6.01 6.01 0 0 0 1.5-.189m-1.5.189a6.01 6.01 0 0 1-1.5-.189m3.75 7.478a12.06 12.06 0 0 1-4.5 0m3.75 2.383a14.406 14.406 0 0 1-3 0M14.25 18v-.192c0-.983.658-1.823 1.508-2.316a7.5 7.5 0 1 0-7.517 0c.85.493 1.509 1.333 1.509 2.316V18" />
            </svg>
            <div className="ml-auto flex gap-1">
              <span className="rounded-full border border-white/10 px-2 py-0.5 text-[9px] text-white/28">
                {lang === "zh" ? persona.focusZh : persona.focusEn}
              </span>
            </div>
          </div>

          {/* Tip Content */}
          <div className="relative overflow-hidden" style={{ minHeight: 120 }}>
            <div
              className={`px-5 py-4 transition-all duration-400 ${
                fadeState === "in" ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-3"
              }`}
            >
              <div className="mb-2 flex items-center gap-2">
                <span className={`inline-block h-1.5 w-1.5 rounded-full ${accentBg}`} />
                <span className={`text-sm font-bold ${accentClass}`}>
                  {lang === "zh" ? currentPoint.titleZh : currentPoint.titleEn}
                </span>
                <span className="ml-auto rounded-full border border-white/5 bg-white/[0.03] px-2 py-0.5 text-[9px] text-white/25">
                  {lang === "zh" ? `${persona.nameZh} 推荐` : `${persona.nameEn} Pick`}
                </span>
              </div>
              <p className="text-sm leading-relaxed text-white/55">
                {lang === "zh" ? currentPoint.tipZh : currentPoint.tipEn}
              </p>
            </div>

            {/* Next preview */}
            <div className="border-t border-white/[0.03] px-5 py-2.5 flex items-center gap-2 opacity-40">
              <span className="text-[10px] text-white/30">
                {lang === "zh" ? "下一条" : "Next"}:
              </span>
              <span className="text-[11px] text-white/30 truncate">
                {lang === "zh" ? nextPoint.titleZh : nextPoint.titleEn}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* CSS animations */}
      <style jsx>{`
        @keyframes float {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-6px); }
        }
        @keyframes spin-slow {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        @keyframes sway {
          0%, 100% { transform: rotate(0deg); }
          50% { transform: rotate(3deg); }
        }
        @keyframes sway-reverse {
          0%, 100% { transform: rotate(0deg); }
          50% { transform: rotate(-3deg); }
        }
        @keyframes swing-arm {
          0%, 100% { transform: rotate(0deg); }
          25% { transform: rotate(-2deg); }
          75% { transform: rotate(2deg); }
        }
        @keyframes hair {
          0%, 100% { transform: rotate(0deg); }
          50% { transform: rotate(2deg) translateX(1px); }
        }
        @keyframes hair-reverse {
          0%, 100% { transform: rotate(0deg); }
          50% { transform: rotate(-2deg) translateX(-1px); }
        }
        @keyframes sparkle {
          0%, 100% { opacity: 0.6; transform: scale(1); }
          50% { opacity: 1; transform: scale(1.4); }
        }
        @keyframes sparkle-slow {
          0%, 100% { opacity: 0.2; }
          50% { opacity: 0.5; }
        }
        @keyframes drift {
          0% { transform: translateX(0) translateY(0); }
          50% { transform: translateX(8px) translateY(-4px); }
          100% { transform: translateX(0) translateY(0); }
        }
        :global(.animate-float) { animation: float 3s ease-in-out infinite; }
        :global(.animate-sway) { animation: sway 2.5s ease-in-out infinite; transform-origin: bottom; }
        :global(.animate-sway-reverse) { animation: sway-reverse 3s ease-in-out infinite; transform-origin: bottom; }
        :global(.animate-swing-arm) { animation: swing-arm 4s ease-in-out infinite; transform-origin: 85px 125px; }
        :global(.animate-hair) { animation: hair 3s ease-in-out infinite; transform-origin: top; }
        :global(.animate-hair-reverse) { animation: hair-reverse 3.5s ease-in-out infinite; transform-origin: top; }
        :global(.animate-sparkle) { animation: sparkle 2s ease-in-out infinite; }
        :global(.animate-sparkle-slow) { animation: sparkle-slow 4s ease-in-out infinite; }
        :global(.animate-drift) { animation: drift 6s ease-in-out infinite; }
        :global(.animate-spin-slow) { animation: spin-slow 2.8s linear infinite; }
      `}</style>
    </div>
  );
}
