/**
 * Client-side prov3「骨架 + 视频时间线」依据说明：从已有分析 JSON 生成结构化要点，
 * 不调用新 API；高信任与低信任均输出（低信任由 UI 另行标注）。
 */

export type Prov3MotionReportSection = {
  id: string;
  title_zh: string;
  title_en: string;
  bullets_zh: string[];
  bullets_en: string[];
};

type PoseLike = {
  frame_index: number;
  timestamp: number;
  angles?: Record<string, number>;
  joints?: Array<{ name: string; visibility: number }>;
  phase_data?: {
    phase_id?: string;
    phase_zh?: string;
    phase_en?: string;
    progress_pct?: number;
  };
};

/** 全部字段可选；报告只输出「JSON 里确有值」的句子，不编造。 */
type ReportInput = {
  pose_frames?: PoseLike[] | null;
  video_meta?: {
    fps?: number;
    duration_s?: number;
    source_frame_count?: number;
    total_pose_frames?: number;
  } | null;
  prediction?: Record<string, unknown> | null;
  swing_phase_evaluations?: Array<{
    phase: string;
    status: string;
    note_zh?: string;
    note_en?: string;
  }> | null;
  keyframes_strip?: {
    timeline?: string;
    analysis_fps?: string | number;
    thumbnails_from_analysis_video?: boolean;
  } | null;
  phase_keyframes?: Record<string, number> | null;
  /** 与条图一致的关键帧行（phase / 时间戳 / frame_index 等来自后端） */
  keyframes_display?: Array<{
    phase?: string;
    label_zh?: string;
    label_en?: string;
    timestamp?: number;
    frame_index?: number;
    display_source_frame_index?: number;
    analysis_timestamp?: number;
  }> | null;
  core_frame_scores?: Record<
    string,
    {
      score?: number | null;
      pass_90?: boolean | null;
      confidence?: number | null;
      comment_zh?: string;
      comment_en?: string;
    }
  > | null;
  gemini_frame_notes?: Array<{
    index: number;
    label?: string | null;
    note_zh?: string;
    note_en?: string;
  }> | null;
  issues?: string[] | null;
  issues_zh?: string[] | null;
  summary?: string | null;
  summary_zh?: string | null;
};

const PHASE_ORDER = [
  "address",
  "takeaway",
  "backswing",
  "top",
  "downswing",
  "impact",
  "follow_through",
  "finish",
] as const;

function num(v: unknown): number | null {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  return null;
}

function pickFrameForPhase(frames: PoseLike[], phase: string): PoseLike | null {
  const pid = phase.toLowerCase();
  const byPhase = frames.filter(
    (f) => String(f.phase_data?.phase_id ?? "").toLowerCase() === pid,
  );
  if (byPhase.length > 0) {
    return byPhase[Math.floor(byPhase.length / 2)];
  }
  return null;
}

function pickImpactish(frames: PoseLike[]): PoseLike | null {
  const impact = pickFrameForPhase(frames, "impact");
  if (impact) return impact;
  const down = pickFrameForPhase(frames, "downswing");
  if (down) return down;
  if (frames.length === 0) return null;
  return frames[Math.min(frames.length - 1, Math.floor(frames.length * 0.72))];
}

function pickTopish(frames: PoseLike[]): PoseLike | null {
  const top = pickFrameForPhase(frames, "top");
  if (top) return top;
  if (frames.length === 0) return null;
  return frames[Math.floor(frames.length * 0.45)];
}

function formatAngles(f: PoseLike | null): string {
  if (!f?.angles) return "";
  const a = f.angles;
  const parts: string[] = [];
  const xf = num(a.x_factor);
  const sr = num(a.shoulder_rotation);
  const hr = num(a.hip_rotation);
  const st = num(a.spine_tilt);
  if (xf != null) parts.push(`X-Factor ${xf.toFixed(1)}°`);
  if (sr != null) parts.push(`肩旋转 ${sr.toFixed(1)}°`);
  if (hr != null) parts.push(`髋旋转 ${hr.toFixed(1)}°`);
  if (st != null) parts.push(`脊柱倾角 ${st.toFixed(1)}°`);
  return parts.join(" · ");
}

function formatAnglesEn(f: PoseLike | null): string {
  if (!f?.angles) return "";
  const a = f.angles;
  const parts: string[] = [];
  const xf = num(a.x_factor);
  const sr = num(a.shoulder_rotation);
  const hr = num(a.hip_rotation);
  const st = num(a.spine_tilt);
  if (xf != null) parts.push(`X-factor ${xf.toFixed(1)}°`);
  if (sr != null) parts.push(`Shoulder rot. ${sr.toFixed(1)}°`);
  if (hr != null) parts.push(`Hip rot. ${hr.toFixed(1)}°`);
  if (st != null) parts.push(`Spine tilt ${st.toFixed(1)}°`);
  return parts.join(" · ");
}

/** 骨架 angles 对象里所有有限数值，按键名排序后原样列出（无则空数组）。 */
function allAnglePairs(f: PoseLike | null): { zh: string[]; en: string[] } {
  if (!f?.angles) return { zh: [], en: [] };
  const entries = Object.entries(f.angles)
    .filter(([, v]) => num(v) != null)
    .sort(([a], [b]) => a.localeCompare(b));
  const zh = entries.map(([k, v]) => `${k} ${num(v)!.toFixed(1)}°`);
  const en = entries.map(([k, v]) => `${k} ${num(v)!.toFixed(1)}°`);
  return { zh, en };
}

function visibilitySummaryZh(f: PoseLike | null): string[] {
  const j = f?.joints;
  if (!Array.isArray(j) || j.length === 0) return [];
  const vis = j.map((x) => num(x.visibility)).filter((v): v is number => v != null);
  if (vis.length === 0) return [];
  const mean = vis.reduce((a, b) => a + b, 0) / vis.length;
  const pick = (name: string) => j.find((x) => String(x.name).toLowerCase().includes(name));
  const lw = pick("left_wrist") ?? pick("wrist");
  const rw = pick("right_wrist");
  const parts = [`本帧共 ${j.length} 个关节点，可见度均值约 ${mean.toFixed(2)}（0–1）。`];
  if (lw && num(lw.visibility) != null) parts.push(`左腕可见度约 ${num(lw.visibility)!.toFixed(2)}。`);
  if (rw && num(rw.visibility) != null) parts.push(`右腕可见度约 ${num(rw.visibility)!.toFixed(2)}。`);
  return parts;
}

function visibilitySummaryEn(f: PoseLike | null): string[] {
  const j = f?.joints;
  if (!Array.isArray(j) || j.length === 0) return [];
  const vis = j.map((x) => num(x.visibility)).filter((v): v is number => v != null);
  if (vis.length === 0) return [];
  const mean = vis.reduce((a, b) => a + b, 0) / vis.length;
  const pick = (name: string) => j.find((x) => String(x.name).toLowerCase().includes(name));
  const lw = pick("left_wrist") ?? pick("wrist");
  const rw = pick("right_wrist");
  const parts = [`${j.length} joints; mean visibility ~${mean.toFixed(2)} (0–1).`];
  if (lw && num(lw.visibility) != null) parts.push(`Left wrist visibility ~${num(lw.visibility)!.toFixed(2)}.`);
  if (rw && num(rw.visibility) != null) parts.push(`Right wrist visibility ~${num(rw.visibility)!.toFixed(2)}.`);
  return parts;
}

function uniquePhasesFromFrames(frames: PoseLike[]): string[] {
  const seen = new Set<string>();
  const order: string[] = [];
  for (const f of frames) {
    const id = String(f.phase_data?.phase_id ?? "").trim();
    if (!id || seen.has(id)) continue;
    seen.add(id);
    order.push(id);
  }
  return order.sort(
    (a, b) =>
      PHASE_ORDER.indexOf(a as (typeof PHASE_ORDER)[number]) -
      PHASE_ORDER.indexOf(b as (typeof PHASE_ORDER)[number]),
  );
}

/**
 * 由当前结果对象生成报告区块（纯数据驱动）。
 */
export function buildProv3MotionEvidenceReport(input: ReportInput): Prov3MotionReportSection[] {
  const frames = Array.isArray(input.pose_frames)
    ? [...input.pose_frames].sort((a, b) => a.frame_index - b.frame_index)
    : [];
  const sections: Prov3MotionReportSection[] = [];

  if (frames.length === 0) {
    sections.push({
      id: "no_pose",
      title_zh: "骨架序列",
      title_en: "Pose sequence",
      bullets_zh: ["本结果暂无逐帧骨架，无法根据骨架与视频叠加生成要点。"],
      bullets_en: ["No per-frame pose in this result—no skeleton-overlay report."],
    });
    return sections;
  }

  const n = frames.length;
  const t0 = frames[0]?.timestamp ?? 0;
  const t1 = frames[n - 1]?.timestamp ?? t0;
  const span = Math.max(0, t1 - t0);
  const durMeta = num(input.video_meta?.duration_s);
  const srcFc = num(input.video_meta?.source_frame_count);
  const fps = num(input.video_meta?.fps);

  sections.push({
    id: "coverage",
    title_zh: "时间线与采样",
    title_en: "Timeline & sampling",
    bullets_zh: [
      `姿态管线共输出 ${n} 个采样时刻，帧索引约从 ${frames[0]?.frame_index ?? 0} 到 ${frames[n - 1]?.frame_index ?? 0}。`,
      durMeta != null
        ? `分析视频时长约 ${durMeta.toFixed(2)} s${fps != null ? `，标称 ${fps.toFixed(1)} fps` : ""}。`
        : span > 0.05
          ? `姿态时间跨度约 ${span.toFixed(2)} s（由时间戳差推算）。`
          : "时间戳跨度较短或为单帧簇，请以视频 scrub 与关键帧条为准对齐观察。",
      srcFc != null && srcFc > 0
        ? `与 OpenCV 时间线对齐的源帧数约 ${srcFc}，用于视频条与骨架帧索引对齐（见「视频分析」叠加）。`
        : "建议在「视频分析」中与叠加骨架一并拖动时间线，核对挥杆节奏。",
      input.keyframes_strip?.timeline
        ? `关键帧条时间基准：${String(input.keyframes_strip.timeline)}${
            input.keyframes_strip.thumbnails_from_analysis_video ? "（缩略图来自分析视频解码）" : ""
          }。`
        : "关键帧条与姿态序列来自同一真 240 分析时间线，可交叉对照顶/击球/送杆等相位。",
    ],
    bullets_en: [
      `Pose pipeline emitted ${n} samples, frame_index roughly ${frames[0]?.frame_index ?? 0} → ${frames[n - 1]?.frame_index ?? 0}.`,
      durMeta != null
        ? `Analysis video duration ~${durMeta.toFixed(2)} s${fps != null ? ` at ~${fps.toFixed(1)} fps` : ""}.`
        : span > 0.05
          ? `Pose timestamp span ~${span.toFixed(2)} s (from timestamps).`
          : "Timestamps are tight or clustered—use the video scrubber and keyframe strip for rhythm.",
      srcFc != null && srcFc > 0
        ? `Source frame count ~${srcFc} (OpenCV timeline) aligns the scrubber with pose frame_index (see Video tab overlay).`
        : "Scrub the timeline in Video with the skeleton overlay to verify tempo.",
      input.keyframes_strip?.timeline
        ? `Keyframe strip timeline: ${String(input.keyframes_strip.timeline)}${
            input.keyframes_strip.thumbnails_from_analysis_video ? " (thumbnails from analysis video)" : ""
          }.`
        : "Keyframes and poses share the same true-240 analysis timeline—cross-check top, impact, and release.",
    ],
  });

  const tpf = num(input.video_meta?.total_pose_frames);
  if (tpf != null && tpf > 0) {
    sections.push({
      id: "meta_pose_count",
      title_zh: "元数据与姿态条数",
      title_en: "Meta vs pose count",
      bullets_zh: [
        `video_meta.total_pose_frames = ${Math.round(tpf)}；当前 pose_frames 采样 ${n} 帧。${
          Math.round(tpf) !== n ? "二者不一致时以本结果内可渲染的 pose_frames 为准。" : ""
        }`,
      ],
      bullets_en: [
        `video_meta.total_pose_frames = ${Math.round(tpf)}; pose_frames samples = ${n}.${
          Math.round(tpf) !== n ? " If they differ, trust the pose_frames present in this payload." : ""
        }`,
      ],
    });
  }

  const kfd = Array.isArray(input.keyframes_display) ? input.keyframes_display : [];
  if (kfd.length > 0) {
    const fmtRow = (k: (typeof kfd)[0], zh: boolean) => {
      const phase = String(k.phase ?? k.label_zh ?? k.label_en ?? "?");
      const ts = num(k.timestamp);
      const fi = num(k.frame_index);
      const dsi = num(k.display_source_frame_index);
      const ats = num(k.analysis_timestamp);
      const parts: string[] = zh ? [`相位 ${phase}`] : [`Phase ${phase}`];
      if (ts != null) parts.push(zh ? `时间 ${ts.toFixed(3)} s` : `t ${ts.toFixed(3)} s`);
      if (fi != null) parts.push(zh ? `pose frame_index ${Math.round(fi)}` : `pose frame_index ${Math.round(fi)}`);
      if (dsi != null) parts.push(zh ? `条图源帧 ${Math.round(dsi)}` : `strip src frame ${Math.round(dsi)}`);
      if (ats != null) parts.push(zh ? `analysis_ts ${ats.toFixed(3)}` : `analysis_ts ${ats.toFixed(3)}`);
      return parts.join(zh ? "；" : "; ");
    };
    sections.push({
      id: "keyframe_rows",
      title_zh: "关键帧条（与界面同源）",
      title_en: "Keyframe strip (same as UI)",
      bullets_zh: [
        `共 ${kfd.length} 行；下列为前 ${Math.min(12, kfd.length)} 条后端字段（用于与视频 scrub、骨架叠加对照）。`,
        ...kfd.slice(0, 12).map((row) => fmtRow(row, true)),
      ],
      bullets_en: [
        `${kfd.length} row(s); first ${Math.min(12, kfd.length)} entries from payload (cross-check scrubber & overlay).`,
        ...kfd.slice(0, 12).map((row) => fmtRow(row, false)),
      ],
    });
  }

  const pk = input.phase_keyframes;
  if (pk && typeof pk === "object" && Object.keys(pk).length > 0) {
    const rows = Object.entries(pk).map(([phase, idx]) => `${phase} → pose_frames[${idx}]`);
    sections.push({
      id: "phase_index_map",
      title_zh: "相位 → 姿态索引",
      title_en: "Phase → pose index",
      bullets_zh: rows,
      bullets_en: rows,
    });
  }

  const phases = uniquePhasesFromFrames(frames);
  if (phases.length > 0) {
    sections.push({
      id: "phases",
      title_zh: "相位覆盖（骨架）",
      title_en: "Phase coverage (pose)",
      bullets_zh: [
        `在姿态序列中检测到相位标签：${phases.join(" → ")}。`,
        "上述相位来自逐帧推理，与「全挥杆」及关键帧 JPEG 同一套时间线；若处于低信任，相位标签可能降级，请以视频叠加为准。",
      ],
      bullets_en: [
        `Phase labels seen in the pose sequence: ${phases.join(" → ")}.`,
        "These come from per-frame inference on the same timeline as Full Swing and keyframe JPEGs; under low trust, labels may be downgraded—trust the video overlay when unsure.",
      ],
    });
  } else {
    sections.push({
      id: "phases_fallback",
      title_zh: "相位与关键帧对齐",
      title_en: "Phases & keyframes",
      bullets_zh: [
        "逐帧姿态未带显式 phase_id 时，使用关键帧条 + 时间戳与视频 scrub 对齐观察；「全挥杆」tab 按标准相位轮播关键帧图。",
        input.phase_keyframes && Object.keys(input.phase_keyframes).length > 0
          ? `后端相位→姿态索引映射含 ${Object.keys(input.phase_keyframes).length} 个相位键，可与关键帧图对照。`
          : "",
      ].filter(Boolean),
      bullets_en: [
        "Without per-frame phase_id, align observations using the keyframe strip, timestamps, and video scrub; the Full Swing tab cycles standard phases.",
        input.phase_keyframes && Object.keys(input.phase_keyframes).length > 0
          ? `Phase→pose index map includes ${Object.keys(input.phase_keyframes).length} phase keys—cross-check keyframe images.`
          : "",
      ].filter(Boolean),
    });
  }

  const topF = pickTopish(frames);
  const impF = pickImpactish(frames);
  const zhTop = formatAngles(topF);
  const zhImp = formatAngles(impF);
  if (zhTop || zhImp) {
    sections.push({
      id: "angles",
      title_zh: "代表性角度（教育参考）",
      title_en: "Representative angles (educational)",
      bullets_zh: [
        zhTop ? `上杆附近（参考 top / 中段）：${zhTop}。` : "",
        zhImp ? `击球附近（参考 impact / 下杆末）：${zhImp}。` : "",
        "数值来自骨架解算，与职业区间对比仅作自学参考，不构成医疗或竞技结论。",
      ].filter(Boolean),
      bullets_en: [
        zhTop ? `Near top / mid-backswing: ${formatAnglesEn(topF)}.` : "",
        zhImp ? `Near impact / late downswing: ${formatAnglesEn(impF)}.` : "",
        "Values come from pose estimation; comparison to tour ranges is educational only—not medical or competitive advice.",
      ].filter(Boolean),
    });
  }

  const topAll = allAnglePairs(topF);
  const impAll = allAnglePairs(impF);
  if (topAll.zh.length > 0 || impAll.zh.length > 0) {
    sections.push({
      id: "angles_full",
      title_zh: "骨架角度（全量键）",
      title_en: "Pose angles (all keys)",
      bullets_zh: [
        ...(topAll.zh.length
          ? [
              `上杆参考帧 frame_index=${topF?.frame_index ?? "—"}：${topAll.zh.join("；")}`,
            ]
          : []),
        ...(impAll.zh.length
          ? [
              `击球参考帧 frame_index=${impF?.frame_index ?? "—"}：${impAll.zh.join("；")}`,
            ]
          : []),
      ],
      bullets_en: [
        ...(topAll.en.length
          ? [
              `Top ref frame_index=${topF?.frame_index ?? "—"}: ${topAll.en.join("; ")}`,
            ]
          : []),
        ...(impAll.en.length
          ? [
              `Impact ref frame_index=${impF?.frame_index ?? "—"}: ${impAll.en.join("; ")}`,
            ]
          : []),
      ],
    });
  }

  const vZh = [...visibilitySummaryZh(topF).map((x) => `上杆参考帧：${x}`), ...visibilitySummaryZh(impF).map((x) => `击球参考帧：${x}`)];
  const vEn = [
    ...visibilitySummaryEn(topF).map((x) => `Top ref: ${x}`),
    ...visibilitySummaryEn(impF).map((x) => `Impact ref: ${x}`),
  ];
  if (vZh.length > 0) {
    sections.push({
      id: "joint_visibility",
      title_zh: "关节可见度（参考帧）",
      title_en: "Joint visibility (ref frames)",
      bullets_zh: vZh,
      bullets_en: vEn,
    });
  }

  const pred = input.prediction ?? {};
  const dist = num(pred.predicted_distance);
  const chs = num(pred.club_head_speed as number);
  const fused = num(pred.fused_speed);
  const blurS = num(pred.blur_speed);
  const shapeZh = String(pred.shot_shape_zh ?? pred.shot_shape ?? "").trim();
  const shapeEn = String(pred.shot_shape ?? "").trim();
  const club = String(pred.club_type ?? "").trim();
  const speedConf = String(pred.speed_confidence ?? "").trim();
  const vfBulletsZh = [
    dist != null && dist > 0 ? `弹道模型预估距离约 ${Math.round(dist)} 码（JSON prediction.predicted_distance）。` : "",
    chs != null && chs > 0 ? `杆头速度约 ${chs.toFixed(1)} mph（prediction.club_head_speed）。` : "",
    fused != null && fused > 0 ? `融合杆速约 ${fused.toFixed(1)} mph（prediction.fused_speed）。` : "",
    blurS != null && blurS > 0 ? `模糊法估计约 ${blurS.toFixed(1)} mph（prediction.blur_speed）。` : "",
    speedConf ? `速度置信标签：${speedConf}（prediction.speed_confidence）。` : "",
    shapeZh ? `球路（中文）：${shapeZh}。` : "",
    club && club !== "UNKNOWN" ? `杆型：${club}。` : "",
  ].filter(Boolean);
  const vfBulletsEn = [
    dist != null && dist > 0 ? `Yardage ~${Math.round(dist)} yds (prediction.predicted_distance).` : "",
    chs != null && chs > 0 ? `Club speed ~${chs.toFixed(1)} mph (prediction.club_head_speed).` : "",
    fused != null && fused > 0 ? `Fused speed ~${fused.toFixed(1)} mph (prediction.fused_speed).` : "",
    blurS != null && blurS > 0 ? `Blur estimate ~${blurS.toFixed(1)} mph (prediction.blur_speed).` : "",
    speedConf ? `Speed confidence: ${speedConf} (prediction.speed_confidence).` : "",
    shapeEn ? `Shot shape: ${shapeEn}.` : "",
    club && club !== "UNKNOWN" ? `Club: ${club}.` : "",
  ].filter(Boolean);
  if (vfBulletsZh.length > 0) {
    vfBulletsZh.push("飞行轨迹 HUD 与骨架叠加共用同一时间线（与视频 scrub 对齐）。");
    vfBulletsEn.push("Flight HUD and pose overlay share the same timeline as the video scrubber.");
    sections.push({
      id: "video_fusion",
      title_zh: "视频融合字段（prediction）",
      title_en: "Video-fusion fields (prediction)",
      bullets_zh: vfBulletsZh,
      bullets_en: vfBulletsEn,
    });
  }

  const evals = Array.isArray(input.swing_phase_evaluations) ? input.swing_phase_evaluations : [];
  if (evals.length > 0) {
    const zh = evals.slice(0, 8).map((e) => {
      const note = (e.note_zh || e.note_en || "").trim();
      return `${e.phase}: ${e.status}${note ? ` — ${note}` : ""}`;
    });
    const en = evals.slice(0, 8).map((e) => {
      const note = (e.note_en || e.note_zh || "").trim();
      return `${e.phase}: ${e.status}${note ? ` — ${note}` : ""}`;
    });
    sections.push({
      id: "phase_eval",
      title_zh: "分相位评估（swing_phase_evaluations）",
      title_en: "Per-phase evaluations (payload)",
      bullets_zh: zh,
      bullets_en: en,
    });
  }

  const cfs = input.core_frame_scores;
  if (cfs && typeof cfs === "object") {
    const entries = Object.entries(cfs).filter(
      ([, v]) =>
        v &&
        (num(v.score) != null ||
          v.pass_90 != null ||
          num(v.confidence) != null ||
          String(v.comment_zh ?? v.comment_en ?? "").trim().length > 0),
    );
    if (entries.length > 0) {
      const fmt = (zh: boolean) =>
        entries.slice(0, 14).map(([phase, v]) => {
          const s = num(v.score);
          const c = num(v.confidence);
          const bits: string[] = [phase];
          if (s != null) bits.push(zh ? `分 ${s.toFixed(1)}` : `score ${s.toFixed(1)}`);
          if (v.pass_90 != null) bits.push(`pass_90=${v.pass_90}`);
          if (c != null) bits.push(zh ? `置信 ${c.toFixed(2)}` : `conf ${c.toFixed(2)}`);
          const q = String(zh ? v.comment_zh ?? v.comment_en : v.comment_en ?? v.comment_zh ?? "").trim();
          if (q) bits.push(q);
          return bits.join(zh ? " · " : " · ");
        });
      sections.push({
        id: "core_frame_scores",
        title_zh: "核心帧评分（core_frame_scores）",
        title_en: "Core frame scores (payload)",
        bullets_zh: fmt(true),
        bullets_en: fmt(false),
      });
    }
  }

  const gf = input.gemini_frame_notes;
  if (Array.isArray(gf) && gf.length > 0) {
    sections.push({
      id: "gemini_frame_notes",
      title_zh: "多模态帧注（gemini_observation）",
      title_en: "Multimodal frame notes (gemini_observation)",
      bullets_zh: gf.slice(0, 12).map((n) => {
        const note = String(n.note_zh ?? n.note_en ?? "").trim();
        return `帧 ${n.index}${n.label ? ` · ${n.label}` : ""}${note ? `：${note}` : ""}`;
      }),
      bullets_en: gf.slice(0, 12).map((n) => {
        const note = String(n.note_en ?? n.note_zh ?? "").trim();
        return `Frame ${n.index}${n.label ? ` · ${n.label}` : ""}${note ? `: ${note}` : ""}`;
      }),
    });
  }

  const issuesZh = (input.issues_zh ?? []).filter(Boolean).slice(0, 6);
  const issuesEn = (input.issues ?? []).filter(Boolean).slice(0, 6);
  if (issuesZh.length > 0 || issuesEn.length > 0) {
    sections.push({
      id: "issues_link",
      title_zh: "问题列表（issues 原文）",
      title_en: "Issues (verbatim)",
      bullets_zh: issuesZh.length > 0 ? issuesZh : issuesEn,
      bullets_en: issuesEn.length > 0 ? issuesEn : issuesZh,
    });
  }

  const sumZh = String(input.summary_zh ?? "").trim();
  const sumEn = String(input.summary ?? "").trim();
  if (sumZh || sumEn) {
    sections.push({
      id: "summary_anchor",
      title_zh: "文字摘要（summary 字段）",
      title_en: "Text summary (summary fields)",
      bullets_zh: sumZh ? [sumZh.slice(0, 600) + (sumZh.length > 600 ? "…" : "")] : [sumEn.slice(0, 600) + (sumEn.length > 600 ? "…" : "")],
      bullets_en: sumEn ? [sumEn.slice(0, 600) + (sumEn.length > 600 ? "…" : "")] : [sumZh.slice(0, 600) + (sumZh.length > 600 ? "…" : "")],
    });
  }

  return sections;
}
