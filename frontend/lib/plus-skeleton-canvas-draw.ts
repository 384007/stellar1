/**
 * Plus-style skeleton + plumb + meteor swing arcs (same visuals as PlusResultView keyframes).
 * Used on VideoAnalysisOverlay for Stellar Pro / Plus parity.
 */

export type PlusSkelJoint = {
  name: string;
  normalized: { x: number; y: number };
  visibility: number;
};

export type PlusSkelPose = {
  joints: PlusSkelJoint[];
  connections?: number[][];
};

export const PLUS_SKELETON_JOINT_COLORS: Record<string, string> = {
  head: "#f5c518",
  nose: "#ef4444",
  left_eye_inner: "#ef4444",
  left_eye: "#ef4444",
  left_eye_outer: "#ef4444",
  right_eye_inner: "#ef4444",
  right_eye: "#ef4444",
  right_eye_outer: "#ef4444",
  left_ear: "#ef4444",
  right_ear: "#ef4444",
  mouth_left: "#ef4444",
  mouth_right: "#ef4444",
  left_shoulder: "#a855f7",
  right_shoulder: "#a855f7",
  left_elbow: "#8b5cf6",
  right_elbow: "#8b5cf6",
  left_wrist: "#f59e0b",
  right_wrist: "#f59e0b",
  left_pinky: "#f59e0b",
  right_pinky: "#f59e0b",
  left_index: "#f59e0b",
  right_index: "#f59e0b",
  left_thumb: "#f59e0b",
  right_thumb: "#f59e0b",
  left_hip: "#06b6d4",
  right_hip: "#06b6d4",
  left_knee: "#22c55e",
  right_knee: "#22c55e",
  left_ankle: "#14b8a6",
  right_ankle: "#14b8a6",
  left_heel: "#14b8a6",
  right_heel: "#14b8a6",
  left_foot_index: "#14b8a6",
  right_foot_index: "#14b8a6",
};

export function letterboxPoseInContainer(
  frameW: number,
  frameH: number,
  cW: number,
  cH: number,
): { offsetX: number; offsetY: number; renderW: number; renderH: number } {
  const fW = frameW || cW;
  const fH = frameH || cH;
  if (!fW || !fH) return { offsetX: 0, offsetY: 0, renderW: cW, renderH: cH };
  const containerAR = cW / cH;
  const frameAR = fW / fH;
  let renderW: number;
  let renderH: number;
  let offsetX: number;
  let offsetY: number;
  if (frameAR >= containerAR) {
    renderW = cW;
    renderH = cW / frameAR;
    offsetX = 0;
    offsetY = (cH - renderH) / 2;
  } else {
    renderH = cH;
    renderW = cH * frameAR;
    offsetX = (cW - renderW) / 2;
    offsetY = 0;
  }
  return { offsetX, offsetY, renderW, renderH };
}

/** Plus scale factor from render box short side (matches PlusResultView). */
export function plusSkeletonScale(renderW: number, renderH: number): number {
  const ref = Math.min(renderW, renderH);
  return Math.max(0.3, Math.min(1.5, ref / 380));
}

/**
 * Draw plumb + meteor arcs + gradient skeleton (no clearRect).
 * `px` maps normalized [0,1] pose coords to canvas pixels (with letterbox).
 */
export function drawPlusStyleSkeletonOverlay(
  ctx: CanvasRenderingContext2D,
  pose: PlusSkelPose,
  px: (nx: number, ny: number) => [number, number],
  s: number,
  offsetY: number,
  renderH: number,
  showSkeleton: boolean,
  showGuideLines: boolean,
): void {
  const joints = pose.joints;
  if (!joints?.length) return;

  if (showGuideLines) {
    const lsh = joints.find((j) => j.name === "left_shoulder");
    const rsh = joints.find((j) => j.name === "right_shoulder");
    const nose = joints.find((j) => j.name === "nose");
    const headJ = joints.find((j) => j.name === "head");
    const rwri = joints.find((j) => j.name === "right_wrist");
    const lwri = joints.find((j) => j.name === "left_wrist");
    const lank = joints.find((j) => j.name === "left_ankle");
    const rank = joints.find((j) => j.name === "right_ankle");
    const hasSh =
      lsh && rsh && lsh.visibility > 0.3 && rsh.visibility > 0.3;
    const refHead =
      nose && nose.visibility > 0.3
        ? nose
        : headJ && headJ.visibility > 0.3
          ? headJ
          : null;
    let midSh: [number, number] | null = null;
    if (hasSh)
      midSh = px(
        (lsh!.normalized.x + rsh!.normalized.x) / 2,
        (lsh!.normalized.y + rsh!.normalized.y) / 2,
      );

    if (refHead) {
      const [nx, ny] = px(refHead.normalized.x, refHead.normalized.y);
      const footY =
        lank && lank.visibility > 0.3
          ? px(0, lank.normalized.y)[1]
          : rank && rank.visibility > 0.3
            ? px(0, rank.normalized.y)[1]
            : offsetY + renderH * 0.92;
      ctx.save();
      ctx.globalAlpha = 0.5;
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = Math.max(1.2, 2 * s);
      ctx.setLineDash([7 * s, 4 * s]);
      ctx.beginPath();
      ctx.moveTo(nx, ny - 10 * s);
      ctx.lineTo(nx, footY);
      ctx.stroke();
      ctx.restore();
    }

    const wrist =
      rwri && rwri.visibility > 0.3
        ? rwri
        : lwri && lwri.visibility > 0.3
          ? lwri
          : null;
    if (wrist && midSh) {
      const [wx, wy] = px(wrist.normalized.x, wrist.normalized.y);
      const arcR = Math.sqrt((wx - midSh[0]) ** 2 + (wy - midSh[1]) ** 2);
      const wAngle = Math.atan2(wy - midSh[1], wx - midSh[0]);
      const segs = 16;
      const trailLen = Math.PI * 0.75;
      const bsColors = ["#1e3a8a", "#2563eb", "#0ea5e9", "#22d3ee"];
      const dsColors = ["#16a34a", "#22c55e", "#eab308", "#f97316", "#ef4444"];

      for (let i = 0; i < segs; i++) {
        const t = i / segs;
        const a0 = wAngle + trailLen * t;
        const a1 = wAngle + trailLen * (t + 1 / segs);
        const ci = Math.min(
          bsColors.length - 1,
          Math.floor((1 - t) * (bsColors.length - 1)),
        );
        ctx.save();
        ctx.globalAlpha = (1 - t) * 0.45;
        ctx.strokeStyle = bsColors[ci];
        ctx.lineWidth = Math.max(1.5, (4 - t * 3) * s);
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.arc(midSh[0], midSh[1], arcR, a0, a1);
        ctx.stroke();
        ctx.restore();
      }

      for (let i = 0; i < segs; i++) {
        const t = i / segs;
        const a0 = wAngle - trailLen * t;
        const a1 = wAngle - trailLen * (t + 1 / segs);
        const ci = Math.min(
          dsColors.length - 1,
          Math.floor((1 - t) * (dsColors.length - 1)),
        );
        ctx.save();
        ctx.globalAlpha = (1 - t) * 0.4;
        ctx.strokeStyle = dsColors[ci];
        ctx.lineWidth = Math.max(1.5, (3.5 - t * 2.5) * s);
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.arc(midSh[0], midSh[1], arcR, a0, a1, true);
        ctx.stroke();
        ctx.restore();
      }

      ctx.save();
      const glow = ctx.createRadialGradient(wx, wy, 0, wx, wy, 10 * s);
      glow.addColorStop(0, "rgba(255,255,255,0.9)");
      glow.addColorStop(0.25, "rgba(34,211,238,0.5)");
      glow.addColorStop(0.6, "rgba(14,165,233,0.2)");
      glow.addColorStop(1, "transparent");
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(wx, wy, 10 * s, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  if (!showSkeleton) return;

  const lineMain = Math.max(1, 2.5 * s);
  const lineGlow1 = Math.max(2, 10 * s);
  const lineGlow2 = Math.max(1.5, 5 * s);
  const colors = PLUS_SKELETON_JOINT_COLORS;

  for (const conn of pose.connections || []) {
    const j1 = joints[conn[0]];
    const j2 = joints[conn[1]];
    if (!j1 || !j2 || j1.visibility < 0.3 || j2.visibility < 0.3) continue;
    const [x1, y1] = px(j1.normalized.x, j1.normalized.y);
    const [x2, y2] = px(j2.normalized.x, j2.normalized.y);

    ctx.save();
    ctx.globalAlpha = 0.07;
    ctx.strokeStyle = "#9f5fff";
    ctx.lineWidth = lineGlow1;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    ctx.globalAlpha = 0.12;
    ctx.lineWidth = lineGlow2;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    ctx.globalAlpha = 0.9;
    const g = ctx.createLinearGradient(x1, y1, x2, y2);
    g.addColorStop(0, colors[j1.name] || "#b97bff");
    g.addColorStop(1, colors[j2.name] || "#f5c518");
    ctx.strokeStyle = g;
    ctx.lineWidth = lineMain;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.restore();
  }

  for (const j of joints) {
    if (j.visibility < 0.3) continue;
    const [x, y] = px(j.normalized.x, j.normalized.y);
    const color = colors[j.name] || "#a855f7";
    const isKey =
      j.name === "head" ||
      j.name.includes("shoulder") ||
      j.name.includes("hip") ||
      j.name.includes("knee") ||
      j.name.includes("elbow") ||
      j.name.includes("wrist");
    const r = isKey ? Math.max(2, 5 * s) : Math.max(1.2, 3 * s);
    const glowR = r * 2.5;

    ctx.save();
    const halo = ctx.createRadialGradient(x, y, 0, x, y, glowR);
    halo.addColorStop(0, color + "33");
    halo.addColorStop(1, "transparent");
    ctx.beginPath();
    ctx.arc(x, y, glowR, 0, Math.PI * 2);
    ctx.fillStyle = halo;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = color + "28";
    ctx.fill();
    ctx.strokeStyle = j.visibility >= 0.7 ? color + "dd" : color + "88";
    ctx.lineWidth = Math.max(0.8, 1.8 * s);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(x, y, Math.max(0.6, r * 0.35), 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,255,255,0.92)";
    ctx.fill();
    ctx.restore();
  }
}
