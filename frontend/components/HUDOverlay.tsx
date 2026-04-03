"use client";

import { useRef, useEffect } from "react";

interface Joint {
  name: string;
  x: number;
  y: number;
  visible: boolean;
  category: string;
  angle: number | null;
  color: string;
  pulse: boolean;
}

interface Connection {
  from: number;
  to: number;
  from_name: string;
  to_name: string;
  visible: boolean;
  gradient: string[];
}

interface HUDData {
  joints?: Joint[];
  connections?: Connection[];
  angles?: Record<string, number>;
  frame_size?: { width: number; height: number };
  stats?: Record<string, number>;
}

interface TrajectoryPoint {
  x: number;
  y: number;
  speed: number;
}

interface HUDOverlayProps {
  hudData: Record<string, unknown>;
  showExtended: boolean;
  mode: "lite" | "pro";
  trajectory?: TrajectoryPoint[];
  lang?: "en" | "zh";
}

export default function HUDOverlay({
  hudData,
  showExtended,
  mode,
  trajectory,
  lang = "zh",
}: HUDOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);
  const trailsRef = useRef<Array<{ joints: Array<{ x: number; y: number }>; alpha: number }>>([]);

  const data = hudData as unknown as HUDData;
  const joints = data?.joints || [];
  const connections = data?.connections || [];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    let t = 0;

    // Build motion trail history
    if (trailsRef.current.length === 0 && joints.length > 0) {
      for (let i = 0; i < 5; i++) {
        const offset = (5 - i) * 0.008;
        trailsRef.current.push({
          joints: joints.map((j) => ({
            x: j.x + (Math.random() - 0.5) * offset,
            y: j.y + (Math.random() - 0.5) * offset,
          })),
          alpha: 0.06 + i * 0.03,
        });
      }
    }

    function shouldShow(joint: Joint) {
      if (showExtended) return true;
      return joint.visible && joint.category !== "extended";
    }

    function shouldShowConn(conn: Connection) {
      if (showExtended) return true;
      if (!conn.visible) return false;
      const fromJ = joints[conn.from];
      const toJ = joints[conn.to];
      return fromJ && toJ && shouldShow(fromJ) && shouldShow(toJ);
    }

    function draw() {
      if (!ctx) return;
      ctx.clearRect(0, 0, W, H);

      // Transparent dark background with subtle gradient
      const bg = ctx.createRadialGradient(W * 0.5, H * 0.4, 0, W * 0.5, H * 0.4, W * 0.6);
      bg.addColorStop(0, "rgba(13,10,26,0.65)");
      bg.addColorStop(1, "rgba(13,10,26,0.85)");
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, W, H);

      drawSubtleGrid(ctx, W, H, t);
      drawMotionTrails(ctx, W, H, t);
      drawConnections(ctx, W, H, t);
      drawJoints(ctx, W, H, t);

      if (mode === "pro" && trajectory && trajectory.length > 1) {
        drawTrajectory(ctx, W, H, t);
      }

      drawAngleLabels(ctx, W, H, t);
      drawStats(ctx, W, H, t);
      drawBrandMark(ctx, W, H);

      t += 0.016;
      animFrameRef.current = requestAnimationFrame(draw);
    }

    function drawSubtleGrid(ctx: CanvasRenderingContext2D, w: number, h: number, t: number) {
      const spacing = 40;
      for (let x = 0; x < w; x += spacing) {
        const alpha = 0.02 + Math.sin(x * 0.01 + t * 0.5) * 0.01;
        ctx.strokeStyle = `rgba(124,58,237,${alpha})`;
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      for (let y = 0; y < h; y += spacing) {
        const alpha = 0.02 + Math.sin(y * 0.01 + t * 0.3) * 0.01;
        ctx.strokeStyle = `rgba(124,58,237,${alpha})`;
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }
    }

    function drawMotionTrails(ctx: CanvasRenderingContext2D, w: number, h: number, t: number) {
      for (const trail of trailsRef.current) {
        ctx.globalAlpha = trail.alpha * (0.7 + Math.sin(t) * 0.3);
        for (const conn of connections) {
          if (!shouldShowConn(conn)) continue;
          const from = trail.joints[conn.from];
          const to = trail.joints[conn.to];
          if (!from || !to) continue;

          ctx.beginPath();
          ctx.moveTo(from.x * w, from.y * h);
          ctx.lineTo(to.x * w, to.y * h);
          ctx.strokeStyle = "rgba(159,95,255,0.4)";
          ctx.lineWidth = 1;
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
    }

    function drawConnections(ctx: CanvasRenderingContext2D, w: number, h: number, t: number) {
      for (const conn of connections) {
        if (!shouldShowConn(conn)) continue;

        const fromJoint = joints[conn.from];
        const toJoint = joints[conn.to];
        if (!fromJoint || !toJoint) continue;

        const x1 = fromJoint.x * w;
        const y1 = fromJoint.y * h;
        const x2 = toJoint.x * w;
        const y2 = toJoint.y * h;

        // Subtle glow line
        const glowGrad = ctx.createLinearGradient(x1, y1, x2, y2);
        glowGrad.addColorStop(0, "rgba(159,95,255,0.08)");
        glowGrad.addColorStop(0.5, "rgba(159,95,255,0.15)");
        glowGrad.addColorStop(1, "rgba(159,95,255,0.08)");
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = glowGrad;
        ctx.lineWidth = 6;
        ctx.stroke();

        // Main connection line with energy flow
        const gradient = ctx.createLinearGradient(x1, y1, x2, y2);
        const c1 = conn.gradient?.[0] || "#9f5fff";
        const c2 = conn.gradient?.[1] || "#f5c518";
        gradient.addColorStop(0, c1 + "99");
        gradient.addColorStop(1, c2 + "99");

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = gradient;
        ctx.lineWidth = mode === "pro" ? 2.5 : 1.8;
        ctx.stroke();

        // Flowing energy particles along connection
        const dist = Math.hypot(x2 - x1, y2 - y1);
        const particleCount = Math.floor(dist / 30);
        for (let p = 0; p < particleCount; p++) {
          const progress = ((t * 0.8 + p * (1 / particleCount)) % 1);
          const px = x1 + (x2 - x1) * progress;
          const py = y1 + (y2 - y1) * progress;
          const pAlpha = Math.sin(progress * Math.PI) * 0.5;

          ctx.beginPath();
          ctx.arc(px, py, 1.5, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(245,197,24,${pAlpha})`;
          ctx.fill();
        }
      }
    }

    function drawJoints(ctx: CanvasRenderingContext2D, w: number, h: number, t: number) {
      for (let idx = 0; idx < joints.length; idx++) {
        const joint = joints[idx];
        if (!shouldShow(joint)) continue;

        const x = joint.x * w;
        const y = joint.y * h;
        const breathe = 1 + 0.15 * Math.sin(t * Math.PI * 1.2 + idx * 0.5);
        const baseR = mode === "pro" ? 6 : 4.5;
        const radius = baseR * breathe;

        // Outer pulse ring
        if (joint.pulse) {
          const pulseR = radius * (2 + Math.sin(t * Math.PI * 2) * 0.5);
          ctx.beginPath();
          ctx.arc(x, y, pulseR, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(245,197,24,${0.15 + Math.sin(t * 2) * 0.1})`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }

        // Soft outer glow
        const glowR = radius * 2.5;
        const glowGradient = ctx.createRadialGradient(x, y, radius * 0.5, x, y, glowR);
        glowGradient.addColorStop(0, joint.color + "33");
        glowGradient.addColorStop(1, "transparent");
        ctx.beginPath();
        ctx.arc(x, y, glowR, 0, Math.PI * 2);
        ctx.fillStyle = glowGradient;
        ctx.fill();

        // Joint body
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = joint.color + "cc";
        ctx.fill();

        // Bright center dot
        ctx.beginPath();
        ctx.arc(x, y, radius * 0.35, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255,255,255,0.8)";
        ctx.fill();
      }
    }

    function drawTrajectory(ctx: CanvasRenderingContext2D, w: number, h: number, t: number) {
      if (!trajectory || trajectory.length < 2) return;

      ctx.beginPath();
      ctx.moveTo(trajectory[0].x * w, trajectory[0].y * h);
      for (let i = 1; i < trajectory.length; i++) {
        ctx.lineTo(trajectory[i].x * w, trajectory[i].y * h);
      }
      ctx.strokeStyle = "rgba(245,197,24,0.35)";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 3]);
      ctx.lineDashOffset = -t * 30;
      ctx.stroke();
      ctx.setLineDash([]);

      // Animated point on trajectory
      const idx = Math.floor((t * 0.5 % 1) * (trajectory.length - 1));
      const tp = trajectory[idx];
      if (tp) {
        ctx.beginPath();
        ctx.arc(tp.x * w, tp.y * h, 3, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(245,197,24,0.7)";
        ctx.fill();
      }
    }

    function drawAngleLabels(ctx: CanvasRenderingContext2D, w: number, h: number, _t: number) {
      for (const joint of joints) {
        if (!shouldShow(joint)) continue;
        if (joint.angle === null || joint.angle === undefined) continue;

        const x = joint.x * w;
        const y = joint.y * h;
        const labelX = x + 14;
        const labelY = y - 14;
        const text = `${joint.angle}°`;

        ctx.font = "bold 9px 'Rajdhani', sans-serif";
        const metrics = ctx.measureText(text);
        const padding = 4;
        const boxW = metrics.width + padding * 2;
        const boxH = 14;

        ctx.fillStyle = "rgba(0,0,0,0.5)";
        ctx.beginPath();
        ctx.roundRect(labelX - padding, labelY - boxH / 2, boxW, boxH, 3);
        ctx.fill();

        ctx.strokeStyle = "rgba(124,58,237,0.3)";
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.roundRect(labelX - padding, labelY - boxH / 2, boxW, boxH, 3);
        ctx.stroke();

        ctx.fillStyle = "#ffd85e";
        ctx.textBaseline = "middle";
        ctx.fillText(text, labelX, labelY);
      }
    }

    function drawStats(ctx: CanvasRenderingContext2D, w: number, h: number, t: number) {
      const stats = data?.stats;
      if (!stats) return;

      const panelW = 150;
      const panelH = mode === "pro" ? 80 : 65;
      const px = 10;
      const py = 10;

      // Semi-transparent panel
      ctx.fillStyle = "rgba(13,10,26,0.6)";
      ctx.beginPath();
      ctx.roundRect(px, py, panelW, panelH, 8);
      ctx.fill();

      ctx.strokeStyle = "rgba(124,58,237,0.2)";
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.roundRect(px, py, panelW, panelH, 8);
      ctx.stroke();

      // Accent line
      const accentAlpha = 0.5 + Math.sin(t * 2) * 0.2;
      ctx.strokeStyle = `rgba(245,197,24,${accentAlpha})`;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(px + 8, py);
      ctx.lineTo(px + 45, py);
      ctx.stroke();

      ctx.font = "bold 10px 'Rajdhani', sans-serif";
      ctx.fillStyle = "rgba(245,197,24,0.8)";
      ctx.fillText("STELLAR HUD", px + 8, py + 16);

      ctx.font = "9px 'Rajdhani', sans-serif";
      let y = py + 30;
      const entries: [string, number | undefined][] = [
        [lang === "zh" ? "肩部旋转" : "Shoulder", stats.shoulder_rotation],
        [lang === "zh" ? "髋部旋转" : "Hip", stats.hip_rotation],
        [lang === "zh" ? "X因子" : "X-Factor", stats.x_factor],
      ];
      if (mode === "pro") {
        entries.push([lang === "zh" ? "脊柱倾斜" : "Spine Tilt", stats.spine_tilt]);
      }

      for (const [label, value] of entries) {
        ctx.fillStyle = "rgba(139,125,181,0.7)";
        ctx.fillText(label, px + 8, y);
        ctx.fillStyle = "rgba(245,197,24,0.8)";
        ctx.fillText(
          `${typeof value === "number" ? value.toFixed(1) : (value ?? "N/A")}°`,
          px + 90,
          y
        );
        y += 12;
      }
    }

    function drawBrandMark(ctx: CanvasRenderingContext2D, w: number, h: number) {
      ctx.font = "8px 'Rajdhani', sans-serif";
      ctx.fillStyle = "rgba(139,125,181,0.25)";
      ctx.textAlign = "right";
      ctx.fillText("STELLAR AI", w - 10, h - 8);
      ctx.textAlign = "left";
    }

    animFrameRef.current = requestAnimationFrame(draw);

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [joints, connections, showExtended, mode, trajectory, data?.stats, lang]);

  // Clear trails when data changes
  useEffect(() => {
    trailsRef.current = [];
  }, [hudData]);

  const visibleJoints = joints.filter((j) =>
    showExtended ? true : (j.visible && j.category !== "extended")
  );

  const defaultCount = 4;
  const extendedCount = visibleJoints.length;

  return (
    <div className="relative overflow-hidden rounded-xl">
      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        className="w-full rounded-xl"
        style={{ aspectRatio: "4/3" }}
      />

      {/* Joint info chips */}
      <div className="mt-3 flex flex-wrap gap-1.5">
        {visibleJoints.map((joint) => (
          <span
            key={joint.name}
            className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] text-white/50 border border-white/5 bg-white/[0.03]"
          >
            <span
              className="h-1.5 w-1.5 rounded-full"
              style={{ backgroundColor: joint.color, opacity: 0.7 }}
            />
            {joint.name.replace(/_/g, " ")}
            {joint.angle !== null && joint.angle !== undefined && (
              <span className="text-brand-gold/60">{joint.angle}°</span>
            )}
          </span>
        ))}
      </div>

      {/* Counter badge */}
      <div className="mt-2 text-[10px] text-white/25">
        {lang === "zh"
          ? `显示 ${showExtended ? extendedCount : Math.min(defaultCount, extendedCount)} / ${joints.length} 个关键部位`
          : `Showing ${showExtended ? extendedCount : Math.min(defaultCount, extendedCount)} / ${joints.length} key points`}
      </div>
    </div>
  );
}
