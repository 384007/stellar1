"use client";

import { useRef, useEffect, useCallback } from "react";

interface Prediction {
  predicted_distance: number;
  lateral_offset: number;
  shot_shape: string;
  shot_shape_zh?: string;
  club_head_speed: number;
  ball_speed: number;
  launch_angle: number;
  spin_rate: number;
  smash_factor: number;
  trajectory?: Array<{ t: number; x: number; y: number; lateral: number }>;
}

interface SimAnimationProps {
  prediction: Prediction;
  lang?: "en" | "zh";
  isPro?: boolean;
}

export default function SimAnimation({ prediction, lang = "zh", isPro }: SimAnimationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);

  const distance = prediction.predicted_distance || 200;
  const deviation = prediction.lateral_offset || 0;
  const shotShape = lang === "zh"
    ? (prediction.shot_shape_zh || prediction.shot_shape || "直球")
    : (prediction.shot_shape || "Straight");
  const carry = Math.round(distance * 0.85);
  const roll = Math.round(distance * 0.15);

  const runAnimation = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    const totalFrames = 90;
    let frame = 0;

    const TEE_X = W * 0.12;
    const TEE_Y = H * 0.72;
    const FAIRWAY_END_X = W * 0.82;
    const devPx = (deviation / 300) * H * 0.6;
    const LAND_X = FAIRWAY_END_X;
    const LAND_Y = TEE_Y - (distance / 350) * H * 0.55 + devPx;

    function drawCourse() {
      if (!ctx) return;
      const sky = ctx.createLinearGradient(0, 0, 0, H * 0.65);
      sky.addColorStop(0, "#0d0a1a");
      sky.addColorStop(1, "#1a1530");
      ctx.fillStyle = sky;
      ctx.fillRect(0, 0, W, H);

      const stars = [[0.1,0.1],[0.3,0.05],[0.6,0.08],[0.8,0.12],[0.9,0.04],[0.5,0.15],[0.15,0.2],[0.75,0.18]];
      stars.forEach(([sx, sy]) => {
        ctx.beginPath();
        ctx.arc(sx * W, sy * H, 1, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255,215,94,0.6)";
        ctx.fill();
      });

      const ground = ctx.createLinearGradient(0, H * 0.6, 0, H);
      ground.addColorStop(0, "#1a2f1a");
      ground.addColorStop(1, "#0f1f0f");
      ctx.fillStyle = ground;
      ctx.fillRect(0, H * 0.62, W, H * 0.38);

      ctx.beginPath();
      ctx.moveTo(TEE_X - 20, TEE_Y + 10);
      ctx.lineTo(FAIRWAY_END_X + 20, LAND_Y + 15);
      ctx.lineTo(FAIRWAY_END_X + 20, LAND_Y - 15);
      ctx.lineTo(TEE_X - 20, TEE_Y - 10);
      ctx.closePath();
      ctx.fillStyle = "rgba(34,60,34,0.8)";
      ctx.fill();

      const horiz = ctx.createLinearGradient(0, H * 0.58, 0, H * 0.68);
      horiz.addColorStop(0, "rgba(124,58,237,0.15)");
      horiz.addColorStop(1, "transparent");
      ctx.fillStyle = horiz;
      ctx.fillRect(0, H * 0.58, W, H * 0.1);

      ctx.strokeStyle = "rgba(124,58,237,0.08)";
      ctx.lineWidth = 0.5;
      for (let i = 0; i < 8; i++) {
        const ly = H * 0.63 + i * 12;
        ctx.beginPath();
        ctx.moveTo(0, ly);
        ctx.lineTo(W, ly);
        ctx.stroke();
      }

      ctx.setLineDash([4, 6]);
      ctx.strokeStyle = "rgba(245,197,24,0.15)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(TEE_X, TEE_Y);
      ctx.lineTo(LAND_X, TEE_Y);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = "#f5c518";
      ctx.beginPath();
      ctx.roundRect(TEE_X - 14, TEE_Y - 5, 28, 10, 3);
      ctx.fill();
      ctx.fillStyle = "rgba(245,197,24,0.3)";
      ctx.beginPath();
      ctx.roundRect(TEE_X - 16, TEE_Y - 7, 32, 14, 4);
      ctx.fill();

      const flagX = FAIRWAY_END_X + 10;
      const flagY = LAND_Y;
      ctx.strokeStyle = "rgba(255,255,255,0.5)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(flagX, flagY - 5);
      ctx.lineTo(flagX, flagY - 30);
      ctx.stroke();
      ctx.fillStyle = "#7c3aed";
      ctx.beginPath();
      ctx.moveTo(flagX, flagY - 30);
      ctx.lineTo(flagX + 14, flagY - 24);
      ctx.lineTo(flagX, flagY - 18);
      ctx.closePath();
      ctx.fill();
    }

    function drawBall(progress: number) {
      if (!ctx) return;
      const eased = progress < 0.5
        ? 2 * progress * progress
        : 1 - Math.pow(-2 * progress + 2, 2) / 2;

      const bx = TEE_X + (LAND_X - TEE_X) * eased;
      const peakY = Math.min(TEE_Y, LAND_Y) - H * 0.38;
      const by = (1 - eased) * TEE_Y + eased * LAND_Y - 4 * eased * (1 - eased) * (TEE_Y - peakY);

      const shadowY = H * 0.68;
      const shadowAlpha = 0.15 + 0.25 * (1 - Math.abs(eased - 0.5) * 2);
      ctx.beginPath();
      ctx.ellipse(bx, shadowY, 12 * shadowAlpha * 6, 4 * shadowAlpha * 4, 0, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0,0,0,${shadowAlpha})`;
      ctx.fill();

      const trailLen = 12;
      for (let i = trailLen; i > 0; i--) {
        const tp = Math.max(0, progress - i * 0.008);
        const te = tp < 0.5 ? 2 * tp * tp : 1 - Math.pow(-2 * tp + 2, 2) / 2;
        const tx = TEE_X + (LAND_X - TEE_X) * te;
        const ty = (1 - te) * TEE_Y + te * LAND_Y - 4 * te * (1 - te) * (TEE_Y - peakY);
        const alpha = (i / trailLen) * 0.4;
        ctx.beginPath();
        ctx.arc(tx, ty, 2.5 * (i / trailLen), 0, Math.PI * 2);
        ctx.fillStyle = `rgba(245,197,24,${alpha})`;
        ctx.fill();
      }

      const glow = ctx.createRadialGradient(bx, by, 0, bx, by, 14);
      glow.addColorStop(0, "rgba(245,197,24,0.5)");
      glow.addColorStop(1, "transparent");
      ctx.beginPath();
      ctx.arc(bx, by, 14, 0, Math.PI * 2);
      ctx.fillStyle = glow;
      ctx.fill();

      const ballGrad = ctx.createRadialGradient(bx - 2, by - 2, 0, bx, by, 6);
      ballGrad.addColorStop(0, "#ffffff");
      ballGrad.addColorStop(1, "#d0d0d0");
      ctx.beginPath();
      ctx.arc(bx, by, 6, 0, Math.PI * 2);
      ctx.fillStyle = ballGrad;
      ctx.fill();
      ctx.strokeStyle = "rgba(245,197,24,0.6)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    function drawLanding() {
      if (!ctx) return;
      for (let r = 0; r < 3; r++) {
        const rProgress = ((frame - totalFrames) / 30 + r * 0.33) % 1;
        if (rProgress < 0) continue;
        const rRadius = rProgress * 28;
        const rAlpha = (1 - rProgress) * 0.6;
        ctx.beginPath();
        ctx.arc(LAND_X, LAND_Y, rRadius, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(245,197,24,${rAlpha})`;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(LAND_X, LAND_Y, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#f5c518";
      ctx.fill();

      // Distance text at landing point
      ctx.font = "bold 14px 'Rajdhani', sans-serif";
      ctx.fillStyle = "#f5c518";
      ctx.textAlign = "center";
      ctx.fillText(`${distance} ${lang === "zh" ? "码" : "yds"}`, LAND_X, LAND_Y - 18);
      ctx.textAlign = "left";
    }

    function drawHUD() {
      if (!ctx) return;
      ctx.fillStyle = "rgba(13,10,26,0.85)";
      ctx.beginPath();
      ctx.roundRect(W * 0.04, H * 0.04, 185, 130, 10);
      ctx.fill();
      ctx.strokeStyle = "rgba(124,58,237,0.3)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(W * 0.04, H * 0.04, 185, 130, 10);
      ctx.stroke();

      ctx.strokeStyle = "#f5c518";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(W * 0.04 + 10, H * 0.04);
      ctx.lineTo(W * 0.04 + 60, H * 0.04);
      ctx.stroke();

      ctx.font = "bold 11px 'Rajdhani', sans-serif";
      ctx.fillStyle = "#ffd85e";
      ctx.fillText("STELLAR AI · SHOT SIM", W * 0.04 + 10, H * 0.04 + 18);

      const lines = lang === "zh" ? [
        ["预测距离", `${distance} 码`],
        ["落点偏差", `${deviation > 0 ? "+" : ""}${deviation.toFixed(1)} 码`],
        ["球路类型", shotShape],
        ["杆头速度", `${prediction.club_head_speed} mph`],
        ["球速", `${prediction.ball_speed} mph`],
        ["发射角", `${prediction.launch_angle}°`],
        ["击球系数", `${prediction.smash_factor}`],
      ] : [
        ["Distance", `${distance} yds`],
        ["Offset", `${deviation > 0 ? "+" : ""}${deviation.toFixed(1)} yds`],
        ["Shot Shape", shotShape],
        ["Club Speed", `${prediction.club_head_speed} mph`],
        ["Ball Speed", `${prediction.ball_speed} mph`],
        ["Launch", `${prediction.launch_angle}°`],
        ["Smash", `${prediction.smash_factor}`],
      ];

      ctx.font = "10px 'Rajdhani', sans-serif";
      lines.forEach(([label, val], i) => {
        const ly = H * 0.04 + 34 + i * 13;
        ctx.fillStyle = "#8b7db5";
        ctx.fillText(label, W * 0.04 + 10, ly);
        ctx.fillStyle = "#f0eaff";
        ctx.fillText(val, W * 0.04 + 100, ly);
      });
    }

    function render() {
      if (!ctx) return;
      ctx.clearRect(0, 0, W, H);
      drawCourse();
      const progress = Math.min(frame / totalFrames, 1);
      if (frame <= totalFrames) {
        drawBall(progress);
      } else {
        ctx.beginPath();
        ctx.arc(LAND_X, LAND_Y, 6, 0, Math.PI * 2);
        ctx.fillStyle = "#ffffff";
        ctx.fill();
        ctx.strokeStyle = "rgba(245,197,24,0.8)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        drawLanding();
      }
      drawHUD();
      frame++;
      if (frame < totalFrames + 60) {
        animRef.current = requestAnimationFrame(render);
      }
    }

    if (animRef.current) cancelAnimationFrame(animRef.current);
    frame = 0;
    animRef.current = requestAnimationFrame(render);
  }, [prediction, distance, deviation, shotShape, carry, lang]);

  useEffect(() => {
    runAnimation();
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [runAnimation]);

  return (
    <div className="glass-card overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2.5 border-b"
           style={{ borderColor: "rgba(124,58,237,0.2)" }}>
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-brand-gold animate-pulse
                           shadow-[0_0_6px_rgba(245,197,24,0.8)]" />
          <span className="text-sm font-bold tracking-widest text-brand-gold font-display">
            {lang === "zh" ? "击球模拟 · SHOT SIM" : "SHOT SIMULATOR"}
          </span>
        </div>
        <button
          onClick={runAnimation}
          className="text-xs font-semibold px-3 py-1 rounded-full
                     border border-[rgba(124,58,237,0.4)] text-white/50
                     hover:border-brand-gold hover:text-brand-gold transition-all">
          ↺ {lang === "zh" ? "重播" : "Replay"}
        </button>
      </div>
      <canvas
        ref={canvasRef}
        width={560}
        height={260}
        className="w-full"
      />
      <div className="grid grid-cols-4 gap-px bg-[rgba(124,58,237,0.15)] border-t"
           style={{ borderColor: "rgba(124,58,237,0.2)" }}>
        {[
          { label: lang === "zh" ? "预测距离" : "Distance", value: `${distance}${lang === "zh" ? "码" : "yds"}`, sub: `Carry ${carry} / Roll ${roll}` },
          { label: lang === "zh" ? "球路" : "Shape", value: shotShape, sub: `${prediction.launch_angle}° launch` },
          { label: lang === "zh" ? "杆头速度" : "Club Speed", value: `${prediction.club_head_speed}`, sub: "mph" },
          { label: lang === "zh" ? "球速" : "Ball Speed", value: `${prediction.ball_speed}`, sub: `Smash ${prediction.smash_factor}` },
        ].map((s, i) => (
          <div key={i} className="flex flex-col items-center py-3 bg-[#0d0a1a]">
            <div className="font-display text-lg" style={{ color: "#f5c518" }}>{s.value}</div>
            <div className="text-xs font-semibold text-[#f0eaff]/80">{s.label}</div>
            <div className="text-[10px] text-[#8b7db5]">{s.sub}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
