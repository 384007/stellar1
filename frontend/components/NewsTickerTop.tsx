"use client";

const TIPS = [
  "✦ 握杆压力：保持轻柔均匀  Grip Pressure: Keep it light and even",
  "✦ 转肩角度：后摆≥90°  Shoulder Turn: Backswing ≥ 90°",
  "✦ 重心转移：前脚蹬地发力  Weight Transfer: Push off front foot",
  "✦ 脊柱角度：击球时保持倾斜  Spine Angle: Maintain tilt through impact",
  "✦ 手腕铰链：后摆早期设定  Wrist Hinge: Set early in backswing",
  "✦ 髋部旋转：臀部先行  Hip Rotation: Lead with the hips",
  "✦ 收杆：高手位完整收杆  Follow Through: High hands, full finish",
  "✦ 球位：左脚跟内侧（一号木）  Ball Position: Inside left heel for driver",
  "✦ 节奏：后摆与下杆比3:1  Tempo: 3:1 backswing to downswing ratio",
  "✦ X因子：最大化肩臀分离  X-Factor: Maximize shoulder-hip separation",
  "✦ 延迟释放：下杆时保持手腕角度  Lag: Maintain wrist angle into downswing",
];

export default function NewsTickerTop() {
  const content = TIPS.join("       ");
  return (
    <div className="relative overflow-hidden py-2.5"
         style={{
           background: "linear-gradient(90deg,#0d0a1a,#1a1530,#231d42,#1a1530,#0d0a1a)",
           borderBottom: "1px solid rgba(124,58,237,0.3)",
         }}>
      {/* Shimmer sweep overlay */}
      <div className="absolute inset-0 pointer-events-none"
           style={{
             background: "linear-gradient(90deg,transparent 0%,rgba(245,197,24,0.04) 50%,transparent 100%)",
             animation: "shimmer 4s linear infinite",
           }} />
      {/* Left/right fade masks */}
      <div className="absolute left-0 top-0 bottom-0 w-16 z-10 pointer-events-none"
           style={{ background: "linear-gradient(90deg,#0d0a1a,transparent)" }} />
      <div className="absolute right-0 top-0 bottom-0 w-16 z-10 pointer-events-none"
           style={{ background: "linear-gradient(-90deg,#0d0a1a,transparent)" }} />

      <div className="animate-ticker-scroll flex whitespace-nowrap">
        {[0,1].map(i => (
          <span key={i} className="inline-block px-6 text-sm font-semibold tracking-wide"
                style={{ color: "#ffd85e", textShadow: "0 0 8px rgba(245,197,24,0.4)" }}>
            {content}
          </span>
        ))}
      </div>
    </div>
  );
}
