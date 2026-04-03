"use client";

export default function HeroSection({ isPro = false }: { isPro?: boolean }) {
  return (
    <section className="relative overflow-hidden py-24 lg:py-36">
      {/* Deep purple radial bg */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2
                        h-[700px] w-[700px] rounded-full
                        bg-[radial-gradient(ellipse,rgba(124,58,237,0.18)_0%,transparent_70%)]" />
        <div className="absolute right-0 top-0 h-96 w-96 rounded-full
                        bg-[radial-gradient(ellipse,rgba(245,197,24,0.07)_0%,transparent_70%)]" />
        <div className="absolute left-0 bottom-0 h-64 w-64 rounded-full
                        bg-[radial-gradient(ellipse,rgba(124,58,237,0.1)_0%,transparent_70%)]" />
      </div>

      <div className="relative z-10 mx-auto max-w-7xl px-6 text-center">
        {/* Badge */}
        <div className="mb-8 inline-flex items-center gap-2 rounded-full px-5 py-2
                        border border-[rgba(124,58,237,0.4)] bg-[rgba(124,58,237,0.1)]
                        backdrop-blur-sm animate-fade-in">
          <span className="h-2 w-2 animate-breathe rounded-full bg-brand-gold shadow-[0_0_8px_rgba(245,197,24,0.8)]" />
          <span className="text-sm font-semibold tracking-wide text-brand-gold-light">
            Stellar AI · Advanced Motion Analysis Engine
          </span>
        </div>

        {/* Headline */}
        <h1 className="mb-6 font-display leading-none tracking-wider animate-slide-up">
          <span className="block text-6xl lg:text-8xl text-white/90">AI-POWERED</span>
          <span className="block text-6xl lg:text-8xl text-gold-gradient"
                style={{
                  background: "linear-gradient(90deg,#c9960a,#ffd85e,#f5c518,#ffd85e,#c9960a)",
                  backgroundSize: "200% auto",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  animation: "shimmer 4s linear infinite",
                }}>
            STELLAR AI
          </span>
          <span className="block text-5xl lg:text-7xl"
                style={{
                  background: "linear-gradient(135deg,#9f5fff,#c084fc)",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                }}>
            GOLF ANALYSIS
          </span>
        </h1>

        {/* Subtext */}
        <p className="mx-auto mb-2 max-w-2xl text-lg text-[#8b7db5] animate-fade-in">
          Upload your swing video and get instant professional-grade analysis
          with skeleton HUD overlay, scoring, and improvement suggestions.
        </p>
        <p className="mx-auto mb-12 max-w-2xl text-base text-[#8b7db5]/70 animate-fade-in">
          上传挥杆视频，获取专业级分析 · 骨架HUD叠加 · 评分建议
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-col items-center justify-center gap-4 sm:flex-row animate-slide-up">
          <a href="/analyze" className="btn-primary text-lg">
            ⚡ Free Analysis · 免费分析
          </a>
          <a href={isPro ? "/pro" : "/pro-login"} className="btn-pro glow-border text-lg">
            ★ PRO Analysis · 专业版
          </a>
          <a href="/shot-lab" className="btn-lab text-lg">
            🔬 Shot Lab · 击球实验室
          </a>
        </div>

        {/* Divider line */}
        <div className="mt-16 flex items-center justify-center gap-4">
          <div className="h-px flex-1 bg-gradient-to-r from-transparent via-[rgba(124,58,237,0.4)] to-transparent max-w-xs" />
          <div className="h-2 w-2 rounded-full bg-brand-gold animate-breathe
                          shadow-[0_0_8px_rgba(245,197,24,0.8)]" />
          <div className="h-px flex-1 bg-gradient-to-r from-transparent via-[rgba(124,58,237,0.4)] to-transparent max-w-xs" />
        </div>

        {/* Stats */}
        <div className="mt-10 grid grid-cols-3 gap-8">
          {[
            { value: "5",     label: "Analysis Dimensions", sub: "分析维度" },
            { value: "8+",    label: "Skeleton Joints",     sub: "骨架关节" },
            { value: "< 30s", label: "Analysis Time",       sub: "分析时间" },
          ].map((stat, i) => (
            <div key={i} className="text-center group">
              <div className="font-display text-4xl text-brand-gold
                              group-hover:drop-shadow-[0_0_12px_rgba(245,197,24,0.8)]
                              transition-all duration-300">
                {stat.value}
              </div>
              <div className="mt-1 text-sm font-semibold text-[#f0eaff]/70">{stat.label}</div>
              <div className="text-xs text-[#8b7db5]">{stat.sub}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
