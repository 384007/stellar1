"use client";

import { useState, useEffect } from "react";
import NewsTickerTop from "@/components/NewsTickerTop";
import HeroSection from "@/components/HeroSection";
import NewsCarousel from "@/components/NewsCarousel";
import { preloadPoseModel } from "@/lib/mediapipe-assets";

export default function HomePage() {
  const [user, setUser] = useState<{ username?: string; email?: string } | null>(null);
  const [isPro, setIsPro] = useState(false);
  const [lang, setLang] = useState<"zh" | "en">("zh");

  useEffect(() => {
    const storedLang = localStorage.getItem("stellar_lang");
    if (storedLang === "zh" || storedLang === "en") {
      setLang(storedLang);
    }

    const stored = localStorage.getItem("stellar_user");
    if (stored) {
      try {
        const u = JSON.parse(stored);
        setUser(u);
        setIsPro(!!u.is_pro);
      } catch { /* ignore */ }
    }

    // Pre-fetch MediaPipe lite model in background so skeleton page starts instantly.
    preloadPoseModel();
  }, []);

  function handleSignOut() {
    localStorage.removeItem("stellar_token");
    localStorage.removeItem("stellar_user");
    setUser(null);
    setIsPro(false);
  }

  const flowSteps = [
    {
      icon: "📤",
      en: "Upload / Capture",
      zh: "上传 / 实拍",
      descEn: "Upload swing video or capture directly with camera.",
      descZh: "上传挥杆视频，或直接使用摄像头实拍。",
    },
    {
      icon: "🧠",
      en: "AI Analysis",
      zh: "AI 分析",
      descEn: "Stellar AI scores movement quality and detects key issues.",
      descZh: "Stellar AI 对动作质量评分并识别关键问题。",
    },
    {
      icon: "📈",
      en: "Training Plan",
      zh: "训练计划",
      descEn: "Get focused drills and step-by-step swing improvement.",
      descZh: "获得针对性训练动作与逐步提升路径。",
    },
  ];

  const caseStudies = [
    {
      name: lang === "zh" ? "用户 A · 右曲球修正" : "User A · Slice Fix",
      before: 58,
      after: 81,
      issue: lang === "zh" ? "下杆由外到内，杆面开放" : "Over-the-top downswing, open clubface",
      result: lang === "zh" ? "球路从右曲变为轻微小左曲，命中率明显提升" : "Ball flight improved from slice to controlled draw; accuracy up",
    },
    {
      name: lang === "zh" ? "用户 B · 顶点稳定" : "User B · Top Position Stability",
      before: 62,
      after: 86,
      issue: lang === "zh" ? "后摆顶点过度内收，节奏紊乱" : "Collapsed top position and unstable tempo",
      result: lang === "zh" ? "后摆结构稳定，X-Factor 提升，击球更扎实" : "More stable backswing structure, better X-Factor, cleaner strike",
    },
    {
      name: lang === "zh" ? "用户 C · 距离提升" : "User C · Distance Gain",
      before: 64,
      after: 88,
      issue: lang === "zh" ? "重心转移不足，提前释放" : "Poor weight shift and early release",
      result: lang === "zh" ? "杆头速度提升，平均开球距离 +17 码" : "Higher clubhead speed, average driving distance +17 yards",
    },
  ];

  const compareRows = [
    { labelEn: "3D Reconstruction", labelZh: "3D 姿态重建", lite: false, pro: true },
    { labelEn: "Swing Flow Playback", labelZh: "Swing Flow 逐步回放", lite: false, pro: true },
    { labelEn: "Pro Comparison", labelZh: "职业球员对比", lite: false, pro: true },
    { labelEn: "7-Day Training Plan", labelZh: "7 天训练计划", lite: false, pro: true },
  ];

  const dailyTips = [
    { en: "Focus today: downswing sequence (hips lead, then torso).", zh: "今日重点：下杆顺序（先髋后躯干）。" },
    { en: "Focus today: keep your head stable through impact.", zh: "今日重点：触球前后头部保持稳定。" },
    { en: "Focus today: hold lag a little longer before release.", zh: "今日重点：延后释放手腕角度（Lag）。" },
    { en: "Focus today: complete finish and hold balance for 3 seconds.", zh: "今日重点：完整收杆并平衡停住 3 秒。" },
    { en: "Focus today: improve setup alignment to target line.", zh: "今日重点：站姿瞄准线对齐目标线。" },
  ];

  const dayIndex = Math.floor(Date.now() / 86400000) % dailyTips.length;
  const todayTip = dailyTips[dayIndex];

  const faqs = [
    {
      qEn: "Why don't I see 3D?",
      qZh: "为什么没有看到 3D？",
      aEn: "3D requires clear body landmarks. Please upload/capture a full-body swing with good lighting.",
      aZh: "3D 需要清晰识别人体关键点，请上传/实拍全身入镜且光线充足的挥杆视频。",
    },
    {
      qEn: "What video quality works best?",
      qZh: "视频拍摄有什么要求？",
      aEn: "Use side-view angle, full body in frame, stable camera, and at least 720p.",
      aZh: "建议侧面机位、全身入框、镜头稳定、分辨率不低于 720p。",
    },
    {
      qEn: "How long does analysis take?",
      qZh: "分析一般要多久？",
      aEn: "Usually 20-60 seconds depending on video length and mode (Lite / Pro).",
      aZh: "通常 20-60 秒，取决于视频长度以及普通/Pro 模式。",
    },
    {
      qEn: "What is the main difference between Lite and Pro?",
      qZh: "普通分析和 Pro 最大区别是什么？",
      aEn: "Pro includes 3D reconstruction, swing flow, pro comparison, and training plans.",
      aZh: "Pro 包含 3D 重建、Swing Flow、职业对比与训练计划。",
    },
    {
      qEn: "Can I use phone camera directly?",
      qZh: "可以直接用手机摄像头吗？",
      aEn: "Yes. Live capture works on modern mobile browsers with camera permissions.",
      aZh: "可以。现代手机浏览器授权摄像头后可直接实拍分析。",
    },
  ];

  return (
    <main className="min-h-screen pb-24">
      <NewsTickerTop />

      <nav className="sticky top-0 z-50 border-b border-white/5 bg-brand-dark/80 backdrop-blur-xl pt-[max(0.5rem,env(safe-area-inset-top,0px))]">
        <div className="mx-auto flex w-full max-w-7xl items-center gap-3 px-4 py-3 sm:gap-4 sm:px-6 sm:py-4">
          <a href="/" className="flex min-w-0 max-w-[48%] shrink items-center gap-2 sm:max-w-none sm:gap-3">
            <img src="/logo.svg" alt="Stellar" className="h-9 w-9 flex-shrink-0 sm:h-10 sm:w-10" />
            <span className="truncate text-lg font-bold tracking-[0.12em] text-brand-gold sm:text-xl md:text-2xl md:tracking-[0.1em]">
              STELLAR
            </span>
          </a>
          <div className="ml-auto flex flex-shrink-0 items-center gap-2 sm:gap-3">
            {user ? (
              <>
                <a
                  href="/history"
                  className="flex max-w-[9rem] items-center gap-1 rounded-lg border border-white/10 px-2 py-1.5 text-xs text-white/70 transition hover:border-brand-gold/30 hover:text-brand-gold sm:max-w-[11rem] sm:gap-1.5 sm:px-3 sm:text-sm"
                >
                  <svg className="h-3.5 w-3.5 flex-shrink-0 sm:h-4 sm:w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A17.933 17.933 0 0 1 12 21.75c-2.676 0-5.216-.584-7.499-1.632Z" />
                  </svg>
                  <span className="truncate font-medium text-brand-gold">{user.username || user.email}</span>
                </a>
                <button
                  onClick={handleSignOut}
                  className="rounded-lg px-2 py-1.5 text-[11px] text-white/40 transition hover:text-white/70 sm:px-3 sm:py-2 sm:text-xs"
                >
                  {lang === "zh" ? "退出" : "Sign Out"}
                </button>
              </>
            ) : (
              <>
                <a
                  href="/login"
                  className="rounded-lg border border-white/15 bg-white/[0.04] px-3 py-1.5 text-xs font-semibold tracking-wide text-white/90 transition hover:border-brand-purple/40 hover:bg-white/[0.07] hover:text-white sm:px-4 sm:py-2 sm:text-sm"
                >
                  {lang === "zh" ? "登录" : "Sign In"}
                </a>
                <a
                  href="/pro-login"
                  className="btn-pro whitespace-nowrap rounded-lg px-2.5 py-1.5 text-[10px] font-semibold leading-tight sm:px-4 sm:py-2 sm:text-sm"
                >
                  PRO Access
                </a>
              </>
            )}
            <button
              onClick={() => {
                const next = lang === "en" ? "zh" : "en";
                setLang(next);
                localStorage.setItem("stellar_lang", next);
              }}
              className="rounded-lg border border-white/10 px-2 py-1.5 text-[11px] text-white/60 transition hover:text-white sm:px-3 sm:text-xs"
            >
              {lang === "en" ? "中文" : "EN"}
            </button>
          </div>
        </div>
      </nav>

      <HeroSection isPro={isPro} />
      <NewsCarousel />

      {/* Features Grid */}
      <section className="mx-auto max-w-7xl px-6 py-24">
        <h2 className="mb-4 text-center text-3xl font-bold text-white">
          Powered by <span className="text-brand-gold">Stellar AI</span>
        </h2>
        <p className="mx-auto mb-16 max-w-2xl text-center text-white/60">
          Cutting-edge technology meets professional golf coaching
        </p>

        <div className="grid items-stretch gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {[
            {
              icon: "🎯",
              title: "AI Swing Scoring",
              titleZh: "AI挥杆评分",
              desc: "5-dimension analysis powered by Stellar AI with real-time scoring",
              descZh: "由Stellar AI驱动的五维度分析与实时评分",
            },
            {
              icon: "🦴",
              title: "Skeleton HUD",
              titleZh: "骨架HUD",
              desc: "MediaPipe pose detection with interactive bone overlay visualization",
              descZh: "MediaPipe姿态检测，交互式骨架叠加可视化",
            },
            {
              icon: "📊",
              title: "Pro Comparison",
              titleZh: "职业球员对比",
              desc: "Compare your swing with Tiger Woods, Rory McIlroy and more",
              descZh: "与泰格·伍兹、麦克罗伊等职业球员对比",
            },
            {
              icon: "🏌️",
              title: "Shot Prediction",
              titleZh: "击球预测",
              desc: "Physics-based ball flight simulation with distance and shape prediction",
              descZh: "基于物理的弹道模拟，预测距离和球路",
            },
          ].map((feature, i) => (
            <div
              key={i}
              className="glass-card group flex h-full flex-col p-6 transition-all duration-300 hover:scale-[1.02] hover:border-brand-purple/30"
            >
              <div className="mb-4 text-4xl">{feature.icon}</div>
              <h3 className="mb-1 text-lg font-semibold text-white">
                {feature.title}
              </h3>
              <p className="mb-2 text-xs text-brand-gold">{feature.titleZh}</p>
              <p className="text-sm text-white/60">{feature.desc}</p>
              <p className="mt-2 text-xs text-white/40">{feature.descZh}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Shot Lab CTA — right after Features Grid */}
      <section className="mx-auto max-w-7xl px-6 py-8">
        <div className="relative overflow-hidden rounded-2xl border border-red-500/30 p-[1px]"
          style={{ background: "linear-gradient(135deg, rgba(185,28,28,0.6), rgba(239,68,68,0.3), rgba(185,28,28,0.5))" }}>
          <div className="relative rounded-2xl overflow-hidden bg-gradient-to-r from-[#1a0606] via-[#0d0a1a] to-[#1a0606] p-8"
            style={{ boxShadow: "inset 0 0 60px rgba(239,68,68,0.08)" }}>
            <div className="pointer-events-none absolute -right-10 -top-10 h-56 w-56 rounded-full blur-3xl"
              style={{ background: "radial-gradient(ellipse, rgba(239,68,68,0.2) 0%, transparent 70%)", animation: "breathe 3s ease-in-out infinite" }} />
            <div className="pointer-events-none absolute -bottom-10 left-10 h-40 w-40 rounded-full blur-3xl"
              style={{ background: "radial-gradient(ellipse, rgba(239,68,68,0.15) 0%, transparent 70%)", animation: "breathe 4s ease-in-out infinite 1s" }} />

            <div className="relative z-10 flex flex-col items-center text-center md:flex-row md:text-left md:gap-10">
              <div className="relative mb-6 flex-shrink-0 md:mb-0">
                <div className="absolute inset-0 rounded-2xl blur-md" style={{ background: "rgba(239,68,68,0.3)", animation: "breathe 2s ease-in-out infinite" }} />
                <div className="relative flex h-20 w-20 items-center justify-center rounded-2xl text-4xl"
                  style={{ background: "linear-gradient(135deg,#7f1d1d,#ef4444,#7f1d1d)", boxShadow: "0 0 24px rgba(239,68,68,0.5)" }}>
                  🔬
                </div>
              </div>
              <div className="flex-1">
                <div className="mb-1 flex items-center justify-center gap-2 md:justify-start">
                  <h3 className="text-2xl font-bold tracking-wide"
                    style={{
                      background: "linear-gradient(90deg, #b91c1c, #ff6060, #ef4444, #ff6060, #b91c1c)",
                      backgroundSize: "200% auto",
                      WebkitBackgroundClip: "text",
                      WebkitTextFillColor: "transparent",
                      animation: "shimmer 3s linear infinite",
                    }}>
                    Shot Lab
                  </h3>
                  <span className="rounded-full px-2.5 py-0.5 text-[10px] font-bold tracking-widest"
                    style={{
                      background: "rgba(239,68,68,0.2)",
                      border: "1px solid rgba(239,68,68,0.5)",
                      color: "#ff6060",
                      boxShadow: "0 0 8px rgba(239,68,68,0.4)",
                    }}>NEW</span>
                </div>
                <p className="mb-1 text-sm font-bold" style={{ color: "#ff6060" }}>击球实验室</p>
                <p className="mb-5 text-sm text-white/55">
                  {lang === "zh"
                    ? "球速 · 起飞角度 · 触球质量 · 节奏分析。上传视频、实拍或屏幕录制，全指标一键输出。"
                    : "Ball speed · launch angle · contact quality · tempo. Upload, capture, or screen-record for full metrics."}
                </p>
                <div className="flex flex-wrap gap-3 justify-center md:justify-start">
                  <a href="/shot-lab" className="btn-lab px-5 py-2.5 text-sm">
                    {lang === "zh" ? "🔬 进入击球实验室" : "🔬 Open Shot Lab"}
                  </a>
                  <a href="/shot-lab"
                    className="rounded-xl border border-white/15 bg-white/[0.04] px-5 py-2.5 text-sm text-white/55 transition hover:bg-white/10 hover:text-white/80">
                    {lang === "zh" ? "📷 实拍分析" : "📷 Live Capture"}
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Real Case Studies */}
      <section className="mx-auto max-w-7xl px-6 py-8">
        <h3 className="mb-6 text-2xl font-bold text-white">
          {lang === "zh" ? "真实案例（前后对比）" : "Real Case Studies (Before/After)"}
        </h3>
        <div className="grid gap-5 lg:grid-cols-3">
          {caseStudies.map((c) => (
            <div key={c.name} className="glass-card p-5">
              <p className="mb-3 text-sm font-semibold text-brand-gold">{c.name}</p>
              <div className="mb-3 flex items-center gap-3">
                <div className="rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-1.5 text-xs text-red-300">
                  Before {c.before}
                </div>
                <div className="text-white/30">→</div>
                <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/10 px-3 py-1.5 text-xs text-emerald-300">
                  After {c.after}
                </div>
              </div>
              <p className="mb-2 text-xs text-white/45">{lang === "zh" ? "关键问题" : "Key Issue"}</p>
              <p className="mb-3 text-sm text-white/65">{c.issue}</p>
              <p className="mb-2 text-xs text-white/45">{lang === "zh" ? "改进结果" : "Result"}</p>
              <p className="text-sm text-white/65">{c.result}</p>
            </div>
          ))}
        </div>
      </section>

      {/* 3-step Flow */}
      <section className="mx-auto max-w-7xl px-6 py-8">
        <h3 className="mb-6 text-2xl font-bold text-white">
          {lang === "zh" ? "三步完成挥杆进阶" : "3-Step Improvement Flow"}
        </h3>
        <div className="grid gap-4 md:grid-cols-3">
          {flowSteps.map((s, i) => (
            <div key={s.en} className="glass-card relative p-5">
              <div className="mb-3 text-3xl">{s.icon}</div>
              <p className="text-base font-semibold text-white">{lang === "zh" ? s.zh : s.en}</p>
              <p className="mt-2 text-sm text-white/60">{lang === "zh" ? s.descZh : s.descEn}</p>
              <span className="absolute right-4 top-4 rounded-full border border-white/10 px-2 py-0.5 text-[10px] text-white/35">
                0{i + 1}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* Lite vs Pro + Daily Tip */}
      <section className="mx-auto grid max-w-7xl gap-5 px-6 py-8 lg:grid-cols-[2fr_1fr]">
        <div className="glass-card p-6">
          <h3 className="mb-4 text-2xl font-bold text-white">
            {lang === "zh" ? "普通分析 vs Pro" : "Lite vs Pro"}
          </h3>
          <div className="space-y-3">
            {compareRows.map((r) => (
              <div key={r.labelEn} className="grid grid-cols-[1fr_90px_90px] items-center gap-2 rounded-lg border border-white/5 bg-white/[0.02] px-3 py-2">
                <span className="text-sm text-white/70">{lang === "zh" ? r.labelZh : r.labelEn}</span>
                <span className="text-center text-xs text-white/35">{r.lite ? "✓" : "—"}</span>
                <span className="text-center text-xs font-semibold text-brand-gold">{r.pro ? "✓" : "—"}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="glass-card p-6">
          <h3 className="mb-2 text-lg font-semibold text-brand-gold">
            {lang === "zh" ? "今日训练建议" : "Today's Training Tip"}
          </h3>
          <p className="text-sm leading-relaxed text-white/70">
            {lang === "zh" ? todayTip.zh : todayTip.en}
          </p>
        </div>
      </section>

      {/* FAQ */}
      <section className="mx-auto max-w-7xl px-6 py-8">
        <h3 className="mb-6 text-2xl font-bold text-white">FAQ</h3>
        <div className="grid gap-4 md:grid-cols-2">
          {faqs.map((item) => (
            <div key={item.qEn} className="glass-card p-5">
              <p className="mb-2 text-sm font-semibold text-white">{lang === "zh" ? item.qZh : item.qEn}</p>
              <p className="text-sm text-white/60">{lang === "zh" ? item.aZh : item.aEn}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/5 bg-brand-dark/50 py-12">
        <div className="mx-auto max-w-7xl px-6 text-center">
          <div className="mb-4 flex items-center justify-center gap-3">
            <img src="/logo.svg" alt="Stellar" className="h-8 w-8" />
            <span className="text-lg font-bold tracking-[0.1em] text-brand-gold">STELLAR</span>
          </div>
          <p className="text-sm text-white/40">
            © 2026 Stellar AI. All rights reserved.
          </p>
          <p className="mt-1 text-xs text-white/30">
            AI驱动的专业高尔夫挥杆分析平台
          </p>
        </div>
      </footer>

      {/* Bottom fixed CTA */}
      <div className="fixed bottom-0 left-0 right-0 z-40 border-t border-white/10 bg-brand-dark/85 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-center gap-3 px-4 py-3">
          <a
            href="/analyze"
            className="btn-primary rounded-lg px-4 py-2 text-sm font-semibold"
          >
            {lang === "zh" ? "立即开始普通分析" : "Start Lite Analysis"}
          </a>
          <a
            href="/shot-lab"
            className="btn-lab rounded-lg px-4 py-2 text-sm font-semibold"
          >
            {lang === "zh" ? "🔬 击球实验室" : "🔬 Shot Lab"}
          </a>
          <a
            href="/pro-login"
            className="btn-pro rounded-lg px-4 py-2 text-sm font-semibold"
          >
            {lang === "zh" ? "升级 Pro" : "Upgrade to Pro"}
          </a>
        </div>
      </div>
    </main>
  );
}
