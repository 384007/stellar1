"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function ProLoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [inviteCode, setInviteCode] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function storeAndRedirect(data: Record<string, unknown>) {
    const token = data.token as string;
    if (!token || !token.includes(".")) {
      setError("登录异常，未获取到有效凭证，请重试");
      return;
    }
    localStorage.setItem("stellar_token", token);
    localStorage.setItem("stellar_user", JSON.stringify({
      user_id: data.user_id,
      email: data.email,
      username: data.username,
      is_pro: data.is_pro,
      is_guest: false,
    }));
    router.push("/pro");
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const res = await fetch("/api/auth", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "pro-login",
          email,
          password,
          username: username.trim() || undefined,
          invite_code: inviteCode,
        }),
      });

      const data = await res.json();

      if (res.ok) {
        storeAndRedirect(data);
        return;
      }

      const msg = data.detail || "";

      if (res.status === 503) {
        setError("服务器暂时不可用，请稍后重试");
        return;
      }

      throw new Error(msg || "认证失败");
    } catch (err: unknown) {
      if (
        err instanceof TypeError &&
        (err.message.includes("fetch") || err.message.includes("network"))
      ) {
        setError("网络连接失败，请检查网络后重试");
        return;
      }
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="relative flex min-h-screen items-center justify-center px-4 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-black via-brand-dark to-black" />
        <div className="absolute left-1/4 top-1/4 h-96 w-96 rounded-full bg-brand-gold/5 blur-[120px]" />
        <div className="absolute bottom-1/4 right-1/4 h-96 w-96 rounded-full bg-brand-gold/5 blur-[120px]" />
        <div
          className="absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage:
              "linear-gradient(rgba(212,175,55,0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(212,175,55,0.5) 1px, transparent 1px)",
            backgroundSize: "60px 60px",
          }}
        />
      </div>

      <div className="relative z-10 w-full max-w-md">
        <div className="mb-8 text-center">
          <a href="/" className="inline-flex items-center gap-2">
            <img src="/logo.svg" alt="Stellar" className="h-14 w-14" />
          </a>
          <h1 className="mt-4 text-4xl font-bold">
            <span className="bg-gradient-to-r from-brand-gold via-brand-gold-light to-brand-gold bg-clip-text text-transparent">
              PRO ACCESS
            </span>
          </h1>
          <p className="mt-2 text-white/50">Unlock the full power of AI swing analysis</p>
          <p className="text-sm text-brand-gold/60">解锁AI挥杆分析的全部功能</p>
        </div>

        <div className="relative overflow-hidden rounded-2xl border border-brand-gold/20 bg-black/60 p-8 backdrop-blur-xl">
          {/* Gold corner accents */}
          <div className="absolute left-0 top-0 h-16 w-px bg-gradient-to-b from-brand-gold to-transparent" />
          <div className="absolute left-0 top-0 h-px w-16 bg-gradient-to-r from-brand-gold to-transparent" />
          <div className="absolute right-0 top-0 h-16 w-px bg-gradient-to-b from-brand-gold to-transparent" />
          <div className="absolute right-0 top-0 h-px w-16 bg-gradient-to-l from-brand-gold to-transparent" />
          <div className="absolute bottom-0 left-0 h-16 w-px bg-gradient-to-t from-brand-gold to-transparent" />
          <div className="absolute bottom-0 left-0 h-px w-16 bg-gradient-to-r from-brand-gold to-transparent" />
          <div className="absolute bottom-0 right-0 h-16 w-px bg-gradient-to-t from-brand-gold to-transparent" />
          <div className="absolute bottom-0 right-0 h-px w-16 bg-gradient-to-l from-brand-gold to-transparent" />

          <div className="mb-6 flex items-center justify-center">
            <span className="rounded-full border border-brand-gold/30 bg-brand-gold/10 px-4 py-1 text-xs font-semibold tracking-widest text-brand-gold">
              ★ PRO MEMBER ★
            </span>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="mb-1 block text-sm text-brand-gold/70">
                用户名 / Username <span className="text-white/30 text-xs">（新注册填写）</span>
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full rounded-xl border border-brand-gold/20 bg-black/50 px-4 py-3 text-white placeholder-white/30 transition focus:border-brand-gold/50 focus:outline-none focus:ring-2 focus:ring-brand-gold/20"
                placeholder="您的昵称"
                maxLength={20}
              />
            </div>

            <div>
              <label className="mb-1 block text-sm text-brand-gold/70">
                Email / 邮箱
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full rounded-xl border border-brand-gold/20 bg-black/50 px-4 py-3 text-white placeholder-white/30 transition focus:border-brand-gold/50 focus:outline-none focus:ring-2 focus:ring-brand-gold/20"
                placeholder="pro@stellar.ai"
                required
              />
            </div>

            <div>
              <label className="mb-1 block text-sm text-brand-gold/70">
                Password / 密码
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full rounded-xl border border-brand-gold/20 bg-black/50 px-4 py-3 text-white placeholder-white/30 transition focus:border-brand-gold/50 focus:outline-none focus:ring-2 focus:ring-brand-gold/20"
                placeholder="••••••••"
                required
                minLength={6}
              />
            </div>

            <div>
              <label className="mb-1 block text-sm text-brand-gold/70">
                邀请码 / Invite Code
              </label>
              <input
                type="password"
                value={inviteCode}
                onChange={(e) => setInviteCode(e.target.value)}
                className="w-full rounded-xl border border-brand-gold/20 bg-black/50 px-4 py-3 text-white placeholder-white/30 transition focus:border-brand-gold/50 focus:outline-none focus:ring-2 focus:ring-brand-gold/20"
                placeholder="请输入邀请码"
                autoComplete="off"
                required
              />
            </div>

            {error && (
              <div className="rounded-lg bg-red-500/10 border border-red-500/20 p-3 text-sm text-red-400">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="btn-pro glow-border w-full disabled:opacity-50"
            >
              {loading ? "Authenticating..." : "Enter PRO / 进入PRO"}
            </button>
          </form>

          <div className="mt-6 space-y-3 border-t border-brand-gold/10 pt-6">
            <h3 className="text-center text-xs font-semibold tracking-widest text-brand-gold/50">
              PRO FEATURES / PRO功能
            </h3>
            <div className="grid grid-cols-2 gap-2 text-xs text-white/50">
              {[
                "Pro Player Comparison / 职业球员对比",
                "7-Day Training Plan / 7天训练计划",
                "Advanced HUD Overlay / 高级HUD叠加",
                "Shot Prediction Sim / 击球预测模拟",
                "Detailed AI Report / 详细AI报告",
                "Unlimited Analysis / 无限分析次数",
              ].map((feature, i) => (
                <div key={i} className="flex items-start gap-1">
                  <span className="text-brand-gold">✦</span>
                  <span>{feature}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-4 text-center">
            <a href="/login" className="text-sm text-white/40 hover:text-white/60">
              Standard login instead / 返回普通登录
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
