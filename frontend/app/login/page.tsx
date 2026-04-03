"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
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
    router.push("/analyze");
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
          action: mode,
          email,
          password,
          ...(mode === "register" ? { username } : {}),
        }),
      });

      if (res.ok) {
        storeAndRedirect(await res.json());
        return;
      }

      const errData = await res.json().catch(() => ({ detail: "" }));
      const serverMsg = errData.detail || "";

      if (res.status === 503) {
        setError("服务器暂时不可用，请稍后重试");
        return;
      }

      throw new Error(serverMsg || "操作失败");
    } catch (err: unknown) {
      if (
        err instanceof TypeError &&
        (err.message.includes("fetch") || err.message.includes("network"))
      ) {
        setError("网络连接失败，请检查网络后重试");
        return;
      }
      setError(err instanceof Error ? err.message : "操作失败");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <div className="w-full max-w-md">
        <div className="mb-8 text-center">
          <a href="/" className="inline-flex items-center gap-2">
            <img src="/logo.svg" alt="Stellar" className="h-12 w-12" />
            <span className="text-3xl font-bold text-brand-gold">STELLAR</span>
          </a>
          <p className="mt-2 text-white/60">
            {mode === "login" ? "欢迎回来" : "创建您的账户"}
          </p>
          <p className="text-sm text-white/40">
            {mode === "login" ? "Welcome back" : "Create your account"}
          </p>
        </div>

        <div className="glass-card p-8">
          <div className="mb-6 flex rounded-xl bg-white/5 p-1">
            <button
              onClick={() => setMode("login")}
              className={`flex-1 rounded-lg py-2 text-sm font-medium transition ${
                mode === "login"
                  ? "bg-brand-purple text-white"
                  : "text-white/60 hover:text-white"
              }`}
            >
              登录 / Sign In
            </button>
            <button
              onClick={() => setMode("register")}
              className={`flex-1 rounded-lg py-2 text-sm font-medium transition ${
                mode === "register"
                  ? "bg-brand-purple text-white"
                  : "text-white/60 hover:text-white"
              }`}
            >
              注册 / Register
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {mode === "register" && (
              <div>
                <label className="mb-1 block text-sm text-white/70">
                  用户名 / Username
                </label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="input-field"
                  placeholder="您的昵称"
                  required
                  minLength={2}
                  maxLength={20}
                />
              </div>
            )}
            <div>
              <label className="mb-1 block text-sm text-white/70">
                邮箱 / Email
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="input-field"
                placeholder="your@email.com"
                required
              />
            </div>
            <div>
              <label className="mb-1 block text-sm text-white/70">
                密码 / Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="input-field"
                placeholder="••••••••"
                required
                minLength={6}
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
              className="btn-primary w-full disabled:opacity-50"
            >
              {loading
                ? "处理中..."
                : mode === "login"
                  ? "登录 / Sign In"
                  : "创建账户 / Create Account"}
            </button>
          </form>

          <div className="mt-4 text-center">
            <a
              href="/pro-login"
              className="text-sm text-brand-gold hover:underline"
            >
              寻找PRO专业版？ / Looking for PRO access?
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
