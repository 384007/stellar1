"use client";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="flex min-h-screen items-center justify-center px-6">
      <div className="text-center max-w-md">
        <h2 className="text-xl font-bold text-white mb-3">页面加载出错</h2>
        <p className="text-sm text-white/50 mb-6">
          {error?.message || "发生了未知错误"}
        </p>
        <div className="flex flex-wrap gap-3 justify-center">
          <button
            onClick={() => reset()}
            className="rounded-lg bg-purple-600 px-5 py-2 text-sm font-medium text-white hover:bg-purple-500 transition"
          >
            重试
          </button>
          <a
            href="/"
            className="rounded-lg border border-white/20 px-5 py-2 text-sm font-medium text-white hover:bg-white/10 transition"
          >
            返回首页
          </a>
          <button
            type="button"
            onClick={() => {
              if ("caches" in window) {
                caches.keys().then((names) => names.forEach((n) => caches.delete(n)));
              }
              if ("serviceWorker" in navigator) {
                navigator.serviceWorker.getRegistrations().then((regs) =>
                  regs.forEach((r) => r.unregister()),
                );
              }
              window.location.href = "/";
            }}
            className="rounded-lg border border-white/20 px-5 py-2 text-sm font-medium text-white hover:bg-white/10 transition"
          >
            清除缓存并刷新
          </button>
        </div>
      </div>
    </div>
  );
}
