"use client";

import dynamic from "next/dynamic";

const SharePageClient = dynamic(() => import("./SharePageClient"), {
  ssr: false,
  loading: () => (
    <div className="flex min-h-screen items-center justify-center bg-brand-dark">
      <div className="flex flex-col items-center gap-4">
        <div className="h-10 w-10 animate-spin rounded-full border-2 border-white/10 border-t-brand-purple" />
        <p className="text-sm text-white/40">加载中…</p>
      </div>
    </div>
  ),
});

export default function SharePageGate({ token }: { token: string }) {
  return <SharePageClient token={token} />;
}
