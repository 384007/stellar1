// 整段 /pro 走 Edge，满足 @cloudflare/next-on-pages 对动态子路由的要求（与 [analysisId]/page 双保险）。
export const runtime = "edge";

export default function ProLayout({ children }: { children: React.ReactNode }) {
  return children;
}
