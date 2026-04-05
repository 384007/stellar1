import ProPageClient from "../ProPageClient";

// @cloudflare/next-on-pages：动态 App 路由必须 Edge，否则 Pages 构建报
// "routes were not configured to run with the Edge Runtime: /pro/[analysisId]".
export const runtime = "edge";

export default async function ProAnalysisByIdPage({
  params,
}: {
  params: Promise<{ analysisId: string }>;
}) {
  const { analysisId } = await params;
  const id = decodeURIComponent(analysisId || "").trim();
  return <ProPageClient deepLinkAnalysisId={id} />;
}
