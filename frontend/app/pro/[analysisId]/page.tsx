import ProPageClient from "../ProPageClient";

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
