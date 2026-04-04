import ProPage from "../page";

export default async function ProAnalysisByIdPage({
  params,
}: {
  params: Promise<{ analysisId: string }>;
}) {
  const { analysisId } = await params;
  const id = decodeURIComponent(analysisId || "").trim();
  return <ProPage deepLinkAnalysisId={id} />;
}
