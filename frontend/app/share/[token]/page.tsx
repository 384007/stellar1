import SharePageGate from "./SharePageGate";

export const runtime = "edge";

export default async function SharePage({ params }: { params: Promise<{ token: string }> }) {
  const { token } = await params;
  return <SharePageGate token={token} />;
}
