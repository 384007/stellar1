import ProPageClient from "./ProPageClient";

// Explicit Edge for /pro (not only layout) — @cloudflare/next-on-pages expects
// App Router segments to declare Edge; client-only page.tsx can be misclassified.
export const runtime = "edge";

export default function ProPage() {
  return <ProPageClient />;
}
