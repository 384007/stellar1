/** Fetch URL or data URL → trigger browser download; optional open-tab fallback for http(s). */

export function triggerBlobDownload(blob: Blob, filename: string) {
  const href = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = href;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(href);
}

export async function downloadHrefAsFile(href: string, filename: string, fallbackOpen: boolean) {
  try {
    if (href.startsWith("data:")) {
      const res = await fetch(href);
      const blob = await res.blob();
      triggerBlobDownload(blob, filename);
      return;
    }
    const r = await fetch(href, { mode: "cors" });
    if (!r.ok) throw new Error(String(r.status));
    const blob = await r.blob();
    triggerBlobDownload(blob, filename);
  } catch {
    if (fallbackOpen && href.startsWith("http")) window.open(href, "_blank", "noopener,noreferrer");
  }
}
