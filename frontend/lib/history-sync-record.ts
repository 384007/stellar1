/**
 * History batch-sync: extract durable video fields from client localStorage records.
 * Rejects blob: URLs (only valid in the browser session that created them).
 */
export function extractVideoFieldsFromSyncRecord(
  rec: Record<string, unknown>,
): { video_url: string; video_r2_key: string } {
  const rawUrl =
    typeof rec.video_url === "string" ? rec.video_url.trim() : "";
  const video_url =
    rawUrl &&
    /^https?:\/\//i.test(rawUrl) &&
    !rawUrl.toLowerCase().startsWith("blob:")
      ? rawUrl
      : "";

  const snake =
    typeof rec.video_r2_key === "string" ? rec.video_r2_key.trim() : "";
  const camel =
    typeof rec.videoR2Key === "string" ? rec.videoR2Key.trim() : "";
  const video_r2_key = snake || camel;

  return { video_url, video_r2_key };
}

/** Persist R2 key on the local history row so batch sync can attach video if server POST failed. */
export function patchLocalHistoryVideoR2Key(
  analysisId: string,
  videoR2Key: string,
): void {
  if (!analysisId || !videoR2Key) return;
  try {
    const key = "stellar_history_local";
    const raw = localStorage.getItem(key);
    if (!raw) return;
    const rows = JSON.parse(raw) as Array<Record<string, unknown>>;
    const next = rows.map((r) =>
      r.id === analysisId ? { ...r, video_r2_key: videoR2Key } : r,
    );
    localStorage.setItem(key, JSON.stringify(next));
  } catch {
    /* quota / parse */
  }
}
