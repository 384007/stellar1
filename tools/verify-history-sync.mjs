#!/usr/bin/env node
/**
 * Contract tests for frontend/lib/history-sync-record.ts (extractVideoFieldsFromSyncRecord).
 * Keep in sync when changing that module.
 */
import assert from "node:assert/strict";

function extractVideoFieldsFromSyncRecord(rec) {
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

assert.deepEqual(extractVideoFieldsFromSyncRecord({}), {
  video_url: "",
  video_r2_key: "",
});
assert.deepEqual(
  extractVideoFieldsFromSyncRecord({ video_r2_key: "videos/u/1.mp4" }),
  { video_url: "", video_r2_key: "videos/u/1.mp4" },
);
assert.deepEqual(extractVideoFieldsFromSyncRecord({ videoR2Key: "k" }), {
  video_url: "",
  video_r2_key: "k",
});
assert.deepEqual(
  extractVideoFieldsFromSyncRecord({ video_url: "blob:http://localhost/x" }),
  { video_url: "", video_r2_key: "" },
);
assert.deepEqual(
  extractVideoFieldsFromSyncRecord({ video_url: "https://cdn.example/v.mp4" }),
  { video_url: "https://cdn.example/v.mp4", video_r2_key: "" },
);
assert.deepEqual(
  extractVideoFieldsFromSyncRecord({
    video_r2_key: "a",
    videoR2Key: "b",
  }),
  { video_url: "", video_r2_key: "a" },
);

console.log("verify-history-sync: OK");
