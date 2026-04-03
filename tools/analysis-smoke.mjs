#!/usr/bin/env node
/**
 * Stellar analysis smoke test — call from CI or locally against any deployed URL.
 *
 * Usage:
 *   node tools/analysis-smoke.mjs
 *   STELLAR_BASE_URL=https://your.pages.dev node tools/analysis-smoke.mjs
 *   STELLAR_TOKEN="eyJ..." node tools/analysis-smoke.mjs   # real JWT for protected routes
 *
 * Without STELLAR_TOKEN: uses local-e2e-smoke token (works when API allows local-* bypass).
 *
 * Exit code 0 = all attempted checks passed; 1 = failure.
 */

const BASE = (process.env.STELLAR_BASE_URL || "http://127.0.0.1:3000").replace(/\/$/, "");
const TOKEN = process.env.STELLAR_TOKEN || "local-e2e-smoke";
const SKIP_PLUS = process.env.STELLAR_SKIP_PLUS === "1";
const SKIP_LIVE_AI = process.env.STELLAR_SKIP_LIVE_AI === "1";

const JPEG_1X1 = Buffer.from(
  "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAA8A/9k=",
  "base64"
);

function log(msg) {
  console.log(`[smoke] ${msg}`);
}

async function main() {
  log(`BASE=${BASE}`);
  const failures = [];

  try {
    const r = await fetch("https://stellar1-backend.onrender.com/health");
    log(`Render /health → ${r.status}`);
  } catch (e) {
    log(`Render /health skip: ${e.message}`);
  }

  // Club detect (edge)
  if (!SKIP_LIVE_AI) {
    const fd = new FormData();
    fd.append("frame", new Blob([JPEG_1X1], { type: "image/jpeg" }), "frame.jpg");
    const { res, json } = await fetch(`${BASE}/api/club-detect`, {
      method: "POST",
      headers: { Authorization: `Bearer ${TOKEN}` },
      body: fd,
    }).then(async (r) => ({ res: r, json: await r.json().catch(() => ({})) }));

    if (res.status === 401 || res.status === 403) {
      log(`club-detect: ${res.status} (auth required on this env)`);
    } else if (!res.ok) {
      failures.push(`club-detect HTTP ${res.status}`);
    } else if (!json.club_type) {
      failures.push("club-detect: missing club_type");
    } else {
      log(`club-detect → ${json.club_type} (${json.confidence})`);
    }
  } else {
    log("club-detect skipped (STELLAR_SKIP_LIVE_AI=1)");
  }

  // Analyze Lite — tiny image (edge AI; needs keys on server)
  if (!SKIP_LIVE_AI) {
    const fd = new FormData();
    fd.append("file", new Blob([JPEG_1X1], { type: "image/jpeg" }), "smoke.jpg");
    const r = await fetch(`${BASE}/api/analyze`, {
      method: "POST",
      headers: { Authorization: `Bearer ${TOKEN}` },
      body: fd,
    });
    const j = await r.json().catch(() => ({}));
    const d = String(j.detail || "");
    if (
      r.status === 503 &&
      (d.includes("密钥") || /key|not configured|503/i.test(d))
    ) {
      log(`analyze lite: 503 (no AI keys on this env — not a failure for smoke)`);
    } else if (!r.ok) {
      failures.push(`analyze lite HTTP ${r.status}: ${(j.detail || "").slice(0, 120)}`);
    } else if (j.total_score == null && j.detail) {
      failures.push(`analyze lite: unexpected body`);
    } else {
      log(`analyze lite → ok (total_score=${j.total_score})`);
    }
  } else {
    log("analyze lite skipped (STELLAR_SKIP_LIVE_AI=1)");
  }

  // Plus — multipart (hits Modal/Render; optional)
  if (!SKIP_PLUS) {
    const fd = new FormData();
    fd.append("file", new Blob([JPEG_1X1], { type: "image/jpeg" }), "smoke.jpg");
    const r = await fetch(`${BASE}/api/plus`, {
      method: "POST",
      headers: { Authorization: `Bearer ${TOKEN}` },
      body: fd,
    });
    const j = await r.json().catch(() => ({}));
    if (r.status === 401 || r.status === 403) {
      log(`plus: ${r.status} (need real STELLAR_TOKEN for this env)`);
    } else if (r.status === 429) {
      log("plus: 429 daily limit — skip as failure");
    } else if (!r.ok) {
      failures.push(`plus HTTP ${r.status}: ${(j.detail || "").slice(0, 160)}`);
    } else {
      log(`plus → ok (analysis_id=${j.analysis_id || "?"})`);
    }
  } else {
    log("plus skipped (STELLAR_SKIP_PLUS=1)");
  }

  if (failures.length) {
    failures.forEach((f) => console.error(`[smoke] FAIL: ${f}`));
    process.exit(1);
  }
  log("all checks passed");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
