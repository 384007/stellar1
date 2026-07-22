#!/usr/bin/env node
/**
 * Post-build guard: fail if known backend / infra strings appear in browser chunks.
 * Run from ``frontend/`` after ``next build`` (see package.json ``build`` script).
 */

import fs from "node:fs";
import path from "node:path";

const cwd = process.cwd();
const staticDir = path.join(cwd, ".next", "static");

const FORBIDDEN = [
  "dytsui--stellar-ai-fastapi-app.modal.run",
  "onrender.com",
  "r2.cloudflarestorage.com",
  "cloudflarestorage.com",
  "generativelanguage.googleapis.com",
  "NEXT_PUBLIC_BACKEND_URL",
  "NEXT_PUBLIC_MODAL_BACKEND_URL",
  "NEXT_PUBLIC_LITE_BACKEND_URL",
  "NEXT_PUBLIC_MEDIAPIPE_CDN_BASE",
  "NEXT_PUBLIC_MEDIAPIPE_ALLOW_FOREIGN_FALLBACK",
  "NEXT_PUBLIC_STELLAR_PROV3_R2_PUBLIC_BASE",
  "upload_url",
  "/api/cdn/p?f=1&z=",
  "f=1&z=",
];

const EXT = /\.(js|mjs|css|json|html|txt|map)$/i;

function walkFiles(dir, out = []) {
  if (!fs.existsSync(dir)) return out;
  for (const name of fs.readdirSync(dir)) {
    const p = path.join(dir, name);
    const st = fs.statSync(p);
    if (st.isDirectory()) walkFiles(p, out);
    else if (st.isFile() && EXT.test(name)) out.push(p);
  }
  return out;
}

function main() {
  const files = walkFiles(staticDir);
  if (files.length === 0) {
    console.error("check-no-client-leaks: no files under .next/static — run next build first");
    process.exit(1);
  }

  /** @type {{ file: string; needle: string }[]} */
  const hits = [];
  for (const file of files) {
    let raw;
    try {
      raw = fs.readFileSync(file, "utf8");
    } catch {
      continue;
    }
    for (const needle of FORBIDDEN) {
      if (raw.includes(needle)) hits.push({ file, needle });
    }
  }

  if (hits.length) {
    console.error("check-no-client-leaks: forbidden substring(s) in client static output:");
    const seen = new Set();
    for (const h of hits) {
      const key = `${h.needle}\0${h.file}`;
      if (seen.has(key)) continue;
      seen.add(key);
      console.error(`  "${h.needle}" → ${path.relative(cwd, h.file)}`);
    }
    process.exit(1);
  }

  console.log("check-no-client-leaks: ok (", files.length, "files scanned )");
}

main();
