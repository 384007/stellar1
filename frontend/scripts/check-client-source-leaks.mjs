#!/usr/bin/env node
/**
 * Fail if client-reachable source files contain forbidden upload / public-env patterns.
 * Run from ``frontend/`` (e.g. after lint or in ``npm run build``).
 */

import fs from "node:fs";
import path from "node:path";

const cwd = process.cwd();

const FORBIDDEN = [
  "fetch(presignBody.upload_url",
  "NEXT_PUBLIC_BACKEND_URL",
  "NEXT_PUBLIC_STELLAR_PROV3_R2_PUBLIC_BASE",
  "r2.cloudflarestorage.com",
  "cdnFullUrl(",
  "?f=1&z=",
  "decodeBase64Url(",
  "allowAbsoluteUrl(",
];

const ROOTS = ["app", "lib", "components"];

function shouldScanFile(absPath, relPath) {
  if (!/\.(tsx|ts|jsx|js)$/.test(relPath)) return false;
  if (relPath.includes(`${path.sep}api${path.sep}`)) return false;
  if (relPath.startsWith(`lib${path.sep}server${path.sep}`)) return false;
  return true;
}

function walkFiles(rootDir, baseRel, out) {
  if (!fs.existsSync(rootDir)) return;
  for (const name of fs.readdirSync(rootDir)) {
    const abs = path.join(rootDir, name);
    const rel = path.join(baseRel, name);
    const st = fs.statSync(abs);
    if (st.isDirectory()) walkFiles(abs, rel, out);
    else if (st.isFile() && shouldScanFile(abs, rel)) out.push({ abs, rel });
  }
}

function main() {
  /** @type {{ abs: string; rel: string }[]} */
  const files = [];
  for (const root of ROOTS) {
    walkFiles(path.join(cwd, root), root, files);
  }

  /** @type {{ rel: string; needle: string }[]} */
  const hits = [];
  for (const { abs, rel } of files) {
    let raw;
    try {
      raw = fs.readFileSync(abs, "utf8");
    } catch {
      continue;
    }
    for (const needle of FORBIDDEN) {
      if (raw.includes(needle)) hits.push({ rel, needle });
    }
  }

  if (hits.length) {
    console.error("check-client-source-leaks: forbidden substring(s) in client-visible source:");
    const seen = new Set();
    for (const h of hits) {
      const key = `${h.needle}\0${h.rel}`;
      if (seen.has(key)) continue;
      seen.add(key);
      console.error(`  "${h.needle}" → ${h.rel}`);
    }
    process.exit(1);
  }

  console.log("check-client-source-leaks: ok (", files.length, "files scanned )");
}

main();
