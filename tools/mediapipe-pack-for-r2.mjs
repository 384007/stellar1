#!/usr/bin/env node
/**
 * Pack MediaPipe Tasks Vision (bundle + wasm + pose models) for upload to R2.
 *
 * Usage (from repo root):
 *   cd frontend && npm install
 *   node ../tools/mediapipe-pack-for-r2.mjs
 *
 * Output: frontend/build/mediapipe-r2/<version>/
 * Upload that folder's *contents* to R2 under your chosen prefix, e.g.:
 *   static/mediapipe/tasks-vision/0.10.33/
 *
 * Then set NEXT_PUBLIC_MEDIAPIPE_CDN_BASE to the public URL of that prefix
 * (no trailing slash), rebuild Pages.
 */

import fs from "node:fs";
import path from "node:path";
import https from "node:https";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.join(__dirname, "..");
const FRONTEND = path.join(REPO_ROOT, "frontend");
const PKG = path.join(FRONTEND, "node_modules", "@mediapipe", "tasks-vision");
const OUT_ROOT = path.join(FRONTEND, "build", "mediapipe-r2");

function readVersion() {
  const j = JSON.parse(fs.readFileSync(path.join(PKG, "package.json"), "utf8"));
  return j.version;
}

function download(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https
      .get(url, (res) => {
        if (res.statusCode === 301 || res.statusCode === 302) {
          const loc = res.headers.location;
          if (!loc) return reject(new Error("redirect without location"));
          file.close();
          fs.unlinkSync(dest);
          return download(loc, dest).then(resolve).catch(reject);
        }
        if (res.statusCode !== 200) {
          file.close();
          fs.unlinkSync(dest);
          return reject(new Error(`GET ${url} → ${res.statusCode}`));
        }
        res.pipe(file);
        file.on("finish", () => {
          file.close(resolve);
        });
      })
      .on("error", (err) => {
        file.close();
        fs.unlink(dest, () => reject(err));
      });
  });
}

async function main() {
  if (!fs.existsSync(PKG)) {
    console.error("Missing package. Run: cd frontend && npm install");
    process.exit(1);
  }

  const version = readVersion();
  const outDir = path.join(OUT_ROOT, version);
  const wasmOut = path.join(outDir, "wasm");
  const modelsOut = path.join(outDir, "models");

  fs.rmSync(outDir, { recursive: true, force: true });
  fs.mkdirSync(wasmOut, { recursive: true });
  fs.mkdirSync(modelsOut, { recursive: true });

  fs.copyFileSync(path.join(PKG, "vision_bundle.mjs"), path.join(outDir, "vision_bundle.mjs"));
  fs.cpSync(path.join(PKG, "wasm"), wasmOut, { recursive: true });

  const models = [
    {
      name: "pose_landmarker_full.task",
      url: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    },
    {
      name: "pose_landmarker_lite.task",
      url: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    },
  ];

  console.log(`Packing MediaPipe tasks-vision@${version} → ${outDir}\n`);

  for (const m of models) {
    const dest = path.join(modelsOut, m.name);
    process.stdout.write(`Downloading ${m.name} … `);
    await download(m.url, dest);
    const mb = (fs.statSync(dest).size / (1024 * 1024)).toFixed(1);
    console.log(`${mb} MB`);
  }

  const du = (dir) => {
    let n = 0;
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      const p = path.join(dir, e.name);
      if (e.isDirectory()) n += du(p);
      else n += fs.statSync(p).size;
    }
    return n;
  };
  const totalMb = (du(outDir) / (1024 * 1024)).toFixed(1);

  console.log(`
Done. Total ~${totalMb} MB.

Next:
1. Upload everything under:
     ${outDir}/
   to R2, preserving structure, e.g. key prefix:
     static/mediapipe/tasks-vision/${version}/

2. Enable public access (R2.dev or custom domain) and CORS GET for your site origin.

3. Cloudflare Pages → Environment variables:
     NEXT_PUBLIC_MEDIAPIPE_CDN_BASE=https://<your-public-host>/static/mediapipe/tasks-vision/${version}

4. Redeploy the frontend.

See docs/mediapipe-r2-selfhost.md for wrangler / dashboard steps.
`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
