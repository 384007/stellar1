#!/usr/bin/env node
/**
 * Copy ALL MediaPipe Tasks Vision files from node_modules to public/mp/
 * so they ship as CF Pages static assets (same domain, same CDN, works in China).
 *
 * Copies from the already-installed npm package — no internet download needed.
 * Also downloads the lite pose model from Google Storage (build servers can reach it).
 *
 * Runs during `npm run build` / `npm run pages:build`.
 */
import { existsSync, mkdirSync, copyFileSync, writeFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PKG_DIR = join(__dirname, "..", "node_modules", "@mediapipe", "tasks-vision");
const DEST = join(__dirname, "..", "public", "mp");

const WASM_FILES = [
  "vision_wasm_internal.js",
  "vision_wasm_internal.wasm",
  "vision_wasm_module_internal.js",
  "vision_wasm_module_internal.wasm",
  "vision_wasm_nosimd_internal.js",
  "vision_wasm_nosimd_internal.wasm",
];

function copyPkg(src, destPath) {
  if (existsSync(destPath)) {
    console.log(`  ✓ ${destPath.split("/public/mp/")[1]} (exists)`);
    return;
  }
  if (!existsSync(src)) {
    console.warn(`  ✗ ${src} not found, skipping`);
    return;
  }
  copyFileSync(src, destPath);
  console.log(`  ✓ ${destPath.split("/public/mp/")[1]}`);
}

async function downloadModel(name, url) {
  const dest = join(DEST, "models", name);
  if (existsSync(dest)) {
    console.log(`  ✓ models/${name} (exists)`);
    return;
  }
  console.log(`  ↓ downloading models/${name} ...`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  const buf = Buffer.from(await res.arrayBuffer());
  writeFileSync(dest, buf);
  console.log(`  ✓ models/${name} (${(buf.length / 1e6).toFixed(1)} MB)`);
}

async function main() {
  console.log("[mediapipe-static] Copying MediaPipe assets to public/mp/");

  // Create directories
  mkdirSync(join(DEST, "wasm"), { recursive: true });
  mkdirSync(join(DEST, "models"), { recursive: true });

  // Copy vision_bundle.mjs
  copyPkg(join(PKG_DIR, "vision_bundle.mjs"), join(DEST, "vision_bundle.mjs"));

  // Copy WASM files
  for (const f of WASM_FILES) {
    copyPkg(join(PKG_DIR, "wasm", f), join(DEST, "wasm", f));
  }

  // Download model (build server can reach Google Storage)
  await downloadModel(
    "pose_landmarker_lite.task",
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
  );

  // Generate vision_bundle_global.js — a non-ESM version of vision_bundle.mjs
  // that assigns exports to window.__mediapipe_vision instead of using `export`.
  // This can be loaded via regular <script src="…"> (no eval, no CSP issue, PWA-safe).
  // Always regenerate (derived file — stale copy could have the wrong transform).
  const globalPath = join(DEST, "vision_bundle_global.js");
  {
    const { readFileSync } = await import("fs");
    let src = readFileSync(join(PKG_DIR, "vision_bundle.mjs"), "utf-8");
    src = src.replace(
      /export\{([^}]+)\};?[\s\S]*$/,
      (_match, exports) => {
        const pairs = exports.split(",").map((s) => s.trim()).filter(Boolean);
        const obj = pairs.map((p) => {
          const parts = p.split(/\s+as\s+/);
          return parts.length === 2 ? `${parts[1].trim()}:${parts[0].trim()}` : `${parts[0].trim()}:${parts[0].trim()}`;
        }).join(",");
        return `window.__mediapipe_vision={${obj}};`;
      }
    );
    if (!src.includes("window.__mediapipe_vision")) {
      console.error("  ✗ FATAL: export replacement failed — regex did not match");
      process.exit(1);
    }
    writeFileSync(globalPath, src);
    console.log("  ✓ vision_bundle_global.js (generated)");
  }

  console.log("[mediapipe-static] Done — all assets in public/mp/");
}

main().catch((e) => {
  console.error("[mediapipe-static] WARNING:", e.message);
  console.error("  Build continues — will try external URLs at runtime.");
  process.exit(0);
});
