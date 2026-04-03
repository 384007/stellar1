/**
 * Lightweight regression check (no Jest/Vitest): PlusResultView shows Gemini observation
 * when frame_notes / observed_phase_keyframes exist, even if formal score is withheld.
 */
import assert from "assert";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const src = fs.readFileSync(path.join(__dirname, "..", "PlusResultView.tsx"), "utf8");

assert.ok(src.includes("gemObs.frame_notes"), "expect frame_notes in visibility");
assert.ok(src.includes("observed_phase_keyframes"), "expect observed_phase_keyframes in visibility");
assert.ok(src.includes("gemObsBulletsRaw.length"), "expect non-empty bullets check");
console.log("plus-gemini-visibility: ok");
