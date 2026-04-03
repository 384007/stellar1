/**
 * Single-file offline HTML report: no zip, no external URLs required to view.
 * All assets are data: URLs from content already on the device.
 */

import { rawBase64ImagePayload } from "@/lib/image-base64";

export type OfflineHtmlLang = "zh" | "en";

export function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onloadend = () => resolve(String(r.result || ""));
    r.onerror = () => reject(r.error);
    r.readAsDataURL(blob);
  });
}

export function downloadOfflineHtml(filename: string, html: string) {
  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function esc(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function attrDataUrl(url: string): string {
  return url.replace(/&/g, "&amp;").replace(/"/g, "&quot;");
}

function pickStr(...vals: unknown[]): string {
  for (const v of vals) {
    if (typeof v === "string" && v.trim()) return v.trim();
  }
  return "";
}

function fmtScore(v: unknown): string {
  if (typeof v === "number" && Number.isFinite(v)) return String(Math.round(v * 10) / 10);
  return "—";
}

function keyframeImgSrc(b64: unknown): string | null {
  if (typeof b64 !== "string" || !b64.trim()) return null;
  const t = b64.trim();
  if (t.startsWith("data:image/")) return t;
  const raw = rawBase64ImagePayload(t);
  if (raw.length < 40) return null;
  return `data:image/jpeg;base64,${raw}`;
}

export function buildOfflineLabHtml(opts: {
  lang: OfflineHtmlLang;
  id: string;
  created_at: string;
}): string {
  const zh = opts.lang === "zh";
  const title = zh ? "Shot Lab 记录" : "Shot Lab record";
  const body = zh
    ? `<p>此条为 Shot Lab 会话记录。完整交互报告请在 App 内打开 <strong>Shot Lab</strong> 查看。</p><p>记录编号：<code>${esc(opts.id)}</code></p><p>时间：${esc(opts.created_at)}</p>`
    : `<p>This is a Shot Lab session. Open <strong>Shot Lab</strong> in the app for the full interactive report.</p><p>Id: <code>${esc(opts.id)}</code></p><p>Time: ${esc(opts.created_at)}</p>`;
  return `<!DOCTYPE html><html lang="${zh ? "zh-CN" : "en"}"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><title>${esc(title)}</title>${baseStyles()}</head><body><main class="wrap"><h1>${esc(title)}</h1><div class="card">${body}</div><p class="foot">${zh ? "本页可完全离线打开，不依赖网络。" : "This page works fully offline."}</p></main></body></html>`;
}

function baseStyles(): string {
  return `<style>
:root{--bg:#0f1115;--card:#1a1d24;--text:#e8eaed;--muted:#9aa0a6;--line:#2d323c;--accent:#c9a227;--ok:#34a853;}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;-webkit-font-smoothing:antialiased;}
.wrap{max-width:720px;margin:0 auto;padding:20px 16px 48px;}
h1{font-size:1.35rem;font-weight:700;margin:0 0 16px;letter-spacing:.02em;}
h2{font-size:1.05rem;font-weight:600;margin:24px 0 10px;color:var(--accent);border-bottom:1px solid var(--line);padding-bottom:6px;}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px;}
.meta{display:grid;grid-template-columns:1fr 1fr;gap:8px 16px;font-size:14px;color:var(--muted);}
.meta b{color:var(--text);}
ul,ol{margin:8px 0;padding-left:1.25rem;}
li{margin:6px 0;}
.scores{width:100%;border-collapse:collapse;font-size:14px;margin-top:8px;}
.scores th,.scores td{border:1px solid var(--line);padding:8px 10px;text-align:left;}
.scores th{background:#22262e;color:var(--muted);font-weight:600;}
.kf-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:12px;margin-top:12px;}
.kf-item{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#12141a;}
.kf-item img{display:block;width:100%;height:auto;vertical-align:middle;}
.kf-cap{padding:8px 10px;font-size:12px;color:var(--muted);text-align:center;}
video{max-width:100%;border-radius:10px;background:#000;}
.vid-wrap{margin-top:12px;}
.note{color:var(--muted);font-size:13px;margin-top:8px;}
.foot{margin-top:32px;font-size:12px;color:var(--muted);line-height:1.5;}
code{font-size:12px;background:#222;padding:2px 6px;border-radius:4px;word-break:break-all;}
</style>`;
}

export function buildOfflineAnalysisHtml(opts: {
  lang: OfflineHtmlLang;
  record: {
    id: string;
    type: string;
    created_at: string;
    total_score: number | null;
  };
  result: Record<string, unknown>;
  videoDataUrl: string | null;
}): string {
  const zh = opts.lang === "zh";
  const r = opts.result;
  const title = zh ? "挥杆分析报告" : "Swing analysis report";

  const summary = pickStr(r.summary_zh, r.summary, r.summary_en);
  const total =
    typeof r.total_score === "number" && Number.isFinite(r.total_score)
      ? r.total_score
      : opts.record.total_score;
  const scores = r.scores && typeof r.scores === "object" && !Array.isArray(r.scores)
    ? (r.scores as Record<string, unknown>)
    : {};

  const scoreRows = ["grip", "stance", "backswing", "downswing", "follow_through"].filter((k) => scores[k] != null);
  const scoreLabel: Record<string, { zh: string; en: string }> = {
    grip: { zh: "握杆", en: "Grip" },
    stance: { zh: "站姿", en: "Stance" },
    backswing: { zh: "上杆", en: "Backswing" },
    downswing: { zh: "下杆", en: "Downswing" },
    follow_through: { zh: "收杆", en: "Follow-through" },
  };

  const issues = Array.isArray(r.issues_zh) ? r.issues_zh : Array.isArray(r.issues) ? r.issues : [];
  const suggestions = Array.isArray(r.suggestions_zh)
    ? r.suggestions_zh
    : Array.isArray(r.suggestions)
      ? r.suggestions
      : [];

  const keyframes = Array.isArray(r.keyframes) ? r.keyframes : [];
  const kfBlocks: string[] = [];
  for (const kf of keyframes) {
    if (!kf || typeof kf !== "object") continue;
    const o = kf as Record<string, unknown>;
    const src = keyframeImgSrc(o.image_base64);
    if (!src) continue;
    const label = zh
      ? pickStr(o.label_zh, o.label_en, o.phase)
      : pickStr(o.label_en, o.label_zh, o.phase);
    kfBlocks.push(
      `<div class="kf-item"><img src="${attrDataUrl(src)}" alt="${esc(label)}"/><div class="kf-cap">${esc(label)}</div></div>`,
    );
  }

  const pred = r.prediction && typeof r.prediction === "object" && !Array.isArray(r.prediction)
    ? (r.prediction as Record<string, unknown>)
    : null;

  const extraNotes = [
    pickStr(r.quick_tip_zh, r.quick_tip_en),
    pickStr(r.problem_description_zh, r.problem_description_en),
    pickStr(r.quality_warning, r.keyframe_warning),
  ].filter(Boolean);

  const training =
    r.training_plan && typeof r.training_plan === "object"
      ? (r.training_plan as Record<string, { focus?: string; drills?: string[]; duration?: string }>)
      : r.training && typeof r.training === "object"
        ? (r.training as Record<string, { focus?: string; drills?: string[]; duration?: string }>)
        : null;

  let predHtml = "";
  if (pred) {
    const rows: string[] = [];
    const add = (k: string, labelZh: string, labelEn: string, v: unknown) => {
      if (v == null || v === "") return;
      rows.push(
        `<tr><td><b>${esc(zh ? labelZh : labelEn)}</b></td><td>${esc(String(v))}</td></tr>`,
      );
    };
    add("dist", "预测距离", "Predicted distance", pred.predicted_distance);
    add("shape", "球路", "Shot shape", pickStr(pred.shot_shape_zh, pred.shot_shape));
    add("chs", "杆头速度", "Club speed", pred.club_head_speed);
    add("bs", "球速", "Ball speed", pred.ball_speed);
    add("la", "起飞角", "Launch angle", pred.launch_angle);
    if (rows.length) {
      predHtml = `<h2>${zh ? "预测与数据" : "Prediction"}</h2><div class="card"><table class="scores"><tbody>${rows.join("")}</tbody></table></div>`;
    }
  }

  let trainingHtml = "";
  if (training && Object.keys(training).length) {
    const parts: string[] = [];
    for (const [phase, block] of Object.entries(training)) {
      if (!block || typeof block !== "object") continue;
      const focus = typeof block.focus === "string" ? block.focus : "";
      const dur = typeof block.duration === "string" ? block.duration : "";
      const drills = Array.isArray(block.drills) ? block.drills.filter((d) => typeof d === "string") : [];
      let inner = `<h3 style="margin:12px 0 6px;font-size:14px;color:var(--text)">${esc(phase)}</h3>`;
      if (focus) inner += `<p>${esc(focus)}</p>`;
      if (dur) inner += `<p class="note">${zh ? "时长：" : "Duration: "}${esc(dur)}</p>`;
      if (drills.length) inner += `<ul>${drills.map((d) => `<li>${esc(d)}</li>`).join("")}</ul>`;
      parts.push(inner);
    }
    if (parts.length) {
      trainingHtml = `<h2>${zh ? "训练计划" : "Training plan"}</h2><div class="card">${parts.join("")}</div>`;
    }
  }

  const videoSection =
    opts.videoDataUrl && opts.videoDataUrl.startsWith("data:")
      ? `<h2>${zh ? "挥杆视频" : "Swing video"}</h2><div class="card vid-wrap"><video controls playsinline src="${attrDataUrl(opts.videoDataUrl)}"></video></div>`
      : `<h2>${zh ? "挥杆视频" : "Swing video"}</h2><div class="card"><p class="note">${zh ? "本机未缓存该条原视频，报告中仅含下列图文与关键帧。若曾在本设备保存过视频，可先展开该记录加载后再导出。" : "Original video was not cached on this device. Only text and keyframes below are included. Open the record once to cache video, then export again."}</p></div>`;

  const scoreTable =
    scoreRows.length > 0
      ? `<table class="scores"><thead><tr><th>${zh ? "项目" : "Item"}</th><th>${zh ? "得分" : "Score"}</th></tr></thead><tbody>${scoreRows
          .map(
            (k) =>
              `<tr><td>${esc(zh ? scoreLabel[k]?.zh ?? k : scoreLabel[k]?.en ?? k)}</td><td>${esc(fmtScore(scores[k]))}</td></tr>`,
          )
          .join("")}</tbody></table>`
      : "";

  const issuesHtml =
    issues.length > 0
      ? `<h2>${zh ? "问题" : "Issues"}</h2><div class="card"><ul>${issues
          .filter((x): x is string => typeof x === "string")
          .map((x) => `<li>${esc(x)}</li>`)
          .join("")}</ul></div>`
      : "";

  const sugHtml =
    suggestions.length > 0
      ? `<h2>${zh ? "建议" : "Suggestions"}</h2><div class="card"><ul>${suggestions
          .filter((x): x is string => typeof x === "string")
          .map((x) => `<li>${esc(x)}</li>`)
          .join("")}</ul></div>`
      : "";

  const kfSection =
    kfBlocks.length > 0
      ? `<h2>${zh ? "动作关键帧" : "Keyframes"}</h2><div class="card"><div class="kf-grid">${kfBlocks.join("")}</div></div>`
      : "";

  const poseFrames = Array.isArray(r.pose_frames) ? r.pose_frames : [];
  const poseImgBlocks: string[] = [];
  const maxPose = 24;
  for (let i = 0; i < Math.min(poseFrames.length, maxPose); i++) {
    const pf = poseFrames[i];
    if (!pf || typeof pf !== "object") continue;
    const src = keyframeImgSrc((pf as Record<string, unknown>).image_base64);
    if (!src) continue;
    const fi = (pf as Record<string, unknown>).frame_index;
    const cap =
      typeof fi === "number"
        ? zh
          ? `帧 ${fi}`
          : `Frame ${fi}`
        : `#${i + 1}`;
    poseImgBlocks.push(
      `<div class="kf-item"><img src="${attrDataUrl(src)}" alt="${esc(cap)}"/><div class="kf-cap">${esc(cap)}</div></div>`,
    );
  }
  const poseSection =
    poseImgBlocks.length > 0
      ? `<h2>${zh ? "姿态采样（节选）" : "Pose samples"}</h2><div class="card"><p class="note" style="margin-top:0">${zh ? `共导出 ${poseImgBlocks.length} 张（最多 ${maxPose} 张），减小文件体积。` : `${poseImgBlocks.length} frames (max ${maxPose}) to limit file size.`}</p><div class="kf-grid">${poseImgBlocks.join("")}</div></div>`
      : "";

  const extraHtml =
    extraNotes.length > 0
      ? `<h2>${zh ? "备注" : "Notes"}</h2><div class="card">${extraNotes.map((t) => `<p>${esc(t)}</p>`).join("")}</div>`
      : "";

  const foot = zh
    ? "本文件已包含当前设备中可用的全部图文与视频（若有）。不依赖网络、不依赖云端存储即可用浏览器打开阅读。"
    : "This file contains all text, images, and video available on this device. Open in any browser — no network or cloud required.";

  return `<!DOCTYPE html><html lang="${zh ? "zh-CN" : "en"}"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><title>${esc(title)}</title>${baseStyles()}</head><body><main class="wrap"><h1>${esc(title)}</h1><div class="card"><div class="meta"><div><b>${zh ? "类型" : "Type"}</b> ${esc(String(opts.record.type || "—").toUpperCase())}</div><div><b>${zh ? "总分" : "Total"}</b> ${esc(fmtScore(total))}</div><div><b>${zh ? "时间" : "Time"}</b> ${esc(opts.record.created_at)}</div><div><b>ID</b> <code>${esc(opts.record.id)}</code></div></div>${summary ? `<p style="margin-top:14px;font-size:15px">${esc(summary)}</p>` : ""}${scoreTable}</div>${videoSection}${predHtml}${trainingHtml}${issuesHtml}${sugHtml}${kfSection}${poseSection}${extraHtml}<p class="foot">${esc(foot)}</p></main></body></html>`;
}
