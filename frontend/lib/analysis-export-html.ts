/**
 * 导出可在手机/电脑浏览器中直接打开的离线 HTML 报告（单文件，内嵌样式与关键帧图）。
 * 视频为单独文件；HTML 内使用相对路径引用，需与视频保存在同一文件夹。
 */

import { rawBase64ImagePayload } from "@/lib/image-base64";

export type ExportRecordMeta = {
  id: string;
  type: string;
  created_at: string;
  total_score: number | null;
};

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function keyframeImgSrc(b64: string): string {
  const t = (b64 || "").trim();
  if (!t) return "";
  if (t.startsWith("data:image/")) return t;
  const raw = rawBase64ImagePayload(t);
  if (raw.length < 40) return "";
  return `data:image/jpeg;base64,${raw}`;
}

function asString(v: unknown): string {
  return typeof v === "string" ? v : "";
}

function asStringArray(v: unknown): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];
}

function asScoreRecord(v: unknown): Record<string, number> {
  if (!v || typeof v !== "object" || Array.isArray(v)) return {};
  const o = v as Record<string, unknown>;
  const out: Record<string, number> = {};
  for (const [k, val] of Object.entries(o)) {
    if (typeof val === "number" && Number.isFinite(val)) out[k] = val;
  }
  return out;
}

export function downloadHtmlFile(filename: string, html: string) {
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

export function buildStellarLabStubHtml(
  rec: ExportRecordMeta,
  lang: "zh" | "en",
  exportedAt: string,
): string {
  const zh = lang === "zh";
  const title = zh ? "Shot Lab 记录" : "Shot Lab record";
  const body = zh
    ? "完整报告请在 Stellar App 的 Shot Lab 页面查看。"
    : "Open the Shot Lab screen in Stellar for the full report.";
  return `<!DOCTYPE html>
<html lang="${zh ? "zh-CN" : "en"}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(title)}</title>
<style>
  body{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;padding:20px;background:#0f1115;color:#e8eaef;line-height:1.6;max-width:640px;margin-left:auto;margin-right:auto;}
  h1{font-size:1.25rem;margin:0 0 12px;font-weight:700;}
  .meta{font-size:13px;color:#8b919a;margin-bottom:20px;}
  .card{background:#1a1d24;border:1px solid #2a2f3a;border-radius:12px;padding:16px;}
</style>
</head>
<body>
  <h1>${escapeHtml(title)}</h1>
  <p class="meta">ID ${escapeHtml(rec.id)} · ${escapeHtml(rec.created_at)}</p>
  <div class="card"><p>${escapeHtml(body)}</p></div>
  <p class="meta" style="margin-top:24px">${zh ? "导出时间" : "Exported"}: ${escapeHtml(exportedAt)}</p>
</body>
</html>`;
}

export function buildStellarAnalysisHtml(opts: {
  record: ExportRecordMeta;
  result: Record<string, unknown>;
  lang: "zh" | "en";
  exportedAt: string;
  /** 与同目录下视频文件名一致，例如 stellar-video-xxx.mp4；无视频则为 null */
  videoFileName: string | null;
}): string {
  const { record, result, lang, exportedAt, videoFileName } = opts;
  const zh = lang === "zh";
  const summary = zh
    ? asString(result.summary_zh) || asString(result.summary)
    : asString(result.summary) || asString(result.summary_zh);
  const issues = zh ? asStringArray(result.issues_zh) : asStringArray(result.issues);
  const suggestions = zh ? asStringArray(result.suggestions_zh) : asStringArray(result.suggestions);
  const scores = asScoreRecord(result.scores);
  const scoreKeys = Object.keys(scores);
  const keyframes = Array.isArray(result.keyframes) ? result.keyframes : [];

  const scoreBars =
    scoreKeys.length === 0
      ? `<p class="muted">${zh ? "无分项得分" : "No sub-scores"}</p>`
      : scoreKeys
          .map((k) => {
            const v = Math.max(0, Math.min(100, scores[k] ?? 0));
            return `<div class="score-row"><span class="score-name">${escapeHtml(k)}</span><div class="bar"><span style="width:${v}%"></span></div><span class="score-val">${v}</span></div>`;
          })
          .join("");

  const issuesBlock =
    issues.length === 0
      ? ""
      : `<h2>${zh ? "问题" : "Issues"}</h2><ul>${issues.map((t) => `<li>${escapeHtml(t)}</li>`).join("")}</ul>`;

  const suggBlock =
    suggestions.length === 0
      ? ""
      : `<h2>${zh ? "建议" : "Suggestions"}</h2><ul>${suggestions.map((t) => `<li>${escapeHtml(t)}</li>`).join("")}</ul>`;

  let kfHtml = "";
  for (const kf of keyframes) {
    if (!kf || typeof kf !== "object" || Array.isArray(kf)) continue;
    const row = kf as Record<string, unknown>;
    const phase = asString(row.phase);
    const label = zh
      ? asString(row.label_zh) || asString(row.label_en) || phase
      : asString(row.label_en) || asString(row.label_zh) || phase;
    const b64 = asString(row.image_base64);
    const src = keyframeImgSrc(b64);
    if (src) {
      kfHtml += `<figure class="kf"><img src="${src}" alt="${escapeHtml(label)}" loading="lazy"/><figcaption>${escapeHtml(label)}</figcaption></figure>`;
    } else {
      kfHtml += `<figure class="kf kf-empty"><figcaption>${escapeHtml(label || phase || (zh ? "关键帧" : "Keyframe"))}</figcaption><p class="muted">${zh ? "（无预览图）" : "(No image)"}</p></figure>`;
    }
  }
  if (!kfHtml) {
    kfHtml = `<p class="muted">${zh ? "本条记录未包含关键帧预览。" : "No keyframe images in this export."}</p>`;
  }

  const pred = result.prediction;
  let predHtml = "";
  if (pred && typeof pred === "object" && !Array.isArray(pred)) {
    const p = pred as Record<string, unknown>;
    const rows: [string, string][] = [];
    const add = (label: string, val: unknown) => {
      if (val === undefined || val === null || val === "") return;
      if (typeof val === "number" && !Number.isFinite(val)) return;
      rows.push([label, String(val)]);
    };
    if (zh) {
      add("预测距离 (m)", p.predicted_distance);
      add("球路", p.shot_shape_zh || p.shot_shape);
      add("杆头速度", p.club_head_speed);
      add("球速", p.ball_speed);
    } else {
      add("Predicted distance (m)", p.predicted_distance);
      add("Shot shape", p.shot_shape || p.shot_shape_zh);
      add("Club head speed", p.club_head_speed);
      add("Ball speed", p.ball_speed);
    }
    if (rows.length > 0) {
      predHtml = `<h2>${zh ? "预测" : "Prediction"}</h2><table class="tbl">${rows.map(([a, b]) => `<tr><th>${escapeHtml(a)}</th><td>${escapeHtml(b)}</td></tr>`).join("")}</table>`;
    }
  }

  const training = result.training_plan;
  let trainHtml = "";
  if (training && typeof training === "object" && !Array.isArray(training)) {
    const entries = Object.entries(training as Record<string, { focus?: string; drills?: string[]; duration?: string }>);
    if (entries.length > 0) {
      trainHtml = `<h2>${zh ? "训练计划" : "Training plan"}</h2>`;
      for (const [key, block] of entries) {
        if (!block || typeof block !== "object") continue;
        const drills = Array.isArray(block.drills) ? block.drills.filter((d): d is string => typeof d === "string") : [];
        trainHtml += `<div class="train-block"><h3>${escapeHtml(key)}</h3>`;
        if (block.focus) trainHtml += `<p>${escapeHtml(block.focus)}</p>`;
        if (block.duration) trainHtml += `<p class="muted">${escapeHtml(block.duration)}</p>`;
        if (drills.length) trainHtml += `<ul>${drills.map((d) => `<li>${escapeHtml(d)}</li>`).join("")}</ul>`;
        trainHtml += `</div>`;
      }
    }
  }

  const contactUrl = asString(result.contact_sheet_url);
  const contactBlock =
    contactUrl && /^https?:\/\//i.test(contactUrl)
      ? `<p><a href="${escapeHtml(contactUrl)}" target="_blank" rel="noopener">${zh ? "联系表 / 拼图（在线）" : "Contact sheet (online)"}</a></p>`
      : "";

  const videoSection =
    videoFileName
      ? `<h2>${zh ? "挥杆视频" : "Swing video"}</h2>
         <p class="hint">${zh ? "请将导出的视频文件与本页面保存在同一文件夹，文件名须为：" : "Save the downloaded video in the same folder as this file. Filename must be:"} <code>${escapeHtml(videoFileName)}</code></p>
         <video controls playsinline preload="metadata" src="./${escapeHtml(videoFileName)}"></video>`
      : `<h2>${zh ? "挥杆视频" : "Swing video"}</h2><p class="muted">${zh ? "本次导出未包含本地视频文件。若曾缓存视频，请先在本机历史页加载后再导出。" : "No local video file in this export."}</p>`;

  const typeLabel = (record.type || "lite").toUpperCase();
  const total =
    record.total_score != null && Number.isFinite(Number(record.total_score))
      ? String(record.total_score)
      : "—";

  return `<!DOCTYPE html>
<html lang="${zh ? "zh-CN" : "en"}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stellar — ${escapeHtml(typeLabel)} ${total === "—" ? "" : escapeHtml(total + (zh ? " 分" : " pts"))}</title>
<style>
  *{box-sizing:border-box;}
  body{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;margin:0;padding:16px;background:#0c0e12;color:#eceff4;line-height:1.55;max-width:720px;margin-left:auto;margin-right:auto;padding-bottom:48px;}
  h1{font-size:1.35rem;margin:0 0 8px;font-weight:700;}
  h2{font-size:1.05rem;margin:28px 0 12px;color:#c4c9d4;font-weight:600;border-bottom:1px solid #2a3040;padding-bottom:8px;}
  h3{font-size:0.95rem;margin:12px 0 6px;color:#aeb4bf;}
  .badge{display:inline-block;padding:3px 10px;border-radius:6px;font-size:11px;font-weight:700;letter-spacing:0.04em;}
  .badge-pro{background:rgba(212,175,55,0.15);color:#e6c564;}
  .badge-plus{background:rgba(168,85,247,0.15);color:#c4a3f5;}
  .badge-lite{background:rgba(124,58,237,0.15);color:#b49cfc;}
  .meta{font-size:13px;color:#8b929e;margin:12px 0 20px;}
  .summary{font-size:15px;color:#d8dde6;margin-bottom:8px;white-space:pre-wrap;}
  .muted{color:#6b7280;font-size:14px;}
  .score-row{display:flex;align-items:center;gap:10px;margin:8px 0;font-size:13px;}
  .score-name{flex:0 0 100px;color:#9ca3af;text-transform:capitalize;}
  .bar{flex:1;height:8px;background:#1e2430;border-radius:4px;overflow:hidden;}
  .bar span{display:block;height:100%;background:linear-gradient(90deg,#7c3aed,#a78bfa);border-radius:4px;}
  .score-val{flex:0 0 28px;text-align:right;color:#cbd5e1;}
  ul{padding-left:1.2rem;margin:8px 0;}
  li{margin:6px 0;}
  .kf-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:14px;margin-top:12px;}
  .kf{margin:0;background:#141820;border:1px solid #252b38;border-radius:10px;overflow:hidden;}
  .kf img{width:100%;height:auto;display:block;vertical-align:middle;background:#0a0c10;}
  .kf figcaption{font-size:12px;padding:8px;color:#9ca3af;text-align:center;}
  .kf-empty{padding:12px;}
  .tbl{width:100%;border-collapse:collapse;font-size:14px;margin-top:8px;}
  .tbl th{text-align:left;padding:8px 10px;color:#9ca3af;font-weight:500;border-bottom:1px solid #252b38;width:42%;}
  .tbl td{padding:8px 10px;border-bottom:1px solid #1e2430;}
  .train-block{background:#141820;border:1px solid #252b38;border-radius:10px;padding:12px;margin-bottom:12px;}
  video{width:100%;max-height:70vh;border-radius:10px;background:#000;margin-top:10px;}
  .hint{font-size:13px;color:#8b929e;margin:8px 0 12px;}
  code{font-size:12px;background:#1e2430;padding:2px 6px;border-radius:4px;word-break:break-all;}
  a{color:#93c5fd;}
  footer{margin-top:36px;padding-top:16px;border-top:1px solid #252b38;font-size:12px;color:#6b7280;}
</style>
</head>
<body>
  <span class="badge badge-${record.type === "pro" ? "pro" : record.type === "plus" ? "plus" : "lite"}">${escapeHtml(typeLabel)}</span>
  <h1>${zh ? "挥杆分析报告" : "Swing analysis report"}</h1>
  <p class="meta">${zh ? "分析时间" : "Analyzed"}: ${escapeHtml(record.created_at)} · ${zh ? "综合得分" : "Total"}: <strong>${escapeHtml(total)}</strong>${zh && total !== "—" ? " 分" : ""}<br/>ID: ${escapeHtml(record.id)}</p>
  ${summary ? `<h2>${zh ? "总结" : "Summary"}</h2><p class="summary">${escapeHtml(summary)}</p>` : ""}
  <h2>${zh ? "分项得分" : "Scores"}</h2>
  ${scoreBars}
  ${issuesBlock}
  ${suggBlock}
  ${predHtml}
  ${trainHtml}
  <h2>${zh ? "关键帧" : "Keyframes"}</h2>
  <div class="kf-grid">${kfHtml}</div>
  ${contactBlock}
  ${videoSection}
  <footer>Stellar · ${zh ? "导出时间" : "Exported"} ${escapeHtml(exportedAt)}<br/>${zh ? "用浏览器打开本文件即可查看，无需联网（视频除外）。" : "Open this file in any browser. Video plays offline if the file is beside this page."}</footer>
</body>
</html>`;
}
