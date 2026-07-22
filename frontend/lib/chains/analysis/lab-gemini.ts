import { getCfEnvVal } from "@/lib/lab-config";
import { getNvidiaApiBase, getNvidiaVideoModel } from "@/lib/gemini-proxy";
import { isVideoFile, normalizeVideoMimeForUpload } from "@/lib/upload-video";
import { LAB_PROMPT } from "./lab-prompts";

function uint8ToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 8192) {
    const end = Math.min(i + 8192, bytes.length);
    for (let j = i; j < end; j++) binary += String.fromCharCode(bytes[j]);
  }
  return btoa(binary);
}

function stripThinkingBlocks(text: string): string {
  return text
    .replace(/<redacted_thinking>[\s\S]*?<\/redacted_thinking>/g, "")
    .replace(/<redacted_thinking>[\s\S]*?<\/think>/g, "")
    .replace(/<think>[\s\S]*?<\/think>/g, "")
    .trim();
}

/** Optional Edge NVIDIA path. Main Shot Lab flow uses Modal `/analyze/vision-lab`. */
export async function labNvidiaAnalysis(file: File, apiKey: string): Promise<Record<string, unknown>> {
  const model = getNvidiaVideoModel(getCfEnvVal);
  const base = getNvidiaApiBase(getCfEnvVal);
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const base64 = uint8ToBase64(bytes);
  const rawType = file.type || "";
  const video = isVideoFile(file, file.name || "");
  const mimeType = video
    ? normalizeVideoMimeForUpload(rawType, file.name || "video.mp4")
    : rawType || "image/jpeg";
  const mediaPart = video
    ? { type: "video_url", video_url: { url: `data:${mimeType};base64,${base64}` } }
    : { type: "image_url", image_url: { url: `data:${mimeType};base64,${base64}` } };

  const res = await fetch(`${base}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: [{ type: "text", text: LAB_PROMPT }, mediaPart] }],
      temperature: 0.3,
      max_tokens: 8192,
      stream: false,
      include_reasoning: false,
      chat_template_kwargs: { enable_thinking: false },
    }),
    signal: AbortSignal.timeout(120_000),
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`NVIDIA 分析错误 [${res.status}]: ${body.substring(0, 200)}`);
  }

  const data = await res.json();
  const raw: string = data.choices?.[0]?.message?.content || "";
  const text = stripThinkingBlocks(raw);
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) throw new Error(`NVIDIA 返回了非JSON响应: ${text.substring(0, 150)}`);
  return JSON.parse(match[0]);
}
