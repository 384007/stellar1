import { getCfEnvVal } from "@/lib/lab-config";
import { rewriteGoogleUrl } from "@/lib/gemini-proxy";
import { isVideoFile, normalizeVideoMimeForUpload } from "@/lib/upload-video";
import { LAB_PROMPT } from "./lab-prompts";

const QWEN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions";
const QWEN_MODEL = "qwen-vl-max-latest";

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
    .trim();
}

/** Parse `[503]`-style status from `labGeminiAnalysis*` error messages. */
export function httpStatusFromBracketMessage(message: string): number {
  const m = message.match(/\[(\d{3})\]/);
  if (!m) return 0;
  const n = parseInt(m[1]!, 10);
  return Number.isFinite(n) ? n : 0;
}

export async function labGeminiAnalysis(file: File, host: string, apiKey: string): Promise<Record<string, unknown>> {
  const model = getCfEnvVal("GEMINI_MODEL") || "gemini-2.5-flash-lite";
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const rawType = file.type || "";
  const isVideo = isVideoFile(file, file.name || "");
  const mimeType = isVideo
    ? normalizeVideoMimeForUpload(rawType, file.name || "video.mp4")
    : rawType || "image/jpeg";

  let contentParts: unknown[];

  if (isVideo) {
    const initRes = await fetch(`${host}/upload/v1beta/files?key=${apiKey}`, {
      method: "POST",
      headers: {
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": String(bytes.length),
        "X-Goog-Upload-Header-Content-Type": mimeType,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ file: { displayName: file.name || "swing.mp4" } }),
      signal: AbortSignal.timeout(20_000),
    });
    if (!initRes.ok) {
      const errBody = await initRes.text().catch(() => "");
      throw new Error(`AI上传初始化失败 [${initRes.status}]: ${errBody.substring(0, 200)}`);
    }

    const rawUploadUri = initRes.headers.get("x-goog-upload-url");
    const uploadUri = rawUploadUri ? rewriteGoogleUrl(rawUploadUri, host) : null;
    if (!uploadUri) throw new Error("AI服务未返回上传URI");

    const uploadRes = await fetch(uploadUri, {
      method: "POST",
      headers: {
        "X-Goog-Upload-Command": "upload, finalize",
        "X-Goog-Upload-Offset": "0",
        "Content-Type": mimeType,
      },
      body: bytes,
      signal: AbortSignal.timeout(30_000),
    });
    if (!uploadRes.ok) {
      const errBody = await uploadRes.text().catch(() => "");
      throw new Error(`视频上传失败 [${uploadRes.status}]: ${errBody.substring(0, 200)}`);
    }

    const uploadData = await uploadRes.json();
    const fileUri = uploadData.file?.uri;
    const fileName = uploadData.file?.name;
    if (!fileUri) throw new Error("文件上传成功但未返回URI");

    if (fileName) {
      for (let i = 0; i < 20; i++) {
        const checkRes = await fetch(`${host}/v1beta/${fileName}?key=${apiKey}`);
        if (checkRes.ok) {
          const fileInfo = await checkRes.json();
          if (fileInfo.state === "ACTIVE") break;
          if (fileInfo.state === "FAILED") throw new Error("视频处理失败，请尝试压缩视频或缩短时长");
        }
        await new Promise((r) => setTimeout(r, 3000));
      }
    }

    contentParts = [{ text: LAB_PROMPT }, { fileData: { mimeType, fileUri } }];
  } else {
    const base64 = uint8ToBase64(bytes);
    contentParts = [{ text: LAB_PROMPT }, { inlineData: { mimeType, data: base64 } }];
  }

  const res = await fetch(`${host}/v1beta/models/${model}:generateContent?key=${apiKey}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ parts: contentParts }],
      generationConfig: { temperature: 0.3, maxOutputTokens: 8192 },
    }),
    signal: AbortSignal.timeout(60_000),
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`AI 分析错误 [${res.status}]: ${body.substring(0, 200)}`);
  }

  const data = await res.json();
  const text = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) throw new Error(`AI 返回了非JSON响应: ${text.substring(0, 150)}`);

  return JSON.parse(match[0]);
}

export async function labQwenAnalysis(file: File): Promise<Record<string, unknown>> {
  const apiKey = getCfEnvVal("QWEN_API_KEY");
  if (!apiKey) throw new Error("通义千问 API 密钥未配置 (QWEN_API_KEY)");

  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const base64 = uint8ToBase64(bytes);
  const rawType = file.type || "";
  const isVideo = isVideoFile(file, file.name || "");
  const mimeType = isVideo
    ? normalizeVideoMimeForUpload(rawType, file.name || "video.mp4")
    : rawType || "image/jpeg";

  const mediaPart = isVideo
    ? { type: "video_url", video_url: { url: `data:${mimeType};base64,${base64}` } }
    : { type: "image_url", image_url: { url: `data:${mimeType};base64,${base64}` } };

  const res = await fetch(QWEN_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: QWEN_MODEL,
      messages: [{ role: "user", content: [{ type: "text", text: LAB_PROMPT }, mediaPart] }],
      temperature: 0.3,
      max_tokens: 8192,
    }),
    signal: AbortSignal.timeout(60_000),
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Qwen 分析错误 [${res.status}]: ${body.substring(0, 200)}`);
  }

  const data = await res.json();
  const raw: string = data.choices?.[0]?.message?.content || "";
  const text = stripThinkingBlocks(raw);
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) throw new Error(`Qwen 返回了非JSON响应: ${text.substring(0, 150)}`);

  return JSON.parse(match[0]);
}

export async function labGeminiAnalysisWithUri(
  fileUri: string,
  mimeType: string,
  host: string,
  apiKey: string,
): Promise<Record<string, unknown>> {
  const model = getCfEnvVal("GEMINI_MODEL") || "gemini-2.5-flash-lite";

  const res = await fetch(`${host}/v1beta/models/${model}:generateContent?key=${apiKey}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ parts: [{ text: LAB_PROMPT }, { fileData: { mimeType, fileUri } }] }],
      generationConfig: { temperature: 0.3, maxOutputTokens: 8192 },
    }),
    signal: AbortSignal.timeout(60_000),
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`AI 分析错误 [${res.status}]: ${body.substring(0, 200)}`);
  }

  const data = await res.json();
  const text = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) throw new Error(`AI 返回了非JSON响应: ${text.substring(0, 150)}`);

  return JSON.parse(match[0]);
}
