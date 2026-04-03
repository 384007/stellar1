import { NextRequest, NextResponse } from "next/server";
import { getRequestContext } from "@cloudflare/next-on-pages";
import { getGeminiHosts, getGeminiKeys, shouldRetryNextGeminiKey } from "@/lib/gemini-proxy";

export const runtime = "edge";

const QWEN_URL =
  "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions";

function getCfEnv(key: string): string {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return ((getRequestContext().env as any)[key] as string) || "";
  } catch {
    return process.env[key] || "";
  }
}

const CLUB_DETECT_PROMPT = `You are an expert golf equipment and stance analyst.

FIRST: Determine if this image shows a person holding or swinging a golf club.
If NO golf club is visible, return: {"club_type": "UNKNOWN", "confidence": 0, "hand": "R"}
Only if you can clearly see a golf club, identify the club type and handedness.

首先判断图中是否有人持有或挥动高尔夫球杆。如果看不到球杆，直接返回 UNKNOWN。

Possible club types (球杆型号):
- UNKNOWN: no golf club visible (看不到球杆)
- WOOD (木杆): 1W, 3W, 5W
- IRON (铁杆): 3I, 4I, 5I, 6I, 7I, 8I, 9I
- WEDGE (挖起杆): PW, AW, SW, LW
- PUTTER (推杆): PT

Respond with ONLY this JSON (no markdown, no backticks):
{
  "club_type": "<UNKNOWN or one of: 1W, 3W, 5W, 3I, 4I, 5I, 6I, 7I, 8I, 9I, PW, AW, SW, LW, PT>",
  "confidence": <float 0.0 to 1.0>,
  "hand": "<R or L>"
}

Club identification tips / 球杆识别要点:
- Wood clubs have large, rounded heads (木杆杆头大而圆)
- Irons have thin, flat blade-like heads (铁杆杆头薄而平)
- Wedges look similar to short irons but with more loft (挖起杆类似短铁杆但角度更大)
- Putters have a flat face and are used on the green (推杆平面杆头，用于果岭)
- If the club head is not clearly visible but a person is swinging, estimate from shaft length
  (杆头不清晰但有人挥杆，可从杆身长度推断)
- If NO golf club or swing is visible at all, use "UNKNOWN" with confidence 0
  (如果完全看不到球杆或挥杆动作，使用 "UNKNOWN"，置信度 0)

Handedness tips / 左右手判断:
- R = right-handed (右手打球): player stands with left shoulder closer to target
- L = left-handed (左手打球): player stands with right shoulder closer to target
- Default to "R" if uncertain (无法判断时默认 "R")`;

const CLUB_GROUP_MAP: Record<string, string> = {
  "1W": "WOOD", "3W": "WOOD", "5W": "WOOD",
  "3I": "IRON", "4I": "IRON", "5I": "IRON", "6I": "IRON",
  "7I": "IRON", "8I": "IRON", "9I": "IRON",
  "PW": "WEDGE", "AW": "WEDGE", "SW": "WEDGE", "LW": "WEDGE",
  "PT": "PUTTER",
};

const FALLBACK = {
  club_type: "UNKNOWN",
  club_group: "IRON",
  confidence: 0.0,
  hand: "R" as const,
  ai_provider: "none" as const,
};

function uint8ToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 8192) {
    const end = Math.min(i + 8192, bytes.length);
    for (let j = i; j < end; j++) binary += String.fromCharCode(bytes[j]);
  }
  return btoa(binary);
}

function parseClubResponse(text: string) {
  const cleaned = text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
  const match = cleaned.match(/\{[\s\S]*?\}/);
  if (!match) return FALLBACK;

  try {
    const parsed = JSON.parse(match[0]);
    const clubType = String(parsed.club_type || "").toUpperCase().trim();
    const confidence = Math.max(0, Math.min(1, Number(parsed.confidence) || 0));
    const hand = String(parsed.hand || "R").toUpperCase().trim() === "L" ? "L" : "R";
    if (clubType === "UNKNOWN" || !CLUB_GROUP_MAP[clubType]) {
      return { ...FALLBACK, hand };
    }
    return {
      club_type: clubType,
      club_group: CLUB_GROUP_MAP[clubType],
      confidence: Math.round(confidence * 100) / 100,
      hand,
    };
  } catch { /* parse error */ }
  return FALLBACK;
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get("frame") as File | null;
    if (!file) return NextResponse.json(FALLBACK);

    const buffer = await file.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    const base64 = uint8ToBase64(bytes);

    // Gemini: host × key loop, then Qwen
    const geminiKeys = getGeminiKeys(getCfEnv);
    if (geminiKeys.length > 0) {
      const model = getCfEnv("GEMINI_MODEL") || "gemini-2.5-flash-lite";
      const country = (request.headers.get("cf-ipcountry") || "").toUpperCase();
      const hosts = getGeminiHosts(getCfEnv, country === "CN");
      const bodyJson = JSON.stringify({
        contents: [
          {
            parts: [
              { text: CLUB_DETECT_PROMPT },
              { inlineData: { mimeType: "image/jpeg", data: base64 } },
            ],
          },
        ],
        generationConfig: { temperature: 0.2, maxOutputTokens: 256 },
      });
      for (const host of hosts) {
        for (const key of geminiKeys) {
          try {
            const res = await fetch(
              `${host}/v1beta/models/${model}:generateContent?key=${key}`,
              {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: bodyJson,
                signal: AbortSignal.timeout(15_000),
              }
            );
            if (res.ok) {
              const data = await res.json();
              const text = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
              return NextResponse.json({ ...parseClubResponse(text), ai_provider: "gemini" as const });
            }
            if (shouldRetryNextGeminiKey(res.status)) continue;
            break; // other error → next host
          } catch { break; }
        }
      }
    }

    // Qwen fallback
    const qwenKey = getCfEnv("QWEN_API_KEY");
    if (qwenKey) {
      try {
        const res = await fetch(QWEN_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${qwenKey}`,
          },
          body: JSON.stringify({
            model: "qwen-vl-max-latest",
            messages: [
              {
                role: "user",
                content: [
                  { type: "text", text: CLUB_DETECT_PROMPT },
                  {
                    type: "image_url",
                    image_url: { url: `data:image/jpeg;base64,${base64}` },
                  },
                ],
              },
            ],
            temperature: 0.2,
            max_tokens: 256,
          }),
          signal: AbortSignal.timeout(15_000),
        });
        if (res.ok) {
          const data = await res.json();
          const text = data.choices?.[0]?.message?.content || "";
          return NextResponse.json({ ...parseClubResponse(text), ai_provider: "qwen" as const });
        }
      } catch { /* Qwen also failed */ }
    }

    return NextResponse.json(FALLBACK);
  } catch {
    return NextResponse.json(FALLBACK);
  }
}
