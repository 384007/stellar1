/**
 * AI API client helpers for server-side usage (Next.js API routes).
 * GEMINI_API_KEY must be set as a Cloudflare Pages Secret — never hardcode it here.
 */

const GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta";

export interface GeminiAnalysisResult {
  scores: {
    grip: number;
    stance: number;
    backswing: number;
    downswing: number;
    follow_through: number;
  };
  total_score: number;
  issues: string[];
  issues_zh: string[];
  suggestions: string[];
  suggestions_zh: string[];
  summary: string;
  summary_zh: string;
}

export async function callGeminiFlash(
  apiKey: string,
  prompt: string,
  images?: string[]
): Promise<string> {
  if (!apiKey) throw new Error("GEMINI_API_KEY not configured");

  const parts: Array<{ text: string } | { inlineData: { mimeType: string; data: string } }> = [
    { text: prompt },
  ];

  if (images) {
    for (const img of images) {
      parts.push({
        inlineData: {
          mimeType: "image/jpeg",
          data: img,
        },
      });
    }
  }

  const response = await fetch(
    `${GEMINI_BASE_URL}/models/gemini-2.5-flash-lite:generateContent?key=${apiKey}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts }],
        generationConfig: {
          temperature: 0.3,
          maxOutputTokens: 4096,
        },
      }),
    }
  );

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(
      `AI API error: ${response.status} - ${JSON.stringify(errorData)}`
    );
  }

  const data = await response.json();
  const text =
    data.candidates?.[0]?.content?.parts?.[0]?.text || "";
  return text;
}

export function parseGeminiAnalysis(text: string): GeminiAnalysisResult {
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    try {
      return JSON.parse(jsonMatch[0]);
    } catch {
      // fall through to default
    }
  }

  return {
    scores: {
      grip: 70,
      stance: 70,
      backswing: 70,
      downswing: 70,
      follow_through: 70,
    },
    total_score: 70,
    issues: ["Unable to parse AI response"],
    issues_zh: ["无法解析AI响应"],
    suggestions: ["Please try again"],
    suggestions_zh: ["请重试"],
    summary: "Analysis could not be completed.",
    summary_zh: "分析未能完成。",
  };
}

export function buildLitePrompt(poseData: Record<string, unknown>): string {
  return `You are a professional golf coach. Analyze the following golf swing data.

Skeleton angle data: ${JSON.stringify(poseData)}

Provide your analysis in the following JSON format ONLY (no markdown, no extra text):
{
  "scores": {"grip":0-100, "stance":0-100, "backswing":0-100, "downswing":0-100, "follow_through":0-100},
  "total_score": 0-100,
  "issues": ["issue 1", "issue 2", "issue 3"],
  "issues_zh": ["问题1", "问题2", "问题3"],
  "suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"],
  "suggestions_zh": ["建议1", "建议2", "建议3"],
  "summary": "English summary (200 words max)",
  "summary_zh": "中文总结（200字以内）"
}`;
}
