import { NextRequest, NextResponse } from "next/server";
import { isLabEnabled } from "@/lib/lab-config";
import { ensureLabSchema, getLabJob } from "@/lib/lab-db";
import { authenticateRequest, getDB, labError } from "@/lib/lab-auth";
import { getRequestContext } from "@cloudflare/next-on-pages";

export const runtime = "edge";

function getR2() {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (getRequestContext().env as any).R2_BUCKET || null;
  } catch {
    return null;
  }
}

export async function GET(
  request: NextRequest,
  ctx: { params: Promise<{ id: string }> }
) {
  if (!isLabEnabled()) {
    return labError("FEATURE_DISABLED", "Shot Lab 功能暂未开放", 404);
  }

  const authResult = await authenticateRequest(request);
  if (authResult instanceof NextResponse) return authResult;
  const { user_id } = authResult;

  const db = getDB();
  const r2 = getR2();
  if (!db || !r2) return labError("DB_UNAVAILABLE", "服务不可用", 503);

  try {
    await ensureLabSchema(db);
  } catch {
    /* non-fatal */
  }

  const { id: jobId } = await ctx.params;
  if (!jobId) return labError("BAD_REQUEST", "缺少任务 ID", 400);

  const job = await getLabJob(db, jobId);
  if (!job) return labError("NOT_FOUND", "任务不存在", 404);
  if ((job.user_id as string) !== user_id) {
    return labError("FORBIDDEN", "无权访问此任务", 403);
  }

  const videoKey = String((job.video_r2_key as string) || "").trim();
  if (!videoKey) {
    return NextResponse.json(
      {
        error: "NO_MEDIA",
        detail: "该记录未保存原始媒体，无法再次分析。请重新上传；新完成的 Shot Lab 会自动备份。",
        detail_en: "No backed-up media for this job. Upload again; new Shot Lab runs are stored for re-analysis.",
      },
      { status: 404 },
    );
  }

  const rangeHeader = request.headers.get("range");

  if (rangeHeader) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const head = await (r2 as any).head(videoKey);
    if (!head) return labError("NOT_FOUND", "文件不存在", 404);

    const totalSize: number = head.size ?? 0;
    const contentType: string = head.httpMetadata?.contentType || "video/mp4";

    const match = rangeHeader.match(/bytes=(\d+)-(\d*)/);
    if (match && totalSize > 0) {
      const start = parseInt(match[1], 10);
      const end = match[2] ? Math.min(parseInt(match[2], 10), totalSize - 1) : totalSize - 1;
      const chunkSize = end - start + 1;

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const obj = await (r2 as any).get(videoKey, { range: { offset: start, length: chunkSize } });
      if (!obj) return labError("NOT_FOUND", "文件不存在", 404);

      return new NextResponse(obj.body, {
        status: 206,
        headers: {
          "Content-Type": contentType,
          "Content-Range": `bytes ${start}-${end}/${totalSize}`,
          "Content-Length": String(chunkSize),
          "Accept-Ranges": "bytes",
          "Cache-Control": "private, max-age=3600",
        },
      });
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const obj = await (r2 as any).get(videoKey);
  if (!obj) return labError("NOT_FOUND", "文件不存在", 404);

  const totalSize: number = obj.size ?? 0;
  const headers: Record<string, string> = {
    "Content-Type": obj.httpMetadata?.contentType || "video/mp4",
    "Accept-Ranges": "bytes",
    "Cache-Control": "private, max-age=3600",
  };
  if (totalSize > 0) headers["Content-Length"] = String(totalSize);

  return new NextResponse(obj.body, { status: 200, headers });
}
