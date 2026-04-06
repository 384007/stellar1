/**
 * GET /api/prov3/analyze/job/[jobId] — poll Pro v3 async job status + result from R2 (same-origin).
 */

import { NextRequest, NextResponse } from "next/server";
import { getRequestContext } from "@cloudflare/next-on-pages";
import { requireProUserForProv3Edge } from "@/lib/prov3-edge-route-auth";

export const runtime = "edge";

const JOB_PREFIX = "prov3-async-jobs";

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
  ctx: { params: Promise<{ jobId: string }> },
) {
  const auth = await requireProUserForProv3Edge(request);
  if (!auth.ok) return auth.response;

  const { jobId: rawId } = await ctx.params;
  const jobId = String(rawId || "").trim();
  if (!/^[a-zA-Z0-9_-]{8,80}$/.test(jobId)) {
    return NextResponse.json({ detail: "Invalid job id" }, { status: 400 });
  }

  const r2 = getR2();
  if (!r2) {
    return NextResponse.json({ detail: "存储不可用" }, { status: 503 });
  }

  const statusKey = `${JOB_PREFIX}/${jobId}/status.json`;
  let statusObj: Record<string, unknown>;
  try {
    const obj = await r2.get(statusKey);
    if (!obj) {
      return NextResponse.json(
        { status: "unknown", detail: "Job not found or not yet created" },
        { status: 404 },
      );
    }
    const text = await obj.text();
    statusObj = JSON.parse(text) as Record<string, unknown>;
  } catch {
    return NextResponse.json({ detail: "Could not read job status" }, { status: 500 });
  }

  const uid = String(statusObj.user_id || "");
  if (uid !== auth.userId) {
    return NextResponse.json({ detail: "Forbidden" }, { status: 403 });
  }

  const st = String(statusObj.status || "");
  if (st === "completed") {
    const resultKey = `${JOB_PREFIX}/${jobId}/result.json`;
    try {
      const robj = await r2.get(resultKey);
      if (robj) {
        const rt = await robj.text();
        const parsed = JSON.parse(rt) as { result?: unknown };
        return NextResponse.json({
          status: "completed",
          job_id: jobId,
          result: parsed.result ?? null,
        });
      }
    } catch {
      /* fall through */
    }
    return NextResponse.json({
      status: "completed",
      job_id: jobId,
      result: null,
      detail: "Result payload missing in storage",
    });
  }

  if (st === "failed") {
    return NextResponse.json({
      status: "failed",
      job_id: jobId,
      detail: typeof statusObj.detail === "string" ? statusObj.detail : "Analyze failed",
    });
  }

  return NextResponse.json({
    status: st || "pending",
    job_id: jobId,
  });
}
