/**
 * GET /api/admin/finance — aggregate payment_orders for bookkeeping.
 * Header: X-Admin-Finance-Secret must match env ADMIN_FINANCE_SECRET (set in Cloudflare Secrets).
 */

import { NextRequest, NextResponse } from "next/server";
import { getRequestContext } from "@cloudflare/next-on-pages";
import { getCfEnvVal } from "@/lib/lab-config";
import { ensurePaymentSchema, financeSummary, listPaymentOrdersAdmin } from "@/lib/payment-db";
import { FINANCE_AI_USER_PROMPT_ZH } from "@/lib/finance-ai-prompt";

export const runtime = "edge";

function getDB() {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (getRequestContext().env as any).DB;
  } catch {
    return null;
  }
}

export async function GET(request: NextRequest) {
  const secret = getCfEnvVal("ADMIN_FINANCE_SECRET");
  if (!secret) {
    return NextResponse.json(
      { error: "DISABLED", detail: "未配置 ADMIN_FINANCE_SECRET，财务接口关闭" },
      { status: 503 }
    );
  }

  const provided = request.headers.get("x-admin-finance-secret") || "";
  if (provided !== secret) {
    return NextResponse.json({ error: "FORBIDDEN" }, { status: 403 });
  }

  const db = getDB();
  if (!db) {
    return NextResponse.json({ error: "DB_UNAVAILABLE" }, { status: 503 });
  }

  await ensurePaymentSchema(db);
  const summary = await financeSummary(db);
  const orders = await listPaymentOrdersAdmin(db, 200);

  const payload = {
    generated_at: new Date().toISOString(),
    summary,
    orders,
  };

  const narrative = request.nextUrl.searchParams.get("pack") === "1";
  if (narrative) {
    return NextResponse.json({
      ...payload,
      paste_for_ai_zh: `${FINANCE_AI_USER_PROMPT_ZH}\n${JSON.stringify(payload, null, 2)}`,
    });
  }

  return NextResponse.json(payload);
}
