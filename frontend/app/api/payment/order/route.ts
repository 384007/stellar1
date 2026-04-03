/**
 * POST /api/payment/order — logged-in user registers intent to pay (pending until you confirm).
 * GET  /api/payment/order — list current user's recent orders.
 */

import { NextRequest, NextResponse } from "next/server";
import { getRequestContext } from "@cloudflare/next-on-pages";
import { jwtVerify } from "jose";
import { getCfEnvVal } from "@/lib/lab-config";
import {
  PAYMENT_CHANNELS,
  type PaymentChannel,
  getSuggestedUsdcAmount,
  getSuggestedUsdtAmount,
} from "@/lib/payment-config";
import {
  ensurePaymentSchema,
  createPaymentOrder,
  listPaymentOrdersForUser,
} from "@/lib/payment-db";

export const runtime = "edge";

function getDB() {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (getRequestContext().env as any).DB;
  } catch {
    return null;
  }
}

async function auth(request: NextRequest): Promise<{ user_id: string; email: string } | NextResponse> {
  const authHeader = request.headers.get("authorization") || "";
  const token = authHeader.startsWith("Bearer ") ? authHeader.slice(7) : "";
  if (!token) {
    return NextResponse.json({ error: "UNAUTHORIZED", detail: "请先登录" }, { status: 401 });
  }
  if (token.startsWith("guest-")) {
    return NextResponse.json({ error: "FORBIDDEN", detail: "游客无法创建支付订单" }, { status: 403 });
  }
  if (token.startsWith("local-")) {
    return { user_id: token, email: "" };
  }
  const secret = getCfEnvVal("JWT_SECRET");
  if (!secret) {
    return NextResponse.json({ error: "SERVER_MISCONFIG", detail: "JWT 未配置" }, { status: 503 });
  }
  try {
    const { payload } = await jwtVerify(token, new TextEncoder().encode(secret));
    return {
      user_id: (payload.user_id as string) || "unknown",
      email: (payload.email as string) || "",
    };
  } catch {
    return NextResponse.json({ error: "UNAUTHORIZED", detail: "登录已过期" }, { status: 401 });
  }
}

function amountHintForChannel(ch: PaymentChannel): string {
  if (ch === "usdc_sol") return getSuggestedUsdcAmount() || "USDC (SPL)";
  if (ch === "usdt_sol") return getSuggestedUsdtAmount() || "USDT (SPL)";
  return "";
}

export async function GET(request: NextRequest) {
  const a = await auth(request);
  if (a instanceof NextResponse) return a;

  const db = getDB();
  if (!db) {
    return NextResponse.json({ error: "DB_UNAVAILABLE" }, { status: 503 });
  }
  await ensurePaymentSchema(db);
  const orders = await listPaymentOrdersForUser(db, a.user_id, 20);
  return NextResponse.json({ orders });
}

export async function POST(request: NextRequest) {
  const a = await auth(request);
  if (a instanceof NextResponse) return a;

  let body: { channel?: string };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "BAD_REQUEST", detail: "JSON 无效" }, { status: 400 });
  }

  const channel = body.channel as PaymentChannel;
  if (!channel || !PAYMENT_CHANNELS.includes(channel)) {
    return NextResponse.json(
      { error: "BAD_REQUEST", detail: `channel 必须是: ${PAYMENT_CHANNELS.join(", ")}` },
      { status: 400 }
    );
  }

  const db = getDB();
  if (!db) {
    return NextResponse.json({ error: "DB_UNAVAILABLE" }, { status: 503 });
  }
  await ensurePaymentSchema(db);

  const id = `pay-${crypto.randomUUID()}`;
  const user_memo = `SL-${id.replace("pay-", "").slice(0, 8).toUpperCase()}`;

  await createPaymentOrder(db, {
    id,
    user_id: a.user_id,
    email: a.email,
    channel,
    amount_hint: amountHintForChannel(channel),
    user_memo,
  });

  return NextResponse.json({
    order_id: id,
    user_memo,
    channel,
    amount_hint: amountHintForChannel(channel),
    status: "pending",
    hint_zh:
      "请使用所选方式完成支付，并在转账备注中填写上方「付款备注码」。到账后由管理员在后台核对并开通 Pro。",
    hint_en:
      "Complete payment using your chosen method and put the memo code in the transfer note. Pro is activated after manual verification.",
  });
}
