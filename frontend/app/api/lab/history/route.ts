/**
 * GET /api/lab/history — Shot Lab analysis history.
 *
 * Free users: last 7 days or 10 items (whichever is stricter).
 * Pro users: up to 200 items, no date restriction.
 */

import { NextRequest, NextResponse } from "next/server";
import {
  isLabEnabled,
  LAB_FREE_DAILY_LIMIT,
  LAB_FREE_HISTORY_DAYS,
  LAB_FREE_HISTORY_MAX_ITEMS,
  LAB_PRO_HISTORY_MAX_ITEMS,
} from "@/lib/lab-config";
import { ensureLabSchema, getLabHistory, getLabUsageToday } from "@/lib/lab-db";
import {
  authenticateRequest,
  getDB,
  buildQuota,
  labError,
} from "@/lib/lab-auth";
import { jsonProduct } from "@/lib/chains";

export const runtime = "edge";

export async function GET(request: NextRequest) {
  try {
    if (!isLabEnabled()) {
      return labError("FEATURE_DISABLED", "Shot Lab 功能暂未开放", 404);
    }

    const authResult = await authenticateRequest(request);
    if (authResult instanceof NextResponse) return authResult;
    const { user_id, is_pro } = authResult;

    const db = getDB();
    if (!db) {
      return jsonProduct(
        {
          items: [],
          total: 0,
          retention: {
            policy: "limited",
            max_items: LAB_FREE_HISTORY_MAX_ITEMS,
            max_days: LAB_FREE_HISTORY_DAYS,
          },
          quota: {
            used: 0,
            limit: is_pro ? null : LAB_FREE_DAILY_LIMIT,
            remaining: is_pro ? -1 : LAB_FREE_DAILY_LIMIT,
            is_pro,
          },
          tier: is_pro ? "pro" : "free",
          db_available: false,
        },
        undefined,
        "lab",
      );
    }

    await ensureLabSchema(db);

    const items = await getLabHistory(db, user_id, {
      maxItems: is_pro ? LAB_PRO_HISTORY_MAX_ITEMS : LAB_FREE_HISTORY_MAX_ITEMS,
      maxAgeDays: is_pro ? undefined : LAB_FREE_HISTORY_DAYS,
    });

    const usage = await getLabUsageToday(db, user_id);

    return jsonProduct(
      {
        items,
        total: items.length,
        retention: is_pro
          ? { policy: "long_term", max_items: LAB_PRO_HISTORY_MAX_ITEMS }
          : {
              policy: "limited",
              max_items: LAB_FREE_HISTORY_MAX_ITEMS,
              max_days: LAB_FREE_HISTORY_DAYS,
            },
        quota: buildQuota(usage, is_pro),
        tier: is_pro ? "pro" : "free",
      },
      undefined,
      "lab",
    );
  } catch (err) {
    console.error("[lab] GET history error:", err);
    return labError("INTERNAL", "查询失败，请稍后重试。", 500);
  }
}
