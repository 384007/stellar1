import type { NextRequest } from "next/server";
import { resolveLiteModalWorkerBase } from "@/lib/server/prov3-upstream";

export type CfEnvGetter = (key: string) => string;

/**
 * Resolve Modal (or LITE_BACKEND_URL) base for server-side upstream calls.
 */
export function modalAnalysisBase(getCfEnv: CfEnvGetter, request: NextRequest): string {
  return resolveLiteModalWorkerBase(getCfEnv).replace(/\/+$/, "");
}

export function forwardHeadersFromRequest(request: NextRequest): Headers {
  const h = new Headers();
  const auth = request.headers.get("authorization");
  if (auth) h.set("Authorization", auth);
  const cf = (request.headers.get("cf-ipcountry") || request.headers.get("CF-IPCountry") || "").trim();
  if (cf) h.set("CF-IPCountry", cf);
  return h;
}
