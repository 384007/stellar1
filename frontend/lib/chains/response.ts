import { NextResponse } from "next/server";
import { sanitizeProductJson, type ProductChain } from "./sanitize";

/**
 * JSON response with denylisted keys stripped (for product APIs).
 */
export function jsonProduct(data: unknown, init?: ResponseInit, chain: ProductChain = "generic"): NextResponse {
  return NextResponse.json(sanitizeProductJson(data, chain), init);
}
