import { getRequestContext } from "@cloudflare/next-on-pages";

/** JWT_SECRET from Pages bindings or process.env (Edge / Node). */
export function getEdgeJwtSecret(): string {
  try {
    const s = (getRequestContext().env as Record<string, string>).JWT_SECRET;
    if (s) return s;
  } catch {
    /* not in CF context */
  }
  return process.env.JWT_SECRET || "";
}
