import { NextRequest, NextResponse } from "next/server";
import { getRequestContext } from "@cloudflare/next-on-pages";
import { buildProv3ModalUrlList, normalizeProHttpApiBase } from "@/lib/server/prov3-upstream";

export const runtime = "edge";

function getCfEnv(key: string): string {
  try {
    return ((getRequestContext().env as Record<string, string>)[key] as string) || "";
  } catch {
    return process.env[key] || "";
  }
}

function r2PublicOrigin(): string {
  const raw =
    getCfEnv("NEXT_PUBLIC_STELLAR_PROV3_R2_PUBLIC_BASE") ||
    process.env.NEXT_PUBLIC_STELLAR_PROV3_R2_PUBLIC_BASE ||
    "";
  return raw.trim().replace(/\/+$/, "");
}

function modalPrimaryOrigin(request: NextRequest): string {
  const cf = (request.headers.get("cf-ipcountry") || request.headers.get("CF-IPCountry") || "").trim();
  const cn = cf.toUpperCase() === "CN";
  const list = buildProv3ModalUrlList(getCfEnv, cn);
  const first = list[0] ? normalizeProHttpApiBase(list[0]) : "";
  return first;
}

function modalOriginAllowHosts(request: NextRequest): Set<string> {
  const cf = (request.headers.get("cf-ipcountry") || request.headers.get("CF-IPCountry") || "").trim();
  const cn = cf.toUpperCase() === "CN";
  const hosts = new Set<string>();
  for (const o of buildProv3ModalUrlList(getCfEnv, cn)) {
    try {
      const u = new URL(o.startsWith("http") ? o : `https://${o}`);
      hosts.add(u.hostname.toLowerCase());
    } catch {
      /* ignore */
    }
  }
  return hosts;
}

function safeFixedPath(raw: string, requiredPrefix: string): string | null {
  let decoded: string;
  try {
    decoded = decodeURIComponent(raw);
  } catch {
    return null;
  }
  if (decoded.includes("\0") || decoded.includes("//")) return null;
  try {
    const u = new URL(decoded, "https://dummy.invalid");
    const path = u.pathname + u.search;
    if (!path.startsWith(requiredPrefix)) return null;
    if (path.includes("/../") || path.includes("\\")) return null;
    return path;
  } catch {
    return null;
  }
}

function decodeBase64Url(s: string): string | null {
  const pad = s.length % 4 === 0 ? "" : "====".slice(s.length % 4);
  const b64 = s.replace(/-/g, "+").replace(/_/g, "/") + pad;
  try {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new TextDecoder().decode(bytes);
  } catch {
    return null;
  }
}

function allowAbsoluteUrl(urlStr: string, request: NextRequest): URL | null {
  let u: URL;
  try {
    u = new URL(urlStr);
  } catch {
    return null;
  }
  if (u.protocol !== "https:" && u.protocol !== "http:") return null;
  const host = u.hostname.toLowerCase();
  if (modalOriginAllowHosts(request).has(host)) return u;
  const ro = r2PublicOrigin();
  if (ro) {
    try {
      const ru = new URL(ro.startsWith("http") ? ro : `https://${ro}`);
      if (host === ru.hostname.toLowerCase()) return u;
    } catch {
      /* ignore */
    }
  }
  return null;
}

function buildUpstreamResponse(ir: Response, method: "GET" | "HEAD"): NextResponse {
  const resHeaders = new Headers();
  const ct = ir.headers.get("content-type");
  if (ct) resHeaders.set("Content-Type", ct);
  const ar = ir.headers.get("accept-ranges");
  if (ar) resHeaders.set("Accept-Ranges", ar);
  const cr = ir.headers.get("content-range");
  if (cr) resHeaders.set("Content-Range", cr);
  const cl = ir.headers.get("content-length");
  if (cl) resHeaders.set("Content-Length", cl);
  const cc = ir.headers.get("cache-control");
  if (cc) resHeaders.set("Cache-Control", cc);
  const etag = ir.headers.get("etag");
  if (etag) resHeaders.set("ETag", etag);
  const lm = ir.headers.get("last-modified");
  if (lm) resHeaders.set("Last-Modified", lm);

  if (method === "HEAD") {
    return new NextResponse(null, { status: ir.status, headers: resHeaders });
  }
  return new NextResponse(ir.body, { status: ir.status, headers: resHeaders });
}

async function runProxy(request: NextRequest, method: "GET" | "HEAD"): Promise<NextResponse> {
  const sp = request.nextUrl.searchParams;
  const m = sp.get("m");
  const r = sp.get("r");
  const f = sp.get("f");
  const pRaw = sp.get("p");
  const z = sp.get("z");

  let target: string | null = null;

  if (m === "1" && pRaw) {
    const path = safeFixedPath(pRaw, "/pro-v3/media/");
    if (!path) return new NextResponse(null, { status: 400 });
    const primary = modalPrimaryOrigin(request);
    if (!primary) return new NextResponse(null, { status: 503 });
    target = `${primary}${path}`;
  } else if (r === "1" && pRaw) {
    const path = safeFixedPath(pRaw, "/prov3-media/");
    if (!path) return new NextResponse(null, { status: 400 });
    const ro = r2PublicOrigin();
    if (!ro) return new NextResponse(null, { status: 503 });
    const base = ro.startsWith("http") ? ro : `https://${ro}`;
    target = `${base.replace(/\/+$/, "")}${path}`;
  } else if (f === "1" && z) {
    if (z.length > 4096) return new NextResponse(null, { status: 400 });
    const decoded = decodeBase64Url(z);
    if (!decoded) return new NextResponse(null, { status: 400 });
    const u = allowAbsoluteUrl(decoded, request);
    if (!u) return new NextResponse(null, { status: 403 });
    target = u.toString();
  } else {
    return new NextResponse(null, { status: 400 });
  }

  const headers = new Headers();
  const range = request.headers.get("range");
  if (range && method === "GET") headers.set("Range", range);

  let ir: Response;
  try {
    ir = await fetch(target, { method, headers, redirect: "follow" });
  } catch {
    return new NextResponse(null, { status: 502 });
  }

  return buildUpstreamResponse(ir, method);
}

export async function GET(request: NextRequest) {
  return runProxy(request, "GET");
}

export async function HEAD(request: NextRequest) {
  return runProxy(request, "HEAD");
}
