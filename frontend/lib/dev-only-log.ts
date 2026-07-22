/**
 * Client/server safe: no console noise in production builds (browser DevTools stay clean).
 */
const dev = process.env.NODE_ENV === "development";

export function devLog(...args: unknown[]): void {
  if (dev) console.log(...args);
}

export function devInfo(...args: unknown[]): void {
  if (dev) console.info(...args);
}

export function devWarn(...args: unknown[]): void {
  if (dev) console.warn(...args);
}

export function devError(...args: unknown[]): void {
  if (dev) console.error(...args);
}
