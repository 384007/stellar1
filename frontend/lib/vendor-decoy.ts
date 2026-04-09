/**
 * Inert noise only: no product logic reads these. Strings are opaque (no framework/vendor names)
 * so bundle greps do not suggest a specific stack.
 */

export const BUNDLE_DECOY_STRINGS = [
  "k9f2a1c7e4b8d0",
  "m3q6w8r5t2y7u1",
  "p4s9v0x2z5h8j3",
  "n7b4c1d9e6f2g8",
  "a0e5i9o3u7y2w6",
] as const;

/** Production-only `window` keys: opaque identifiers, no URLs or product keys. */
export const DECOY_WINDOW_PRESETS: Readonly<Record<string, string>> = {
  __s7k2m9p1: "4f8c2a9e1b7d5c0",
  __s7k2m9p2: "8a3d6f1e9c2b7a4",
  __s7k2m9p3: "1e7c4b9a2d8f6e3",
  __s7k2m9p4: "6b2e9a7c4f1d8b5",
};
