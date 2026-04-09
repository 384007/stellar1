"use client";

import { useEffect } from "react";
import { BUNDLE_DECOY_STRINGS, DECOY_WINDOW_PRESETS } from "@/lib/vendor-decoy";

/**
 * Production-only: harmless `window` keys + retained bundle strings (no network, no product reads).
 */
export default function ClientDecoyRuntime() {
  useEffect(() => {
    if (process.env.NODE_ENV !== "production") return;
    const w = window as unknown as Record<string, unknown>;
    for (const [k, v] of Object.entries(DECOY_WINDOW_PRESETS)) {
      try {
        if (!(k in w)) w[k] = v;
      } catch {
        /* ignore */
      }
    }
    void BUNDLE_DECOY_STRINGS[BUNDLE_DECOY_STRINGS.length - 1];
  }, []);
  return null;
}
