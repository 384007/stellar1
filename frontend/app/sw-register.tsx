"use client";

import { useEffect } from "react";
import { devError } from "@/lib/dev-only-log";

// IMPORTANT: bump this on each deployment to bypass any stale /sw.js cache
// that may be controlled by an older service worker.
const SW_VERSION = "v33";
const SW_URL = `/sw.js?v=${SW_VERSION}`;

export default function SwRegister() {
  useEffect(() => {
    if (typeof window === "undefined" || !("serviceWorker" in navigator)) return;

    let registration: ServiceWorkerRegistration | null = null;
    let reloaded = false;

    // Reload once when controller changes or update notification arrives.
    const doReload = () => {
      if (reloaded) return;
      reloaded = true;
      window.location.reload();
    };

    navigator.serviceWorker.addEventListener("controllerchange", doReload);
    const onMessage = (event: MessageEvent) => {
      if (event.data?.type === "SW_UPDATED") doReload();
    };
    navigator.serviceWorker.addEventListener("message", onMessage);

    const onVisible = () => {
      if (document.visibilityState === "visible") {
        registration?.update().catch(() => {});
      }
    };

    const register = async () => {
      try {
        // Try to wake any existing registration first.
        const regs = await navigator.serviceWorker.getRegistrations();
        await Promise.all(regs.map((r) => r.update().catch(() => {})));

        // Register with versioned SW URL to guarantee update even if an older
        // SW cached /sw.js itself.
        registration = await navigator.serviceWorker.register(SW_URL, { scope: "/" });

        // Force-check for a new SW immediately on every launch
        registration.update().catch(() => {});

        // Re-check whenever the app comes back to foreground (home-screen tap / tab switch)
        document.addEventListener("visibilitychange", onVisible);
      } catch (err) {
        devError("[SW] Registration failed:", err);
      }
    };

    if (document.readyState === "complete") {
      register();
    } else {
      window.addEventListener("load", register);
    }

    return () => {
      navigator.serviceWorker.removeEventListener("controllerchange", doReload);
      navigator.serviceWorker.removeEventListener("message", onMessage);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);

  return null;
}
