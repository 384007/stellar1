// Service worker — PWA installable + offline homepage + MediaPipe asset cache.

const APP_CACHE = "stellar-ai-v33";
const MP_CACHE = "mediapipe-0.10.33";

self.addEventListener("install", (event) => {
  // Wipe old app caches but KEEP the MediaPipe cache (large files, rarely change).
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((k) => k !== MP_CACHE).map((k) => caches.delete(k))
      ))
      .then(() => caches.open(APP_CACHE))
      .then((cache) => cache.addAll(["/"]).catch(() => {}))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((k) => k !== APP_CACHE && k !== MP_CACHE).map((k) => caches.delete(k))
      ))
      .then(() => self.clients.claim())
      .then(() => self.clients.matchAll({ includeUncontrolled: true, type: "window" }))
      .then((clients) => {
        clients.forEach((c) => c.postMessage({ type: "SW_UPDATED" }));
      })
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  // ── MediaPipe assets: cache-first (persist across app updates) ──
  if (url.pathname.startsWith("/mp/")) {
    event.respondWith(
      caches.open(MP_CACHE).then((cache) =>
        cache.match(request).then((cached) => {
          if (cached) return cached;
          return fetch(request).then((response) => {
            if (response.ok) cache.put(request, response.clone());
            return response;
          });
        })
      )
    );
    return;
  }

  // ── Page navigations: network-first with offline fallback ──
  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request).catch(() =>
        caches.match("/").then((r) => r || new Response("Offline", { status: 503 }))
      )
    );
  }
});
