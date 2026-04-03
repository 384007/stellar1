import type { Page } from "@playwright/test";

/** Local token bypasses JWT on /api/analyze; Plus allows local- in edge routes. */
export async function injectLocalAuth(page: Page) {
  await page.addInitScript(() => {
    localStorage.setItem("stellar_token", "local-e2e-playwright");
    localStorage.setItem("stellar_user", JSON.stringify({ username: "e2e", email: "e2e@test.local" }));
  });
}

/** Skip MediaPipe in ScreenModeCapture — required for stable headless 实拍 tests. */
export async function injectSkipMediaPipe(page: Page) {
  await page.addInitScript(() => {
    (window as unknown as { __STELLAR_E2E_SKIP_MEDIAPIPE__?: boolean }).__STELLAR_E2E_SKIP_MEDIAPIPE__ = true;
  });
}

/** Fake camera + screen share so 实拍 / 屏幕录制 flows work headlessly. */
export async function injectFakeMediaDevices(page: Page) {
  await page.addInitScript(() => {
    function canvasStream(): MediaStream {
      const canvas = document.createElement("canvas");
      canvas.width = 640;
      canvas.height = 480;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.fillStyle = "#1a472a";
        ctx.fillRect(0, 0, 640, 480);
        ctx.fillStyle = "#fff";
        ctx.font = "24px sans-serif";
        ctx.fillText("E2E fake video", 40, 240);
      }
      return canvas.captureStream(30);
    }

    const md = navigator.mediaDevices;
    if (!md) return;

    try {
      Object.defineProperty(md, "getUserMedia", {
        configurable: true,
        writable: true,
        value: async () => canvasStream(),
      });
      Object.defineProperty(md, "getDisplayMedia", {
        configurable: true,
        writable: true,
        value: async () => canvasStream(),
      });
    } catch {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (md as any).getUserMedia = async () => canvasStream();
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (md as any).getDisplayMedia = async () => canvasStream();
    }
  });
}
