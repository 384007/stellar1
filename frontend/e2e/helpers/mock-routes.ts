import type { Page } from "@playwright/test";
import { mockAnalyzeLite, mockAnalyzePro, mockClubDetect, mockPlusResult } from "../fixtures/mock-api";

/**
 * Intercept heavy APIs so E2E stays fast and does not need Modal/Gemini keys.
 */
export async function installAnalysisMocks(page: Page, mode: "lite" | "pro") {
  const body = JSON.stringify(mode === "pro" ? mockAnalyzePro : mockAnalyzeLite);

  await page.route("**/api/club-detect", async (route) => {
    if (route.request().method() !== "POST") {
      await route.continue();
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockClubDetect),
    });
  });

  await page.route("**/api/analyze", async (route) => {
    if (route.request().method() !== "POST") {
      await route.continue();
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body,
    });
  });
}

export async function installPlusMocks(page: Page) {
  await page.route("**/api/club-detect", async (route) => {
    if (route.request().method() !== "POST") {
      await route.continue();
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockClubDetect),
    });
  });

  // Exact pathname — glob `**/api/plus` can match `/api/plus/usage` first and confuse ordering in some runners
  await page.route(
    (url) => {
      try {
        return url.pathname === "/api/plus";
      } catch {
        return false;
      }
    },
    async (route) => {
      if (route.request().method() !== "POST") {
        await route.continue();
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockPlusResult),
      });
    }
  );
}
