import { test, expect } from "@playwright/test";
import { injectFakeMediaDevices, injectLocalAuth, injectSkipMediaPipe } from "./helpers/inject";
import { installAnalysisMocks } from "./helpers/mock-routes";

const tinyMp4 = Buffer.from([
  0x00, 0x00, 0x00, 0x20, 0x66, 0x74, 0x79, 0x70, 0x69, 0x73, 0x6f, 0x6d, 0x00, 0x00, 0x02, 0x00,
  0x69, 0x73, 0x6f, 0x6d, 0x69, 0x73, 0x6f, 0x32, 0x6d, 0x70, 0x34, 0x31, 0x00, 0x00, 0x00, 0x08,
]);

test.beforeEach(async ({ page }) => {
  await injectLocalAuth(page);
});

test.describe("Analyze — upload (mocked APIs)", () => {
  test("Lite: video upload reaches results", async ({ page }) => {
    await installAnalysisMocks(page, "lite");
    await page.goto("/analyze");

    await page.getByRole("button", { name: /普通分析|Standard/ }).click();
    await page.getByTestId("e2e-upload-video-input").setInputFiles({
      name: "swing-e2e.mp4",
      mimeType: "video/mp4",
      buffer: tinyMp4,
    });

    await expect(page.getByText(/分析结果|Analysis Results/)).toBeVisible({ timeout: 60_000 });
    await expect(page.getByText("E2E 模拟分析")).toBeVisible();
  });

  test("Pro: video upload reaches results", async ({ page }) => {
    await installAnalysisMocks(page, "pro");
    await page.goto("/analyze");

    await page.getByRole("button", { name: /Pro|深度/ }).click();
    await page.getByTestId("e2e-upload-video-input").setInputFiles({
      name: "swing-e2e.mp4",
      mimeType: "video/mp4",
      buffer: tinyMp4,
    });

    await expect(page.getByText(/分析结果|Analysis Results/)).toBeVisible({ timeout: 60_000 });
  });
});

test.describe("Analyze — 实拍 (mocked APIs + canvas getUserMedia)", () => {
  test.beforeEach(async ({ page }) => {
    await injectSkipMediaPipe(page);
    await injectFakeMediaDevices(page);
  });

  test("实拍: record short clip → results", async ({ page }) => {
    test.setTimeout(120_000);
    await installAnalysisMocks(page, "lite");
    await page.goto("/analyze");
    await expect(page.getByRole("button", { name: /上传视频|Upload/ })).toBeVisible({ timeout: 30_000 });

    await page.getByRole("button", { name: /实拍模式|^Camera$/ }).click();
    const openCam = page.getByRole("button", { name: /打开摄像头|Open Camera/ });
    await openCam.waitFor({ state: "visible", timeout: 15_000 });
    await openCam.click();

    await expect(page.getByTestId("e2e-capture-record")).toBeEnabled({ timeout: 60_000 });

    await page.getByTestId("e2e-capture-record").click();
    await page.waitForTimeout(2500);
    await page.getByTestId("e2e-capture-record").click();

    await expect(page.getByText(/分析结果|Analysis Results/)).toBeVisible({ timeout: 60_000 });
  });
});

test.describe("Analyze — 屏幕 (mocked APIs + fake display capture)", () => {
  test.beforeEach(async ({ page }) => {
    await injectSkipMediaPipe(page);
    await injectFakeMediaDevices(page);
  });

  test("屏幕: screen record → results", async ({ page }) => {
    test.setTimeout(90_000);
    await installAnalysisMocks(page, "lite");
    await page.goto("/analyze");
    await expect(page.getByRole("button", { name: /上传视频|Upload/ })).toBeVisible({ timeout: 30_000 });

    await page.getByRole("button", { name: /屏幕模式|^Screen$/ }).click();
    await page.getByRole("button", { name: /录制屏幕|Record Screen/ }).click();

    await expect(page.getByText(/正在录制屏幕|Recording Screen/)).toBeVisible({ timeout: 15_000 });
    await page.getByRole("button", { name: /停止录制并分析|Stop & Analyze/ }).click();

    await expect(page.getByText(/分析结果|Analysis Results/)).toBeVisible({ timeout: 60_000 });
  });
});
