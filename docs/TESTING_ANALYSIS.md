# 分析功能测试（永久方案）

## 1. Playwright E2E（推荐：本地 / CI）

在 **`frontend/`** 目录：

```bash
npm install
npx playwright install chromium
npm run test:e2e
```

### 行为说明

- 自动执行 **`npm run build`** 后 **`next start`**（绑定 `127.0.0.1:3000`），与生产行为一致，避免 `next dev` 下偶发 `.next` 损坏或 HMR 干扰。
- 若已手动启动站点：`STELLAR_SKIP_WEBSERVER=1 npm run test:e2e`（需与 `STELLAR_BASE_URL` 一致）。
- 使用 **`local-e2e-playwright`** 令牌注入 `localStorage`，与线上 `local-*` 放行逻辑一致。
- **`/api/analyze`、`/api/plus`、`/api/club-detect`** 在测试中 **被 Mock**，不依赖 Modal / Gemini / 通义，速度快、稳定。
- **实拍 / 屏幕**：通过 `injectFakeMediaDevices` 伪造 `getUserMedia` / `getDisplayMedia`（画布流），可在无摄像头环境跑通。
- **实拍稳定性**：测试里会设置 `window.__STELLAR_E2E_SKIP_MEDIAPIPE__`，跳过从 CDN 加载 MediaPipe（避免 CI 超时）；生产用户不会设置该变量。

### 环境变量

| 变量 | 含义 |
|------|------|
| `STELLAR_BASE_URL` | 默认 `http://127.0.0.1:3000`，可改为预发 / 生产 URL |
| `STELLAR_SKIP_WEBSERVER=1` | 不自动起 dev，需已手动启动站点 |
| `CI=1` | 失败重试、单 worker |

### 仅跑部分用例

```bash
npx playwright test e2e/analyze-three-paths.spec.ts
npx playwright test e2e/plus-three-paths.spec.ts
npx playwright test --grep "upload"
```

### UI 调试

```bash
npm run test:e2e:ui
```

---

## 2. HTTP 烟雾脚本（真接口）

仓库根目录：

```bash
# 本地（需 next 已启动且配置好 AI Key 才会 200）
node tools/analysis-smoke.mjs

# 线上
STELLAR_BASE_URL=https://你的.pages.dev node tools/analysis-smoke.mjs

# 使用真实 JWT（Plus / 严格 JWT 环境）
STELLAR_TOKEN="eyJhbG..." STELLAR_BASE_URL=https://... node tools/analysis-smoke.mjs
```

| 变量 | 含义 |
|------|------|
| `STELLAR_SKIP_LIVE_AI=1` | 跳过需要边缘 AI 密钥的 `/api/club-detect`、图片 Lite |
| `STELLAR_SKIP_PLUS=1` | 跳过 `/api/plus`（避免扣次数或长耗时） |

退出码：`0` 成功，`1` 失败。

---

## 3. 测试覆盖矩阵

| 页面 | 上传 | 实拍 | 屏幕 |
|------|------|------|------|
| `/analyze` Lite | `analyze-three-paths.spec.ts` | 同文件 capture | 同文件 screen |
| `/analyze` Pro | 同文件 | — | — |
| `/plus` | `plus-three-paths.spec.ts` | 同文件 | 同文件 |

`data-testid`：

- `e2e-upload-video-input` — `UploadZone` 隐藏 file input  
- `e2e-capture-record` — 实拍录制按钮  

---

## 4. CI 建议

在 GitHub Actions 中（仅前端变更时）：

```yaml
- uses: actions/setup-node@v4
  with:
    node-version: "22"
- run: npm ci
  working-directory: frontend
- run: npx playwright install chromium --with-deps
  working-directory: frontend
- run: npm run build && npm run test:e2e
  working-directory: frontend
  env:
    CI: true
```

（若不想在 CI 里 `build`，可改 `webServer` 为 `npm run dev`，但首次会较慢。）

---

## 5. 与产品路径的对应关系

- **Lite**：边缘 `/api/analyze`（Gemini/Qwen），E2E 中已 Mock。  
- **Pro / Plus**：边缘代理 Modal→Render，E2E 中已 Mock。  
- 要验证 **真实 Modal 链路**，请用 `analysis-smoke.mjs` 或对预发环境关闭 Mock 的手动测试。
