# PR #50 — Lite 分析路径测试报告

**仓库**: `dytsui/stellar1`  
**分支**: `feat/modal-centric-backend-proxies`  
**预览**: [https://pr-50.stellar1.pages.dev](https://pr-50.stellar1.pages.dev)  
**报告日期**: 2026-04-10  

---

## 1. Lite 路径说明（产品行为）

- **入口 UI**: `/analyze`，选择 **「普通分析 / Standard」**（`analysisMode === "lite"`）。
- **浏览器请求**: 同源 **`POST /api/lite/analyze-proxy`**（multipart：`file`、`request_id`；请求头：`X-Stellar-Idempotency-Key`；若 JWT 含 `.` 则带 `Authorization`）。
- **与 Pro 的差异**: Pro 依赖 **`POST /api/history/upload-video`** 与 **`/api/prov3/...`**，且 `upload-video` 对 `local-*` token **不解析 userId**（见主报告）。Lite 代理路由 `requireAuth` 对 **`local-*` 放行**（与主站 JWT 校验逻辑一致，见 `frontend/app/api/lite/analyze-proxy/route.ts`）。

---

## 2. 摘要

| 项目 | 结果 |
|------|------|
| `frontend` **build**（含 leak 脚本） | **pass** |
| Lite **API**（curl → 预览站 `/api/lite/analyze-proxy`） | **pass**（链路可达，见下） |
| Lite **浏览器 UI**（Playwright：极小测试 mp4） | **部分 / 未通过可见终态**（120s 内未出现「分析结果」或匹配错误文案） |
| Lite 场景下 **Network 请求 URL** 泄漏巡检（Playwright 监听） | **pass**（未命中 `modal.run` / `onrender.com` / `r2.*` / `upload_url` / `f=1&z=` 等） |

---

## 3. 构建验证

在 `frontend/` 执行 `npm run build`（2026-04-10）：

- **check-no-client-leaks.mjs**: `ok ( 76 files scanned )`
- **check-client-source-leaks.mjs**: `ok ( 95 files scanned )`

---

## 4. Lite API 验证（curl → pr-50）

### 4.1 鉴权与路由

使用 **`Authorization: Bearer local-e2e-playwright`**（`local-*`），并携带 **`X-Stellar-Idempotency-Key`** 与 multipart 体。

### 4.2 极小 JPEG

- **响应**: HTTP **200**，正文为 **SSE**（`text/event-stream`），首段含 `lite-start`，随后 `data:` JSON。
- **业务结果示例**: `ok:false`，`code:"LITE_PIPELINE_FAILED"`，`detail` 含 `lite_clean_video_empty`（测试图非有效「可清洗」视频，属**上游管线对输入的拒绝**，不是 401/503 网关未配置）。

### 4.3 极小 MP4（32 字节桩文件）

- **响应**: HTTP **200**，SSE `data:` JSON 中 `code:"LITE_PIPELINE_FAILED"`，`detail` 含 **`lite_clean_video_unreadable`**。
- **结论**: 浏览器到 **同源 Edge 代理 → 上游 Lite 管线** 的往返在 **数秒内完成**，返回**结构化错误**，**非**长时间无响应、**非**仅转圈无反馈。

---

## 5. 浏览器 UI 验证（Playwright）

- **步骤**: 打开 `/analyze` → 点击「普通分析 / Standard」→ 通过 `e2e-upload-video-input` 上传与 API 测试相同的 **极小 mp4**。
- **结果**: **120s 内**未匹配到「分析结果 / Analysis Results」或脚本中的错误关键词（可能与实际 UI 文案、异步状态有关）。
- **说明**: **API 层已证明** Lite 代理与管线可达；UI 层在本轮 **无效短视频** 下**未收敛到可断言的终态文案**，若要验证「结果卡片」需换 **有效短视频** 或延长等待并细化选择器。

---

## 6. 泄漏巡检（Lite 上传会话）

在 Lite Playwright 脚本中监听 **request URL**，对以下子串做命中检查：

`upload_url`、`r2.cloudflarestorage.com`、`cloudflarestorage.com`、`modal.run`、`onrender.com`、`/api/cdn/p?f=1&z=`

- **结果**: **0 命中**（`leakCount: 0`）。

---

## 7. 发现与建议

1. **Lite API** 在预览站可用，`local-*` 可用于**自动化探测代理**，无需真实 JWT（与 Pro 上传存储路径不同）。
2. **UI 终态** 在本轮 **极小样例** 下未断言成功；建议用 **真实短 swing 视频** 复测 UI，或根据 `readLiteAnalyzeResult` / 页面错误展示补充 E2E 选择器。
3. 若需与 [PR-50 总报告](./PR-50-verification-report.md) 对照：总报告中 **Pro 上传** 依赖 JWT userId；**Lite 分析** 以本报告为准。

---

## 8. 执行命令（摘录）

```bash
# 构建
cd frontend && npm run build

# Lite 代理（示例：极小 mp4；IDEM 每次唯一）
curl -sS -w "\nHTTP %{http_code}\n" --max-time 120 -X POST \
  "https://pr-50.stellar1.pages.dev/api/lite/analyze-proxy" \
  -H "Authorization: Bearer local-e2e-playwright" \
  -H "X-Stellar-Idempotency-Key: $IDEM" \
  -F "file=@/path/to/lite-smoke.mp4;type=video/mp4" \
  -F "request_id=$IDEM"
```

---

*本文件仅记录 Lite 路径测试结论；未修改业务代码。*
