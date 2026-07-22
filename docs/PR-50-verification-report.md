# PR #50 测试与验证报告

**仓库**: `dytsui/stellar1`  
**PR**: [#50](https://github.com/dytsui/stellar1/pull/50)  
**分支**: `feat/modal-centric-backend-proxies`  
**验证 HEAD**: `641646b1`（`fix(prov3): drop unknown absolute media URLs; strict sanitize + leak check`）  
**预览环境**: [https://pr-50.stellar1.pages.dev](https://pr-50.stellar1.pages.dev)  
**报告日期**: 2026-04-10  

---

## 摘要

| 项目 | 结果 |
|------|------|
| build | **pass** |
| smoke | **pass** |
| upload analyze | **fail**（无有效 JWT，无法走完真实上传链路） |
| leak audit（本机构建） | **pass** |
| leak audit（pr-50 线上静态资源） | **fail**（见下文） |

---

## 1. 环境与步骤

1. 检出 PR head：`git fetch origin pull/50/head:pr-50-head`，`641646b1` 与 `origin/feat/modal-centric-backend-proxies` 一致。
2. 目录：`frontend/`
3. 依赖：`npm install`（仓库无单一 root lockfile 约束下的 `npm ci` 前提，使用 `npm install`）。
4. 构建：`npm run build`（含 `check-no-client-leaks.mjs`、`check-client-source-leaks.mjs`）。
5. 浏览器冒烟：Playwright（Chromium），目标 `https://pr-50.stellar1.pages.dev`；脚本在本地 `/tmp` 执行，通过 `NODE_PATH` 解析 `frontend/node_modules/playwright`。全权限运行以避免沙箱内 Chromium SIGSEGV。
6. 补充：`curl` 抽检 HTTP 与静态 chunk；本机 `next start` 探测上传 API 行为。

---

## 2. build：pass

- `npm run build` 成功结束。
- **check-no-client-leaks.mjs**：`ok ( 76 files scanned )`
- **check-client-source-leaks.mjs**：`ok ( 95 files scanned )`
- 备注：Next 构建时提示多份 `package-lock.json`（含用户目录下 lockfile），与结论无关。

---

## 3. smoke：pass

- 对预览站自动化访问：`/analyze`、`/plus`、`/history`、`/share/invalid-token-0000` 可加载；会话内 **无 `pageerror`**。
- `curl -sI https://pr-50.stellar1.pages.dev/pro` 返回 **200**，HTML 含 Pro 上传区与 `data-testid="e2e-upload-video-input"`。
- Playwright 汇总中 `nav.pro` 可能显示为空对象：因 `page.goto` 返回的 `Response` 在部分情况下为 `null`，`JSON.stringify` 省略 `undefined` 字段；与 `curl` 验证的 **/pro 可访问** 不矛盾。

---

## 4. upload analyze：fail

- 使用与仓库 E2E 相同的 **`local-e2e-playwright`**（`localStorage`）在预览站走 Pro 上传：**240s 内**未出现「分析结果 / Analysis Results」或脚本匹配的错误文案（`wait_timeout`）。
- **代码层面原因**：`frontend/app/api/history/upload-video/route.ts` 中 `getUserId` 对 `token.startsWith("local-")` **直接返回 `null`**，随后 `if (!userId) return 401`（`{"detail":"未登录"}`）。
- **实测**：`Authorization: Bearer local-e2e-playwright` 调用 `POST /api/history/upload-video` 在 **预览站与本机 `next start`** 均为 **401**。
- **结论**：在未提供 **有效 JWT** 的前提下，无法完成「上传 → 分析 → 结果或明确失败」的端到端验证；仓库内 **`local-*` 并非上传 API 的合法用户 token**（与 E2E mock 流程不同）。

---

## 5. leak audit

### 5.1 本仓库 `npm run build` 产物：pass

- 对 `frontend/.next/static` 检索 **`f=1`** / **`f=1&z=`**：**无匹配**。
- 与构建阶段 `check-no-client-leaks.mjs`、`check-client-source-leaks.mjs` 通过一致。

### 5.2 预览站静态资源：fail

- 在  
  `https://pr-50.stellar1.pages.dev/_next/static/chunks/1648-8bca1d5dd2bb06b3.js`  
  的响应体中检出 **`/api/cdn/p?f=1&z=`**（与「不应再出现旧 CDN 查询形态」的验收目标冲突）。
- 同 chunk 抽检 **未** 发现 `modal.run`、`onrender.com`、`r2.cloudflarestorage.com`、`upload_url` 等（本次扫描范围内）。

### 5.3 推论

- 当前 **HEAD 本地构建** 与 **pr-50 当前提供的 chunk** 在「是否含 `f=1&z=`」上 **不一致**，较可能原因包括：预览 **未对齐最新 commit 构建**、或 **CDN/缓存** 仍服务旧 artifact。需以运维侧核对部署为准。

---

## 6. 环境与工具说明

- Playwright 绑定的 Chromium 在 **受限沙箱** 中启动时出现 **SIGSEGV**；在 **全权限** 宿主环境下可正常运行。属执行环境限制，非应用逻辑结论。

---

## 7. 问题清单（逐条）

1. **pr-50 静态 JS** 仍含 `/api/cdn/p?f=1&z=`，与同 commit **本地** `.next/static` 不一致 → 部署/缓存与 **当前构建** 可能未对齐。
2. **`/api/history/upload-video` 拒绝 `local-*` token**，自动化无法用现有 E2E token 验证真实上传链路；需 **有效 JWT** 或内部测试账号流程。
3. **Playwright + Chromium** 在部分 CI/沙箱中需全权限或等价配置，否则可能崩溃。

---

## 8. 后续最小动作建议（非本次代码修改范围）

1. **泄漏（线上仍见 `f=1&z=`）**：确认 Cloudflare Pages 上 PR #50 预览是否已用 **`641646b1` 重新构建** 并刷新静态资源；若已对齐仍出现，再在 `frontend/` 内排查仍写入 client bundle 的源码路径（本地当前构建已扫不出该串）。
2. **上传 E2E**：使用 **真实 Bearer JWT** 对预览站重跑「上传 → Pro 分析」与 Network 审计；或文档化「无 JWT 无法测通上传 API」。

---

## 9. 实际执行的命令（摘录）

```bash
git fetch origin pull/50/head:pr-50-head
git rev-parse pr-50-head   # 641646b1...

cd frontend && npm install && npm run build

# 预览站抽检
curl -sI "https://pr-50.stellar1.pages.dev/pro"
curl -s "https://pr-50.stellar1.pages.dev/_next/static/chunks/1648-8bca1d5dd2bb06b3.js" | grep -o 'f=1[^"]*' | head

# 上传 API（预期 401）
curl -s -w "\n%{http_code}" -X POST "https://pr-50.stellar1.pages.dev/api/history/upload-video" \
  -H "Authorization: Bearer local-e2e-playwright" -F "file=@/dev/null"
```

---

*本文件仅记录测试与验证结论；未包含业务代码修改。*
