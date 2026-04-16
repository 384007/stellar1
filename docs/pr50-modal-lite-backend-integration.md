# PR50：Stellar1 Lite × Modal 后端对接说明

> Branch: `feat/modal-centric-backend-proxies`  
> PR: #50 `Test feat/modal-centric-backend-proxies against main`

这份文档是给 **stellar1 Lite 分析链路** 用的落地说明，目标不是重写整套后端，而是把 **Lite 上传分析** 稳定接到 **Modal 后端**，同时保留回退路径，避免影响现有主站与 Pro v3。

---

## 1. 目标

### 这次要做到
- Lite 前端上传视频后，默认走 **自家前端 API / proxy**，不要让浏览器直接打 Modal。
- 前端 proxy 再转发到 **Modal Lite 后端**。
- Modal Lite 不可用时，可以按开关回退到现有 Render / backend 服务。
- 先把 **Lite 跑通**，不要把 Pro v3 和旧路径一起搅进去。
- PR50 先以 **联调 / 验证 / 预览可测** 为目标。

### 这次先不要做
- 不把浏览器直接绑死到 Modal 域名。
- 不把 Pro v3 `/pro-v3/analyze` 混进 Lite 接口。
- 不在这次 PR 里强推 3D、240fps、Modal 专属重型逻辑到 Lite 主链路。

---

## 2. 已有仓库信息（对接时要记住）

### Modal 主应用
仓库里已经有：
- `modal_app.py`
- `backend/.env.example`

从 `modal_app.py` 可以确认：
- 主 Modal 部署命令：`modal deploy modal_app.py`
- 主 ASGI 入口：`main:app`
- Pro v3 路由在 `/pro-v3/*`
- 文件顶部已经注明：**独立 Lite Modal：`modal deploy modal_app_lite.py`**

所以这次 **Lite 对接优先使用独立 Lite Modal 服务**，不要直接挤进 Pro v3。

---

## 3. 建议架构（PR50 按这个走）

```text
Lite Upload Page
  -> frontend internal API (same-origin)
    -> Modal Lite backend
      -> Gemini / pose / keyframes / result

失败时（可选开关）
  -> fallback Render backend
```

### 为什么要这样接
1. **隐藏 Modal 域名**，后面换服务不用改浏览器代码。
2. **方便做 header / auth / timeout / retry / fallback**。
3. Cloudflare / Next 这层可以统一处理错误文案与返回结构。
4. 以后要切回 Render 或双活，不需要改前端页面组件。

---

## 4. 前端怎么接

### 统一原则
Lite 页面只打你自己的前端接口，例如：
- `POST /api/lite/analyze`
- `GET /api/lite/analysis/:id`
- `GET /api/lite/health`

不要在浏览器里直接出现：
- Modal 原始 URL
- Render 原始 URL
- Gemini 相关敏感路径

### 推荐前端环境变量
建议新增这几个，统一由 proxy 决定流量去向：

```bash
STELLAR_LITE_BACKEND_MODE=modal
STELLAR_MODAL_LITE_ORIGIN=https://your-modal-lite-origin.example.com
STELLAR_RENDER_BACKEND_ORIGIN=https://your-render-backend.example.com
STELLAR_LITE_FALLBACK_ENABLED=1
STELLAR_LITE_PROXY_TIMEOUT_MS=90000
```

### 模式说明
- `STELLAR_LITE_BACKEND_MODE=modal`
  - Lite 默认走 Modal Lite。
- `STELLAR_LITE_FALLBACK_ENABLED=1`
  - Modal 失败时允许回退 Render。
- `STELLAR_LITE_BACKEND_MODE=render`
  - 临时切回旧后端。

---

## 5. Proxy 层职责

前端 proxy 不要只做机械转发，至少要做这几件事：

### A. 统一请求入口
接收前端表单 / 文件上传，然后转发给真正分析服务。

### B. 统一超时
Lite 上传分析要有统一 timeout，避免浏览器无限等。

### C. 统一返回结构
不管后面打的是 Modal 还是 Render，前端组件都尽量收到同一种 JSON 结构。

### D. 统一 fallback
当 Modal 返回 5xx / timeout / health fail 时，按开关切 Render。

### E. 统一日志
proxy 日志里至少打：
- request id
- selected backend = modal / render
- upstream status
- latency
- fallback happened or not

---

## 6. Modal Lite 后端配合要求

### 先做最小闭环
Lite Modal 后端只要先保证下面三件事：

1. `GET /health`
2. `POST /analyze/lite`（或等价 Lite 分析入口）
3. 返回结构尽量跟现有 Lite 一致

### health 检查建议
返回至少包含：

```json
{
  "ok": true,
  "runtime": "modal",
  "service": "stellar-lite",
  "version": "..."
}
```

这样前端 proxy 可以明确知道当前打到的是不是 Modal。

### Lite 返回结构建议统一字段
至少保证这些字段稳定：

```json
{
  "ok": true,
  "analysisId": "...",
  "score": 0,
  "summary": "...",
  "keyframes": [],
  "issues": [],
  "source": "modal-lite"
}
```

字段名如果暂时和旧后端不同，优先在 proxy 层做 normalize，不要立刻改动所有前端组件。

---

## 7. Modal 相关环境变量

### Modal Secret
`modal_app.py` 里已经写了：
- 使用 `custom-secret`

所以 Modal 控制台里要保证至少有：
- Gemini 所需 key
- 代理相关 key（如果需要 CN/代理策略）

### backend/.env.example 已经给出的重点变量
后端联调时重点关注：
- `GEMINI_API_KEY`
- `GEMINI_PROXY_ALI`
- `GEMINI_PROXY_JD`
- `FRONTEND_URL`
- `JWT_SECRET`

如果 Lite 也复用这套后端环境语义，前后端就更容易对齐。

---

## 8. PR50 推荐实现顺序

### 第一步：确认 Lite Modal 存活
- 能 `modal deploy modal_app_lite.py`
- `GET /health` 返回 `runtime=modal`

### 第二步：前端加内部 proxy
建议新建：

```text
frontend/app/api/lite/analyze/route.ts
frontend/app/api/lite/health/route.ts
frontend/lib/lite-backend.ts
```

### 第三步：在 `lite-backend.ts` 里收口
统一封装：
- 选 Modal 还是 Render
- health check
- fallback
- timeout
- response normalize

### 第四步：Lite 页面只认自己的 `/api/lite/*`
页面组件不要再自己判断 Modal / Render。

### 第五步：补验收文档和截图
PR50 要能回答下面这些问题：
- Preview 能不能打开？
- 上传视频能不能完成一次 Lite 分析？
- Modal 挂掉时前端会不会直接炸？
- 回退 Render 后页面是否还能出结果？

---

## 9. 推荐的返回策略

### 正常
```json
{
  "ok": true,
  "backend": "modal",
  "data": { ...normalized analysis... }
}
```

### Modal 失败但已回退
```json
{
  "ok": true,
  "backend": "render-fallback",
  "fallback": true,
  "data": { ...normalized analysis... }
}
```

### 双后端都失败
```json
{
  "ok": false,
  "backend": "modal",
  "error": "lite_analysis_failed",
  "message": "Lite analysis temporarily unavailable"
}
```

---

## 10. 验收清单

### 功能
- [ ] Lite 页面上传视频，能走前端 proxy
- [ ] Proxy 默认命中 Modal Lite
- [ ] `GET /api/lite/health` 能正确反映 Modal 状态
- [ ] Modal 失败时可按开关回退 Render
- [ ] 页面不暴露真实 Modal 域名
- [ ] 前端 UI 不需要知道后端到底是 Modal 还是 Render

### 稳定性
- [ ] 大文件上传不会因为浏览器直连跨域出问题
- [ ] Proxy 有 timeout
- [ ] Proxy 有 request id / upstream latency 日志
- [ ] 分析失败时前端报错文案一致

### PR 可验证性
- [ ] PR50 页面里有本说明文档
- [ ] 说明清楚当前 Lite 打的是哪个后端
- [ ] 提供最小测试步骤

---

## 11. 最小测试步骤

### 健康检查
1. 打开 PR50 预览
2. 请求 `/api/lite/health`
3. 确认返回 `runtime=modal` 或明确 fallback 状态

### Lite 上传
1. 上传一个短 MP4
2. 看 proxy 日志是否先打 Modal
3. 确认能返回 Lite 分析结果

### fallback
1. 临时改坏 `STELLAR_MODAL_LITE_ORIGIN` 或让 Modal 返回失败
2. 再上传一次
3. 确认系统按开关切到 Render

---

## 12. 这次 PR50 建议先不碰的范围

- 不在这次里改 Pro v3 `/pro-v3/*`
- 不在这次里合并 240fps / minterpolate / SwingNet / MMAction2 重逻辑进 Lite
- 不让 Lite 页面直接依赖 Modal 专用返回结构
- 不在浏览器里硬编码 Modal URL

---

## 13. 推荐提交说明

如果后面要继续拆提交，建议按下面节奏：

1. `docs: add PR50 modal lite backend integration plan`
2. `feat(frontend): add lite backend proxy + backend selector`
3. `feat(frontend): add modal health check + render fallback`
4. `chore(env): document modal lite env variables`

---

## 14. 一句话结论

**PR50 先把 Lite 分析接成：前端统一打自己的 `/api/lite/*`，proxy 默认转 Modal Lite，失败再按开关回退 Render。**

这样最稳，也最容易在 PR 页面验证。