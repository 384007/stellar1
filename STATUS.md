# STELLAR AI 项目状态文档
**最后更新：2026-03-20**

---

## 项目地址
- **前端（Cloudflare Pages）**：https://stellar1.pages.dev
- **后端（Render）**：https://stellar1-backend.onrender.com
- **GitHub 仓库**：https://github.com/dytsui/stellar1
- **本地代码**：/Users/mac/Downloads/stellar-ai/

---

## 当前状态

### ✅ 已经工作正常
1. **前端普通分析**（Cloudflare → Gemini 直接调用）
   - 图片上传：✓ 正常（2-3秒）
   - 视频上传：✓ 正常（用 Gemini Resumable File API，15-20秒）
   - 模型：gemini-2.5-flash-lite
   - API Key 存在 Cloudflare Secret（不在代码里）

2. **前端 Pro 分析**（Cloudflare → Render 后端）
   - HTTP 200 返回
   - MediaPipe 骨架检测正常（15-16帧）
   - 关键帧提取正常（6帧：准备→后摆→顶点→下杆→击球→收杆）
   - 弹道预测正常
   - ✅ Gemini AI 评分已修复（新 key 已验证 PASS）

3. **用户认证**
   - 游客登录：✓ 正常
   - D1 数据库：已配置
   - JWT token：正常生成

4. **摄像头实拍模式**
   - MediaPipe 骨架检测：已修复（CDN 加载）
   - 8 个关键部位卡片：已修复（useState 替代 useRef）
   - Pro 参考角度：TPI 权威数据已更新
   - 未检测到的部位显示 0°

5. **后端健康**
   - 5/5 路由加载成功
   - /health 返回正常
   - Gemini API key 已验证通过

### ✅ 已解决的问题
**API Key 泄露事件（已修复）**
- 原因：commit `151e165` 曾将 GEMINI_API_KEY 明文写入 `wrangler.toml` 并推送到公开 GitHub 仓库
- 后果：key 被自动扫描 bot 盗用，配额耗尽（429 错误）
- 修复：旧 key 已撤销，新 key（来自全新 Google Cloud 项目）仅存放在 Cloudflare Secret 和 Render Environment 中
- `/debug/gemini` 调试端点已删除

---

## 架构说明

### 分析流程
```
普通分析: 用户 → Cloudflare Edge Function → Gemini File API → 返回结果
Pro分析:  用户 → Cloudflare Edge Function → Render后端 → MediaPipe骨架+Gemini → 返回结果
```

### 关键文件
```
frontend/app/api/analyze/route.ts    — 分析 API 路由（普通=Gemini直接，Pro=后端）
frontend/app/api/auth/route.ts       — 认证（D1 数据库）
frontend/components/ScreenModeCapture.tsx — 摄像头+骨架实时检测
frontend/wrangler.toml               — Cloudflare 配置（无 GEMINI_API_KEY）
backend/main.py                      — FastAPI 入口，安全启动
backend/services/gemini_service.py   — Gemini 调用（懒加载）
backend/render.yaml                  — Render 部署配置（rootDir: backend）
```

### 环境变量
**Cloudflare（Secret）：**
- `GEMINI_API_KEY` = 新 key（AIzaSyA5 开头，仅存于 Dashboard）

**Cloudflare（vars）：**
- `GEMINI_MODEL` = gemini-2.5-flash-lite
- `NEXT_PUBLIC_BACKEND_URL` = https://stellar1-backend.onrender.com
- `JWT_SECRET` = ...
- `ENVIRONMENT` = production

**Render：**
- `GEMINI_API_KEY` = 新 key（AIzaSyA5 开头，仅存于 Dashboard）
- `GEMINI_MODEL` = gemini-2.5-flash-lite
- `JWT_SECRET` = 已设置
- `FRONTEND_URL` = https://stellar1.pages.dev
- `PYTHON_VERSION` = 3.11.11（重要！mediapipe 0.10.21 不支持 3.12）

---

## 安全注意事项

### 绝对不要做的事
1. **永远不要把 API key 写进代码/配置文件然后提交到 Git**
2. Key 只能存放在：Cloudflare Dashboard Secret / Render Dashboard Environment
3. 如果怀疑 key 泄露，立刻在 Google AI Studio 撤销旧 key，并从全新项目生成新 key

### 历史泄露记录
- commit `151e165`：`AIzaSyBIo...` 曾明文写入 `wrangler.toml`（已撤销）
- Git 历史中仍有记录，但 key 已失效

---

## 已完成的重要修复历史
- **API Key 泄露修复**：发现 key 在 git 历史中暴露，换新项目新 key
- Port scan timeout：改用 _safe_load 安全启动 + rootDir: backend
- 前端视频上传 "metadata too large"：改用 Gemini Resumable File API
- 前端摄像头卡片不更新：cardsRef → useState + 200ms 节流
- 死亡降级模型：删除 gemini-2.0-flash/lite 降级
- numpy 依赖冲突：numpy<2（mediapipe 0.10.21 要求）
- D1 用户表缺 username 列：已创建 migration

---

## 给下一个 AI 的指令

用户叫 mac，项目在 /Users/mac/Downloads/stellar-ai/

**当前状态**：所有功能已正常工作，无已知 bug。

如果用户反馈 429 错误：
1. 先检查是否是配额自然耗尽（免费版有每日限额）
2. 如果是泄露，用 `git log --all -p -S "AIzaSy"` 检查 git 历史
3. 撤销旧 key，从全新 Google Cloud 项目生成新 key
4. 更新 Cloudflare Secret + Render Environment，重新部署
