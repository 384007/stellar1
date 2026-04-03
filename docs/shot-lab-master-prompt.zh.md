# Shot Lab（击球实验室）— 整合主提示词（中文版）

> **用途**：可作为 **产品 + 技术** 主规格供人阅读或与英文版对照。  
> **给 Opus / 编码模型**：请只用 **唯一合并提示词** `docs/shot-lab-opus-prompt.zh.md`（Cursor：`@docs/shot-lab-opus-prompt.zh.md`），勿再以本文档作为独立任务源。  
> **仓库**：`stellar-ai`，主应用目录 **`frontend/`**（Next.js，部分 API 为 Edge / Cloudflare；已有 `frontend/app/api/analyze/route.ts`、`frontend/lib/d1.ts`、`frontend/lib/r2.ts`、`frontend/lib/capture-quality.ts`、`frontend/lib/pose-filters.ts`、`frontend/app/history` 等）。  
> **实施原则**：**patch / targeted fix**；**禁止整包推翻**；**禁止破坏** 现有 `/analyze`、`/plus`、`/pro`、`/history` 等主链路；持久化 **D1 + R2**，不新增第二套数据库栈。

---

## 角色指令（给 AI 执行者）

你是本仓库高级全栈 + 计算机视觉工程师。目标是 **新增独立板块 Shot Lab（击球实验室）**，并做成 **核心付费板块之一**；**现有「分析」流程（如 `/analyze`、`/api/analyze` 及相关页面）必须保留，不得改名或迁移为 Shot Lab 的唯一入口。Shot Lab 须有 **独立路由、独立 API 命名空间（建议 `/api/lab`）、独立历史/配额**（或与用户表关联但字段隔离），仅可 **复用** `frontend/lib/*` 等公共能力。同时实现 **手机仅依赖摄像头/麦克风/（可选）陀螺仪与加速度计** 的真实可运行分析管线（禁止写死假数据、禁止把估算伪装成雷达测量）。

交付物必须 **可开发、可测试、可运营**：含数据模型、API、前后端权限校验、UI 分层、配额与订阅逻辑、日志与降级，而不是营销空话。

---

## 一、Shot Lab 产品定义

### 1.1 定位

**Shot Lab（击球实验室）** 是与 **现有「分析」并列的新板块**：主打 **手机视觉 launch monitor 类** 深度指标、分层付费与实验室历史；**不替代** 当前分析页的产品与路由。

**与「分析」的关系（必须遵守）：**

- **保留**：既有分析入口、文案、API 行为与数据语义不变（除非另有独立 PR）。  
- **新增**：Shot Lab 自有页面（如 `/lab` 或 `/shot-lab`）、自有任务与结果模型。  
- **复用**：拍摄质量、存储、鉴权等 **库级代码** 可共享；**业务 JSON schema 与历史表** 与旧分析 **分离**，避免混用同一 `type` 导致前端误解析。

用户在此可以：

- 上传或录制挥杆视频  
- 查看 AI 分析结果  
- 查看球速、节奏、起飞参数、轨迹、动作问题  
- 获取训练建议  
- 查看历史报告与对比（权限分层）  

### 1.2 体验模型：可体验 + 可升级

必须做成 **双层体系**：

- **Free**：快速感受价值，但 **不释放全部核心价值**，自然引导升级 **Pro**  
- **Pro**：完整专业面板，差异 **非常明显**，通过 **高级模块解锁** 体现，而非廉价弹窗轰炸  

### 1.3 硬约束（与下列技术约束同时满足，取更严）

1. **不依赖任何额外硬件**  
2. **不依赖** 雷达、外接传感器、手表、IMU、三脚架专用设备  
3. **仅允许** 普通智能手机 **摄像头、麦克风**；**陀螺仪/加速度计** 仅在系统/WebView/原生壳 **实际可用** 时使用，否则 **降级**  
4. **优先 iPhone**，架构 **兼容 Android**  
5. **真实可运行产品**：非 demo；指标须有 **计算链、置信度、失败原因**；禁止胡编数值当「测量」  
6. **基于现有仓库 patch**，禁止推倒重写导致功能丢失  

### 1.4 诚实边界（必须在 UI 与 API 字段中体现）

- **Measured-like**：来自视频/音频/几何/时序的直接估计  
- **Estimated / Inferred**：来自模型或经验物理混合；**必须标注**  
- **第一版不承诺**：军规雷达精度；等同 TrackMan/GCQuad；全环境全机型一致高精度；无充分验证的精准总旋/侧旋/spin axis  

---

## 二、用户分层（Free / Pro）

### A. 普通用户（Free）

**目标**：快速感受价值；保留核心付费动机。

**建议权限（可微调数值，但须在规格中写死默认值与可配置项）：**

1. **每日分析次数限制**：默认 **每天 3 次**（服务端强制）  
2. **基础视频分析**（Shot Lab **自有** 分析管线或任务流，但 Free  tier **输出裁剪/模块锁定**）  
3. **基础指标可见**：Ball speed、Launch angle、Launch direction、Tempo、**基础** shot tracer（可降采样/短时）  
4. **基础 AI 总结**（短摘要，非完整报告）  
5. **历史**：仅 **最近 7 天** 或 **最近 10 条**（取更严或产品可配置，须服务端执行）  
6. **不开放**：高级对比  
7. **不开放**：完整问题库（仅 Top 3 或子集）  
8. **不开放**：高级导出  
9. **不开放**：完整 drill 库（部分预览 + 锁定）  
10. 分析结果页 **Pro 升级引导**（软提示为主）  

### B. Pro 用户（Pro）

**目标**：高级、完整、专业感知强。

**建议权限：**

1. 分析次数 **不限** 或 **极高上限**（服务端配置）  
2. **完整分析指标**（含 backswing/downswing/top pause、轨迹 overlay 等按 MVP 定义）  
3. **完整 AI 报告**（结构化长报告 + 与指标一致）  
4. **完整动作问题识别**（全量 issue 列表）  
5. **历史长期保存**（明确保留策略，符合隐私与成本）  
6. **挥杆对比**  
7. **趋势分析**  
8. **多 session 汇总**  
9. **完整 drill 推荐**  
10. **导出 / 分享**  
11. **Pro 专属 UI 标识**（徽章、模块样式，克制高级）  
12. 预留：**simulator / club history / personalized baseline**（可二期）  

---

## 三、Shot Lab 页面结构（含 Free / Pro 展示差异）

### 3.1 必须包含的模块

1. **顶部标题**：主标题 **Shot Lab**，副标题 **击球实验室**  
2. **上传/录制入口**  
3. **最近分析卡片**  
4. **当前分析结果**  
5. **指标区**（Free 部分可见 + 锁定预览；Pro 全量）  
6. **轨迹可视化区**（Free 基础 tracer；Pro 完整 tracer + club/hand overlay 若可用）  
7. **AI 诊断区**（Free 短摘要；Pro 完整报告）  
8. **Drill 建议区**（Free 预览；Pro 全库/多组）  
9. **历史记录区**（Free 7 天/10 条；Pro 长期）  
10. **升级 Pro 区块**（内联模块，非唯一打断手段）  

### 3.2 UX 原则

- Free 用户界面 **不能显得残缺或廉价**：用 **「预览 / 解锁后可见」** 的高级模块卡片，而非大块空白  
- Pro 差异通过 **高级模块解锁、更深数据维度、更长历史** 体现  
- 弹窗仅用于 **关键门槛**（如次数用尽）；其余用 **内联 CTA、模块脚标、轻量 badge**  

---

## 四、功能分层表（必须输出为工程师可实现的表格）

| 功能名称 | Free | Pro | 展示方式 | 升级触发点 | 分层理由 |
|---------|------|-----|----------|------------|----------|
| 单次分析 | ✓（受限次数） | ✓（不限或高上限） | 同一入口；超额拦截 | 次数用尽 / 高级模块点击 | 核心钩子 |
| 每日分析次数 | 默认 3 | 不限/高上限 | 顶栏/设置显示剩余 | 用尽时 | 自然转化 |
| Ball speed | ✓ 基础展示 | ✓ 全量+置信度详情 | 指标卡 | 查看不确定性分解（可选 Pro） | 保留深度给 Pro |
| Launch angle | ✓ | ✓ | 指标卡 | 高阶可视化 | 基础信任建立 |
| Launch direction | ✓ | ✓ | 指标卡 | 对比视图 | 同上 |
| Tempo | ✓ | ✓ | 指标卡 | 趋势需 Pro | Free 体验节奏价值 |
| Backswing / downswing / top pause | 锁定或模糊 | ✓ 完整 | 时间轴模块 | 展开时间轴 | 深度专业度 |
| Shot tracer | ✓ 基础 | ✓ 完整 | 视频叠加层 | 更长轨迹/多段 | 视觉冲击力分层 |
| Club / hand trajectory | 预览/低帧 | ✓ 完整 | overlay | 点击解锁 | 计算成本高 |
| Weight shift / shoulder / hip 等问题 | 仅 Top 3 子集 | ✓ 全量 | 问题列表 | 「还有 N 项」 | 教练价值分层 |
| Top 3 issue | ✓ | ✓ | 诊断区 | — | Free 也需诊断感 |
| Full issue list | ✗ | ✓ | 可滚动列表 | 展开全部 | Pro 核心 |
| AI summary | ✓ 短 | ✓ 长 | 文本区 | 「完整报告」 | 内容深度 |
| Full AI report | ✗ | ✓ | 结构化报告页 | 生成完整报告 | Pro 核心 |
| Drill recommendation | ✓ 1–2 条 | ✓ 完整库/多组 | drill 卡片 | 更多 drill | 训练闭环 |
| History retention | 7 天/10 条 | 长期 | 历史列表灰显/锁定 | 打开旧记录 | 数据资产 |
| Compare swings | ✗ | ✓ | 对比页 | 选两条对比 | 高阶功能 |
| Trend analytics | ✗ | ✓ | 图表 dashboard | 趋势 Tab | 长期价值 |
| Export/share | ✗ | ✓ | 按钮 | 导出 | 专业用户刚需 |
| Coach mode / advanced report | ✗ | ✓ | 模式切换 | Pro 徽章 | 品牌区分 |

---

## 五、权限与计费逻辑（工程必须落地）

### 5.1 数据字段（建议）

**用户 / 配置（D1 或现有用户表扩展，禁止破坏现有字段语义）：**

- `user_id`  
- `plan`：`free` | `pro`  
- `subscription_status`：`active` | `canceled` | `past_due` | `expired`  
- `pro_expires_at`（若适用）  
- `entitlements_version`（规则版本，便于迁移）  

**用量（须服务端权威）：**

- `usage_daily_analysis_count`  
- `usage_daily_analysis_date`（按用户时区或 UTC，须统一文档）  
- `usage_lifetime_analysis_count`（可选）  

### 5.2 规则

1. **每日重置**：按 `usage_daily_analysis_date` 与当前日期比较重置计数  
2. **扣减**：分析任务 **接受/开始处理** 时扣减（定义幂等 `analysis_id`，防止重复扣）  
3. **Pro 到期**：定时或请求时校验 → `plan=free`，**不删除** Pro 期历史但 **锁定访问规则**（或按产品：仅锁新功能，旧 Pro 报告只读——须明确一种策略并写清）  
4. **历史保留**：Free 在服务端 **查询时裁剪** + **定期清理任务**（若需）；禁止仅靠前端隐藏  
5. **API 校验**：**每个** 创建分析、拉取完整报告、对比、趋势、导出接口 **必须** 校验 `plan` + `quota`  
6. **前端校验**：仅用于 UX；**绝不能**作为唯一防线  

### 5.3 防绕过

- 高级结果字段 **服务端裁剪**：Free 请求返回 `report_tier: "free"` 与 **截断字段**  
- 敏感端点 **拒绝** 或返回 **402/403** + 统一错误码（如 `PRO_REQUIRED` / `QUOTA_EXCEEDED`）  
- **同一分析任务** 禁止用不同端点拼凑完整 Pro 字段  

---

## 六、升级转化设计（Shot Lab 内）

1. **强引导时机**：每日次数用尽；点击对比/趋势/导出；请求完整 issue 列表；打开超窗历史  
2. **软提示时机**：完成分析后底部「深入训练计划」；指标区「置信度与分解」脚标；drill 区「还有 4 项建议」  
3. **预览后锁定**：完整时间轴、全量 issue、完整报告前 30% 可见  
4. **直接锁定**：导出、对比、趋势、超窗历史  
5. **文案风格**：克制、专业、训练工具感；避免尖叫式促销  
6. **升级理由主轴**：更深分析、更长历史、更专业训练闭环  

---

## 七、文案风格与示例（中文）

**风格**：高级、专业、克制；像 **高端高尔夫训练工具**，不像泛健身 App。

1. **标题**：Shot Lab  
2. **副标题（Free）**：用手机完成每一杆的专业级洞察。  
3. **副标题（Pro）**：完整数据、长期历史与深度训练计划，为认真练习者而设。  
4. **次数用尽**：今日免费分析次数已用完。明天再来，或升级 Pro 继续练习。  
5. **升级 CTA**：解锁 Pro：完整报告与长期进步曲线。  
6. **功能锁定卡片**：此模块包含完整动作诊断与训练序列。升级 Pro 查看全部。  
7. **历史锁定**：更早的记录已归档至 Pro 历史库。  
8. **高级对比锁定**：并排对比与差异分析为 Pro 功能，用于追踪技术演变。  

（英文版见 `docs/shot-lab-master-prompt.en.md`。）

---

## 八、技术：MVP / 二期 / 不承诺

### MVP（必做，真实可用 + 置信度 + 降级）

1. Ball speed（estimate）  
2. Vertical launch angle  
3. Horizontal launch direction  
4. Backswing time  
5. Downswing time  
6. Tempo  
7. Swing sequence segmentation（address → finish）  
8. Club/hand trajectory overlay（不可用则降级）  
9. Shot tracer  
10. Carry distance **estimate**（强制标注 estimate）  
11. Contact / strike quality score（禁止伪造 smash factor）  
12. Session / club history / trend（**Pro 全量；Free 受限**）  
13. AI text report（分层）  
14. Drill recommendation（分层）  

### 二期

Spin rate/axis estimate、AoA estimate、club path refinement、face angle、室内网模式、模拟器导出、多球种校准、个性化 baseline、多机位 fusion。

### 系统架构（文字版，须实现数据流）

Mobile/Web（Shot Lab）→ 上传/录制 pipeline → 视频预处理 → 挥杆事件检测 → 2D 姿态 → 手/杆跟踪 → 球检测与起飞提取 → 指标引擎 → 错误模式检测 → AI 报告层 → 历史存储（D1/R2）→ 校准/profile 层。

### 核心算法（落地要求）

- **击球时刻**：音频峰值 + 姿态突变 + 杆头高速 + 球运动，融合得 impact frame 与 confidence  
- **球与起飞**：静止球、帧差、模糊条纹、RANSAC/直线拟合、透视与标定、帧率与音频对齐  
- **指标表**：每个指标需 **定义、输入、公式/几何、异常、confidence**（禁止固定假表当测量）  

### 动作识别（至少 10 类）

Weight shift insufficient；Shoulder up；Hip slide；Reverse pivot；Early extension；OTT；Casting；Chicken wing；Head lift；Finish imbalance — 每项含 **关键点、逻辑、阈值来源、confidence、解释、drill**。

---

## 九、数据模型与 API（概要，实施须扩展为 OpenAPI 级）

- **表**：`lab_sessions`、`lab_shots`、`lab_metrics`、`lab_reports`、`lab_usage_daily`、`user_subscription`（或扩展现表）  
- **API 示例**：`POST /api/lab/analyze`（创建任务）、`GET /api/lab/jobs/:id`、`GET /api/lab/history`、`GET /api/lab/compare`（Pro）、`POST /api/lab/export`（Pro）  
- **响应**：统一 `tier: free|pro`、`fields_visibility`、`quota`  

---

## 十、分阶段开发

### 第一阶段（MVP）

- **新增** Shot Lab 路由与导航入口（与现有「分析」并列；**不**把旧分析重命名成 Shot Lab）  
- Free/Pro **服务端**配额与响应裁剪（仅针对 **lab** 端点与 lab 历史）  
- 主链路：上传/录制 → 分析 → 指标 → 报告 → 历史（Free 限制）  
- 日志、超时、重试、特性开关（关闭时 **仅隐藏/停用 Shot Lab**，**不影响** 原有 `/analyze`）  

### 第二阶段

趋势、对比、导出、完整 drill 库、校准与个性化、更多 estimate 指标。

---

## 十一、第一阶段应触碰的文件清单（示例，实施时以仓库为准）

- `frontend/app/lab/**` 或 `frontend/app/shot-lab/**`（**新建**页面目录，二选一，全仓一致即可）  
- `frontend/app/api/lab/**`（任务与权限）  
- `frontend/lib/**`（配额、entitlement、D1 schema 迁移）  
- 导航入口：`frontend/app/layout.tsx`、`frontend/app/page.tsx` 等（最小改动）  
- 复用：`capture-quality.ts`、`pose-filters.ts`、`video-store.ts`、`fetch-retry.ts`  
- **避免** 在未授权下扩大 `frontend/app/api/analyze/route.ts` 的「假预测」字段语义；新能力走 **lab** 命名空间  

---

## 十二、测试入口与回滚

- **测试**：E2E 覆盖 Free 超额、Pro 全量字段、历史裁剪；`curl`/集成测试覆盖 API 403/402  
- **回滚**：特性开关禁用 Shot Lab 新 API 与入口；用户仍使用 **原有分析页**；数据库迁移可逆  

---

## 十三、自检清单

- [ ] Shot Lab 为 **新增独立板块**；与既有「分析」并存；lab 为核心付费场景之一  
- [ ] Free/Pro **后端强制**，不可前端绕过  
- [ ] 手机-only 硬约束满足  
- [ ] 无写死假数据；estimate 已标注  
- [ ] 不破坏现有功能；patch 交付  
- [ ] D1+R2；无新外部 DB 栈  
- [ ] 主链路可运行；文档含测试与回滚  

---

## 十四、工程落地输出（严格按此顺序交付文档/PR）

1. **Shot Lab 产品定义**（见第一节 + 技术诚实边界）  
2. **Free / Pro 功能分层表**（见第四节表格，实施时附 CSV/Notion 同步）  
3. **页面结构图（文字版）**（见第三节）  
4. **数据模型设计**（用户/订阅/用量/会话/击球/指标/报告/迁移策略）  
5. **API 权限设计**（端点列表、错误码、`tier` 与字段裁剪规则、幂等）  
6. **分析次数与订阅逻辑**（日重置、扣减时机、Pro 到期、历史保留）  
7. **升级转化设计**（见第六节 + 模块级触发点表）  
8. **UI 文案示例**（见第七节）  
9. **MVP 第一阶段应开发内容**（见第十节 + 第十一节文件清单）  
10. **第二阶段增强内容**（见第八节二期 + 第十节第二阶段）  

---

**结束**。将本文件全文复制即可作为单一主提示词使用。
