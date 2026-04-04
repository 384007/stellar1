# Pro v2 Screen Mode — Full Flow Codex PR Prompt

你现在在 `dytsui/stellar1` 仓库里，直接完成 **pro_v2 的 Screen Mode 全流程改造**，并提交 PR 到 `main`。

不要只写方案，不要只改 prompt，不要 demo，不要假数据，不要伪逻辑，不要只写 README。  
要直接改真实代码，保留现有项目结构，优先精准修改，不要大爆炸重构，不要破坏 plus 和旧 `/analyze/pro`。

---

## 一、最终产品规则（必须实现）

这 8 条是硬规则，禁止违背：

1. AI先看视频，输出后端处理策略
2. 后端按策略提取8张关键帧
3. AI对6张核心关键帧逐张评分
4. 若任一核心关键帧评分 < 90，则后端按失败原因重做
5. 若两轮后仍未达到核心关键帧90分标准，则标记为 `low_trust`
6. 前端显示关键帧评分与 `low_trust` 标记
7. 只有非 `low_trust` 状态，AI 才能输出正式动作报告
8. 若为 `low_trust`，AI 只能输出低信任报告，并明确注明“关键帧不符，结论受限”

---

## 二、Screen Mode 业务目标

本次只先做 **Screen Mode 板块**，但要做完整链路，不是半成品。

Screen Mode 指：
- 拍摄显示器/投影/平板/手机屏幕上的挥杆视频
- 翻拍屏幕的视频
- 非正常真机直拍视频

目标：
- 用户在前端明确进入 Screen Mode
- 请求真实传 `screen_mode=true`
- 后端先做 screen preprocess，再统一做 240fps 分析
- screen mode 下的关键帧不能只信任 motion-only
- 必须接入 AI 路由 + AI 核心关键帧评分 + retry + low_trust + formal/limited report 分流

---

## 三、关键帧定义

最终仍输出 8 张关键帧：

1. `address`
2. `takeaway`
3. `backswing_mid`
4. `top`
5. `early_downswing`
6. `impact`
7. `release`
8. `finish`

其中：
- `address` 和 `finish` 是固定参考帧，不参与 90 分硬门槛
- 核心 6 张为：
  - `takeaway`
  - `backswing_mid`
  - `top`
  - `early_downswing`
  - `impact`
  - `release`

两轮后只要这 6 张里任一张 `< 90`，就必须 `low_trust`。

---

## 四、当前仓库现状（基于现有代码演进，不要重写）

当前仓库已经有：
- `backend/routers/pro_v2_api.py`
- `backend/services/pro_v2_video_analysis_service.py`
- `backend/services/pro_v2_screen_preprocess_service.py`
- `backend/services/pro_v2_keyframe_picker_service.py`
- `backend/services/pro_v2_simple_gate_service.py`
- `backend/services/pro_v2_report_service.py`
- `backend/services/gemini_service.py`
- 前端已有 `pro_v2` 相关上传/结果展示代码
- 当前已有 `screen_mode` 参数和 `screen preprocess` 分支
- 当前 `pro_v2` 已走 `240fps + motion-first` 方向

但是当前还缺：
- AI首轮视频路由
- 6张核心关键帧逐张评分
- 两轮重做
- 两轮失败后 `low_trust`
- `formal / limited report` 分流
- 前端关键帧评分与 `low_trust` 展示

你要在现有结构上补齐，不要推翻全部重写。

---

## 五、必须修改 / 新增的文件

优先修改这些文件：

### 后端
- `backend/routers/pro_v2_api.py`
- `backend/services/pro_v2_video_analysis_service.py`
- `backend/services/pro_v2_screen_preprocess_service.py`
- `backend/services/pro_v2_keyframe_picker_service.py`
- `backend/services/pro_v2_simple_gate_service.py`
- `backend/services/pro_v2_report_service.py`
- `backend/services/gemini_service.py`

### 必要时新增
- `backend/services/pro_v2_ai_routing_service.py`
- `backend/services/pro_v2_keyframe_review_service.py`

### 前端
- `frontend/lib/pro-v2-analyze-client.ts`
- `frontend/app/pro/page.tsx`
- `frontend/app/analyze/page.tsx`
- `pro_v2` 结果页/详情页相关组件
- 与 `keyframe score / low_trust / report_mode` 展示有关的组件

---

## 六、完整目标链路（必须按这个实现）

### 阶段 0：前端进入 Screen Mode
前端要有明确 Screen Mode 板块 / 入口 / 开关：
- 用户选中 Screen Mode
- 上传请求必须真实带 `screen_mode=true`
- 上传中和结果页必须明确显示当前是 Screen Mode

### 阶段 1：AI 首轮视频路由
新增 AI 视频路由能力。  
AI 第一轮不是写报告，不是直接选帧，而是先看视频并输出后端处理策略。

允许方式：
- 看原视频抽样帧
- 看联系图 / storyboard
- 看低成本预处理样本
- 不要求整段视频重型模型推理，但要足以做 routing

AI 路由输出至少要包含：

```json
{
  "screen_mode_confirmed": true,
  "recommended_pipeline": "screen_mode_pipeline",
  "quality_level": "high|medium|low",
  "use_deblur": true,
  "use_heavy_club_tracking": true,
  "pose_priority": false,
  "expected_confidence_ceiling": 0.84
}
```

后端必须能读懂这个结构化结果，不要依赖自然语言建议。

### 阶段 2：Screen preprocess + 240fps 分析
如果 `screen_mode=true`：
1. 先跑 `screen preprocess`
2. 尽量识别屏幕区域
3. 裁剪掉明显边框/UI/字幕干扰
4. 再统一生成 240fps 分析视频
5. 后续 `swing window / dense scan / keyframe pick` 基于该分析视频执行

要求：
- 不能只把 `screen_mode` 当成前端开关
- 必须真实影响后端链路
- 保留原有普通模式逻辑，不要破坏

### 阶段 3：后端提取 8 张关键帧
后端按 `routing strategy` 产出 8 张关键帧初稿：
- `address`
- `takeaway`
- `backswing_mid`
- `top`
- `early_downswing`
- `impact`
- `release`
- `finish`

如果现有 keyframe picker 的阶段命名不匹配，请在不破坏兼容的前提下做映射或升级：
- `backswing -> backswing_mid`
- `downswing -> early_downswing`
- `follow_through -> release`

`address / finish` 保留为固定参考帧。

### 阶段 4：AI 审核 6 张核心关键帧
新增 AI 核心关键帧审核服务：
- `takeaway`
- `backswing_mid`
- `top`
- `early_downswing`
- `impact`
- `release`

AI 对这 6 张逐张输出：
- `score`
- `pass_90`
- `confidence`
- `reason_codes / retry_reasons`

输出格式至少要像：

```json
{
  "review_round": 1,
  "core_frame_scores": {
    "takeaway": {"score": 92, "pass_90": true, "confidence": 0.93},
    "backswing_mid": {"score": 88, "pass_90": false, "confidence": 0.79},
    "top": {"score": 95, "pass_90": true, "confidence": 0.95},
    "early_downswing": {"score": 90, "pass_90": true, "confidence": 0.91},
    "impact": {"score": 96, "pass_90": true, "confidence": 0.97},
    "release": {"score": 91, "pass_90": true, "confidence": 0.90}
  },
  "retry_required": true,
  "retry_reasons": ["BACKSWING_MID_BELOW_90"]
}
```

### 阶段 5：失败重做（第2轮）
若任一核心关键帧 `< 90`：
- 后端必须按 `retry_reasons` 重做
- 不能机械重复跑同一逻辑
- 要根据失败原因调整：
  - `top` 重选
  - `impact` 重选
  - `release` 重选
  - `screen preprocess` 参数微调
  - `screen ROI/motion` 权重调整
  - `late-strip spacing` 调整
  - `club tracking` 权重调整

重做后再跑第二轮 AI 核心关键帧审核。

### 阶段 6：trust 决策
若第二轮后所有核心关键帧都 `>= 90`：
- `analysis_trust = "high_trust"`
- `report_mode = "formal"`

若第二轮后仍有任一核心关键帧 `< 90`：
- `analysis_trust = "low_trust"`
- `report_mode = "limited"`
- `keyframe_mismatch_notice = true`

### 阶段 7：报告分流
`high_trust`：
- 才允许正式动作报告
- `formal report`

`low_trust`：
- 只能 `limited report`
- 必须明确注明：
  **“关键帧不符，结论受限”**
- 禁止写得像高置信正式报告

---

## 七、后端 API / 返回字段要求

对前端公开 API 仍尽量保持少而干净：
- `POST /pro-v2/analyze`
- `GET /pro-v2/media/{analysis_id}/{filename}`
- 如果现有前端已使用其它取结果方式，保持兼容

但最终 `POST /pro-v2/analyze` 返回结果里，必须新增/补齐这些字段（命名可微调，但语义不能丢）：

```json
{
  "analysis_id": "...",
  "status": "completed",
  "pro_v2_screen_pipeline": true,
  "screen_mode": true,
  "screen_cropped_video_url": "...",
  "playback_video_url": "...",
  "analysis_trust": "high_trust | low_trust",
  "report_mode": "formal | limited",
  "review_round": 1,
  "keyframe_mismatch_notice": false,
  "core_frame_scores": {
    "takeaway": {"score": 0, "pass_90": false, "confidence": 0},
    "backswing_mid": {"score": 0, "pass_90": false, "confidence": 0},
    "top": {"score": 0, "pass_90": false, "confidence": 0},
    "early_downswing": {"score": 0, "pass_90": false, "confidence": 0},
    "impact": {"score": 0, "pass_90": false, "confidence": 0},
    "release": {"score": 0, "pass_90": false, "confidence": 0}
  },
  "retry_required": false,
  "retry_reasons": [],
  "keyframes": [],
  "summary": "...",
  "summary_zh": "...",
  "warning": "关键帧不符，结论受限"
}
```

---

## 八、前端必须完成的显示

前端结果页必须显示：

1. 当前是否为 Screen Mode
2. 若有 `screen_cropped_video_url`，显示可播放入口
3. 6 张核心关键帧逐张评分
4. `low_trust` 标记
5. `report_mode`（`formal / limited`）
6. 若 `low_trust`：
   - 明确显示“关键帧不符，结论受限”
   - 不要把报告视觉表现做得像高信任正式报告

要求：
- 前端不能只显示 `total_score`
- 必须真实展示 `core_frame_scores`
- 必须真实展示 `low_trust / formal / limited` 差异

---

## 九、Prompt / AI 逻辑修改要求

你必须补齐两类 AI prompt / service：

### A. 视频路由 prompt
AI 先看视频，输出结构化 routing 策略：
- `recommended_pipeline`
- `quality_level`
- `use_deblur`
- `use_heavy_club_tracking`
- `pose_priority`
- `expected_confidence_ceiling`
- `screen_mode_confirmed`

### B. 核心关键帧审核 prompt
AI 对 6 张核心关键帧逐张打分：
- `0-100`
- `pass_90`
- `confidence`
- `retry reasons`

### C. 报告 prompt 分流
必须有两套：
- `formal report prompt`
- `limited report prompt`

`limited report prompt` 必须强约束：
- 明确写出关键帧未通过高信任校验
- 结论受限
- 不得假装关键帧准确
- 不得写成高置信正式报告

---

## 十、日志要求

请增加并统一日志，方便在 modal / backend 排查：

- `[PRO_V2][SCREEN]`
- `[PRO_V2][ROUTE]`
- `[PRO_V2][KF_REVIEW]`
- `[PRO_V2][RETRY]`
- `[PRO_V2][LOW_TRUST]`
- `[PRO_V2][REPORT_MODE]`

日志里至少要能看见：
- `screen_mode=true/false`
- `analysis_input=raw|screen_cropped`
- `review_round=1|2`
- `retry_required=true/false`
- `analysis_trust=high_trust|low_trust`
- `report_mode=formal|limited`

---

## 十一、禁止事项

1. 不要回退到旧 `/analyze/pro`
2. 不要破坏 plus 和其它旧功能
3. 不要只改 prompt，不改后端状态机
4. 不要只改后端，不改前端显示
5. 不要只做 `low_trust` 文案，必须有真实字段和真实分流逻辑
6. 不要把 AI 评分写死
7. 不要只在日志里标记，要把结果真正返回前端
8. 不要做伪“通过”，两轮后 `< 90` 就必须 `low_trust`

---

## 十二、验收标准

完成后必须满足：

1. 前端能明确进入 Screen Mode
2. 请求里真的传 `screen_mode=true`
3. 后端 `screen preprocess` 真正参与链路
4. 返回结果里有 `analysis_trust / report_mode / core_frame_scores`
5. 两轮失败后会真实出现 `low_trust`
6. 前端能显示关键帧评分和 `low_trust` 标记
7. `low_trust` 时只能 `limited report`，并显示“关键帧不符，结论受限”
8. `high_trust` 时才 `formal report`

---

## 十三、交付要求

直接修改代码并提交 PR。  
在 PR 描述里写清楚：

- 改了哪些文件
- 每个文件做了什么
- Screen Mode 现在完整链路
- 哪些字段新增了
- 前端如何显示 `low_trust` 和关键帧评分
- 还有哪些已知限制

现在开始改代码并提交 PR，不要只写计划。
