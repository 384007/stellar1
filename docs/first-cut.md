# First Cut — 先把分析结果修准

## 目标
这次只做第一刀：**先把分析结果修准**。  
不要重写项目，不要先做新功能，不要先做大架构改造，不要先做 MMPose / DeepLabCut / Shot Lab 全家桶。

这次只修 4 件事：

1. 关键帧错误
2. 视频分析结论错误
3. AI 文案自相矛盾、置信度虚高
4. share 页可能展示旧/错结果

---

## 当前已确认的真实问题

### 问题 1：关键帧明显错误
我已经通过 HAR 抓到真实 share 返回结果，确认：

- share 返回了 8 张 keyframes
- 前 6 张高度相似
- `top` 不像顶点
- `impact` 不像击球瞬间
- 只有 `follow_through` / `finish` 比较明显不同

结论：

**不是 share 页自己选错图，而是上游分析结果本身就错。**

---

### 问题 2：视频分析本身也错
HAR 中的 plus 分析结果里：

- `primary_diagnosis.title_zh` 是问题描述
- 但 `primary_diagnosis.status_zh` 却是“做得好”
- `ai_confidence = 95`
- `prediction.hand = UNKNOWN`
- `prediction.club_type = null`
- 同时 keyframes 又明显错位

结论：

**不只是关键帧错，视频分析也错，而且报告层没有做一致性和降级。**

---

### 问题 3：plus phase 选择机制太危险
当前 plus 链路大概率是：

1. 提取 poses
2. 均匀抽 16 帧
3. 调 `detect_phases_from_frames(uniform_frames)`
4. 只要 Gemini 返回 phase index，就直接采用
5. 再映射回 pose / keyframe / AI 分析

这里的问题是：

- 没有 phase confidence
- 没有和 kinematic phase 做一致性校验
- 没有最小间距约束
- 没有 near-duplicate keyframe 去重
- 没有 top / impact 二次校验

---

### 问题 4：share API 可能拿旧结果
`frontend/app/api/share/[token]/route.ts` 当前逻辑是：

- 先读 D1 `result_json`
- 只要存在 `result_r2_key`
- 就优先用 R2 结果覆盖 D1

这会导致：
- 如果 R2 是旧结果
- D1 是新结果
- share 页仍然显示旧错误内容

---

## 这次只改这 6 个文件

1. `backend/routers/plus_analyze.py`
2. `backend/services/gemini_service.py`
3. `backend/services/keyframe_service.py`
4. `backend/services/swing_flow_utils.py`
5. `frontend/app/api/share/[token]/route.ts`
6. `frontend/app/share/[token]/page.tsx`

不要扩散到一堆无关文件。  
这次先把“结果做准”。

---

## 每个文件必须怎么修

### 1）`backend/routers/plus_analyze.py`
目标：

**不要再盲信 Gemini phase detection。**

必须做到：

- `detect_phases_from_frames()` 返回后不能直接采用
- 必须和 `detect_swing_phases(poses)` / `get_phase_keyframes(...)` 做一致性比对
- 增加 phase validation gate：
  - 顺序合法
  - 相邻 phase 间距合理
  - `top` 合理
  - `impact` 合理
- 如果 Gemini 结果可疑，自动 fallback 到 kinematic
- 输出 debug 字段：
  - `phase_source`
  - `phase_validation`
  - `phase_keyframe_debug`

---

### 2）`backend/services/gemini_service.py`
目标：

**Gemini 只能做辅助，不能再高置信瞎写。**

必须做到：

- `detect_phases_from_frames()` 增加严格校验
- phase 结果异常时返回 `None`
- 如果 phase 间距过小、顺序虽对但明显可疑，直接判失败
- 视频分析结果增加一致性检查
- 当以下任一不可靠时，必须降低 `ai_confidence`：
  - keyframes unreliable
  - phase validation failed
  - hand unknown
  - club unknown
  - tracking weak
- 禁止：
  - diagnosis title 和 status 自相矛盾
  - 低证据还给很高置信度

---

### 3）`backend/services/keyframe_service.py`
目标：

**8 张关键帧必须视觉上真的像 8 个阶段。**

必须做到：

- 增加相邻 keyframe 最小时间间距
- 增加相邻 keyframe 视觉差异阈值
- 如果相邻 phase 图像太像，必须重新选
- `top` 和 `impact` 必须单独重点校验
- 返回更多 debug 字段：
  - `source_pose_idx`
  - `source_frame_index`
  - `visual_diff_from_prev`
  - `phase_validation_passed`

---

### 4）`backend/services/swing_flow_utils.py`
目标：

**让 kinematic fallback 更可信。**

必须做到：

- 强化 `top` / `impact` / `follow_through` 的事件规则
- 提升在 Gemini 失败时的 fallback 质量
- 保证 `top` 更接近顶点
- 保证 `impact` 更接近击球瞬间

---

### 5）`frontend/app/api/share/[token]/route.ts`
目标：

**不要再盲目优先展示旧 R2 结果。**

必须做到：

- 比较 D1 `result_json` 和 R2 `result_r2_key`
- 优先选择：
  - 更新
  - 字段更完整
  - keyframes 更合理
  - 结果更可信
  的那一份
- 如果 R2 结果明显异常，就回退 D1
- 保持兼容，但不能再盲目 R2 覆盖 D1

---

### 6）`frontend/app/share/[token]/page.tsx`
目标：

**展示修复后的真实可信结果，不要包装错结果。**

必须做到：

- 如果后端返回 low reliability，前端明确提示
- 不要把低可信结果展示得像完全正确
- 保持现有页面结构，少改 UI，重点配合结果质量显示

---

## 这次必须新增的验证机制

### `phase_validation`
必须包含：

- 顺序
- 间距
- top 合理性
- impact 合理性

---

### `keyframe_validation`
必须包含：

- near-duplicate 检测
- 时间分布异常
- visually wrong 检查

---

### `analysis_reliability`
综合：

- `phase_validation`
- `keyframe_validation`
- hand detection
- club detection
- tracking quality

这个结果必须影响：

- `ai_confidence`
- `summary`
- `primary_diagnosis.status`
- share 页低可信提示

---

## HAR 文件使用要求
我已附上 HAR 文件。  
请重点检查 HAR 里的：

1. `/api/share/{token}` 响应体
2. `result_json`
3. `keyframes`
4. `phase_keyframes`
5. `primary_diagnosis`
6. `ai_confidence`
7. `result_r2_key`

不要忽略 HAR。  
这次要以 **HAR + 代码** 一起定位根因，不要只看代码猜。

---

## 验收标准

1. 8 张 keyframes 不再前 6 张几乎一样
2. `top` 明显接近挥杆顶点
3. `impact` 明显接近击球瞬间
4. 视频分析结论和关键帧一致
5. `primary_diagnosis.title_*` 和 `status_*` 不再矛盾
6. 证据不足时 `ai_confidence` 明显下降
7. share 不再优先吃旧 R2 错结果
8. share 页能显示低可信提示（如适用）

---

## 输出要求
请按这个顺序输出：

### 第 1 部分：根因分析
逐条解释：
- 关键帧为什么错
- phase 为什么错
- 视频分析为什么错
- share 为什么可能展示旧结果

### 第 2 部分：文件级修改方案
按这 6 个文件逐个写：
- 改什么
- 为什么改
- 解决什么问题

### 第 3 部分：完整 patch / 完整代码
优先直接给这 6 个文件的完整修改版代码

### 第 4 部分：验证方法
告诉我如何验证：
- keyframes 修好了
- top / impact 修好了
- 文案不再乱写
- share 不再吃旧结果

---

## 工程约束
1. targeted patch
2. 不重写项目
3. 不先做新功能
4. 第一目标是“把结果做准”
5. 如果某项做不到 100%，必须明确 fallback 和限制
6. 不要空谈，直接面向代码实施