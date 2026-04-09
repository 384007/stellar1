# PR50 Lite A-only 主路径：不允许直接失败，必须持续重选直到选出当前窗口内最合理关键帧

你现在直接在 `dytsui/stellar1` 的当前工作分支 `feat/modal-centric-backend-proxies` 上修改 **Lite A-only 主路径**。

这次不要讨论方案，不要停在解释，不要改前端，不要提 B，不要引入 B。  
你要直接改后端代码，让 Lite 关键帧更准，尤其是中间 6 个动作。  
并且这次有一个强硬要求：

> **不允许因为语义不通过就直接 fail 退出。**  
> **必须继续做局部重选，直到为每个动作选出当前窗口内最合理的帧。**

---

## 一、当前问题（你必须先理解清楚）

当前主路径是：

1. `run_lite_a_infer_only(...)`
2. `run_lite_a_gate(rows)`
3. 结束

当前 `orchestrator.py` 明确还是：

- A infer
- A gate
- no local refine

这导致：
- 中间动作一旦错位
- 当前流程容易直接 fail / low-trust
- 但不会真正“把错帧修正到更合理位置”

本次任务就是把它改成：

1. A infer 先给第一版候选关键帧
2. 对中间 6 个动作做语义检查
3. 某个动作不满足语义 → **局部重选**
4. 重选后再检查
5. 继续重选，直到该动作拿到当前窗口内最合理的帧
6. 最终必须输出完整 8 帧
7. 不允许因为中间动作第一次语义不通过就直接返回 fail

---

## 二、这次只处理 A-only 主路径

不要提 B。  
不要写 B。  
不要引入 B。  
不要说“以后再接 B”。  
不要改 heuristic B 相关任何逻辑。

本次任务只改 **A-only 主路径**。

---

## 三、中间 6 个动作必须全部做语义检查

本次重点不是 8 个全量动作，而是中间 6 个动作：

1. `Toe-up`
2. `Mid-backswing`
3. `Top`
4. `Mid-downswing`
5. `Impact`
6. `Mid-follow-through`

`Address` 和 `Finish` 继续保留边界作用即可，  
但真正必须强化的是上面这 6 个动作。

---

## 四、明确“谁来做语义检查”

这次必须把职责拆清楚，不能混乱。

### 1. A infer 只负责“提名”
A 模型只负责给第一版候选关键帧。  
它不是最终裁决者。

### 2. 新增一个模块，真正做语义检查和局部重选
你必须新建一个最小后端模块，例如：

`backend/services/lite_ab_mirror/phase_semantic.py`

这个模块负责：

- 对中间 6 个动作逐个做语义检查
- 给每个动作打语义分
- 决定哪些动作需要局部重选
- 在局部窗口里重新选帧
- 返回修正后的 rows
- 返回语义失败/重选日志

### 3. a_gate.py 只负责最后审批
`a_gate.py` 不负责做复杂语义判断本身，  
它负责接收 semantic 模块的结果，然后做最后 pass / low-trust / fail 审批。

---

## 五、这次最重要的要求：不允许直接失败，必须继续选

当前错误做法是：

- infer fail
- 或 gate fail
- 直接返回

这次不允许这样。

新的行为必须是：

### 规则 1
即使 `infer.a_status == "fail"`，  
也不能立刻退出。

必须：
- 继续拿已有候选关键帧 / 候选位置
- 进入 `phase_semantic.py`
- 对中间 6 动作逐个做语义修正与局部重选

### 规则 2
即使第一次语义检查不通过，
也不能立刻 fail。

必须：
- 在该动作所属的局部窗口继续重选
- 直到选出当前窗口内最合理的一帧

### 规则 3
最终必须输出完整 8 帧  
不允许：
- 缺帧
- 直接空返回
- 因为中间动作语义第一次不过就直接终止

### 规则 4
允许 low-trust，但不允许不选
也就是说：
- 可以输出 `trust_low`
- 但不能因为难例就不选关键帧
- 必须选出“当前窗口内最合理”的帧，再把 trust 降低

一句话：
> **不允许“我不会，所以失败”；必须“我继续搜，直到选出当前最合理帧”。**

---

## 六、局部重选必须这样做（不是整条 A 重跑）

注意：
- 不是整条 A 从头重跑很多次
- 不是重复完整模型推理
- 只是对不合格动作做 **局部窗口重选**

### 1. Toe-up
如果语义不通过：
- 在 `Address → Mid-backswing` 窗口里重选

### 2. Mid-backswing
如果语义不通过：
- 在 `Toe-up → Top` 窗口里重选

### 3. Top
如果语义不通过：
- 在当前 Top 附近的小窗口里重选

### 4. Mid-downswing（最重要）
如果语义不通过：
- 在 `Top → Impact` 窗口里重选
- 不允许再用“Impact 前固定偏移帧”直接当最终下杆

### 5. Impact
如果语义不通过：
- 在 `impact hint` 附近小窗口里重选

### 6. Mid-follow-through
如果语义不通过：
- 在 `Impact → Finish` 窗口里重选

---

## 七、每个动作必须做什么语义检查

### Toe-up
至少检查：
- 已经离开 Address
- 不能还像静止准备
- 不能已经进入 Mid-backswing

### Mid-backswing
至少检查：
- 已进入上杆中段
- 不能太像 Toe-up
- 不能太像 Top

### Top
至少检查：
- 接近顶点 / 上行结束
- 不能已经明显开始下杆
- 不能离 Impact 太近

### Mid-downswing
至少检查：
- 必须位于 Top 与 Impact 之间
- 已明显从 Top 开始下行
- 不能太像 Top
- 不能已经贴近 Impact
- 不能只是“Impact 前固定偏移帧”

### Impact
至少检查：
- 接近合理击球瞬间
- 不要与 Mid-downswing 混淆
- 不要已经进入 Follow-through

### Mid-follow-through
至少检查：
- 已明显过击球
- 不能还停留在 Impact
- 不能已经接近 Finish

---

## 八、你必须新建的函数接口（建议）

在 `phase_semantic.py` 里建议实现一个类似函数：

```python
def refine_lite_a_rows_with_phase_semantics(
    rows: list[dict],
    *,
    analysis_frames: list[dict],
    preprocess_meta: dict,
    poses: list[dict] | None = None,
    timeline: list[dict] | None = None,
    motions: list[float] | None = None,
    impact_hint_frame_index: int | None = None,
) -> tuple[list[dict], list[str], dict]:
    ...
```

返回：
1. `refined_rows`
2. `semantic_fail_reasons`
3. `semantic_debug`

其中：
- `refined_rows`：修正后的 8 帧
- `semantic_fail_reasons`：如果还有低信任原因，带出来
- `semantic_debug`：便于日志排查

---

## 九、关键要求：局部重选要“持续进行”，不是只试一次

这次不要做成：

- 语义检查一次
- 不过
- 随便换一帧
- 还不过
- 直接 fail

你必须改成：

### 对每个动作：
1. 先算当前候选帧的语义分
2. 若低于阈值：
   - 在对应窗口中枚举候选
   - 对窗口中候选逐个打分
   - 选语义分最高者
3. 替换当前动作帧
4. 再检查是否仍违反硬约束
5. 如仍不理想，可继续一轮局部重选
6. 最终至少要选出窗口中“最合理”的那一帧

也就是说：

> **必须持续重选，直到拿到当前窗口内最合理帧；不能因为第一次语义不通过就直接停。**

注意：
- “当前窗口内最合理” 不等于“永远百分百正确”
- 但它必须优于原来那个明显错误帧
- 也不能直接退回 fail 退出

---

## 十、orchestrator.py 必须怎么改

`backend/services/lite_ab_mirror/orchestrator.py` 必须改成这个责任链：

1. `run_lite_a_infer_only(...)`
2. 拿到 infer 结果
3. 无论 infer 当前是 pass 还是 fail，都进入 `phase_semantic.py`
4. 在 `phase_semantic.py` 中对中间 6 动作逐个做语义检查 + 局部重选
5. 得到 `refined_rows`
6. 再调用 `run_lite_a_gate(refined_rows, semantic_fail_reasons=...)`
7. 输出结果

关键要求：
- **删掉/改掉当前那种一看到 `infer.a_status == "fail"` 就直接 return 的行为**
- 必须先让 semantic recovery 跑完
- 最终再决定 trust / gate

---

## 十一、a_gate.py 必须怎么改

`a_gate.py` 不能再只是：
- order
- gap
- avg confidence
- top/impact/finish confidence

你必须让它还接收：
- 中间 6 动作的语义检查结果
- semantic fail reasons

建议新增失败原因：
- `toeup_semantic_invalid`
- `mid_backswing_semantic_invalid`
- `top_semantic_invalid`
- `mid_downswing_semantic_invalid`
- `impact_semantic_invalid`
- `mid_followthrough_semantic_invalid`

但注意：
- 这些原因是最终 trust / gate 判断依据
- 不是为了让流程直接早退
- **流程必须先完成局部重选，再做审批**

---

## 十二、你绝对不能做的事

1. 不要提 B
2. 不要引入 B
3. 不要整条 A 重跑很多次
4. 不要第一次语义不过就直接 fail
5. 不要缺帧
6. 不要动前端
7. 不要只调阈值假装修复
8. 不要只写注释
9. 不要把“不会选”当成结束理由

---

## 十三、最终验收标准

这次改完，必须满足：

1. 主路径仍然是 A-only
2. 中间 6 个动作全部有语义检查
3. 语义检查的执行者是 `phase_semantic.py`
4. `a_gate.py` 只做最终审批
5. infer fail 时不允许直接退出，必须继续 semantic recovery
6. 中间动作语义不过时，不允许直接退出，必须继续局部重选
7. 最终必须输出完整 8 帧
8. 允许 low-trust，但不允许“不选”
9. `Mid-downswing` 不再只是 Impact 前固定偏移
10. 不改前端
11. 不引入 B

---

## 十四、最后输出格式

完成后只输出：

1. 改了哪些文件
2. 每个文件改了什么
3. 你如何实现“谁来做语义检查”
4. 你如何实现“infer fail 也不直接退出”
5. 你如何实现“中间动作语义不过就持续局部重选”
6. 为什么现在不会因为第一次不通过就直接失败
7. 为什么这次会让关键帧更准
8. 还剩哪些难例

---

## 最后一条死命令

> 不允许“第一次语义不过 = 直接 fail”。  
> 必须继续局部重选，直到选出当前窗口内最合理的帧，再由 gate 决定 trust。  
> 可以低信任，但不能不选。
