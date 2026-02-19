下面是一份“可复制粘贴、长期复用”的上下文文档（你之后和 GPT 继续聊时，直接把这份发过去就能快速恢复全部背景与共识）。我按**毕设目标 → 问题定义 → 系统方案 → 模拟器设计 → 实现演进 → 评估计划 → 叙事版本选择**整理，尽量做到闭环、无歧义。

---

# 本科毕设上下文文档：在线调度中的 LLM 推理模拟器

## 0. 项目一句话概述

本毕设研究在线 LLM 推理服务集群的全局调度：为每个到达请求选择执行实例。为提升调度质量，引入在线推理模拟器（预测模块）对“请求在不同实例上的执行表现”进行快速预测。模拟器必须满足在线开销预算，并通过与真实实例同步避免状态漂移。

---

## 1. 我需要解决的问题（Problem）

### 1.1 背景

* 在线 LLM serving 中，调度器需要在请求到达时选择实例，关注指标包括：

  * TTFT（首 token 延迟）
  * TPOT（每 token 时间 / token 间隔）
  * 吞吐（QPS）
  * tail（p95/p99）通常对体验与 SLO 更关键
* 简单调度依据（队列长度/GPU 利用率/静态规则）在动态 batching、请求长度异质时容易失准。

### 1.2 核心问题

在线调度需要一个预测模块：对新请求 r、候选实例 i，在当前实例状态下预测关键性能指标 (\hat{M}(r,i))，以指导全局调度决策。

### 1.3 关键挑战

1. **在线开销约束**：full-fidelity 模拟器过重，无法支撑在线预测吞吐（预测路径预算极小）。
2. **状态漂移 divergence**：真实实例和全局调度器/模拟器之间存在乱序、延迟，以及实例侧策略行为，导致模拟器预测会逐渐偏离真实。
3. **并行候选预测**：候选实例评分通常并行计算，调度决策延迟由关键路径（max(pred time) + queueing）决定，不随候选数 K 简单线性增长。

---

## 2. 系统整体方案（System Overview）

### 2.1 组件与职责

* **Global Scheduler（全局调度器）**：请求到达 → 调用预测模块评估多个实例 → 选择实例 → 指派请求
* **Instance Scheduler（实例调度器/运行时）**：连续 batching 执行 → 每批结束上报执行轨迹与状态摘要
* **Online Simulator（在线模拟器）**：维护每实例的轻量状态副本，提供预测接口，并通过批边界同步修正状态

### 2.2 在线闭环（关键）

调度闭环：
`Predict（反事实评估） → Assign（写入决策） → Execute（真实执行） → Sync（批边界上报） → Reconcile（修正模拟器状态）`

---

## 3. 在线模拟器的语义与接口（Semantics & API）

在线模拟器不同于离线事件循环模拟：它面向“单请求、候选实例”的反事实预测，并需要与真实系统持续同步。

建议的最小接口集合：

1. `EnqueueRequest(instance_id, request_meta, ts)`

   * 调度完成后把请求加入该实例模拟器的等待视角
2. `Predict(instance_id, request_meta, action_hint) -> metrics`

   * 输出用于比较的指标：TTFT/TPOT/完成时间或 SLO 风险等（以调度策略实际使用的 score 为准）
3. `SyncBatchResult(instance_id, batch_trace)`

   * 实例每批完成时上报：batch_id、prefill_len、decode_len、duration、完成请求信息（或数量）、（可选）抢占/取消等事件标志

同步以批边界为主：成本低，表达力足够用于修正核心状态。

---

## 4. 状态抽象：pattern-level state（核心设计选择）

为了降低在线预测开销，模拟器不做完整 request-level schedule + KVCache block 级管理，而采用轻量状态抽象：

* **pattern-level state（批/模式粒度）**：

  * 上一批 pattern：`(prefill_len, decode_len)`
  * 最近批结束时间/逻辑时钟
  * 可选：少量队列聚合统计（waiting_count 等）
* **KVCache 不在模拟器内细管**：KVCache 的 allocate/evict/可用性等由 vLLM/实例侧上报摘要，模拟器不维护 block 级数据结构
* **预测粒度从“逐请求构批”降为“按上次 batch pattern 推进”**：构批无需遍历 waiting/running 队列，从而降低预测常数与复杂度

同步在批边界对齐，防止 pattern 抽象带来长期漂移。

---

## 5. 系统实现演进（Implementation Path & Lessons）

### v1：Full-fidelity reference（不可行，但用于语义对齐/开销归因）

* 复刻 vLLM：KVCache block 管理（allocate/evict）、running/waiting queue 遍历构批
* 每次预测 fork 当前实例状态做反事实推演
* 结果：预测吞吐极差，无法满足在线（例：8 卡仅 ~20 QPS，而系统服务能力可 50QPS+甚至 100+）
* **定位**：v1 是 reference（正确性/归因），不是 baseline，避免“baseline 做烂”的质疑

### v2：Lightweight（可部署）

* 不再管理 KVCache block：由 vLLM 上报 allocate/evict/摘要
* schedule 简化：以 batch/pattern 粒度管理，不遍历 request 队列构批
* 预测吞吐显著提升，满足在线需求

### v2+：Prediction reuse（默认实现/增强项）

* 动机：对堆积 waiting 请求，若每次新请求都从 gt fork 重算 backlog，存在大量重复计算
* 做法：复用上次预测推进后的“末状态”，对新请求做增量预测（prediction reuse）
* 观察：关闭 reuse 后（只保留 batch 粒度 + KVCache 外包）系统吞吐几乎不变，但单次预测 latency 更高；说明吞吐瓶颈在系统其他环节（如请求发送/调度系统其他逻辑），但 reuse 能降低控制面延迟/尾部风险

  * 你给出的典型数值：reuse 开启 ~20µs，关闭 ~400µs（差微秒级）

并行候选预测背景下，不用“随 K 线性放大”的叙事；更合理的是：

* 决策延迟由 `max(pred compute) + queueing` 决定
* reuse 的价值主要体现在降低关键路径尾部与减少预测服务排队（尤其 burst/高并发时）
* 若需要更强证据，可做少量插桩分解：submit/start/end → compute vs queueing（可选，允许改一点代码）

---

## 6. Evaluation（你已有的三部分 + 推荐补充）

你计划的 Evaluation 已包含：

1. **端到端**：预测-based/混合调度 vs 简单指标混合策略对比
2. **预测准确性对端到端影响**（可用误差/敏感性分析）
3. **预测模块吞吐是否满足在线需求**

建议可选增强（本科毕设可只选一个加分项）：

* **Cross-table（跨预测表/跨模型）一致性分析**：你观察到使用 Qwen3-30B-A3B 的预测表预测 Qwen2.5-7B，mean 指标差小（<3%），但 p95/p99 差更明显（10%+）。

  * 解释路线：调度更依赖相对排序，mean 不敏感但 tail/SLO 更敏感
  * 易做指标：

    * **Spearman 相关**：对每个请求，比较两预测表对候选实例打分排序一致性
    * **Agree@1**：两表 top1 选择一致率（非 oracle 正确率，仅一致性）
* **同步机制消融（可选）**：No-Sync vs Full-Sync（或 Coarse-Sync），展示漂移/端到端 tail 退化

  * 注意：No-Sync 仍可定义为“模拟器自推进不接收 batch_trace”，用来展示 drift 的必要性

---

## 7. 论文结构（本科毕设推荐版本）

### 第1章 引言

* 动机、问题、挑战、贡献（3 条）

### 第2章 背景与问题定义

* prefill/decode、batching 基础
* 指标定义（TTFT/TPOT/吞吐/tail）
* 形式化问题：在线预测用于实例选择

### 第3章 系统总体设计

* 架构：Global Scheduler / Instance Scheduler / Online Simulator
* 在线闭环时序（Predict→Assign→Execute→Sync→Reconcile）

### 第4章 在线模拟器设计

* 接口语义与字段（Enqueue/Predict/SyncBatchResult）
* pattern-level state 抽象
* 批边界同步与漂移原因（乱序/策略事件）

### 第5章 实现与优化（工程闭环）

* v1 reference：full-fidelity 不可行
* v2 lightweight：KVCache 外包 + batch/pattern schedule 简化
* prediction reuse：作为默认实现（解释价值：降低重复计算/控制面延迟 tail）

### 第6章 实验评估

* 设置
* 端到端对比
* 预测模块开销
* 可选：cross-table Spearman/Agree@1 或同步消融（二选一）

### 第7章 总结与展望

* 总结与未来工作（异构硬件、在线校准、更复杂实例策略等）

---

## 8. 写作/叙事策略（避免被质疑的关键点）

1. **v1 不叫 baseline，叫 reference**：用于语义对齐与开销归因，证明 full-fidelity 在线不可行。
2. **v2（lightweight）是可部署核心**：满足在线吞吐预算的关键来自“pattern-level 状态抽象 + schedule 简化 + KVCache 外包”。
3. **prediction reuse 是增强项/默认实现**：即便吞吐不变，也能降低预测 latency/tail、减少排队与控制面抖动；并行预测场景下用 max+queueing 的关键路径解释，而非 K 线性增长。
4. **tail 差异是有意义的**：p95/p99 与 SLO/拥塞放大更相关，mean 差小不代表无影响。

---

## 9. 我之后和 GPT 继续讨论时，希望 GPT 记住/沿用的共识

* 这是本科毕设：优先保证主线闭环与可完成性，不追求系统论文级别细节（如完整在线校准、复杂一致性协议）。
* 性能模型/校准章节可以删除或压缩（当前以离线预测表查表为主，没有成熟在线更新实现）。
* Cross-table 用 Spearman/Agree@1 做一致性分析是性价比高的补充。
* 实现优化闭环要避免“把 baseline 做烂”的质疑：v1=reference；baseline 在 lightweight 空间内比较。

---

如果你愿意，我也可以把这份文档再“压缩成 10 行版摘要”，方便你每次对话只贴极短上下文；或者把它改成 LaTeX 的 `\section{Project Context}` 形式，直接放进附录里当项目说明。
