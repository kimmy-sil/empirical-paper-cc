---
name: empirical-pipeline
description: "实证论文全流程调度，Stage -1 到 Stage 5，含 Gate 检查与 Worker-Critic 配对"
disable-model-invocation: true
---

# 一站式实证论文生成流水线

Pipeline: S-1 → S0 → S1 → S2 → S3 → ◆Gate1 → S4 → S5

---

## Stage -1: 研究规格书生成（条件触发）

触发条件：`quality_reports/research_spec_*.md` 不存在时自动触发。如已存在 → 跳过。

### 入口分流（1 题）

提问："你当前的研究想法处于哪个阶段？"
- A "有明确问题和大致方法" → 6 步快问快答
- B "有方向但没想清楚" → 6 步 + 每步允许"不确定"，提供选项辅助
- C "只有模糊兴趣" → 自由对话，收敛到可检验问题后转入 B 模式

模式 C 策略：依次问"对什么现象好奇？→ 关心它的原因还是影响？→ 最近有相关政策/数据变化吗？" 用户选定方向后转 B。3 轮仍无法收敛 → 建议先读 5 篇文献再回来。

### 6 步访谈

| 步骤 | 提问 | "不确定"时的辅助（仅 B 模式） |
|------|------|-------------------------------|
| 1. 现象 | "你想研究什么现象？为什么重要？" | 提供该领域近年热点（读取 domain-profile） |
| 2. 机制 | "X 通过什么机制影响 Y？" | 提示常见渠道：信息/激励/资源/制度 |
| 3. 数据 | "你有什么数据？或计划用什么？" | 列出领域常用数据源 |
| 4. 识别 | "有什么外生冲击/准实验变异？" | 提示领域常用策略；仍无法回答 → ⚠️ 提示缺乏因果基础 |
| 5. 预期 | "预期结果什么方向？相反意味着什么？" | 可标记 [TBD]，不阻塞 |
| 6. 贡献 | "和最接近的 3 篇文献相比，差异点？" | 可标记 [TBD]，或触发 literature-review 辅助 |

A 模式：6 步必答。B 模式：步骤 5-6 可输出 [TBD]，Stage 0 补全。

### 输出

1. `quality_reports/research_spec_[topic].md` — 基于 `templates/research-spec-template.md` 填充
2. `.claude/references/domain-profile.md`（如不存在）— 基于 `templates/domain-profile-template.md` 填充

research_spec 含 [TBD] 字段时 → Stage 0 优先补全，再走 0.1-0.5。

---

## Stage 0: 研究设计协作

### 0.0 前置文件检查

| 文件 | 路径 | 缺失时 |
|------|------|--------|
| 研究规格书 | `quality_reports/research_spec_*.md` | 回退 Stage -1 |
| 领域档案 | `.claude/references/domain-profile.md` | 回退 Stage -1 |
| 期刊画像 | `.claude/references/journal-profiles.md` | 提示检查仓库 |
| 内生性路由 | `.claude/references/endogeneity-routing.md` | 提示检查仓库 |

### 0.1 因果问题拆解（依次提问）

提问顺序（不可跳过）：

1. X → Y 的因果故事是什么？（理论机制）
2. 最大内生性威胁是什么？
   - 遗漏变量 / 反向因果 / 测量误差 / 选择偏误
   - 信息不对称 / 动态内生性 / 同时性
3. 有什么外生冲击或准实验变异？

处理规则：
- 无法回答第3题 → ⚠️ 无可信因果路径，建议改为描述性分析
- 三题全答 → 进入0.2 DAG绘制

### 0.2 DAG 绘制（全局基础）

文字描述模板：
```
节点：[列出所有变量]
箭头：X→Y, Z→X, Z→Y（混淆）, M→Y（中介）
不可观测：U→X, U→Y（遗漏变量）
```

DAG 贯穿全流程：
- Stage 3 控制变量审核：基于DAG判断前定/后定/对撞
- Stage 4 机制分析：基于DAG确认中介变量M的位置

### 0.3 识别策略匹配

决策树（与CLAUDE.md板块5保持一致）：

```
数据结构判断：
├── 有id + time → panel-data（默认）
├── 纯横截面 → ols-regression
└── 单截面多时期 → time-series

识别策略判断：
├── 政策/处理交错实施 → did-analysis
├── 数值门槛/切断点 → rdd-analysis
├── 合理外生工具变量 → iv-estimation
├── 含y_{t-1}且T<20 → panel-data（GMM）
├── 处理单位≤5 → synthetic-control
├── 高维协变量+因果设计已有 → ml-causal（DML）
├── 纯横截面无因果设计 → ols-regression（描述性）
└── 以上都不适用 → ⚠️ 告知用户，建议：
    (a) 重新寻找准实验变异
    (b) 转描述性分析，声明不作因果解释
    (c) 需要反事实模拟 → 超出本pipeline范围（结构估计）
```

### 0.4 文献定位

用户提供3-5篇参考文献后，系统依次提问：
- 你的数据来源与他们有何不同？
- 你的识别策略与他们有何不同？
- 你的样本（时间/地区/人群）有何不同？

输出贡献定位模板：
```
本文在[方法/数据/样本]上拓展了[参考文献]，
通过[识别策略]估计[Estimand]，
识别了[X→Y]的[局部/平均]因果效应。
```

### 0.5 Research Design Memo

必须在 Gate 1 前完成，保存到 `quality_reports/research_design_memo_[topic].md`。

| 字段 | 内容 |
|------|------|
| 研究问题 | X对Y的因果效应 |
| DAG | 文字描述（节点+箭头） |
| 识别策略 | 方法名称+外生性来源 |
| Estimand | ATT/LATE/ATE/ATO + 定义 |
| 数据 | 来源/样本/时间跨度/观测数 |
| 方法路径 | Skill路由 |
| 文献贡献 | 与参考文献的差异 |
| 风险+Plan B | 主要风险+备选方案 |

---

### ◆ Gate 1：确认研究设计

**必停点。不得跳过。**

向用户呈现：
1. 数据概况（观测数、变量范围、缺失情况）
2. 识别策略选择及理由
3. Estimand 声明（估计的是谁的效应，在什么条件下）
4. Research Design Memo 摘要

等待用户明确确认（"确认"/"继续"/"OK"）后方可进入 Stage 1。

---

## Stage 1: 数据清洗 + 描述统计

### 1.1 数据清洗
调用 `data-cleaning` skill，执行：
- 缺失值检测（MCAR/MAR/MNAR机制判断）
- 面板平衡性检查（有id+time时）
- 连续变量1%/99%缩尾（Winsorize）
- 变量类型规范化
- 输出清洗日志

### 1.2 描述统计
调用 `stats` skill，生成：
- Table 1（Summary Statistics）
- Balance Table（处理组vs对照组）
- 缺失值模式图
- VIF（控制变量共线性检查，VIF>10预警）

**AI检查点**：数据量级和变量范围是否符合经济学合理范围？
- 例：工资变量出现负数→预警；GDP增速>50%→预警

---

## Stage 2: 描述性分析

### 2.1 趋势图与分布图
调用 `figure` skill：
- 处理组/对照组结果变量时间趋势（DID前置）
- 运行变量分布密度图（RDD前置）
- 散点图（X vs Y）

### 2.2 相关性初探
- 双变量相关系数矩阵
- 预分析检验（避免数据挖掘）

---

## Stage 3: 基准回归

### 3.1 控制变量审核（基于DAG）
在运行回归前，逐一审查控制变量：

| 类型 | 判断 | 行动 |
|------|------|------|
| 前定变量（X的原因，Y的原因，不在X→Y路径上） | 可控制 | 加入回归 |
| 后定变量（X→M→Y路径上的M） | 禁止 | 移除，改为机制分析 |
| 对撞变量（X→C←Y） | 禁止 | 绝对不控制 |
| 工具变量的影响路径（Z→X） | 禁止控制Z | 保留Z仅作工具 |

### 3.2 聚类层级决策
- 聚类层级 = 处理分配层级（不是个体层级）
- 例：政策在城市层面分配 → 城市层面聚类
- 双向聚类：个体+时间（可选，附理由）

### 3.3 执行基准回归
调用对应方法 skill：
- 运行主回归
- 输出标准三线表至 `output/tables/`
- 声明 Estimand

方法特有前置检验：
- DID → 预趋势检验（事件研究图）
- RDD → 密度检验（McCrary/rddensity）+ 协变量平衡
- IV → 第一阶段F统计量（F>10预警，F>100理想）
- GMM → 工具变量数≤N，Sargan/Hansen J检验

**AI检查点**：系数符号和量级是否符合经济学逻辑？
- 例：最低工资↑→就业率↑（符号异常，需解释）
- 例：政策效果是全国GDP的10%（量级异常，预警）

### ◆ Gate 2（条件触发）：基准回归异常时停

触发条件（满足任一）：
- 系数符号与理论预期相反且幅度显著
- 量级远超合理范围（>3倍行业均值）
- 预趋势检验显著失败
- 第一阶段F<10

弹出内容：
- 报告具体异常
- 提供3种解释（数据问题/识别策略问题/真实效果）
- 建议下一步行动
- 等待用户指示是否继续

---

## Stage 4: 稳健性 + 内生性 + 机制 + 异质性

### 4.1 稳健性三层分层

#### 必做层（绑定识别策略，不可省略）

**DID：**
- 事件研究图（动态效应）
- Bacon 分解（交错DID必做，负权重≥10% → 改用CS/SA）
- 安慰剂检验（500次置换，真实估计量排名百分位）
- HonestDiD（预趋势敏感性分析，Rambachan & Roth 2023）
- Callaway-Sant'Anna 或 Sun-Abraham（若Bacon显示问题）

**IV：**
- 第一阶段F统计量
- 倍增比（Reduced-form / First-stage，量级合理性）
- Hansen J / Sargan检验（过度识别时）
- Anderson-Rubin 弱工具稳健推断
- plausexog（部分识别，Conley等2012）
- Jackknife IV（高杠杆观测）

**RDD：**
- 密度检验（McCrary/rddensity）
- 协变量平衡（各协变量在断点处的RDD估计应不显著）
- 带宽敏感性（0.5h, 0.75h, 1.25h, 1.5h）
- 安慰剂断点（在非断点处运行RDD）
- MSE点估计 + CER置信区间

#### 推荐层（每项必须附经济学理由）

| 检验 | 适用情境 | 说明 |
|------|----------|------|
| 替换被解释变量 | 结果指标可多角度测量 | 说明哪个是主指标 |
| 替换聚类层级 | 不确定分配机制层级 | 比较SE变化 |
| 缩短样本区间 | 担心时期选择性 | 避免包含其他干扰事件 |
| PSM-DID | 担心可比性 | 注意：解决样本问题，非内生性 |
| 子样本分析 | 有理论预测异质性 | 先分组，再检验组间差异 |

#### 情境层（数据特征自动触发）

| 数据特征 | 触发检验 |
|----------|----------|
| 因变量含大量0（>30%） | ppmlhdfe（泊松HDFE） |
| 同期存在多政策 | 排除干扰政策的子样本 |
| 处理可能提前泄露/预期 | 预期效应检验（提前一期） |
| 面板T<5 | 报告偏差修正SE |
| 小样本（N<200） | 野bootstrap SE |

### 4.2 内生性处理

| 方法 | 触发条件 | Skill |
|------|----------|-------|
| IV/2SLS | 有合理工具变量 | iv-estimation |
| Heckman两阶段 | 样本选择问题（观测非随机） | ols-regression |
| System GMM | 含y_{t-1}且T<20 | panel-data |
| Oster (2019) | 无工具变量时的系数稳定性 | ols-regression |
| 滞后处理变量 | 缓解反向因果（辅助，非替代）| 内嵌 |

### 4.3 机制分析

调用 `mechanism-analysis` skill。

**强制前置：确认DAG中M的位置**
- M 是 X→M→Y 路径上的中间变量？→ 合法中介
- M 是 X→Y 和 M→Y 的共同原因？→ 对撞，禁止
- M 是 X 的原因？→ 前定，不是机制

推荐方法（按适用性排序）：

| 方法 | 适用条件 | 注意 |
|------|----------|------|
| 两步法（首选） | M和Y均可观测 | M方程也需满足识别条件 |
| Baron-Kenny四步法 | 线性系统 | 需Sobel检验或bootstrap |
| 反事实中介分析 | 需要分解直接/间接效应 | 强假设（Sequential Ignorability）|
| 渠道排除 | M不可观测但可排除 | 子样本或交互项 |
| 信息渠道检验 | 信息不对称机制 | 分组+交互 |

⚠️ Causal Forest 只能在主回归已建立因果效应后，用于探索异质性，不能作为因果识别工具。

### 4.4 异质性分析

必做：
- 分组回归（按理论预测的维度）
- 组间系数差异检验（交互项或Suest）

可选（在主回归因果成立后）：
- Causal Forest（grf/econml）探索异质性
- 分位数回归（分布效应）

### 4.5 扩展分析（可选）

- 门槛效应（Hansen 1999 面板门槛）
- 动态/滞后效应（t+1, t+2, t+3期效应）
- 空间溢出（若有地理维度，需声明：超出本pipeline核心范围）

---

## Stage 5: 论文 + 输出

### 5.1 论文写作

调用 `paper-writing` skill，按节依次生成。

推荐写作顺序：`background → strategy → data → results → conclusion → intro → abstract`

argument 格式：`[section] [--journal NAME] [--lang cn|en]`

paper-writing skill 会自动读取：
- `.claude/references/domain-profile.md` → 领域惯例校准
- `.claude/references/journal-profiles.md` → 期刊风格校准
- `quality_reports/research_spec_*.md` → 研究问题/贡献
- `quality_reports/research_design_memo_*.md` → 识别策略/Estimand

详细写作规范、LaTeX 模板、节级模板见 `skills/paper-writing/`。

### 5.2 输出规范

| 类型 | 规范 | 工具 |
|------|------|------|
| 回归表 | booktabs三线表，\input{}引用 | table skill |
| 图形 | DPI≥300, 字号≥12pt, 黑白友好 | figure skill |
| 引用格式 | APA 7 或 GB/T 7714 | literature-review skill |
| 代码 | Python + R，可复现 | output/replication/ |

**禁止手动抄数**：所有表格数字必须通过代码生成后 `\input{}` 引用。

### 5.3 审稿模拟（条件触发）

触发条件：用户请求，或论文完成后的可选步骤。

调用 `methods-referee` agent（Task 分派），执行：
- 6维审稿评分（STRUCTURAL / CREDIBILITY / MEASUREMENT / POLICY / THEORY / SKEPTIC）
- 可指定目标期刊：`--peer [journal]`（读取 `.claude/references/journal-profiles.md` 校准）
- 输出审稿报告至 `quality_reports/referee_report_*.md`

### 5.4 Beamer（可选）
调用 `beamer-ppt` skill，生成15-20页学术汇报：
Title → Motivation → Literature → Contribution → Data → Strategy → Results → Robustness → Heterogeneity → Mechanism → Conclusion → Appendix

### 5.5 最终 AI 检查点

论文与回归输出一致性核查：
- 正文提到的系数是否与表格一致？
- 显著性描述是否与星号一致？
- 样本量是否与Table 1一致？
- Estimand声明是否存在？

建议：使用多模型交叉审核（Claude + 其他模型）检查数字一致性。

### 5.6 审稿回复（R&R 阶段，条件触发）

触发条件：收到审稿意见后，用户手动触发。

调用 `respond-to-referees` agent（Task 分派），执行：
- 解析审稿意见 → 逐条分类（方法/数据/写作/格式）
- 生成 Response Letter（逐条回复 + 修改位置标注）
- 调用 paper-writing skill 修改对应章节
- 输出 `output/response_letter.tex` + 修改稿 diff

---

## 附录A：Skill 调用速查

| Stage | 调用 Skill | 调用 Agent |
|-------|-----------|-----------|
| S-1 | literature-review（可选辅助） | — |
| S0 | （内嵌） | — |
| S1 | data-cleaning, stats | — |
| S2 | figure, stats | — |
| S3 | 方法 skill（did/iv/rdd/panel/sc/ols/ml-causal/ts） | — |
| S4 | 同上 + mechanism-analysis + figure + table | methods-referee（Gate 2 触发时） |
| S5 | paper-writing, literature-review, table, figure, beamer-ppt | methods-referee, respond-to-referees |

辅助 skill（按需调用）：
- `data-fetcher`：获取 FRED/WB/OECD/akshare 数据
- `stats`：描述统计/Balance Table/VIF
- `figure`：事件研究图/系数图/RDD图
- `table`：三线表/回归表/LaTeX格式化

## 附录B：共享文件路径速查

| 文件 | 路径 | 写入者 | 读取者 |
|------|------|--------|--------|
| domain-profile | `.claude/references/domain-profile.md` | Stage -1 | pipeline, paper-writing, lit-review, referee |
| journal-profiles | `.claude/references/journal-profiles.md` | 预置 | paper-writing, referee |
| endogeneity-routing | `.claude/references/endogeneity-routing.md` | 预置 | 方法 skill |
| research-spec | `quality_reports/research_spec_*.md` | Stage -1 | pipeline, paper-writing |
| research-design-memo | `quality_reports/research_design_memo_*.md` | Stage 0.5 | pipeline, paper-writing |
""".strip()

import os
os.makedirs("output", exist_ok=True)
with open("output/empirical-pipeline-SKILL.md", "w", encoding="utf-8") as f:
    f.write(pipeline_skill)

original = 6730
new = len(pipeline_skill)
prev_version = 9313

print(f"原版 (file:118):     {original} chars")
print(f"上一版 (v2):         {prev_version} chars")
print(f"本版 (v3):           {new} chars")
print(f"vs 原版:             +{new - original} ({(new-original)/original*100:.1f}%)")
print(f"vs 上一版:           {new - prev_version:+d} ({(new-prev_version)/prev_version*100:.1f}%)")

# Count the Stage -1 section specifically
s1_start = pipeline_skill.index("## Stage -1")
s0_start = pipeline_skill.index("## Stage 0")
stage_minus1 = pipeline_skill[s1_start:s0_start]
print(f"\nStage -1 单独:       {len(stage_minus1)} chars")
print(f"Stage 0-5 + 附录:    {new - len(stage_minus1)} chars (≈原版)")