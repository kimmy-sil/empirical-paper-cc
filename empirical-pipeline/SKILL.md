# 一站式实证论文生成流水线

全流程编排器，覆盖从数据审计到终稿输出 + Beamer 幻灯片的完整研究流程。

本 skill 为 Layer 1 编排器。各阶段会自动调用 `skills/` 下的独立模块（Layer 2）。

## Architecture

9-stage pipeline with 2 human decision gates:

```
S0:Design → S1:DataAudit → S2:QuestionGen → S3:PreAnalysis → ◆Gate1
→ S4:Analysis → S5:Drafting → S6:Review → ◆Gate2 → S7:Finalize → S8:Beamer
```

## Entry Points

**A) Has data file** → Start Stage 1
**B) Has topic, no data** → Start Stage 0 (方法咨询模式)
**C) Both data + idea** → Start Stage 0 (完整模式)
**D) Has specific task** → 直接调用对应 skill（如 "帮我跑个 DID"）
**E) Mid-stream** → Ask user's current progress, jump to matching stage

---

## Stage 0: 研究设计协作（Research Design）

Goal: 将模糊的研究 idea 转化为可执行的因果推断设计，或对已有数据进行诊断和策略匹配。

### 入口模式

- **数据探索模式**：用户只上传了数据 → 审计数据结构 → 建议可能的研究问题和识别策略
- **方法咨询模式**：用户只给了 idea → 拆解因果问题 → 建议需要什么数据和方法
- **完整模式**：两者都有 → 数据诊断 + 策略匹配 → Gate 1 确认后继续

### 阶段 0.1：因果问题拆解

引导用户回答三个核心问题（依次提问，不一次全抛）：

1. **X → Y 的因果故事是什么？** 用一句话描述。例如："AI试验区政策 → 地区TFP提升"
2. **最大的内生性威胁是什么？**
   - 遗漏变量（有不可观测因素同时影响 X 和 Y）
   - 反向因果（Y 反过来影响 X）
   - 测量误差
   - 选择偏误（样本非随机）
   - 信息不对称
   - 动态内生性 / Nickell 偏误（含滞后因变量）
   - 同时性
3. **你有什么外生冲击可以利用？** 政策变动、制度断点、随机分配、历史事件

如果用户无法回答第3题 → ⚠️ "当前可能没有可信的因果推断路径。建议：(a) 寻找自然实验 (b) 考虑DML+Selection on Observables (c) 改为描述性分析"

### 阶段 0.2：DAG 绘制（全局基础，贯穿后续所有阶段）

引导用户画因果图（可用文字描述）：
- X（处理变量）→ Y（结果变量）
- W（控制变量）→ X, W → Y
- M（机制变量）：X → M → Y
- 标注哪些变量是前定的（处理前确定）、哪些是后定的（受处理影响）

这张 DAG 将在以下阶段复用：
- Stage 3：控制变量审核（排除坏控制）
- Stage 4：机制分析前置检查
- Stage 5：论文理论框架描述

### 阶段 0.3：识别策略匹配

```
你有什么样的外生变异？
├── 政策在不同地区/时间交错实施 → Staggered DID (→ did-analysis)
├── 政策同一时间统一实施 → 标准DID / RDiT
├── 存在明确的数值门槛 → RDD (→ rdd-analysis)
├── 有合理的工具变量 → IV/2SLS (→ iv-estimation)
├── 单一处理单位 → Synthetic Control (→ synthetic-control)
├── 含滞后因变量 + T小 → 动态面板GMM (→ panel-data GMM-lite)
├── 无明确自然实验，有丰富协变量 → DML (→ ml-causal)
├── 时间序列/宏观 → VAR/VECM (→ time-series)
└── ⚠️ 以上都不适用 → 告知用户，建议重新寻找识别策略
```

### 阶段 0.4：文献定位（用户提供 + 系统提问）

让用户提供 3-5 篇核心参考文献（标题或DOI），然后提问：
- 你的数据覆盖了什么新时期或新地区？
- 你的识别策略有什么改进？
- 你用了什么新的度量方式？

输出一段："与 XX (2023) 相比，本文的边际贡献在于……"

### 阶段 0.5：生成 Research Design Memo

输出结构化备忘录，用户确认后才进入 Stage 1：

| 部分 | 内容 |
|------|------|
| 研究问题 | 一句话因果问题 |
| 理论框架 | DAG（文字版或dagitty代码） |
| 识别策略 | 外生变异来源 + 为什么可信 |
| Estimand | 估计的是什么效应（ATT/LATE/ATE/ATO） |
| 数据需求 | 变量、样本期、数据来源 |
| 方法路径 | 主回归 + 稳健性方向 + 机制方向 |
| 预期贡献 | 相对文献的边际贡献 |
| 主要风险 | Plan B |

### ◆ Gate 1：确认研究设计

呈现 Research Design Memo。用户必须确认后才继续。

---

## Stage 1: Data Audit

Goal: Understand data structure, quality, and research potential to constrain hypothesis generation.

Calls: `skills/data-cleaning`（如数据需清洗）, `skills/stats`（描述性统计）

### Steps

1. Load data → print shape, dtypes, time range, entity count
2. Classify variables: panel IDs, outcome candidates, treatment candidates, controls, instruments
3. Quality check: missing rates, panel balance, outliers, within-variation
4. Generate `output/data_audit_report.md`

Read `references/data-audit-checklist.md` for detailed diagnostics.

### Output

`output/data_audit_report.md` containing:
- Data overview table
- Variable dictionary (name, type, description, missing%, distribution)
- Panel structure diagnosis
- Potential research directions
- Data limitations

> **AI防幻觉检查点**：检查数据量级、变量范围是否合理（防幻觉）。核实样本量、时间范围、关键变量取值域与用户描述一致；如发现异常立即提示用户确认。

---

## Stage 2: Research Question Generation

Goal: Generate 3-5 feasible, dataset-aware research questions with scoring.

Calls: `skills/literature-review`（文献搜索）

### Steps

1. **Dataset-aware constraint**: Each question must use variables that exist with <30% missing, sufficient within-variation, adequate sample size
2. For each candidate, produce structured assessment: see `templates/question-template.md`
3. Quick literature scan (search for existing work, identify gaps)
4. Score each question on 4 dimensions (25pts each = 100 total):
   - Data adequacy
   - Identification credibility
   - Novelty
   - Policy relevance

用户可提供 Research Prompt（参考 `templates/research-prompt-template.md` 格式），直接跳过 Stage 1-2，从 Stage 3 开始。

### ◆ HUMAN GATE: Select Research Question

Present all candidates with scores. Ask user to:
1. Select one question, OR
2. Modify/combine candidates, OR
3. Request regeneration

---

## Stage 3: Pre-Analysis Plan

Goal: Design rigorous causal identification strategy and write complete pre-analysis plan.

### Method Selection Router

Based on data structure and research question, select from method toolbox:

| Condition | Skill to Call |
|-----------|--------------|
| 政策冲击 + 处理/对照组 | `skills/did-analysis` |
| 运行变量 + 断点 | `skills/rdd-analysis` |
| 内生性 + 外生工具 | `skills/iv-estimation` |
| 面板数据 + 固定效应 | `skills/panel-data` |
| 单一处理单位 | `skills/synthetic-control` |
| 高维控制变量 | `skills/ml-causal` |
| 时间序列 / 宏观 | `skills/time-series` |
| 截面基础分析 | `skills/ols-regression` |

Read `references/did-methodology.md` for DID details.
Read `references/panel-fe-methods.md` for panel FE details.
Read `references/other-methods.md` for RDD, IV, and other methods.

### 控制变量审核（基于 Stage 0 的 DAG）

用户指定控制变量后，逐一检查：
1. 这个变量是否受处理变量 D 影响？→ 后定变量，不应控制
2. 这个变量是否是 D 和 Y 的共同结果？→ 对撞变量，不应控制
3. 这个变量在处理发生之前就确定了吗？→ 前定变量，可以控制

聚类层级自动决策：
- 处理在个体层面 → 聚类到个体
- 处理在省/城市层面 → 聚类到省/城市
- 聚类数 < 30 → Wild Bootstrap

### DID Checklist (most common)

1. Define treatment & control groups clearly
2. Identify policy implementation date
3. Plan parallel trends test (event study plot)
4. Choose standard error clustering level (match treatment level)
5. Plan placebo tests (fake timing / fake treatment group)
6. If staggered: use CS(2021) or SA(2021), NOT naive TWFE
7. Plan heterogeneity analysis
8. Rule out confounding concurrent policies
9. Plan anticipation test (lead coefficients should be insignificant)
10. Plan sample composition test

### Output

Generate `output/pre_analysis_plan.md` following `templates/pre-analysis-template.md`.

### ◆ HUMAN GATE: Confirm Strategy

Present pre-analysis plan. Ask user to confirm:
1. Is identification strategy sound?
2. Any variable adjustments needed?
3. Additional robustness checks to add?

---

## Stage 4: Econometric Analysis

Goal: Execute all statistical analysis per pre-analysis plan.

Calls: Selected method skill (e.g. `skills/did-analysis`), `skills/stats`, `skills/figure`, `skills/table`

### Steps

1. **Data prep**: Clean, construct variables, handle missing values
2. **Descriptive stats**: Table 1 (by treatment/control), mean difference tests → call `skills/stats`
3. **Main regression**: Execute chosen identification strategy → call selected method skill
4. **Parallel trends / identification tests**: Event study plot, pre-trend F-test → call `skills/figure`
5. **Robustness**: See layered robustness framework below
6. **Heterogeneity**: Sub-sample regressions, interaction terms

Use scripts in `scripts/` as templates (Python/R available). All analysis code saved to `output/replication/`.

### Multi-Language Code Generation

Default: Python. If user requests R:
- R template: `scripts/did_analysis.R` (fixest + did packages)
- Python template: `scripts/did_analysis.py` (linearmodels)

### 稳健性检验（三层分层）

**必做层**（与识别策略绑定）：
- DID → 平行趋势(事件研究图) + HonestDiD敏感性(默认Relative Magnitudes) + 安慰剂(空间500次+分布图)
- IV → 第一阶段F + 倍增比(>5警告) + plausexog/Lee bounds
- RDD → 密度检验 + 协变量平衡 + 带宽敏感性 + CER置信区间

**推荐层**（应对审稿人常见质疑，每项附经济学理由）：
- 替换被解释变量 — "排除度量方式对结论的影响"
- 替换聚类层级 — "检验统计推断对聚类选择的敏感性"
- 缩短样本区间 — "排除极端年份/疫情的干扰"
- PSM-DID — "检验结论对样本构成的敏感性"

**情境层**（按数据特征触发）：
- 因变量含大量0 → ppmlhdfe
- 多个同期政策 → 排除其他政策干扰
- 处理可能提前泄露 → 预期效应检验
- 需要穷举规格 → Specification Curve

### 内生性处理（解决层）

- IV / 2SLS — 反向因果 + 遗漏变量 + 测量误差
- Heckman 两阶段 — 样本选择偏误 / MNAR
- System GMM — 动态面板 + 滞后因变量内生性（触发条件：含y_{t-1}且T<20且N>T）
- Oster (2019) 系数稳定性 — 评估遗漏变量需要多强才能推翻结论
- 滞后处理变量 — 缓解反向因果（辅助）

### 机制分析

调用 skills/mechanism-analysis。强制 DAG 前置（复用 Stage 0 的 DAG）。默认推荐两步法。

### 异质性分析

分组回归 + 组间系数差异检验（必做）。
可选：Causal Forest 探索（⚠️ 仅在基准回归已建立因果关系后使用）。

### 扩展分析（可选）

- 门槛效应（Hansen门槛模型）
- 滞后效应检验（滞后1-3期处理变量）

### Output

- `output/tables/` — all regression tables (Markdown + LaTeX) → call `skills/table`
- `output/figures/` — event study plot, trend plots, coefficient plots → call `skills/figure`
- `output/analysis_log.md` — complete analysis log

> **AI防错检查点**：检查系数符号和量级是否符合经济学逻辑（防计量逻辑错误）；TWFE后自动触发Bacon分解，负权重>10%时强制切换CS/SA并告知用户。

---

## Stage 5: Paper Drafting

Goal: Write complete academic paper based on analysis results.

Calls: `skills/paper-writing`, `skills/literature-review`

### Structure

Follow `templates/paper-structure.md` for section-by-section guidance.

Standard sections:
1. Abstract (250-300 words)
2. Introduction (1500-2000 words)
3. Institutional Background (1000-1500 words)
4. Literature Review (1500-2000 words)
5. Data & Variables (1000-1500 words)
6. Empirical Strategy (1000-1500 words)
7. Results (2000-3000 words)
8. Conclusion (800-1000 words)
9. References (≥25)

### Writing Quality Rules

- Academic register: rigorous, objective, professional
- Full paragraph prose only (NO bullet lists in body text)
- Every claim backed by data or citations
- Careful causal language (only causal claims if identification is credible)
- Chinese style: 经管期刊风格，专业但不晦涩
- AVOID: "值得注意的是", "综上所述", "取决于", excessive dashes, exclamation marks

### Output Formats

- `output/paper_draft.md` (Markdown)
- `output/paper_draft.tex` (LaTeX, using `templates/paper-latex.tex` as base)
- `output/paper_summary.md` (1-page summary)

> **AI防错检查点**：检查论文正文引用的数字与回归输出是否一致（防抄写错误）；在 Section 3（实证策略）强制插入 Estimand 声明段，明确说明估计的是 ATT/LATE/ATE/ATO 及其适用范围。

---

## Stage 6: Review & Revision

Goal: Simulate peer review and iterate.

### Review Dimensions (0-20 each, 100 total)

| Dimension | What to evaluate |
|-----------|-----------------|
| Identification | Causal credibility, assumption validity, test sufficiency |
| Novelty | Research gap, contribution to literature |
| Policy Relevance | Practical implications for policy/management |
| Execution | Data handling, analysis rigor, result presentation |
| Writing | Logic, completeness, formatting |

### Review Format Checks

- [ ] Paper ≥ 25 pages (incl. tables/figures, excl. references)
- [ ] References ≥ 25
- [ ] Full paragraph prose (no bullet lists in body)
- [ ] Each major section has 3-4 substantive paragraphs
- [ ] All figures/tables numbered, titled, with notes
- [ ] No placeholder data ("XXX", "TBD", "Figure ??")

### Process

Round 1: Generate `output/review_round1.md` with verdict (Accept/Minor/Major/Reject) + detailed comments → Revise → Generate `output/revision_response.md`

Round 2 (if needed): Re-review revised draft → Final revise

Max 2 revision rounds.

### ◆ Gate 2（条件触发）：Approve Final

Gate 2 仅在审稿发现重大问题（Identification评分<12/20，或发现明显计量错误）时弹出，否则静默继续。

弹出时呈现 review report + revision plan，询问用户：
1. Agree with revision plan?
2. Additional comments?
3. Satisfied with quality?

---

## Stage 7: Finalization

### Output Package

```
output/
├── paper_final.md
├── paper_final.tex          (LaTeX 版本)
├── tables/
│   ├── table1_descriptive.md
│   ├── table2_main_results.md
│   ├── table3_robustness.md
│   └── table4_heterogeneity.md
├── figures/
│   ├── fig1_trends.png
│   ├── fig2_event_study.png
│   └── fig3_coefficients.png
├── pre_analysis_plan.md
├── data_audit_report.md
├── analysis_log.md
├── review_round1.md
├── revision_response.md
└── replication/
    ├── analysis.py (or .R)
    └── README.md
```

### Replication Package Requirements

1. Complete analysis code (Python or R)
2. Data dictionary
3. Run instructions (dependencies, execution order)
4. Expected output description

> **AI防错检查点**：建议用户在新会话中让另一个模型独立审核（防单模型盲区）。提示："本论文由同一模型全流程生成，建议开启新对话，使用独立模型（如另一个Claude实例）对终稿进行盲审，以发现潜在的系统性偏差。"

---

## Stage 8: Beamer Slides

Goal: Generate academic presentation slides based on the finalized paper.

Calls: `skills/beamer-ppt`

### Steps

1. Extract key content from paper: RQ, contribution, strategy, main results, robustness, heterogeneity, conclusion
2. Generate 15-20 slide Beamer using `templates/beamer-slides.tex` as base
3. Include:
   - Title slide
   - Motivation & research question
   - Contribution (3 points)
   - Institutional background (with timeline)
   - Data description
   - Empirical strategy (with equation)
   - Main results table
   - Event study / parallel trends figure
   - Robustness summary
   - Heterogeneity
   - Conclusion & policy implications
   - Appendix backup slides

### Output

`output/beamer_slides.tex`
