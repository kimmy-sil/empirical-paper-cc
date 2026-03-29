# Empirical Paper Pipeline (实证论文生成流水线)

一站式实证论文生成工具，覆盖从数据审计到终稿输出的完整研究流程。

## Architecture: Two-Layer Design

**Layer 1 — Pipeline Orchestrator（全流程编排器）**

统一编排 8 个阶段，带人工决策门。适合从零开始一篇完整论文。

```
S0:Intake → S1:DataAudit → S2:QuestionGen → ◆SELECT → S3:PreAnalysis → ◆CONFIRM
→ S4:Analysis → S5:Drafting → S6:Review → ◆APPROVE → S7:Finalize → S8:Beamer
```

**Layer 2 — Independent Skills（独立模块）**

每个 skill 可单独调用，也可被 pipeline 编排器在对应阶段自动调用。

## Skills Catalog

### 计量方法

| Skill | 功能 | 独立调用场景 |
|-------|------|-------------|
| `did-analysis` | DID / 交错DID / PSM-DID / DDD | 只需跑个DID |
| `iv-estimation` | 2SLS / 弱IV检验 / 过度识别 | 需要IV估计 |
| `rdd-analysis` | Sharp/Fuzzy RDD / rdrobust | 断点回归分析 |
| `panel-data` | 面板FE / 高维FE / Hausman | 面板回归 |
| `synthetic-control` | SCM / 置换推断 / gsynth | 合成控制 |
| `ols-regression` | OLS / 逐步回归 / VIF | 基础回归 |
| `ml-causal` | DML / Causal Forest / LASSO | ML因果推断 |
| `time-series` | 单位根 / 协整 / VAR / IRF | 时间序列 |

### 数据处理

| Skill | 功能 | 独立调用场景 |
|-------|------|-------------|
| `data-cleaning` | 缺失值 / 异常值 / 面板平衡 | 清洗数据 |
| `data-fetcher` | FRED / World Bank / Census API | 自动拉数据 |
| `stats` | Table 1 / 相关系数 / 统计检验 | 描述性统计 |

### 输出与写作

| Skill | 功能 | 独立调用场景 |
|-------|------|-------------|
| `figure` | 事件研究图 / 系数图 / RDD图 | 生成图表 |
| `table` | 三线表 / 回归表 / LaTeX排版 | 做表格 |
| `paper-writing` | IMRaD结构 / 学术写作规范 | 写论文 |
| `literature-review` | 文献搜索 / 综述写作 / BibTeX | 文献综述 |
| `beamer-ppt` | 学术汇报Beamer幻灯片 | 做PPT |

## Routing Rules

### 全流程模式
- 用户给出模糊 idea（如 "AI 对就业的影响"）→ Pipeline Stage 0（需求诊断）开始
- 用户上传数据文件（CSV/Excel/Stata）→ Pipeline Stage 1 开始
- 用户给出 Research Prompt → Pipeline Stage 3 开始
- 用户已有初稿 → 跳至 Pipeline Stage 6

### 单模块模式
- 用户说"跑个DID" → 直接调用 `did-analysis` skill
- 用户说"帮我做描述性统计" → 直接调用 `stats` skill
- 用户说"生成事件研究图" → 直接调用 `figure` skill
- 用户说"做个Beamer" → 直接调用 `beamer-ppt` skill
- 用户说"帮我拉数据" → 直接调用 `data-fetcher` skill
- 用户说"写文献综述" → 直接调用 `literature-review` skill

### 方法选择路由（Pipeline Stage 3 自动调用）
- 有政策冲击 + 处理/对照组 → `did-analysis`
- 有运行变量 + 断点 → `rdd-analysis`
- 内生性 + 外生工具 → `iv-estimation`
- 面板数据 + 固定效应 → `panel-data`
- 单一处理单位 → `synthetic-control`
- 高维控制变量 / 异质性探索 → `ml-causal`
- 时间序列 / 宏观数据 → `time-series`
- 截面数据基础分析 → `ols-regression`

## Multi-Language Code Output

所有计量方法 skill 均提供三语言代码模板：
- **Python**: pandas + statsmodels + linearmodels (默认)
- **R**: tidyverse + fixest + did + modelsummary
- **Stata**: reghdfe + csdid + esttab

用户可在任何阶段指定语言偏好，默认 Python。

## Key Rules

- 所有假设必须 dataset-aware（基于数据可行性约束）
- DID 必做平行趋势检验 + 安慰剂检验
- 交错 DID 不得盲目使用 TWFE，必须使用 CS(2021)/SA(2021) 等新方法
- 标准误聚类层级必须匹配处理层级
- 正文使用完整段落散文，不使用项目符号列表
- 参考文献 ≥ 25 篇
- 中文写作避免 AI 腔（不用"值得注意的是"、"综上所述"、"取决于"）
- 不过度使用破折号、不用感叹号

## Output Structure

```
output/
├── data_audit_report.md
├── candidate_questions.md
├── pre_analysis_plan.md
├── analysis_log.md
├── tables/
├── figures/
├── paper_draft.md          (Markdown)
├── paper_final.tex         (LaTeX)
├── beamer_slides.tex       (Beamer)
├── review_round1.md
├── revision_response.md
├── paper_final.md
└── replication/
    ├── analysis.py
    ├── analysis.R
    ├── analysis.do
    └── README.md
```

## Default Preferences

- **Language**: 中文（除非用户指定英文）
- **Analysis Tool**: Python (默认) — 可切换到 R 或 Stata
- **Citation Style**: APA 7 或 GB/T 7714
- **Domain Focus**: 数据治理、AI Agent、数字经济、公共管理、组织行为
