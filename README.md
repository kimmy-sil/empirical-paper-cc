# Empirical Paper Pipeline（实证论文生成流水线）

一站式实证论文生成工具 for Claude Code。从数据审计到终稿输出 + Beamer 幻灯片，覆盖完整研究流程。

## Two-Layer Architecture

### Layer 1: Pipeline Orchestrator（全流程编排器）

8 阶段流水线 + 3 个人工决策门，确保研究规范性：

```
数据审计 → 研究问题 → ◆选择 → 预分析计划 → ◆确认
→ 计量分析 → 论文撰写 → 审稿修订 → ◆批准 → 终稿 → Beamer
```

### Layer 2: Independent Skills（16 个独立模块）

每个模块可独立调用，也可被 pipeline 自动编排：

**计量方法** (8)：DID、IV、RDD、面板FE、合成控制、OLS、ML因果推断、时间序列

**数据处理** (3)：数据清洗、自动拉数据、描述性统计

**输出与写作** (5)：图表、表格、论文写作、文献综述、Beamer 幻灯片

## Features

- 支持 DID / IV / RDD / Panel FE / Synthetic Control / OLS / DML / Causal Forest
- 交错 DID 自动使用 Callaway-Sant'Anna / Sun-Abraham（不盲目用 TWFE）
- Python / R / Stata 三语言代码模板，全部可复用
- LaTeX 论文模板（AER/QJE 风格）+ Beamer 学术汇报模板
- 自动从 FRED / World Bank / Census 等 API 拉取公开数据
- Dataset-aware 假设生成（先审计数据再提假设，避免幻觉）
- 内置审稿模拟（5 维度打分，最多 2 轮修订）

## Quick Start

### 全流程模式

```
> 我有一份省级面板数据（2010-2022），想研究数据要素市场化对经济增长的影响
```

Claude 会从 Stage 1 开始，自动走完 8 个阶段。

### 单模块模式

```
> 帮我跑一个 DID，处理变量是 policy，处理时间是 2018 年
> 生成一个事件研究图
> 帮我做个 Beamer 汇报
> 拉一下 World Bank 的 GDP 数据
```

直接调用对应的独立 skill。

## Project Structure

```
.claude/
  CLAUDE.md                  # 路由规则 + skills 目录

empirical-pipeline/          # Layer 1: 全流程编排器
  SKILL.md                   # 8 阶段 pipeline 指令
  scripts/
    did_analysis.py          # Python DID 模板
    did_analysis.R           # R DID 模板 (fixest)
    did_analysis.do          # Stata DID 模板 (reghdfe)
    fetch_data.py            # 自动数据获取
  templates/
    paper-latex.tex          # LaTeX 论文模板
    beamer-slides.tex        # Beamer 幻灯片模板
    paper-structure.md       # 论文结构指南
    pre-analysis-template.md # 预分析计划模板
    question-template.md     # 研究问题模板
  references/
    did-methodology.md       # DID 方法论
    panel-fe-methods.md      # 面板固定效应
    other-methods.md         # RDD, IV, PSM, SC
    data-audit-checklist.md  # 数据审计清单
    data-sources.md          # 公开数据源

skills/                      # Layer 2: 16 个独立模块
  did-analysis/SKILL.md
  iv-estimation/SKILL.md
  rdd-analysis/SKILL.md
  panel-data/SKILL.md
  synthetic-control/SKILL.md
  ols-regression/SKILL.md
  ml-causal/SKILL.md
  time-series/SKILL.md
  data-cleaning/SKILL.md
  data-fetcher/SKILL.md
  stats/SKILL.md
  figure/SKILL.md
  table/SKILL.md
  paper-writing/SKILL.md
  literature-review/SKILL.md
  beamer-ppt/SKILL.md
```

## Installation

### Claude Code

```bash
git clone https://github.com/kimmy-sil/empirical-paper-cc.git
cd empirical-paper-cc
claude   # 启动 Claude Code
```

Claude 会自动读取 `.claude/CLAUDE.md` 和所有 skills。

### Cursor / Windsurf

将 `.claude/CLAUDE.md` 内容合并到项目的 rules 文件中。

## Supported Methods

| Method | Package (Python) | Package (R) | Package (Stata) |
|--------|-----------------|-------------|-----------------|
| DID (TWFE) | linearmodels | fixest | reghdfe |
| Staggered DID | pydid | did, fixest(sunab) | csdid, did_multiplegt |
| IV / 2SLS | linearmodels | fixest, ivreg | ivreghdfe |
| RDD | rdrobust | rdrobust | rdrobust |
| Panel FE | linearmodels | fixest, plm | reghdfe, xtreg |
| Synthetic Control | SparseSC | Synth, tidysynth | synth |
| DML | econml, doubleml | DoubleML | — |
| Causal Forest | econml | grf | — |
| Time Series | statsmodels | vars, urca | var, vec |

## License

MIT

## Author

kim
