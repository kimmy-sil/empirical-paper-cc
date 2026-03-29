# Empirical Paper Pipeline（实证论文生成流水线）

AI 辅助的一站式实证论文生成工具。从研究 idea 到终稿 + Beamer 幻灯片，覆盖完整研究流程。

> AI 负责把数据变成规范稿件，研究者负责把稿件变成好研究。

## Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT（三种入口）                              │
│  A) 上传数据文件 (.csv/.xlsx/.dta)                               │
│  B) 给出研究主题 ("数字普惠金融对农村消费的影响")                    │
│  C) 提供 Research Prompt（APE 风格，完整研究设计方案）              │
└───────────┬─────────────────────────┬───────────────────────────┘
            │                         │
            ▼                         ▼
┌──── STAGE 1 ────┐          ┌──── STAGE 2 ────┐
│   数据审计       │          │  研究问题生成     │
│  · 面板结构诊断   │          │  · 3-5 个候选问题  │
│  · 变量分类      │          │  · 文献快扫       │
│  · 质量检查      │    ┌─────│  · 四维评分       │
│  · 自动清洗      │    │     └────────┬────────┘
└────────┬────────┘    │              │
         │             │     ◆ 人工决策门 1：选择研究问题
         └─────────────┘              │
                                      ▼
                            ┌──── STAGE 3 ────┐
                            │  识别策略设计     │
                            │  · 方法自动路由：  │
                            │    DID / IV / RDD │
                            │    Panel FE / SC  │
                            │    DML / 时间序列  │
                            │  · 预分析计划     │
                            └────────┬────────┘
                                     │
                            ◆ 人工决策门 2：确认识别策略
                                     │
                                     ▼
                            ┌──── STAGE 4 ────┐
                            │   计量分析       │
                            │  · 描述性统计     │
                            │  · 主回归        │
                            │  · 平行趋势检验   │
                            │  · 稳健性检验     │
                            │  · 异质性分析     │
                            │  · 三语言代码输出  │
                            └────────┬────────┘
                                     │
                                     ▼
                            ┌──── STAGE 5 ────┐
                            │   论文撰写       │
                            │  · IMRaD 结构    │
                            │  · Markdown + LaTeX │
                            └────────┬────────┘
                                     │
                                     ▼
                            ┌──── STAGE 6 ────┐
                            │  审稿模拟 & 修订  │
                            │  · 5 维度打分     │
                            │  · 最多 2 轮修订  │
                            └────────┬────────┘
                                     │
                            ◆ 人工决策门 3：批准终稿
                                     │
                                     ▼
                  ┌──── STAGE 7 ────┐   ┌──── STAGE 8 ────┐
                  │    终稿输出      │   │  Beamer 幻灯片   │
                  │  · 论文 (md+tex) │   │  · 15-20 页      │
                  │  · 图表         │   │  · 含公式/三线表   │
                  │  · 复制包       │   │  · 备用幻灯片     │
                  └────────────────┘   └────────────────┘
```

## Two-Layer Architecture

### Layer 1: Pipeline Orchestrator（全流程编排器）

上面的 8 阶段流水线 + 3 个人工决策门。适合从零开始一篇完整论文。

### Layer 2: 16 Independent Skills（独立模块）

每个模块可单独调用，不必走完整 pipeline：

| 类别 | Skills | 独立调用示例 |
|------|--------|------------|
| **计量方法** (8) | DID, IV, RDD, Panel FE, Synthetic Control, OLS, ML-Causal, Time Series | "帮我跑一个 DID" |
| **数据处理** (3) | Data Cleaning, Data Fetcher, Stats | "拉一下 World Bank 的 GDP 数据" |
| **输出写作** (5) | Figure, Table, Paper Writing, Literature Review, Beamer | "生成事件研究图" |

## Features

### 因果识别方法

| 方法 | 覆盖内容 | Python | R | Stata |
|------|---------|--------|---|-------|
| DID | TWFE, Stacked DID, CS(2021), SA(2021), DDD, PSM-DID | linearmodels, pydid | fixest, did | reghdfe, csdid |
| DID 诊断 | Bacon 分解, twowayfeweights, pretrends 功效分析, HonestDiD 敏感性 | — | bacondecomp, pretrends, HonestDiD | bacondecomp, twowayfeweights |
| DID 拓展 | Synthetic DID, fect (矩阵补全反事实) | — | synthdid, fect | sdid |
| IV / 2SLS | LATE/MTE 框架, 四大流派, Bartik shift-share, 法官设计, Conley-Hansen-Rossi | linearmodels | fixest, ivreg, bartik.weight | ivreghdfe, ssaggregate |
| RDD | Sharp, Fuzzy, RDiT (时间断点), 地理 RDD, 多门槛 RDD | rdrobust | rdrobust, rddensity, rdmc | rdrobust |
| Panel FE | Entity/Time/Two-way FE, 高维 FE, Hausman, Driscoll-Kraay SE | linearmodels | fixest, plm | reghdfe, xtreg |
| Synthetic Control | Abadie SCM, 置换推断, gsynth | SparseSC | Synth, tidysynth, gsynth | synth |
| DML | PLM vs IRM 选择指南, Neyman 正交性, 与 IV/DID/RDD 嵌套, 六条实践指南 | econml, doubleml | DoubleML | — |
| Causal Forest | CATE 估计, 变量重要性 | econml | grf | — |
| Time Series | ADF/PP/KPSS, 协整, VAR/VECM, Granger, IRF, ARDL, 结构突变 | statsmodels | vars, urca | var, vec |

### 方法论深度（不只是代码模板）

- **DID**：Goodman-Bacon 分解原理、负权重问题解释、Roth(2022) 功效分析、Rambachan-Roth(2023) 敏感性边界
- **IV**：Imbens-Angrist(1994) LATE 精确定义、Heckman-Vytlacil(2005) MTE 统一框架、IV 四大流派分类与外生性来源比较、Bartik shift-share 的三场方法论争论
- **RDD**：五种断点类型分类（分数/年龄/时间/地理/指标）、RDiT 时间自相关处理、地理 RDD 的 GIS 距离计算
- **DML**："先识别后估计"的准确定位、PLM vs IRM 的选择判断、何时不该用 DML

### 其他功能

- Python / R / Stata 三语言代码模板，全部可复用
- LaTeX 论文模板（AER/QJE 风格）+ Beamer 学术汇报模板
- 自动从 FRED / World Bank / Census / OECD API 拉取公开数据
- 中国数据库字段指引（CSMAR / CNRDS / Wind / 国统局）
- APE 风格 Research Prompt 模板（含 3 个完整示例）
- Balance Table 专项模块（标准化差异 + 匹配前后对比）
- Dataset-aware 假设生成（先审计数据再提假设，避免幻觉）
- 内置审稿模拟（5 维度打分，最多 2 轮修订）

## Quick Start

### 安装

```bash
git clone https://github.com/kimmy-sil/empirical-paper-cc.git
cd empirical-paper-cc
```

### 在 Claude Code 中使用

```bash
claude   # 启动 Claude Code，自动加载所有 skills
```

### 全流程模式

```
> 我有一份省级面板数据（2010-2022），想研究数据要素市场化对经济增长的影响
```

Claude 从 Stage 1 开始，自动走完 8 阶段。在 3 个决策门会停下来等你确认。

### 单模块模式

```
> 帮我跑一个 DID，处理变量是 policy，处理时间是 2018 年
> 生成一个事件研究图
> 帮我做个 Beamer 汇报
> 拉一下 World Bank 的 GDP 数据
> 帮我做一个 Balance Table
> 用 Callaway-Sant'Anna 跑交错 DID
```

### Research Prompt 模式

提供一份 APE 风格的 Research Prompt（参考 `empirical-pipeline/templates/research-prompt-template.md`），跳过 Stage 1-2，直接从 Stage 3 开始。

### 在 Cursor / Windsurf 中使用

将 `.claude/CLAUDE.md` 内容合并到项目的 rules 文件中。

## Project Structure

```
.claude/
  CLAUDE.md                          # 路由规则 + skills 目录

empirical-pipeline/                  # Layer 1: 全流程编排器
  SKILL.md                           # 8 阶段 pipeline 指令
  scripts/
    did_analysis.py                  # Python DID 模板 (linearmodels)
    did_analysis.R                   # R DID 模板 (fixest + did)
    did_analysis.do                  # Stata DID 模板 (reghdfe + csdid)
    fetch_data.py                    # 自动数据获取 (FRED/WB/Census/OECD)
  templates/
    paper-latex.tex                  # LaTeX 论文模板 (AER/QJE 风格)
    beamer-slides.tex                # Beamer 幻灯片模板
    research-prompt-template.md      # APE 风格 Research Prompt 模板 (含 3 个示例)
    paper-structure.md               # 论文结构指南
    pre-analysis-template.md         # 预分析计划模板
    question-template.md             # 研究问题评分模板
  references/
    did-methodology.md               # DID 方法论 (含交错 DID)
    panel-fe-methods.md              # 面板固定效应
    other-methods.md                 # RDD, IV, PSM, Synthetic Control
    data-audit-checklist.md          # 数据审计清单
    data-sources.md                  # 公开数据源目录

skills/                              # Layer 2: 16 个独立模块
  did-analysis/SKILL.md              # 1286 行 — DID 全套 + Stacked/honestdid/pretrends/fect/sdid
  iv-estimation/SKILL.md             # 918 行 — IV + LATE/MTE/四大流派/Bartik/Conley-Hansen-Rossi
  rdd-analysis/SKILL.md              # 843 行 — RDD + RDiT/地理RDD/多门槛
  panel-data/SKILL.md                # 面板 FE + 高维 FE + Hausman
  synthetic-control/SKILL.md         # SCM + 置换推断 + gsynth
  ols-regression/SKILL.md            # OLS + VIF + 异方差检验
  ml-causal/SKILL.md                 # 1049 行 — DML(PLM/IRM) + Causal Forest + 六条实践指南
  time-series/SKILL.md               # 单位根/协整/VAR/IRF/ARDL/结构突变
  data-cleaning/SKILL.md             # 缺失值/异常值/面板平衡/变量构造
  data-fetcher/SKILL.md              # FRED/World Bank/Census/OECD API
  stats/SKILL.md                     # Table 1 / 相关系数 / 统计检验
  figure/SKILL.md                    # 事件研究图/系数图/RDD图/SC图
  table/SKILL.md                     # 三线表 / 回归表 / LaTeX 排版
  paper-writing/SKILL.md             # IMRaD 结构 / 学术写作规范
  literature-review/SKILL.md         # 文献搜索 / 综述写作 / BibTeX
  beamer-ppt/SKILL.md                # Beamer 学术汇报幻灯片
```

## Output Structure

运行完整 pipeline 后生成的文件：

```
output/
├── data_audit_report.md             # 数据审计报告
├── candidate_questions.md           # 研究问题候选（含评分）
├── pre_analysis_plan.md             # 预分析计划
├── analysis_log.md                  # 分析日志
├── tables/
│   ├── table1_descriptive.md        # 描述性统计（Balance Table）
│   ├── table2_main_results.md       # 主回归结果
│   ├── table3_robustness.md         # 稳健性检验
│   └── table4_heterogeneity.md      # 异质性分析
├── figures/
│   ├── fig1_trends.png              # 平行趋势图
│   ├── fig2_event_study.png         # 事件研究图
│   └── fig3_coefficients.png        # 系数图
├── paper_draft.md                   # 论文初稿 (Markdown)
├── paper_final.tex                  # 论文终稿 (LaTeX)
├── beamer_slides.tex                # Beamer 幻灯片
├── review_round1.md                 # 审稿意见
├── revision_response.md             # 修改回应
└── replication/
    ├── analysis.py                  # Python 复制代码
    ├── analysis.R                   # R 复制代码
    ├── analysis.do                  # Stata 复制代码
    └── README.md                    # 运行说明
```

## Design Principles

1. **Dataset-aware**：先审计数据再生成假设，不会提出数据不支持的研究问题
2. **Human-in-the-loop**：3 个决策门确保研究者对关键判断拥有最终决定权
3. **方法论理解优先于代码执行**：每个 skill 不只是代码模板，还包含方法论原理、适用条件、常见错误
4. **DML 是工具不是策略**：明确区分识别策略（DID/IV/RDD）和估计方法（DML），防止误用
5. **交错 DID 不盲目用 TWFE**：自动路由到 CS(2021)/SA(2021)，附 Bacon 分解诊断

## Inspired By

- [APE](https://github.com/SocialCatalystLab/ape-papers) — Automated Policy Evaluation (Social Catalyst Lab)
- [HLER](https://github.com/maxwell2732/hler-working-papers) — Human-LLM Empirical Research
- [academic-research-skills](https://github.com/Imbad0202/academic-research-skills) — Academic Research Skills for Claude Code
- [stata-skill](https://github.com/dylantmoore/stata-skill) — Stata expertise for AI agents

## License

MIT

## Author

kim ([@kimmy-sil](https://github.com/kimmy-sil))
