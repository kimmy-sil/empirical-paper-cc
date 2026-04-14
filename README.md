# empirical-paper-cc

一站式实证论文生成 Claude Code Skill | Empirical Paper Pipeline for Claude Code

> 扔数据 + 说想法 → 出论文。没数据也能启动。内置因果推断防护网 + 全流程审计日志。

---

## Quick Start

```bash
git clone https://github.com/kimmy-sil/empirical-paper-cc.git
cd empirical-paper-cc
claude
```

---

## Who Is This For

面向应用微观实证研究者（劳动 / 公共 / 发展 / 管理 / 城市 / 环境 / 健康经济学）。

方法论定位：Angrist-Imbens-Card 可信度革命范式（Design-based / Reduced-form）。

不覆盖：结构估计（BLP/DSGE）、空间计量（SAR/SDM）、一般均衡反事实模拟。

### 三类用户，三种入口

| 你的状态 | 怎么开始 | 系统行为 |
|---------|---------|---------|
| 有数据 + 有想法 | 上传数据 + 描述研究问题 | 完整模式：直接进入 Pipeline |
| 只有数据，还没想法 | 上传数据 | 探索模式：先跑 EDA，再补理论 |
| 只有模糊想法，没数据 | 描述研究方向 | 咨询模式：三步引导 → `data-fetcher` 推荐数据源 |

---

## Workflow

```
用户输入（数据 / 想法 / 两者 / 都没有）
        │
        ▼
  S0: 研究设计协作
  ├── 模糊用户 → 三步引导（X→Y？外生来源？数据？）
  ├── 无数据 → data-fetcher 推荐数据源
  ├── 因果问题拆解 + DAG 绘制
  ├── 识别策略匹配 + 文献定位
  └── Research Design Memo
        │
   ◆ Gate 1（必停 · 三档分流）
   ├── 档位A：因果推断模式（有 X→Y + 外生变异）
   ├── 档位B：辅助补全（有方向，识别不清 → literature-review 介入）
   └── 档位C：探索/描述模式（先看数据，随时可升级）
        │
        ▼
  S1: 数据清洗 + 描述统计
  data-cleaning ▸ stats
        │
        ▼
  S2: 描述性分析
  figure ▸ trends ▸ distributions
        │
        ▼
  S3: 基准回归
  控制变量审核（DAG）▸ 主回归 ▸ 前置检验
        │
   ◆ Gate 2（条件触发）
   结果异常时弹出
        │
        ▼
  S4: 稳健性 + 机制 + 异质性（档位A/B）
  必做层 ▸ 推荐层 ▸ 情境层
  内生性处理 ▸ 机制分析（DAG先行）▸ 异质性
        │
        ▼
  S5: 论文 + 输出（分部分 → 合并）
  paper-writing ▸ 逐段输出 ▸ 人工审查 ▸ 合并全文 ▸ 审稿模拟
```

**Gate 1**（必停 · 三档分流）：
- **档位A**：用户能说清 X→Y 和外生变异来源 → 因果推断全流程，论文使用因果语言
- **档位B**：用户有方向但识别不清 → 自动触发 `literature-review` 检索先例，生成识别方案建议 × 2-3 → 用户选定后升级为 A，无法选定降级为 C
- **档位C**：用户只想先看数据 → 描述性分析（S1-S3），禁止因果表述，禁止进入 S4；用户随时可补充理论后升级

**Gate 2**（条件触发）：系数符号异常 / 量级失控 / 预趋势显著失败时弹出，停止并报告。

**回退与升级**：档位C 用户在 EDA 后产生因果直觉，可随时回到 Gate 1 补充识别策略，已有结果保留复用。

---

## Audit Logs

全流程自动生成审计日志到 `output/logs/`，确保每一步决策可追溯：

| 日志文件 | 记录内容 |
|---------|---------|
| `decision-log.md` | 识别策略选择原因、候选方案比较、用户确认、档位变更 |
| `design-log.md` | 研究设计版本（Estimand/假设/风险点），含时间戳与版本号 |
| `analysis-log.md` | 分析执行顺序、每步对应输出文件路径 |
| `results-log.md` | 关键结果存档、异常与解释、是否触发 Gate 2 |

规则：日志增量追加，不覆盖旧版本。`paper-writing` 只能引用已落盘的日志和结果文件，禁止编造。

---

## Paper Writing: Section-by-Section

默认**分部分输出**，不直接整篇生成：

| 部分 | 输出文件 | 中文触发词 |
|------|---------|-----------|
| intro | `output/draft_sections/01-intro.md` | "写引言" |
| literature | `output/draft_sections/02-literature.md` | "写文献" |
| hypothesis | `output/draft_sections/03-hypothesis.md` | "写假设" |
| methods | `output/draft_sections/04-methods.md` | "写方法" |
| results | `output/draft_sections/05-results.md` | "写结果" |
| mechanism | `output/draft_sections/06-mechanism.md` | "写机制" |
| discussion | `output/draft_sections/07-discussion.md` | "写讨论" |
| conclusion | `output/draft_sections/08-conclusion.md` | "写结论" |
| abstract | `output/draft_sections/09-abstract.md` | "写摘要" |

用户可逐段审查、修改、退回重写。确认后触发"合并全文"：
1. **Section-level review**：逻辑一致性、变量名、表图引用、因果表述越界检查
2. **合并**为 `output/paper_draft.md` + `output/paper_draft.tex`
3. **Paper-level review**：章节衔接、重复表述、摘要与正文一致性

---

## Slash Commands

| 命令 | 功能 |
|------|------|
| `/new_project` | 启动 Stage 0 研究设计协作 |
| `/run_did <data>` | DID 全流程（含 Bacon/CS-SA/HonestDiD）|
| `/run_iv <data>` | IV 全流程（含弱工具/plausexog/AR推断）|
| `/run_rdd <data>` | RDD 全流程（含密度检验/带宽敏感性）|
| `/run_robustness <data>` | 分层稳健性检验（必做+推荐+情境）|
| `/run_mechanism <data>` | 机制分析（DAG 先行）|
| `/gen_table <results>` | 生成 LaTeX booktabs 三线表 |
| `/gen_beamer <paper>` | 生成 15-20 页 Beamer 学术汇报 |

---

## Two-Layer Architecture

```
Layer 1: Pipeline（Stage 0-5）
         ─ 端到端流程，Gate 1/2 把关
         ─ 入口：empirical-pipeline/SKILL.md

Layer 2: 17 Independent Skills
         ─ 每个 Skill 可单独调用
         ─ 目录：skills/
```

两层解耦：Pipeline 调用 Skill，Skill 也可单独使用（无需走完整 Pipeline）。

---

## Skills Catalog

### 计量方法 Skills (9)

| Skill | 功能 | 文件 |
|-------|------|------|
| did-analysis | DID / 交错DID / Bacon分解 / HonestDiD | SKILL + ADVANCED |
| iv-estimation | IV / LATE / MTE / Bartik / plausexog | SKILL + ADVANCED |
| rdd-analysis | RDD / RDiT / 地理RDD / RD-DD | SKILL + ADVANCED |
| panel-data | Panel FE / RE / Hausman / Mundlak | SKILL + GMM-DYNAMIC |
| synthetic-control | SCM / scpi / augsynth | SKILL |
| ols-regression | OLS / Spec Curve / Oster系数稳定性 | SKILL |
| ml-causal | DML / Causal Forest / DynamicDML | SKILL |
| time-series | 单位根 / VAR / VECM / ARDL / LP-IRF | SKILL |
| mechanism-analysis | 5种机制方法 / DAG / MHT校正 | SKILL |

### 数据处理 Skills (3)

| Skill | 功能 | 文件 |
|-------|------|------|
| data-cleaning | 缺失机制 / Winsorize / 面板平衡 | SKILL + ADVANCED |
| data-fetcher | FRED / WB / OECD / CNRDS / CSMAR / akshare | SKILL + ADVANCED |
| stats | Table1 / Balance Table / VIF / 缺失模式 | SKILL + ADVANCED |

### 输出 Skills (5)

| Skill | 功能 | 文件 |
|-------|------|------|
| figure | 事件研究图 / 系数图 / RDD图 / 密度图 | SKILL |
| table | 三线表 / 回归表 / LaTeX booktabs | SKILL |
| paper-writing | 分段输出 / 合并审查 / Estimand声明段 | SKILL |
| literature-review | 文献综述 / 识别策略先例检索 / BibTeX | SKILL |
| beamer-ppt | Beamer 学术汇报 / 15-20页模板 | SKILL |

---

## Key Design Principles

**1. 三档分流，不卡死用户**
Gate 1 根据用户准备程度分为因果推断 / 辅助补全 / 探索描述三档。没有识别策略也能跑，但禁止因果表述。随时可升级。

**2. Estimand 声明贯穿全流程**
所有 9 种方法 skill 均要求在输出中明确声明 Estimand（ATT/LATE/ATE/ATO），并给出形式化定义和识别假设。

**3. 稳健性三层分层**
- 必做层：绑定识别策略，不可省略（DID→Bacon+HonestDiD，IV→AR推断+plausexog，RDD→密度+带宽）
- 推荐层：每项附经济学理由，非默认全做
- 情境层：由数据特征自动触发（大量零值→ppmlhdfe，小样本→野bootstrap）

**4. 多重假设检验校正**
机制分析 / 异质性分析涉及 ≥3 个子组或中介变量时，强制 MHT 校正（BH/FDR 或 Romano-Wolf），结果表含校正前后 p 值双列。

**5. DAG 持久性**
Stage 0 画一次 DAG，在 Stage 3 控制变量审核和 Stage 4 机制分析中复用。避免反复讨论同一变量是否应该控制。

**6. 审计日志 + 分段写作**
分析结果先落盘到 logs，再由 paper-writing 按段组装。论文先分 9 部分逐段输出审查，再合并为完整 draft。

**7. 方法论边界透明**
Reduced-form only。需要结构估计（BLP/DSGE/IO）或空间计量（SAR/SDM）时，系统明确告知超出范围，而非强行适配。

---

## Methods Coverage

| 方法 | Python | R |
|------|--------|---|
| DID (TWFE / CS / SA / Stacked) | linearmodels | fixest, did |
| HonestDiD | — | HonestDiD |
| IV / 2SLS / LIML | linearmodels | fixest, ivreg |
| plausexog | — | plausexog |
| RDD Sharp / Fuzzy | rdd, rdrobust | rdrobust |
| Panel FE / RE / HDFE | linearmodels | fixest |
| System GMM | — | plm::pgmm |
| SCM | scpi_pkg | tidysynth, scpi, augsynth |
| DML (PLM / IRM) | doubleml, econml | DoubleML |
| Causal Forest | econml | grf |
| Time Series (VAR/VECM/ARDL) | statsmodels | vars, urca, ARDL |
| Local Projections IRF | statsmodels | lpirfs |

注：本 pipeline 不生成 Stata 代码。所有分析脚本输出为 Python 和 R，保存至 `output/replication/`。

---

## Output Structure

```
output/
├── draft_sections/  # 分部分论文草稿（逐段审查）
├── logs/            # 审计日志（decision / design / analysis / results）
├── tables/          # .tex 三线表（\input{} 引用）
├── figures/         # .pdf / .png 图形（DPI≥300）
└── replication/     # Python + R 可复现脚本
```

表格规范：所有数字通过代码生成后 `\input{}` 引用，禁止手动抄数。

图形规范：DPI≥300，字号≥12pt，黑白友好（线型区分色彩区分并存）。

---

## Prerequisites

- Claude Code 订阅（Claude Max 或 Pro）
- Node.js（Claude Code 运行环境）
- Python 3.8+（pandas, numpy, matplotlib, statsmodels, linearmodels, doubleml, econml）
- R（可选，用于 fixest, did, rdrobust, DoubleML, grf, scpi 等）

---

## License

MIT

## Author

kim ([@kimmy-sil](https://github.com/kimmy-sil))