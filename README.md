# empirical-paper-cc

一站式实证论文生成 Claude Code Skill

> 扔数据 + 说想法 → 出论文。内置因果推断防护网。

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

---

## Prerequisites

- Claude Code 订阅（Claude Max 或 Pro）
- Node.js（Claude Code 运行环境）
- Python 3.8+（pandas, numpy, matplotlib, statsmodels, linearmodels, doubleml, econml）
- R（可选，用于 fixest, did, rdrobust, DoubleML, grf, scpi 等）

---

## Workflow

```
用户输入（数据 / 想法 / 两者）
        │
        ▼
  S0: 研究设计协作
  ├── 因果问题拆解
  ├── DAG 绘制
  ├── 识别策略匹配
  ├── 文献定位
  └── Research Design Memo
        │
   ◆ Gate 1（必停）
   确认：数据 + 策略 + Estimand
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
  S4: 稳健性 + 机制 + 异质性
  必做层 ▸ 推荐层 ▸ 情境层
  内生性处理 ▸ 机制分析（DAG先行）▸ 异质性
        │
        ▼
  S5: 论文 + 输出
  paper-writing ▸ LaTeX ▸ 审稿模拟 ▸ Beamer（可选）
```

**Gate 1**（必停）：确认数据概况 + 识别策略 + Estimand，等用户明确确认方可继续。

**Gate 2**（条件触发）：系数符号异常 / 量级失控 / 预趋势显著失败时弹出，停止并报告。

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
| mechanism-analysis | 5种机制方法 / DAG / 坏控制警告 | SKILL |

### 数据处理 Skills (3)

| Skill | 功能 | 文件 |
|-------|------|------|
| data-cleaning | 缺失机制 / Winsorize / 面板平衡 | SKILL + ADVANCED |
| data-fetcher | FRED / WB / OECD / akshare | SKILL + ADVANCED |
| stats | Table1 / Balance Table / VIF / 缺失模式 | SKILL + ADVANCED |

### 输出 Skills (5)

| Skill | 功能 | 文件 |
|-------|------|------|
| figure | 事件研究图 / 系数图 / RDD图 / 密度图 | SKILL |
| table | 三线表 / 回归表 / LaTeX booktabs | SKILL |
| paper-writing | IMRaD / 写作规范 / Estimand声明段 | SKILL |
| literature-review | 文献综述写作 / BibTeX / 贡献定位 | SKILL |
| beamer-ppt | Beamer 学术汇报 / 15-20页模板 | SKILL |

---

## Key Design Principles

**1. Estimand 声明贯穿全流程**
所有 9 种方法 skill 均要求在输出中明确声明 Estimand（ATT/LATE/ATE/ATO），并给出形式化定义和识别假设。

**2. 稳健性三层分层**
- 必做层：绑定识别策略，不可省略（DID→Bacon+HonestDiD，IV→AR推断+plausexog，RDD→密度+带宽）
- 推荐层：每项附经济学理由，非默认全做
- 情境层：由数据特征自动触发（大量零值→ppmlhdfe，小样本→野bootstrap）

**3. DAG 持久性**
Stage 0 画一次 DAG，在 Stage 3 控制变量审核和 Stage 4 机制分析中复用。避免反复讨论同一变量是否应该控制。

**4. Causal Forest ≠ 因果识别**
Causal Forest 只能在主回归已建立因果效应后，用于探索异质性。不能作为初始因果识别工具使用。

**5. 方法论边界透明**
Reduced-form only。需要结构估计（BLP/DSGE/IO）或空间计量（SAR/SDM）时，系统明确告知超出范围，而非强行适配。

**6. 禁止坏控制变量**
系统在 Stage 3 前强制审核所有控制变量（后定/对撞/中介）。发现后定或对撞变量时，阻断回归并给出理由。

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
├── tables/          # .tex 三线表（\input{} 引用）
├── figures/         # .pdf / .png 图形（DPI≥300）
└── replication/     # Python + R 可复现脚本
```

表格规范：所有数字通过代码生成后 `\input{}` 引用，禁止手动抄数。

图形规范：DPI≥300，字号≥12pt，黑白友好（线型区分色彩区分并存）。

---

## License

MIT

## Author

kim ([@kimmy-sil](https://github.com/kimmy-sil))
