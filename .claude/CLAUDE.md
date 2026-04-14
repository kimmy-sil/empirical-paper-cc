# Empirical Paper Pipeline

## 1. 项目定位
Design-based / Reduced-form 因果推断 pipeline。从研究 idea 或数据到完整论文。

覆盖：DID / IV / RDD / Panel FE / GMM / SC / DML / Causal Forest / 时间序列 / 机制分析
不覆盖：结构估计（BLP/DSGE）、空间计量（SAR/SDM）、反事实政策模拟

## 2. 用户输入与流程
用户输入（任一即可启动）：
- 数据文件（.csv/.dta/.xlsx）→ 数据探索模式
- 研究想法（自然语言）→ 方法咨询模式
- 两者都有 → 完整模式

推荐补充：变量说明、参考文献3-5篇、识别策略偏好、目标期刊

Pipeline: S0:Design → S1:DataClean → S2:Descriptive → S3:BaseRegression → ◆Gate1 → S4:Robustness+Mechanism → S5:Paper+Output

### Stage 0 行为
- 完整模式 → 数据诊断 + 策略匹配 → Gate 1
- 仅数据 → 数据诊断 → Gate 1 停住等补全想法
- 仅想法 → 方法咨询模式（见下方主动提问流程）→ Gate 1 停住，等用户选择：
  - 用户有自己的数据 → 上传后转完整模式
  - 用户无数据 → 触发 `data-fetcher`，按研究问题匹配可用数据源（CNRDS/CSMAR/Wind/FRED/World Bank/CCER 等），输出数据源清单，用户确认后转完整模式
- 用户随时可补全，系统自动切换完整模式

### 方法咨询模式：主动提问（不可跳步）
Q1：研究问题是什么？（X 是什么，Y 是什么，研究对象是谁）
Q2：有没有可以把控处理/政策变异的外生来源？（用于判断 IV/DID/RDD 可行性）
Q3：数据可及性？（有自己的数据 / 需要外接数据源 / 还不确定）
→ 三问回答后生成研究设计备选方案 × 2-3，列出每种方案的识别假设强度、数据需求、发表难度
→ 用户选定方案后进入 Gate 1

### Gate 1（必停）：三档分流

**档位A — 因果推断模式（完整流程）**
触发条件：用户能说清 X→Y + 外生变异来源
Gate 1 要求：识别假设的理论依据（非统计检验）→ 确认后进入 S3-S5 全流程
论文输出：因果语言（"effect of X on Y"）

**档位B — 有假设、无识别（辅助补全模式）**
触发条件：用户有 X→Y 方向，但说不清外生性 / 回答"不确定"
Gate 1 行为：
① 自动触发 `literature-review`，检索同领域识别策略先例
② 输出「识别方案建议 × 2-3」（含每种方案的数据需求 + 假设强度）
③ 用户选定 → 升级为档位A；用户仍无法选定 → 降级为档位C
论文输出：相关性语言 + 显式内生性讨论段

**档位C — 探索/描述模式（最小可用流程）**
触发条件：用户什么都没想清楚 / 明确表示只想看数据规律
Gate 1 行为：告知用户当前进入描述性分析，跳过识别策略
可用阶段：S1 → S2 → S3（OLS/相关性，禁止因果表述）→ S5（描述性论文）
⚠️ 禁止进入 S4（稳健性 + 机制），因为无识别策略的稳健性检验无意义
论文输出：`paper-writing` 自动切换描述性模板
升级路径：见下方回退与升级机制

Gate 2（条件触发）：基准回归异常时弹出

### 回退与升级机制
- 档位C 不是终点，用户可在任意阶段回到 Stage 0 / Gate 1
- 当用户基于 EDA / 描述性结果补充了理论、机制直觉或外生变异来源时，系统重新评估识别策略
- 满足条件则从档位C升级为档位B或A；此前的描述性统计、变量构造、样本限定与图表结果保留复用，不重复生成
- 升级时必须写入 log：记录升级原因、触发证据与用户确认

## 3. 方法论红线
- 必须声明 Estimand（ATT/LATE/ATE/ATO）（档位A/B适用）
- TWFE 必须先 Bacon 分解，负权重≥10% → CS/SA
- 聚类层级匹配处理分配层级
- RDD 多项式 ≤ 2 阶
- GMM 工具变量数 ≤ N
- Causal Forest = 异质性工具，NOT 因果识别
- DML = 估计工具，NOT 识别策略
- 不能把 FE 哑变量当高维X放入DML-PLM
- 机制分析前必须画 DAG
- 禁止"坏控制变量"（后定/对撞/中介）
- 机制分析 / 异质性分析涉及 ≥3 个子组或 ≥3 个中介变量时，必须报告 MHT 校正（默认 BH/FDR；处理组稀少时用 Romano-Wolf）；结果表须含校正前后 p 值双列
- 档位C 下禁止因果表述（effect/impact/causal），`paper-writing` 自动执行

## 4. 数据规范
输入：CSV/Excel/Stata(.dta)
面板数据须含：个体ID + 时间
连续变量默认1%/99%缩尾
输出：output/（tables/ figures/ replication/）
代码：Python + R（不含Stata）

## 5. 审计日志

系统在每个关键节点写入 `output/logs/`，生成以下文件：

`decision-log.md` — 方法选择与变更记录：
当前阶段 / 用户输入摘要 / 候选识别策略 / 最终选择 / 选择原因 / 未选其他方案的原因 / 用户是否确认 / 档位变更记录

`design-log.md` — 研究设计版本记录：
研究问题 / 样本定义 / 变量定义 / Estimand / 识别假设 / 风险点（内生性/测量误差/选择偏差等）/ 每次修改的时间戳与版本号

`analysis-log.md` — 实证分析执行记录：
数据清洗步骤 / 缺失值处理 / 缩尾标准化 / 描述统计 / 主回归模型 / 稳健性检验 / 机制异质性分析 / 每步对应输出文件路径

`results-log.md` — 关键结果记录：
主结果表 / 图形 / DID事件研究/平行趋势/安慰剂（如适用）/ 结果异常与解释 / 是否进入Gate 2

规则：
- 日志增量追加，不得覆盖旧版本
- 每次模型切换、样本变化、变量重定义，都必须写 log
- `paper-writing` 只能读取已记录在 log 中且已存在文件路径的结果，禁止编造

## 6. Skill 路由表

> **优先级：6a → 6b → 6c。必须按顺序执行，不可跳步。**

### 6a. 数据层次判断（所有路由的前提）

拿到数据后，先回答三个问题，再进入任何 skill：

**Q1：数据里有没有唯一个体 ID，且同一 ID 出现在多个时期？**
- 有 → 真正面板数据，继续 Q2
- 没有 / 每年重新抽样不同个体 → ⚠️ 混合截面，按截面处理，**禁止使用固定效应模型**

**Q2：N 和 T 的大小关系？**
- T > 20 且 T 接近或超过 N → 长面板特征，预加载 `panel-data/LONG-PANEL.md`，进入回归前先做单位根检验
- 个体观测期数不相等（T_i 不同）→ 预加载 `panel-data/UNBALANCED.md`，先诊断缺失机制（MCAR/MAR/MNAR）

**Q3：因变量 Y 是什么类型？**
- 连续变量 → 走主流程
- 二元 / 计数 / 有序变量 → 预加载 `panel-data/DISCRETE.md`

### 6b. 数据结构 → Skill 路由

| 数据结构 | 路由 |
|---------|------|
| 真正面板（同一批个体 × 多期） | `panel-data` |
| 纯截面（每个个体只有一期） | `ols-regression` |
| 单截面多时期（N=1，T>1） | `time-series` |
| 混合截面（不同批，有 year 列但非追踪） | `ols-regression`，⚠️ 禁止 FE |

### 6c. 识别策略 → Skill 路由

- 政策交错实施 → `did-analysis`
- 数值门槛 → `rdd-analysis`
- 合理工具变量 → `iv-estimation`
- 含滞后因变量 + T 小 → `panel-data` GMM 子文件
- 处理单位 ≤ 5 → `synthetic-control`
- 高维协变量 → `ml-causal`
- 纯横截面无因果设计 → `ols-regression`
- 主回归后机制分析 → `mechanism-analysis`
- ⚠️ 以上都不适用 → 告知用户，不强行套用

### 6d. Sub-agent 路由

| 触发条件 | 调用 agent |
|---------|------------|
| Stage 1 开始前 / `/audit` | `data-auditor`（只读审计，输出隔离） |
| 代码报错 / 结果异常 / 跨语言验证 UNRESOLVED | `debug-agent`（根因 → 最小修复 → 验证） |
| 投稿前 `/peer [journal]` | `methods-referee`（审稿模拟 + Python/R 双验证） |

辅助 skill：`data-cleaning` / `data-fetcher` / `stats` / `figure` / `table` / `paper-writing` / `literature-review` / `beamer-ppt`

## 7. 论文输出规范
双格式：Markdown + LaTeX
表格：\input{} 引用代码输出（禁止手动抄数）
图表：DPI≥300, 字号≥12pt, 黑白友好
引用：APA 7 或 GB/T 7714
Stage 5 强制 Estimand 声明段（档位A/B适用）

### paper-writing 输出模式
默认分部分输出，不直接整篇生成：

| 部分 | 输出文件 | 中文触发词 |
|------|---------|-----------|
| intro | `output/draft_sections/01-intro.md` | "写引言" |
| literature | `output/draft_sections/02-literature.md` | "写文献" |
| hypothesis | `output/draft_sections/03-hypothesis.md` | "写假设" |
| methods | `output/draft_sections/04-methods.md` | "写方法"/"写识别策略" |
| results | `output/draft_sections/05-results.md` | "写结果" |
| mechanism | `output/draft_sections/06-mechanism.md` | "写机制" |
| discussion | `output/draft_sections/07-discussion.md` | "写讨论" |
| conclusion | `output/draft_sections/08-conclusion.md` | "写结论" |
| abstract | `output/draft_sections/09-abstract.md` | "写摘要" |

合并流程（用户触发"合并全文"）：
1. section-level review：检查逻辑一致性、变量名称一致、表图引用一致、因果表述是否越界
2. 合并为 `output/paper_draft.md` + `output/paper_draft.tex`
3. paper-level review：检查章节衔接、重复表述、摘要与正文一致性、结论是否超出证据

档位C 自动使用描述性模板；档位A/B 才允许 methods 中出现因果识别表述