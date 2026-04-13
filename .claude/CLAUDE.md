# Empirical Paper Pipeline

## 1. 项目定位
Design-based / Reduced-form 因果推断 pipeline。从研究 idea 或数据到完整论文。

覆盖：DID / IV / RDD / Panel FE / GMM / SC / DML / Causal Forest / 时间序列 / 机制分析
不覆盖：结构估计（BLP/DSGE）、空间计量（SAR/SDM）、反事实政策模拟

## 2. 用户输入与流程
用户输入（二选一即可启动）：
- 数据文件（.csv/.dta/.xlsx）→ 数据探索模式
- 研究想法（自然语言）→ 方法咨询模式
- 两者都有 → 完整模式

推荐补充：变量说明、参考文献3-5篇、识别策略偏好、目标期刊

Pipeline: S0:Design → S1:DataClean → S2:Descriptive → S3:BaseRegression → ◆Gate1 → S4:Robustness+Mechanism → S5:Paper+Output

Gate 1（必停）：数据概况 + 识别策略 + Estimand声明 → 等用户确认
Gate 2（条件触发）：基准回归异常时弹出

Stage 0 行为：
- 完整模式 → 数据诊断 + 策略匹配 → Gate 1
- 单输入模式 → 提供建议 → Gate 1 停住等补全
- 用户随时可补全，系统自动切换完整模式

## 3. 方法论红线
- 必须声明 Estimand（ATT/LATE/ATE/ATO）
- TWFE 必须先 Bacon 分解，负权重≥10% → CS/SA
- 聚类层级匹配处理分配层级
- RDD 多项式 ≤ 2 阶
- GMM 工具变量数 ≤ N
- Causal Forest = 异质性工具，NOT 因果识别
- DML = 估计工具，NOT 识别策略
- 不能把 FE 哑变量当高维X放入DML-PLM
- 机制分析前必须画 DAG
- 禁止"坏控制变量"（后定/对撞/中介）
- 中文写作：不用"值得注意的是""综上所述""取决于"，不用感叹号

## 4. 数据规范
输入：CSV/Excel/Stata(.dta)
面板数据须含：个体ID + 时间
连续变量默认1%/99%缩尾
输出：output/（tables/ figures/ replication/）
代码：Python + R（不含Stata）

## 5. Skill 路由表

> **优先级：5a → 5b → 5c。必须按顺序执行，不可跳步。**

### 5a. 数据层次判断（所有路由的前提）

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

### 5b. 数据结构 → Skill 路由

| 数据结构 | 路由 |
|---------|------|
| 真正面板（同一批个体 × 多期） | `panel-data` |
| 纯截面（每个个体只有一期） | `ols-regression` |
| 单截面多时期（N=1，T>1） | `time-series` |
| 混合截面（不同批，有 year 列但非追踪） | `ols-regression`，⚠️ 禁止 FE |

**新手常见陷阱（进入 skill 前必读）：**
- 🚨 混合截面 ≠ 面板：不同年份重新抽样的数据，格式上有 year 列，也不是面板
- 🚨 长面板 / 时序不检验平稳性 = 伪回归风险（R² 虚高，结果无效）
- 🚨 核心变量 X 几乎不随时间变化 → FE 会吸收 X，系数不显著 ≠ 无效，先做 within 方差分解
- 🚨 工具变量外生性不能靠统计检验，只能靠理论论证

### 5c. 识别策略 → Skill 路由

- 政策交错实施 → `did-analysis`
- 数值门槛 → `rdd-analysis`
- 合理工具变量 → `iv-estimation`
- 含滞后因变量 + T 小 → `panel-data` GMM 子文件
- 处理单位 ≤ 5 → `synthetic-control`
- 高维协变量 → `ml-causal`
- 纯横截面无因果设计 → `ols-regression`
- 主回归后机制分析 → `mechanism-analysis`
- ⚠️ 以上都不适用 → 告知用户，不强行套用

### 5d. Sub-agent 路由

| 触发条件 | 调用 agent |
|---------|------------|
| Stage 1 开始前 / `/audit` | `data-auditor`（只读审计，输出隔离） |
| 代码报错 / 结果异常 / 跨语言验证 UNRESOLVED | `debug-agent`（根因 → 最小修复 → 验证） |
| 投稿前 `/peer [journal]` | `methods-referee`（审稿模拟 + Python/R 双验证） |

辅助 skill：`data-cleaning` / `data-fetcher` / `stats` / `figure` / `table` / `paper-writing` / `literature-review` / `beamer-ppt`

## 6. 论文输出规范
双格式：Markdown + LaTeX
表格：\input{} 引用代码输出（禁止手动抄数）
图表：DPI≥300, 字号≥12pt, 黑白友好
引用：APA 7 或 GB/T 7714
Stage 5 强制 Estimand 声明段
