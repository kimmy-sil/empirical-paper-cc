# Empirical Paper Pipeline

## 1. 项目定位

Design-based / Reduced-form 因果推断的一站式辅助 pipeline。从研究 idea 或数据到完整论文。

**覆盖**：DID（含交错DID全套）、IV/2SLS（含LATE/MTE/Bartik）、RDD（含RDiT/地理RDD/RD-DD）、Panel FE、Synthetic Control、DML（PLM/IRM/DynamicDML）、Causal Forest（异质性探索）、System GMM（动态面板）、PSM、时间序列

**不覆盖**：结构估计（BLP/DSGE/选择模型/MLE恢复效用函数）、空间计量（SAR/SDM/SEM）、一般均衡模型、反事实政策模拟

当用户的研究问题需要以上"不覆盖"的方法时，Stage 0 应明确告知。

## 2. 用户输入与流程

用户输入（二选一即可启动）：
- 数据文件（.csv/.dta/.xlsx）→ 进入数据探索模式
- 研究想法（自然语言）→ 进入方法咨询模式
- 两者都有 → 进入完整模式，直接执行全流程

推荐补充（非必须）：
- 变量说明（Y/X/控制变量）
- 参考文献 3-5 篇
- 识别策略偏好、聚类层级偏好、目标期刊

Stage 0 行为：
- 完整模式 → 数据诊断 + 策略匹配 → Gate 1 确认后继续
- 单输入模式 → 尽可能提供建议 → Gate 1 停住，等用户补全另一半
- 用户随时可补全缺失的输入，系统自动切换到完整模式继续

Pipeline 流程：
S0:Design → S1:DataAudit → S2:QuestionGen → S3:PreAnalysis → ◆Gate1 → S4:Analysis → S5:Drafting → S6:Review → ◆Gate2 → S7:Finalize → S8:Beamer

Gate 1（必停）：呈现数据概况 + 识别策略推荐 + Estimand声明 → 等用户确认
Gate 2（条件触发）：审稿模拟发现重大问题时弹出，否则静默继续
其他阶段：静默执行，不停

## 3. 方法论约束（红线，绝对不能违反）

- 必须声明 Estimand（估计的是什么效应：ATT/LATE/ATE/ATO）
- TWFE 必须先做 Bacon 分解，负权重>10%时切换 CS/SA
- 聚类标准误层级必须匹配处理分配层级
- RDD 多项式阶数不超过 2 阶（推荐1阶）
- GMM 工具变量数不得超过截面单位数 N
- 不能声称"验证了因果关系"除非有可信的识别策略
- 控制变量不加"坏控制"（后定变量/对撞变量/中介变量）
- Causal Forest 不是因果识别方法，只能在主回归建立因果后用于异质性探索
- DML 是估计工具不是识别策略，不能替代 IV/DID/RDD
- 不能将 FE 作为高维协变量放入 DML-PLM 再解读为 DID
- 中文写作避免 AI 腔（不用"值得注意的是""综上所述""取决于"，不过度使用破折号，不用感叹号）

## 4. 数据规范

- 输入格式：CSV/Excel/Stata (.dta)
- 面板数据须含：个体ID列 + 时间列
- 连续变量默认1%/99%缩尾
- 输出目录：output/（tables/ figures/ replication/）
- 代码语言：Python + R（本轮不含Stata）

## 5. Skill 路由表

### 计量方法（9）
| Skill | 功能 | 独立调用 |
|-------|------|---------|
| did-analysis | DID/交错DID/PSM-DID/DDD/Stacked/BJS/dCDH | "跑个DID" |
| iv-estimation | 2SLS/LATE/MTE/Bartik/plausexog/Lee bounds | "做IV估计" |
| rdd-analysis | Sharp/Fuzzy/RDiT/地理RDD/RD-DD | "断点回归" |
| panel-data | Panel FE/RE/Hausman/GMM-lite/门槛效应/ppmlhdfe | "面板回归" |
| synthetic-control | SCM/置换推断/gsynth | "合成控制" |
| ols-regression | OLS/逐步回归/VIF/Spec Curve/Oster | "跑回归" |
| ml-causal | DML(PLM/IRM)/DynamicDML/Causal Forest/SHAP | "DML分析" |
| time-series | 单位根/协整/VAR/IRF/ARDL | "时间序列" |
| mechanism-analysis | 7种机制分析方法/DAG/坏控制变量警告 | "做机制检验" |

### 数据处理（3）
| Skill | 功能 | 独立调用 |
|-------|------|---------|
| data-cleaning | 缺失值/异常值/面板平衡/Heckman/缺失机制诊断 | "清洗数据" |
| data-fetcher | FRED/World Bank/Census/OECD API | "拉数据" |
| stats | Table 1/相关系数/统计检验/Balance Table | "描述性统计" |

### 输出与写作（5）
| Skill | 功能 | 独立调用 |
|-------|------|---------|
| figure | 事件研究图/系数图/RDD图/安慰剂分布图 | "画图" |
| table | 三线表/回归表/LaTeX排版 | "做表格" |
| paper-writing | IMRaD结构/学术写作规范/Estimand声明段 | "写论文" |
| literature-review | 文献搜索/综述/BibTeX | "文献综述" |
| beamer-ppt | Beamer学术汇报 | "做PPT" |

### 方法匹配路由（Stage 0 自动调用）
- 政策交错实施 + 处理/对照组 → did-analysis
- 明确数值门槛 → rdd-analysis
- 有合理工具变量 → iv-estimation
- 面板 + 固定效应 → panel-data
- 含滞后因变量 + T小 → panel-data (GMM-lite)
- 单一处理单位 → synthetic-control
- 高维协变量 + 需要灵活控制 → ml-causal
- 时间序列/宏观 → time-series
- ⚠️ 以上都不适用 → 告知用户"当前没有可信的因果推断路径"

## 6. 论文输出规范

- 双格式：Markdown + LaTeX
- 表格：LaTeX \input{} 直接引用代码输出（禁止手动抄数）
- 图表：DPI≥300，字号≥12pt，黑白友好
- 引用：APA 7 或 GB/T 7714
- Stage 5 强制插入 Estimand 声明段（论文 Section 3）
- 中文风格：经管期刊，专业但不晦涩
