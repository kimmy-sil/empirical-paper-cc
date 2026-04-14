---
name: data-cleaning
description: >
  Performs data cleaning and preprocessing for empirical research panels:
  missing value diagnosis (MCAR/MAR/MNAR) and MICE imputation,
  outlier detection (IQR/Z-score), panel balancing decisions,
  variable construction (lags/interactions/differences),
  variable transformation guardrails (log/centering/ratios/categorization),
  and safe dataset merging with diagnostics.
  Use when processing raw data files (.csv/.xlsx/.dta/.parquet),
  handling missing data, constructing or transforming variables,
  or preparing clean panel datasets before regression analysis.
---

# 数据清洗

## 概述

经管类实证研究标准数据清洗流程。
**语言**：Python + R（NO Stata）

**适用场景**：原始面板数据清洗、缺失值诊断处理、异常值检测、面板平衡化决策、变量构造、变量转换决策、数据合并诊断。

**架构**：
- Winsorize 模块在 stats skill（调用之，本 skill 不重复）
- Heckman 选择模型在分析阶段执行，本 skill 仅诊断并标注 MNAR 风险
- Step 4.5（变量转换防护栏）：决策树 + 误区警告在此，完整代码 → 见 ADVANCED.md

---

## Step 0: 环境检测 + 格式自动读取

```python
import importlib.util, os, pandas as pd

def check_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

HAS_MISSINGNO = check_package("missingno")
HAS_SKLEARN   = check_package("sklearn")

if not HAS_MISSINGNO:
    print("⚠️ missingno不可用 → 使用matplotlib/seaborn替代")
if not HAS_SKLEARN:
    print("⚠️ sklearn不可用 → MICE插补建议在R端执行")

def auto_read(filepath: str, **kwargs) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()
    readers = {
        ".csv": pd.read_csv, ".xlsx": pd.read_excel, ".xls": pd.read_excel,
        ".dta": pd.read_stata, ".parquet": pd.read_parquet, ".feather": pd.read_feather,
    }
    if ext == ".sav":
        import pyreadstat
        df, _ = pyreadstat.read_sav(filepath, **kwargs)
        return df
    if ext not in readers:
        raise ValueError(f"不支持的格式: {ext}")
    return readers[ext](filepath, **kwargs)

def quick_overview(df: pd.DataFrame) -> None:
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum()/1024**2:.1f} MB")
    miss = df.isnull().sum()
    if miss.sum() > 0:
        print(f"\nVariables with missing ({(miss>0).sum()}):")
        print(miss[miss>0].sort_values(ascending=False).head(10).to_string())
```

---

## Step 1: 缺失值处理

### 1a: 缺失机制诊断

```
缺失值 → 判断缺失机制
├── MCAR（完全随机缺失）
│   检验: Little's MCAR（R naniar::mcar_test）
│         ⚠️ n>5000 时几乎总拒绝（功效过高），结合 t 检验判断
│   处理: → 直接删除（listwise deletion）
│
├── MAR（随机缺失，与可观测变量相关）
│   标志: 缺失组 vs 非缺失组协变量 t 检验 p<0.05
│   处理: → MICE 多重插补（见 1b）
│
└── MNAR（非随机缺失，与不可观测因素相关）
    例: 高亏损企业不披露利润 / 低质量企业退市
    处理: → 标注风险，Heckman 在分析阶段执行
```

**缺失率分级处理**（必须遵循）：

| 缺失率 | 处理策略 | 论证要求 |
|--------|---------|---------|
| < 5% | 可直接删除 | 报告缺失率即可 |
| 5–10% | MICE 或删除均可 | 需论证缺失机制（MCAR→删除；MAR→MICE） |
| 10–30% | **必须** MICE 多重插补 | 需报告机制诊断 + Rubin's rules 合并 |
| > 30% | 该变量不宜作为核心变量 | 考虑替代指标或放入稳健性检验 |

> **⚠️ 误区：未经论证直接删除缺失数据。** Listwise deletion 是多数软件默认设置，但它要求 MCAR 假设。必须先诊断缺失机制，再选择处理方法，并在论文中说明依据。

**Python（缺失组诊断）**
```python
from scipy.stats import ttest_ind

def diagnose_missing(df, target_col, covariate_cols):
    miss_flag = df[target_col].isnull()
    print(f"{target_col}: {miss_flag.sum()} missing ({miss_flag.mean()*100:.1f}%)")
    rows = []
    for cov in covariate_cols:
        s_m, s_nm = df.loc[miss_flag, cov].dropna(), df.loc[~miss_flag, cov].dropna()
        if len(s_m) < 5: continue
        t, p = ttest_ind(s_m, s_nm, equal_var=False)
        rows.append({
            "Covariate": cov, "Mean_miss": round(s_m.mean(),4),
            "Mean_obs": round(s_nm.mean(),4), "p": round(p,4),
            "Diagnosis": "⚠️ MAR" if p<0.05 else "MCAR compatible",
        })
    return pd.DataFrame(rows).set_index("Covariate")
```

**R**
```r
library(naniar)
mcar_result <- mcar_test(df)
# ⚠️ n>5000 时结合实质意义判断，不能只看 p 值
```

### 1b: MICE 多重插补（MAR / 缺失率 5–30%）

**要点**：M=5 次独立插补 → 分别回归 → Rubin's rules 合并（**不能取均值！**）

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def mice_impute(df, imp_vars, M=5, max_iter=10):
    datasets = []
    for m in range(M):
        imputer = IterativeImputer(
            max_iter=max_iter, random_state=m,
            sample_posterior=True,  # 贝叶斯后验（关键！）
        )
        df_imp = df.copy()
        df_imp[imp_vars] = imputer.fit_transform(df[imp_vars])
        datasets.append(df_imp)
    return datasets  # 返回 M 个 DataFrame
```

```r
library(mice)
imp <- mice(df[, vars], m=5, method="pmm", seed=42, printFlag=FALSE)
fit <- with(imp, lm(y ~ x1 + x2 + x3))
pooled <- pool(fit)  # Rubin's rules 自动应用
summary(pooled)
```

> Rubin's rules Python 完整实现 → 见 ADVANCED.md

---

## Step 2: 异常值处理

```python
def detect_outliers_iqr(series, k=1.5):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return (series < Q1 - k*IQR) | (series > Q3 + k*IQR)

from scipy import stats as scipy_stats
def detect_outliers_zscore(series, threshold=3.0):
    z = scipy_stats.zscore(series.dropna())
    return pd.Series(abs(z) > threshold, index=series.dropna().index)
```

**Winsorize → 调用 stats skill**（Python 中 `mstats.winsorize().data` 必须加 `.data`）

---

## Step 3: 面板平衡化

### 决策树

```
FE / RE → 非平衡面板可用（推荐）
事件研究 → 需窗口内连续观测（[-k,+k]期）
差分GMM  → 接近平衡即可
系统GMM  → 强烈建议平衡面板

稳健性：主回归用非平衡面板，稳健性用平衡面板对比
```

### 插值风险三重警告

1. **信息失真**：线性插值低估真实波动（SD 缩小）
2. **低估标准误**：伪观测高估有效样本量 → 过度拒绝 H0
3. **加剧偏误**：对 MNAR 缺失插值 → 偏误更大

```python
def check_panel_balance(df, entity_col="firm_id", time_col="year"):
    obs_count = df.groupby(entity_col)[time_col].count()
    T_total = df[time_col].nunique()
    print(f"面板: N={df[entity_col].nunique():,}, T={T_total}")
    print(f"平衡: {obs_count.max() == obs_count.min()}")
    balanced_ids = obs_count[obs_count == T_total].index
    df_bal = df[df[entity_col].isin(balanced_ids)].copy()
    print(f"平衡子样本: {len(balanced_ids):,}/{df[entity_col].nunique():,}"
          f" ({len(df_bal)/len(df)*100:.1f}%)")
    return df_bal
```

> 面板平衡 R 完整代码（fixest 并行对比）、插值诊断完整代码 → 见 ADVANCED.md

---

## Step 4: 变量构造

```python
# 交互项
df["treat_post"] = df["treated"] * df["post"]

# 滞后项（必须 groupby！）
df = df.sort_values(["firm_id","year"])
df["revenue_lag1"]  = df.groupby("firm_id")["revenue"].shift(1)
df["revenue_lead1"] = df.groupby("firm_id")["revenue"].shift(-1)
# ⚠️ 错误: df["revenue"].shift(1)  → 跨企业错位！

# 一阶差分
df["d_revenue"] = df.groupby("firm_id")["revenue"].diff(1)
```

```r
df <- df %>%
  arrange(firm_id, year) %>%
  group_by(firm_id) %>%
  mutate(revenue_lag1=lag(revenue,1), revenue_lead1=lead(revenue,1),
         d_revenue=revenue-lag(revenue,1)) %>%
  ungroup()
```

---

## Step 4.5: 变量转换防护栏

> **核心原则**：所有变量转换必须有理论依据，不能机械操作。本节为 Claude 提供"做之前先检查"的防护栏。

### 4.5a: 对数转换决策

```
需要对数转换？
├── 理论上 X 和 Y 是乘法/比例关系？
│   ├── 是 → 对数转换合理（弹性解释）
│   └── 否 → 不要仅因"偏态"而取对数
│
├── 对数转换的是因变量 Y？
│   ├── Y 含零值？
│   │   ├── 是 → ⚠️ 禁止 log(Y+1)，改用 GLM-QMLE（Poisson + log link）
│   │   └── 否 → OLS + log(Y) 仍有不一致性风险
│   └── 优先考虑 GLM-QMLE 替代 OLS+log(Y)
│       Santos Silva & Tenreyro (2006): QMLE 在异方差下一致
│
└── 对数转换的是自变量 X？
    └── 解释为半弹性（log-level）或弹性（log-log），需理论支撑
```

**⚠️ 误区**：
- "取对数让分布正态" → 函数形式由理论决定，非机械降偏态
- `log(Y+1)` 中 "+1" 是任意常数 → 引入参数偏差、p 值失真
- OLS + log(Y) 在异方差下不一致 → GLM-QMLE 是更优方案

**GLM-QMLE 最小示例**：
```python
import statsmodels.api as sm
model = sm.GLM(y, X, family=sm.families.Poisson(link=sm.families.links.Log()))
result = model.fit(cov_type="HC1")
```

```r
fit <- glm(y ~ x1 + x2, family=quasipoisson(link="log"), data=df)
# RESET 检验
library(lmtest)
resettest(lm(y ~ x1 + x2, data=df), power=2:3, type="fitted")
```

> GLM-QMLE 完整实现 + RESET/Link/BoxCox/Park 检验 → 见 ADVANCED.md

### 4.5b: 中心化决策

```
需要中心化？
├── 目的是"消除多重共线性"？
│   └── ⚠️ 错误！均值中心化不改变本质性共线性，
│       不改变完整模型结果、交互作用显著性或边际效应
│       (Dalal & Zickar, 2012; Echambadi & Hess, 2007)
│
├── 目的是解释便利（让截距有实质意义）？
│   └── ✅ 可以，但需说明中心点选择依据
│
├── 是潜变量调节 SEM？
│   └── ✅ 建议中心化 (Marsh et al., 2004)
│
└── 是多层模型 (HLM)？
    ├── 组均值中心化 (group-mean centering)
    │   → 分离 within/between 效应（通常推荐）
    ├── 总体均值中心化 (grand-mean centering)
    │   → 通常无用 (Antonakis et al., 2021)
    └── ⚠️ 多层模型中心化需极其谨慎，错误选择改变系数含义
```

### 4.5c: 比率变量处理

```
模型中使用了比率（如 ROA = 利润/资产）？
├── 比率作为因变量 Y？
│   └── ⚠️ 结论可能由分母离散度驱动 (Wiseman, 2009)
│       → 改用: Y=分子（利润），控制分母（资产）
│
├── 比率作为自变量 X？
│   └── ⚠️ 同样风险：虚假关系
│       → 改用: X=分子，控制分母
│
└── 是自然比例变量（0-1 范围，如创新销售额/总销售额）？
    └── ✅ 可直接使用（部分与整体的相对关系）
        → 考虑 fractional response model（见 ADVANCED.md）
```

### 4.5d: 差异分数检验

```
使用差异分数（Δy = ya - yb 或 Δx = xa - xb）？
├── 作为因变量？
│   └── 隐含约束: X 对 ya 和 yb 影响相同
│       → 先分别建模 ya 和 yb，Wald 检验系数是否等量反号
│
├── 作为自变量？
│   └── 隐含约束: xa 和 xb 对 Y 影响符号相反、大小相等
│       → 先将 xa、xb 分别纳入，Wald 检验 β_a = -β_b
│
└── 是一阶差分 Δy_t = y_t - y_{t-1}？
    └── ✅ 这是面板差分策略，不是此处讨论的"差异分数"
```

> Wald 检验完整代码 → 见 ADVANCED.md

### 4.5e: 连续变量类别化

> **⚠️ 禁令：不要将连续变量按中位数/均值分割为虚拟变量。**

- 信息损失 → I 类或 II 类错误（Cohen, 1983; MacCallum et al., 2002）
- 处理效应估计可能不一致
- 如理论上确需分类 → 考虑有限混合模型 (McLachlan & Peel, 2004)

```
连续变量 X 想分组？
├── 有理论依据支持离散分组？
│   └── 考虑有限混合模型（数据驱动分组）→ 见 ADVANCED.md
└── 仅为"方便"或"非线性"？
    ├── 非线性关系 → 加入 X² 或使用 GLM
    └── 方便解释 → 保留连续，用边际效应图展示
```

### 4.5f: 复合指标风险

```
构建了复合指标（多维度加权求和）？
├── 不同维度由不同过程驱动？
│   └── ⚠️ 概念模糊：高A低B 和 低A高B 得到相同复合值
│       → 分别纳入各维度，独立估计效应
│
├── 层级式复合指标（先选择是否参与，再测量结果）？
│   └── ⚠️ 严重方法论问题：两个不同决策过程被压缩为单一变量
│       → 使用双变量概率模型 (bivariate probit) 分别建模
│
└── 主成分/因子分析得分？
    └── ✅ 可用，但需报告方差解释率和载荷矩阵
```

---

## Step 5: 数据合并

```python
def safe_merge(df_left, df_right, on, how="left",
               left_name="left", right_name="right"):
    for name, d in [(left_name, df_left), (right_name, df_right)]:
        dups = d.duplicated(subset=on).sum()
        if dups > 0:
            print(f"⚠️ {name} key不唯一: {dups}个重复")
    df_merged = pd.merge(df_left, df_right, on=on, how=how, indicator=True)
    counts = df_merged["_merge"].value_counts()
    print("=== Merge诊断 ===")
    print(counts.to_string())
    pct = counts.get("both",0)/len(df_merged)*100
    print(f"合并率: {pct:.1f}%")
    if counts.get("left_only",0)/len(df_merged) > 0.1:
        print("⚠️ left_only >10% → 检查key格式（int/str/前导零）")
    return df_merged.drop(columns=["_merge"])
```

```r
stopifnot(!any(duplicated(df_firm[,c("firm_id","year")])))
df_merged <- left_join(df_firm, df_macro, by="year")
cat("未匹配:", nrow(anti_join(df_firm, df_macro, by="year")), "\n")
```

---

## 检验清单

**缺失值**
- [ ] 缺失机制已诊断（MCAR/MAR/MNAR），论文中说明处理依据
- [ ] 缺失率分级处理已遵循（<5%/5-10%/10-30%/>30%）
- [ ] MICE: M=5, `sample_posterior=True`, Rubin's rules 合并（NOT 取均值）
- [ ] MNAR 已标注风险，Heckman 推迟到分析阶段

**异常值与面板**
- [ ] Winsorize `.data` 已执行（Python）
- [ ] 插值比例 >10% 已报告并做非平衡稳健性

**变量构造**
- [ ] 滞后项: `groupby("firm_id").shift()`，未跨企业错位
- [ ] 合并 key 类型统一，合并率已报告

**变量转换防护栏**（Step 4.5）
- [ ] 对数转换有理论依据，非仅因偏态
- [ ] 因变量对数转换：已考虑 GLM-QMLE 替代
- [ ] 未使用 log(Y+1) 处理零值（改用 GLM 或 arcsinh）
- [ ] 中心化：未声称"消除共线性"；多层模型中心化已论证
- [ ] 比率变量：已改为分子建模+分母控制（或论证为自然比例）
- [ ] 差异分数：已做 Wald 检验验证等量反号约束
- [ ] 连续变量未按中位数分割
- [ ] 复合指标：各维度独立估计（或论证合成合理性）

## 常见错误

1. **`from heckman import Heckman`**：该包不存在！Python Heckman 用 `statsmodels.Probit` 手动两阶段。
2. **MICE 后取均值回归**：必须对 M 个数据集分别回归，Rubin's rules 合并。
3. **面板滞后未 groupby**：`df["lag1"]=df["revenue"].shift(1)` 跨企业错位。
4. **插值 >10% 未报告**：必须做非平衡稳健性。
5. **"取对数让分布正态"**：函数形式由理论驱动，非机械操作。
6. **OLS + log(Y+1)**：任意常数引入偏差，改用 GLM-QMLE。
7. **中心化消除共线性**：不改变本质性共线性和模型结果。
8. **中位数分割连续变量**：信息损失，I/II 类错误风险。

## 输出规范

| 文件 | 路径 |
|------|------|
| 清洗后数据 | `data/processed/panel_clean.parquet` |
| 清洗日志 | `logs/data_cleaning_log.txt` |
| 缺失模式图 | `output/figures/missing_heatmap.png` |

**样本筛选流程模板**（论文附录必备，每步须注明依据）：
```
初始样本（A股上市公司 2010-2022）           N = 38,500
  剔除金融行业（行业特殊性）                N = 35,200
  剔除ST/PT企业（财务异常）                 N = 33,800
  剔除主要变量缺失>30%（不宜作核心变量）     N = 31,400
  Winsorize 1%/99%（不删除观测）            N = 31,400
最终样本（个体: 3,080, 平均T: 10.2期）     N = 31,400
```
