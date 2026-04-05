# 数据清洗

## 概述

经管类实证研究标准数据清洗流程。  
**语言**：Python + R（NO Stata）

**适用场景**：原始面板数据清洗、缺失值诊断处理、异常值检测、面板平衡化决策、变量构造、数据合并诊断。

**架构**：Winsorize模块在stats skill（调用之，本skill不重复）。Heckman选择模型在分析阶段执行，本skill仅诊断并标注MNAR风险。

**高级内容**：MICE完整Rubin's rules合并、面板平衡诊断R完整代码、插值风险详细代码示例 → 见 ADVANCED.md

---

## Step 0: 环境检测 + 格式自动读取

```python
import importlib.util
import os
import pandas as pd

def check_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

HAS_MISSINGNO = check_package("missingno")
HAS_SKLEARN   = check_package("sklearn")

if not HAS_MISSINGNO:
    print("⚠️ missingno不可用 → 使用matplotlib/seaborn替代（已内置支持）")
if not HAS_SKLEARN:
    print("⚠️ sklearn不可用 → MICE插补建议在R端执行")

def auto_read(filepath: str, **kwargs) -> pd.DataFrame:
    """
    自动检测文件格式并读取
    支持: .csv .xlsx .dta .sav .parquet .feather
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        return pd.read_csv(filepath, **kwargs)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(filepath, **kwargs)
    elif ext == ".dta":
        return pd.read_stata(filepath, **kwargs)
    elif ext == ".sav":
        try:
            import pyreadstat
            df, _ = pyreadstat.read_sav(filepath, **kwargs)
            return df
        except ImportError:
            raise ImportError("SPSS .sav需要pyreadstat: pip install pyreadstat")
    elif ext == ".parquet":
        return pd.read_parquet(filepath, **kwargs)
    elif ext == ".feather":
        return pd.read_feather(filepath, **kwargs)
    else:
        raise ValueError(f"不支持的格式: {ext}，请转换为CSV后重试")

def quick_overview(df: pd.DataFrame) -> None:
    """数据基本信息快览"""
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum()/1024**2:.1f} MB")
    miss = df.isnull().sum()
    if miss.sum() > 0:
        print(f"\nVariables with missing values ({(miss>0).sum()}):")
        print(miss[miss>0].sort_values(ascending=False).head(10).to_string())
```

---

## Step 1: 缺失值处理

### 1a: 缺失机制诊断（决策树）

```
缺失值 → 判断缺失机制
├── MCAR（完全随机缺失）
│   检验: Little's MCAR（R naniar::mcar_test）
│         ⚠️ n>5000时几乎总拒绝MCAR（功效过高）
│         更实用: 缺失组 vs 非缺失组 t检验
│   处理: → 直接删除（listwise deletion）
│
├── MAR（随机缺失，与可观测变量相关）
│   标志: 缺失与其他观测变量显著相关（t检验p<0.05）
│   处理: → MICE多重插补（见1b）
│         → 或控制相关变量后用非平衡面板
│
└── MNAR（非随机缺失，与不可观测因素相关）
    例: 高亏损企业不披露利润 / 低质量企业退市
    ⚠️ 本skill仅标注风险，Heckman在分析阶段执行
    处理: → 在OLS/DID skill中使用Heckman两阶段
          → 说明缺失比例和偏误方向
```

**Python（缺失组诊断）**
```python
from scipy.stats import ttest_ind

def diagnose_missing(df: pd.DataFrame, target_col: str,
                     covariate_cols: list) -> pd.DataFrame:
    """
    缺失组 vs 非缺失组协变量t检验（实用MAR诊断）
    比Little's MCAR检验更直观，不受样本量影响
    """
    miss_flag = df[target_col].isnull()
    print(f"{target_col}: {miss_flag.sum()} missing ({miss_flag.mean()*100:.1f}%)")
    rows = []
    for cov in covariate_cols:
        s_m  = df.loc[miss_flag,  cov].dropna()
        s_nm = df.loc[~miss_flag, cov].dropna()
        if len(s_m) < 5: continue
        t, p = ttest_ind(s_m, s_nm, equal_var=False)
        rows.append({
            "Covariate":     cov,
            "Mean_missing":  round(s_m.mean(), 4),
            "Mean_observed": round(s_nm.mean(), 4),
            "p-value":       round(p, 4),
            "Diagnosis":     "⚠️ 疑似MAR" if p<0.05 else "无显著差异（可能MCAR）",
        })
    return pd.DataFrame(rows).set_index("Covariate")
```

**R（Little's MCAR检验）**
```r
library(naniar)
mcar_result <- mcar_test(df)
cat("Little's MCAR: χ²=", mcar_result$statistic, "p=", mcar_result$p.value, "\n")
cat("⚠️ 注意: n>5000时几乎总拒绝MCAR，结合实质意义判断\n")

# 缺失组 vs 非缺失组（更实用）
df %>%
  mutate(miss_flag = is.na(revenue)) %>%
  group_by(miss_flag) %>%
  summarise(across(c(assets,leverage,age), mean, na.rm=TRUE))
```

### 1b: 缺失值处理策略

#### A: 直接删除（MCAR / 缺失率 < 5%）

```python
key_vars = ["revenue","employees","year","firm_id"]
n_before = len(df)
df_clean = df.dropna(subset=key_vars)
print(f"删除 {n_before-len(df_clean)} 行 ({(n_before-len(df_clean))/n_before*100:.1f}%)")
```

```r
df_clean <- df %>% drop_na(revenue, employees)
```

#### B: MICE多重插补（MAR / 缺失率 5-30%）

**正确实现要点**：
1. `sample_posterior=True`（贝叶斯MICE，不确定性更准确）
2. M=5次独立插补
3. **不能**对5个数据集取均值后回归 → 用Rubin's rules合并（见ADVANCED.md）

```python
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

def mice_impute(df: pd.DataFrame, imp_vars: list,
                M: int=5, max_iter: int=10) -> list:
    """
    MICE多重插补，返回M个插补后的DataFrame列表
    后续在回归skill中对M个数据集分别回归，Rubin's rules合并
    """
    datasets = []
    for m in range(M):
        imputer = IterativeImputer(
            max_iter=max_iter,
            random_state=m,          # 每次不同seed
            sample_posterior=True,   # 贝叶斯后验采样（关键！）
        )
        X_imp = imputer.fit_transform(df[imp_vars])
        df_imp = df.copy()
        df_imp[imp_vars] = X_imp
        datasets.append(df_imp)
    return datasets  # 返回M个DataFrame，勿取均值！
```

```r
library(mice)
imp_data <- mice(
  df[, c("revenue","assets","leverage","age")],
  m=5, method="pmm", seed=42, printFlag=FALSE
)
# 回归+合并（Rubin's rules自动应用）
fit    <- with(imp_data, lm(y ~ revenue + assets + leverage + age))
pooled <- pool(fit)
summary(pooled)
```

---

## Step 2: 异常值处理

### IQR法检测

```python
def detect_outliers_iqr(series: pd.Series, k: float=1.5) -> pd.Series:
    """IQR法，k=1.5（标准）或k=3（保守）"""
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return (series < Q1 - k*IQR) | (series > Q3 + k*IQR)
```

### Z-score法

```python
from scipy import stats as scipy_stats

def detect_outliers_zscore(series: pd.Series, threshold: float=3.0) -> pd.Series:
    z = scipy_stats.zscore(series.dropna())
    return pd.Series(abs(z) > threshold, index=series.dropna().index)
```

### Winsorize → 调用stats skill

```python
# 见 stats/SKILL.md Step 3
# Python: mstats.winsorize(...).data  （必须加.data！）
# R:      DescTools::Winsorize(...)
```

---

## Step 3: 面板平衡化

### 决策树

```
FE / RE → 非平衡面板可用（推荐）
事件研究 → 需窗口内连续观测（[-k,+k]期）
差分GMM  → 接近平衡即可
系统GMM  → 强烈建议平衡面板

稳健性做法：主回归用非平衡面板，稳健性用平衡面板对比
```

### 插值风险三重警告（任何插值前必读）

**警告1：信息失真**  
线性插值假设变量匀速变化，会**低估真实波动**（标准差缩小），影响相关结构。

**警告2：低估标准误**  
插值创造"伪观测"：有效样本量被高估 → 标准误偏低 → 过度拒绝H0 → 置信区间过窄。  
建议：若必须插值，稳健性检验用原始非平衡面板对比。

**警告3：引入测量误差**  
对MNAR缺失插值 → **加剧偏误**（缺失本身携带信息）。

```python
# 插值前后标准差对比诊断
df_sorted = df.sort_values(["firm_id","year"])
df_sorted["rev_filled"] = (
    df_sorted.groupby("firm_id")["revenue"]
    .transform(lambda x: x.interpolate(method="linear", limit=2))
)
df_sorted["imputed_flag"] = df_sorted["revenue"].isnull() & df_sorted["rev_filled"].notna()
pct = df_sorted["imputed_flag"].mean() * 100
print(f"原始SD: {df_sorted['revenue'].std():.4f}")
print(f"插值后SD: {df_sorted['rev_filled'].std():.4f}")
print(f"插值比例: {pct:.1f}%")
if pct > 10:
    print("⚠️ 插值超过10%！极度谨慎，考虑Heckman或报告非平衡稳健性")
```

### Python诊断代码

```python
def check_panel_balance(df: pd.DataFrame, entity_col: str="firm_id",
                        time_col: str="year") -> pd.DataFrame:
    """诊断面板平衡性，返回平衡子样本"""
    obs_count = df.groupby(entity_col)[time_col].count()
    T_total   = df[time_col].nunique()
    print(f"面板维度: N={df[entity_col].nunique():,}, T={T_total}")
    print(f"平衡: {obs_count.max() == obs_count.min()}")
    print("个体观测数分布:")
    print(obs_count.value_counts().sort_index().to_string())
    balanced_ids = obs_count[obs_count == T_total].index
    df_balanced  = df[df[entity_col].isin(balanced_ids)].copy()
    print(f"\n平衡子样本: {len(balanced_ids):,}/{df[entity_col].nunique():,} "
          f"个体 ({len(df_balanced)/len(df)*100:.1f}%)")
    return df_balanced
```

---

## Step 4: 变量构造

### 交互项

```python
df["treat_post"] = df["treated"] * df["post"]        # DID核心交互项
df["size_lev"]   = df["log_assets"] * df["leverage"]
```

```r
df <- df %>% mutate(treat_post = treated * post, size_lev = log_assets * leverage)
```

### 滞后项与前向项

```python
# 必须groupby(firm_id)再shift，防止跨企业错位！
df = df.sort_values(["firm_id","year"])
df["revenue_lag1"]  = df.groupby("firm_id")["revenue"].shift(1)
df["revenue_lag2"]  = df.groupby("firm_id")["revenue"].shift(2)
df["revenue_lead1"] = df.groupby("firm_id")["revenue"].shift(-1)

# ⚠️ 错误做法：
# df["revenue_lag1"] = df["revenue"].shift(1)  # 跨企业错位！
```

```r
df <- df %>%
  arrange(firm_id, year) %>%
  group_by(firm_id) %>%
  mutate(revenue_lag1=lag(revenue,1), revenue_lag2=lag(revenue,2),
         revenue_lead1=lead(revenue,1)) %>%
  ungroup()
```

### 一阶差分

```python
df["d_revenue"] = df.groupby("firm_id")["revenue"].diff(1)
```

### 对数变换

```python
import numpy as np
df["log_revenue"]   = np.log(df["revenue"].clip(lower=1e-6))  # 处理零值
df["log1p_revenue"] = np.log1p(df["revenue"])                  # log(1+x)
df["ihs_profit"]    = np.arcsinh(df["profit"])                 # 双曲正弦（允许负值）
```

---

## Step 5: 数据合并

```python
def safe_merge(df_left: pd.DataFrame, df_right: pd.DataFrame,
               on: list, how: str="left",
               left_name: str="left", right_name: str="right") -> pd.DataFrame:
    """
    安全合并：检查key唯一性（防many-to-many笛卡尔积），输出_merge诊断
    """
    for name, df in [(left_name, df_left), (right_name, df_right)]:
        dups = df.duplicated(subset=on).sum()
        if dups > 0:
            print(f"⚠️ {name}中key不唯一: {dups}个重复")

    df_merged = pd.merge(df_left, df_right, on=on, how=how, indicator=True)
    counts = df_merged["_merge"].value_counts()
    print("\n=== Merge诊断 ===")
    print(counts.to_string())
    pct_both = counts.get("both",0)/len(df_merged)*100
    print(f"合并率: {pct_both:.1f}%")
    if counts.get("left_only",0)/len(df_merged) > 0.1:
        print("⚠️ left_only >10% → 检查key格式（int/str/前导零）")
    return df_merged.drop(columns=["_merge"])
```

```r
library(dplyr)
stopifnot(!any(duplicated(df_firm[,c("firm_id","year")])))
df_merged <- left_join(df_firm, df_macro, by="year")
unmatched  <- anti_join(df_firm, df_macro, by="year")
cat("未匹配行数:", nrow(unmatched), "\n")
```

---

## 检验清单 + 常见错误

### 检验清单

- [ ] **环境检测**：missingno/sklearn可用性已检查，降级方案已准备
- [ ] **缺失机制**：MCAR/MAR/MNAR已诊断，文章中说明处理依据
- [ ] **大样本警告**：n>5000时不单纯依赖Little's检验
- [ ] **MICE**：M=5次，`sample_posterior=True`，后续Rubin's rules合并（NOT取均值）
- [ ] **Winsorize.data**：Python中`.data`已执行
- [ ] **滞后项分组**：`groupby("firm_id").shift()`，未跨企业错位
- [ ] **对数变换**：零值已处理（clip/log1p/arcsinh）
- [ ] **插值警告**：>10%已报告并做非平衡稳健性检验
- [ ] **合并key**：类型统一，合并率已报告
- [ ] **MNAR标注**：Heckman推迟到分析阶段，本skill仅标注风险

### 常见错误

1. **`from heckman import Heckman`**：该包不存在！Python端Heckman用`statsmodels.Probit`手动两阶段（见OLS skill）。
2. **missingno不可用时报错**：应降级为seaborn热力图（Step 0已处理）。
3. **MICE后取均值回归**：错误！必须对M个数据集分别回归，Rubin's rules合并。
4. **面板滞后未groupby**：`df["lag1"]=df["revenue"].shift(1)`跨企业错位，结果严重错误。
5. **插值比例>10%未报告**：必须做非平衡面板稳健性检验并讨论。
6. **合并键类型不一致**：int vs str无法匹配，合并前统一类型。

## 输出规范

| 文件 | 路径 |
|------|------|
| 清洗后数据 | `data/processed/panel_clean.parquet`（首选） |
| 清洗日志 | `logs/data_cleaning_log.txt`（含每步N变化） |
| 缺失模式图 | `output/figures/missing_heatmap.png` |

**样本筛选流程模板（附录）**：
```
初始样本（A股上市公司 2010-2022）          N = 38,500
  剔除金融行业                             N = 35,200
  剔除ST/PT企业                            N = 33,800
  剔除主要变量缺失                          N = 31,400
  Winsorize 1%/99%（不删除观测）            N = 31,400
最终样本（个体: 3,080, 平均T: 10.2期）     N = 31,400
```
