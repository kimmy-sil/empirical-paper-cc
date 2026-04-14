---
name: stats
description: "描述统计与统计检验，含分布诊断、假设检验、Bootstrap"
---

# 描述性统计与统计检验

## 概述

**基础设施层**。其他skill（DID/IV/RDD等）引用本skill的Balance Table、Winsorize、VIF模块。

**本skill覆盖**：
- 全样本描述统计表（Table 1格式）
- 面板描述性统计（overall/between/within分解）
- 分组描述与Balance Table（标准化差异SMD）
- Winsorize缩尾（全样本 + 按年）
- 相关系数矩阵（高效实现，非O(n²)循环）
- VIF多重共线性检验
- 正态性检验 + QQ图
- 缺失值模式分析
- 多重检验校正

**语言**：Python + R（NO Stata）  
**高级内容**：见 ADVANCED.md（R面板分解、cobalt Love Plot完整代码、threeparttable高级模板）

---

## 前置条件

```python
# Python核心依赖
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mstats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
```

```r
# R核心依赖（按需安装）
# install.packages(c("gtsummary","cobalt","plm","car","DescTools","Hmisc","modelsummary","naniar","psych"))
library(dplyr); library(gtsummary); library(cobalt)
library(plm);   library(car);       library(DescTools)
library(Hmisc); library(modelsummary)
```

---

## Step 1: 基础描述性统计

### 1a: 全样本描述统计表（Table 1格式）

统计量：N, Mean, SD, Min, P25, Median, P75, Max

**Python**
```python
def describe_stats(df: pd.DataFrame, vars_list: list) -> pd.DataFrame:
    """
    生成发表级描述性统计表
    返回: DataFrame，行=变量，列=N/Mean/SD/Min/P25/Median/P75/Max
    """
    rows = []
    for v in vars_list:
        s = df[v].dropna()
        rows.append({
            "Variable": v,
            "N":      int(len(s)),
            "Mean":   s.mean(),
            "SD":     s.std(),
            "Min":    s.min(),
            "P25":    s.quantile(0.25),
            "Median": s.median(),
            "P75":    s.quantile(0.75),
            "Max":    s.max(),
        })
    return pd.DataFrame(rows).set_index("Variable")

VARS = ["revenue", "assets", "leverage", "roa", "age"]
tbl = describe_stats(df, VARS)

# LaTeX输出（用df.to_latex(booktabs=True)，NOT手写tabular）
latex_str = tbl.round(3).to_latex(
    booktabs=True,
    caption="Descriptive Statistics",
    label="tab:descriptive",
)
with open("output/tables/table1_descriptive.tex", "w") as f:
    f.write(latex_str)
tbl.round(3).to_excel("output/tables/table1_descriptive.xlsx")
```

**R**
```r
library(modelsummary)
# 直接输出LaTeX（推荐）
datasummary_skim(df[, VARS], output = "output/tables/table1_descriptive.tex")

# 分组描述（gtsummary，含p值）
library(gtsummary)
df %>%
  select(all_of(c(VARS, "treated"))) %>%
  tbl_summary(
    by = treated,
    statistic = list(all_continuous() ~ "{mean} ({sd})"),
    digits    = all_continuous() ~ 3
  ) %>%
  add_p() %>% add_overall() %>% as_gt()
```

### 1b: 面板描述性统计（overall/between/within分解）

**Python**
```python
def panel_describe(df: pd.DataFrame, var: str,
                   entity_col: str = "firm_id",
                   time_col: str = "year") -> dict:
    """
    面板三向方差分解（Mundlak分解）
    Overall: 全样本方差
    Between: 个体均值间方差（跨企业差异）
    Within:  个体内时间变异（去均值残差）
    """
    s = df[[entity_col, time_col, var]].dropna()
    overall_mean = s[var].mean()
    entity_means = s.groupby(entity_col)[var].mean()
    s = s.copy()
    s["within_dev"] = s[var] - s.groupby(entity_col)[var].transform("mean") + overall_mean

    return {
        "Variable":     var,
        "Overall_N":    len(s),
        "Overall_Mean": round(overall_mean, 4),
        "Overall_SD":   round(s[var].std(), 4),
        "Between_N":    len(entity_means),
        "Between_SD":   round(entity_means.std(), 4),
        "Within_SD":    round(s["within_dev"].std(), 4),
    }

panel_tbl = pd.DataFrame([panel_describe(df, v) for v in VARS]).set_index("Variable")
```

**R**
```r
library(plm)
pdata <- pdata.frame(df, index = c("firm_id","year"))
# plm自动报告overall/between/within
summary(pdata[, "revenue"])
```

### 1c: 分组描述（处理组vs对照组）

```python
from scipy import stats as scipy_stats

def group_comparison_table(df: pd.DataFrame, vars_list: list,
                           group_col: str,
                           group_labels: dict = None) -> pd.DataFrame:
    """
    分组描述统计 + Welch t检验（不假设方差相等）
    返回: Control_Mean/Treated_Mean/Diff/t-stat/p-value/Sig
    """
    if group_labels is None:
        group_labels = {0: "Control", 1: "Treated"}
    rows = []
    for v in vars_list:
        s0 = df.loc[df[group_col] == 0, v].dropna()
        s1 = df.loc[df[group_col] == 1, v].dropna()
        t_stat, p_val = scipy_stats.ttest_ind(s1, s0, equal_var=False)
        row = {"Variable": v}
        for g, s in [(0, s0), (1, s1)]:
            lbl = group_labels[g]
            row[f"{lbl}_Mean"] = round(s.mean(), 4)
            row[f"{lbl}_SD"]   = round(s.std(), 4)
        row["Diff"]    = round(s1.mean() - s0.mean(), 4)
        row["t-stat"]  = round(t_stat, 3)
        row["p-value"] = round(p_val, 4)
        row["Sig"] = "***" if p_val<0.01 else "**" if p_val<0.05 else "*" if p_val<0.10 else ""
        rows.append(row)
    return pd.DataFrame(rows).set_index("Variable")
```

---

## Step 2: Balance Table（标准化差异）

SMD公式：
$$\text{SMD} = \frac{|\bar{X}_{treat} - \bar{X}_{control}|}{\sqrt{(SD_{treat}^2 + SD_{control}^2)/2}}$$

判断：|SMD| < 0.1 ✓，< 0.25 △，≥ 0.25 ✗

**Python**
```python
def balance_table(df: pd.DataFrame, vars_list: list,
                  treat_col: str = "treated") -> pd.DataFrame:
    """
    标准Balance Table（SMD + Welch t检验）
    """
    rows = []
    for v in vars_list:
        s0 = df.loc[df[treat_col] == 0, v].dropna()
        s1 = df.loc[df[treat_col] == 1, v].dropna()
        mean0, mean1 = s0.mean(), s1.mean()
        sd0,   sd1   = s0.std(),  s1.std()
        smd = abs(mean1 - mean0) / np.sqrt((sd0**2 + sd1**2) / 2)
        t_stat, p_val = scipy_stats.ttest_ind(s1, s0, equal_var=False)
        rows.append({
            "Variable":     v,
            "Control_Mean": round(mean0, 4),
            "Treated_Mean": round(mean1, 4),
            "SMD":          round(smd, 4),
            "Balance":      "✓" if smd<0.1 else "△" if smd<0.25 else "✗",
            "p-value":      round(p_val, 4),
        })
    tbl = pd.DataFrame(rows).set_index("Variable")
    # LaTeX输出（booktabs）
    print(tbl.to_latex(booktabs=True))
    return tbl

# tableone包（若可用）
import importlib.util
if importlib.util.find_spec("tableone"):
    from tableone import TableOne
    tbl1 = TableOne(df, columns=VARS, groupby="treated", smd=True, pval=True)
    tbl1.to_excel("output/tables/balance_table.xlsx")
```

**R（cobalt包 + Love Plot）**
```r
library(cobalt)
bal_result <- bal.tab(
  treated ~ revenue + assets + leverage + roa + age,
  data       = df,
  thresholds = c(m = 0.1)
)
print(bal_result)

# Love Plot
love.plot(bal_result, threshold=0.1, abs=TRUE, title="Covariate Balance: SMD")

# 输出LaTeX
datasummary_balance(~treated, data=df[,c(VARS,"treated")],
                    output="output/tables/balance_table.tex")
```

---

## Step 3: Winsorization缩尾

**关键**：`mstats.winsorize()`返回masked array，必须加`.data`转换！

**Python**
```python
from scipy.stats import mstats

def winsorize_vars(df: pd.DataFrame, vars_list: list,
                   limits: list = [0.01, 0.01],
                   suffix: str = "_w") -> pd.DataFrame:
    """全样本缩尾（默认上下各1%）"""
    df = df.copy()
    for v in vars_list:
        # .data转换为普通ndarray（必须！否则后续操作异常）
        df[f"{v}{suffix}"] = mstats.winsorize(df[v].values, limits=limits).data
    return df

def winsorize_by_year(df: pd.DataFrame, vars_list: list,
                      year_col: str = "year",
                      limits: list = [0.01, 0.01],
                      suffix: str = "_w") -> pd.DataFrame:
    """按年份分组缩尾（金融/会计研究标准做法）"""
    df = df.copy()
    for v in vars_list:
        df[f"{v}{suffix}"] = (
            df.groupby(year_col)[v]
            .transform(lambda x: mstats.winsorize(x.values, limits=limits).data)
        )
    return df
```

**R**
```r
library(DescTools)
# 全样本
df$revenue_w <- Winsorize(df$revenue, probs = c(0.01, 0.99))
# 按年
df <- df %>%
  group_by(year) %>%
  mutate(across(all_of(VARS), ~Winsorize(., probs=c(0.01,0.99)), .names="{.col}_w")) %>%
  ungroup()
```

---

## Step 4: 相关系数矩阵

**原则**：`df.corr()`一行获取系数矩阵（NOT O(n²)循环），只对下三角计算p值。

**Python**
```python
from scipy.stats import pearsonr

def corr_matrix_with_pval(df: pd.DataFrame, vars_list: list) -> pd.DataFrame:
    """
    相关系数矩阵（下三角系数+星号，对角线1.000，上三角空白）
    效率: O(n²/2)，只跑下三角
    """
    coef_mat = df[vars_list].corr().values  # 一行获取所有系数
    n = len(vars_list)
    result = pd.DataFrame("", index=vars_list, columns=vars_list)
    for i in range(n):
        result.iloc[i, i] = "1.000"
        for j in range(i):  # 只跑下三角
            common = df[vars_list].iloc[:, [i,j]].dropna()
            if len(common) < 3:
                continue
            _, p = pearsonr(common.iloc[:,0], common.iloc[:,1])
            stars = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""
            result.iloc[i,j] = f"{coef_mat[i,j]:.3f}{stars}"
    return result

corr_tbl = corr_matrix_with_pval(df, VARS)
print(corr_tbl.to_latex(booktabs=True))
```

**R**
```r
library(Hmisc)
corr_res <- rcorr(as.matrix(df[, VARS]), type="pearson")
# 系数矩阵: corr_res$r，p值矩阵: corr_res$P
# 格式化函数见 ADVANCED.md
```

---

## Step 5: VIF多重共线性

VIF > 5 → 中度（关注），VIF > 10 → 严重（必须处理）

**Python**
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def compute_vif(df: pd.DataFrame, vars_list: list) -> pd.DataFrame:
    """计算VIF，自动加常数项"""
    X = sm.add_constant(df[vars_list].dropna())
    vif_data = pd.DataFrame({
        "Variable": X.columns,
        "VIF":      [variance_inflation_factor(X.values, i)
                     for i in range(X.shape[1])]
    })
    vif_data = vif_data[vif_data["Variable"] != "const"].copy()
    vif_data["VIF"]  = vif_data["VIF"].round(2)
    vif_data["Flag"] = vif_data["VIF"].apply(
        lambda x: "✗ 严重" if x>10 else "△ 中度" if x>5 else "✓"
    )
    return vif_data.set_index("Variable")
```

**R**
```r
library(car)
model <- lm(y ~ revenue + assets + leverage + roa + age, data=df)
vif_vals <- car::vif(model)
data.frame(VIF=round(vif_vals,2), Flag=ifelse(vif_vals>10,"严重",ifelse(vif_vals>5,"中度","正常")))
```

---

## Step 6: 正态性检验

**注意**：n > 5000时Shapiro-Wilk几乎总拒绝正态假设（高功效），大样本应结合QQ图和偏度峰度判断。

**Python（检验 + QQ图）**
```python
from scipy import stats

def normality_tests(series: pd.Series, alpha: float=0.05,
                    plot: bool=True, save_path: str=None) -> pd.DataFrame:
    """多种正态性检验，QQ图用scipy.stats.probplot（无需外部依赖）"""
    s = series.dropna()
    results = []
    if len(s) <= 5000:
        stat, p = stats.shapiro(s)
        results.append({"Test":"Shapiro-Wilk","Statistic":round(stat,4),"p-value":round(p,4),"Reject H0":p<alpha})
    stat, p = stats.jarque_bera(s)
    results.append({"Test":"Jarque-Bera","Statistic":round(stat,4),"p-value":round(p,4),"Reject H0":p<alpha})
    stat, p = stats.kstest(s, "norm", args=(s.mean(),s.std()))
    results.append({"Test":"KS","Statistic":round(stat,4),"p-value":round(p,4),"Reject H0":p<alpha})
    results.append({"Test":"Skewness","Statistic":round(stats.skew(s),4),"p-value":None,"Reject H0":None})
    results.append({"Test":"Kurtosis","Statistic":round(stats.kurtosis(s),4),"p-value":None,"Reject H0":None})

    if plot:
        fig, ax = plt.subplots(figsize=(5,5))
        stats.probplot(s, dist="norm", plot=ax)
        ax.set_title(f"Q-Q Plot: {series.name or 'variable'}")
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return pd.DataFrame(results).set_index("Test")
```

**R**
```r
shapiro.test(df$revenue)                    # Shapiro-Wilk（n<5000）
tseries::jarque.bera.test(df$revenue)       # Jarque-Bera
qqnorm(df$revenue); qqline(df$revenue, col="red", lwd=2)
```

---

## Step 7: 缺失值模式分析

**大样本限制**：Little's MCAR检验在n>5000时几乎总拒绝MCAR（功效过高）。  
**更实用**：比较缺失组 vs 非缺失组的协变量均值差异（t检验）。

**Python（matplotlib/seaborn热力图，无需missingno依赖）**
```python
import seaborn as sns

def missing_pattern_analysis(df: pd.DataFrame, save_path: str=None) -> pd.DataFrame:
    """缺失比例表 + 热力图（seaborn，无需missingno）"""
    miss_tbl = pd.DataFrame({
        "Missing_N":   df.isnull().sum(),
        "Missing_Pct": (df.isnull().mean()*100).round(2),
    }).sort_values("Missing_Pct", ascending=False)
    miss_tbl = miss_tbl[miss_tbl["Missing_N"] > 0]
    print(miss_tbl)

    fig, ax = plt.subplots(figsize=(12,5))
    sns.heatmap(df.isnull().astype(int).T, cbar=False, cmap="YlOrRd",
                xticklabels=False, ax=ax)
    ax.set_title("Missing Value Heatmap (Yellow=Missing)")
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return miss_tbl

def missing_vs_nonmissing(df: pd.DataFrame, target_col: str,
                          covariate_cols: list) -> pd.DataFrame:
    """
    缺失组 vs 非缺失组协变量t检验
    更实用的MAR诊断（不受样本量影响）
    """
    from scipy.stats import ttest_ind
    miss_flag = df[target_col].isnull()
    rows = []
    for cov in covariate_cols:
        s_m  = df.loc[miss_flag,  cov].dropna()
        s_nm = df.loc[~miss_flag, cov].dropna()
        if len(s_m) < 5: continue
        t, p = ttest_ind(s_m, s_nm, equal_var=False)
        rows.append({"Covariate":cov,"Mean_missing":round(s_m.mean(),4),
                     "Mean_observed":round(s_nm.mean(),4),
                     "p-value":round(p,4),
                     "Diagnosis":"⚠️ 疑似MAR" if p<0.05 else "无显著差异"})
    return pd.DataFrame(rows).set_index("Covariate")
```

**R（naniar）**
```r
library(naniar)
mcar_result <- mcar_test(df)
cat("Little's MCAR: p=", mcar_result$p.value, "\n")
cat("注意: n>5000时几乎总拒绝MCAR，结合实质意义判断\n")
vis_miss(df); gg_miss_var(df)
```

---

## Step 8: 多重检验校正

同时检验多个假设时（如Balance Table所有变量），需校正多重检验。

**Python**
```python
from statsmodels.stats.multitest import multipletests

def adjust_pvalues(pvals: list, method: str="fdr_bh",
                   alpha: float=0.05) -> pd.DataFrame:
    """
    method: 'bonferroni'（最保守）, 'fdr_bh'（Benjamini-Hochberg推荐）
    """
    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method=method)
    return pd.DataFrame({"p_original":[round(p,4) for p in pvals],
                         "p_adjusted":[round(p,4) for p in p_adj],
                         "reject_H0": reject})
```

**R**
```r
pvals <- bal_tbl$p_value  # 从Balance Table提取
p.adjust(pvals, method = "BH")        # Benjamini-Hochberg（推荐）
p.adjust(pvals, method = "bonferroni") # Bonferroni（最保守）
```

---

## LaTeX输出规范

```python
# 标准做法：用df.to_latex(booktabs=True)，NOT手写tabular代码
# threeparttable完整模板见 ADVANCED.md

latex_str = tbl.round(3).to_latex(
    booktabs=True,
    escape=True,      # 自动转义_等特殊字符
    caption="Descriptive Statistics",
    label="tab:descriptive",
)
```

---

## 检验清单

- [ ] **样本一致性**：描述统计N数与回归样本N一致
- [ ] **变量范围**：连续变量Mean/SD/Min/Max在合理范围内
- [ ] **Balance Table**：SMD全部 < 0.1（或说明 ≥ 0.25变量的处理）
- [ ] **Winsorize.data**：Python中`mstats.winsorize().data`转换已执行
- [ ] **相关系数**：核心变量 < 0.7；VIF < 10
- [ ] **df.corr()**：一行获取系数矩阵，未使用O(n²)手写循环
- [ ] **正态性**：大样本（>5000）不单纯依赖Shapiro-Wilk，结合QQ图
- [ ] **缺失值**：已报告缺失组vs非缺失组比较，说明MCAR/MAR判断
- [ ] **多重校正**：同时检验多变量时已做BH/Bonferroni校正
- [ ] **LaTeX**：用`df.to_latex(booktabs=True)`，不手写tabular

## 输出规范

| 文件 | 路径 |
|------|------|
| 描述统计表 | `output/tables/table1_descriptive.tex` + `.xlsx` |
| Balance Table | `output/tables/balance_table.tex` + `.xlsx` |
| 相关系数矩阵 | `output/tables/table_corr.tex` + `.xlsx` |
| VIF表 | `output/tables/vif_table.xlsx` |
| QQ图 | `output/figures/qqplot_{varname}.png` |
| 缺失模式图 | `output/figures/missing_heatmap.png` |

**数字格式**：均值/SD保留3-4位小数，N为整数，p值保留4位小数  
**显著性**：`***` p<0.01，`**` p<0.05，`*` p<0.10，底部注释说明
