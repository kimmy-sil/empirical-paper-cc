# 描述性统计与统计检验 (Stats)

## 概述

本 skill 提供实证论文中标准统计分析流程，包括描述性统计表（Table 1）、相关系数矩阵、各类统计检验，输出满足期刊投稿要求的发表级三线表格式。

**适用场景**：
- 论文 Table 1（描述性统计，含分组对比）
- 相关系数矩阵（附显著性标注）
- 变量分布检验（正态性、方差齐性）
- 组间差异检验（t 检验、Wilcoxon、卡方）
- 数据附录统计表

---

## 前置条件

| 工具 | 依赖库 |
|------|--------|
| Python | `pandas`, `numpy`, `scipy`, `statsmodels`, `tabulate`, `openpyxl` |
| R | `dplyr`, `psych`, `Hmisc`, `gtsummary`, `modelsummary` |
| Stata | 内置 `sum`, `ttest`, `tab`；扩展 `asdoc`（ssc install）, `estout` |

---

## 分析步骤

### 步骤 1：描述性统计表（Table 1 格式）

#### 全样本描述性统计

**Python**
```python
import pandas as pd
import numpy as np
from scipy import stats

def describe_stats(df, vars_list, percentiles=[0.25, 0.75]):
    """
    生成发表级描述性统计表
    返回: DataFrame，行=变量，列=N/Mean/SD/Min/P25/P75/Max
    """
    rows = []
    for v in vars_list:
        s = df[v].dropna()
        rows.append({
            "Variable": v,
            "N":        len(s),
            "Mean":     s.mean(),
            "SD":       s.std(),
            "Min":      s.min(),
            "P25":      s.quantile(0.25),
            "Median":   s.median(),
            "P75":      s.quantile(0.75),
            "Max":      s.max(),
        })
    return pd.DataFrame(rows).set_index("Variable")

VARS = ["revenue", "assets", "leverage", "roa", "age", "employees"]
tbl = describe_stats(df, VARS)

# 格式化（保留3位小数）
tbl_fmt = tbl.round(3)
print(tbl_fmt.to_string())

# 保存为 Excel
tbl_fmt.to_excel("output/tables/table1_descriptive.xlsx")
```

#### 分组对比表（处理组 vs 对照组）

```python
def group_comparison_table(df, vars_list, group_col, group_labels=None):
    """
    生成分组描述性统计 + 均值差异检验
    group_col: 0/1 二值变量（如 treated）
    """
    groups = df[group_col].unique()
    if group_labels is None:
        group_labels = {g: f"Group {g}" for g in groups}

    rows = []
    for v in vars_list:
        row = {"Variable": v}

        # 分组统计
        for g in sorted(groups):
            s = df.loc[df[group_col] == g, v].dropna()
            label = group_labels[g]
            row[f"{label}_N"]    = len(s)
            row[f"{label}_Mean"] = s.mean()
            row[f"{label}_SD"]   = s.std()

        # 均值差异检验
        s0 = df.loc[df[group_col] == 0, v].dropna()
        s1 = df.loc[df[group_col] == 1, v].dropna()

        # Levene 检验方差齐性 → 选择适当 t 检验
        _, p_levene = stats.levene(s0, s1)
        equal_var = p_levene > 0.05

        t_stat, p_val = stats.ttest_ind(s1, s0, equal_var=equal_var)
        row["t-stat"] = round(t_stat, 3)
        row["p-value"] = round(p_val, 3)
        row["Diff"] = round(s1.mean() - s0.mean(), 3)

        # 显著性星号
        if p_val < 0.01:
            row["Sig"] = "***"
        elif p_val < 0.05:
            row["Sig"] = "**"
        elif p_val < 0.10:
            row["Sig"] = "*"
        else:
            row["Sig"] = ""

        rows.append(row)

    return pd.DataFrame(rows).set_index("Variable")

tbl_compare = group_comparison_table(
    df, VARS, group_col="treated",
    group_labels={0: "Control", 1: "Treated"}
)
print(tbl_compare.round(3).to_string())
```

**R (gtsummary — 最推荐)**
```r
library(gtsummary)
library(dplyr)

df %>%
  select(revenue, assets, leverage, roa, age, treated) %>%
  tbl_summary(
    by = treated,                        # 分组变量
    statistic = list(
      all_continuous() ~ "{mean} ({sd})" # 格式：均值（标准差）
    ),
    digits = all_continuous() ~ 3
  ) %>%
  add_p() %>%                            # 自动选择检验方法
  add_overall() %>%                      # 加总体列
  modify_header(label ~ "**Variable**") %>%
  as_gt()
```

**Stata**
```stata
* 全样本
sum revenue assets leverage roa age employees

* 更详细的格式
sum revenue assets leverage roa age, detail

* 分组对比
bysort treated: sum revenue assets leverage roa

* 均值差异 t 检验
ttest revenue, by(treated) unequal   // Welch t 检验（不假设方差相等）

* 批量 t 检验（循环）
foreach v in revenue assets leverage roa {
    quietly ttest `v', by(treated) unequal
    di "Variable: `v' | t = " r(t) " | p = " r(p)
}
```

---

### 步骤 2：相关系数矩阵

**Python**
```python
import numpy as np
from scipy.stats import pearsonr

def corr_matrix_with_pval(df, vars_list, method="pearson"):
    """
    生成相关系数矩阵，附显著性标注
    上三角显示 p 值星号，下三角显示系数
    """
    n = len(vars_list)
    coef_mat = np.zeros((n, n))
    star_mat = np.full((n, n), "", dtype=object)

    for i in range(n):
        for j in range(n):
            x = df[vars_list[i]].dropna()
            y = df[vars_list[j]].dropna()
            # 取公共非缺失索引
            common = df[[vars_list[i], vars_list[j]]].dropna()
            if len(common) < 3:
                continue
            r, p = pearsonr(common.iloc[:,0], common.iloc[:,1])
            coef_mat[i, j] = round(r, 3)

            if p < 0.01:
                star_mat[i, j] = "***"
            elif p < 0.05:
                star_mat[i, j] = "**"
            elif p < 0.10:
                star_mat[i, j] = "*"

    # 组合系数与星号
    result = pd.DataFrame(index=vars_list, columns=vars_list)
    for i in range(n):
        for j in range(n):
            if i == j:
                result.iloc[i, j] = "1.000"
            elif i > j:
                result.iloc[i, j] = f"{coef_mat[i,j]:.3f}{star_mat[i,j]}"
            else:
                result.iloc[i, j] = ""  # 上三角留空（仅显示下三角）
    return result

corr_tbl = corr_matrix_with_pval(df, VARS)
print(corr_tbl.to_string())
corr_tbl.to_excel("output/tables/correlation_matrix.xlsx")
```

**R**
```r
library(Hmisc)
library(dplyr)

corr_result <- rcorr(as.matrix(df[, VARS]), type = "pearson")

# 系数矩阵
round(corr_result$r, 3)
# p 值矩阵
round(corr_result$P, 3)

# 格式化输出（下三角 + 星号）
format_corr <- function(r_mat, p_mat) {
  n <- nrow(r_mat)
  out <- matrix("", n, n, dimnames = list(rownames(r_mat), colnames(r_mat)))
  for (i in 1:n) {
    for (j in 1:i) {
      r <- round(r_mat[i, j], 3)
      if (i == j) {
        out[i, j] <- "1.000"
      } else {
        p <- p_mat[i, j]
        stars <- ifelse(p < 0.01, "***", ifelse(p < 0.05, "**", ifelse(p < 0.10, "*", "")))
        out[i, j] <- paste0(r, stars)
      }
    }
  }
  as.data.frame(out)
}
```

**Stata**
```stata
pwcorr revenue assets leverage roa age, sig star(0.1)
* sig: 显示 p 值
* star(0.1): 在 10% 显著性水平标注星号
```

---

### 步骤 3：正态性检验

```python
from scipy import stats

def normality_tests(series, alpha=0.05):
    """
    多种正态性检验方法
    """
    s = series.dropna()
    results = {}

    # Shapiro-Wilk（n < 5000 时推荐）
    if len(s) <= 5000:
        stat_sw, p_sw = stats.shapiro(s)
        results["Shapiro-Wilk"] = {"statistic": stat_sw, "p_value": p_sw}

    # Jarque-Bera（适合大样本）
    stat_jb, p_jb = stats.jarque_bera(s)
    results["Jarque-Bera"] = {"statistic": stat_jb, "p_value": p_jb}

    # KS 检验（对正态分布）
    stat_ks, p_ks = stats.kstest(s, "norm", args=(s.mean(), s.std()))
    results["Kolmogorov-Smirnov"] = {"statistic": stat_ks, "p_value": p_ks}

    # 偏度和峰度
    results["Skewness"] = {"value": stats.skew(s)}
    results["Kurtosis"] = {"value": stats.kurtosis(s)}

    return pd.DataFrame(results).T

for v in VARS:
    print(f"\n--- {v} ---")
    print(normality_tests(df[v]))
```

```stata
* Stata: 正态性检验
foreach v in revenue assets leverage roa {
    sktest `v'          // Skewness-Kurtosis test
    swilk `v'          // Shapiro-Wilk
}
```

---

### 步骤 4：方差齐性检验

```python
# Levene 检验（不依赖正态假设）
groups = [df.loc[df["treated"] == g, "revenue"].dropna() for g in [0, 1]]
stat, p = stats.levene(*groups)
print(f"Levene 检验: F = {stat:.4f}, p = {p:.4f}")

# Bartlett 检验（假设正态分布）
stat_b, p_b = stats.bartlett(*groups)
print(f"Bartlett 检验: stat = {stat_b:.4f}, p = {p_b:.4f}")
```

```stata
robvar revenue, by(treated)   // Levene 等方差检验
```

---

### 步骤 5：统计检验

#### t 检验

```python
# 独立样本 t 检验
s0 = df.loc[df["treated"] == 0, "revenue"].dropna()
s1 = df.loc[df["treated"] == 1, "revenue"].dropna()
t, p = stats.ttest_ind(s1, s0, equal_var=False)   # Welch t 检验
print(f"t = {t:.4f}, p = {p:.4f}")

# 配对 t 检验
t_paired, p_paired = stats.ttest_rel(df["before"], df["after"])
```

#### Wilcoxon 符号秩检验（非参数，适用于非正态）

```python
# 独立样本（Mann-Whitney U 检验）
stat_mw, p_mw = stats.mannwhitneyu(s1, s0, alternative="two-sided")
print(f"Mann-Whitney U = {stat_mw:.1f}, p = {p_mw:.4f}")

# 配对样本
stat_w, p_w = stats.wilcoxon(df["before"] - df["after"])
```

```stata
ranksum revenue, by(treated)   // Mann-Whitney U
signrank before = after        // Wilcoxon 配对检验
```

#### 卡方检验

```python
# 独立性检验
ct = pd.crosstab(df["industry"], df["treated"])
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)
print(f"χ² = {chi2:.4f}, df = {dof}, p = {p_chi:.4f}")
```

```stata
tab industry treated, chi2 row col
```

---

### 步骤 6：输出为发表级三线表

**LaTeX 模板（booktabs 风格）**

```python
def to_latex_table(df, caption, label, note=""):
    """将 DataFrame 输出为 LaTeX 三线表代码"""
    n_cols = len(df.columns)
    col_fmt = "l" + "c" * n_cols

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{" + caption + "}",
        r"  \label{" + label + "}",
        r"  \begin{threeparttable}",
        r"  \begin{tabular}{" + col_fmt + "}",
        r"    \toprule",
    ]

    # 表头
    header = " & ".join(["Variable"] + list(df.columns)) + r" \\"
    lines.append(f"    {header}")
    lines.append(r"    \midrule")

    # 数据行
    for idx, row in df.iterrows():
        vals = " & ".join([str(idx)] + [str(v) for v in row.values])
        lines.append(f"    {vals} \\\\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
    ]

    if note:
        lines += [
            r"  \begin{tablenotes}",
            r"    \small",
            r"    \item \textit{Notes}: " + note,
            r"  \end{tablenotes}",
        ]

    lines += [
        r"  \end{threeparttable}",
        r"\end{table}",
    ]
    return "\n".join(lines)

latex_code = to_latex_table(
    tbl.round(3),
    caption="Descriptive Statistics",
    label="tab:descriptive",
    note="***, **, and * indicate significance at 1\%, 5\%, and 10\% levels, respectively."
)
with open("output/tables/table1_descriptive.tex", "w") as f:
    f.write(latex_code)
```

**R (modelsummary)**
```r
library(modelsummary)

# 快速描述性统计表
datasummary_skim(
  df[, c("revenue","assets","leverage","roa","age")],
  output = "output/tables/table1.tex"   # 直接输出 LaTeX
)

# 分组描述性统计
datasummary_balance(
  ~treated,
  data = df[, c("revenue","assets","leverage","roa","treated")],
  output = "output/tables/table1_grouped.tex"
)
```

---

## 检验清单

- [ ] 描述性统计 N 数与回归样本一致
- [ ] 连续变量的 Mean/SD/Min/Max 在合理范围内（无明显数据错误）
- [ ] 分组检验方法选择正确（正态→t 检验，非正态→Wilcoxon）
- [ ] 相关系数矩阵中核心变量相关系数未超过 0.7（VIF 检查多重共线性）
- [ ] 三线表格式：顶线、中线（表头后）、底线，无竖线
- [ ] 表格注释包含星号说明
- [ ] 所有表格输出为 `.tex` 和 `.xlsx` 双格式

---

## 常见错误提醒

1. **t 检验前未检验方差齐性**：默认 `equal_var=True` 会低估 p 值。建议始终用 Welch t 检验（`equal_var=False`）。
2. **大样本下正态检验过于敏感**：n > 1000 时几乎任何分布都会被 Shapiro-Wilk 拒绝正态。此时查看 QQ 图和偏度峰度更有意义。
3. **相关系数忽略缺失值**：`df.corr()` 默认 pairwise 删除，不同格中 N 不同，需注明。
4. **分类变量均值无意义**：虚拟变量（0/1）的均值可以报告，但其 SD 意义有限，通常报告频率（%）更直观。
5. **LaTeX 特殊字符未转义**：变量名中的 `_` 需转义为 `\_`，`%` 转义为 `\%`。

---

## 输出规范

- 描述性统计表：`output/tables/table1_descriptive.tex` + `.xlsx`
- 相关系数矩阵：`output/tables/table_corr.tex` + `.xlsx`
- 统计检验结果：`output/tables/ttest_results.tex`
- 三线表格式要求：顶线 `\toprule`，中线 `\midrule`，底线 `\bottomrule`，禁用竖线
- 数字格式：均值保留3位小数，N 为整数，p 值保留4位小数
- 显著性标注：`***` p<0.01，`**` p<0.05，`*` p<0.10，标注在底部注释
