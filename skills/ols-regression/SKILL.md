# OLS 回归 — 普通最小二乘法

## 概述

OLS（Ordinary Least Squares）是实证研究的基础工具，适用于截面数据分析、基础面板回归及描述性因果关系的建立。在因果推断中，OLS 通常作为基准模型（baseline），为后续的 IV、DID、RDD 等提供参照。本 skill 覆盖完整的 OLS 流程，包括逐步回归策略、诊断检验和规范的结果输出格式。

**适用场景：**
- 截面数据的相关性/描述性分析
- 实验数据（RCT）的平均处理效应估计
- 作为 IV/DID/RDD 的基准规格（OLS baseline）
- 简单面板数据（含固定效应哑变量的 LSDV）
- 机器学习项目中的线性基准模型

---

## 前置条件

### 数据结构要求

```
截面数据：
  - 结果变量 Y（连续变量为主；二元变量考虑 Logit/LPM）
  - 核心解释变量 X
  - 控制变量 W（多个）
  - 分组/固定效应变量（industry, province，可选）

面板数据（简单情形）：
  - 见 panel-data/SKILL.md 获取完整面板分析流程

变量类型检查：
  - 连续变量取值范围合理
  - 分类变量已转为哑变量（保留一个基准类）
  - 无完全多重共线性（全部 dummy 陷阱）
```

### 前置描述性统计

```python
# Python: 描述性统计
import pandas as pd

desc = df[all_vars].describe().T
desc['missing_pct'] = df[all_vars].isnull().mean() * 100
print(desc.round(3))

# 相关系数矩阵
import seaborn as sns
corr_matrix = df[all_vars].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True)
import matplotlib.pyplot as plt
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('output/ols_corr_heatmap.png', dpi=150)
```

```r
# R: 描述性统计
library(modelsummary)

datasummary_skim(df[, all_vars])  # 或
datasummary(All(df) ~ Mean + SD + Min + Max + N, data = df)
```

```stata
* Stata: 描述性统计
summarize y x1 x2 x3 control1 control2, detail
pwcorr y x1 x2 x3 control1 control2, star(0.05)
```

---

## 分析步骤

### Step 1：逐步回归策略（Baseline → +Controls → +FE）

逐步加入控制变量和固定效应，展示核心系数的稳定性。若核心系数随控制变量加入发生剧烈变化（> 30%），说明遗漏变量偏误可能较大。

```python
# Python: statsmodels OLS 逐步回归
import statsmodels.formula.api as smf
import pandas as pd

results = {}

# 列 (1): 仅核心变量
results['(1) Baseline'] = smf.ols(f"y ~ x + {' + '.join(industry_dummies)}", df).fit(cov_type='HC3')

# 列 (2): + 控制变量
results['(2) +Controls'] = smf.ols(
    f"y ~ x + {' + '.join(controls)} + {' + '.join(industry_dummies)}", df
).fit(cov_type='HC3')

# 列 (3): + 省份固定效应
results['(3) +Province FE'] = smf.ols(
    f"y ~ x + {' + '.join(controls)} + C(province) + C(industry)", df
).fit(cov_type='HC3')

# 汇总：提取核心系数
coef_table = pd.DataFrame({
    name: {
        'coef': res.params['x'],
        'se':   res.HC3_se['x'],
        'pval': res.pvalues['x'],
        'N':    int(res.nobs),
        'R2':   res.rsquared
    }
    for name, res in results.items()
})
print(coef_table.round(4))
```

```r
# R: fixest 逐步回归（推荐）
library(fixest)

res1 <- feols(y ~ x,                         data = df, vcov = "hetero")
res2 <- feols(y ~ x + control1 + control2,   data = df, vcov = "hetero")
res3 <- feols(y ~ x + control1 + control2 | industry, data = df, cluster = ~industry)
res4 <- feols(y ~ x + control1 + control2 | industry + province, data = df, cluster = ~province)

etable(res1, res2, res3, res4,
       title   = "OLS Stepwise Regressions",
       headers = c("Baseline", "+Controls", "+Industry FE", "+Province FE"),
       fitstat = ~ r2 + n)
```

```stata
* Stata: 逐步回归
reg y x, robust
eststo model1

reg y x control1 control2, robust
eststo model2

areg y x control1 control2, absorb(industry) robust
eststo model3

areg y x control1 control2 i.province, absorb(industry) cluster(province)
eststo model4

esttab model1 model2 model3 model4, ///
    b(3) se(3) star(* 0.1 ** 0.05 *** 0.01) ///
    stats(N r2 r2_a) mtitles("Baseline" "+Controls" "+Industry FE" "+Province FE")
```

---

### Step 2：多重共线性检验（VIF）

多重共线性不影响 OLS 估计量的无偏性，但会膨胀标准误，导致系数不显著。VIF > 10 为严重问题，VIF > 5 需关注。

```python
# Python: VIF 检验
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import pandas as pd

X_matrix = sm.add_constant(df[['x'] + controls])
vif_data = pd.DataFrame({
    'variable': X_matrix.columns,
    'VIF': [variance_inflation_factor(X_matrix.values, i)
            for i in range(X_matrix.shape[1])]
})
print(vif_data.sort_values('VIF', ascending=False))
# VIF > 10: 严重共线性问题
# VIF 5-10: 中等，需关注
```

```r
# R: VIF（car 包）
library(car)

res_vif <- lm(y ~ x + control1 + control2 + control3, data = df)
vif_vals <- vif(res_vif)
print(vif_vals)
# 若存在分类变量，vif() 返回 GVIF（广义VIF）
# 比较 GVIF^(1/(2*Df)) vs sqrt(10) ≈ 3.16
```

```stata
* Stata: VIF
reg y x control1 control2 control3
vif
* VIF > 10: 严重共线性
```

**处理高VIF：**
1. 查看相关系数矩阵，找到高度相关的变量对
2. 移除冗余变量（如 GDP 和人均 GDP 不同时放入）
3. 对高相关变量做主成分（PCA）降维
4. 中心化（对交乘项）

---

### Step 3：异方差检验

```python
# Python: White 检验和 Breusch-Pagan 检验
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
import statsmodels.api as sm

res_ols = smf.ols(f"y ~ x + {' + '.join(controls)}", df).fit()
X_exog  = sm.add_constant(df[['x'] + controls])

# White 检验
white_stat, white_p, _, _ = het_white(res_ols.resid, X_exog)
print(f"White test: chi2={white_stat:.3f}, p={white_p:.4f}")

# Breusch-Pagan 检验
bp_stat, bp_p, _, _ = het_breuschpagan(res_ols.resid, X_exog)
print(f"Breusch-Pagan: LM={bp_stat:.3f}, p={bp_p:.4f}")

# p < 0.05 → 存在异方差 → 使用稳健标准误
if white_p < 0.05:
    res_robust = smf.ols(f"y ~ x + {' + '.join(controls)}", df).fit(cov_type='HC3')
    print("Using HC3 robust SE")
```

```r
# R: lmtest + sandwich
library(lmtest)
library(sandwich)

res_lm <- lm(y ~ x + control1 + control2, data = df)

# Breusch-Pagan 检验
bptest(res_lm)

# White 检验
bptest(res_lm, ~ fitted(res_lm) + I(fitted(res_lm)^2))

# 若异方差显著，使用 HC3 稳健 SE
coeftest(res_lm, vcov = vcovHC(res_lm, type = "HC3"))

# 或直接用 fixest 的 vcov="hetero"
```

```stata
* Stata: 异方差检验
reg y x control1 control2
estat hettest            * Breusch-Pagan
estat imtest, white      * White 检验

* 使用稳健 SE
reg y x control1 control2, robust
* 聚类 SE
reg y x control1 control2, cluster(industry)
```

---

### Step 4：残差诊断图

```python
# Python: 4 格残差诊断图
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

res_ols = smf.ols(f"y ~ x + {' + '.join(controls)}", df).fit()
fitted  = res_ols.fittedvalues
resid   = res_ols.resid
std_resid = resid / resid.std()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 1. Residuals vs Fitted
axes[0,0].scatter(fitted, resid, alpha=0.3)
axes[0,0].axhline(0, color='red', linestyle='--')
axes[0,0].set(xlabel='Fitted values', ylabel='Residuals', title='Residuals vs Fitted')

# 2. Q-Q Plot（正态性检验）
stats.probplot(resid, dist="norm", plot=axes[0,1])
axes[0,1].set_title('Normal Q-Q Plot')

# 3. Scale-Location（同方差性）
axes[1,0].scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.3)
axes[1,0].set(xlabel='Fitted values', ylabel='√|Std Residuals|', title='Scale-Location')

# 4. 残差直方图
axes[1,1].hist(resid, bins=30, edgecolor='black')
axes[1,1].set(xlabel='Residuals', title='Residuals Distribution')

plt.tight_layout()
plt.savefig('output/ols_residual_diagnostics.png', dpi=150)
```

```r
# R: 残差诊断（base R 或 ggplot2）
par(mfrow = c(2, 2))
plot(res_lm)  # 自动生成4格诊断图

# 或使用 ggplot2 + ggfortify
library(ggfortify)
autoplot(res_lm)
```

---

### Step 5：异常值检验

```python
# Python: 影响力诊断
influence = res_ols.get_influence()
cooks_d   = influence.cooks_distance[0]
leverage  = influence.hat_matrix_diag

# Cook's Distance > 4/N 为潜在异常值
threshold = 4 / len(df)
outliers  = df[cooks_d > threshold].index
print(f"Influential observations (Cook's D > {threshold:.4f}): {len(outliers)}")

# 稳健性：剔除异常值后重跑
df_clean = df.drop(outliers)
res_clean = smf.ols(f"y ~ x + {' + '.join(controls)}", df_clean).fit(cov_type='HC3')
```

```r
# R: 异常值与影响力
# Cook's D
cooksd <- cooks.distance(res_lm)
plot(cooksd, pch = "*", main = "Cook's Distance")
abline(h = 4/nrow(df), col = "red", lty = 2)

# 高杠杆点
hat_vals <- hatvalues(res_lm)
high_lev  <- which(hat_vals > 2 * mean(hat_vals))

# 稳健回归（作为稳健性检验）
library(MASS)
res_robust_reg <- rlm(y ~ x + control1 + control2, data = df)
```

---

### Step 6：标准化系数（Beta coefficients）

在控制变量单位不同、需要比较效应大小时，报告标准化系数（beta）。

```r
# R: 标准化系数（lm.beta 或手动）
library(lm.beta)
lm.beta(res_lm)   # 返回标准化系数

# 手动：Z-score 标准化后回归
df_std <- df %>% mutate(across(all_of(c("y", "x", controls)), scale))
res_std <- lm(y ~ x + control1 + control2, data = df_std)
summary(res_std)  # 系数即为 beta
```

---

## 必做检验清单

| 检验 | 方法 | 通过标准 / 说明 |
|------|------|---------------|
| VIF 多重共线性 | car::vif() | 所有 VIF < 10（理想 < 5） |
| 异方差检验 | Breusch-Pagan / White | 若显著则使用稳健 SE |
| 残差正态性 | Q-Q 图 + Shapiro-Wilk | 大样本下不强制要求 |
| 残差 vs 拟合 | 散点图 | 无系统性模式 |
| 异常值/影响点 | Cook's D | 检查 D > 4/N 的观测 |
| 逐步回归稳定性 | 系数变化幅度 | 加入控制后变化 < 30% |
| F 检验（模型整体） | 联合显著性 | p < 0.05 |
| 样本外预测（可选） | Cross-validation | 防止过拟合 |

---

## 常见错误提醒

> **错误 1：截距项被遗漏**
> 几乎所有 OLS 模型都应包含截距项。在 Stata `reg` 和 R `lm` 中默认包含，Python `statsmodels` 的 `smf.ols` 也默认包含，但 `sm.OLS` 需要手动 `add_constant()`。

> **错误 2：哑变量陷阱（Dummy Variable Trap）**
> 对分类变量（如行业）创建哑变量时，必须删去一个基准类（reference category）。R `lm()` 和 Stata `reg i.industry` 自动处理；Python 需要 `pd.get_dummies(..., drop_first=True)`。

> **错误 3：直接比较不同样本的 R²**
> R² 随控制变量增加而必然上升，不能用于比较不同规格。应报告 adjusted R²，或在 FE 模型中报告 within R²。

> **错误 4：显著性星号当成效应大小**
> 显著性（p < 0.05）≠ 实际重要。大样本下微小效应也会显著。**必须同时报告系数大小和置信区间**，并结合经济意义讨论（如弹性、标准差单位效应）。

> **错误 5：忽略标准误类型**
> 截面数据若存在异方差，必须使用稳健 SE（HC3）。面板/聚类数据需聚类 SE。不同 SE 计算方法可能导致结论大幅变化，必须在表格脚注中明确注明。

> **错误 6：OLS 在非线性关系中的滥用**
> OLS 假设线性关系。在提交前用散点图（Y vs X）检查是否明显非线性；如有需要加入平方项或取对数。二元结果变量用 LPM（OLS 可行但有边界问题）或 Logit/Probit。

> **错误 7：不报告样本量和缺失值处理**
> 必须在表格下方注明 N，以及如何处理缺失值（listwise deletion？多重填补？）。

---

## 输出规范

### 三线表格式（Markdown）

```markdown
**Table X: OLS Regression Results**

|                    | (1) Baseline | (2) +Controls | (3) +Industry FE | (4) +Province FE |
|--------------------|:------------:|:-------------:|:----------------:|:----------------:|
| Core Variable X    |  0.245***    |  0.198***     |  0.176**         |  0.163**         |
|                    |  (0.052)     |  (0.048)      |  (0.071)         |  (0.069)         |
| Control 1          |              |  0.034        |  0.028           |  0.031           |
|                    |              |  (0.021)      |  (0.022)         |  (0.023)         |
| Constant           |  2.341***    |  1.876***     |  —               |  —               |
| Industry FE        |  No          |  No           |  Yes             |  Yes             |
| Province FE        |  No          |  No           |  No              |  Yes             |
| N                  |  1,250       |  1,250        |  1,250           |  1,250           |
| R²                 |  0.142       |  0.231        |  0.318           |  0.367           |

*Notes: Robust standard errors in parentheses (HC3). *** p<0.01, ** p<0.05, * p<0.1.*
```

### LaTeX 输出

```r
# R: kableExtra / modelsummary LaTeX 输出
library(modelsummary)

modelsummary(
  list("(1)" = res1, "(2)" = res2, "(3)" = res3, "(4)" = res4),
  stars   = c("*" = 0.1, "**" = 0.05, "***" = 0.01),
  vcov    = "robust",    # 或 ~cluster_var
  gof_map = c("nobs", "r.squared", "adj.r.squared"),
  output  = "output/ols_main_table.tex",
  title   = "OLS Regression Results"
)
```

```stata
* Stata: esttab LaTeX 输出
esttab model1 model2 model3 model4 using "output/ols_main_table.tex", ///
    b(3) se(3) star(* 0.1 ** 0.05 *** 0.01) ///
    booktabs label replace ///
    stats(N r2 r2_a, fmt(%9.0f %9.3f %9.3f) labels("N" "R²" "Adj. R²")) ///
    note("Robust standard errors in parentheses.")
```

**表格必须包含：**
1. 逐步规格列（展示核心系数稳定性）
2. 括号内为标准误（注明类型）
3. 显著性星号（注明水平）
4. 固定效应/控制变量 Yes/No 标注
5. N、R²（或 within R²）

### 文件命名

```
output/
  ols_descriptive_stats.csv      # 描述性统计
  ols_corr_heatmap.png           # 相关系数热力图
  ols_vif_check.csv              # VIF 检验
  ols_heteroskedasticity.txt     # 异方差检验
  ols_residual_diagnostics.png   # 残差诊断图
  ols_main_table.csv             # 主结果表（CSV）
  ols_main_table.tex             # 主结果表（LaTeX）
  ols_robustness.csv             # 稳健性检验
```
