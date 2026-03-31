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

---

### Section: Specification Curve Analysis

穷举所有合理的控制变量组合（2^k种），把所有回归的核心系数画在一张图上，直观展示结论在不同规格下的稳健性。

#### R代码（specr包）

```r
# R: Specification Curve Analysis
# install.packages("specr")
library(specr)
library(ggplot2)
library(dplyr)

# Step 1: 定义规格空间
# y：因变量（可多个）
# x：核心解释变量（可多个）
# controls：候选控制变量（2^k种组合）
# model：估计方法

specs <- setup(
  data     = df,
  y        = c("y"),                                  # 因变量
  x        = c("x1"),                                 # 核心解释变量
  model    = c("lm"),                                 # 估计方法（"lm", "glm"等）
  controls = c("control1", "control2", "control3",    # 候选控制变量（穷举组合）
               "control4"),
  add_to_formula = "| entity_id + year"               # 固定效应（可选）
)

# Step 2: 运行所有规格
results <- run_specs(
  specs      = specs,
  .progress  = TRUE                                   # 显示进度
)

# Step 3: 汇总结果
summary(results)
# 关键看：核心系数的中位数、方向一致性、显著比例

# Step 4: Specification Curve图
p <- plot_specs(
  results,
  choices     = c("controls"),                        # 显示控制变量选择
  rel_heights = c(1, 2),                              # 上下图比例
  desc        = FALSE                                 # 按系数大小排序
)
ggsave("output/specification_curve.png", p, width = 12, height = 8, dpi = 300)

# Step 5: 统计摘要
cat("规格总数:", nrow(results), "\n")
cat("系数中位数:", median(results$estimate), "\n")
cat("系数方向一致性:", mean(results$estimate > 0), "\n")
cat("统计显著比例(p<0.05):", mean(results$p.value < 0.05), "\n")

# Step 6: 自定义多面板图（更美观）
# 上图：所有规格的系数+置信区间
# 下图：对应的控制变量选择矩阵
plot_specs(results,
           choices  = c("controls", "model"),
           ci       = TRUE,
           ribbon   = TRUE)
```

#### Python代码（循环 + matplotlib图）

```python
# Python: Specification Curve Analysis（手动穷举）
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.formula.api as smf
from tqdm import tqdm

# 定义候选控制变量
candidate_controls = ['control1', 'control2', 'control3', 'control4']
core_var = 'x1'
outcome  = 'y'

# 穷举所有控制变量组合（2^k种）
all_specs = []
for k in range(len(candidate_controls) + 1):
    for combo in itertools.combinations(candidate_controls, k):
        all_specs.append(list(combo))

print(f"总规格数: {len(all_specs)}")

# 运行所有规格
results = []
for controls in tqdm(all_specs):
    formula_parts = [core_var] + controls
    formula = f"{outcome} ~ {' + '.join(formula_parts)}"
    try:
        res = smf.ols(formula, data=df).fit(cov_type='HC3')
        results.append({
            'controls':  tuple(controls),
            'n_controls': len(controls),
            'coef':  res.params[core_var],
            'se':    res.bse[core_var],
            'ci_low': res.conf_int().loc[core_var, 0],
            'ci_high': res.conf_int().loc[core_var, 1],
            'pval':  res.pvalues[core_var],
            'r2':    res.rsquared,
            'n':     int(res.nobs),
            **{f'has_{c}': (c in controls) for c in candidate_controls}
        })
    except Exception as e:
        continue

df_spec = pd.DataFrame(results).sort_values('coef').reset_index(drop=True)

# ============================================================
# 绘制Specification Curve图
# ============================================================
fig = plt.figure(figsize=(14, 9))
gs  = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.05)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# 上图：系数+置信区间
colors = ['#2196F3' if p < 0.05 else '#BBDEFB' for p in df_spec['pval']]
ax1.scatter(range(len(df_spec)), df_spec['coef'], c=colors, s=15, zorder=3)
ax1.fill_between(range(len(df_spec)), df_spec['ci_low'], df_spec['ci_high'],
                 alpha=0.2, color='#2196F3')
ax1.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax1.set_ylabel(f'{core_var} 系数', fontsize=11)
ax1.set_title('Specification Curve Analysis', fontsize=13, fontweight='bold')
ax1.set_xlim(-0.5, len(df_spec) - 0.5)

# 统计标注
sig_pct = mean(df_spec['pval'] < 0.05) * 100
pos_pct = mean(df_spec['coef'] > 0) * 100
ax1.text(0.02, 0.95, f"显著(p<0.05): {sig_pct:.0f}% | 正向: {pos_pct:.0f}%",
         transform=ax1.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# 下图：控制变量选择矩阵
for i, ctrl in enumerate(candidate_controls):
    col_name = f'has_{ctrl}'
    y_pos    = len(candidate_controls) - 1 - i
    for j, has_it in enumerate(df_spec[col_name]):
        ax2.scatter(j, y_pos, color='#1976D2' if has_it else '#E0E0E0',
                    s=8, marker='s')

ax2.set_yticks(range(len(candidate_controls)))
ax2.set_yticklabels(candidate_controls[::-1], fontsize=9)
ax2.set_xlim(-0.5, len(df_spec) - 0.5)
ax2.set_xlabel('规格排序（按系数大小）', fontsize=10)
ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

plt.savefig('output/specification_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"规格曲线图已保存")
print(df_spec[['coef','se','pval','r2']].describe().round(4))
```

---

### Section: Oster (2019) 系数稳定性检验

Oster (2019) 检验遗漏变量偏误的方法：通过计算δ值（bias-adjusted coefficient），衡量"遗漏变量需要多强才能推翻结论"。

**核心逻辑：**
- $\tilde{\beta}$：加入控制变量后的系数
- $\dot{\beta}$：仅有核心变量的系数
- δ：遗漏变量需要与处理变量的相关程度（相对于控制变量）才能使真实效应为0
- **若 δ > 1 → 结论稳健**（遗漏变量需比已有控制变量与处理变量的相关性还强，才能推翻结论）

**R²_max设定原则：**
- 通常取 1.3 × R²_controlled（Oster 2019建议）
- 或取理论上界（如 R²=0.9，因变量不可能被完美解释）
- 敏感性分析中尝试多个R²_max值

#### R代码（手动计算）

```r
# R: Oster (2019) 系数稳定性检验
# 参考 Stata psacalc 的逻辑

#' Oster (2019) δ系数计算
#' @param beta_restricted  仅有核心变量X的系数
#' @param beta_controlled  加入控制变量后的系数
#' @param r2_restricted    仅有核心变量X的R²
#' @param r2_controlled    加入控制变量后的R²
#' @param r2_max           R²上界（默认1.3 × r2_controlled）
#' @param beta_null        原假设下的系数（通常=0）
#' @return δ值

oster_delta <- function(beta_restricted,
                        beta_controlled,
                        r2_restricted,
                        r2_controlled,
                        r2_max = NULL,
                        beta_null = 0) {
  if (is.null(r2_max)) {
    r2_max <- min(1.3 * r2_controlled, 0.99)  # Oster建议 + 上限1
  }

  # Oster (2019) 公式
  # δ = (β_controlled - β_null) / (β_restricted - β_controlled)
  #     × (R²_max - R²_controlled) / (R²_controlled - R²_restricted)

  delta <- ((beta_controlled - beta_null) / (beta_restricted - beta_controlled)) *
            ((r2_max - r2_controlled) / (r2_controlled - r2_restricted))

  return(delta)
}

# ============================================================
# 使用示例
# ============================================================
library(fixest)

# 跑两个规格
res_restricted <- feols(y ~ x1, data = df, vcov = "hetero")
res_controlled <- feols(y ~ x1 + control1 + control2 + control3,
                        data = df, vcov = "hetero")

# 提取所需数值
b_res  <- coef(res_restricted)["x1"]
b_ctrl <- coef(res_controlled)["x1"]
r2_res  <- r2(res_restricted)["r2"]
r2_ctrl <- r2(res_controlled)["r2"]

# 计算δ（三种R²_max假设）
r2_max_vals <- c(
  "1.3×R²_ctrl" = 1.3 * r2_ctrl,
  "1.5×R²_ctrl" = 1.5 * r2_ctrl,
  "理论上界0.9"  = 0.9
)

cat("=== Oster (2019) 系数稳定性检验 ===\n")
cat(sprintf("仅含X的系数 (β_res):  %.4f  R²: %.4f\n", b_res, r2_res))
cat(sprintf("含控制变量系数 (β_ctrl): %.4f  R²: %.4f\n", b_ctrl, r2_ctrl))
cat(sprintf("系数变化幅度: %.1f%%\n", abs(b_ctrl - b_res) / abs(b_res) * 100))
cat("\nδ值（遗漏变量强度倍数）：\n")

for (name in names(r2_max_vals)) {
  delta <- oster_delta(b_res, b_ctrl, r2_res, r2_ctrl, r2_max_vals[name])
  flag  <- if (delta > 1) "✓ 稳健" else "⚠️ 需关注"
  cat(sprintf("  R²_max = %s: δ = %.3f  → %s\n", name, delta, flag))
}

cat("\n解读：若δ > 1 → 遗漏变量需比现有控制变量更强才能推翻结论\n")

# ============================================================
# bias-adjusted系数（β*）
# ============================================================
# 若δ=1（假设遗漏变量偏误与观测到的偏误等强），β*是多少？

beta_star <- function(b_res, b_ctrl, r2_res, r2_ctrl, r2_max, delta = 1) {
  # 当δ=1时求解β_null（即bias-adjusted coefficient）
  # 公式变形：β* = β_ctrl - delta × (β_res - β_ctrl) × (R²_max - R²_ctrl)/(R²_ctrl - R²_res)
  b_star <- b_ctrl - delta * (b_res - b_ctrl) *
              (r2_max - r2_ctrl) / (r2_ctrl - r2_res)
  return(b_star)
}

b_star <- beta_star(b_res, b_ctrl, r2_res, r2_ctrl, 1.3 * r2_ctrl)
cat(sprintf("\nbias-adjusted β*(δ=1, R²_max=1.3×R²_ctrl): %.4f\n", b_star))
cat(sprintf("（若β*与0同号且显著，结论更稳健）\n"))
```

---

### Section: 经济显著性三层解读

统计显著性（p值）≠ 经济显著性。必须同时报告实际效应大小。

#### 第一层：基础换算（标准差单位效应）

$$\text{标准化效应} = \frac{\hat{\beta} \times SD(X)}{Mean(Y)} \times 100\%$$

即：X增加1个标准差，Y相对于其均值变化多少百分比。

#### 第二层：文献对比

提供参考框架，帮助读者判断效应是否合理。例如："同类DID研究中政策效果一般在3-8%之间，本文估计的4.2%处于正常区间。"

#### 第三层：政策换算（如适用）

将系数换算为可操作的政策含义。例如："每增加1单位政策投入，对应产出增加X万元，投入产出比为Y。"

#### 自动计算代码模板

```python
# Python: 经济显著性三层解读自动计算
import numpy as np
import pandas as pd

def economic_significance(
    coef,           # OLS系数
    se,             # 标准误
    x_std,          # 自变量标准差
    y_mean,         # 因变量均值
    y_std,          # 因变量标准差
    variable_name="X",
    outcome_name="Y",
    literature_range=None,  # 文献中效应范围 (min, max) %
    policy_cost=None,       # 每单位政策成本（如元）
    policy_unit="元"
):
    """经济显著性三层解读"""
    print(f"{'='*50}")
    print(f"经济显著性解读: {variable_name} → {outcome_name}")
    print(f"{'='*50}")

    # 第一层：标准差单位效应
    effect_1sd = coef * x_std
    effect_1sd_pct = effect_1sd / y_mean * 100

    print(f"\n【第一层：标准差单位效应】")
    print(f"  系数 β = {coef:.4f}")
    print(f"  {variable_name} 标准差 = {x_std:.4f}")
    print(f"  β × SD({variable_name}) = {effect_1sd:.4f}")
    print(f"  → {variable_name}增加1个标准差，{outcome_name}变化: {effect_1sd:.4f}")
    print(f"  → 相当于{outcome_name}均值的 {effect_1sd_pct:.1f}%")

    # 标准化效应（Cohen's d类比）
    effect_std = coef * x_std / y_std
    print(f"  → 标准化效应 = {effect_std:.3f} 个{outcome_name}标准差")

    # 第二层：文献对比
    print(f"\n【第二层：文献对比】")
    if literature_range:
        lo, hi = literature_range
        in_range = lo <= effect_1sd_pct <= hi
        print(f"  文献参考区间: [{lo}%, {hi}%]")
        print(f"  本文估计: {effect_1sd_pct:.1f}%")
        status = "在文献区间内 ✓" if in_range else "超出文献区间，需解释 ⚠️"
        print(f"  → {status}")
    else:
        print(f"  （请填入同类文献的效应范围作为参照）")

    # 第三层：政策换算
    print(f"\n【第三层：政策换算】")
    if policy_cost is not None:
        output_per_unit = coef * policy_cost
        print(f"  每投入 1{policy_unit} 政策成本")
        print(f"  → {outcome_name}变化: {output_per_unit:.4f}")
        print(f"  → 投入产出比: 1:{abs(output_per_unit):.2f}")
    else:
        print(f"  （如适用：填入政策成本进行换算）")

    print(f"\n{'='*50}")
    return {
        'coef': coef, 'effect_1sd': effect_1sd,
        'effect_pct': effect_1sd_pct, 'effect_std': effect_std
    }

# 使用示例
econ_result = economic_significance(
    coef          = 0.045,      # β系数
    se            = 0.012,
    x_std         = df['x1'].std(),
    y_mean        = df['y'].mean(),
    y_std         = df['y'].std(),
    variable_name = "数字化水平",
    outcome_name  = "企业绩效",
    literature_range = (2, 8),  # 文献中2-8%
    policy_cost   = 10000,      # 每万元补贴
    policy_unit   = "万元"
)
```

```r
# R: 经济显著性三层解读
economic_significance_r <- function(
  model,              # lm或feols对象
  x_var,              # 核心解释变量名
  data,               # 原始数据框
  y_var,              # 因变量名
  literature_range = NULL,
  policy_cost = NULL,
  policy_unit = "元"
) {
  coef_val <- coef(model)[x_var]
  se_val   <- sqrt(vcov(model)[x_var, x_var])

  x_std  <- sd(data[[x_var]], na.rm = TRUE)
  y_mean <- mean(data[[y_var]], na.rm = TRUE)
  y_std  <- sd(data[[y_var]], na.rm = TRUE)

  effect_1sd     <- coef_val * x_std
  effect_1sd_pct <- effect_1sd / y_mean * 100
  effect_std_    <- effect_1sd / y_std

  cat("=== 经济显著性解读 ===\n")
  cat(sprintf("系数 β = %.4f (SE=%.4f)\n", coef_val, se_val))
  cat(sprintf("\n【第一层】X增加1SD → Y变化 %.4f (均值的 %.1f%%，SD的 %.3f)\n",
              effect_1sd, effect_1sd_pct, effect_std_))

  if (!is.null(literature_range)) {
    in_range <- effect_1sd_pct >= literature_range[1] & effect_1sd_pct <= literature_range[2]
    cat(sprintf("\n【第二层】文献区间[%g%%, %g%%]，本文 %.1f%% → %s\n",
                literature_range[1], literature_range[2],
                effect_1sd_pct, if(in_range) "在区间内 ✓" else "超出区间 ⚠️"))
  }

  if (!is.null(policy_cost)) {
    cat(sprintf("\n【第三层】每投入1%s → Y变化 %.4f（投入产出比 1:%.2f）\n",
                policy_unit, coef_val * policy_cost, abs(coef_val * policy_cost)))
  }
}

# 使用示例
economic_significance_r(res_controlled, "x1", df, "y",
                        literature_range = c(2, 8),
                        policy_cost = 10000, policy_unit = "万元")
```

---

### Section: OLS Estimand声明

| 设定 | Estimand | 声明要求 |
|------|---------|---------|
| OLS+控制变量（同质效应假设） | ATE（平均处理效应） | 需论证效应为何可视为同质（如RCT或强可信的CIA） |
| OLS+控制变量（异质效应） | 处理方差加权平均≠ATE | 标注这**不是**ATE，如需ATE建议IPW/matching |
| OLS作为基准（描述性） | 条件相关系数 | 明确声明非因果，仅描述性 |

**声明模板（论文正文方法节）：**

```
本文OLS估计的核心系数为处理变量X的条件均值效应。
在效应同质性假设下，该系数可解释为平均处理效应（ATE）。
然而，若效应存在异质性，OLS系数为处理方差加权平均，
可能偏离ATE（Angrist and Pischke, 2009）。
为此，第X节提供基于IPW的ATE估计作为稳健性检验。
```

---

### Section: 错误补充

**⚠️ 错误8（新增）："默认OLS系数=ATE"**

**错误表现：** 直接将OLS系数报告为"政策的平均处理效应（ATE）"，未说明效应同质性假设。

**为何错误：** 在异质效应下，OLS系数是**处理方差加权平均**（treatment-variance weighted average），数学上等价于：

$$\hat{\beta}_{OLS} = \frac{\sum_i (X_i - \bar{X})^2 \cdot \tau_i}{\sum_i (X_i - \bar{X})^2}$$

其中 $\tau_i$ 是个体处理效应。权重取决于处理强度的方差，而非个体的代表性，**不等于**真正的ATE。

**正确做法：**
- 若确信效应同质（如RCT），明确声明并解释理由
- 若可能存在异质效应，标注系数是"处理方差加权平均"
- 如需ATE，使用IPW（逆概率加权）或matching方法

```r
# R: IPW估计ATE（与OLS对比）
library(WeightIt)

# 估计倾向得分并计算IPW权重
w_out <- weightit(
  x1 ~ control1 + control2 + control3,
  data   = df,
  method = "ps",      # 倾向得分加权
  estimand = "ATE"    # 目标：ATE（非ATT）
)
summary(w_out)

# 用IPW权重做加权回归
library(survey)
svy_design <- svydesign(ids = ~1, data = df, weights = w_out$weights)
res_ipw <- svyglm(y ~ x1, design = svy_design)
summary(res_ipw)

# 对比OLS vs IPW
cat("\n=== OLS vs IPW ATE 对比 ===\n")
cat(sprintf("OLS系数（处理方差加权平均）: %.4f\n", coef(res_controlled)["x1"]))
cat(sprintf("IPW系数（ATE）: %.4f\n", coef(res_ipw)["x1"]))
cat("差异越大 → 效应异质性越强，OLS越偏离ATE\n")
```
