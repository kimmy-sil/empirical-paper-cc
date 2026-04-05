# OLS 回归分析

适用场景：截面/面板基准模型、RCT平均效应估计、描述性相关分析。触发关键词：OLS、线性回归、逐步回归、VIF、异方差、稳健标准误、系数稳定性。

---

## 概述与前置条件

OLS是实证研究的基础工具，通常作为基准模型（baseline）为IV/DID/RDD提供参照。在因果推断中OLS估计量的解读取决于效应同质性假设（见第7节Estimand声明）。

**前置数据要求：**
```
截面数据：
  - 结果变量 Y（连续变量；二元变量考虑LPM/Logit）
  - 核心解释变量 X
  - 控制变量 W（多个）
  - 分组变量（industry, province，可选）

变量检查：
  - 连续变量取值范围合理，无异常尖峰
  - 分类变量已转为哑变量（保留一个基准类）
  - 无完全多重共线性
```

```python
# Python: 前置描述性统计（返回dict/DataFrame，不print）
import pandas as pd
import numpy as np

def describe_vars(df, all_vars):
    desc = df[all_vars].describe().T
    desc['missing_pct'] = df[all_vars].isnull().mean() * 100
    desc['skewness']    = df[all_vars].skew()
    return desc.round(3)

desc_table = describe_vars(df, [outcome] + [core_var] + controls)
```

```r
# R: 前置描述性统计
library(modelsummary)
datasummary_skim(df[, c(outcome, core_var, controls)])
```

---

## Step 1: 逐步回归

逐步加入控制变量和固定效应，展示核心系数稳定性。核心系数变化 > 30% 预示遗漏变量偏误。

> **⚠️ 仅供探索：** 逐步回归用于展示系数稳健性，不得作为因果推断论文的正式识别策略。
> **替代方案：** 需要正式稳健性证据时，用 Step 5（Specification Curve）或 LASSO 变量选择。

```python
# Python: 逐步回归（statsmodels）
import statsmodels.formula.api as smf
import pandas as pd

def stepwise_ols(df, outcome, core_var, controls, fe_vars=None):
    """逐步OLS回归，返回结果dict（非print）"""
    specs = {
        '(1) Baseline': f"{outcome} ~ {core_var}",
        '(2) +Controls': f"{outcome} ~ {core_var} + {' + '.join(controls)}",
    }
    if fe_vars:
        fe_str = ' + '.join([f'C({v})' for v in fe_vars])
        specs['(3) +FE'] = f"{outcome} ~ {core_var} + {' + '.join(controls)} + {fe_str}"

    results = {}
    for name, formula in specs.items():
        res = smf.ols(formula, df).fit(cov_type='HC3')
        results[name] = {
            'coef': res.params[core_var],
            'se':   res.bse[core_var],
            'pval': res.pvalues[core_var],
            'N':    int(res.nobs),
            'R2':   round(res.rsquared, 4)
        }

    coef_table = pd.DataFrame(results).T
    # 检查系数稳定性
    if '(1) Baseline' in results and '(3) +FE' in results:
        b0 = results['(1) Baseline']['coef']
        b3 = list(results.values())[-1]['coef']
        change_pct = abs(b3 - b0) / abs(b0) * 100
        coef_table['note'] = ''
        if change_pct > 30:
            print(f"⚠️ 核心系数变化 {change_pct:.1f}%，遗漏变量偏误风险高")
    return coef_table

result_table = stepwise_ols(df, 'Y', 'X', controls, fe_vars=['industry', 'province'])
```

```r
# R: 逐步回归（fixest，推荐）
library(fixest)

res1 <- feols(Y ~ X,                          data = df, vcov = "hetero")
res2 <- feols(Y ~ X + control1 + control2,    data = df, vcov = "hetero")
res3 <- feols(Y ~ X + control1 + control2 | industry,       data = df, cluster = ~industry)
res4 <- feols(Y ~ X + control1 + control2 | industry + province, data = df, cluster = ~province)

etable(res1, res2, res3, res4,
       headers = c("Baseline", "+Controls", "+Industry FE", "+Province FE"),
       fitstat = ~ r2 + n)
```

---

## Step 2: 诊断检验

### 2a: VIF 多重共线性

VIF > 10：严重问题；VIF > 5：需关注。

```python
# Python: VIF检验（返回DataFrame）
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import pandas as pd

def calc_vif(df, predictors):
    X = sm.add_constant(df[predictors])
    vif_df = pd.DataFrame({
        'variable': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    return vif_df[vif_df['variable'] != 'const'].sort_values('VIF', ascending=False)

vif_result = calc_vif(df, [core_var] + controls)
# 处理高VIF：检查相关系数矩阵，移除冗余变量，或PCA降维
```

```r
# R: VIF（car包）
library(car)
res_vif <- lm(Y ~ X + control1 + control2 + control3, data = df)
vif_vals <- vif(res_vif)
# 分类变量返回GVIF，比较 GVIF^(1/(2*Df)) vs sqrt(10) ≈ 3.16
print(vif_vals)
```

---

### 2b: 异方差检验（White / BP）

```python
# Python: White + Breusch-Pagan检验（返回dict）
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
import statsmodels.api as sm
import statsmodels.formula.api as smf

def heteroskedasticity_tests(df, formula):
    res = smf.ols(formula, df).fit()
    X_exog = sm.add_constant(res.model.exog)
    white_stat, white_p, _, _ = het_white(res.resid, X_exog)
    bp_stat,    bp_p,    _, _ = het_breuschpagan(res.resid, X_exog)
    return {
        'white_stat': round(white_stat, 4), 'white_p': round(white_p, 4),
        'bp_stat':    round(bp_stat, 4),    'bp_p':    round(bp_p, 4),
        'use_robust': white_p < 0.05 or bp_p < 0.05
    }

het_result = heteroskedasticity_tests(df, f"Y ~ X + {' + '.join(controls)}")
# 若 use_robust=True → 使用HC3稳健标准误
```

```r
# R: 异方差检验（lmtest）
library(lmtest)
library(sandwich)

res_lm <- lm(Y ~ X + control1 + control2, data = df)
bptest(res_lm)                                        # Breusch-Pagan
bptest(res_lm, ~ fitted(res_lm) + I(fitted(res_lm)^2)) # White近似

# 若显著，使用HC3稳健SE
coeftest(res_lm, vcov = vcovHC(res_lm, type = "HC3"))
```

---

### 2c: Ramsey RESET 函数形式检验

检验是否存在遗漏非线性项（平方项、交叉项）。

```r
# R: Ramsey RESET（lmtest::resettest）
library(lmtest)
resettest(res_lm, power = 2:3, type = "fitted")
# H0: 函数形式正确
# p < 0.05 → 考虑加入平方项或取对数
```

---

### 2d: 残差诊断图（QQ图 + 残差vs拟合）

```python
# Python: 残差诊断图
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

def plot_residual_diagnostics(df, formula, save_path='output/ols_residual_diagnostics.png'):
    res = smf.ols(formula, df).fit()
    fitted, resid = res.fittedvalues, res.resid
    std_resid = resid / resid.std()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # QQ图（正态性）
    stats.probplot(resid, dist="norm", plot=axes[0])
    axes[0].set_title('Normal Q-Q Plot')

    # 残差 vs 拟合值
    axes[1].scatter(fitted, resid, alpha=0.3, s=15)
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set(xlabel='Fitted values', ylabel='Residuals', title='Residuals vs Fitted')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return {'n': int(res.nobs), 'r2': round(res.rsquared, 4)}

plot_residual_diagnostics(df, f"Y ~ X + {' + '.join(controls)}")
```

```r
# R: 残差诊断（base R 4格图）
par(mfrow = c(2, 2))
plot(res_lm)   # 自动生成：Residuals vs Fitted, QQ, Scale-Location, Cook's D
```

---

### 2e: 异常值 Cook's D

```python
# Python: Cook's D异常值检测（返回dict）
def cooks_distance_check(res_ols, df):
    influence = res_ols.get_influence()
    cooks_d   = influence.cooks_distance[0]
    threshold = 4 / len(df)
    outlier_idx = df.index[cooks_d > threshold].tolist()
    return {
        'threshold': round(threshold, 5),
        'n_outliers': len(outlier_idx),
        'outlier_indices': outlier_idx,
        'pct_outliers': round(len(outlier_idx) / len(df) * 100, 2)
    }

import statsmodels.formula.api as smf
res_main = smf.ols(f"Y ~ X + {' + '.join(controls)}", df).fit(cov_type='HC3')
cooks_result = cooks_distance_check(res_main, df)
```

```r
# R: Cook's D
cooksd    <- cooks.distance(res_lm)
threshold <- 4 / nrow(df)
outliers  <- which(cooksd > threshold)
cat(sprintf("Cook's D > %.4f: %d个观测 (%.1f%%)\n",
            threshold, length(outliers), length(outliers)/nrow(df)*100))
```

---

## Step 3: 标准误选择

| 场景 | 推荐标准误 | Python | R |
|------|-----------|--------|---|
| 截面数据有异方差 | HC3 | `cov_type='HC3'` | `vcovHC(., "HC3")` |
| 面板/聚类数据 | 聚类SE | `cov_type='cluster'` | `cluster = ~group` |
| 聚类数 < 50 | Wild Bootstrap | `wildboottest` | `fwildclusterboot` |
| 不确定聚类层级 | 报告多种 | — | — |

**聚类层级选择规则：**
- 聚类到**处理分配层级**（处理在省级 → 聚类到省；处理在企业级 → 聚类到企业）
- 聚类数 < 50 → Wild Bootstrap（经典渐近推断不可靠）
- 不确定时，报告实体/时间/双向聚类多种结果，选最保守的

```python
# Python: 聚类标准误
import statsmodels.formula.api as smf

res_cluster = smf.ols(f"Y ~ X + {' + '.join(controls)}", df).fit(
    cov_type='cluster',
    cov_kwds={'groups': df['cluster_var']}
)
```

```r
# R: Wild Bootstrap（聚类数<50时）
library(fwildclusterboot)

res_fe <- feols(Y ~ X + control1 | industry, data = df)
boot_res <- boottest(
  res_fe,
  clustid = "province",   # 聚类变量
  param   = "X",          # 检验的系数
  B       = 9999
)
summary(boot_res)
```

---

## Step 4: Oster (2019) 系数稳定性

通过δ值衡量"遗漏变量需要多强才能推翻结论"。**δ > 1 → 结论稳健。**

使用 **robomit 包**（R）的 `o_delta()` 和 `o_beta()`，不手写公式。

R²_max 默认设为 1.3 × R²_controlled（Oster 2019建议）。

```r
# R: Oster (2019) — 使用 robomit 包
# install.packages("robomit")
library(robomit)
library(fixest)

res_restricted <- feols(Y ~ X,                          data = df, vcov = "hetero")
res_controlled <- feols(Y ~ X + control1 + control2 + control3, data = df, vcov = "hetero")

# o_delta: 计算δ（遗漏变量需要多强才能使系数=0）
delta_result <- o_delta(
  y        = "Y",
  x        = "X",
  con      = "control1 + control2 + control3",
  delta    = 1,              # 基准：与观测偏误等强
  R2max    = 1.3 * r2(res_controlled)[["r2"]],   # 1.3×R²_ctrl
  type     = "lm",
  data     = df
)
print(delta_result)

# o_beta: 在delta=1假设下的bias-adjusted系数
beta_star <- o_beta(
  y     = "Y",
  x     = "X",
  con   = "control1 + control2 + control3",
  delta = 1,
  R2max = 1.3 * r2(res_controlled)[["r2"]],
  type  = "lm",
  data  = df
)
print(beta_star)

# 多个R²_max假设下的敏感性
for (r2_mult in c(1.3, 1.5, 2.0)) {
  d <- o_delta(y="Y", x="X", con="control1 + control2",
               delta=1, R2max=r2_mult * r2(res_controlled)[["r2"]],
               type="lm", data=df)
  cat(sprintf("R²max = %.1f×R²_ctrl: δ = %.3f %s\n",
              r2_mult, d$delta, if(d$delta > 1) "✓ 稳健" else "⚠️ 需关注"))
}
```

---

## Step 5: Specification Curve

穷举所有合理控制变量组合，展示核心系数在不同规格下的分布。

```r
# R: Specification Curve（specr包，新API）
# install.packages("specr")
library(specr)
library(ggplot2)

# 定义规格空间（使用新API: setup() + run_specs()）
specs <- setup(
  data     = df,
  y        = c("Y"),
  x        = c("X"),
  model    = c("lm"),
  controls = c("control1", "control2", "control3", "control4")
)

# 运行所有规格
results <- run_specs(specs, .progress = TRUE)

# 统计摘要
cat("规格总数:", nrow(results), "\n")
cat("系数中位数:", median(results$estimate), "\n")
cat("方向一致性:", mean(results$estimate > 0), "\n")
# ✅ 正确写法：
cat("显著比例(p<0.05):", (sum(results$p.value < 0.05) / nrow(results)), "\n")
# ❌ 错误写法（勿用）：mean(df_spec['pval'] < 0.05)  ← Python语法错误

# Specification Curve图
p_curve <- plot(results)
ggsave("output/specification_curve.png", p_curve, width = 12, height = 8, dpi = 300)
```

```python
# Python: Specification Curve（手动循环）
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.formula.api as smf
from tqdm import tqdm

candidate_controls = ['control1', 'control2', 'control3', 'control4']
core_var = 'X'
outcome  = 'Y'

# 穷举所有控制变量组合
all_specs = []
for k in range(len(candidate_controls) + 1):
    for combo in itertools.combinations(candidate_controls, k):
        all_specs.append(list(combo))

records = []
for ctrl_list in tqdm(all_specs):
    parts = [core_var] + ctrl_list
    formula = f"{outcome} ~ {' + '.join(parts)}"
    try:
        res = smf.ols(formula, data=df).fit(cov_type='HC3')
        records.append({
            'coef': res.params[core_var],
            'ci_low':  res.conf_int().loc[core_var, 0],
            'ci_high': res.conf_int().loc[core_var, 1],
            'pval': res.pvalues[core_var],
            **{f'has_{c}': (c in ctrl_list) for c in candidate_controls}
        })
    except Exception:
        continue

df_spec = pd.DataFrame(records).sort_values('coef').reset_index(drop=True)

# ✅ 正确：使用括号避免Python运算符优先级问题
sig_pct = (df_spec['pval'] < 0.05).mean()
pos_pct = (df_spec['coef'] > 0).mean()

# 绘图
fig = plt.figure(figsize=(14, 9))
gs  = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.05)
ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

colors = ['#2196F3' if p < 0.05 else '#BBDEFB' for p in df_spec['pval']]
ax1.scatter(range(len(df_spec)), df_spec['coef'], c=colors, s=15, zorder=3)
ax1.fill_between(range(len(df_spec)), df_spec['ci_low'], df_spec['ci_high'], alpha=0.2)
ax1.axhline(0, color='red', linestyle='--', linewidth=1)
ax1.set_ylabel(f'{core_var} 系数')
ax1.set_title(f'Specification Curve（显著: {sig_pct:.0%} | 正向: {pos_pct:.0%}）')

for i, ctrl in enumerate(candidate_controls):
    y_pos = len(candidate_controls) - 1 - i
    for j, has_it in enumerate(df_spec[f'has_{ctrl}']):
        ax2.scatter(j, y_pos, color='#1976D2' if has_it else '#E0E0E0', s=8, marker='s')
ax2.set_yticks(range(len(candidate_controls)))
ax2.set_yticklabels(candidate_controls[::-1], fontsize=9)
ax2.set_xlabel('规格排序（按系数大小）')

plt.savefig('output/specification_curve.png', dpi=300, bbox_inches='tight')
plt.close()
```

---

## Step 6: 经济显著性三层解读

统计显著 ≠ 经济显著。必须同时报告效应大小，函数返回 dict（不print）。

```python
# Python: 经济显著性（返回dict）
def economic_significance(coef, x_std, y_mean, y_std,
                           variable_name="X", outcome_name="Y",
                           literature_range=None, policy_cost=None):
    """
    返回经济显著性三层解读dict。
    - 第一层：标准差单位效应（β×SD(X)/Mean(Y)）
    - 第二层：文献对比（若提供literature_range）
    - 第三层：政策换算（若提供policy_cost）
    """
    effect_1sd     = coef * x_std
    effect_pct     = effect_1sd / y_mean * 100
    effect_std_    = effect_1sd / y_std

    result = {
        'coef': coef,
        'effect_1sd': round(effect_1sd, 4),
        'effect_pct_of_mean': round(effect_pct, 2),
        'effect_std_units':   round(effect_std_, 3)
    }

    if literature_range:
        lo, hi = literature_range
        result['in_lit_range'] = lo <= effect_pct <= hi
        result['lit_range']    = literature_range

    if policy_cost is not None:
        result['policy_output_per_unit'] = round(coef * policy_cost, 4)

    return result

# 使用示例
econ = economic_significance(
    coef     = 0.045,
    x_std    = df['X'].std(),
    y_mean   = df['Y'].mean(),
    y_std    = df['Y'].std(),
    literature_range = (2, 8),   # 文献中2-8%效应
    policy_cost = 10000          # 每万元
)
```

```r
# R: 经济显著性（返回list）
economic_significance_r <- function(model, x_var, data, y_var,
                                     literature_range = NULL, policy_cost = NULL) {
  coef_val <- coef(model)[x_var]
  x_std    <- sd(data[[x_var]], na.rm = TRUE)
  y_mean   <- mean(data[[y_var]], na.rm = TRUE)
  y_std    <- sd(data[[y_var]], na.rm = TRUE)

  effect_1sd  <- coef_val * x_std
  effect_pct  <- effect_1sd / y_mean * 100
  effect_std_ <- effect_1sd / y_std

  result <- list(
    coef             = coef_val,
    effect_1sd       = effect_1sd,
    effect_pct       = effect_pct,
    effect_std_units = effect_std_,
    in_lit_range     = if (!is.null(literature_range))
                         effect_pct >= literature_range[1] & effect_pct <= literature_range[2]
                       else NA,
    policy_output    = if (!is.null(policy_cost)) coef_val * policy_cost else NA
  )
  return(result)
}
```

---

## 7. OLS Estimand 声明

| 设定 | Estimand | 声明要求 |
|------|----------|---------|
| OLS + 控制变量（效应同质） | ATE | 需论证效应为何同质（如RCT或强CIA） |
| OLS + 控制变量（效应异质） | 处理方差加权平均 ≠ ATE | 标注这**不是**ATE；如需ATE用IPW/Matching |
| OLS作为基准（描述性） | 条件相关系数 | 明确声明非因果，仅描述性 |

> **⚠️ 错误8（高频错误）：默认OLS系数=ATE**
>
> 在异质效应下，OLS系数是**处理方差加权平均**（treatment-variance weighted average）：
> \[\hat{\beta}_{OLS} = \frac{\sum_i (X_i - \bar{X})^2 \cdot \tau_i}{\sum_i (X_i - \bar{X})^2}\]
> 权重取决于处理强度方差，不等于真正的ATE。若需ATE，使用IPW或Matching。

**声明模板：**
> "本文OLS估计的核心系数为处理变量X的条件均值效应。在效应同质性假设下，该系数可解释为ATE。若效应存在异质性，OLS系数为处理方差加权平均，可能偏离ATE（Angrist & Pischke, 2009）。第X节提供IPW估计的ATE作为稳健性检验。"

---

## 检验清单

| 步骤 | 检验 | 通过标准 |
|------|------|---------|
| Step 1 | 逐步系数变化 | 加入控制后变化 < 30% |
| Step 2a | VIF | 所有 VIF < 10（理想 < 5） |
| Step 2b | White / BP异方差 | 若显著 → 使用HC3 |
| Step 2c | Ramsey RESET | p > 0.05（函数形式正确） |
| Step 2d | QQ图 + 残差vs拟合 | 无系统模式，近似正态 |
| Step 2e | Cook's D | 检查 D > 4/N 的观测 |
| Step 3 | 标准误类型 | 与数据结构匹配（聚类/HC3） |
| Step 4 | Oster δ | δ > 1（稳健） |
| Step 5 | Spec Curve | 多数规格方向一致 |
| Step 6 | 经济显著性 | 效应量有实质意义 |

---

## 常见错误

> **错误1：截距项遗漏** — `sm.OLS` 需手动 `add_constant()`；`smf.ols` 和 `lm()` 默认含截距。

> **错误2：哑变量陷阱** — 分类变量创建哑变量必须删一个基准类。`pd.get_dummies(..., drop_first=True)`；R `lm()` 自动处理。

> **错误3：比较不同样本的R²** — R²随控制变量增加必然上升。报告adjusted R²或within R²。

> **错误4：星号当效应大小** — p显著 ≠ 实际重要。大样本下微小效应也会显著。**必须同时报告系数+置信区间+经济显著性。**

> **错误5：标准误类型未声明** — 表格脚注必须明确注明SE类型（HC3/Clustered/Wild Bootstrap）。

> **错误6：非线性关系强行线性** — 提交前用散点图检查Y vs X；明显非线性则加平方项或取对数。

> **错误7：未报告样本量和缺失处理** — 表格下方必须注明N及缺失值处理方式。

> **错误8：默认OLS系数=ATE** — 见第7节Estimand声明。

---

## 输出规范

**表格必须包含：**
1. 逐步规格列（展示核心系数稳定性）
2. 括号内标准误（注明类型）
3. 显著性星号（注明水平）
4. 固定效应 Yes/No 标注
5. N、R²（或 within R²）

```r
# R: modelsummary LaTeX输出
library(modelsummary)
modelsummary(
  list("(1)" = res1, "(2)" = res2, "(3)" = res3, "(4)" = res4),
  stars   = c("*" = 0.1, "**" = 0.05, "***" = 0.01),
  vcov    = "robust",
  gof_map = c("nobs", "r.squared", "adj.r.squared"),
  output  = "output/ols_main_table.tex",
  title   = "OLS Regression Results"
)
```

**文件命名：**
```
output/
  ols_descriptive_stats.csv
  ols_vif_check.csv
  ols_residual_diagnostics.png
  ols_specification_curve.png
  ols_main_table.csv
  ols_main_table.tex
  ols_robustness.csv
```
