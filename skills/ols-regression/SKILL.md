---
name: ols-regression
description: >
  OLS baseline regression for empirical research: stepwise specification
  with coefficient stability checks, full diagnostics suite (VIF, heteroskedasticity,
  RESET, Cook's D, residual plots), standard error selection (HC3/cluster/wild bootstrap),
  Oster (2019) delta and sensemakr sensitivity analysis, specification curve,
  economic significance interpretation, LPM for binary outcomes, and estimand declaration.
  Use when running baseline regressions, checking coefficient robustness,
  diagnosing model assumptions, or preparing OLS result tables.
  NOT for panel FE/RE (use panel-data skill) or causal identification (use DID/IV/RDD skills).
---

# OLS 回归分析

## 概述

OLS 是实证研究的基础工具，通常作为基准模型（baseline）为 IV/DID/RDD 提供参照。
**语言**：Python + R（NO Stata）

**本 skill 负责**：截面 OLS、Pooled OLS（面板基准对照）、LPM、诊断全套、标准误选择、系数稳定性（Oster + sensemakr）、Specification Curve、经济显著性、Estimand 声明。

**本 skill 不负责**（转到其他 skill）：
- FE/RE 选择与估计 → `panel-data` skill
- GLM-QMLE（因变量含零/比率）→ `data-cleaning` Step 4.5a
- IV 2SLS → `iv-estimation` skill
- DID/RDD → 对应 skill
- Matching/IPW → `matching` skill

**架构**：sensemakr + Heckman 完整代码 → 见 ADVANCED.md

---

## Step 0: OLS 适用性判断

> 在开始回归前，先确认 OLS 是否适合当前场景。

```
数据结构？
├── 截面数据 → ✅ OLS 是默认起点
├── 面板数据
│   ├── 作为基准 Pooled OLS（与 FE 对照）→ ✅ 本 skill
│   ├── 需要 FE/RE → panel-data skill
│   └── F 检验 + Hausman → pipeline Stage 2 FE gate 决策
└── 重复截面 → Pooled OLS + 年份 FE

因变量类型？
├── 连续 → ✅ OLS
├── 二元（0/1）
│   ├── 目标是平均边际效应（大多数实证论文）？
│   │   └── ✅ LPM (OLS) — 系数直读为概率变化
│   │       优势: IV-2SLS 直接适用 (Angrist, 2001)
│   │              离散内生变量无需特殊处理 (Wooldridge, 2014)
│   │       劣势: 预测值可能 <0 或 >1；极端值处偏效应不准
│   └── 目标是预测 / 极端值偏效应？
│       └── Logit/Probit — 需报告边际效应而非原始系数
│           ⚠️ 非线性 IV 对简化形式设定错误敏感 (Ramalho & Ramalho, 2017)
├── 有界比率（如 ROA）
│   └── OLS 可用于估计平均边际效应 (Wooldridge, 2010)
│       → 稳健性用 fractional response model（见 data-cleaning ADVANCED.md）
├── 计数 / 含零 → GLM-QMLE（data-cleaning 4.5a）
└── 自然比例 0-1 → fractional logit（data-cleaning ADVANCED.md）
```

> **⚠️ OLS vs GLM 常见误区**：因变量有界时直接放弃 OLS 转 GLM。若研究重点是估计边际效应（而非预测），OLS 通常能很好地近似有界变量的效应 (Wooldridge, 2010)。非线性模型的 IV 估计量对简化形式的模型设定错误非常敏感，且难以处理离散型内生变量。**OLS/LPM 是稳健的默认起点。**

---

## Step 1: 逐步回归

逐步加入控制变量，展示核心系数稳定性。核心系数变化 > 30% 预示遗漏变量偏误。

> **⚠️ 仅供探索：** 逐步回归用于展示系数稳健性，不得作为因果推断的正式识别策略。
> 需要正式稳健性证据时，用 Step 5（Specification Curve）或 Oster δ。

```python
import statsmodels.formula.api as smf
import pandas as pd

def stepwise_ols(df, outcome, core_var, controls, fe_vars=None):
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
            'coef': res.params[core_var], 'se': res.bse[core_var],
            'pval': res.pvalues[core_var], 'N': int(res.nobs),
            'R2': round(res.rsquared, 4),
        }

    coef_table = pd.DataFrame(results).T
    if len(results) >= 2:
        b_first = list(results.values())[0]['coef']
        b_last  = list(results.values())[-1]['coef']
        change_pct = abs(b_last - b_first) / abs(b_first) * 100 if b_first != 0 else float('inf')
        if change_pct > 30:
            print(f"⚠️ 核心系数变化 {change_pct:.1f}%，遗漏变量偏误风险高")
    return coef_table
```

```r
library(fixest)

res1 <- feols(Y ~ X,                          data = df, vcov = "hetero")
res2 <- feols(Y ~ X + control1 + control2,    data = df, vcov = "hetero")
res3 <- feols(Y ~ X + control1 + control2 | industry,       data = df, cluster = ~industry)
res4 <- feols(Y ~ X + control1 + control2 | industry + year, data = df, cluster = ~industry)

etable(res1, res2, res3, res4,
       headers = c("Baseline", "+Controls", "+Ind FE", "+Ind+Year FE"),
       fitstat = ~ r2 + n)
```

---

## Step 2: 诊断检验

### 2a: VIF 多重共线性

VIF > 10：严重；VIF > 5：需关注。

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

def calc_vif(df, predictors):
    X = sm.add_constant(df[predictors].dropna())
    vif_df = pd.DataFrame({
        'variable': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    return vif_df[vif_df['variable'] != 'const'].sort_values('VIF', ascending=False)
```

```r
library(car)
vif_vals <- vif(lm(Y ~ X + control1 + control2, data = df))
# 分类变量返回 GVIF，比较 GVIF^(1/(2*Df)) vs sqrt(10) ≈ 3.16
```

### 2b: 异方差检验（White / BP）

```python
from statsmodels.stats.diagnostic import het_white, het_breuschpagan

def heteroskedasticity_tests(df, formula):
    res = smf.ols(formula, df).fit()
    X_exog = sm.add_constant(res.model.exog)
    white_stat, white_p, _, _ = het_white(res.resid, X_exog)
    bp_stat, bp_p, _, _ = het_breuschpagan(res.resid, X_exog)
    return {
        'white_p': round(white_p, 4), 'bp_p': round(bp_p, 4),
        'use_robust': white_p < 0.05 or bp_p < 0.05,
    }
# 若 use_robust=True → 使用 HC3 稳健标准误
```

```r
library(lmtest); library(sandwich)
bptest(res_lm)                                         # Breusch-Pagan
bptest(res_lm, ~ fitted(res_lm) + I(fitted(res_lm)^2)) # White 近似
# 若显著 → coeftest(res_lm, vcov = vcovHC(res_lm, type = "HC3"))
```

### 2c: Ramsey RESET

```r
library(lmtest)
resettest(res_lm, power = 2:3, type = "fitted")
# p < 0.05 → 考虑加入平方项或取对数
```

### 2d: 残差诊断图

```python
import matplotlib.pyplot as plt
from scipy import stats

def plot_residual_diagnostics(df, formula, save_path='output/ols_residual_diagnostics.png'):
    res = smf.ols(formula, df).fit()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    stats.probplot(res.resid, dist="norm", plot=axes[0])
    axes[0].set_title('Normal Q-Q Plot')
    axes[1].scatter(res.fittedvalues, res.resid, alpha=0.3, s=15)
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set(xlabel='Fitted values', ylabel='Residuals', title='Residuals vs Fitted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
```

```r
par(mfrow = c(2, 2))
plot(res_lm)  # Residuals vs Fitted, QQ, Scale-Location, Cook's D
```

### 2e: Cook's D 异常值

```python
def cooks_distance_check(res_ols, df):
    cooks_d = res_ols.get_influence().cooks_distance[0]
    threshold = 4 / len(df)
    outlier_idx = df.index[cooks_d > threshold].tolist()
    return {'threshold': round(threshold, 5), 'n_outliers': len(outlier_idx),
            'pct_outliers': round(len(outlier_idx) / len(df) * 100, 2)}
```

---

## Step 3: 标准误选择

| 场景 | 推荐 | Python | R |
|------|------|--------|---|
| 截面 + 异方差 | HC3 | `cov_type='HC3'` | `vcovHC(., "HC3")` |
| 面板 / 聚类 | 聚类 SE | `cov_type='cluster'` | `cluster = ~group` |
| 聚类数 < 50 | Wild Bootstrap | `wildboottest` | `fwildclusterboot` |

**聚类层级规则**：
- 聚类到**处理分配层级**（省级处理→聚类到省）
- 聚类数 < 50 → Wild Bootstrap
- 不确定时，报告多种聚类结果，选最保守的

```r
# Wild Bootstrap（聚类数 < 50）
library(fwildclusterboot)
res_fe <- feols(Y ~ X + control1 | industry, data = df)
boot_res <- boottest(res_fe, clustid = "province", param = "X", B = 9999)
```

---

## Step 4: Oster (2019) 系数稳定性

δ > 1 → 结论稳健。R²_max = 1.3 × R²_controlled（Oster 2019 建议）。

```r
library(robomit); library(fixest)

res_ctrl <- feols(Y ~ X + control1 + control2 + control3, data = df, vcov = "hetero")

# δ 值
delta_result <- o_delta(
  y = "Y", x = "X", con = "control1 + control2 + control3",
  delta = 1, R2max = 1.3 * r2(res_ctrl)[["r2"]], type = "lm", data = df
)

# bias-adjusted β*
beta_star <- o_beta(
  y = "Y", x = "X", con = "control1 + control2 + control3",
  delta = 1, R2max = 1.3 * r2(res_ctrl)[["r2"]], type = "lm", data = df
)

# 多 R²_max 敏感性
for (r2_mult in c(1.3, 1.5, 2.0)) {
  d <- o_delta(y="Y", x="X", con="control1 + control2",
               delta=1, R2max=r2_mult * r2(res_ctrl)[["r2"]], type="lm", data=df)
  cat(sprintf("R²max = %.1f×R²_ctrl: δ = %.3f %s\n",
              r2_mult, d$delta, if(d$delta > 1) "✓ 稳健" else "⚠️ 需关注"))
}
```

> sensemakr (Cinelli & Hazlett, 2020) 完整 R+Python 代码 → 见 ADVANCED.md

---

## Step 5: Specification Curve

穷举所有合理控制变量组合，展示核心系数在不同规格下的分布。

```r
library(specr); library(ggplot2)

specs <- setup(data = df, y = c("Y"), x = c("X"), model = c("lm"),
               controls = c("control1", "control2", "control3", "control4"))
results <- run_specs(specs, .progress = TRUE)

cat("规格总数:", nrow(results), "\n")
cat("系数中位数:", median(results$estimate), "\n")
cat("方向一致性:", mean(results$estimate > 0), "\n")
cat("显著比例(p<0.05):", (sum(results$p.value < 0.05) / nrow(results)), "\n")

p_curve <- plot(results)
ggsave("output/specification_curve.png", p_curve, width = 12, height = 8, dpi = 300)
```

```python
import itertools
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

candidate_controls = ['control1', 'control2', 'control3', 'control4']

all_specs = []
for k in range(len(candidate_controls) + 1):
    for combo in itertools.combinations(candidate_controls, k):
        all_specs.append(list(combo))

records = []
for ctrl_list in all_specs:
    parts = [core_var] + ctrl_list
    formula = f"{outcome} ~ {' + '.join(parts)}"
    try:
        res = smf.ols(formula, data=df).fit(cov_type='HC3')
        records.append({
            'coef': res.params[core_var],
            'ci_low': res.conf_int().loc[core_var, 0],
            'ci_high': res.conf_int().loc[core_var, 1],
            'pval': res.pvalues[core_var],
            **{f'has_{c}': (c in ctrl_list) for c in candidate_controls},
        })
    except Exception:
        continue

df_spec = pd.DataFrame(records).sort_values('coef').reset_index(drop=True)
sig_pct = (df_spec['pval'] < 0.05).mean()
pos_pct = (df_spec['coef'] > 0).mean()

# 绘图
fig = plt.figure(figsize=(14, 9))
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.05)
ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
colors = ['#2196F3' if p < 0.05 else '#BBDEFB' for p in df_spec['pval']]
ax1.scatter(range(len(df_spec)), df_spec['coef'], c=colors, s=15, zorder=3)
ax1.fill_between(range(len(df_spec)), df_spec['ci_low'], df_spec['ci_high'], alpha=0.2)
ax1.axhline(0, color='red', linestyle='--', linewidth=1)
ax1.set_ylabel(f'{core_var} coefficient')
ax1.set_title(f'Specification Curve (sig: {sig_pct:.0%} | positive: {pos_pct:.0%})')
for i, ctrl in enumerate(candidate_controls):
    y_pos = len(candidate_controls) - 1 - i
    for j, has_it in enumerate(df_spec[f'has_{ctrl}']):
        ax2.scatter(j, y_pos, color='#1976D2' if has_it else '#E0E0E0', s=8, marker='s')
ax2.set_yticks(range(len(candidate_controls)))
ax2.set_yticklabels(candidate_controls[::-1], fontsize=9)
plt.savefig('output/specification_curve.png', dpi=300, bbox_inches='tight')
plt.close()
```

---

## Step 6: 经济显著性三层解读

统计显著 ≠ 经济显著。**必须同时报告效应大小。**

```python
def economic_significance(coef, x_std, y_mean, y_std,
                           literature_range=None, policy_cost=None):
    effect_1sd = coef * x_std
    effect_pct = effect_1sd / y_mean * 100
    effect_std = effect_1sd / y_std
    result = {
        'effect_1sd': round(effect_1sd, 4),
        'effect_pct_of_mean': round(effect_pct, 2),
        'effect_std_units': round(effect_std, 3),
    }
    if literature_range:
        lo, hi = literature_range
        result['in_lit_range'] = lo <= effect_pct <= hi
    if policy_cost is not None:
        result['policy_output_per_unit'] = round(coef * policy_cost, 4)
    return result
```

```r
economic_significance_r <- function(model, x_var, data, y_var,
                                     literature_range = NULL) {
  coef_val <- coef(model)[x_var]
  x_std <- sd(data[[x_var]], na.rm = TRUE)
  y_mean <- mean(data[[y_var]], na.rm = TRUE)
  effect_1sd <- coef_val * x_std
  effect_pct <- effect_1sd / y_mean * 100
  list(coef = coef_val, effect_1sd = effect_1sd, effect_pct = effect_pct,
       in_lit_range = if (!is.null(literature_range))
         effect_pct >= literature_range[1] & effect_pct <= literature_range[2] else NA)
}
```

---

## Step 7: Estimand 声明

| 设定 | Estimand | 声明要求 |
|------|----------|---------|
| OLS + 控制（效应同质） | ATE | 需论证效应同质（如 RCT 或强 CIA） |
| OLS + 控制（效应异质） | 处理方差加权平均 ≠ ATE | 标注非 ATE；如需 ATE 用 IPW/Matching |
| OLS 作为基准（描述性） | 条件相关系数 | 明确声明非因果 |

> **⚠️ 高频错误：默认 OLS 系数 = ATE。** 在异质效应下，OLS 系数是处理方差加权平均，权重取决于处理强度方差，不等于真正的 ATE。若需 ATE → IPW 或 Matching。

**声明模板**：
> "本文 OLS 估计的核心系数为处理变量 X 的条件均值效应。在效应同质性假设下，该系数可解释为 ATE。若效应存在异质性，OLS 系数为处理方差加权平均，可能偏离 ATE（Angrist & Pischke, 2009）。"

---

## 检验清单

**诊断**
- [ ] VIF：所有 < 10（理想 < 5）
- [ ] White / BP 异方差：若显著 → HC3
- [ ] Ramsey RESET：p > 0.05（函数形式正确）
- [ ] QQ 图 + 残差 vs 拟合：无系统模式
- [ ] Cook's D：检查 D > 4/N 的观测

**稳健性**
- [ ] 逐步回归系数变化 < 30%
- [ ] 标准误类型与数据结构匹配
- [ ] Oster δ > 1
- [ ] Specification Curve 多数规格方向一致
- [ ] 经济显著性有实质意义

**报告**
- [ ] Estimand 已声明
- [ ] 表格含 SE 类型脚注、N、R²
- [ ] LPM 已说明预测值可能越界

## 常见错误

1. **`sm.OLS` 无截距**：需手动 `add_constant()`；`smf.ols` 和 `lm()` 默认含截距。
2. **哑变量陷阱**：`pd.get_dummies(..., drop_first=True)`；R `lm()` 自动处理。
3. **比较不同样本的 R²**：报告 adjusted R² 或 within R²。
4. **星号当效应大小**：大样本微小效应也会显著。**必须报告系数 + CI + 经济显著性。**
5. **SE 类型未声明**：表格脚注必须注明 HC3 / Clustered / Wild Bootstrap。
6. **非线性关系强行线性**：散点图检查 Y vs X；非线性则加平方项或取对数。
7. **默认 OLS 系数 = ATE**：见 Step 7。
8. **因变量有界就放弃 OLS**：若目标是边际效应，LPM 通常够用 (Wooldridge, 2010)。

## 输出规范

**表格必须包含**：逐步规格列、括号内 SE（注明类型）、显著性星号（注明水平）、FE Yes/No、N、R²。

```r
library(modelsummary)
modelsummary(
  list("(1)" = res1, "(2)" = res2, "(3)" = res3, "(4)" = res4),
  stars = c("*" = 0.1, "**" = 0.05, "***" = 0.01),
  vcov = "robust", gof_map = c("nobs", "r.squared", "adj.r.squared"),
  output = "output/ols_main_table.tex", title = "OLS Regression Results"
)
```

```
output/
  ols_descriptive_stats.csv
  ols_vif_check.csv
  ols_residual_diagnostics.png
  ols_specification_curve.png
  ols_main_table.csv / .tex
  ols_robustness.csv
```
