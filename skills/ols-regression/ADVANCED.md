# OLS 回归 — 高级内容

本文件按需加载（ADVANCED.md）。包含：
- sensemakr 完整实现（R + Python）
- Heckman 两阶段完整代码
- LPM 诊断与修正
- pyfixest 多规格批量回归
- 手动 Oster delta Python 实现

---

## sensemakr: 遗漏变量敏感性分析 (Cinelli & Hazlett, 2020)

> 量化"遗漏变量需要多强才能推翻结论"。与 Oster delta 互补：Oster 用 R-squared 变化度量，sensemakr 用 partial R-squared 基准化。

### R（推荐，原生包）

```r
# install.packages("sensemakr")
library(sensemakr)

# 1. 拟合基准模型
model <- lm(Y ~ X + control1 + control2 + control3, data = df)

# 2. 运行 sensemakr
sense <- sensemakr(
  model    = model,
  treatment = "X",
  benchmark_covariates = c("control1", "control2"),  # 基准协变量
  kd = 1:3,        # 遗漏变量强度倍数（1x, 2x, 3x 基准协变量）
  ky = 1:3,
  q  = 1,          # 效应缩减比例（1 = 推翻到 0）
  alpha = 0.05
)

# 3. 核心输出
summary(sense)
# RV_q=1   : 能使系数=0 的遗漏变量 partial R-sq 阈值
# RV_q=1,a : 能使 CI 包含 0 的阈值（更保守）
# 若 RV > 基准协变量的 partial R-sq --> 结论稳健

# 4. 二维等高线图（核心可视化）
png("output/sensemakr_contour.png", width = 800, height = 600, res = 150)
plot(sense, type = "contour")  # X轴: R2_Y~Z|X,W  Y轴: R2_D~Z|W
dev.off()

# 5. 极端情景图
png("output/sensemakr_extreme.png", width = 800, height = 600, res = 150)
plot(sense, type = "extreme")  # 最坏情景下系数变化
dev.off()

# 6. 报告模板
cat(sprintf(
  "Robustness Value (RV_q=1): %.3f\n  遗漏变量需解释 %.1f%% 的处理和结果残差方差才能使系数归零。\n  最强基准协变量 (%s) 仅解释 %.1f%%。\n  结论: %s\n",
  sense$sensitivity_stats$rv_q,
  sense$sensitivity_stats$rv_q * 100,
  sense$info$benchmark_covariates[1],
  max(sense$bounds$r2dz.x) * 100,
  if (sense$sensitivity_stats$rv_q > max(sense$bounds$r2dz.x)) "OK robust" else "WARN needs attention"
))
```

### Python（PySensemakr）

```python
# pip install PySensemakr
import statsmodels.formula.api as smf

try:
    from PySensemakr import sensemakr as pysense

    model = smf.ols("Y ~ X + control1 + control2 + control3", data=df).fit()

    sense = pysense.Sensemakr(
        model=model,
        treatment="X",
        benchmark_covariates=["control1", "control2"],
        kd=[1, 2, 3],
        ky=[1, 2, 3],
        q=1.0,
        alpha=0.05,
    )

    # 核心统计量
    sense.summary()

    # 等高线图
    sense.plot(plot_type="contour")

    # 极端情景图
    sense.plot(plot_type="extreme")

except ImportError:
    print("PySensemakr not available, use R sensemakr instead")
    print("   or compute partial R-squared bounds manually (see below)")

# 手动 partial R-squared 计算（不依赖包）
def partial_r2(model_full, model_restricted):
    ssr_r = model_restricted.ssr
    ssr_f = model_full.ssr
    return (ssr_r - ssr_f) / ssr_r

# 基准协变量的 partial R-squared
model_without_c1 = smf.ols("Y ~ X + control2 + control3", data=df).fit()
model_full = smf.ols("Y ~ X + control1 + control2 + control3", data=df).fit()
pr2_c1 = partial_r2(model_full, model_without_c1)
print(f"control1 partial R-sq = {pr2_c1:.4f}")
print(f"Omitted variable must exceed {pr2_c1:.4f} to be as strong as control1")
```

---

## Heckman 两阶段选择模型

> 处理样本选择偏误（MNAR 缺失）。需至少一个排他性变量（选择方程显著、结果方程不显著）。

### R（推荐，标准误自动纠正）

```r
library(sampleSelection)

# 排他性变量: z（影响是否被观测，但不直接影响 Y）
# 例: 分析师覆盖（影响信息披露，但不直接影响盈利能力）

# 两阶段 Heckman
heck <- selection(
  selection = observed ~ x1 + x2 + z,      # 第一阶段: Probit
  outcome   = Y ~ x1 + x2,                  # 第二阶段: OLS
  data      = df,
  method    = "2step"                        # 或 "ml"（极大似然）
)
summary(heck)
# 关注:
# 1. inverse Mills ratio (lambda) 是否显著 -> 显著说明存在选择偏误
# 2. 排他性变量 z 在选择方程中是否显著
# 3. 结果方程系数与普通 OLS 的差异

# 诊断
cat("Lambda (IMR) p-value:", summary(heck)$estimate["(Intercept):lambda", "Pr(>|z|)"], "\n")
cat("If lambda is insignificant -> selection bias is not severe, OLS results are reliable\n")
```

### Python（statsmodels 手动两阶段）

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
import numpy as np

def heckman_2step(df, selection_formula, outcome_formula, exclusion_var):
    # Stage 1: Probit
    probit = smf.probit(selection_formula, data=df).fit(disp=0)
    print("=== Stage 1: Probit (Selection) ===")
    print(probit.summary().tables[1])

    # 排他性变量检查
    if exclusion_var in probit.params.index:
        z_stat = abs(probit.tvalues[exclusion_var])
        print(f"\nExclusion var '{exclusion_var}': z={z_stat:.3f}, "
              f"p={probit.pvalues[exclusion_var]:.4f}")
        if probit.pvalues[exclusion_var] > 0.05:
            print("WARN: exclusion var insignificant in selection eq -> weak identification")

    # Inverse Mills Ratio
    xb = probit.predict(df, which="linear")
    endog_name = probit.model.endog_names
    df_sel = df[df[endog_name] == 1].copy()
    xb_sel = xb[df[endog_name] == 1]
    df_sel["imr"] = norm.pdf(xb_sel) / norm.cdf(xb_sel)

    # Stage 2: OLS + IMR
    outcome_with_imr = outcome_formula + " + imr"
    stage2 = smf.ols(outcome_with_imr, data=df_sel).fit(cov_type="HC3")
    print("\n=== Stage 2: OLS + Inverse Mills Ratio ===")
    print(stage2.summary().tables[1])

    # Lambda significance
    lambda_p = stage2.pvalues.get("imr", 1.0)
    print(f"\nIMR (lambda) p-value: {lambda_p:.4f}")
    if lambda_p > 0.05:
        print("-> Selection bias not severe, plain OLS results reliable")
    else:
        print("-> Significant selection bias, use Heckman-corrected coefficients")

    # NOTE: manual 2-step SE is not correct (generated regressor issue)
    # For formal reporting, use R sampleSelection::selection()
    print("\nNOTE: manual 2-step SE not corrected for generated regressor.")
    print("   Use R sampleSelection::selection() for formal reporting.")

    return stage2
```

---

## LPM 诊断与修正

```python
def lpm_diagnostics(df, formula):
    res = smf.ols(formula, data=df).fit(cov_type="HC3")  # LPM must use HC3
    fitted = res.fittedvalues

    out_of_range = ((fitted < 0) | (fitted > 1)).sum()
    pct_oor = out_of_range / len(fitted) * 100

    print(f"=== LPM Diagnostics ===")
    print(f"Out-of-range predictions: {out_of_range} ({pct_oor:.1f}%)")
    if pct_oor > 5:
        print("WARN: >5% out of range, consider Logit/Probit as robustness check")
    else:
        print("OK: out-of-range ratio acceptable, LPM results reliable")

    print(f"\nCoefficients as probability changes:")
    for var in res.params.index:
        if var == "Intercept":
            continue
        print(f"  {var}: {res.params[var]:+.4f} "
              f"(+1 unit -> P(Y=1) changes by {res.params[var]*100:+.2f} pp)")

    return res
```

```r
# R: LPM diagnostics
lpm_fit <- feols(Y_binary ~ X + control1 + control2, data = df, vcov = "hetero")

# Out-of-range check
fitted_vals <- fitted(lpm_fit)
oor <- sum(fitted_vals < 0 | fitted_vals > 1)
cat(sprintf("Out-of-range: %d (%.1f%%)\n", oor, oor/length(fitted_vals)*100))

# LPM vs Logit comparison
logit_fit <- glm(Y_binary ~ X + control1 + control2, family = binomial, data = df)
logit_ame <- margins::margins(logit_fit)
cat("LPM  coef:", coef(lpm_fit)["X"], "\n")
cat("Logit AME:", summary(logit_ame)[summary(logit_ame)$factor == "X", "AME"], "\n")
cat("Small difference -> LPM reliable\n")
```

---

## pyfixest 多规格批量回归

```python
import pyfixest as pf

# Multi-DV x stepwise FE in one line
fit = pf.feols(
    "Y + Y2 ~ X + control1 + control2 | csw0(industry, year)",
    data=df, vcov={"CRV1": "industry"}
)
pf.etable(fit)

# Wild Bootstrap (cluster count < 50)
fit_single = pf.feols("Y ~ X + control1 | industry", data=df)
boot = fit_single.wildboottest(param="X", B=9999, cluster="province")
print(f"Wild Bootstrap p-value: {boot['Pr(>|t|)']:.4f}")

# Romano-Wolf multiple testing correction
from pyfixest.multcomp import rwolf
rwolf_result = rwolf(
    "Y + Y2 + Y3 ~ X + control1 | industry",
    param="X", B=9999, data=df
)
```

---

## 手动 Oster delta Python 实现

```python
import numpy as np
import statsmodels.formula.api as smf

def oster_delta(df, outcome, core_var, controls, r2_max_mult=1.3):
    # Short regression (restricted)
    res_r = smf.ols(f"{outcome} ~ {core_var}", data=df).fit()
    beta_r = res_r.params[core_var]
    r2_r = res_r.rsquared

    # Long regression (controlled)
    ctrl_str = " + ".join(controls)
    res_c = smf.ols(f"{outcome} ~ {core_var} + {ctrl_str}", data=df).fit()
    beta_c = res_c.params[core_var]
    r2_c = res_c.rsquared

    # R-squared max
    r2_max = r2_max_mult * r2_c

    # delta calculation (Oster 2019)
    if abs(beta_r - beta_c) < 1e-12 or abs(r2_c - r2_r) < 1e-12:
        delta = float('inf')
    else:
        delta = ((beta_c - 0) * (r2_max - r2_c)) / ((beta_r - beta_c) * (r2_c - r2_r))

    # bias-adjusted beta*
    if abs(r2_max - r2_c) > 1e-12:
        beta_star = beta_c - delta * (beta_r - beta_c) * (r2_max - r2_c) / (r2_c - r2_r)
    else:
        beta_star = beta_c

    print(f"=== Oster (2019) Sensitivity ===")
    print(f"beta_restricted = {beta_r:.4f} (R-sq = {r2_r:.4f})")
    print(f"beta_controlled = {beta_c:.4f} (R-sq = {r2_c:.4f})")
    print(f"R-sq_max = {r2_max:.4f} ({r2_max_mult}x R-sq_controlled)")
    robust_str = "OK robust" if abs(delta) > 1 else "WARN needs attention"
    print(f"delta = {delta:.3f} {robust_str}")
    print(f"beta* (bias-adjusted, delta=1) = {beta_star:.4f}")

    return {
        'delta': delta, 'beta_star': beta_star,
        'beta_r': beta_r, 'beta_c': beta_c,
        'r2_r': r2_r, 'r2_c': r2_c, 'r2_max': r2_max,
    }

# Multi R-sq_max sensitivity
for mult in [1.3, 1.5, 2.0]:
    result = oster_delta(df, "Y", "X", controls, r2_max_mult=mult)
```

---

## 描述性统计模板

```python
def describe_vars(df, all_vars):
    desc = df[all_vars].describe().T
    desc['missing_pct'] = df[all_vars].isnull().mean() * 100
    desc['skewness'] = df[all_vars].skew()
    return desc.round(3)
```

```r
library(modelsummary)
datasummary_skim(df[, c(outcome, core_var, controls)])
```
