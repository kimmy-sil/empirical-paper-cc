# 时间序列分析 (Time Series)

## 概述

本 skill 提供宏观经济学与金融实证研究中标准时间序列分析流程，涵盖单位根检验、协整检验、VAR/VECM 模型、格兰杰因果、ARDL 模型、脉冲响应函数及结构突变检验，支持 Python / R / Stata 三种工具。

**适用场景**：
- 宏观经济变量的时间序列性质检验（平稳性）
- 多变量长期均衡关系（协整）分析
- 政策冲击的动态传导路径（IRF）
- 货币政策、财政政策乘数估计
- 经济预测与因果推断

---

## 前置条件

| 工具 | 依赖库 |
|------|--------|
| Python | `statsmodels >= 0.14`, `pandas`, `numpy`, `matplotlib` |
| R | `urca`, `tseries`, `vars`, `tsDyn`, `strucchange`, `mFilter` |
| Stata | `dfuller`, `pperron`, `varsoc`, `var`, `vec`, `irf` 内置；`egranger`（ssc install）|

数据要求：
- 时间序列已按时间顺序排列
- 无重复时间点
- 缺失值已处理（时间序列插值或删除）
- 理解数据频率（日/月/季/年）

---

## 分析步骤

### 步骤 1：单位根检验（Stationarity Tests）

在进行时间序列分析之前，必须检验各变量的积分阶数 I(d)。

#### ADF 检验（Augmented Dickey-Fuller）

原假设：序列存在单位根（非平稳）

**Python**
```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
import matplotlib.pyplot as plt

def adf_test(series, name="", max_lag=None, regression="ct"):
    """
    ADF 检验
    regression: 'c' = 仅截距, 'ct' = 截距+趋势, 'n' = 无截距无趋势
    """
    result = adfuller(series.dropna(), maxlag=max_lag,
                      regression=regression, autolag="AIC")
    t_stat, p_val, used_lags, nobs, crit_vals = result[:5]

    print(f"\n{'='*50}")
    print(f"ADF Test: {name}")
    print(f"  Test Statistic : {t_stat:.4f}")
    print(f"  p-value        : {p_val:.4f}")
    print(f"  Used Lags      : {used_lags}")
    print(f"  Critical Values: 1%: {crit_vals['1%']:.4f}, "
          f"5%: {crit_vals['5%']:.4f}, 10%: {crit_vals['10%']:.4f}")
    conclusion = "Non-stationary (unit root)" if p_val > 0.05 else "Stationary"
    print(f"  Conclusion     : {conclusion}")
    return {"stat": t_stat, "p": p_val, "lags": used_lags,
            "crit": crit_vals, "stationary": p_val <= 0.05}

# 对水平值和一阶差分都进行检验
for col in ["gdp", "m2", "cpi"]:
    adf_test(df[col], name=f"{col} (level)")
    adf_test(df[col].diff().dropna(), name=f"D.{col} (first diff)")
```

#### PP 检验（Phillips-Perron）

```python
from statsmodels.tsa.stattools import adfuller

# PP 检验（statsmodels 通过 arch 库或自实现）
# 推荐使用 R 的 PP.test 或 Stata 的 pperron

# Python 替代：使用 arch 包
# pip install arch
from arch.unitroot import PhillipsPerron

pp = PhillipsPerron(df["gdp"].dropna(), trend="ct", lags=4)
print(pp.summary())
```

#### KPSS 检验（Kwiatkowski-Phillips-Schmidt-Shin）

原假设：序列平稳（与 ADF 相反）

```python
from statsmodels.tsa.stattools import kpss

def kpss_test(series, name="", regression="ct"):
    """
    regression: 'c' = 截距平稳, 'ct' = 趋势平稳
    """
    stat, p_val, lags, crit_vals = kpss(series.dropna(),
                                         regression=regression, nlags="auto")
    print(f"\nKPSS Test: {name}")
    print(f"  Test Statistic : {stat:.4f}")
    print(f"  p-value        : {p_val:.4f}")
    print(f"  Critical Values: 1%: {crit_vals['1%']}, 5%: {crit_vals['5%']}")
    conclusion = "Stationary" if p_val > 0.05 else "Non-stationary (unit root)"
    print(f"  Conclusion     : {conclusion}")

kpss_test(df["gdp"], name="GDP (level)")
```

**R**
```r
library(urca)
library(tseries)

# ADF 检验（urca 包，更灵活）
adf_result <- ur.df(df$gdp, type = "trend", lags = 4, selectlags = "AIC")
summary(adf_result)

# Phillips-Perron
pp.test(df$gdp)

# KPSS
kpss.test(df$gdp, null = "Trend")

# 批量检验函数
unit_root_table <- function(series_list, names) {
  results <- data.frame()
  for (i in seq_along(series_list)) {
    x <- na.omit(series_list[[i]])
    adf <- ur.df(x, type = "trend", selectlags = "AIC")@teststat[1]
    pp  <- pp.test(x)$statistic
    kpss_s <- kpss.test(x)$statistic
    results <- rbind(results, data.frame(
      Variable = names[i],
      ADF = round(adf, 3),
      PP  = round(pp, 3),
      KPSS = round(kpss_s, 3)
    ))
  }
  return(results)
}
```

**Stata**
```stata
* ADF 检验
dfuller gdp, trend lags(4) regress
dfuller d.gdp, trend lags(3) regress    // 一阶差分

* PP 检验
pperron gdp, trend lags(4)

* KPSS 检验（需安装）
* ssc install kpss
kpss gdp, trend

* 批量检验
foreach v in gdp m2 cpi investment {
    dfuller `v', trend lags(4)
    dfuller d.`v', trend lags(3)
}
```

**检验结论矩阵**（通常整理为表格展示）：

| 变量 | ADF (水平) | PP (水平) | KPSS (水平) | ADF (一差) | 积分阶 |
|------|-----------|----------|------------|-----------|-------|
| GDP | -1.23 | -1.45 | 0.89** | -5.67*** | I(1) |
| M2  | -0.87 | -1.02 | 1.12** | -4.89*** | I(1) |

---

### 步骤 2：协整检验

若多个 I(1) 序列之间存在协整关系，则长期均衡成立，可建立 VECM。

#### Engle-Granger 两步法（双变量协整）

```python
from statsmodels.tsa.stattools import coint

# EG 协整检验
score, pvalue, crit_values = coint(df["gdp"], df["m2"])
print(f"EG 协整检验: stat = {score:.4f}, p = {pvalue:.4f}")
print(f"临界值: 1%: {crit_values[0]:.4f}, 5%: {crit_values[1]:.4f}")
if pvalue < 0.05:
    print("结论：存在协整关系（长期均衡）")
```

#### Johansen 检验（多变量协整）

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_test(df_vars, det_order=0, k_ar_diff=1):
    """
    det_order: -1=无截距, 0=截距不在协整方程, 1=截距在协整方程
    k_ar_diff: VAR 滞后阶数（差分形式，VAR(p) 对应 k_ar_diff=p-1）
    """
    result = coint_johansen(df_vars.dropna(), det_order=det_order, k_ar_diff=k_ar_diff)

    print("\nJohansen 协整检验")
    print("-" * 60)
    print("迹检验 (Trace Test):")
    for i in range(len(result.lr1)):
        print(f"  H0: r <= {i} | 迹统计量 = {result.lr1[i]:.4f} | "
              f"5% 临界值 = {result.cvt[i, 1]:.4f} | "
              f"{'拒绝 H0*' if result.lr1[i] > result.cvt[i,1] else '不拒绝'}")

    print("\n最大特征值检验 (Max-Eigen Test):")
    for i in range(len(result.lr2)):
        print(f"  H0: r = {i} | Max-Eigen = {result.lr2[i]:.4f} | "
              f"5% 临界值 = {result.cvm[i, 1]:.4f} | "
              f"{'拒绝 H0*' if result.lr2[i] > result.cvm[i,1] else '不拒绝'}")
    return result

vars_data = df[["gdp", "m2", "cpi"]].dropna()
johansen_result = johansen_test(vars_data)
```

**R (urca)**
```r
library(urca)

vars_matrix <- cbind(df$gdp, df$m2, df$cpi)
jo <- ca.jo(vars_matrix, type = "trace", K = 2,
             ecdet = "const", spec = "longrun")
summary(jo)
```

**Stata**
```stata
* Johansen 协整检验
vecrank gdp m2 cpi, trend(constant) lags(2)
* 输出：迹统计量和最大特征值统计量，及不同协整秩下的选择
```

---

### 步骤 3：VAR 模型（向量自回归）

适用于变量均为 I(0) 或差分后平稳，无协整关系的情况。

```python
from statsmodels.tsa.vector_ar.var_model import VAR

# 选择滞后阶数
model_order = VAR(df_stationary[["d_gdp", "d_m2", "d_cpi"]])
lag_order = model_order.select_order(maxlags=8)
print(lag_order.summary())   # AIC、BIC、HQIC 选择最优滞后

# 估计 VAR(p)
p_opt = lag_order.aic        # 使用 AIC 准则
var_model = model_order.fit(p_opt, trend="c")
print(var_model.summary())

# 残差诊断
var_model.test_normality()   # 残差正态性
var_model.test_whiteness(nlags=10)  # Portmanteau 白噪声检验
```

**R (vars)**
```r
library(vars)

# 选择滞后阶数
lag_select <- VARselect(df_stationary, lag.max = 8, type = "const")
print(lag_select$selection)  # AIC、HQ、SC、FPE 最优滞后

# 估计 VAR
var_model <- VAR(df_stationary, p = 2, type = "const")
summary(var_model)

# 诊断检验
serial.test(var_model, lags.pt = 12, type = "PT.asymptotic")  # 序列相关
arch.test(var_model, lags.multi = 5)                           # ARCH 效应
normality.test(var_model)                                      # 正态性
```

**Stata**
```stata
* VAR 滞后阶数选择
varsoc d_gdp d_m2 d_cpi, maxlag(8)

* 估计 VAR(2)
var d_gdp d_m2 d_cpi, lags(1/2)

* 稳定性检验（特征根在单位圆内）
varstable, graph
```

---

### 步骤 4：VECM 模型（存在协整时）

```python
from statsmodels.tsa.vector_ar.vecm import VECM

# 估计 VECM（k_ar_diff = VAR 滞后 - 1）
vecm_model = VECM(df[["gdp", "m2", "cpi"]].dropna(),
                  k_ar_diff=1,      # 对应 VAR(2)
                  coint_rank=1,     # 协整向量数（来自 Johansen 检验）
                  deterministic="ci")  # 协整方程含截距
vecm_fit = vecm_model.fit()
print(vecm_fit.summary())
```

**R (tsDyn)**
```r
library(tsDyn)

vecm_model <- VECM(df_vars, lag = 1, r = 1, estim = "ML",
                    include = "const")
summary(vecm_model)
```

**Stata**
```stata
* 估计 VECM（协整秩 = 1）
vec gdp m2 cpi, trend(constant) rank(1) lags(2)
```

---

### 步骤 5：格兰杰因果检验

```python
from statsmodels.tsa.stattools import grangercausalitytests

def granger_test_matrix(df_vars, maxlag=4):
    """生成格兰杰因果检验矩阵（F 检验）"""
    cols = df_vars.columns.tolist()
    n = len(cols)
    results = pd.DataFrame(index=cols, columns=cols,
                            dtype=object).fillna("")

    for i, y in enumerate(cols):
        for j, x in enumerate(cols):
            if i == j:
                results.loc[y, x] = "—"
                continue
            test_data = df_vars[[y, x]].dropna()
            gc = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
            # 取最小 p 值
            p_vals = [gc[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag+1)]
            min_p = min(p_vals)
            sig = "***" if min_p < 0.01 else ("**" if min_p < 0.05
                  else ("*" if min_p < 0.10 else ""))
            results.loc[y, x] = f"{min_p:.3f}{sig}"

    return results

gc_matrix = granger_test_matrix(df_stationary[["d_gdp","d_m2","d_cpi"]], maxlag=4)
print("格兰杰因果检验矩阵（p 值，行被列格兰杰导致）:")
print(gc_matrix)
```

**R (vars)**
```r
library(vars)

# 对 VAR 模型做格兰杰因果检验
causality(var_model, cause = "d_m2")$Granger
# H0: d_m2 不格兰杰导致其他变量
```

**Stata**
```stata
* VAR 后格兰杰因果检验
vargranger
```

---

### 步骤 6：脉冲响应函数（IRF）

```python
# Cholesky 分解正交化 IRF
irf_result = var_model.irf(periods=12)  # 12 期 IRF

# 绘制 IRF
fig = irf_result.plot(impulse="d_m2", response="d_gdp",
                       orth=True,    # 正交化
                       cumsum=False)
plt.suptitle("IRF: Response of GDP to M2 Shock", fontsize=14)
plt.savefig("output/figures/irf_m2_to_gdp.pdf", dpi=300, bbox_inches="tight")

# 方差分解（FEVD）
fevd = var_model.fevd(10)
fevd.plot()
plt.savefig("output/figures/fevd.pdf", dpi=300, bbox_inches="tight")
```

**R (vars)**
```r
library(vars)

# IRF（正交化，Cholesky）
irf_result <- irf(var_model,
                  impulse = "d_m2", response = "d_gdp",
                  n.ahead = 12, ortho = TRUE,
                  boot = TRUE, ci = 0.95, runs = 1000)
plot(irf_result)

# 方差分解
fevd_result <- fevd(var_model, n.ahead = 10)
plot(fevd_result)
```

**Stata**
```stata
* 估计 IRF（先建立 IRF 文件）
irf create myirf, step(12) set(myirf) replace

* 绘制 IRF
irf graph oirf, impulse(d_m2) response(d_gdp)    ///
    yline(0) xlabel(0(1)12)                        ///
    title("Response of GDP to M2 Shock")
graph export "output/figures/irf_m2_gdp.pdf", replace

* 方差分解
irf table fevd
```

---

### 步骤 7：ARDL 模型（Autoregressive Distributed Lag）

适用于变量积分阶混合（I(0) 和 I(1) 混合）时的边界检验（Bounds Test）。

```python
# Python: ardl 包
# pip install ardl
# 或使用 statsmodels ARDL（>=0.14）
from statsmodels.tsa.ardl import ARDL, ardl_select_order, bounds_test

# 自动选择滞后阶数
res_order = ardl_select_order(
    df["gdp"].dropna(),
    df[["m2", "cpi"]].dropna(),
    maxlag=4, maxorder=4,
    ic="aic"
)
print(res_order.model.order)

# 估计 ARDL
ardl_model = ARDL(
    df["gdp"].dropna(),
    lags=2,                        # 被解释变量滞后阶
    exog=df[["m2","cpi"]].dropna(),
    order={"m2": 2, "cpi": 1},    # 解释变量分布滞后阶
    trend="c"
)
ardl_fit = ardl_model.fit()
print(ardl_fit.summary())

# Bounds 检验（协整边界检验）
bounds = bounds_test(ardl_fit, case=3, alpha=0.05)
print(bounds)
```

**R (ARDL 包)**
```r
library(ARDL)
library(dplyr)

# 自动选择最优 ARDL
auto_ardl <- auto_ardl(gdp ~ m2 + cpi, data = df,
                        max_order = 4, selection = "AIC")
print(auto_ardl$best_order)

# 估计最优 ARDL
ardl_model <- ardl(gdp ~ m2 + cpi, data = df,
                   order = auto_ardl$best_order)
summary(ardl_model)

# Bounds 检验（PSS 2001）
bounds_result <- bounds_f_test(ardl_model, case = 3)
print(bounds_result)
```

**Stata**
```stata
* ssc install ardl
ardl gdp m2 cpi, aic maxlag(4)       // 自动选择 + AIC
* 边界检验
estat btest
```

---

### 步骤 8：结构突变检验（Bai-Perron）

```python
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.tsa.stattools import breakvar_heteroskedasticity_test

# CUSUM 检验（参数稳定性）
from statsmodels.regression.recursive_ls import RecursiveLS

rls_model = RecursiveLS(df["gdp"], df[["m2", "const"]].dropna())
rls_result = rls_model.fit()

# 绘制 CUSUM
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
rls_result.plot_cusum(ax=axes[0])
rls_result.plot_cusum_squares(ax=axes[1])
plt.savefig("output/figures/cusum_test.pdf", dpi=300, bbox_inches="tight")
```

**R (strucchange — 最完整的 Bai-Perron 实现)**
```r
library(strucchange)

# Bai-Perron 结构突变检验
bp_test <- breakpoints(gdp ~ m2 + cpi, data = df, h = 0.15,
                        breaks = 5)  # 最多 5 个突变点
summary(bp_test)
plot(bp_test)

# CUSUM 检验
cusum_test <- efp(gdp ~ m2 + cpi, data = df, type = "OLS-CUSUM")
plot(cusum_test)
sctest(cusum_test)     # 正式检验统计量
```

**Stata**
```stata
* Chow 检验（已知突变点，如 2008 年）
gen post2008 = (year > 2008)
reg gdp m2 cpi
* 使用 estat sbknown（已知突变点）
estat sbknown, breakpoint(2008)

* 未知突变点：Quandt-Andrews 检验
estat sbsingle

* Bai-Perron（多个突变点）
* ssc install sbsingle
sbsingle gdp m2 cpi, trim(0.15) max_breaks(5)
```

---

## 检验清单

- [ ] 所有变量的积分阶数已确认（ADF + PP + KPSS 三种检验）
- [ ] 单位根检验的趋势设定与数据实际趋势一致（有无确定性趋势？）
- [ ] VAR 滞后阶数已通过信息准则（AIC/BIC）和残差诊断确定
- [ ] VAR 稳定性已检验（特征根全在单位圆内）
- [ ] Johansen 检验的协整秩已确定，用于指导 VECM 设定
- [ ] IRF 置信带通过 Bootstrap 构建（非解析解）
- [ ] 结构突变已检验，若存在突变需在模型中处理（虚拟变量或分段估计）
- [ ] 所有检验结果汇总为表格，报告检验统计量和临界值

---

## 常见错误提醒

1. **对 I(1) 序列直接做 OLS**：造成伪回归（R² 虚高，DW 值极低）。解决：先做一阶差分，或确认存在协整后建 VECM。
2. **VAR 变量顺序影响 IRF**：Cholesky 分解对变量排序敏感，需根据经济理论确定因果次序（最外生的变量排第一）。
3. **KPSS 与 ADF 结论矛盾**：若 ADF 不拒绝单位根 + KPSS 也不拒绝平稳（两者都不拒绝），数据信息不足，加大样本或换频率。
4. **Johansen 检验中 det_order 设定错误**：det_order 决定协整方程是否含截距和趋势，设定错误会导致错误结论。
5. **ARDL 边界检验（Bounds Test）样本过小**：Pesaran 等（2001）原始临界值基于大样本渐近，小样本（T<30）需用有限样本临界值。
6. **格兰杰因果 ≠ 经济因果**：格兰杰因果是预测意义上的先验信息，不是结构因果，切勿在文章中混淆。

---

## 输出规范

- 单位根检验表：`output/tables/tableA_unit_root.tex`
- 协整检验结果：`output/tables/tableA_cointegration.tex`
- VAR/VECM 估计结果：`output/tables/table_var.tex`
- 格兰杰因果矩阵：`output/tables/table_granger.tex`
- IRF 图：`output/figures/fig_irf_[impulse]_[response].pdf`
- FEVD 图：`output/figures/fig_fevd.pdf`
- 结构突变图（CUSUM）：`output/figures/fig_cusum.pdf`

在论文中汇报格式示例：
```
Table X: Granger Causality Tests
Row: Response variable. Column: Impulse variable.
Entries are p-values. *** p<0.01, ** p<0.05, * p<0.10.
Optimal lag length selected by AIC. Sample: 2000Q1–2023Q4.
```
