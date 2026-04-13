---
name: time-series
description: "时间序列分析，含平稳性检验、协整、VAR/VECM/ARDL、SVAR四种识别、BVAR、TVP-VAR、MS-VAR、GARCH、预测评估"
---

# 时间序列分析 (Time Series)

## 概述

本 skill 提供宏观经济学与金融实证研究中标准时间序列分析流程，涵盖单位根检验、协整检验、VAR/VECM 模型、格兰杰因果、ARDL 模型、脉冲响应函数及结构突变检验，支持 Python / R 两种工具。

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
| Python | `statsmodels >= 0.14`, `pandas`, `numpy`, `matplotlib`, `arch` |
| R | `urca`, `tseries`, `vars`, `tsDyn`, `strucchange`, `mFilter`, `lpirfs`, `svars`, `BVAR`, `bvarsv`, `MSwM` |
| Stata | 不使用 Stata |

数据要求：
- 时间序列已按时间顺序排列
- 无重复时间点
- 缺失值已处理（时间序列插值或删除）
- 理解数据频率（日/月/季/年）

---

## 自动化决策树（10步）

> **必须按顺序执行，不可跳步。**

```
Step 1: 单位根检验 → 确定 I(d)
    所有 I(0) → VAR（水平值）
    所有 I(1) → Step 2
    混合阶     → ARDL + Bounds Test，跳至 Step 4

Step 2: 协整检验 → 确定长期关系
    存在协整 → VECM，跳至 Step 4
    无协整   → 差分后 VAR，继续 Step 3

Step 3: 基础模型选择
    I(0)           → VAR（水平值）
    I(1) + 协整    → VECM
    混合阶         → ARDL

Step 4: 是否需要结构因果识别？
    有外部工具变量（代理变量）→ Proxy-SVAR / LP-IV
    仅知冲击方向               → 符号约束（Sign Restrictions）
    有理论因果排序             → Cholesky 递归
    有理论长期约束             → Blanchard-Quah 长期零约束
    无需结构识别               → 继续（简化型 VAR/VECM）

Step 5: 参数是否时变？
    是 → TVP-VAR（或 TVP-FAVAR）
    否 → 继续

Step 6: 是否存在状态转换 / 非线性？
    可观测阈值（如 GDP 缺口正负）→ TVAR / STAR
    不可观测潜在状态             → MS-VAR
    否                           → 线性模型

Step 7: 变量过多 / 样本短？
    变量多 + 大数据集可提取因子 → FAVAR
    变量多 + 无大数据集         → BVAR（Minnesota 先验收缩）
    正常规模（n≤8, T>80）       → 标准 VAR/VECM

Step 8: 残差是否有 ARCH 效应？
    是 → 加入 GARCH / VAR-GARCH
    否 → 标准误即可

Step 9: IRF + FEVD + Granger（见下方各步骤）

Step 10: 预测评估
    → 滚动窗口预测 + RMSE/MAE + Diebold-Mariano 检验
```

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
    result = adfuller(series.dropna(), maxlag=max_lag,
                      regression=regression, autolag="AIC")
    t_stat, p_val, used_lags, nobs, crit_vals = result[:5]
    print(f"\nADF Test: {name}")
    print(f"  Stat={t_stat:.4f}  p={p_val:.4f}  Lags={used_lags}")
    print(f"  Crit: 1%={crit_vals['1%']:.4f} 5%={crit_vals['5%']:.4f}")
    return {"stat": t_stat, "p": p_val, "stationary": p_val <= 0.05}

for col in ["gdp", "m2", "cpi"]:
    adf_test(df[col], name=f"{col} (level)")
    adf_test(df[col].diff().dropna(), name=f"D.{col}")
```

#### KPSS 检验（原假设：平稳，与 ADF 互补）

```python
from statsmodels.tsa.stattools import kpss

def kpss_test(series, name="", regression="ct"):
    stat, p_val, lags, crit_vals = kpss(series.dropna(), regression=regression, nlags="auto")
    conclusion = "Stationary" if p_val > 0.05 else "Non-stationary"
    print(f"KPSS {name}: stat={stat:.4f} p={p_val:.4f} → {conclusion}")

kpss_test(df["gdp"], "GDP")
```

#### Zivot-Andrews 检验（含结构突变）

```python
from statsmodels.tsa.stattools import zivot_andrews

result = zivot_andrews(df["gdp"].dropna(), regression="ct", autolag="AIC")
za_stat, p_val, crit_vals, break_date, _ = result
print(f"ZA: stat={za_stat:.4f} p={p_val:.4f} break={break_date}")
```

**R**
```r
library(urca); library(tseries)
adf_result <- ur.df(df$gdp, type="trend", selectlags="AIC")
summary(adf_result)
pp.test(df$gdp)
kpss.test(df$gdp, null="Trend")
```

**检验结论矩阵**（报告表格）：

| 变量 | ADF (水平) | PP (水平) | KPSS (水平) | ADF (一差) | 积分阶 |
|------|-----------|----------|------------|-----------|-------|
| GDP | -1.23 | -1.45 | 0.89** | -5.67*** | I(1) |
| M2  | -0.87 | -1.02 | 1.12** | -4.89*** | I(1) |

---

### 步骤 2：协整检验

#### Johansen 检验（多变量）

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_test(df_vars, det_order=0, k_ar_diff=1):
    result = coint_johansen(df_vars.dropna(), det_order=det_order, k_ar_diff=k_ar_diff)
    print("迹检验:")
    for i in range(len(result.lr1)):
        sig = "*" if result.lr1[i] > result.cvt[i,1] else ""
        print(f"  r<={i}: stat={result.lr1[i]:.3f} crit5%={result.cvt[i,1]:.3f} {sig}")
    return result

johansen_test(df[["gdp","m2","cpi"]])
```

**R**
```r
library(urca)
jo <- ca.jo(cbind(df$gdp,df$m2,df$cpi), type="trace", K=2, ecdet="const")
summary(jo)
```

---

### 步骤 3：VAR / VECM / ARDL

#### VAR（所有变量 I(0) 或差分后）

```python
from statsmodels.tsa.vector_ar.var_model import VAR

model_order = VAR(df_stationary)
lag_order = model_order.select_order(maxlags=8)
p_opt = lag_order.selected_orders['aic']
var_fit = model_order.fit(p_opt, trend="c")
print(var_fit.summary())
print("稳定性:", var_fit.is_stable())
```

**R**
```r
library(vars)
lag_select <- VARselect(df_stationary, lag.max=8, type="const")
var_model <- VAR(df_stationary, p=lag_select$selection["AIC(n)"], type="const")
serial.test(var_model, lags.pt=12)
arch.test(var_model, lags.multi=5)
```

#### VECM（存在协整时）

```python
from statsmodels.tsa.vector_ar.vecm import VECM

vecm_fit = VECM(df[["gdp","m2","cpi"]].dropna(),
                k_ar_diff=1, coint_rank=1,
                deterministic="ci").fit()
print(vecm_fit.summary())
```

#### ARDL Bounds Test（混合积分阶）

```python
from statsmodels.tsa.ardl import ARDL, ardl_select_order, UECM

res_order = ardl_select_order(df["gdp"].dropna(), df[["m2","cpi"]].dropna(),
                               maxlag=4, maxorder=4, ic="aic")
uecm = UECM.from_ardl(ARDL(df["gdp"].dropna(), lags=res_order.model.order[0],
                            exog=df[["m2","cpi"]].dropna(),
                            order=res_order.model.order[1:]))
print(uecm.fit().bounds_test(case=3))
```

**R**
```r
library(ARDL)
auto <- auto_ardl(gdp ~ m2 + cpi, data=df, max_order=4, selection="AIC")
ardl_model <- ardl(gdp ~ m2 + cpi, data=df, order=auto$best_order)
bounds_f_test(ardl_model, case=3)
```

---

### 步骤 4：SVAR 结构识别（四种方案）

> **理论框架**：简化型 VAR 残差 u_t 与结构冲击 ε_t 的关系：u_t = A₀⁻¹ε_t。识别目标是从 Σ_u 中恢复 A₀，需要 n(n-1)/2 个约束。

| 方法 | 核心约束 | 适用场景 |
|------|---------|----------|
| Cholesky 递归 | 变量排序决定当期因果方向 | 有清晰理论因果链 |
| 短期零约束 | A₀ 中某元素=0（当期无影响） | 需强理论支持 |
| 长期零约束（Blanchard-Quah） | 某冲击长期影响为零 | 供给/需求冲击分离 |
| Proxy-SVAR / SVAR-IV | 外部工具变量识别结构冲击 | 有可信外生代理序列 |
| 符号约束（Sign Restrictions） | 仅约束 IRF 符号方向 | 理论仅能判断方向 |

#### 4a. Cholesky 识别（稳健性参考）

```python
# Cholesky 排序：最外生变量在前
# 示例：[货币政策利率] → [产出] → [价格]
irf = var_fit.irf(periods=12)
irf.plot(impulse="rate", response="gdp", orth=True)
# 稳健性：尝试不同排序，对比 IRF 形状
```

#### 4b. 长期零约束（Blanchard-Quah）

```r
library(svars)
# BQ 识别：需求冲击对产出无长期影响
bq_var <- VAR(df_bq, p=2, type="const")
bq_svar <- BQ(bq_var)
summary(bq_svar)
irf_bq <- irf(bq_svar, n.ahead=20, boot=TRUE)
plot(irf_bq)
```

#### 4c. Proxy-SVAR / SVAR-IV（核心补充）

代理变量要求：
- **相关性**：与目标结构冲击相关（弱工具检验 F > 10）
- **外生性**：与其他结构冲击不相关

经典代理序列：
- 货币政策冲击：Romer & Romer（2004）叙事冲击序列
- 财政政策冲击：军事支出新闻变量（Ramey 2011）
- 石油价格冲击：石油产量意外（Caldara & Herbst 2019）

```r
library(svars)
var_fit <- VAR(df_vars, p=2, type="const")
proxy_svar <- id.iv(var_fit, external.instruments=proxy_series)
summary(proxy_svar)
irf_proxy <- irf(proxy_svar, n.ahead=20, boot=TRUE, runs=500)
plot(irf_proxy)
```

**LP-IV（Local Projections with Instrumental Variables）**

```r
library(lpirfs)
lp_iv <- lp_lin_iv(
  endog_data     = df_vars,
  shock          = proxy_series,
  lags_endog_lin = 2,
  exog_data      = controls,
  hor            = 20,
  confint        = 1.96
)
plot(lp_iv)

# 手动实现（fixest）
library(fixest); library(dplyr)
lp_iv_coefs <- sapply(0:20, function(h) {
  df_h <- df_vars %>% mutate(y_ahead = lead(gdp, h))
  coef(feols(y_ahead ~ 1 | rate ~ proxy + l(rate,1:2) + l(gdp,1:2), data=df_h))["fit_rate"]
})
```

#### 4d. 符号约束（Sign Restrictions）

```r
library(svars)
# 定义符号矩阵：货币紧缩冲击 → 利率↑ 产出↓ 通胀↓
sign_mat <- matrix(c(1,NA,NA, -1,NA,NA, -1,NA,NA), nrow=3, ncol=3)
sign_svar <- id.dc(var_fit, sign_restr=sign_mat, ndraws=2000)
summary(sign_svar)
irf_sign <- irf(sign_svar, n.ahead=20)
plot(irf_sign)
# ⚠️ Sign Restrictions 是集合识别，报告 IRF 分位数区间，不是点估计
```

---

### 步骤 5：TVP-VAR（时变参数）

```r
library(bvarsv)
set.seed(42)
tvp_result <- bvar.sv.tvp(
  Y = as.matrix(df_vars),
  nlag = 2, nburn = 5000, nrep = 10000
)
plot(tvp_result)
```

> ⚠️ TVP-VAR 计算量大，建议变量数 ≤ 4，T > 80；先验灵敏度分析不可省。

---

### 步骤 6：MS-VAR（Markov 状态转换）

```r
library(MSwM)
base_lm <- lm(gdp ~ m2 + cpi, data=df)
ms_model <- msmFit(base_lm, k=2, sw=c(TRUE,TRUE,TRUE,TRUE),
                   control=list(parallel=FALSE))
summary(ms_model)
plotProb(ms_model, which=1)

# 多变量 MS-VAR
library(tsDyn)
msvar <- MSAR(df_vars, p=2, M=2)
summary(msvar)
```

---

### 步骤 7：BVAR（贝叶斯 VAR）

```r
library(BVAR)
mn_prior <- bv_minnesota(
  lambda = bv_lambda(mode=0.2, sd=0.4),
  alpha  = bv_alpha(mode=2),
  psi    = bv_psi()
)
bvar_fit <- bvar(data=as.matrix(df_vars), lags=4,
                 n_draw=10000, n_burn=5000,
                 priors=bv_priors(hyper=mn_prior))
bvar_irf   <- irf(bvar_fit, n_ahead=20, conf_bands=c(0.16,0.84))
plot(bvar_irf)
bvar_fcast <- predict(bvar_fit, horizon=8)
plot(bvar_fcast)
```

> **Minnesota 先验三参数**：λ（总体收缩）、α（滞后衰减）、ψ（交叉变量相对自变量收缩比）。λ 越小收缩越强。

---

### 步骤 8：GARCH 族

```python
from arch import arch_model
garch = arch_model(residuals_series, vol="Garch", p=1, q=1, dist="normal")
print(garch.fit(disp="off").summary())
```

```r
library(rugarch)
spec <- ugarchspec(
  variance.model     = list(model="sGARCH", garchOrder=c(1,1)),
  mean.model         = list(armaOrder=c(1,0), include.mean=TRUE),
  distribution.model = "std"
)
show(ugarchfit(spec, data=residuals_series))
```

---

### 步骤 9：格兰杰因果 + IRF + FEVD

#### 格兰杰因果

```python
from statsmodels.tsa.stattools import grangercausalitytests

def granger_test_matrix(df_vars, opt_lag=2):
    """
    opt_lag 应来自 VAR AIC 确定的最优滞后，在该滞后处单次检验。
    不要取各滞后最小 p 值——会导致多重检验膨胀。
    """
    cols = df_vars.columns.tolist()
    results = pd.DataFrame(index=cols, columns=cols, dtype=object).fillna("")
    for y in cols:
        for x in cols:
            if y == x:
                results.loc[y,x] = "—"; continue
            gc = grangercausalitytests(df_vars[[y,x]].dropna(), maxlag=opt_lag, verbose=False)
            p = gc[opt_lag][0]["ssr_ftest"][1]
            sig = "***" if p<0.01 else ("**" if p<0.05 else ("*" if p<0.10 else ""))
            results.loc[y,x] = f"{p:.3f}{sig}"
    return results

print(granger_test_matrix(df_stationary, opt_lag=2))
```

#### IRF

```python
irf = var_fit.irf(periods=12)
fig = irf.plot(impulse="d_m2", response="d_gdp", orth=True, cumsum=False)
plt.savefig("output/figures/irf_m2_to_gdp.pdf", dpi=300, bbox_inches="tight")
# GIRF（不依赖排序）
fig_g = var_fit.irf(periods=12).plot(impulse="d_m2", response="d_gdp", orth=False)
```

**R**
```r
irf_result <- irf(var_model, impulse="d_m2", response="d_gdp",
                  n.ahead=12, ortho=TRUE, boot=TRUE, ci=0.95, runs=1000)
plot(irf_result)
fevd_result <- fevd(var_model, n.ahead=10)
plot(fevd_result)
```

#### LP-IRF（Jordà 2005）

```r
library(lpirfs)
lp_result <- lp_lin(endog_data=df_stationary, lags_endog_lin=2,
                     trend=0, shock_type=1, confint=1.96, hor=12)
plot(lp_result)
```

---

### 步骤 10：预测评估

```python
import numpy as np
from scipy import stats

def diebold_mariano(e1, e2, h=1):
    """比较两个模型预测精度，H0: 两模型相同"""
    d = e1**2 - e2**2
    d_bar = np.mean(d)
    T = len(d)
    gamma0 = np.var(d, ddof=0)
    gammas = [np.cov(d[j:], d[:-j])[0,1] if j>0 else gamma0 for j in range(h)]
    var_d = (gamma0 + 2*sum(gammas[1:])) / T
    dm_stat = d_bar / np.sqrt(var_d)
    p_val = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    print(f"DM stat={dm_stat:.4f}  p={p_val:.4f}")
    return dm_stat, p_val
```

**R**
```r
library(forecast)
dm.test(e1=resid_model1, e2=resid_model2, h=1, power=2)
```

---

### 结构突变检验（Bai-Perron）

```r
library(strucchange)
bp <- breakpoints(gdp ~ m2 + cpi, data=df, h=0.15, breaks=5)
summary(bp); plot(bp)
cusum <- efp(gdp ~ m2 + cpi, data=df, type="OLS-CUSUM")
sctest(cusum)
```

---

## 检验清单

> **如何判断自己属于哪一层？**
> - 探索性分析（了解数据特征、初步验证想法）→ 只需保证🔴红线
> - 工作论文/投稿稿→ 红线 + 所有适用的🟡条件项
> - 正式发表/审稿后修改 → 三层全部

---

### 🔴 红线（始终必須，4条）

- [ ] 所有变量积分阶数已确认（ADF + PP + KPSS三种检验）
- [ ] VAR 稳定性已检验（特征根全在单位圆内）
- [ ] 所有 IRF 的识别方案已声明（哪种 SVAR ，依据是什么）
- [ ] 格兰杰因果已明确说明≠经济因果

---

### 🟡 条件触发（满足前提才需要，5条）

- [ ] 若用 **Proxy-SVAR**：工具变量相关性已检验（F > 10）
- [ ] 若用 **符号约束**：已报告识别集（IRF 分位数区间）而非点估计
- [ ] 若 arch.test **残差有 ARCH 效应**：已建模 GARCH
- [ ] 若 CUSUM 显著或理论判断 **参数时变**：已使用 TVP-VAR
- [ ] 若 **变量 > 8 或 T < 80**：已使用 BVAR Minnesota 先验

---

### ⬜ 推荐（正式投稿前补，3条）

- [ ] GIRF 稳健性对比 Cholesky IRF
- [ ] BVAR 先验灵敏度分析（尝试不同 λ 弱/强收缩）
- [ ] 预测模型滚动评估 + Diebold-Mariano 检验

---

## 常见错误提醒

1. **对 I(1) 直接 OLS**：伪回归（R² 虚高，DW 极低）→ 先差分或 VECM
2. **Cholesky 排序随意**：IRF 排序敏感 → 用 GIRF 或 Proxy-SVAR 做稳健性
3. **Proxy 工具弱（F<10）**：弱工具偏误，不如用符号约束
4. **符号约束报告点估计**：Sign Restrictions 是集合识别，应报告分位数区间
5. **BVAR 先验未调**：λ 过大/过小影响结论 → 做先验灵敏度分析
6. **KPSS 与 ADF 矛盾**：数据信息不足，加大样本或换频率
7. **格兰杰因果 ≠ 经济因果**：预测意义的先验信息，文章中必须说明
8. **ARDL Bounds Test 样本小（T<30）**：Pesaran 渐近临界值失效 → 用有限样本临界值

---

## Estimand 声明

| 方法 | Estimand | 必须声明 |
|------|----------|---------|
| VAR（差分后） | 短期动态乘数 ΔY_{t+h}/Δε_t | 冲击标准化方式 |
| VECM | 长期均衡系数 + 误差修正速度 α | 协整向量标准化约束 |
| ARDL | 短期系数 + 长期系数 β/(1-ρ) | 明确区分短长期 |
| IRF-Cholesky | 排序依赖的脉冲响应 | 排序经济依据 + GIRF 稳健性 |
| **IRF-Proxy** | **外生货币/财政冲击的因果响应** | **代理变量相关性和外生性论证** |
| **IRF-Sign** | **满足符号约束的 IRF 识别集** | **报告集合而非点估计** |
| LP-IRF | 直接估计 h 期后脉冲响应 | 是否控制混淆动态路径 |
| BVAR | 先验收缩后的贝叶斯后验 IRF | 先验设定 + 灵敏度分析 |
| 格兰杰因果 | 预测意义先验信息，非结构因果 | 明确声明不等于经济因果 |

---

## 输出规范

- 单位根检验表：`output/tables/tableA_unit_root.tex`
- 协整检验结果：`output/tables/tableA_cointegration.tex`
- VAR/VECM/SVAR 估计：`output/tables/table_var.tex`
- 格兰杰因果矩阵：`output/tables/table_granger.tex`
- IRF 图：`output/figures/fig_irf_[impulse]_[response].pdf`
- FEVD 图：`output/figures/fig_fevd.pdf`
- 预测评估表：`output/tables/table_forecast_eval.tex`
- 结构突变图：`output/figures/fig_cusum.pdf`
