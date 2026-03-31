# 面板数据分析 — 固定效应与随机效应

## 概述

面板数据（Panel Data）包含多个个体在多个时间点的观测，允许控制不随时间变化的个体异质性（个体固定效应）和不同个体间共同的时间趋势（时间固定效应）。本 skill 涵盖经典双向固定效应、高维固定效应、标准误选择及相关诊断检验。

**适用场景：**
- 数据为面板结构（同一个体跨多期观测）
- 需要控制个体/时间层面的不可观测异质性
- 关注个体内（within）变异而非个体间（between）变异
- 适用于企业、省份、国家、个人等各类单位

---

## 前置条件

### 数据结构要求

```
长格式（long format）：
  - 个体 ID 列（entity_id）：企业/地区编码，字符或数值
  - 时间列（time）：年份/季度，数值型
  - 结果变量（Y）
  - 核心解释变量（X）
  - 控制变量（W）
  - 固定效应分组变量（可以是 entity_id, time, industry, province 等）

平衡/非平衡面板均可，但需注意标准误选择
```

### 面板基本描述

```python
# Python: 面板描述
import pandas as pd

print("Panel dimensions:")
print(f"  N (entities): {df['entity_id'].nunique()}")
print(f"  T (periods):  {df['time'].nunique()}")
print(f"  NT (obs):     {len(df)}")

# 是否平衡
obs_per_entity = df.groupby('entity_id')['time'].count()
print(f"  Balanced: {obs_per_entity.std() == 0}")
print(obs_per_entity.describe())

# 组内变异检查（within variation）
for var in key_vars:
    within_std = df.groupby('entity_id')[var].transform('std').mean()
    total_std  = df[var].std()
    print(f"  {var}: within/total std = {within_std/total_std:.3f}")
```

```r
# R: 面板描述（plm / dplyr）
library(dplyr)

cat("N entities:", n_distinct(df$entity_id), "\n")
cat("T periods: ", n_distinct(df$time), "\n")
cat("NT obs:    ", nrow(df), "\n")

# 组内变异（within variation）—— FE 靠这个识别
df %>%
  group_by(entity_id) %>%
  summarise(within_sd_x = sd(x, na.rm=TRUE)) %>%
  summarise(mean_within_sd = mean(within_sd_x)) %>%
  print()
```

---

## 分析步骤

### Step 1：组内变异检验（Within Variation Check）

固定效应估计靠**组内变异**识别。如果核心变量的组内变异极小，FE 估计不可靠（标准误膨胀）。

```python
# Python: 组内 vs 组间变异分解
def within_between_var(df, var, group):
    overall_mean = df[var].mean()
    between_var  = df.groupby(group)[var].mean().var()       # 组间方差
    within_var   = df.groupby(group)[var].transform(lambda x: (x - x.mean())).var()  # 组内方差
    total_var    = df[var].var()
    print(f"\n{var} Variance Decomposition (group={group}):")
    print(f"  Overall  SD: {total_var**0.5:.4f}")
    print(f"  Between  SD: {between_var**0.5:.4f}")
    print(f"  Within   SD: {within_var**0.5:.4f}")
    print(f"  Within / Total: {within_var / total_var:.3f}")

for var in key_vars:
    within_between_var(df, var, 'entity_id')
```

```stata
* Stata: xtsum
xtset entity_id time
xtsum y x1 x2
* 重点看 "within" 一行的 std dev
* 若某变量 within std ≈ 0，固定效应无法识别
```

---

### Step 2：Hausman 检验（FE vs RE 选择）

```python
# Python: Hausman 检验（linearmodels）
from linearmodels.panel import PanelOLS, RandomEffects

df_panel = df.set_index(['entity_id', 'time'])

fe_res = PanelOLS(df_panel['y'], df_panel[exog_vars], entity_effects=True).fit()
re_res = RandomEffects(df_panel['y'], df_panel[exog_vars]).fit()

from linearmodels.panel.results import compare
comparison = compare({'FE': fe_res, 'RE': re_res})
print(comparison)

# Hausman test: H0=RE 一致（个体效应与 X 不相关）
from linearmodels.panel.utility import hausman
h_stat, h_pval, dof = hausman(fe_res, re_res)
print(f"Hausman H={h_stat:.3f}, p={h_pval:.4f} (df={dof})")
```

```r
# R: plm Hausman 检验
library(plm)

fe_res <- plm(y ~ x1 + x2 + x3, data = df, index = c("entity_id", "time"), model = "within")
re_res <- plm(y ~ x1 + x2 + x3, data = df, index = c("entity_id", "time"), model = "random")

phtest(fe_res, re_res)
# H0: RE 一致（个体效应与解释变量不相关）
# p < 0.05 → 拒绝 H0 → 用 FE
```

```stata
* Stata: Hausman 检验
xtreg y x1 x2 x3, fe
estimates store fe_model
xtreg y x1 x2 x3, re
estimates store re_model
hausman fe_model re_model
* p < 0.05 → 用固定效应
```

**实践提示：** 大多数经管研究默认使用 FE（更保守，控制不可观测个体异质性），RE 仅在有理论依据或 Hausman 检验通过时使用。

---

### Step 3：固定效应模型估计

#### 3a. Entity FE（个体固定效应）

$$Y_{it} = \alpha_i + \beta X_{it} + \gamma W_{it} + \varepsilon_{it}$$

```python
# Python: Entity FE
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

df_panel = df.set_index(['entity_id', 'time'])

mod = PanelOLS(
    dependent  = df_panel['y'],
    exog       = sm.add_constant(df_panel[['x1', 'x2', 'x3']]),
    entity_effects = True,
    time_effects   = False
)
res_fe = mod.fit(cov_type='clustered', cluster_entity=True)
print(res_fe.summary)
```

```r
# R: fixest（Entity FE）
library(fixest)

res_efe <- feols(y ~ x1 + x2 + x3 | entity_id, data = df, cluster = ~entity_id)
etable(res_efe)
```

```stata
* Stata: xtreg fe
xtreg y x1 x2 x3, fe cluster(entity_id)
```

#### 3b. Two-Way FE（双向固定效应）

$$Y_{it} = \mu_i + \lambda_t + \beta X_{it} + \gamma W_{it} + \varepsilon_{it}$$

```python
# Python: TWFE
mod_twfe = PanelOLS(
    dependent    = df_panel['y'],
    exog         = sm.add_constant(df_panel[['x1', 'x2', 'x3']]),
    entity_effects = True,
    time_effects   = True
)
res_twfe = mod_twfe.fit(cov_type='clustered', cluster_entity=True)
```

```r
# R: TWFE（推荐 fixest）
res_twfe <- feols(y ~ x1 + x2 + x3 | entity_id + time, data = df, cluster = ~entity_id)
```

```stata
* Stata: TWFE（reghdfe，推荐）
ssc install reghdfe

reghdfe y x1 x2 x3, absorb(entity_id time) cluster(entity_id)
```

#### 3c. 高维固定效应（High-Dimensional FE）

例如：个体 × 行业 FE，省份 × 年份 FE，个体 × 时间双向。

```r
# R: fixest 高维 FE（速度极快）
res_hdfe <- feols(
  y ~ x1 + x2 | entity_id + time + industry^year,  # ^ 表示交互 FE
  data    = df,
  cluster = ~entity_id
)

# 逐步加入 FE（展示识别来源）
res1 <- feols(y ~ x1 + x2 | entity_id, df, cluster=~entity_id)
res2 <- feols(y ~ x1 + x2 | entity_id + time, df, cluster=~entity_id)
res3 <- feols(y ~ x1 + x2 | entity_id + time + industry^year, df, cluster=~entity_id)
etable(res1, res2, res3)
```

```stata
* Stata: reghdfe 高维 FE
reghdfe y x1 x2, absorb(entity_id time industry#year) cluster(entity_id)
* absorb() 支持交互固定效应
```

---

### Step 4：标准误选择

| 场景 | 推荐标准误 | 代码 |
|------|-----------|------|
| 标准面板 | 聚类（个体层面） | `cluster(entity_id)` |
| 处理在组层面 | 聚类（组层面） | `cluster(province)` |
| 少于 30 个聚类 | Wild Bootstrap | `boottest` / `fwildclusterboot` |
| 截面相关（宏观数据） | Driscoll-Kraay | `xtscc` in Stata |
| 空间相关 | Conley SE | `conleyreg` in R |
| 时间序列相关 | Newey-West HAC | `NeweyWest` in R |

```r
# R: Driscoll-Kraay SE（适合宏观面板，T 较大，截面相关）
library(lmtest)
library(sandwich)

res_plm <- plm(y ~ x1 + x2, data = df, index = c("entity_id", "time"), model = "within")
coeftest(res_plm, vcov = vcovSCC(res_plm, type = "HC3", maxlag = 2))  # DK SE
```

```r
# R: Wild Bootstrap（少聚类）
library(fwildclusterboot)
boottest(res_twfe, clustid = "province_id", param = "x1", B = 9999)
```

```stata
* Stata: Driscoll-Kraay（xtscc）
ssc install xtscc

xtscc y x1 x2, fe lag(2)

* Stata: Wild Bootstrap
ssc install boottest
boottest x1, cluster(province_id) reps(9999)
```

---

### Step 5：逐步回归（渐进加入控制）

```r
# R: 逐步加入控制变量，展示系数稳定性
res_baseline  <- feols(y ~ x1 | entity_id, df, cluster=~entity_id)
res_controls  <- feols(y ~ x1 + control1 + control2 | entity_id, df, cluster=~entity_id)
res_twfe      <- feols(y ~ x1 + control1 + control2 | entity_id + time, df, cluster=~entity_id)
res_hdfe      <- feols(y ~ x1 + control1 + control2 | entity_id + time + industry^year, df, cluster=~entity_id)

etable(res_baseline, res_controls, res_twfe, res_hdfe,
       title   = "Panel FE: Stepwise Results",
       headers = c("Entity FE", "+Controls", "+Time FE", "+Industry×Year FE"),
       fitstat = ~ r2 + r2_within + n)
```

---

### Step 6：诊断检验

#### 6a. 序列相关检验（Serial Correlation）

```r
# R: Wooldridge test for serial correlation in panel data
library(plm)
pbgtest(res_plm)   # Breusch-Godfrey test for panels
# 若显著 → 须用 HAC SE 或 Driscoll-Kraay
```

```stata
* Stata: Wooldridge serial correlation test
ssc install xtserial
xtserial y x1 x2
```

#### 6b. 截面相关检验（Cross-sectional Dependence）

```r
# R: Pesaran CD test
library(plm)
pcdtest(res_plm, test = "cd")
# 若显著 → 用 Driscoll-Kraay SE
```

```stata
* Stata: Pesaran CD test
xtcd2 y x1, pesaran   * 或
xttest2                * 需要 xttest2 包
```

#### 6c. 面板单位根检验（Panel Unit Root，长面板必做）

```r
# R: IPS test（Im, Pesaran, Shin）
library(plm)
purtest(y ~ 1, data = df, index = c("entity_id", "time"),
        pmax = 2, test = "ips", exo = "trend")
# H0: 所有个体均有单位根
# p < 0.05 → 拒绝单位根（平稳）
```

```stata
* Stata: xtunitroot
xtunitroot ips y         * Im-Pesaran-Shin
xtunitroot fisher y, dfuller lags(2)
```

---

## 必做检验清单

| 检验 | 方法 | 通过标准 / 说明 |
|------|------|---------------|
| 组内变异 | xtsum / 方差分解 | Within SD > 0（有足够变异） |
| Hausman 检验 | FE vs RE | 通常 FE 更稳健 |
| 序列相关 | Wooldridge test | 若显著，升级SE |
| 截面相关 | Pesaran CD | 若显著，用 DK SE |
| 面板单位根 | IPS / Fisher | 长面板（T>10）必做 |
| 异方差 | Breusch-Pagan | 若显著，用稳健 SE |
| 逐步回归稳定性 | 系数变化 < 30% | 核心系数方向和显著性稳定 |

---

## 常见错误提醒

> **错误 1：FE 后无法估计不随时间变化的变量**
> 个体固定效应会吸收所有个体层面不随时间变化的变量（如性别、地区、行业代码）。若这些变量是研究核心，考虑 RE 或 Mundlak/Correlated RE 方法。

> **错误 2：双向 FE 不等于已控制所有混淆**
> TWFE 只控制了加性的个体效应和时间效应，时变的个体特征（如企业某年新政策）仍会引起内生性问题。

> **错误 3：标准误聚类层级不对**
> 聚类应在处理分配层级（处理在省级就聚类到省）。聚类到太细的层级会低估标准误，聚类到太粗的层级会高估。

> **错误 4：忽略序列相关**
> 面板数据中同一个体跨期误差项通常正相关。不处理序列相关会严重低估标准误（过度拒绝 H0）。需检验并使用聚类标准误（自动处理同组内相关）。

> **错误 5：R² 报告错误**
> 固定效应模型的 R² 有多种：overall R²、between R²、within R²。结果表应报告 **within R²**（组内拟合优度），不要报告 overall R²（被 FE 虚高）。

> **错误 6：高维 FE 中忘记检查吸收了多少自由度**
> 高维 FE 占用大量自由度，应报告 FE 的维度和剩余自由度，避免实际上过度参数化。

---

## 输出规范

### 主结果表格式

```r
# R: fixest etable 输出
etable(res1, res2, res3, res4,
       title   = "Main Results: Panel Fixed Effects",
       headers = c("(1)Entity FE", "(2)+Controls", "(3)TWFE", "(4)HDFE"),
       fitstat = ~ r2_within + n + G,   # G = 聚类数
       dict    = c(x1 = "Core Variable X", ...),
       tex     = FALSE)
```

**必须注明：**
1. 固定效应类型（个体 FE、年份 FE、行业×年份 FE）
2. 标准误类型（Clustered at entity level）
3. Within R²、N、聚类数 G

### 文件命名

```
output/
  panel_within_variation.csv   # 组内变异分解
  panel_hausman.txt            # Hausman 检验结果
  panel_main_results.csv       # 主回归结果表
  panel_serial_corr.txt        # 序列相关检验
  panel_unit_root.txt          # 单位根检验（长面板）
  panel_robustness.csv         # 稳健性检验（不同SE/样本）
```

---

### Section: GMM-lite（动态面板System GMM）

动态面板模型：当因变量的滞后项（$Y_{i,t-1}$）作为解释变量，静态FE估计会因"Nickell偏误"（Nickell, 1981）产生有偏估计，此时需要系统GMM（Blundell & Bond, 1998）。

#### 触发条件

```
是否需要动态模型？
├── 理论上：存在调整成本/习惯形成/状态依赖？
│   （如：企业投资存在调整成本，当期投资受上期影响）
├── 实证上：残差存在序列相关？
│   （FE残差的AR(1)检验显著）
├── 数据上：T < 15-20 使Nickell偏误显著？
│   （N大T小的典型企业面板：N=3000, T=8 → Nickell偏误≈1/T≈12.5%）
│
├── 至少两条为是 → 使用System GMM
├── 都不满足 → 静态FE即可
└── 不确定 → 同时跑FE和GMM比较（ρ_FE < ρ_GMM < ρ_OLS应成立）
```

**Nickell偏误大小估算：** 当 T=8，偏误约 1/T ≈ 12.5%；T=20 时降至 5%；T>30 基本可忽略。

#### System GMM代码模板（R代码，plm::pgmm）

```r
# R: 动态面板 System GMM
library(plm)
library(lmtest)

# ============================================================
# Step 1: 数据准备
# ============================================================
# 确保数据为pdata.frame格式
df_panel <- pdata.frame(df, index = c("entity_id", "year"))

# 检查是否需要动态模型
# （1）理论检查：文献中是否有调整成本/状态依赖讨论
# （2）序列相关检验（用静态FE残差）
fe_static <- plm(y ~ x1 + x2 + control1 + control2,
                 data  = df_panel,
                 model = "within",
                 effect = "twoways")
pbgtest(fe_static, order = 1)  # AR(1)显著 → 考虑动态模型

# ============================================================
# Step 2: System GMM估计
# ============================================================
# 公式：y ~ lag(y,1) + x1 + x2 | lag(y, 2:4) + lag(x1, 1:2)
# 含义：
#   左侧 y ~ lag(y,1) + x1 + x2：回归方程（含滞后因变量）
#   右侧 | 后：工具变量（滞后阶数范围）

gmm_sys <- pgmm(
  y ~ lag(y, 1) + x1 + x2 + control1 + control2  # 回归方程
  | lag(y, 2:4) + lag(x1, 1:2),                  # 工具变量（层差方程）
  data        = df_panel,
  effect      = "twoways",          # 双向固定效应
  model       = "twosteps",         # 两步GMM（更有效率）
  collapse    = TRUE,               # ⚠️ 重要：减少工具变量数量（防止过多）
  transformation = "ld"             # "ld"=系统GMM（差分+水平方程）
)

summary(gmm_sys, robust = TRUE)     # 使用稳健标准误

# ============================================================
# Step 3: 差分GMM对比（仅差分方程，Arellano-Bond）
# ============================================================
gmm_diff <- pgmm(
  y ~ lag(y, 1) + x1 + x2 + control1 + control2
  | lag(y, 2:4) + lag(x1, 1:2),
  data           = df_panel,
  effect         = "twoways",
  model          = "twosteps",
  collapse       = TRUE,
  transformation = "d"              # "d"=差分GMM
)
summary(gmm_diff, robust = TRUE)
```

#### 强制检验清单（自动执行）

```
GMM估计完成后自动检查：
├── AR(1) p < 0.05？ → 预期中（差分残差存在一阶相关是正常的）
├── AR(2) p > 0.10？ → 支持工具变量有效
│   └── AR(2) p < 0.10 → ⚠️ 严重：增加工具变量滞后阶数（如lag(y, 3:5)）
├── Hansen p > 0.10？ → 工具变量联合有效（过度识别检验）
│   └── p > 0.90 → ⚠️ 工具变量可能过多（Hansen统计量失效），用collapse
├── 工具变量数 ≤ N？ → 安全
│   └── > N → ⚠️ 必须减少（严重过度拟合）
├── ρ_FE < ρ_GMM < ρ_OLS？ → GMM落在合理区间
│   └── 超出 → ⚠️ 模型可能设定错误（工具变量、滞后阶数问题）
└── 长期效应 β/(1-ρ) 自动计算（当|ρ|<1时）
```

```r
# R: GMM强制检验清单完整实现
check_gmm <- function(gmm_model, fe_model_rho = NULL, ols_rho = NULL, N = NULL) {
  cat("=== GMM强制检验清单 ===\n\n")

  sum_gmm <- summary(gmm_model, robust = TRUE)

  # ---- 1. AR检验 ----
  ar_tests <- mtest(gmm_model, order = 1:2, robust = TRUE)
  ar1_p <- ar_tests$p.value[1]
  ar2_p <- ar_tests$p.value[2]

  cat(sprintf("【1】AR(1) p = %.4f", ar1_p))
  cat(if (ar1_p < 0.05) " ✓ 预期中（显著）\n" else " ⚠️ AR(1)不显著，可能有问题\n")

  cat(sprintf("【2】AR(2) p = %.4f", ar2_p))
  if (ar2_p > 0.10) {
    cat(" ✓ 工具变量有效（不显著）\n")
  } else {
    cat(" ⚠️ 严重：AR(2)显著！增加工具变量滞后阶数（建议lag(y, 3:6)）\n")
  }

  # ---- 2. Hansen J检验 ----
  hansen_j <- sum_gmm$sargan       # 或 sum_gmm$hansen
  if (!is.null(hansen_j)) {
    hansen_p <- hansen_j$p.value
    cat(sprintf("【3】Hansen p = %.4f", hansen_p))
    if (hansen_p < 0.10) {
      cat(" ⚠️ 工具变量可能无效！检查排他性约束\n")
    } else if (hansen_p > 0.90) {
      cat(" ⚠️ p>0.90：工具变量可能过多（Hansen失效），使用collapse\n")
    } else {
      cat(" ✓ 工具变量联合有效\n")
    }
  }

  # ---- 3. 工具变量数量检查 ----
  n_instr <- length(gmm_model$instruments)
  cat(sprintf("【4】工具变量数 = %d", n_instr))
  if (!is.null(N)) {
    cat(sprintf("，样本N = %d", N))
    if (n_instr > N) {
      cat(" ⚠️ 必须减少！工具变量数>N → 强制collapse=TRUE或缩短滞后阶数\n")
    } else if (n_instr > N * 0.5) {
      cat(" ⚠️ 工具变量较多（>N/2），建议collapse\n")
    } else {
      cat(" ✓ 在安全范围内\n")
    }
  } else {
    cat("\n")
  }

  # ---- 4. ρ区间检查 ----
  rho_gmm <- coef(gmm_model)["lag(y, 1)"]
  cat(sprintf("【5】ρ_GMM = %.4f", rho_gmm))
  if (!is.null(fe_model_rho) && !is.null(ols_rho)) {
    cat(sprintf("（ρ_FE=%.4f, ρ_OLS=%.4f）", fe_model_rho, ols_rho))
    if (rho_gmm > fe_model_rho && rho_gmm < ols_rho) {
      cat(" ✓ GMM在FE-OLS区间内\n")
    } else {
      cat(" ⚠️ GMM不在合理区间，检查工具变量设定\n")
    }
  } else {
    cat("\n  （建议同时提供FE和OLS的ρ进行对比）\n")
  }

  # ---- 5. 长期效应计算 ----
  beta_x  <- coef(gmm_model)["x1"]    # 核心变量短期系数
  if (abs(rho_gmm) < 1) {
    lr_effect <- beta_x / (1 - rho_gmm)
    cat(sprintf("【6】短期效应 β = %.4f\n", beta_x))
    cat(sprintf("     长期效应 β/(1-ρ) = %.4f / (1 - %.4f) = %.4f\n",
                beta_x, rho_gmm, lr_effect))
    cat("     必须在论文中区分短期和长期效应（Estimand声明要求）\n")
  } else {
    cat("⚠️ |ρ| ≥ 1，不平稳！不能计算长期效应，检查模型设定\n")
  }

  # ---- 6. 样本量警告 ----
  if (!is.null(N) && N < 50) {
    cat("\n⚠️ 自动警告：N < 50，样本量可能不足以支撑GMM（需至少50个个体）\n")
  }
}

# 使用示例
# 先获取OLS和FE的ρ系数
ols_dyn <- lm(y ~ lag_y + x1 + x2, data = df)
rho_ols <- coef(ols_dyn)["lag_y"]
rho_fe  <- coef(fe_static)["lag(y,1)"]  # 从静态FE中获取（有Nickell偏误，作下界）

check_gmm(gmm_sys, fe_model_rho = rho_fe, ols_rho = rho_ols, N = n_distinct(df$entity_id))
```

#### 自动警告规则

```r
# 自动警告触发条件（嵌入check_gmm函数）：
# 1. N < 50 → "⚠️ 样本量可能不足以支撑GMM"
# 2. 工具变量数 > N → "⚠️ 必须减少，强制collapse=TRUE"
# 3. ρ不在[ρ_FE, ρ_OLS]区间 → "⚠️ 模型可能设定错误"
# 4. AR(2) p < 0.10 → "⚠️ 工具变量无效，增加滞后阶数"
# 5. Hansen p > 0.90 → "⚠️ 工具变量过多"
```

#### Estimand声明

动态面板GMM → **必须区分短期效应和长期效应**：

| 效应 | 公式 | 解释 |
|------|------|------|
| 短期效应 | $\hat{\beta}$ | X变化1单位后，当期Y的即时变化 |
| 长期效应 | $\hat{\beta}/(1-\hat{\rho})$ | X持续维持变化后，Y达到新稳态的总变化 |

**声明模板：**
> "本文GMM估计的短期效应为 $\hat{\beta}=X$，长期效应（稳态乘数）为 $\hat{\beta}/(1-\hat{\rho})=Y$，说明处理效应会通过调整过程逐渐放大。"

---

### Section: 重叠性检验

重叠性假设（Common Support）：处理组和对照组在协变量空间上有充分重叠，才能进行有效的因果比较。

#### 倾向得分分布图（处理组vs对照组）

```r
# R: 倾向得分分布图 + Common Support检验
library(ggplot2)
library(dplyr)

# Step 1: 估计倾向得分（Propensity Score）
ps_model <- glm(
  treated ~ control1 + control2 + control3 + control4,
  family = binomial(link = "logit"),
  data   = df
)
df$ps <- predict(ps_model, type = "response")

# Step 2: 重叠性统计
ps_treated  <- df$ps[df$treated == 1]
ps_control  <- df$ps[df$treated == 0]

# Common Support区域
cs_min <- max(min(ps_treated), min(ps_control))
cs_max <- min(max(ps_treated), max(ps_control))

cat(sprintf("=== 重叠性检验 ===\n"))
cat(sprintf("处理组PS范围: [%.4f, %.4f]\n", min(ps_treated), max(ps_treated)))
cat(sprintf("对照组PS范围: [%.4f, %.4f]\n", min(ps_control), max(ps_control)))
cat(sprintf("Common Support: [%.4f, %.4f]\n", cs_min, cs_max))

# 在Common Support外的样本比例
outside_cs <- mean(df$ps < cs_min | df$ps > cs_max)
cat(sprintf("Common Support外样本: %.1f%%\n", outside_cs * 100))

# Step 3: 倾向得分分布图（镜像图）
p_overlap <- ggplot(df, aes(x = ps, fill = factor(treated))) +
  # 镜像密度图
  geom_density(data = df %>% filter(treated == 1),
               aes(y = after_stat(density)), alpha = 0.5) +
  geom_density(data = df %>% filter(treated == 0),
               aes(y = -after_stat(density)), alpha = 0.5) +
  # Common Support区域标注
  annotate("rect",
           xmin = cs_min, xmax = cs_max,
           ymin = -Inf, ymax = Inf,
           alpha = 0.1, fill = "blue") +
  geom_vline(xintercept = c(cs_min, cs_max),
             linetype = "dashed", color = "blue", linewidth = 0.8) +
  scale_fill_manual(values  = c("#E57373", "#42A5F5"),
                    labels  = c("对照组", "处理组"),
                    name    = "组别") +
  labs(title    = "倾向得分分布（处理组 vs 对照组）",
       subtitle = sprintf("Common Support: [%.3f, %.3f]（蓝色阴影区域）", cs_min, cs_max),
       x        = "倾向得分（Propensity Score）",
       y        = "密度（处理组↑ / 对照组↓）") +
  theme_bw(base_size = 12)

ggsave("output/overlap_ps_distribution.png", p_overlap,
       width = 10, height = 6, dpi = 300)

# Step 4: 截断Common Support（用于主回归稳健性）
df_cs <- df %>% filter(ps >= cs_min & ps <= cs_max)
cat(sprintf("\nCommon Support样本: %d (原始: %d, 保留: %.1f%%)\n",
            nrow(df_cs), nrow(df), nrow(df_cs)/nrow(df)*100))
```

---

### Section: 门槛效应（Hansen门槛模型）

适用场景：核心变量与因变量之间存在非线性关系，且预期存在结构突变点（门槛）。例如：企业规模超过某阈值后，规模效应改变符号；债务率超过门槛后，对投资的影响从正转负。

#### R代码（grid search + bootstrap p值）

```r
# R: Hansen (1999, 2000) 面板门槛模型
# 方法1: 使用 threshold 包（如可用）
# install.packages("threshold")  # 或从GitHub安装

# 方法2: 手动grid search（通用）
library(plm)
library(dplyr)

#' Hansen面板门槛模型（手动网格搜索）
#' @param df 数据框
#' @param y 因变量名
#' @param x 核心变量名（被门槛分割）
#' @param q 门槛变量名（可以是x本身）
#' @param controls 控制变量向量
#' @param entity_id 个体ID列名
#' @param time_id 时间ID列名
#' @param n_boot Bootstrap次数
hansen_threshold <- function(df, y, x, q, controls,
                              entity_id = "entity_id",
                              time_id   = "year",
                              n_boot    = 300) {

  # 候选门槛值（分位数网格）
  q_vals  <- quantile(df[[q]], probs = seq(0.15, 0.85, by = 0.01), na.rm = TRUE)
  q_vals  <- unique(q_vals)

  # 网格搜索：对每个候选门槛值计算SSR
  ssr_fun <- function(threshold) {
    df_tmp <- df %>%
      mutate(
        x_low  = ifelse(.data[[q]] <= threshold, .data[[x]], 0),
        x_high = ifelse(.data[[q]] >  threshold, .data[[x]], 0)
      )
    formula_str <- paste(y, "~", paste(c("x_low", "x_high", controls), collapse = " + "))
    tryCatch({
      mod <- plm(as.formula(formula_str),
                 data  = pdata.frame(df_tmp, index = c(entity_id, time_id)),
                 model = "within")
      sum(residuals(mod)^2)
    }, error = function(e) Inf)
  }

  ssr_vals <- sapply(q_vals, ssr_fun)
  best_idx <- which.min(ssr_vals)
  gamma_hat <- q_vals[best_idx]
  ssr_min  <- ssr_vals[best_idx]

  cat(sprintf("=== Hansen门槛模型结果 ===\n"))
  cat(sprintf("门槛估计值 γ̂ = %.4f\n", gamma_hat))
  cat(sprintf("SSR（门槛模型）= %.4f\n", ssr_min))

  # 对应的线性模型SSR（无门槛）
  formula_linear <- paste(y, "~", paste(c(x, controls), collapse = " + "))
  mod_linear <- plm(as.formula(formula_linear),
                    data  = pdata.frame(df, index = c(entity_id, time_id)),
                    model = "within")
  ssr_linear <- sum(residuals(mod_linear)^2)
  F_stat <- (ssr_linear - ssr_min) / (ssr_min / nrow(df))
  cat(sprintf("线性模型 SSR = %.4f\n", ssr_linear))
  cat(sprintf("F统计量 = %.4f\n", F_stat))

  # Bootstrap p值（非标准分布，需bootstrap）
  cat(sprintf("\n正在计算Bootstrap p值（%d次）...\n", n_boot))
  F_boot <- numeric(n_boot)
  for (b in seq_len(n_boot)) {
    df_boot <- df[sample(nrow(df), replace = TRUE), ]
    q_boot  <- quantile(df_boot[[q]], probs = seq(0.15, 0.85, by = 0.05), na.rm = TRUE)
    ssr_boot <- sapply(unique(q_boot), function(th) {
      tryCatch({
        df_b <- df_boot %>%
          mutate(x_low  = ifelse(.data[[q]] <= th, .data[[x]], 0),
                 x_high = ifelse(.data[[q]] >  th, .data[[x]], 0))
        m <- plm(as.formula(paste(y, "~", paste(c("x_low","x_high",controls), collapse="+"))),
                 data = pdata.frame(df_b, index = c(entity_id, time_id)), model = "within")
        sum(residuals(m)^2)
      }, error = function(e) Inf)
    })
    m_lin_b <- plm(as.formula(formula_linear),
                   data = pdata.frame(df_boot, index = c(entity_id, time_id)), model = "within")
    F_boot[b] <- (sum(residuals(m_lin_b)^2) - min(ssr_boot)) /
                  (min(ssr_boot) / nrow(df_boot))
  }
  p_val <- mean(F_boot >= F_stat, na.rm = TRUE)
  cat(sprintf("Bootstrap p值 = %.4f\n", p_val))
  if (p_val < 0.05) cat("✓ 门槛效应在5%%水平显著\n") else cat("⚠️ 门槛效应不显著\n")

  # 返回完整结果
  list(gamma = gamma_hat, F = F_stat, p_boot = p_val, ssr_grid = data.frame(q = q_vals, ssr = ssr_vals))
}

# 使用示例
result_thresh <- hansen_threshold(
  df, y = "y", x = "x1", q = "x1",
  controls = c("control1", "control2"), n_boot = 300
)

# 可视化：SSR vs 候选门槛值
library(ggplot2)
ggplot(result_thresh$ssr_grid, aes(x = q, y = ssr)) +
  geom_line(color = "#1976D2") +
  geom_vline(xintercept = result_thresh$gamma, color = "red", linetype = "dashed") +
  labs(title = "Hansen门槛搜索：SSR随候选门槛值变化",
       x = "候选门槛值 γ", y = "残差平方和（SSR）",
       subtitle = sprintf("最优门槛 γ̂ = %.4f（红色虚线）", result_thresh$gamma)) +
  theme_bw()
ggsave("output/hansen_threshold_search.png", dpi = 300, width = 8, height = 5)
```

---

### Section: MDE统计功效分析

最小可检测效应（Minimum Detectable Effect, MDE）：给定样本量、显著性水平和统计功效，能够检验出的最小真实效应。

**核心逻辑：** 若MDE > 文献中合理效应量 → 预警"数据可能不够有力，即使存在真实效应也难以检验出"。

$$MDE = (z_{1-\alpha/2} + z_{1-\kappa}) \times \sqrt{\frac{\sigma^2}{N_{treat}} + \frac{\sigma^2}{N_{control}}}$$

含聚类修正：
$$MDE_{cluster} = MDE \times \sqrt{1 + (n_{cluster\_size} - 1) \times ICC}$$

#### R代码

```r
# R: MDE统计功效分析
library(dplyr)

#' 计算面板/聚类数据的MDE
#' @param n_treat    处理组样本量
#' @param n_control  对照组样本量
#' @param sigma      因变量标准差
#' @param alpha      显著性水平（默认0.05）
#' @param power      统计功效（默认0.80）
#' @param n_clusters 聚类数量（NULL=不聚类）
#' @param cluster_size 每个聚类平均样本量
#' @param icc        组内相关系数（Intraclass Correlation Coefficient）
mde_calc <- function(
  n_treat, n_control,
  sigma,
  alpha        = 0.05,
  power        = 0.80,
  n_clusters   = NULL,
  cluster_size = NULL,
  icc          = NULL,
  two_sided    = TRUE,
  lit_effect   = NULL     # 文献中合理效应量（用于对比）
) {
  if (two_sided) alpha <- alpha / 2

  z_alpha <- qnorm(1 - alpha)
  z_power <- qnorm(power)

  # 基础MDE（不聚类）
  mde_base <- (z_alpha + z_power) * sigma * sqrt(1/n_treat + 1/n_control)

  # 聚类修正（DEFF: Design Effect）
  mde_cluster <- mde_base
  deff        <- 1
  if (!is.null(icc) && !is.null(cluster_size)) {
    deff        <- 1 + (cluster_size - 1) * icc
    mde_cluster <- mde_base * sqrt(deff)
  }

  # 标准化效应（Cohen's d）
  mde_cohen_d <- mde_cluster / sigma

  cat("=== MDE统计功效分析 ===\n")
  cat(sprintf("设定: N_treat=%d, N_control=%d, σ=%.4f, α=%.2f, 功效=%.2f\n",
              n_treat, n_control, sigma, alpha*ifelse(two_sided,2,1), power))
  if (!is.null(icc)) {
    cat(sprintf("聚类修正: cluster_size=%g, ICC=%.3f, DEFF=%.3f\n",
                cluster_size, icc, deff))
  }
  cat(sprintf("\nMDE（原始单位）: %.4f\n", mde_cluster))
  cat(sprintf("MDE（Cohen's d）: %.4f\n", mde_cohen_d))
  cat(sprintf("MDE（因变量均值的%%）: 需提供均值计算\n"))

  if (!is.null(lit_effect)) {
    cat(sprintf("\n文献合理效应量: %.4f\n", lit_effect))
    if (mde_cluster > abs(lit_effect)) {
      cat("⚠️ 预警：MDE > 文献效应量！数据可能不够有力\n")
      cat("   即使真实存在文献量级的效应，当前样本也难以检测出\n")
      cat("   建议：扩大样本 / 降低显著性要求 / 考虑功效分析驱动的样本设计\n")
    } else {
      cat("✓ MDE < 文献效应量，样本量足以检测合理效应\n")
    }
  }

  # 功效曲线（不同真实效应 vs 功效）
  effect_range <- seq(0, mde_cluster * 3, length.out = 100)
  power_curve  <- pnorm(effect_range / (sigma * sqrt(1/n_treat + 1/n_control)) - z_alpha) +
                  pnorm(-effect_range / (sigma * sqrt(1/n_treat + 1/n_control)) - z_alpha)

  return(invisible(list(
    mde        = mde_cluster,
    mde_base   = mde_base,
    deff       = deff,
    cohen_d    = mde_cohen_d,
    power_data = data.frame(effect = effect_range, power = power_curve)
  )))
}

# 使用示例
n_total  <- nrow(df)
n_treat  <- sum(df$treated == 1)
n_ctrl   <- sum(df$treated == 0)
y_sigma  <- sd(df$y, na.rm = TRUE)
n_clust  <- n_distinct(df$province_id)   # 聚类数

# 计算聚类层面的ICC
library(lme4)
icc_model <- lmer(y ~ 1 + (1 | province_id), data = df)
icc_val   <- as.numeric(VarCorr(icc_model)$province_id) /
              (as.numeric(VarCorr(icc_model)$province_id) + sigma(icc_model)^2)

mde_result <- mde_calc(
  n_treat      = n_treat,
  n_control    = n_ctrl,
  sigma        = y_sigma,
  icc          = icc_val,
  cluster_size = n_total / n_clust,
  lit_effect   = 0.05 * mean(df$y, na.rm = TRUE)  # 文献中5%效应量
)

# 可视化功效曲线
library(ggplot2)
ggplot(mde_result$power_data, aes(x = effect, y = power)) +
  geom_line(color = "#1976D2", linewidth = 1) +
  geom_hline(yintercept = 0.80, linetype = "dashed", color = "red") +
  geom_vline(xintercept = mde_result$mde, linetype = "dashed", color = "orange") +
  labs(title  = "统计功效曲线",
       x      = "真实效应大小",
       y      = "统计功效（1-β）",
       caption = sprintf("红线=80%%功效水平；橙线=MDE=%.4f", mde_result$mde)) +
  theme_bw()
ggsave("output/power_curve.png", dpi = 300, width = 8, height = 5)
```

---

### Section: PSM完善

#### 半径匹配 + 核匹配（除最近邻外）

```r
# R: 多种PSM匹配方法（MatchIt包）
library(MatchIt)
library(cobalt)
library(ggplot2)

# ============================================================
# Step 1: 估计倾向得分
# ============================================================
ps_formula <- treated ~ control1 + control2 + control3 + control4 + control5

# 方法A: 最近邻匹配（1:1，有替换）
match_nn <- matchit(
  ps_formula, data = df,
  method   = "nearest",
  distance = "logit",
  ratio    = 1,
  replace  = FALSE
)

# 方法B: 半径（Caliper）匹配
# caliper=0.1 表示倾向得分差距不超过0.1个标准差
match_caliper <- matchit(
  ps_formula, data = df,
  method   = "nearest",
  distance = "logit",
  caliper  = 0.1,        # 半径（logit-PS的0.1 SD）
  std.caliper = TRUE
)

# 方法C: 核匹配（Kernel Matching）
match_kernel <- matchit(
  ps_formula, data = df,
  method   = "full",     # 完全匹配（近似核匹配）
  distance = "logit"
)

# 汇总各方法匹配效果
cat("=== 各匹配方法样本量 ===\n")
cat(sprintf("最近邻: 处理组=%d, 对照组=%d\n",
            sum(match_nn$weights[df$treated==1] > 0),
            sum(match_nn$weights[df$treated==0] > 0)))
cat(sprintf("半径(caliper=0.1): 处理组=%d, 对照组=%d\n",
            sum(match_caliper$weights[df$treated==1] > 0),
            sum(match_caliper$weights[df$treated==0] > 0)))

# ============================================================
# Step 2: 平衡性检验 Love Plot（cobalt包）
# ============================================================

# 未匹配 vs 匹配后对比
bal_table <- bal.tab(
  match_nn,
  stats   = c("mean.diffs", "variance.ratios"),
  thresholds = c(m = 0.1, v = 2),   # m<0.1 且 VR∈[0.5,2]为平衡
  un      = TRUE                     # 同时显示匹配前
)
print(bal_table)

# Love Plot（标准化均值差可视化）
p_love <- love.plot(
  match_nn,
  stats     = "mean.diffs",
  threshold = 0.1,                   # 平衡性阈值（|SMD| < 0.1）
  binary    = "std",
  abs       = TRUE,
  var.order = "unadjusted",
  colors    = c("#E57373", "#42A5F5"),
  shapes    = c("circle", "triangle"),
  title     = "协变量平衡性 Love Plot（|标准化均值差|）"
)
ggsave("output/psm_love_plot.png", p_love, width = 10, height = 7, dpi = 300)

# ============================================================
# Step 3: 共同支撑域图（处理组/对照组PS分布对比）
# ============================================================
df$ps_score <- match_nn$distance   # 倾向得分

p_support <- ggplot(df, aes(x = ps_score, fill = factor(treated))) +
  geom_histogram(data = df %>% filter(treated == 1),
                 aes(y =  after_stat(count)), bins = 40, alpha = 0.7) +
  geom_histogram(data = df %>% filter(treated == 0),
                 aes(y = -after_stat(count)), bins = 40, alpha = 0.7) +
  scale_fill_manual(values = c("#42A5F5","#E57373"),
                    labels = c("对照组","处理组"), name="") +
  geom_hline(yintercept = 0, linewidth = 0.5) +
  labs(title = "倾向得分分布：共同支撑域检验",
       x = "倾向得分（logit scale）",
       y = "频数（处理组↑ / 对照组↓）") +
  theme_bw(base_size = 12)
ggsave("output/psm_common_support.png", p_support, width = 10, height = 6, dpi = 300)

# ============================================================
# Step 4: 匹配后ATT估计
# ============================================================
df_matched <- match.data(match_nn)

# 回归估计ATT（含聚类SE）
library(fixest)
att_est <- feols(
  y ~ treated + control1 + control2 + control3,
  data    = df_matched,
  weights = ~weights,
  cluster = ~entity_id
)
summary(att_est)
# 核心系数 "treated" 即为PSM-ATT
```

---

### Section: ppmlhdfe

泊松伪极大似然估计（PPML / PPMLHDFE）：当因变量含**大量零值**（如贸易流量、专利数、投资额）时，log-OLS会因log(0)丢失样本且产生偏误，应使用PPML。

**适用场景：**
- 因变量 ≥ 0 且含大量零（零值比例 > 30%）
- 贸易引力模型（贸易流量大量为零）
- 创新产出（专利申请/授权数，大量企业为零）
- FDI流量、融资额等右偏+零堆积变量

**优势：**
- 不丢弃零值样本
- 对异方差稳健
- 允许半对数解释（系数可解释为弹性）

#### R代码（fixest fepois）

```r
# R: PPML高维固定效应（fixest::fepois）
library(fixest)

# ============================================================
# Step 1: 检查是否需要PPML
# ============================================================
cat("=== 因变量分布检验 ===\n")
cat(sprintf("零值比例: %.1f%%\n", mean(df$y == 0, na.rm=TRUE) * 100))
cat(sprintf("均值: %.4f, 方差: %.4f\n", mean(df$y, na.rm=TRUE), var(df$y, na.rm=TRUE)))
cat(sprintf("方差/均值 = %.2f（>1说明过度离散）\n", var(df$y, na.rm=TRUE)/mean(df$y, na.rm=TRUE)))

# ============================================================
# Step 2: PPML估计（fepois = Poisson FE）
# ============================================================

# 基础PPML（无固定效应）
ppml_base <- fepois(
  y ~ x1 + x2 + control1 + control2,
  data    = df,
  vcov    = "hetero"          # 异方差稳健SE（PPML标准做法）
)

# 含固定效应的PPML（高维）
ppml_fe <- fepois(
  y ~ x1 + x2 + control1 + control2 | entity_id + year,
  data    = df,
  cluster = ~entity_id        # 聚类SE
)

# 含交互固定效应
ppml_hdfe <- fepois(
  y ~ x1 + x2 + control1 + control2 | entity_id + year + industry^year,
  data    = df,
  cluster = ~entity_id
)

# 汇总结果
etable(
  ppml_base, ppml_fe, ppml_hdfe,
  title   = "PPML估计结果",
  headers = c("基础PPML", "+个体/年FE", "+行业×年FE"),
  fitstat = ~ pr2 + n   # 伪R²
)

# ============================================================
# Step 3: 系数解读（半对数弹性）
# ============================================================
coef_x1 <- coef(ppml_fe)["x1"]
cat(sprintf("\nPPML系数解读:\n"))
cat(sprintf("  x1系数 = %.4f\n", coef_x1))
cat(sprintf("  → x1增加1单位，y的条件期望变化: (exp(%.4f)-1)×100%% = %.2f%%\n",
            coef_x1, (exp(coef_x1) - 1) * 100))

# ============================================================
# Step 4: PPML vs log-OLS 对比（稳健性）
# ============================================================
# Log-OLS（仅保留y>0样本，对比用）
df_pos <- df %>% filter(y > 0)
ols_log <- feols(log(y) ~ x1 + x2 + control1 + control2 | entity_id + year,
                 data = df_pos, cluster = ~entity_id)

cat("\n=== PPML vs log-OLS 对比 ===\n")
cat(sprintf("PPML (全样本, N=%d): x1系数 = %.4f\n", nrow(df), coef(ppml_fe)["x1"]))
cat(sprintf("log-OLS (仅y>0, N=%d): x1系数 = %.4f\n", nrow(df_pos), coef(ols_log)["x1"]))
cat(sprintf("丢弃零值样本: %d (%.1f%%)\n", nrow(df)-nrow(df_pos), (nrow(df)-nrow(df_pos))/nrow(df)*100))
```

---

### Section: Estimand声明

本节汇总面板数据各方法的Estimand声明要求。

| 方法 | Estimand | 声明要求 |
|------|---------|---------|
| PSM-DID | ATT（处理组的平均处理效应） | 标注共同支撑域损失样本数量及比例 |
| Panel FE（TWFE） | 组内变异识别的效应（加性ATT） | 明确"效应由组内变异识别"，注明within R² |
| 动态面板GMM | 短期效应β + 长期效应β/(1-ρ) | 必须区分短期和长期，|ρ|<1才能计算长期 |
| PPML | 条件期望的乘法效应 | 系数解读为 (exp(β)-1)×100%，非线性 |
| 门槛模型 | 门槛两侧的分段效应 | 报告两侧系数及门槛估计值和置信区间 |

**PSM-DID Estimand声明模板：**
> "本文PSM-DID估计的Estimand为处理组的平均处理效应（ATT）。倾向得分匹配后保留共同支撑域内样本，共损失 X 个观测（占原始样本的 X.X%）。PSM+DID的识别需同时满足条件独立假设（CIA）和平行趋势假设。"

**Panel FE Estimand声明模板：**
> "固定效应估计利用个体内（within）跨期变异识别效应，Estimand为以组内变异加权的平均处理效应。固定效应控制了不随时间变化的个体异质性，但无法排除时变的遗漏变量。"

```r
# R: 自动生成Estimand声明（PSM示例）
generate_psm_estimand <- function(df_orig, df_matched, treated_col = "treated") {
  n_orig     <- nrow(df_orig)
  n_matched  <- nrow(df_matched)
  n_lost     <- n_orig - n_matched
  pct_lost   <- n_lost / n_orig * 100

  n_treat_matched <- sum(df_matched[[treated_col]] == 1)
  n_ctrl_matched  <- sum(df_matched[[treated_col]] == 0)

  cat("=== PSM Estimand声明 ===\n")
  cat(sprintf("目标Estimand: ATT（处理组平均处理效应）\n"))
  cat(sprintf("原始样本: %d（处理组: %d, 对照组: %d）\n",
              n_orig,
              sum(df_orig[[treated_col]] == 1),
              sum(df_orig[[treated_col]] == 0)))
  cat(sprintf("匹配后样本: %d（处理组: %d, 对照组: %d）\n",
              n_matched, n_treat_matched, n_ctrl_matched))
  cat(sprintf("共同支撑域损失: %d个观测（%.1f%%）\n", n_lost, pct_lost))

  if (pct_lost > 20) {
    cat("⚠️ 警告：超过20%%样本被排除，ATT估计的外部有效性受限\n")
    cat("   建议：检查处理组和对照组的基础特征差异，说明损失样本的特征\n")
  }
}
```
