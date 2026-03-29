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
