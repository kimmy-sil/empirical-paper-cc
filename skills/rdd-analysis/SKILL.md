# RDD 分析 — 断点回归设计 (Regression Discontinuity Design)

## 概述

断点回归利用个体在某个"评分变量"（running variable / forcing variable）上的阈值（cutoff）来识别因果效应。在阈值附近的个体因为接近随机分配，具有很高的内部效度。

**适用场景：**
- 处理分配基于某个连续变量（评分变量）是否超过阈值
- 示例：考试分数超过 600 分获得奖学金、企业规模超过 50 人须遵守某法规、地方官员投票超过 50% 当选
- Sharp RDD：超过阈值 100% 被处理
- Fuzzy RDD：超过阈值大幅提高处理概率（不完全服从），此时 RDD 类似 IV

---

## 前置条件

### 数据结构要求

```
必须包含：
  - 结果变量 Y（outcome）
  - 评分变量 R（running variable），已知阈值 c
  - 处理变量 D（0/1），Sharp RDD 中 D = 1(R ≥ c)
  - 协变量 X（controls，用于协变量平衡检验）

推荐：
  - 将评分变量中心化：r = R - c，使阈值在 0 处
  - 检查是否有评分变量操控迹象（McCrary test）
```

### 中心化

```python
# Python: 评分变量中心化
df['r_centered'] = df['running_var'] - cutoff
df['above_cutoff'] = (df['r_centered'] >= 0).astype(int)
```

```r
# R
df <- df %>%
  mutate(r_centered = running_var - cutoff,
         above_cutoff = as.integer(r_centered >= 0))
```

---

## 分析步骤

### Step 1：可视化断点

绘制断点散点图，直观确认断点效应存在。

```python
# Python: RDD 可视化（分箱散点图）
import numpy as np
import matplotlib.pyplot as plt

bins = np.arange(df['r_centered'].min(), df['r_centered'].max(), bandwidth/10)
df['bin'] = pd.cut(df['r_centered'], bins=bins)
bin_means = df.groupby('bin').agg(
    r_mid=('r_centered', 'mean'),
    y_mean=('outcome', 'mean')
).reset_index()

fig, ax = plt.subplots(figsize=(8, 5))
left  = bin_means[bin_means['r_mid'] < 0]
right = bin_means[bin_means['r_mid'] >= 0]
ax.scatter(left['r_mid'],  left['y_mean'],  color='blue', label='Control', alpha=0.7)
ax.scatter(right['r_mid'], right['y_mean'], color='red',  label='Treated', alpha=0.7)
# 拟合局部多项式曲线
from numpy.polynomial import polynomial as P
for side, color in [(left, 'blue'), (right, 'red')]:
    x = side['r_mid'].values
    y = side['y_mean'].values
    coef = P.polyfit(x, y, deg=1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, P.polyval(x_line, coef), color=color)
ax.axvline(0, color='black', linestyle='--')
ax.set_xlabel('Running Variable (centered)')
ax.set_ylabel('Outcome')
ax.set_title('RDD: Visual Discontinuity')
ax.legend()
plt.savefig('output/rdd_main_plot.png', dpi=150)
```

```r
# R: rdplot（rdrobust 包）
library(rdrobust)

rdplot(y   = df$outcome,
       x   = df$r_centered,
       c   = 0,
       title = "RDD Plot",
       x.label = "Running Variable (centered)",
       y.label = "Outcome")
```

---

### Step 2：最优带宽选择（Bandwidth Selection）

带宽选择是 RDD 的核心——太窄方差大，太宽偏误大。推荐使用 `rdrobust` 的 MSE 最优带宽（Calonico, Cattaneo, Titiunik 2014）。

```python
# Python: rdrobust
from rdrobust import rdrobust, rdbwselect

# 最优带宽选择
bw = rdbwselect(y=df['outcome'].values, x=df['r_centered'].values, c=0, p=1)
print(bw.summary())
# 主带宽 h, 偏误修正带宽 b

# 主估计（局部线性，MSE最优带宽，bias-corrected robust CI）
rdd_res = rdrobust(
    y=df['outcome'].values,
    x=df['r_centered'].values,
    c=0,
    p=1,      # 局部线性（推荐）
    kernel='triangular',  # 三角核（推荐）
    bwselect='mserd'      # MSE-optimal
)
print(rdd_res.summary())
```

```r
# R: rdrobust（标准方法）
library(rdrobust)

# 主估计（bias-corrected + robust CI，即 rbc 列）
rdd_main <- rdrobust(
  y        = df$outcome,
  x        = df$r_centered,
  c        = 0,
  p        = 1,          # 局部线性
  kernel   = "triangular",
  bwselect = "mserd"     # MSE-optimal bandwidth
)
summary(rdd_main)
# 报告: Conventional, Bias-corrected, Robust 三列
# 推荐报告: Robust (bias-corrected robust) 的置信区间
```

```stata
* Stata: rdrobust
ssc install rdrobust

rdrobust outcome r_centered, c(0) p(1) kernel(triangular) bwselect(mserd)
* 重点: "Robust" 列的系数和 CI
```

---

### Step 3：McCrary 密度检验（操控检验）

检验评分变量在阈值处是否存在堆积（manipulation），即个体是否能操控自己的评分来进入处理组。

```python
# Python: rddensity
from rddensity import rddensity, rdplotdensity

density_test = rddensity(X=df['r_centered'].values, c=0)
print(density_test.summary())
# H0: 无操控（密度在断点处连续）
# p > 0.10 通过检验

# 可视化
rdplotdensity(density_test, X=df['r_centered'].values)
plt.savefig('output/rdd_density_test.png', dpi=150)
```

```r
# R: rddensity（Cattaneo et al. 2018，优于 McCrary 2008）
library(rddensity)

density_res <- rddensity(X = df$r_centered, c = 0)
summary(density_res)  # 看 p-value: H0=连续（无操控）

# 密度图
rdplotdensity(density_res, X = df$r_centered,
              title = "Density Test: No Manipulation")
```

```stata
* Stata: rddensity（优于 DCdensity）
ssc install rddensity

rddensity r_centered, c(0) plot
* p > 0.10 → 无操控证据
```

**注意：** McCrary (2008) 的 DCdensity 检验已被 Cattaneo et al. (2018) 的 rddensity 取代，后者在有限样本下表现更好。

---

### Step 4：协变量平衡检验（Covariate Balance）

在断点处检验前定协变量是否连续（不应有跳跃）。若协变量在断点处显著跳跃，说明样本可能在阈值处被选择。

```r
# R: 对每个协变量做 RDD，期望系数不显著
covariate_balance <- lapply(covariates, function(cov) {
  res <- rdrobust(y = df[[cov]], x = df$r_centered, c = 0, p = 1, bwselect = "mserd")
  data.frame(
    covariate = cov,
    coef      = res$coef["Conventional", ],
    se        = res$se["Conventional", ],
    p_val     = res$pv["Robust", ]
  )
}) %>% bind_rows()

print(covariate_balance)
# 所有协变量的 p > 0.10 → 平衡良好
```

```python
# Python: 协变量平衡
balance_results = []
for cov in covariates:
    res = rdrobust(y=df[cov].values, x=df['r_centered'].values, c=0, p=1)
    balance_results.append({
        'covariate': cov,
        'coef': res.coef.loc['Conventional'].values[0],
        'p_robust': res.pv.loc['Robust'].values[0]
    })
pd.DataFrame(balance_results).to_csv('output/rdd_covariate_balance.csv')
```

---

### Step 5：主估计 — Sharp RDD

```r
# R: 多种规格比较
specs <- list()
for (p in 1:3) {  # 局部线性, 二次, 三次
  specs[[p]] <- rdrobust(
    y = df$outcome, x = df$r_centered,
    c = 0, p = p, kernel = "triangular", bwselect = "mserd"
  )
}

# 提取结果
results_df <- data.frame(
  spec   = c("Local Linear", "Local Quadratic", "Local Cubic"),
  coef   = sapply(specs, function(r) r$coef["Bias-Corrected", 1]),
  ci_low = sapply(specs, function(r) r$ci["Robust", 1]),
  ci_hi  = sapply(specs, function(r) r$ci["Robust", 2]),
  h      = sapply(specs, function(r) r$bws["h", 1])  # 带宽
)
print(results_df)
```

---

### Step 6：Fuzzy RDD（不完全服从）

当阈值只是提高而非完全决定处理概率时，使用 Fuzzy RDD（等价于在断点附近做 IV）。

```r
# R: Fuzzy RDD
# fuzzy 参数传入实际处理变量（非 above_cutoff）
rdd_fuzzy <- rdrobust(
  y     = df$outcome,
  x     = df$r_centered,
  c     = 0,
  fuzzy = df$actual_treatment,   # 实际处理 D（不等于阈值规则）
  p     = 1,
  bwselect = "mserd"
)
summary(rdd_fuzzy)
# 报告的是 Fuzzy RD（LATE at the cutoff）

# 第一阶段（discontinuity in treatment）
rdd_first_stage <- rdrobust(
  y = df$actual_treatment,
  x = df$r_centered,
  c = 0, p = 1
)
cat("First Stage Discontinuity:", rdd_first_stage$coef["Conventional",1], "\n")
```

```stata
* Stata: Fuzzy RDD
rdrobust outcome r_centered, c(0) fuzzy(actual_treatment) p(1)
```

---

### Step 7：稳健性检验

#### 7a. 带宽敏感性

```r
# R: 多带宽稳健性
h_main <- rdd_main$bws["h", 1]   # 主带宽

bw_sensitivity <- lapply(c(0.5, 0.75, 1.0, 1.25, 1.5) * h_main, function(h) {
  res <- rdrobust(y = df$outcome, x = df$r_centered, c = 0, p = 1, h = h)
  data.frame(
    bandwidth = h,
    coef      = res$coef["Bias-Corrected", 1],
    ci_low    = res$ci["Robust", 1],
    ci_hi     = res$ci["Robust", 2],
    n_left    = res$N_h[1],
    n_right   = res$N_h[2]
  )
}) %>% bind_rows()

# 系数图：横轴带宽，纵轴估计量（含CI）
ggplot(bw_sensitivity, aes(x = bandwidth, y = coef)) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_hi), alpha = 0.2) +
  geom_line() + geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Bandwidth Sensitivity", x = "Bandwidth", y = "RDD Estimate") +
  theme_minimal()
```

#### 7b. 安慰剂断点检验（Placebo Cutoffs）

```r
# R: 在假断点处估计（应不显著）
fake_cutoffs <- c(-2, -1, 1, 2)  # 在真断点两侧设置假断点

placebo_res <- lapply(fake_cutoffs, function(fake_c) {
  # 只用真断点一侧的数据
  sub_df <- df %>% filter(r_centered < 0)  # 仅控制侧
  res <- rdrobust(y = sub_df$outcome, x = sub_df$r_centered, c = fake_c, p = 1)
  data.frame(fake_cutoff = fake_c,
             coef = res$coef["Conventional",1],
             p_robust = res$pv["Robust",1])
}) %>% bind_rows()
print(placebo_res)
# 期望: 所有假断点处 p > 0.10
```

#### 7c. Donut Hole 检验

排除最接近阈值的观测（可能存在操控），检验结论是否稳健。

```r
# R: Donut hole（排除 |r| < donut_width 的观测）
donut_widths <- c(0.1, 0.2, 0.5)

donut_res <- lapply(donut_widths, function(d) {
  sub_df <- df %>% filter(abs(r_centered) >= d)
  res <- rdrobust(y = sub_df$outcome, x = sub_df$r_centered, c = 0, p = 1)
  data.frame(donut = d,
             coef  = res$coef["Bias-Corrected", 1],
             ci_low = res$ci["Robust", 1],
             ci_hi  = res$ci["Robust", 2])
}) %>% bind_rows()
print(donut_res)
```

---

## 必做检验清单

| 检验 | 方法 | 通过标准 |
|------|------|----------|
| 断点可视化 | rdplot | 肉眼可见跳跃 |
| 密度/操控检验 | rddensity | p > 0.10（无操控） |
| 协变量平衡 | 对每个协变量做 RDD | 所有 p > 0.10 |
| 主估计 | rdrobust（局部线性，MSE带宽） | 报告 Robust CI |
| 多项式阶数敏感性 | p = 1, 2, 3 | 结论稳定 |
| 带宽敏感性 | 0.5h ~ 1.5h | 系数方向一致 |
| 安慰剂断点 | 假 cutoff（两侧） | 系数不显著 |
| Donut hole | 排除 cutoff 附近观测 | 结论稳定 |

---

## 常见错误提醒

> **错误 1：直接用 OLS 加 above_cutoff 哑变量**
> 传统 OLS（如 `y ~ r + above + r*above`）的估计量和标准误均不可靠，带宽选择是 ad-hoc 的。应使用 `rdrobust` 的局部多项式估计，自动选择 MSE 最优带宽。

> **错误 2：报告 Conventional 估计量而非 Robust**
> rdrobust 输出三行：Conventional、Bias-corrected、Robust。正确报告顺序：**点估计用 Bias-corrected，置信区间用 Robust**（Calonico et al. 2014 的推荐）。不要只报告 Conventional。

> **错误 3：多项式阶数过高**
> 使用高阶多项式（p ≥ 4）是危险的——它们对断点附近以外的数据过于敏感，容易产生虚假的精确性。推荐局部线性（p=1）为主规格，p=2 作为稳健性。

> **错误 4：忽略操控检验**
> 如果主体（个体/企业）知道阈值并能操控评分变量，样本在阈值两侧不再"准随机"。必须报告密度检验。

> **错误 5：Fuzzy RDD 忘记报告第一阶段强度**
> Fuzzy RDD 本质是 IV，第一阶段（阈值处的处理概率跳跃）必须足够大（类似 IV 的 F 检验），需明确报告。

> **错误 6：混淆 RDD 的 LATE 解释**
> RDD 识别的是**阈值处（at the cutoff）**的局部处理效应，不能外推到评分变量远离阈值的个体。

---

## 输出规范

### 主结果表

```r
# R: 手动整理结果表
rdd_table <- data.frame(
  spec    = c("Local Linear (MSE-opt bw)", "Local Quadratic", "Local Linear (0.5h)", "Local Linear (1.5h)"),
  coef_bc = ...,   # Bias-corrected 系数
  ci_low  = ...,   # Robust CI 下界
  ci_hi   = ...,   # Robust CI 上界
  bw_h    = ...,   # 主带宽
  n_eff   = ...,   # 有效样本量（断点两侧各自）
  p       = ...    # 多项式阶数
)
```

**表格必须包含：**
1. 主规格（p=1，MSE最优带宽）
2. 多项式阶数变化的稳健性
3. 带宽变化的稳健性
4. 有效样本量（断点两侧）
5. CI 类型说明（bias-corrected robust）

### 文件命名

```
output/
  rdd_main_plot.png           # 断点散点图
  rdd_density_test.png        # 密度/操控检验图
  rdd_main_results.csv        # 主估计结果
  rdd_covariate_balance.csv   # 协变量平衡
  rdd_bandwidth_sensitivity.csv  # 带宽敏感性
  rdd_placebo_cutoffs.csv     # 安慰剂断点
  rdd_donut_hole.csv          # Donut hole 检验
```

---

## RDD 变体与方法论深度补充

### A. 五种断点类型分类

| 断点类型 | 代表场景 | 适用条件 | 核心假设 | 常见陷阱 | 代表文献 |
|----------|----------|----------|----------|----------|----------|
| **分数/声誉断点** | 高考分数线、录取分数线、信用评级阈值 | 评分规则公开、断点唯一且确定 | Sharp RDD: D = 1(R ≥ c)，无报名上造假操控 | 分数堆积（最常见）；跨年分数比较 | Lee (2008) 选举; Hoekstra (2009) 旗舰大学 |
| **年龄断点** | 退休年龄、养老金参保年龄、医保资格年龄 | 处理和取消分段实施；坐标嵎稳定 | Fuzzy RDD（选择性提前/延后参保） | 递归现象（情堅者最山值时玴年）；年龄测量误差 | Card et al. (2008) 医保; Angrist & Krueger (1991) |
| **时间断点（RDiT）** | 政策实施日、法律修改日、交易开放日 | 时间序列数据；政策在确定日期生效 | 潜在结果趋势可控、无其他政策同年交叉 | 时间序列自相关；季节性操控不当 | Hausman & Rapson (2018); Davis (2008) 污染 |
| **地理断点** | 秦岭-淦河线、行政边界、气候分界线 | 地理边界干净、边界两侧可比 | 到边界距离为评分变量；人口特征在边界处连续 | 边界吒合选择（商业选址、移民）；空间异质性 | Dell (2010) 内战; Huang et al. (2014) 秦岭-淦河 |
| **指标阈值** | AQI 阈值、资产负值线、财务门槛 | 阈值备巴且清晰已知；个体无法精确操控指标 | 指标评分无操控导致不连续；多阈值不产生交互 | 太多阈值陨阱（带宽不够）；安慰剂断点被拉低 | Chen et al. (2013) AQI; Grembi et al. (2016) 财务规则 |

---

### B. RDiT（时间断点回归，Regression Discontinuity in Time）

#### B1. 与标准 RDD 的关键区别

| 维度 | 标准截面 RDD | RDiT（时间 RDD） |
|------|--------------|-------------------|
| 评分变量 | 截面指标（考分、指标值） | 时间（日/周/月） |
| 操控检验 | McCrary 密度检验 | 不可直接应用（时间无法“堆积”）→ 方法文献不要求 |
| 主要威胁 | 操控、协变量不平衡 | 时间序列自相关、季节性、同时政策 |
| 内部效度假设 | 断点两侧连续性 | 政策实施日前后的潜在结果趋势可控 |
| 需要额外检验 | 协变量平衡，Donut hole | 自相关检验、季节性控制、预期效应检验 |

**要点提示：** 如果政策是渐进推行（phased-in）而非立即生效，RDiT 可能高估带宽或低估效应，需要单独处理。

#### B2. R 代码模板

```r
# RDiT: 时间断点回归模板
library(rdrobust)
library(fixest)
library(dplyr)
library(lubridate)   # 日期处理
library(tseries)     # 自相关检验

# ---- 第一步：构造时间评分变量 ----
# policy_date: 政策实施日（如 as.Date("2015-01-01")）
policy_date <- as.Date("2015-01-01")

df <- df %>%
  mutate(
    # 时间评分变量：政策日前后的日数（中心化）
    t_centered = as.numeric(date - policy_date),
    # 政策虚拟变量
    post_policy = as.integer(t_centered >= 0),
    # 季节性控制（月份虽然不完全解决季节性，但是一个起点）
    month_of_year = month(date),
    day_of_week   = wday(date)
  )

# ---- 第二步：自相关检验（内部效度关键）----
# 检验年内结果变量的自相关，验证时间序列假设是否成立
outcome_ts <- ts(df$outcome[order(df$date)], frequency = 365)
Box.test(outcome_ts, lag = 20, type = "Ljung-Box")  # H0: 无自相关

# 如果显著自相关，需要在模型中控制
df <- df %>%
  arrange(date) %>%
  mutate(outcome_lag1 = lag(outcome, 1))  # 加入滞后项

# ---- 第三步：季节性控制 ----
# 方法1：加入月份固定效应
# 方法2：用历史同时期（去年同期）作为控制
# 方法3： Fourier 项控制年内周期性
df <- df %>%
  mutate(
    sin_yearly  = sin(2 * pi * yday(date) / 365),
    cos_yearly  = cos(2 * pi * yday(date) / 365),
    sin_weekly  = sin(2 * pi * wday(date) / 7),
    cos_weekly  = cos(2 * pi * wday(date) / 7)
  )

# ---- 第四步：RDiT 主估计（rdrobust 应用于时间维度）----
# 以 t_centered 为评分变量
rdit_main <- rdrobust(
  y        = df$outcome,
  x        = df$t_centered,
  c        = 0,
  p        = 1,
  kernel   = "triangular",
  bwselect = "mserd"
)
summary(rdit_main)
# 注意：带宽选择会基于时间单位（如日），解释时要说明带宽单位

# ---- 第五步：带季节性控制的 RDiT（feols 方法）----
res_rdit_fe <- feols(
  outcome ~ t_centered + i(post_policy, t_centered) +  # 断点两侧不同斜率
    sin_yearly + cos_yearly + sin_weekly + cos_weekly | # Fourier 控制季节性
    month_of_year + day_of_week,                        # 月份和周内日固定效应
  data = df %>% filter(abs(t_centered) <= rdit_main$bws["h", 1])  # 限制带宽内
)
summary(res_rdit_fe)

# ---- 第六步：预期效应检验（应不显著）----
# 在政策实施日唤取若干天前，检验是否已有预期行为
df_pre_window <- df %>%
  filter(t_centered >= -60, t_centered < 0) %>%  # 政策前 60 天
  mutate(placebo_post = as.integer(t_centered >= -30))  # 假设 30 天前为假断点

res_anticipation <- feols(
  outcome ~ t_centered + i(placebo_post, t_centered) +
    sin_yearly + cos_yearly | month_of_year,
  data = df_pre_window
)
summary(res_anticipation)
# 期望： placebo_post 系数不显著（无预期效应）

# ---- 第七步：带宽敏感性（时间 RDD 特别重要）----
bw_days <- c(30, 60, 90, 180, 365)  # 不同日数带宽

rdit_bw_res <- lapply(bw_days, function(h) {
  res <- rdrobust(y = df$outcome, x = df$t_centered, c = 0, p = 1, h = h)
  data.frame(
    bw_days = h,
    coef    = res$coef["Bias-Corrected", 1],
    ci_low  = res$ci["Robust", 1],
    ci_hi   = res$ci["Robust", 2]
  )
}) %>% bind_rows()
print(rdit_bw_res)
```

**RDiT 额外检验应包含：**
1. 自相关检验（Ljung-Box / ADF）
2. 季节性控制（Fourier 项或月份固定效应）
3. 预期效应检验（政策日前窗口内假断点）
4. 同时政策检验（添加对操组/其他窗口赴 DiD 检验）

---

### C. 地理 RDD（Geographic / Spatial RDD）

#### C1. 方法框架

以地理边界（如行政边界、气候分界线、历史分割线）为断点，个体到边界的距离为评分变量：

$$R_i = \text{dist}(\text{location}_i, \text{boundary})$$

- $R_i > 0$：边界一侧（处理组）
- $R_i < 0$：边界另一侧（控制组）

**内部效度假设：** 边界两侧的个体除了受到不同制度/政策外，其他特征应该连续

**额外识别挑战：**
- 边界吧合选择（人口、企业在边界两侧分布不均衡）
- 空间溢出效应（spillovers）导致边界两侧潜在结果不完全独立
- 地理近邻内生性（地理近邻 = 经济联系密切）

#### C2. R 代码框架（sf + rdrobust）

```r
# 地理 RDD 完整流程
library(sf)         # 空间数据
library(rdrobust)   # RDD 估计
library(dplyr)
library(ggplot2)

# ---- 第一步：读入地理数据 ----
# df_points: 含个体地理坐标（经纬度）和结果变量
df_sf <- st_as_sf(df_points, coords = c("longitude", "latitude"), crs = 4326)

# 边界 shp 文件（如凝河-秦岭线、行政分界线）
boundary_line <- st_read("data/boundary_line.shp") %>%
  st_transform(crs = 4326)  # 确保坐标系一致

# ---- 第二步：计算到边界的符号距离 ----
# 第二步关键：距离需要带“方向”（哪侧为处理、哪侧为控制）

# 计算每个个体到边界直线距离
df_sf$dist_to_boundary <- as.numeric(
  st_distance(df_sf, st_union(boundary_line))  # 单位：米
) / 1000  # 转换为公里

# 确定方向：例如北方为处理组
# 方法：用点在大边界多边形内的判断
treatment_area <- st_read("data/treatment_region.shp") %>%
  st_transform(crs = 4326)

df_sf$treated <- as.integer(st_within(df_sf, treatment_area, sparse = FALSE)[, 1])

# 符号化距离：处理组为正，控制组为负
df_sf <- df_sf %>%
  mutate(r_geo = ifelse(treated == 1, dist_to_boundary, -dist_to_boundary))

df <- df_sf %>% st_drop_geometry()  # 去掉圆形信息，转为普通 df

# ---- 第三步：地理 RDD 可视化 ----
ggplot(df_sf) +
  geom_sf(aes(color = as.factor(treated)), alpha = 0.3, size = 0.5) +
  geom_sf(data = boundary_line, color = "black", linewidth = 1) +
  scale_color_manual(values = c("blue", "red"), labels = c("Control", "Treated")) +
  labs(title = "Geographic RDD: Sample Distribution",
       color = "Treatment Status") +
  theme_minimal()

# RDD 断点图（距离 vs 结果）
rdplot(y = df$outcome, x = df$r_geo, c = 0,
       title = "Geographic RDD Plot",
       x.label = "Distance to Boundary (km, signed)",
       y.label = "Outcome")

# ---- 第四步：主估计 ----
rdd_geo <- rdrobust(
  y        = df$outcome,
  x        = df$r_geo,
  c        = 0,
  p        = 1,
  kernel   = "triangular",
  bwselect = "mserd"
)
summary(rdd_geo)

# ---- 第五步：女0位特征连续性检验（边界两侧人口特征平衡）----
geo_balance <- lapply(geo_covariates, function(cov) {
  res <- rdrobust(y = df[[cov]], x = df$r_geo, c = 0, p = 1, bwselect = "mserd")
  data.frame(
    covariate = cov,
    coef      = res$coef["Conventional", 1],
    p_robust  = res$pv["Robust", 1]
  )
}) %>% bind_rows()

print(geo_balance)
# 期望：所有人口特征（年龄结构、心里收入、为业结构）系数不显著

# ---- 第六步：空间溢出检验 ----
# 颞边界维度（地理 RDD 特有）
df <- df %>%
  mutate(
    # 个体在边界上的“坐标”（投影到边界线上的位置）
    along_boundary = # ... GIS 计算投影坐标
  )

# 如果结果因"along"方向而异质，考虑加入 along_boundary 一阶项
res_geo_ctrl <- feols(
  outcome ~ r_geo + post_boundary + r_geo:post_boundary +
    along_boundary + along_boundary:post_boundary,  # 控制地理投影方向
  data = df %>% filter(abs(r_geo) <= rdd_geo$bws["h", 1])
)
summary(res_geo_ctrl)
```

**地理 RDD 额外检验：**

| 检验 | 方法 | 目的 |
|------|------|------|
| 边界两侧人口特征 | 协变量 RDD | 证明边界两侧人口特征连续 |
| 空间溢出评估 | 湛边界对控组延伸结果检验 | 处理和控制组在边界两侧是否相似 |
| 坷擔 | Donut hole（排除最近边界的个体） | 排除可能选择性迁居 |
| 安慰剂边界 | 用附近平行边界作为安慰剂 | 边界不应显示断点 |
| 大圆场控制 | 加入 along_boundary 项 | 控制边界展开方向的地理异质性 |

---

### D. 多门槛 RDD（Multi-Cutoff RDD）

#### D1. 适用场景

- 全国不同地区 / 学校有不同录取分数线
- 同一政策在不同年份变动阈值
- 指标类行业（如 AQI 预警级别：一级、2级、3级阈值不同）

**核心思路（Cattaneo et al. 2016）：**

将评分变量标准化到各自阈值：$\tilde{R}_{ic} = R_i - c_j（c_j 是个体 i 面对的阈值）$，再合并估计。

#### D2. R 代码（rdmc 包）

```r
# 安装
library(rdrobust)  # 包含 rdmc 函数
library(dplyr)

# ---- 情景一：不同地区的不同录取分数线 ----
# df 包含: running_var, outcome, school_id, cutoff（每所学校的录取分数线不同）

# 方法1： 标准化并合并（Normalisation-and-Pooling）
df_norm <- df %>%
  mutate(
    r_normalized = running_var - cutoff,  # 将每个个体的评分变量对Α自就的阈值中心化
    above_cutoff  = as.integer(r_normalized >= 0)
  )

# 合并后用 rdrobust（最简单的方法，但假设所有断点等同）
rdd_pooled <- rdrobust(
  y        = df_norm$outcome,
  x        = df_norm$r_normalized,
  c        = 0,
  p        = 1,
  bwselect = "mserd"
)
summary(rdd_pooled)  # 报告 pooled LATE

# ---- 方法2： rdmc（Cattaneo et al. 推荐方法）----
# 主要优势：允许带宽在不同断点间变化

# 准备数据：居分 cutoff 索引
# rdmc 要求：所有断点的列表
cutoffs <- unique(df$cutoff)
cat("Total cutoffs:", length(cutoffs), "\n")
print(sort(cutoffs))

# 运行 rdmc
rdd_mc <- rdmc(
  y = df_norm$outcome,   # 结果变量
  x = df_norm$r_normalized,  # 标准化后的评分变量
  c = sort(unique(df_norm$r_normalized[df_norm$above_cutoff == 1 & df_norm$r_normalized == 0]))  # 每个断点位置
  # 注：标准化后所有断点都在 0 处 → 就是 c=0 的 pooled 问题
)
# 如果每个断点不同，不要标准化，直接传入原始阈值
rdd_mc_raw <- rdmc(
  y = df$outcome,
  x = df$running_var,
  c = sort(unique(df$cutoff))  # 每个个体对应的阈值
)
summary(rdd_mc_raw)
# 输出：每个断点的个体 RDD 估计 + 加权平均（pooled estimate）

# ---- 方法3：分断点单独估计 + 概证平均 ----
# 对每个阈值单独做 RDD
by_cutoff <- lapply(cutoffs, function(c_val) {
  sub_df <- df %>% filter(cutoff == c_val) %>%
    mutate(r_c = running_var - c_val)
  
  if (sum(sub_df$r_c >= 0) < 20 | sum(sub_df$r_c < 0) < 20) {
    return(NULL)  # 带宽内样本量不足，跳过
  }
  
  res <- rdrobust(y = sub_df$outcome, x = sub_df$r_c, c = 0, p = 1)
  data.frame(
    cutoff    = c_val,
    coef_bc   = res$coef["Bias-Corrected", 1],
    ci_low    = res$ci["Robust", 1],
    ci_hi     = res$ci["Robust", 2],
    n_left    = res$N_h[1],
    n_right   = res$N_h[2]
  )
}) %>% bind_rows()

print(by_cutoff)

# 绘制多断点系数图
ggplot(by_cutoff, aes(x = as.factor(cutoff), y = coef_bc)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_hi), width = 0.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray") +
  labs(
    title = "RDD Estimates by Cutoff",
    x     = "Cutoff Value",
    y     = "RDD Estimate (Bias-corrected Robust CI)"
  ) +
  coord_flip() +
  theme_minimal()

# ---- 方法4：异质性检验（各断点效应是否相同）----
# Wald 检验：各断点之间的差异是否显著
if (nrow(by_cutoff) >= 2) {
  # 简单估计各断点系数的加权平均
  w <- 1 / ((by_cutoff$ci_hi - by_cutoff$ci_low) / (2 * 1.96))^2  # 逆方差加权
  pooled_est <- weighted.mean(by_cutoff$coef_bc, w)
  cat(sprintf("\nPooled RDD estimate (inverse-variance weighted): %.4f\n", pooled_est))
  cat("Individual cutoff estimates:\n")
  print(by_cutoff[, c("cutoff", "coef_bc", "ci_low", "ci_hi")])
}
```

**Stata 代码（rdmulti）：**

```stata
* 安装
* ssc install rdrobust   (* 包含 rdmc rdplot rdrobust rdmulti）

* ---- 方法1： rdmc 多断点 RDD ----
* 数据需要：审分变量 r、结果 y、每个个体对应的 cutoff 变量 c

* 标准化并合并（全部对各自阈值中心化）
gen r_norm = running_var - cutoff

* Pooled RDD
rdrobust outcome r_norm, c(0) p(1) kernel(triangular) bwselect(mserd)

* ---- 方法2： rdmc（Cattaneo-Titiunik-Vazquez-Bare 2020）----
* 需要对每个断点单独提供维度
rdmc outcome running_var, c(cutoff_list) p(1) bwselect(mserd)
* cutoff_list = 空格分隔的各个阈值值（如 "60 70 80 90"）
* 输出：每个断点的单独估计 + 加权平均

* ---- 方法3： 多阈值切点的安慰剂检验 ----
* 对每个断点，在其两侧各设一个安慰剂断点
foreach c of numlist 60 70 80 90 {
    * 假断点往左企移 5 分
    local fake_c = `c' - 5
    qui rdrobust outcome running_var if running_var < `c', c(`fake_c') p(1)
    scalar placebo_coef_`c' = e(coef)
    scalar placebo_pv_`c'   = e(pv_rb)
}
```

**多门槛 RDD 实践建议：**

| 问题 | 建议 |
|------|------|
| 各断点样本量差异很大 | 用逆方差加权合并 | 
| 断点特征不同（如不同地区） | 分断点报告 + 差异性横截面回归 |
| 担心分断点检验样本量不足 | 每个断点至少需要 200+ 个样本 | 
| 各断点和并后报告哪个 | 主估计用 pooled，武为稳健性单独一一列出 |

---

### RD-DD 方法（断点 + 双重差分结合）

**适用场景：** 断点附近存在**随时间变化的混杂因素**（time-varying confounders），仅靠截面 RDD 无法控制，需要结合 DID 吸收时间趋势。

**典型情况：**
- 政策同时在时间上发生（某年某月），且在阈值两侧分配（评分 ≥ 阈值才受到政策）
- 经济冲击与断点在时间上相关，可能影响阈值两侧（pure RDD 无法控制）
- 需要控制个体/单元固定效应（消除时不变混淆）

**与标准 RDD 和 DID 的对比：**

| 方法 | 识别变异 | 控制 | 限制 |
|------|---------|------|------|
| 标准 RDD | 截面：阈值两侧横向比较 | 无时间趋势控制 | 无法控制时变混淆 |
| 标准 DID | 时间：处理前后纵向比较 | 无空间/分数控制 | 需要平行趋势 |
| RD-DD | 阈值 × 时间双重变异 | 同时控制 FE + 评分多项式 | 需要面板数据 + 双重假设 |

**识别假设：**
1. **RDD 假设：** 阈值附近连续性（无精确操控评分变量）
2. **DID 假设：** 阈值两侧的时间趋势在政策前平行（条件平行趋势）

```r
# R: RD-DD 代码模板（rdrobust + fixest 结合）
library(rdrobust)
library(fixest)
library(dplyr)

# ---- 数据结构要求 ----
# 面板数据：单元 × 时间
# 必须包含：评分变量 r_centered、时间虚拟 post、处理虚拟 above_cutoff

df <- df %>%
  mutate(
    r_centered   = running_var - cutoff,
    above_cutoff = as.integer(r_centered >= 0),
    # 交互项：处理指示 × 政策后哑变量（RD-DD 识别变量）
    rddd_treat   = above_cutoff * post
  )

# ---- Step 1：确定最优带宽（仅用截面 RDD，基于政策前数据）----
bw_ref <- rdrobust(
  y = df$outcome[df$post == 0],
  x = df$r_centered[df$post == 0],
  c = 0, p = 1, bwselect = "mserd"
)
h_opt <- bw_ref$bws["h", 1]
cat(sprintf("RDD 最优带宽: %.4f\n", h_opt))

# ---- Step 2：限制带宽内样本 ----
df_bw <- df %>% filter(abs(r_centered) <= h_opt)

# ---- Step 3：RD-DD 主估计 ----
# 模型：Y_it = α + β·(above × post) + f(r)·post + f(r) + FE + ε
# 其中 f(r) 为评分变量的局部多项式
res_rddd <- feols(
  outcome ~ rddd_treat +
    r_centered + above_cutoff + post +           # 主效应
    r_centered:post + r_centered:above_cutoff +  # 评分多项式 × 时间/处理交互
    control1 + control2 |
    unit_fe + year_fe,                           # 单元 + 时间 FE
  data    = df_bw,
  cluster = ~unit_id
)
summary(res_rddd)
# 关注：rddd_treat 的系数 = RD-DD 估计量

# ---- Step 4：预趋势检验（条件平行趋势）----
# 在政策前期，阈值两侧的趋势应平行
df_pre <- df_bw %>% filter(post == 0)
# 用年份 × above_cutoff 交互检验政策前趋势差异
res_pretrend <- feols(
  outcome ~ i(year, above_cutoff, ref = base_year) + r_centered | unit_fe,
  data    = df_pre,
  cluster = ~unit_id
)
iplot(res_pretrend, main = "Pre-trend Check: Threshold × Year")
# 期望：政策前所有年份的 above_cutoff × year 系数在0附近

# ---- Step 5：带宽敏感性 ----
bw_sensitivity_rddd <- lapply(c(0.5, 0.75, 1.0, 1.25, 1.5) * h_opt, function(h) {
  df_h <- df %>% filter(abs(r_centered) <= h)
  res  <- feols(
    outcome ~ rddd_treat + r_centered + above_cutoff + post +
      r_centered:post + r_centered:above_cutoff | unit_fe + year_fe,
    data = df_h, cluster = ~unit_id
  )
  data.frame(bw = h, coef = coef(res)["rddd_treat"],
             se  = se(res)["rddd_treat"],
             n   = nrow(df_h))
}) |> bind_rows()
print(bw_sensitivity_rddd)
```

---

### Bleemer-Mehta (2022) RDD 机制分析

**方法来源：** Bleemer & Mehta (2022) *"Income-Based Affirmative Action in Undergraduate Admissions"*，使用 RDD 框架定量分解机制路径的贡献度。

**核心逻辑：**
1. 主 RDD：$\hat{\tau}_{total}$（处理变量 D 对结果 Y 的总效应）
2. 机制 RDD：用机制变量 M（如"是否进入名校"）预测结果变量 → 得到 $\hat{Y}_M = f(\hat{M})$
3. 将预测值 $\hat{Y}_M$ 作为因变量对 RDD 重跑 → 得到 $\hat{\tau}_{M \to Y}$
4. 机制解释力度 = $\hat{\tau}_{M \to Y} / \hat{\tau}_{total}$

**适用条件：**
- 机制变量 M 本身在断点处也有跳跃（或与 M 的变化相关）
- 机制变量 M 不存在逆因果（M 必须先于 Y）
- 多个机制时，各机制解释力度之和应约等于总效应

```r
# R: Bleemer-Mehta RDD 机制分析代码模板
library(rdrobust)
library(fixest)
library(dplyr)

# ---- 假设：主 RDD 已完成 ----
# rdd_main: rdrobust 对 outcome ~ r_centered 的主估计
# tau_total: 主 RDD 估计量（总效应）
tau_total <- rdd_main$coef["Bias-Corrected", 1]
h_main    <- rdd_main$bws["h", 1]

# ---- 机制分析：以机制变量 M 为中介 ----
# 步骤1：用控制变量和机制变量预测结果（样本内拟合）
# 使用带宽外样本拟合，带宽内预测（避免循环内生）
df_outside_bw <- df %>% filter(abs(r_centered) > h_main)
df_inside_bw  <- df %>% filter(abs(r_centered) <= h_main)

# 用带宽外数据训练 Y ~ M 的关系
fit_mechanism <- lm(outcome ~ mechanism_var + control1 + control2,
                    data = df_outside_bw)

# 步骤2：预测带宽内个体的结果（基于机制变量）
df_inside_bw$y_pred_via_M <- predict(fit_mechanism, newdata = df_inside_bw)

# 步骤3：用预测值作为因变量跑 RDD → 得到机制贡献的效应
rdd_mechanism <- rdrobust(
  y = df_inside_bw$y_pred_via_M,
  x = df_inside_bw$r_centered,
  c = 0, p = 1,
  h = h_main  # 固定与主 RDD 相同的带宽，便于比较
)
tau_mechanism <- rdd_mechanism$coef["Bias-Corrected", 1]

# 步骤4：计算机制解释力度
mechanism_share <- tau_mechanism / tau_total
cat(sprintf("总效应 τ_total       = %.4f\n", tau_total))
cat(sprintf("机制效应 τ_mechanism = %.4f\n", tau_mechanism))
cat(sprintf("机制解释力度         = %.1f%%\n", mechanism_share * 100))

# ---- 多机制分解 ----
mechanisms <- c("mechanism1", "mechanism2", "mechanism3")
mech_results <- lapply(mechanisms, function(m) {
  fit_m <- lm(as.formula(paste("outcome ~", m, "+ control1 + control2")),
              data = df_outside_bw)
  df_inside_bw$y_pred <- predict(fit_m, newdata = df_inside_bw)

  rdd_m <- rdrobust(y = df_inside_bw$y_pred, x = df_inside_bw$r_centered,
                    c = 0, p = 1, h = h_main)
  tau_m <- rdd_m$coef["Bias-Corrected", 1]
  data.frame(mechanism = m, tau_m = tau_m, share = tau_m / tau_total * 100)
}) |> bind_rows()

cat("\n=== 机制分解结果 ===\n")
print(mech_results)
cat(sprintf("各机制之和: %.1f%%（应约等于100%%，否则机制不完整）\n",
            sum(mech_results$share)))
```

---

### CER 最优带宽

`rdrobust` 提供两类带宽：

| 带宽类型 | 英文全称 | 用途 | `bwselect` 参数 |
|---------|---------|------|----------------|
| **MSE 带宽** | Mean Squared Error optimal | 点估计最优（偏误-方差权衡） | `"mserd"` |
| **CER 带宽** | Coverage Error Rate optimal | 置信区间覆盖率最优 | `"cerrd"` |

**关键区别：**
- MSE 带宽最小化 $E[(\hat{\tau} - \tau)^2]$，适合**点估计报告**
- CER 带宽最小化置信区间的覆盖误差，适合**统计推断和置信区间报告**
- Calonico et al. (2020) 建议：**点估计用 MSE 带宽，置信区间用 Robust CI（基于 CER 带宽计算偏误修正）**

```r
# R: CER 带宽 vs MSE 带宽比较
library(rdrobust)

# MSE 最优带宽（点估计）
rdd_mse <- rdrobust(
  y = df$outcome, x = df$r_centered, c = 0,
  p = 1, bwselect = "mserd"
)

# CER 最优带宽（置信区间）
rdd_cer <- rdrobust(
  y = df$outcome, x = df$r_centered, c = 0,
  p = 1, bwselect = "cerrd"
)

cat(sprintf("MSE 带宽: h = %.4f, b = %.4f\n",
            rdd_mse$bws["h", 1], rdd_mse$bws["b", 1]))
cat(sprintf("CER 带宽: h = %.4f, b = %.4f\n",
            rdd_cer$bws["h", 1], rdd_cer$bws["b", 1]))

# CER 带宽通常比 MSE 带宽更窄（牺牲点估计精度，换取更准确的推断）
cat(sprintf("CER/MSE 比值: %.2f（通常 <1，CER更保守）\n",
            rdd_cer$bws["h", 1] / rdd_mse$bws["h", 1]))

# ---- 推荐报告方式 ----
# 点估计：用 MSE 带宽的 Bias-Corrected 列
# 置信区间：用 MSE 或 CER 带宽的 Robust 列（两者均可，CER 更严谨）
cat("\n=== 主估计（MSE 带宽，Bias-Corrected 点估计 + Robust CI）===\n")
summary(rdd_mse)

cat("\n=== 稳健性（CER 带宽）===\n")
summary(rdd_cer)

# ---- 输出对比表 ----
results_bw <- data.frame(
  spec         = c("MSE 带宽", "CER 带宽"),
  bandwidth    = c(rdd_mse$bws["h",1], rdd_cer$bws["h",1]),
  coef_bc      = c(rdd_mse$coef["Bias-Corrected",1], rdd_cer$coef["Bias-Corrected",1]),
  ci_lo_robust = c(rdd_mse$ci["Robust",1], rdd_cer$ci["Robust",1]),
  ci_hi_robust = c(rdd_mse$ci["Robust",2], rdd_cer$ci["Robust",2])
)
print(round(results_bw, 4))
```

---

### ITT vs TOT 区分

| 类型 | 全称 | 等价关系 | rdrobust 实现 |
|------|------|---------|--------------|
| **ITT** | Intent-to-Treat | Sharp RDD = ITT = TOT | 标准 `rdrobust(y, x)` |
| **TOT** | Treatment-on-the-Treated | Fuzzy RDD: TOT = IV/RDD 估计量 | `rdrobust(y, x, fuzzy = actual_treatment)` |

**Sharp RDD：**
- 阈值严格决定处理（$D = \mathbf{1}[R \geq c]$，100% 服从）
- 此时 ITT = TOT，因为每个达到阈值的人都被处理了
- 不需要区分 ITT 和 TOT

**Fuzzy RDD：**
- 阈值仅提高处理概率（不完全服从），存在 Always-taker 和 Never-taker
- $\text{ITT} = E[Y | R \geq c] - E[Y | R < c]$（常规 rdrobust 估计）
- $\text{TOT} = \text{LATE at cutoff} = \text{ITT} / \text{First Stage}$（Fuzzy rdrobust 估计）

```r
# R: Sharp RDD — ITT = TOT
rdd_sharp <- rdrobust(
  y = df$outcome,
  x = df$r_centered,
  c = 0, p = 1, bwselect = "mserd"
  # 不需要 fuzzy 参数
)
summary(rdd_sharp)
# 此时估计量既是 ITT 也是 TOT（因为 D = 1(R ≥ 0)）

# ---- Fuzzy RDD：明确区分 ITT 和 TOT ----

# ITT（常规 rdrobust，以 above_cutoff 为"处理"）
rdd_itt <- rdrobust(
  y = df$outcome,
  x = df$r_centered,
  c = 0, p = 1, bwselect = "mserd"
  # 不传 fuzzy → 估计量是 ITT
)
tau_itt <- rdd_itt$coef["Bias-Corrected", 1]
cat(sprintf("ITT（意向处理效应）= %.4f\n", tau_itt))

# 第一阶段：处理概率在阈值处的跳跃
rdd_first_stage <- rdrobust(
  y = df$actual_treatment,  # 实际处理变量（非 above_cutoff）
  x = df$r_centered,
  c = 0, p = 1, bwselect = "mserd"
)
first_stage_jump <- rdd_first_stage$coef["Bias-Corrected", 1]
cat(sprintf("第一阶段（处理概率跳跃）= %.4f\n", first_stage_jump))

# TOT（Fuzzy RDD，本质是 IV：IV = above_cutoff，内生变量 = actual_treatment）
rdd_tot <- rdrobust(
  y     = df$outcome,
  x     = df$r_centered,
  c     = 0,
  fuzzy = df$actual_treatment,  # 实际处理变量
  p     = 1, bwselect = "mserd"
)
tau_tot <- rdd_tot$coef["Bias-Corrected", 1]
cat(sprintf("TOT（处理效应，LATE at cutoff）= %.4f\n", tau_tot))

# 验证：TOT ≈ ITT / 第一阶段
cat(sprintf("验证: ITT / First Stage = %.4f ≈ TOT = %.4f\n",
            tau_itt / first_stage_jump, tau_tot))

# ---- 输出对比表 ----
cat("\n=== ITT vs TOT 对比 ===\n")
results_fuzzy <- data.frame(
  estimand    = c("ITT（意向处理）", "First Stage（处理跳跃）", "TOT（LATE at cutoff）"),
  estimate    = c(tau_itt, first_stage_jump, tau_tot),
  ci_lo       = c(rdd_itt$ci["Robust",1], rdd_first_stage$ci["Robust",1], rdd_tot$ci["Robust",1]),
  ci_hi       = c(rdd_itt$ci["Robust",2], rdd_first_stage$ci["Robust",2], rdd_tot$ci["Robust",2])
)
print(round(results_fuzzy, 4))
```

---

### 控制变量加入方式

在 RDD 中加入控制变量的**正确方法**：控制变量以**线性可分**的形式加入，**不与处理变量（或断点指示变量）交互**。

**正确做法（线性可分）：**
$$Y = \tau D + f(R) + \mathbf{X}'\gamma + \varepsilon$$

**错误做法（控制变量 × 处理交互）：**
$$Y = \tau D + f(R) + \mathbf{X}'\gamma + D \cdot \mathbf{X}'\delta + \varepsilon$$
（后者会改变识别对象，混淆 LATE 的解释）

**控制变量的作用：** 降低残差方差，提高估计精度，但不改变点估计（在正确规格下）

```r
# R: RDD 控制变量加入示例

# ---- 无控制变量（基准）----
rdd_no_ctrl <- rdrobust(
  y = df$outcome, x = df$r_centered, c = 0,
  p = 1, bwselect = "mserd"
)

# ---- 加入控制变量（covs 参数，线性可分）----
# rdrobust 通过 covs 参数加入控制变量（内部线性部分化）
rdd_with_ctrl <- rdrobust(
  y    = df$outcome,
  x    = df$r_centered,
  c    = 0,
  p    = 1,
  bwselect = "mserd",
  covs = df[, c("control1", "control2", "control3")]  # 控制变量矩阵
)

cat("=== 无控制变量 ===\n"); summary(rdd_no_ctrl)
cat("\n=== 加入控制变量 ===\n"); summary(rdd_with_ctrl)

# 注意：点估计应基本不变，但标准误会降低（精度提升）
cat(sprintf("\n点估计变化: %.4f → %.4f（差异应 <5%%）\n",
            rdd_no_ctrl$coef["Bias-Corrected",1],
            rdd_with_ctrl$coef["Bias-Corrected",1]))
cat(sprintf("SE 变化: %.4f → %.4f（应降低）\n",
            rdd_no_ctrl$se["Robust",1],
            rdd_with_ctrl$se["Robust",1]))

# ---- 用 feols 手动实现（带宽内线性控制变量）----
# 限制在最优带宽内
h_opt <- rdd_with_ctrl$bws["h", 1]
df_bw <- df %>% filter(abs(r_centered) <= h_opt)

# 线性可分控制变量（不交互）
res_rdd_ols <- feols(
  outcome ~ above_cutoff + r_centered + above_cutoff:r_centered +
    control1 + control2 + control3,  # 控制变量独立加入，不交互
  data    = df_bw,
  weights = ~ triangular_weight,    # 三角核权重
  cluster = ~unit_id
)
summary(res_rdd_ols)
```

---

### Donut Hole 检验定位调整

**原有内容（Step 7c）补充警告：**

> ⚠️ **Donut RD 精度警告（Cattaneo et al. 2023）：**
>
> Donut RD 通过排除最接近断点的观测（$|R| < d$）来处理潜在操控。但 Cattaneo, Titiunik & Vazquez-Bare (2023) 指出：
>
> 1. **估计精度下降**：Donut 排除了信息最密集的断点区域，导致方差显著增大、置信区间变宽。
> 2. **改变估计对象**：Donut RD 估计的不再是断点处（$R = 0$）的 LATE，而是 $|R| \geq d$ 处的效应，存在外推问题。
> 3. **建议仅在以下条件下使用 Donut RD：**
>    - 密度检验（rddensity）显示有操控迹象（$p < 0.10$）
>    - 必须同时报告标准 RD 结果作为对比
>    - 说明 Donut 宽度 $d$ 的选择依据（如操控嫌疑区域的范围）
>
> **替代建议：** 若密度检验未发现操控迹象，不需要做 Donut RD。若担心离散评分变量的测量误差，考虑使用连续化方法（如 rdrobust 的 `mass_points` 参数）。

```r
# R: 正确的 Donut Hole 检验——配合密度检验结果决定是否做
library(rddensity)
library(rdrobust)

# 先做密度检验
density_res <- rddensity(X = df$r_centered, c = 0)
density_pval <- density_res$test$p_jk  # Jackknife p-value
cat(sprintf("密度检验 p 值: %.4f\n", density_pval))

if (density_pval < 0.10) {
  cat("⚠️ 检测到潜在操控迹象，进行 Donut RD\n")

  # 标准 RD（对比基准）
  rdd_standard <- rdrobust(y = df$outcome, x = df$r_centered, c = 0, p = 1)
  tau_standard  <- rdd_standard$coef["Bias-Corrected", 1]

  # Donut RD（排除 |r| < d 的观测）
  donut_widths <- c(0.1, 0.2, 0.5) * rdd_standard$bws["h", 1]  # 相对于带宽的比例
  donut_results <- lapply(donut_widths, function(d) {
    df_d <- df %>% filter(abs(r_centered) >= d)
    res  <- rdrobust(y = df_d$outcome, x = df_d$r_centered, c = 0, p = 1)
    data.frame(donut = d, coef_bc = res$coef["Bias-Corrected",1],
               ci_lo = res$ci["Robust",1], ci_hi = res$ci["Robust",2],
               se    = res$se["Robust",1])
  }) |> bind_rows()

  cat("\n=== Donut RD 对比（密度检验显示操控嫌疑时才使用）===\n")
  cat(sprintf("标准 RD: %.4f\n", tau_standard))
  print(donut_results)

} else {
  cat("✓ 密度检验未发现操控证据（p > 0.10），不需要 Donut RD\n")
  cat("  直接报告标准 rdrobust 结果即可\n")
}
```

---

### Estimand 声明

**RDD → LATE（断点处局部效应）**

在论文中，每次报告 RDD 结果时，**必须**包含以下声明：

| 声明项目 | 内容要求 |
|----------|---------|
| 估计量定义 | 明确标注"本文 RDD 估计量为 LATE（断点处局部平均处理效应）" |
| 局部性限制 | 声明结论仅对评分变量接近阈值（$R \approx c$）的个体成立 |
| 外推限制 | 明确声明不能外推到远离断点的个体（远离阈值者可能有完全不同的效应） |
| Fuzzy vs Sharp | Fuzzy RDD 报告 TOT（LATE at cutoff）而非 ITT，需明确区分 |
| 带宽说明 | 说明估计所用带宽及其选择方法（MSE/CER），有效样本量 |

**标准声明模板（论文正文或脚注）：**
```
本文 RDD 估计量识别的是断点 [阈值描述] 附近的局部平均处理效应（LATE at the cutoff）。
该估计量仅对评分变量 [评分变量名] 接近阈值 [c 值] 的个体具有直接因果解释，
不能外推到评分远低于或远高于阈值的群体，
这些群体在可观测和不可观测特征上可能与断点附近个体存在系统性差异。
估计采用局部线性回归，MSE 最优带宽 h = [h 值]，有效样本量为 [N_left]/[N_right]（阈值左/右侧）。
```
