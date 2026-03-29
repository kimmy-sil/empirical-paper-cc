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
