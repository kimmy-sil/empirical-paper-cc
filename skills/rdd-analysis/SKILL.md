# 断点回归设计 (RDD)

## 概述

断点回归利用个体在某个"评分变量"（running variable / forcing variable）上的阈值（cutoff）来识别因果效应。在阈值附近，个体因接近随机分配而具有很高的内部效度。

**适用场景：**
- 处理分配基于某个连续变量（评分变量）是否超过阈值
- 示例：考试分数超过 600 分获得奖学金、企业规模超过 50 人须遵守某法规、地方官员投票超过 50% 当选
- **Sharp RDD**：超过阈值 100% 被处理
- **Fuzzy RDD**：超过阈值大幅提高处理概率（不完全服从），等价于在断点附近做 IV

**识别假设：** 潜在结果 E[Y(0)|R] 和 E[Y(1)|R] 在阈值 c 处连续（无操控）。

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
  - 检查是否有评分变量操控迹象（McCrary/rddensity）
```

### 中心化

```python
# Python: 评分变量中心化
import pandas as pd
import numpy as np

cutoff = 0.5   # 替换为实际阈值

df['r_centered']  = df['running_var'] - cutoff
df['above_cutoff'] = (df['r_centered'] >= 0).astype(int)
```

```r
# R: 中心化
library(dplyr)

cutoff <- 0.5   # 替换为实际阈值

df <- df %>%
  mutate(
    r_centered   = running_var - cutoff,
    above_cutoff = as.integer(r_centered >= 0)
  )
```

### 包依赖

```r
# R（主要工作流）
library(rdrobust)    # 主估计、带宽选择
library(rddensity)   # 密度/操控检验
library(dplyr)
library(ggplot2)
library(fixest)      # 控制变量 + FE（RD-DD）
```

```python
# Python（简化版，推荐 R 为主）
from rdrobust import rdrobust, rdbwselect, rdplot
from rddensity import rddensity, rdplotdensity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

---

## Step 1：RD Plot 可视化

### R：rdplot（IMSE + MV bandwidth，推荐）

```r
# R: rdplot（rdrobust 包，自动选择分箱数）
library(rdrobust)

# IMSE-optimal 分箱（默认，推荐）
rdplot(
  y       = df$outcome,
  x       = df$r_centered,
  c       = 0,
  nbins   = NULL,         # NULL = 自动 IMSE 最优
  title   = "RDD Plot",
  x.label = "Running Variable (centered)",
  y.label = "Outcome"
)
# 推荐报告：将此图直接保存为论文图

# Mimicking Variance (MV) 分箱（更平滑，推荐对比）
rdplot(y = df$outcome, x = df$r_centered, c = 0,
       binselect = "qsmv",    # MV 选择准则
       title = "RDD Plot (MV binning)")
```

### Python：matplotlib 分箱散点图（简化版）

```python
# Python: 手动分箱散点图（简化版，推荐用 R rdplot 作最终图）
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def rdd_plot_python(df, r_col, y_col, n_bins=40, title="RDD Plot"):
    """
    Python 简化版 RD 图（推荐用 R rdplot 替代）。
    
    注意：使用 np.linspace 避免 bandwidth undefined 错误。
    """
    r = df[r_col].values
    y = df[y_col].values
    
    # 使用 np.linspace 固定 40 个区间（避免 bandwidth undefined 错误）
    bin_edges_left  = np.linspace(r.min(), 0, n_bins // 2 + 1)
    bin_edges_right = np.linspace(0, r.max(), n_bins // 2 + 1)
    bin_edges = np.unique(np.concatenate([bin_edges_left, bin_edges_right]))
    
    # 计算每个 bin 的均值
    bin_idx = np.digitize(r, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_edges) - 2)
    
    bin_data = []
    for b in range(len(bin_edges) - 1):
        mask = bin_idx == b
        if mask.sum() >= 3:
            bin_data.append({
                'r_mid': np.mean(r[mask]),
                'y_mean': np.mean(y[mask]),
                'side': 'right' if np.mean(r[mask]) >= 0 else 'left'
            })
    
    bdf = pd.DataFrame(bin_data)
    left  = bdf[bdf['side'] == 'left']
    right = bdf[bdf['side'] == 'right']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(left['r_mid'],  left['y_mean'],  color='steelblue', label='Control', alpha=0.8)
    ax.scatter(right['r_mid'], right['y_mean'], color='tomato',    label='Treated',  alpha=0.8)
    
    # 拟合线（局部线性）
    from numpy.polynomial.polynomial import polyfit, polyval
    for side_df, color in [(left, 'steelblue'), (right, 'tomato')]:
        if len(side_df) >= 2:
            x_s = side_df['r_mid'].values
            y_s = side_df['y_mean'].values
            coef = polyfit(x_s, y_s, 1)
            x_line = np.linspace(x_s.min(), x_s.max(), 100)
            ax.plot(x_line, polyval(x_line, coef), color=color, linewidth=2)
    
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Running Variable (centered)')
    ax.set_ylabel('Outcome')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig

fig = rdd_plot_python(df, r_col='r_centered', y_col='outcome')
fig.savefig('output/rdd_plot.png', dpi=150)
```

---

## Step 2：带宽选择（MSE + CER 合并）

**两类带宽：**
- **MSE 带宽**（`bwselect="mserd"`）：最小化均方误差，用于**点估计**
- **CER 带宽**（`bwselect="cerrd"`）：最小化置信区间覆盖误差，用于**统计推断**
- 实践规则：点估计用 MSE 带宽，报告 Robust CI（bias-corrected）

```r
# R: 带宽选择（rdrobust，同时计算 MSE 和 CER）
library(rdrobust)

# MSE 最优带宽（点估计）
bw_mse <- rdbwselect(
  y        = df$outcome,
  x        = df$r_centered,
  c        = 0,
  p        = 1,          # 局部线性
  bwselect = "mserd"
)
h_mse <- bw_mse$bws["h", 1]
cat(sprintf("MSE 最优带宽 h = %.4f\n", h_mse))

# CER 最优带宽（置信区间）
bw_cer <- rdbwselect(
  y        = df$outcome,
  x        = df$r_centered,
  c        = 0,
  p        = 1,
  bwselect = "cerrd"
)
h_cer <- bw_cer$bws["h", 1]
cat(sprintf("CER 最优带宽 h = %.4f（通常 < MSE 带宽）\n", h_cer))
cat(sprintf("CER/MSE 比值 = %.2f\n", h_cer / h_mse))
```

```python
# Python: rdbwselect
from rdrobust import rdbwselect

bw = rdbwselect(y=df['outcome'].values, x=df['r_centered'].values, c=0, p=1,
                bwselect='mserd')
# 注意 Python API：bw.bws 为矩阵，取 [0,0] 为主带宽 h
h_mse = bw.bws[0, 0]
print(f"MSE 最优带宽 h = {h_mse:.4f}")
```

---

## Step 3：密度检验（McCrary / rddensity）

检验评分变量在阈值处是否存在堆积（manipulation），H₀：无操控（密度连续）。

```r
# R: rddensity（Cattaneo et al. 2018，优于 McCrary 2008）
library(rddensity)

density_res <- rddensity(X = df$r_centered, c = 0)
summary(density_res)   # 看 p-value（H0：连续 = 无操控）

# 密度图（可视化）
rdplotdensity(density_res, X = df$r_centered,
              title = "Density Test: No Manipulation")

# 提取 p 值
density_pval <- density_res$test$p_jk
cat(sprintf("密度检验 p 值 = %.4f\n", density_pval))
if (density_pval < 0.10) {
  cat("⚠️  检测到潜在操控迹象（p < 0.10），考虑 Donut RD\n")
} else {
  cat("✓  无操控证据（p > 0.10）\n")
}
```

```python
# Python: rddensity（注意 API 差异）
from rddensity import rddensity, rdplotdensity

density_test = rddensity(X=df['r_centered'].values, c=0)
density_test.summary()

# Python API 关键：
# p 值在 density_test.p_jk（非 .summary() 的对象）
# 系数：density_test.hat['left'] 和 ['right']
density_pval_py = density_test.p_jk
print(f"密度检验 p 值 = {density_pval_py:.4f}")

rdplotdensity(density_test, X=df['r_centered'].values)
plt.savefig('output/rdd_density_test.png', dpi=150)
```

**说明：** rddensity（Cattaneo et al. 2018）在有限样本下优于 McCrary (2008) DCdensity 检验，已成为标准方法。

---

## Step 4：协变量平衡检验

前定协变量（pre-determined covariates）在断点处不应有跳跃，否则提示操控或选择。

```r
# R: 对每个协变量做 rdrobust，期望 p > 0.10
library(rdrobust)
library(dplyr)

covariates <- c("age", "income_pre", "education", "female")

covariate_balance <- lapply(covariates, function(cov) {
  res <- rdrobust(y = df[[cov]], x = df$r_centered, c = 0, p = 1,
                  bwselect = "mserd")
  data.frame(
    covariate = cov,
    coef_bc   = res$coef["Bias-Corrected", 1],
    se_robust = res$se["Robust", 1],
    p_robust  = res$pv["Robust", 1],
    h         = res$bws["h", 1],
    n_left    = res$N_h[1],
    n_right   = res$N_h[2]
  )
}) %>% bind_rows()

print(covariate_balance)
cat("\n期望：所有协变量 p_robust > 0.10（无跳跃）\n")
```

```python
# Python: 协变量平衡（修复 API：用 res.coef[0] 而非 res.coef.loc['Conventional']）
from rdrobust import rdrobust
import pandas as pd

covariates = ['age', 'income_pre', 'education', 'female']
balance_results = []

for cov in covariates:
    res = rdrobust(y=df[cov].values, x=df['r_centered'].values, c=0, p=1)
    # 正确 API（Python rdrobust）：
    # res.coef[0]  = Conventional 系数
    # res.coef[1]  = Bias-corrected 系数
    # res.pv[2]    = Robust p 值（索引 2，非 'Robust'）
    balance_results.append({
        'covariate': cov,
        'coef_bc':   res.coef[1],    # Bias-corrected
        'p_robust':  res.pv[2]       # Robust p-value
    })

balance_df = pd.DataFrame(balance_results)
print(balance_df)
balance_df.to_csv('output/rdd_covariate_balance.csv', index=False)
```

---

## Step 5：主估计

### Sharp RDD

```r
# R: Sharp RDD 主估计（rdrobust）
library(rdrobust)

rdd_main <- rdrobust(
  y        = df$outcome,
  x        = df$r_centered,
  c        = 0,
  p        = 1,           # 局部线性（推荐主规格）
  kernel   = "triangular", # 三角核（推荐）
  bwselect = "mserd"      # MSE 最优带宽
)
summary(rdd_main)
# 报告：Bias-corrected 点估计 + Robust CI（第三行）
# 不要只报告 Conventional 估计

# 提取关键结果
coef_bc   <- rdd_main$coef["Bias-Corrected", 1]
ci_lo     <- rdd_main$ci["Robust", 1]
ci_hi     <- rdd_main$ci["Robust", 2]
h_main    <- rdd_main$bws["h", 1]
n_left    <- rdd_main$N_h[1]
n_right   <- rdd_main$N_h[2]

cat(sprintf("RDD 估计（Bias-corrected）= %.4f\n", coef_bc))
cat(sprintf("Robust 95%% CI = [%.4f, %.4f]\n", ci_lo, ci_hi))
cat(sprintf("MSE 带宽 h = %.4f，有效样本 n = %d (%d left, %d right)\n",
            h_main, n_left + n_right, n_left, n_right))
```

```python
# Python: Sharp RDD
from rdrobust import rdrobust

rdd_main = rdrobust(y=df['outcome'].values, x=df['r_centered'].values,
                    c=0, p=1, kernel='triangular', bwselect='mserd')
rdd_main.summary()

# Python API：
# rdd_main.coef[1]   = Bias-corrected 估计
# rdd_main.ci[2, :]  = Robust CI [下界, 上界]
# rdd_main.bws[0, 0] = 主带宽 h
coef_bc = rdd_main.coef[1]
ci_lo, ci_hi = rdd_main.ci[2, 0], rdd_main.ci[2, 1]
print(f"RDD 估计 = {coef_bc:.4f}, Robust CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
```

### Fuzzy RDD（ITT vs TOT）

```r
# R: Fuzzy RDD — 明确区分 ITT 和 TOT
library(rdrobust)

# ITT（Intent-to-Treat）= 标准 Sharp rdrobust（以 above_cutoff 为处理）
rdd_itt <- rdrobust(
  y = df$outcome, x = df$r_centered, c = 0, p = 1, bwselect = "mserd"
)
tau_itt <- rdd_itt$coef["Bias-Corrected", 1]

# 第一阶段：处理概率在阈值处的跳跃
rdd_fs <- rdrobust(
  y = df$actual_treatment,   # 实际处理变量（非 above_cutoff）
  x = df$r_centered, c = 0, p = 1, bwselect = "mserd"
)
tau_fs <- rdd_fs$coef["Bias-Corrected", 1]
cat(sprintf("第一阶段跳跃 = %.4f（Complier 比例 ≈ %.1f%%）\n",
            tau_fs, tau_fs * 100))

# TOT（Treatment-on-the-Treated）= Fuzzy RDD（等价 IV）
rdd_tot <- rdrobust(
  y     = df$outcome,
  x     = df$r_centered,
  c     = 0,
  fuzzy = df$actual_treatment,   # 传入实际处理变量
  p     = 1,
  bwselect = "mserd"
)
tau_tot <- rdd_tot$coef["Bias-Corrected", 1]

# Wald 验证：ITT / 第一阶段 ≈ TOT
wald_verify <- tau_itt / tau_fs
cat(sprintf("TOT（Fuzzy RDD）= %.4f\n", tau_tot))
cat(sprintf("Wald 验证：ITT/First_Stage = %.4f ≈ TOT = %.4f\n",
            wald_verify, tau_tot))

# 输出对比
results_fuzzy <- data.frame(
  estimand = c("ITT", "First Stage", "TOT (LATE at cutoff)", "Wald (ITT/FS)"),
  estimate = c(tau_itt, tau_fs, tau_tot, wald_verify),
  ci_lo    = c(rdd_itt$ci["Robust",1], rdd_fs$ci["Robust",1],
               rdd_tot$ci["Robust",1], NA),
  ci_hi    = c(rdd_itt$ci["Robust",2], rdd_fs$ci["Robust",2],
               rdd_tot$ci["Robust",2], NA)
)
print(round(results_fuzzy, 4))
```

### 控制变量加入

控制变量以**线性可分**形式加入（`covs` 参数），**不与处理交互**。

```r
# R: rdrobust covs 参数（正确方式）
rdd_with_ctrl <- rdrobust(
  y        = df$outcome,
  x        = df$r_centered,
  c        = 0,
  p        = 1,
  bwselect = "mserd",
  covs     = df[, c("control1", "control2", "control3")]  # 控制变量矩阵
)

# 对比：点估计应基本不变，标准误降低
cat(sprintf("无控制变量: %.4f (SE=%.4f)\n",
            rdd_main$coef["Bias-Corrected",1], rdd_main$se["Robust",1]))
cat(sprintf("有控制变量: %.4f (SE=%.4f)\n",
            rdd_with_ctrl$coef["Bias-Corrected",1], rdd_with_ctrl$se["Robust",1]))
```

---

## Step 6：稳健性检验

### 6a：带宽敏感性（0.5h ~ 1.5h）

```r
# R: 带宽敏感性
library(rdrobust)
library(dplyr)
library(ggplot2)

h_main <- rdd_main$bws["h", 1]

bw_sensitivity <- lapply(c(0.5, 0.75, 1.0, 1.25, 1.5) * h_main, function(h) {
  res <- rdrobust(y = df$outcome, x = df$r_centered, c = 0, p = 1, h = h)
  data.frame(
    bandwidth = round(h / h_main, 2),
    h_value   = h,
    coef_bc   = res$coef["Bias-Corrected", 1],
    ci_lo     = res$ci["Robust", 1],
    ci_hi     = res$ci["Robust", 2],
    n_left    = res$N_h[1],
    n_right   = res$N_h[2]
  )
}) %>% bind_rows()

ggplot(bw_sensitivity, aes(x = bandwidth, y = coef_bc)) +
  geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), alpha = 0.2, fill = "steelblue") +
  geom_line(color = "steelblue") + geom_point(color = "steelblue", size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  labs(title = "带宽敏感性分析", x = "带宽倍数（h_main = 1.0）", y = "RDD 估计") +
  theme_minimal()
```

### 6b：多项式阶数（p = 1, 2，不高于 2）

```r
# R: p = 1（主规格）, p = 2（稳健性）
# ⚠️  不使用 p >= 3，高阶多项式产生虚假精确性

poly_results <- lapply(1:2, function(p) {
  res <- rdrobust(y = df$outcome, x = df$r_centered, c = 0, p = p,
                  kernel = "triangular", bwselect = "mserd")
  data.frame(
    poly_order = p,
    coef_bc    = res$coef["Bias-Corrected", 1],
    ci_lo      = res$ci["Robust", 1],
    ci_hi      = res$ci["Robust", 2],
    h          = res$bws["h", 1]
  )
}) %>% bind_rows()
print(poly_results)
```

### 6c：安慰剂断点

```r
# R: 安慰剂断点（控制组一侧）
fake_cutoffs <- c(-2, -1.5, -1, 1, 1.5, 2) * (h_main / 2)

placebo_res <- lapply(fake_cutoffs, function(fc) {
  # 使用远离真实断点的区域
  sub_df <- if (fc < 0) df %>% filter(r_centered < -abs(fc)/2) else
                        df %>% filter(r_centered >  abs(fc)/2)
  tryCatch({
    res <- rdrobust(y = sub_df$outcome, x = sub_df$r_centered, c = fc, p = 1)
    data.frame(fake_cutoff = fc,
               coef_bc = res$coef["Bias-Corrected", 1],
               p_robust = res$pv["Robust", 1])
  }, error = function(e) data.frame(fake_cutoff = fc, coef_bc = NA, p_robust = NA))
}) %>% bind_rows()

print(placebo_res)
cat("期望：所有假断点 p_robust > 0.10（系数不显著）\n")
```

### 6d：Donut Hole（仅在操控检验显著时使用）

```r
# R: Donut Hole — 仅当密度检验 p < 0.10 时才使用
# ⚠️  Cattaneo-Titiunik-Vazquez-Bare (2023) 警告：
#   1. 精度下降（排除信息最密集区域）
#   2. 估计对象改变（不再是 LATE at cutoff，而是 |R| >= d 处的效应）
#   3. 必须同时报告标准 RD 作对比，说明 d 的选择依据

if (density_pval < 0.10) {
  cat("⚠️  密度检验显示操控嫌疑，执行 Donut RD\n")
  
  donut_widths <- c(0.05, 0.10, 0.20) * h_main
  donut_results <- lapply(donut_widths, function(d) {
    df_d <- df %>% filter(abs(r_centered) >= d)
    res  <- rdrobust(y = df_d$outcome, x = df_d$r_centered, c = 0, p = 1)
    data.frame(donut_pct = d / h_main * 100,
               donut_val = d,
               coef_bc   = res$coef["Bias-Corrected", 1],
               ci_lo     = res$ci["Robust", 1],
               ci_hi     = res$ci["Robust", 2])
  }) %>% bind_rows()
  
  cat("\n=== Donut RD 结果（与标准 RD 对比）===\n")
  cat(sprintf("标准 RD: %.4f\n", coef_bc))
  print(donut_results)
} else {
  cat("✓  密度检验未发现操控（p > 0.10），不需要 Donut RD\n")
}
```

---

## Step 7：离散评分变量（Mass Points）

当评分变量为整数或有大量重复值时，自动处理。

```r
# R: rdrobust masspoints="adjust"（默认行为，自动检测和调整）
# rdrobust 默认 masspoints="adjust"，无需手动设置
rdd_discrete <- rdrobust(
  y          = df$outcome,
  x          = df$r_centered,
  c          = 0,
  p          = 1,
  masspoints = "adjust"  # 默认值，自动处理离散评分变量
)
summary(rdd_discrete)
# 若存在大量 mass points，rdrobust 会自动调整带宽和标准误

# 诊断：评分变量的唯一值数量
n_unique <- length(unique(df$r_centered))
cat(sprintf("评分变量唯一值数量：%d\n", n_unique))
if (n_unique < 30) {
  cat("⚠️  唯一值较少（< 30），离散评分变量情形\n")
  cat("   masspoints='adjust' 已自动处理，但结论需谨慎\n")
  cat("   建议：报告 masspoints='check' 的诊断信息\n")
  
  rdd_check <- rdrobust(y = df$outcome, x = df$r_centered, c = 0,
                        masspoints = "check")
  cat("Mass points 诊断信息：\n")
  print(rdd_check$masspoints)
}
```

---

## 检验清单

| 检验 | 方法 | 通过标准 |
|------|------|----------|
| 断点可视化 | `rdplot`（R）或分箱散点图（Python） | 肉眼可见跳跃，图形合理 |
| 密度/操控检验 | `rddensity`（Cattaneo 2018） | p > 0.10（无操控） |
| 协变量平衡 | `rdrobust` 对每个协变量 | 所有 p_robust > 0.10 |
| 主估计 | `rdrobust` p=1，MSE带宽 | 报告 Bias-corrected + Robust CI |
| 多项式稳健性 | p = 1, 2（不高于 2） | 系数方向和显著性一致 |
| 带宽敏感性 | 0.5h ~ 1.5h | 系数方向一致，幅度稳定 |
| 安慰剂断点 | 假 cutoff（控制组侧） | 系数不显著 |
| Donut Hole | 仅密度检验显著时 | 与标准 RD 结论一致 |
| Fuzzy：第一阶段 | `rdrobust(y=D, x=R, c=0)` | 跳跃显著，Complier 比例合理 |
| Fuzzy：Wald 验证 | ITT / FS ≈ TOT | 比值接近（< 5% 差异） |

---

## 常见错误（6 条）

> **错误 1：直接用 OLS 加 above_cutoff 哑变量**
> 传统 `y ~ r + above + r*above` 的估计量和标准误均不可靠，带宽 ad-hoc。应使用 `rdrobust` 局部多项式 + MSE 最优带宽。

> **错误 2：只报告 Conventional 估计量**
> rdrobust 输出三行：Conventional、Bias-corrected、Robust CI。正确做法：**点估计报告 Bias-corrected，置信区间报告 Robust**（Calonico et al. 2014）。不要只报告 Conventional。

> **错误 3：多项式阶数过高（p ≥ 3）**
> 高阶多项式对断点附近以外的数据过于敏感，产生虚假精确性（Gelman & Imbens 2019）。仅使用 p = 1（主规格）和 p = 2（稳健性）。

> **错误 4：忽略密度/操控检验**
> 若个体能操控评分变量，断点两侧不再准随机。必须报告 rddensity 结果。

> **错误 5：Fuzzy RDD 不报告第一阶段强度**
> Fuzzy RDD 等价 IV，必须报告阈值处的处理概率跳跃（第一阶段）及其 95% CI。跳跃 < 10 个百分点时结论应特别谨慎。

> **错误 6：Donut Hole 在无操控迹象时使用**
> Donut RD 改变估计对象（不再是 LATE at cutoff），且降低精度。只在密度检验 p < 0.10 时才使用，且必须与标准 RD 对比报告（Cattaneo-Titiunik-Vazquez-Bare 2023）。

---

## Estimand 声明

**RDD → LATE at cutoff（断点处局部平均处理效应）**

| 声明项目 | 内容要求 |
|----------|----------|
| 估计量定义 | "本文 RDD 估计量为 LATE，即评分变量 R 在阈值 c 处的局部处理效应" |
| 局部性限制 | 结论仅对 R ≈ c 的个体成立，不能外推 |
| 外推限制 | 明确声明不适用于 R 远离阈值的个体 |
| Fuzzy vs Sharp | Fuzzy 报告 TOT（LATE at cutoff），需区分 ITT |
| 带宽说明 | 报告 MSE 带宽 h 值、有效样本量（左/右各自） |

**标准声明模板（论文正文或脚注）：**
```
本文 RDD 估计量识别的是断点 [阈值描述] 处的局部平均处理效应（LATE at the cutoff）。
该估计量仅对评分变量 [评分变量名] 接近阈值 [c 值] 的个体具有直接因果解释，
不能外推到评分远低于或远高于阈值的群体（这些群体在可观测和不可观测特征上
可能与断点附近个体存在系统性差异）。
估计采用局部线性回归，MSE 最优带宽 h = [h 值]，
有效样本量为 [N_left]/[N_right]（阈值左/右侧）。
```

---

## 输出规范

```r
# R: 汇总主结果表
results_table <- data.frame(
  spec     = c("Local Linear (MSE bw)", "Local Quadratic (MSE bw)",
               "Local Linear (0.5h)", "Local Linear (1.5h)"),
  coef_bc  = c(rdd_main$coef["Bias-Corrected", 1], ...),
  ci_lo    = c(rdd_main$ci["Robust", 1], ...),
  ci_hi    = c(rdd_main$ci["Robust", 2], ...),
  h        = c(rdd_main$bws["h", 1], ...),
  n_left   = c(rdd_main$N_h[1], ...),
  n_right  = c(rdd_main$N_h[2], ...),
  poly_p   = c(1, 2, 1, 1)
)
```

**表格必须包含：**
1. 主规格（p=1，MSE 最优带宽）
2. 多项式阶数稳健性（p=2）
3. 带宽变化稳健性（0.5h、1.5h）
4. 有效样本量（断点两侧分别报告）
5. CI 类型说明（bias-corrected robust）

### 文件命名

```
output/
  rdd_main_plot.png              # RD 断点散点图
  rdd_density_test.png           # 密度/操控检验图
  rdd_main_results.csv           # 主估计结果
  rdd_covariate_balance.csv      # 协变量平衡
  rdd_bandwidth_sensitivity.csv  # 带宽敏感性
  rdd_placebo_cutoffs.csv        # 安慰剂断点
  rdd_donut_hole.csv             # Donut hole（仅操控时）
  rdd_fuzzy_itt_tot.csv          # ITT/TOT 对比（Fuzzy 时）
```
