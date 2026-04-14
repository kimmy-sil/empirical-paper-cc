---
name: rdd-analysis
description: "断点回归设计，含 RD 图、带宽选择、McCrary 密度检验、安慰剂检验"
---

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

## 适用性自检（决策前诊断）

> **在开始 RDD 分析之前，必须通过以下四项自检。任一不通过则 RDD 可能不是最佳方法。**

| # | 自检问题 | 通过标准 | 不通过时的替代 |
|---|----------|----------|----------------|
| 1 | **是否存在明确的、外生的阈值？** | 阈值由制度/法规明确规定，非研究者自选 | 阈值模糊或多变 → 考虑 DID 或 IV |
| 2 | **阈值附近是否有足够多的观测？** | 带宽内两侧各有 ≥ 50 个有效观测 | 断点附近数据稀疏 → 扩大数据或换方法 |
| 3 | **个体能否精确操控评分变量？** | 个体无法精确操控（如官方考试分数） | 可精确操控（如自评指标） → RDD 内部效度崩塌 |
| 4 | **LATE at cutoff 是否有政策含义？** | 断点附近效应即为研究关注的核心 | 关心全样本 ATE → 考虑 DID、IV 或实验 |

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

### 离散评分变量预诊断

> ⚠️ **在进入主流程之前，先检查评分变量是否为离散型（整数或大量重复值）。**
> 离散评分变量影响带宽选择和标准误计算，应在此处提前诊断，而非留到最后。

```r
# R: 离散评分变量预诊断
n_unique <- length(unique(df$r_centered))
cat(sprintf("评分变量唯一值数量：%d\n", n_unique))
if (n_unique < 30) {
  cat("⚠️  唯一值较少（< 30），离散评分变量情形\n")
  cat("   后续所有 rdrobust 调用将使用 masspoints='adjust'（默认）\n")
  cat("   带宽选择和标准误将自动调整，但结论需更谨慎\n")
}
```

```python
# Python: 离散评分变量预诊断
n_unique = df['r_centered'].nunique()
print(f"评分变量唯一值数量：{n_unique}")
if n_unique < 30:
    print("⚠️  唯一值较少（< 30），离散评分变量情形")
    print("   后续 rdrobust 调用使用 masspoints='adjust'")
```

### 包依赖

```r
# R（主要工作流）
library(rdrobust)    # 主估计、带宽选择
library(rddensity)   # 密度/操控检验
library(dplyr)
library(ggplot2)
library(fixest)      # 控制变量 + FE（RD-DD）+ 参数法 RDD
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

## 分析流程总览

```
Step 1: RD Plot（初步目检）
Step 2: 密度/操控检验（内部效度门槛）      ← 先验证可行性
Step 3: 协变量平衡检验（辅助验证）          ← 先验证可行性
Step 4: 带宽选择（MSE + CER）
Step 5: 主估计（Sharp / Fuzzy）
Step 6: 稳健性（带宽 + 多项式 + 安慰剂 + Donut + 参数法）
Step 7: 离散评分变量处理（如适用）
```

> **逻辑：先验证 RDD 可行性（Step 2-3），再做估计（Step 4-5），最后稳健性（Step 6-7）。**
> 如果 Step 2 密度检验失败（存在操控），后续所有步骤的结论都要打问号。

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

# Mimicking Variance (MV) 分箱（更平滑，推荐对比）
rdplot(y = df$outcome, x = df$r_centered, c = 0,
       binselect = "qsmv",    # MV 选择准则
       title = "RDD Plot (MV binning)")
```

### Python：rdplot（推荐）

```python
# Python: 使用 rdrobust 包的 rdplot（推荐，与 R 版本一致）
from rdrobust import rdplot
import matplotlib.pyplot as plt

# IMSE-optimal 分箱（默认）
rdplot_out = rdplot(
    y = df['outcome'].values,
    x = df['r_centered'].values,
    c = 0,
    nbins = None,          # 自动 IMSE 最优
    title = "RDD Plot",
    x_label = "Running Variable (centered)",
    y_label = "Outcome"
)
plt.savefig('output/rdd_plot.png', dpi=150)
plt.show()
```

---

## Step 2：密度检验（McCrary / rddensity）

> ⚠️ **这是内部效度的门槛检验。如果密度检验失败（p < 0.10），RDD 的基本假设受到质疑，后续所有结果需谨慎解读。**

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
  cat("⚠️  检测到潜在操控迹象（p < 0.10），后续结论需特别谨慎\n")
  cat("   考虑：(1) Donut RD 作为稳健性；(2) 重新审视研究设计\n")
} else {
  cat("✓  无操控证据（p > 0.10），可继续后续分析\n")
}
```

```python
# Python: rddensity（注意 API 差异）
from rddensity import rddensity, rdplotdensity

density_test = rddensity(X=df['r_centered'].values, c=0)
density_test.summary()

density_pval_py = density_test.p_jk
print(f"密度检验 p 值 = {density_pval_py:.4f}")

rdplotdensity(density_test, X=df['r_centered'].values)
plt.savefig('output/rdd_density_test.png', dpi=150)
```

**说明：** rddensity（Cattaneo et al. 2018）在有限样本下优于 McCrary (2008) DCdensity 检验，已成为标准方法。

---

## Step 3：协变量平衡检验

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
# Python: 协变量平衡
from rdrobust import rdrobust
import pandas as pd

covariates = ['age', 'income_pre', 'education', 'female']
balance_results = []

for cov in covariates:
    res = rdrobust(y=df[cov].values, x=df['r_centered'].values, c=0, p=1)
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

## Step 4：带宽选择（MSE + CER 合并）

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
  # 样本量保护：子样本过小时跳过
  if (nrow(sub_df) < 50) return(NULL)
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

### 6e：参数法 RDD 对比（稳健性，非主规格）

> **说明：** 主规格使用 `rdrobust`（非参数局部多项式），参数法 OLS 仅作稳健性对比。
> 两者结果应**方向一致**。若方向相反，需检查带宽选择或模型设定。

```r
# R: 参数法 RDD（全样本 OLS，稳健性对比）
library(fixest)

# 方法1：全样本参数法
res_param_full <- feols(
  outcome ~ r_centered * above_cutoff | fe_var,
  data    = df,
  cluster = ~cluster_var
)

# 方法2：限制在 MSE 最优带宽内的参数法（推荐对比）
df_bw <- df %>% filter(abs(r_centered) <= h_main)
res_param_bw <- feols(
  outcome ~ r_centered * above_cutoff | fe_var,
  data    = df_bw,
  cluster = ~cluster_var
)

# 三规格对比
cat("=== 非参数 vs 参数法对比 ===\n")
cat(sprintf("非参数（rdrobust，主规格）: %.4f [%.4f, %.4f]\n",
            coef_bc, ci_lo, ci_hi))
cat(sprintf("参数法（带宽内 OLS）:       %.4f (SE=%.4f)\n",
            coef(res_param_bw)["above_cutoff"],
            se(res_param_bw)["above_cutoff"]))
cat(sprintf("参数法（全样本 OLS）:       %.4f (SE=%.4f)\n",
            coef(res_param_full)["above_cutoff"],
            se(res_param_full)["above_cutoff"]))
cat("\n期望：三者方向一致，幅度接近\n")
```

```python
# Python: 参数法 RDD（statsmodels OLS）
import statsmodels.formula.api as smf

# 限制在带宽内
h_main_py = rdd_main.bws[0, 0]
df_bw = df[df['r_centered'].abs() <= h_main_py].copy()

res_param = smf.ols(
    'outcome ~ r_centered * above_cutoff',
    data=df_bw
).fit(cov_type='cluster', cov_kwds={'groups': df_bw['cluster_var']})

print(f"参数法 RDD（带宽内 OLS）: {res_param.params['above_cutoff']:.4f}")
print(f"非参数（rdrobust）:       {coef_bc:.4f}")
```

---

## Step 7：离散评分变量（Mass Points）

> ⚠️ **离散评分变量影响带宽选择和标准误。应在 Step 4 带宽选择前已完成预诊断（见前置条件）。**
> 本步骤提供详细的诊断和调整方法。

当评分变量为整数或有大量重复值时，自动处理。

```r
# R: rdrobust masspoints="adjust"（默认行为，自动检测和调整）
rdd_discrete <- rdrobust(
  y          = df$outcome,
  x          = df$r_centered,
  c          = 0,
  p          = 1,
  masspoints = "adjust"  # 默认值，自动处理离散评分变量
)
summary(rdd_discrete)

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
| 适用性自检 | 4 问诊断 | 4 项全部通过 |
| 断点可视化 | `rdplot`（R / Python） | 肉眼可见跳跃，图形合理 |
| 密度/操控检验 | `rddensity`（Cattaneo 2018） | p > 0.10（无操控） |
| 协变量平衡 | `rdrobust` 对每个协变量 | 所有 p_robust > 0.10 |
| 主估计 | `rdrobust` p=1，MSE带宽 | 报告 Bias-corrected + Robust CI |
| 多项式稳健性 | p = 1, 2（不高于 2） | 系数方向和显著性一致 |
| 带宽敏感性 | 0.5h ~ 1.5h | 系数方向一致，幅度稳定 |
| 安慰剂断点 | 假 cutoff（控制组侧） | 系数不显著 |
| Donut Hole | 仅密度检验显著时 | 与标准 RD 结论一致 |
| 参数法对比 | OLS（带宽内 + 全样本） | 与非参数主规格方向一致 |
| Fuzzy：第一阶段 | `rdrobust(y=D, x=R, c=0)` | 跳跃显著，Complier 比例合理 |
| Fuzzy：Wald 验证 | ITT / FS ≈ TOT | 比值接近（< 5% 差异） |

---

## 常见错误（6 条）

> **错误 1：直接用 OLS 加 above_cutoff 哑变量**
> 传统 `y ~ r + above + r*above` 的估计量和标准误均不可靠，带宽 ad-hoc。应使用 `rdrobust` 局部多项式 + MSE 最优带宽。参数法 OLS 仅可作为稳健性对比（见 Step 6e），不应作为主规格。

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
               "Local Linear (0.5h)", "Local Linear (1.5h)",
               "Parametric OLS (MSE bw)"),
  coef_bc  = c(rdd_main$coef["Bias-Corrected", 1], ...),
  ci_lo    = c(rdd_main$ci["Robust", 1], ...),
  ci_hi    = c(rdd_main$ci["Robust", 2], ...),
  h        = c(rdd_main$bws["h", 1], ...),
  n_left   = c(rdd_main$N_h[1], ...),
  n_right  = c(rdd_main$N_h[2], ...),
  poly_p   = c(1, 2, 1, 1, NA)
)
```

**表格必须包含：**
1. 主规格（p=1，MSE 最优带宽）
2. 多项式阶数稳健性（p=2）
3. 带宽变化稳健性（0.5h、1.5h）
4. 参数法 OLS 对比（带宽内）
5. 有效样本量（断点两侧分别报告）
6. CI 类型说明（bias-corrected robust）

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
  rdd_parametric_comparison.csv  # 参数法对比
  rdd_fuzzy_itt_tot.csv          # ITT/TOT 对比（Fuzzy 时）
```

---

## Few-Shot 示例：贫困县政策（Meng 2013）

> 以下为 RDD 经典应用的完整分析思路示例，用于帮助 Claude Code 理解预期输入输出。

**场景描述：** 中国国家级贫困县认定基于各县人均收入是否低于某一阈值。低于阈值的县获得中央财政转移支付和优惠政策。研究问题：贫困县政策是否促进了经济增长？

**断点类型：** 指标阈值（人均收入）

**Sharp / Fuzzy：** Fuzzy RDD — 阈值以下的县大概率被认定为贫困县，但并非 100%（存在政治因素干预）

**检验要点：**
1. 密度检验：人均收入是否在阈值附近存在堆积（地方政府可能操控统计数据）
2. 协变量平衡：阈值两侧的地理特征、人口特征等前定变量是否连续
3. 第一阶段：阈值处"被认定为贫困县"的概率跳跃幅度（Fuzzy 关键）
4. 安慰剂：在远离真实阈值处设假断点，检验系数是否为零

**预期结果：**
- 第一阶段跳跃显著（≥ 30 个百分点），Fuzzy RDD 可行
- ITT 和 TOT 方向一致，TOT 幅度 > ITT（因 Complier 比例 < 1）
- 密度检验需特别关注——如果地方政府操控收入统计，RDD 效度受威胁，需 Donut RD

```r
# 伪代码示例
cutoff <- 400   # 人均收入阈值（元）
df <- df %>% mutate(
  r_centered   = per_capita_income - cutoff,
  above_cutoff = as.integer(r_centered >= 0),    # 高于阈值 = 不贫困
  is_poor_county = ...                            # 实际贫困县认定（Fuzzy）
)

# 主估计
rdd_fuzzy <- rdrobust(y = df$gdp_growth, x = df$r_centered, c = 0,
                       fuzzy = df$is_poor_county, p = 1, bwselect = "mserd")
```
