---
name: iv-estimation
description: >
  工具变量估计(IV/2SLS/LIML), 含 Step 0 DWH 内生性诊断、弱工具检验
  (CD Wald F + KP rk Wald F, Lee et al. 2022 标准 F>104.7)、
  过度识别 Hansen J、Anderson-Rubin 弱 IV 稳健推断、倍增比、
  内生交互项工具化、测量误差 IV 修正、Jackknife 影响力、
  plausExog 排他性论证、LATE Complier 特征分析。
  触发关键词: 工具变量、IV、2SLS、LIML、内生性、弱工具、排他性约束、LATE。
  NOT for DID/RDD (用对应 skill)。
---

# 工具变量估计 (IV / 2SLS)

## 概述

工具变量法通过寻找只影响内生解释变量、但不直接影响结果变量的外生变量 (工具变量 Z), 解决内生性问题。标准方法是两阶段最小二乘 (2SLS)。

**三个识别条件 (必须在论文中分别论证)**:
1. **相关性 (Relevance)**: Cov(Z, X) != 0 -> 第一阶段 F 检验验证
2. **排他性 (Exclusion Restriction)**: Z 只通过 X 影响 Y -> 需理论论证, 恰好识别时无法统计检验
3. **单调性 (Monotonicity)**: 无 Defier, 保证 LATE 解释成立

**IV 随机性 != 排他性 (常混淆, Bastardoz et al. 2023)**:
- **随机性 (exogeneity/independence)**: Z 与误差项 e 不相关 -> 无法直接检验, 需理论论证 "Z 近似随机"
- **排他性 (exclusion restriction)**: Z 只通过 X 影响 Y -> 需理论论证唯一渠道
- 两者是**不同条件**: 随机性是 "Z 本身外生", 排他性是 "Z->Y 的唯一路径经过 X"
- 论文必须**分别论证**两者, 不能混为一谈

**IV 可处理的三类内生性**: 遗漏变量、反向因果、测量误差 (Durbin, 1954; Wald, 1940)。
论文中应明确声明 IV 修正了哪些内生性来源, 不要笼统说 "处理内生性"。

**常见 IV 来源**: 地理/气候特征、政策变化、历史数据、随机抽签、Bartik shift-share

---

## 前置条件

```
必须包含:
  - 结果变量 Y
  - 内生变量 X (可多个)
  - 工具变量 Z (数量 >= 内生变量数)
  - 外生控制变量 W (可选)
  - 聚类/固定效应变量 (panel IV 必须)

恰好识别 (Z数 = X数): 无法过度识别检验
过度识别 (Z数 > X数): 可做 Hansen J 检验
```

### 包依赖

```python
# Python
import pyfixest as pf          # 主力: feols IV 语法, 与 R fixest 统一
import pandas as pd, numpy as np
# 备用: from linearmodels.iv import IV2SLS, IVLIML (pyfixest 不支持时)
```

```r
# R
library(fixest)       # feols IV (推荐)
library(ivreg)        # 完整诊断 (Hansen J, Wu-Hausman, CD/KP F)
library(sandwich); library(lmtest)
```

---

## Step 0: Durbin-Wu-Hausman 内生性诊断

> **目的**: 统计验证内生性是否存在。Stage 0 已从理论层面判断需要 IV, Step 0 是统计确认。

**H0**: X 外生 (OLS 一致且更有效率)  |  **H1**: X 内生 (需要 IV)

- **p < 0.05** -> 拒绝外生性, IV 有统计正当性
- **p > 0.05** -> 不能拒绝外生性, 但**不代表没有内生性** (检验力可能不足, Hahn et al. 2011; Hausman et al. 2005)。此时同时报告 OLS 和 IV, 声明 "基于理论仍报告 IV"

```r
# R: ivreg diagnostics (推荐, 一行搞定)
library(ivreg)
iv_fit <- ivreg(Y ~ X + ctrl1 + ctrl2 | Z1 + Z2 + ctrl1 + ctrl2, data = df)
summ <- summary(iv_fit, diagnostics = TRUE)
print(summ$diagnostics)  # Wu-Hausman + Sargan + Weak IV F 一次性输出

dwh_p <- summ$diagnostics["Wu-Hausman", "p-value"]
cat(sprintf("Wu-Hausman p = %.4f\\n", dwh_p))
if (dwh_p < 0.05) {
  cat("-> Reject exogeneity, IV justified\\n")
} else {
  cat("-> Cannot reject exogeneity, report both OLS and IV\\n")
  cat("   Note: test may lack power, not evidence of no endogeneity\\n")
}
```

```python
# Python: augmented regression test
import statsmodels.formula.api as smf

def dwh_test(df, outcome, endog_var, instruments, controls):
    ctrl_str = " + ".join(controls) if controls else "1"
    iv_str = " + ".join(instruments)
    # Step 1: first-stage residuals
    fs = smf.ols(f"{endog_var} ~ {iv_str} + {ctrl_str}", data=df).fit()
    df_t = df.copy(); df_t["_fs_resid"] = fs.resid
    # Step 2: augmented regression
    aug = smf.ols(f"{outcome} ~ {endog_var} + {ctrl_str} + _fs_resid", data=df_t).fit(cov_type="HC3")
    dwh_p = aug.pvalues["_fs_resid"]
    status = "IV justified" if dwh_p < 0.05 else "Report both OLS and IV"
    print(f"DWH: t={aug.tvalues['_fs_resid']:.3f}, p={dwh_p:.4f} -> {status}")
    return {"dwh_p": round(dwh_p, 4), "endogenous": dwh_p < 0.05}
```

> **关键**: DWH 不是 "决定是否用 IV" 的唯一依据。理论判断优先。DWH 不显著时不要放弃 IV, 而是同时报告两者。当 IV 本身无效时 (弱工具/排他性违反), DWH 结果也不可靠 (Baum et al. 2003)。

---

## Step 1: 第一阶段 F 统计量

**弱工具判断标准**:
- F > 10 (Staiger-Stock 1997): 最低线
- **F > 104.7 (Lee et al. 2022): 严格标准** (5% 检验水平 tF 程序)
- 多内生变量: Sanderson-Windmeijer conditional F

**必须同时报告两个 F 统计量**:

| 统计量 | 适用条件 | 说明 |
|--------|---------|------|
| Cragg-Donald Wald F | i.i.d. 误差 | 同方差基准 |
| Kleibergen-Paap rk Wald F | 聚类/异方差 SE | 实证论文几乎必报 |

> **常见错误**: 报告包含控制变量的第一阶段整体 F, 或报告第二阶段 F。正确做法是报告**排除性工具变量** (未纳入 Y 方程的 Z) 在控制已纳入变量后的联合 F (Stock & Yogo 2005)。

```r
library(fixest); library(ivreg)

# feols: KP rk Wald F (聚类 SE 下自动计算)
iv_main <- feols(Y ~ ctrl | unit_fe + year_fe | X ~ Z1 + Z2,
                 data = df, cluster = ~unit_id)
fitstat(iv_main, type = "ivf")

# ivreg: diagnostics 同时获取 CD Wald F + Wu-Hausman + Sargan
iv_diag <- ivreg(Y ~ X + ctrl | Z1 + Z2 + ctrl, data = df)
summary(iv_diag, diagnostics = TRUE)

check_iv_strength <- function(f_cd, f_kp, var_name = "X") {
  cat(sprintf("\\n=== %s ===\\n  CD Wald F = %.2f  |  KP rk Wald F = %.2f\\n", var_name, f_cd, f_kp))
  if (f_kp < 10)       cat("  FAIL: F < 10, severe weak IV\\n")
  else if (f_kp < 104.7) cat("  WARN: 10 < F < 104.7, report AR CI + LIML\\n")
  else                    cat("  PASS: F >= 104.7 (Lee et al. 2022)\\n")
}
```

```python
import pyfixest as pf
iv_main = pf.feols("Y ~ ctrl | unit_fe + year_fe | X ~ Z1 + Z2",
                    data=df, vcov={"CRV1": "unit_id"})
pf.etable(iv_main)  # includes first-stage F
```

**F 在 10-104.7 之间**: 报告 AR 置信区间 (Step 4b) + LIML 对照 (Step 2), 而非仅依赖 2SLS t 检验。

---

## Step 2: 2SLS + LIML 主回归

> **禁止手动分步跑 2SLS**: 先回归 X~Z 取拟合值再回归 Y~X_hat, 标准误是错误的 (Bollen et al. 2007)。必须使用 feols/ivreg 内置 IV 命令。

```r
# R: 2SLS (fixest, 推荐)
iv_main <- feols(Y ~ ctrl1 + ctrl2 | unit_fe + year_fe | X ~ Z1 + Z2,
                 data = df, cluster = ~unit_id)
summary(iv_main)

# LIML (ivreg) -- 弱工具下偏误更小
library(ivreg)
liml_res <- ivreg(Y ~ X + ctrl1 + ctrl2 | Z1 + Z2 + ctrl1 + ctrl2,
                  data = df, method = "LIML")
summary(liml_res, diagnostics = TRUE)

# OLS 对照 (主表必须包含)
ols_res <- feols(Y ~ X + ctrl1 + ctrl2 | unit_fe + year_fe,
                 data = df, cluster = ~unit_id)
etable(ols_res, iv_main, headers = c("OLS", "2SLS"))
```

```python
import pyfixest as pf
iv_2sls = pf.feols("Y ~ ctrl1 + ctrl2 | unit_fe + year_fe | X ~ Z1 + Z2",
                    data=df, vcov={"CRV1": "unit_id"})
ols_ref = pf.feols("Y ~ X + ctrl1 + ctrl2 | unit_fe + year_fe",
                    data=df, vcov={"CRV1": "unit_id"})
pf.etable([ols_ref, iv_2sls])

# LIML backup (linearmodels)
from linearmodels.iv import IVLIML
liml = IVLIML(dependent=df["Y"], exog=df[exog_cols],
              endog=df[["X"]], instruments=df[instruments]).fit(cov_type="robust")
```

---

## Step 2b: 内生交互项工具化

**当模型含 内生变量 M x 外生调节变量 X 的交互项时, M*X 也是内生的, 必须同时工具化。**

```
模型:   Y = b0 + b1*M + b2*X + b3*(M*X) + e
        其中 M 内生, X 外生, Z 是 M 的工具变量

IV 集:  {Z, X, Z*X}  ->  内生集: {M, M*X}
第一阶段1: M  ~ Z + X + Z*X + controls
第一阶段2: MX ~ Z + X + Z*X + controls (Wooldridge 2010 Ch.6)
```

```r
# R: feols 内生交互项 (fixest 自动处理)
df$MX <- df$M * df$X   # 内生交互项
df$ZX <- df$Z * df$X   # 工具交互项

# 同时工具化 M 和 MX
iv_interact <- feols(Y ~ X + ctrl | unit_fe + year_fe | M + MX ~ Z + ZX,
                     data = df, cluster = ~unit_id)
summary(iv_interact)
# 注意: 所有工具 (Z, ZX) 必须进入所有第一阶段方程
# X 必须出现在所有方程中 (Wooldridge 2010 Ch.9)
```

```python
df["MX"] = df["M"] * df["X"]
df["ZX"] = df["Z"] * df["X"]
iv_interact = pf.feols("Y ~ X + ctrl | unit_fe + year_fe | M + MX ~ Z + ZX",
                        data=df, vcov={"CRV1": "unit_id"})
```

> **仅在外生调节变量 X 严格外生时成立。** 若 X 也内生, 则需额外工具。多内生变量时需检查 SW conditional F。

---

## Step 3: 倍增比 (Amplification Ratio)

```python
def amplification_ratio(beta_2sls, beta_ols, endog_var="X"):
    import numpy as np
    if abs(beta_ols) < 1e-10: return None
    ratio = abs(beta_2sls / beta_ols)
    direction = "same sign" if np.sign(beta_2sls) == np.sign(beta_ols) else "SIGN FLIP"
    print(f"  OLS={beta_ols:.4f}  2SLS={beta_2sls:.4f}  ratio={ratio:.2f} ({direction})")
    if ratio > 5: print("  WARN: ratio > 5, check exclusion / weak IV / complier heterogeneity")
    return ratio
```

```r
amplification_ratio <- function(iv_res, ols_res, var) {
  r <- abs(coef(iv_res)[var] / coef(ols_res)[var])
  cat(sprintf("  OLS=%.4f  2SLS=%.4f  ratio=%.2f\\n", coef(ols_res)[var], coef(iv_res)[var], r))
  if (r > 5) cat("  WARN: ratio > 5\\n")
  invisible(r)
}
```

---

## Step 4: 诊断检验

### 4a: Hansen J 过度识别检验

**仅过度识别时可用 (Z数 > X数)。** H0: 所有 IV 满足排他性。

```r
iv_oid <- ivreg(Y ~ X + ctrl | Z1 + Z2 + Z3 + ctrl, data = df)
summ <- summary(iv_oid, diagnostics = TRUE)
hansen_p <- summ$diagnostics["Sargan", "p-value"]
cat(sprintf("Hansen J p = %.4f\\n", hansen_p))
```

> 过度识别检验依赖 "至少一个 IV 有效" 的前提 (Hansen 1982; Sargan 1958)。拒绝 H0 不必然代表问题 -- 可能某些 IV 比其他更好。不要以决定论方式解读, 需结合经济逻辑。恰好识别时无法计算。

### 4b: Anderson-Rubin 弱 IV 稳健推断

**F < 104.7 时必须报告 AR CI。** AR 不依赖第一阶段 F。

```r
library(ivmodel)
iv_m <- ivmodel(Y = df$Y, D = df$X, Z = as.matrix(df[, instruments]),
                X = as.matrix(df[, controls]))
ar_result <- AR.test(iv_m)
print(ar_result$ci)
```

### 4c: Jackknife 聚类影响力

逐一剔除聚类, 检验结果稳定性。单聚类剔除后变化 > 20% 则标记。

> 完整 Jackknife 代码 -> ADVANCED.md

---

## Step 5: 排他性约束论证

### 5a: Conley UCI (plausExog 包)

允许 IV 对 Y 有微小直接影响 delta, 检验结论在 delta 范围内是否稳健。

```r
library(plausexog)
delta_max <- 0.1 * abs(coef(iv_main)["fit_X"])
plaus_res <- uci(y = df$Y, x = df$X, z = df[, instruments],
                 w = df[, controls], delta = delta_max, level = 0.95)
# delta 范围内 CI 仍不含 0 -> 排他性轻度违反不影响结论
```

### 5b: Lee Bounds / sensemakr

> 完整代码 -> ADVANCED.md

> **ITCV 等敏感性分析 (如 Frank 2000; Oster 2019) 仅为参考信息, 不能替代研究设计。** 多个遗漏变量时 ITCV 可能导致错误判断 (Lonati & Wulff 2023)。只有合适的识别策略能真正解决遗漏变量问题。

---

## Step 6: Leave-One-Out IV 构造

```r
library(dplyr)
construct_loo_iv <- function(df, endog_col, group_col, time_col = NULL) {
  group_keys <- c(group_col, time_col)
  df %>%
    group_by(across(all_of(group_keys))) %>%
    mutate(loo_iv = (sum(.data[[endog_col]], na.rm=TRUE) - .data[[endog_col]]) /
                    (sum(!is.na(.data[[endog_col]])) - 1)) %>%
    ungroup()
}
```

```python
def leave_one_out_iv(df, endog_col, group_col, time_col=None):
    df = df.copy()
    keys = [group_col] + ([time_col] if time_col else [])
    g_sum = df.groupby(keys)[endog_col].transform("sum")
    g_cnt = df.groupby(keys)[endog_col].transform("count")
    df["loo_iv"] = (g_sum - df[endog_col]) / (g_cnt - 1)
    return df
```

> 加权版本 + 详细诊断 -> ADVANCED.md

---

## Step 7: 测量误差的 IV 修正

当 IV 的使用目的是修正测量误差 (而非遗漏变量/反向因果) 时, 需额外注意:

- **有多个测量指标时** (如问卷条目): 优先用 SEM 潜变量模型, 而非 IV
- **有单指标 + 已知信度时**: 用误差变量回归 (error-in-variables) 或 SEM 约束扰动方差
- **无可靠度量时**: 用 IV 估计 (经典方法, Durbin 1954; Wald 1940)

```r
# 信度已知时的简单修正 (不用 IV, 直接校正衰减偏误)
beta_corrected <- beta_ols / reliability_coefficient

# IV 修正测量误差 (标准 2SLS)
# Z 必须与 X 的真值相关, 但与测量误差无关
iv_me <- feols(Y ~ ctrl | FE | X_measured ~ Z, data = df, cluster = ~id)
# 论文中声明: "IV 估计同时修正了 X 的测量误差偏误"
```

> **非线性模型中的测量误差** (如二次项、交互项) 需特殊估计量 (Brandt et al. 2020; Klein & Moosbrugger 2000), 常需强分布假设, 需谨慎检验。

---

## LATE Estimand 声明

IV 识别的是 **LATE** (Complier 群体的处理效应), 不是 ATE。

| 情形 | 识别量 | 必须声明 |
|------|--------|---------|
| 二元 X | LATE = ITT / 第一阶段 | Complier 比例 + 特征 |
| 连续 X | 加权 LATE | 哪些单位 X 受 Z 影响最大 |
| 多 IV | 不同 IV -> 不同 LATE | 每个 IV 的 Complier 可能不同 |

**声明模板**:
> "本文 IV 估计量识别 LATE, 即因 [Z 经济含义] 变化而改变 [X] 的 Complier 群体的平均处理效应。Complier 占总样本约 [比例]%。根据 Abadie (2003) 分析, Complier 在 [特征] 上与全样本存在差异, 结论外推需考虑此局限。"

> Abadie kappa-weighting Complier 特征分析完整代码 -> ADVANCED.md

---

## 常见错误

1. **只报告 F > 10 不报告 Lee et al. 标准** -- F in (10, 104.7) 时必须补 AR CI + LIML。
2. **只报告一个 F** -- 必须同时报告 CD Wald F (i.i.d.) 和 KP rk Wald F (robust/cluster)。
3. **报告错误的 F 统计量** -- 正确的是排除性工具变量的联合 F, 不是含控制变量的整体 F, 也不是第二阶段 F (Bastardoz et al. 2023)。
4. **无 DWH 检验** -- 审稿人常问 "怎么知道 X 内生?"。提供 DWH 统计证据, 但谨慎解读 (注明检验力局限)。
5. **DWH 不显著就放弃 IV** -- 检验力可能不足 (Hahn et al. 2011), 理论判断优先。IV 无论如何都 consistent。
6. **手动分步跑 2SLS** -- 标准误错误 (Bollen et al. 2007)。必须用 feols/ivreg 内置命令。
7. **过度识别时不做 Hansen J** -- Z数 > X数时必须报告。但不要以决定论方式解读。
8. **第一阶段与主回归 FE/聚类不一致** -- feols 自动确保一致性。
9. **倍增比 > 5 未解释** -- 提示排他性/弱工具/Complier 异质性问题, 必须论证。
10. **内生交互项未工具化** -- M*X 中 M 内生时, M*X 也内生。必须用 Z*X 作工具 (Wooldridge 2010 Ch.6)。
11. **不声明 LATE** -- 2SLS = LATE, 非 ATE。必须报告 Complier 特征。
12. **IV 修正测量误差时未说明** -- 论文应声明 IV 同时修正了哪些内生性来源 (遗漏变量/反向因果/测量误差), 不要笼统说 "处理内生性"。
13. **IV 随机性与排他性混淆** -- 两者是不同条件, 必须分别论证 (见概述)。

---

## 检验清单

| 检验 | 通过标准 |
|------|---------|
| DWH 内生性 | p < 0.05 -> IV justified; p > 0.05 -> 报告两者 |
| CD Wald F + KP rk Wald F | **两个都报告**; KP F > 10 (基本), > 104.7 (严格) |
| SW conditional F (多内生) | 每个内生变量 F > 10 |
| Hansen J (过度识别) | p > 0.05 (谨慎解读) |
| AR CI (F < 104.7 时) | 报告 AR CI 替代 Wald CI |
| LIML 对照 | 系数与 2SLS 接近 |
| 倍增比 | 报告并解释; > 5 需论证 |
| 内生交互项 | M*X 已用 Z*X 工具化 |
| Jackknife | 单聚类剔除后变化 < 20% |
| plausExog | delta 范围内 CI 不含 0 |
| LATE Complier | 论文中声明 Complier 特征 |
| 测量误差 | 声明 IV 修正了哪些内生性来源 |

---

## 输出规范

**主表必须包含**: OLS 对照列、2SLS 主估计、LIML 稳健性 (若弱工具)、第一阶段 F (CD + KP 双报告)、Hansen J p (过度识别时)、DWH p、FE 结构说明。

```r
library(modelsummary)
modelsummary(
  list("OLS" = ols_res, "2SLS" = iv_main, "LIML" = liml_res),
  stars = c("*" = 0.1, "**" = 0.05, "***" = 0.01),
  add_rows = data.frame(
    term = c("First-stage F (KP)", "First-stage F (CD)", "Hansen J p", "DWH p"),
    OLS  = c("--", "--", "--", "--"),
    `2SLS` = c(sprintf("%.1f", f_kp), sprintf("%.1f", f_cd),
               sprintf("%.3f", hansen_p), sprintf("%.3f", dwh_p)),
    LIML   = c(sprintf("%.1f", f_kp), sprintf("%.1f", f_cd), "--", "--")
  ),
  output = "output/iv_main_table.tex"
)
```

```
output/
  iv_dwh_test.csv              # DWH 内生性检验
  iv_first_stage.csv           # F (CD + KP)
  iv_main_table.csv / .tex     # OLS / 2SLS / LIML
  iv_amplification_ratio.csv
  iv_overid_test.csv           # Hansen J
  iv_ar_ci.csv                 # Anderson-Rubin CI
  iv_jackknife_diag.csv
  iv_complier_chars.csv        # Abadie kappa-weighting
  iv_plausexog.csv
```
"""

with open("output/iv-estimation-SKILL.md", "w", encoding="utf-8") as f:
    f.write(content)

import os
size = os.path.getsize("output/iv-estimation-SKILL.md")
chars = len(content)
print(f"iv-estimation-SKILL.md")
print(f"  File size:  {size:,} bytes ({size/1024:.1f} KB)")
print(f"  Characters: {chars:,}")
print(f"  Target <15k chars: {'PASS' if chars < 15000 else 'FAIL'}")
