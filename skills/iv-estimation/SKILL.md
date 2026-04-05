# 工具变量估计 (IV / 2SLS)

## 概述

工具变量法通过寻找只影响内生解释变量、但不直接影响结果变量的外生变量（工具变量 Z），解决内生性问题（遗漏变量、测量误差、反向因果）。标准方法是两阶段最小二乘（2SLS）。

**适用场景：**
- 核心解释变量 X 与误差项相关（内生性）
- 能找到满足相关性（relevance）和排他性（exclusion restriction）的工具变量 Z
- 常见工具变量来源：地理/气候特征、政策变化、历史数据、随机抽签、Bartik shift-share

**三个识别条件（必须在论文中论证）：**
1. **相关性（Relevance）**：Cov(Z, X) ≠ 0 → 第一阶段 F 检验验证
2. **排他性（Exclusion Restriction）**：Cov(Z, ε) = 0 → 逻辑论证，恰好识别时无法统计检验
3. **单调性（Monotonicity）**：无 Defier，保证 LATE 解释成立

---

## 前置条件

### 数据结构要求

```
截面数据：一行一观测
面板数据：长格式，含个体 ID 和时间列

必须包含：
  - 结果变量 Y（被解释变量）
  - 内生变量 X（endogenous variable，可多个）
  - 工具变量 Z（可以多个，数量 ≥ 内生变量数）
  - 外生控制变量 W（exogenous controls，可选）
  - 聚类/固定效应变量（panel IV 必须）

工具变量数量 ≥ 内生变量数量：
  - 恰好识别（just-identified）：Z数 = X数，无法过度识别检验
  - 过度识别（over-identified）：Z数 > X数，可做 Hansen J 检验
```

### 包依赖

```python
# Python
from linearmodels.iv import IV2SLS, IVLIML
import pandas as pd
import numpy as np
from scipy import stats
```

```r
# R
library(fixest)       # feols IV 语法（推荐，速度快，支持多FE）
library(ivreg)        # 完整诊断检验（Hansen J, Wu-Hausman）
library(sandwich)     # 稳健标准误
library(lmtest)       # coeftest
```

---

## Step 1：第一阶段回归 + F 统计量

第一阶段：用工具变量 Z 和外生控制变量 W 回归内生变量 X。
**弱工具判断标准：**
- Staiger-Stock (1997) 经验规则：F > 10
- Lee et al. (2022) 保守标准：F > 104.7（tF 程序，5% 检验水平）
- 多内生变量：Sanderson-Windmeijer conditional F

```python
# Python: 第一阶段 F 统计量 (linearmodels IV2SLS)
import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS

def iv_first_stage(df, outcome, endogenous, instruments, exog_controls=None, 
                   entity_effects=False, time_effects=False):
    """
    运行 IV 第一阶段，返回 F 统计量和弱工具诊断。
    
    Parameters
    ----------
    df          : pandas DataFrame（长格式面板，需有 MultiIndex(entity, time) 或普通截面）
    outcome     : str，结果变量列名
    endogenous  : list，内生变量列名
    instruments : list，工具变量列名
    exog_controls : list or None，外生控制变量
    entity_effects : bool，是否含个体固定效应
    time_effects   : bool，是否含时间固定效应
    
    Returns
    -------
    dict 含 first_stage_results, f_stats, weak_instrument_warning
    """
    exog = exog_controls if exog_controls else []
    
    # linearmodels 格式：IV2SLS(dep, exog, endog, instruments)
    # 注意：exog 必须包含截距（1 列）
    exog_cols = ['const'] + exog
    if 'const' not in df.columns:
        df = df.copy()
        df['const'] = 1.0
    
    results = {}
    f_stats = {}
    
    for endog_var in endogenous:
        # 第一阶段：endog_var ~ exog + instruments
        model_fs = IV2SLS(
            dependent   = df[endog_var],
            exog        = df[exog_cols],
            endog       = None,      # 第一阶段本身 OLS
            instruments = None
        )
        # 实际上用 OLS 回归内生变量
        from linearmodels.ols import OLS
        fs_model = OLS(df[endog_var], df[exog_cols + instruments])
        fs_res   = fs_model.fit(cov_type='robust')
        
        # 提取对 instruments 的联合 F 统计量
        # Wald 检验：instruments 系数联合为 0
        from linearmodels.iv.model import IV2SLS as IV2SLS_full
        restriction_inds = [fs_res.params.index.get_loc(z) for z in instruments]
        f_stat = fs_res.f_statistic.stat
        
        results[endog_var] = fs_res
        f_stats[endog_var] = f_stat
        
        # 弱工具诊断
        if f_stat < 10:
            print(f"⚠️  [{endog_var}] 弱工具！F = {f_stat:.2f} < 10 (Staiger-Stock)")
        elif f_stat < 104.7:
            print(f"⚠️  [{endog_var}] F = {f_stat:.2f}，未达 Lee et al.(2022) 标准 104.7")
        else:
            print(f"✓  [{endog_var}] F = {f_stat:.2f}，通过 Lee et al.(2022) 标准")
    
    return {"first_stage": results, "f_stats": f_stats}
```

```r
# R: 第一阶段 F 统计量 (fixest feols)
library(fixest)

# feols IV 语法：outcome ~ exog_controls | FE | endogenous ~ instruments
# 第一阶段单独查看：
fs_res <- feols(
  endogenous_x ~ exog_control1 + exog_control2 + instrument_z1 + instrument_z2 | 
    unit_fe + year_fe,
  data    = df,
  cluster = ~unit_id
)
summary(fs_res)

# 提取 F 统计量（对 instruments 的联合检验）
# fitstat 获取第一阶段 F
fs_fstat <- fitstat(fs_res, type = "ivf")  # IV 第一阶段 F
print(fs_fstat)

# 多内生变量：Sanderson-Windmeijer conditional F
# 需要 ivreg 包
library(ivreg)
iv_sw <- ivreg(
  outcome ~ endogenous_x1 + endogenous_x2 + exog_ctrl | 
    instrument_z1 + instrument_z2 + instrument_z3 + exog_ctrl,
  data = df
)
# SW conditional F（分别检验每个内生变量对应的工具组）
summary(iv_sw, diagnostics = TRUE)
# 输出：Weak instruments（Sanderson-Windmeijer F for each endogenous var）

# 弱工具诊断函数
check_iv_strength <- function(f_stat, var_name = "X") {
  cat(sprintf("\n=== 工具变量强度诊断: %s ===\n", var_name))
  cat(sprintf("  第一阶段 F = %.2f\n", f_stat))
  if (f_stat < 10)
    cat("  ⚠️  弱工具！F < 10 (Staiger-Stock 1997)\n")
  else if (f_stat < 104.7)
    cat("  ⚠️  F < 104.7，未达 Lee et al.(2022) 保守标准\n")
  else
    cat("  ✓  F >= 104.7，通过 Lee et al.(2022) 标准\n")
}
check_iv_strength(fitstat(fs_res, "ivf")[[1]]$stat)
```

**Lee et al. (2022) 说明：** 当存在弱工具时，传统 F > 10 标准低估了尺寸扭曲。Lee et al. 建议使用 tF 程序；F > 104.7 对应 5% 显著性水平的临界值（保守规则）。弱工具下 LIML 偏误更小（见 Step 2）。

---

## Step 2：2SLS 主回归 + LIML 对照

```python
# Python: IV2SLS 主回归 (linearmodels)
from linearmodels.iv import IV2SLS, IVLIML
import pandas as pd

def run_iv_main(df, outcome, endogenous, instruments, exog_controls=None,
                entity_col=None, time_col=None):
    """
    运行 2SLS 和 LIML，返回结果字典。
    
    Returns
    -------
    dict 含 tsls_result, liml_result
    """
    exog = exog_controls if exog_controls else []
    df_ = df.copy()
    if 'const' not in df_.columns:
        df_['const'] = 1.0
    exog_cols = ['const'] + exog
    
    # 设置 MultiIndex（面板数据）
    if entity_col and time_col:
        df_ = df_.set_index([entity_col, time_col])
    
    # 2SLS
    tsls = IV2SLS(
        dependent   = df_[outcome],
        exog        = df_[exog_cols],
        endog       = df_[endogenous],
        instruments = df_[instruments]
    ).fit(cov_type='clustered', cluster_entity=True if entity_col else False,
          cov_kwds={'clusters': df_[entity_col]} if entity_col and not entity_col in df_.index.names else {})
    
    # LIML（弱工具下偏误更小）
    liml = IVLIML(
        dependent   = df_[outcome],
        exog        = df_[exog_cols],
        endog       = df_[endogenous],
        instruments = df_[instruments]
    ).fit(cov_type='robust')
    
    print("=== 2SLS 结果 ===")
    print(tsls.summary.tables[1])
    print("\n=== LIML 结果（弱工具稳健性）===")
    print(liml.summary.tables[1])
    
    return {"tsls": tsls, "liml": liml}
```

```r
# R: feols IV 语法（推荐，支持多固定效应 + 聚类）
library(fixest)

# 标准 2SLS（feols IV 语法）
# 公式格式：y ~ exog | FE | endog ~ instruments
iv_main <- feols(
  outcome ~ exog_ctrl1 + exog_ctrl2 | unit_fe + year_fe | 
    endogenous_x ~ instrument_z1 + instrument_z2,
  data    = df,
  cluster = ~unit_id
)
summary(iv_main)

# LIML（弱工具下偏误更小，恰好识别时 = 2SLS）
# fixest 不直接支持 LIML，使用 ivreg
library(ivreg)
liml_res <- ivreg(
  outcome ~ endogenous_x + exog_ctrl1 + exog_ctrl2 | 
    instrument_z1 + instrument_z2 + exog_ctrl1 + exog_ctrl2,
  data   = df,
  method = "LIML"
)
summary(liml_res, diagnostics = TRUE)

# 截面 IV（无固定效应）
iv_cross <- feols(
  outcome ~ exog_ctrl | endogenous_x ~ instrument_z,
  data = df, vcov = "hetero"
)
etable(iv_main, iv_cross, headers = c("Panel IV", "Cross-section IV"))
```

**LIML vs 2SLS：**
- 恰好识别时 LIML = 2SLS
- 过度识别 + 弱工具时，LIML 中位数偏误更小（Anderson-Rubin 稳健）
- 实践：2SLS 为主结果，LIML 作稳健性对照

---

## Step 3：倍增比（Amplification Ratio）自动计算

倍增比 = |β_2SLS / β_OLS|，反映处理效应在控制内生性后的放大/缩小程度。

```python
# Python: 自动计算倍增比
import numpy as np

def amplification_ratio(beta_2sls, beta_ols, endog_var="X"):
    """计算并解读倍增比。"""
    if abs(beta_ols) < 1e-10:
        print(f"⚠️  OLS 系数接近 0，倍增比无意义")
        return None
    ratio = abs(beta_2sls / beta_ols)
    direction = "同向" if np.sign(beta_2sls) == np.sign(beta_ols) else "反向（符号翻转）"
    print(f"\n=== 倍增比诊断: {endog_var} ===")
    print(f"  OLS  β = {beta_ols:.4f}")
    print(f"  2SLS β = {beta_2sls:.4f}")
    print(f"  倍增比 |β_2SLS/β_OLS| = {ratio:.2f}  ({direction})")
    if ratio > 5:
        print(f"  ⚠️  倍增比 > 5，请检查：")
        print(f"      1. 工具变量是否满足排他性约束？")
        print(f"      2. 是否存在弱工具（F统计量）？")
        print(f"      3. Complier 群体特征是否与全样本差异很大？")
    elif ratio < 0.2:
        print(f"  ℹ️  倍增比 < 0.2，IV 估计量小于 OLS，可能存在衰减偏误修正")
    return ratio
```

```r
# R: 倍增比计算
amplification_ratio <- function(iv_res, ols_res, endog_var = "endogenous_x") {
  beta_2sls <- coef(iv_res)[endog_var]
  beta_ols  <- coef(ols_res)[endog_var]
  
  if (is.na(beta_ols) | abs(beta_ols) < 1e-10) {
    cat("⚠️  OLS 系数缺失或近零，倍增比无意义\n"); return(invisible(NULL))
  }
  
  ratio     <- abs(beta_2sls / beta_ols)
  direction <- ifelse(sign(beta_2sls) == sign(beta_ols), "同向", "反向（符号翻转）")
  
  cat(sprintf("\n=== 倍增比诊断: %s ===\n", endog_var))
  cat(sprintf("  OLS  β = %.4f\n", beta_ols))
  cat(sprintf("  2SLS β = %.4f\n", beta_2sls))
  cat(sprintf("  |β_2SLS/β_OLS| = %.2f  (%s)\n", ratio, direction))
  
  if (ratio > 5)
    cat("  ⚠️  倍增比 > 5，请检查排他性约束、工具强度、Complier异质性\n")
  else if (ratio < 0.2)
    cat("  ℹ️  倍增比 < 0.2，IV 估计量远小于 OLS，可能为测量误差修正\n")
  
  invisible(list(ratio = ratio, beta_2sls = beta_2sls, beta_ols = beta_ols))
}

# 使用示例
ols_res <- feols(outcome ~ endogenous_x + exog_ctrl | unit_fe + year_fe,
                 data = df, cluster = ~unit_id)
amplification_ratio(iv_main, ols_res)
```

---

## Step 4：诊断检验

### Step 4a：过度识别检验（Hansen J）

**仅在过度识别时可用（工具数 > 内生变量数）。**
H₀：所有工具变量均满足排他性约束。

```r
# R: ivreg::summary(diagnostics=TRUE)（不要手写）
library(ivreg)

iv_ivreg <- ivreg(
  outcome ~ endogenous_x + exog_ctrl1 + exog_ctrl2 | 
    instrument_z1 + instrument_z2 + instrument_z3 + exog_ctrl1 + exog_ctrl2,
  data = df
)
# diagnostics=TRUE 自动计算：
# - Wu-Hausman 内生性检验
# - Sargan/Hansen J 过度识别检验
# - 弱工具 F 检验
summary(iv_ivreg, diagnostics = TRUE)

# 提取 Hansen J p 值
diag <- summary(iv_ivreg, diagnostics = TRUE)$diagnostics
hansen_j_pval <- diag["Sargan", "p-value"]
cat(sprintf("Hansen J p 值 = %.4f\n", hansen_j_pval))
if (hansen_j_pval < 0.05) {
  cat("⚠️  过度识别检验显著（p < 0.05）：至少一个工具变量可能不满足排他性\n")
  cat("   建议：逐一检查每个工具变量的经济逻辑；考虑恰好识别规格\n")
} else {
  cat("✓  过度识别检验未拒绝（p > 0.05）：工具组合整体一致\n")
}
```

**注意：** 恰好识别时 Hansen J 无法计算。过度识别检验拒绝 H₀ 并非必然代表问题——可能是某些工具比其他工具更好，需结合经济逻辑判断。

### Step 4b：Anderson-Rubin 弱 IV 稳健推断

AR 检验对弱工具稳健，不依赖第一阶段 F 的大小。

```r
# R: Anderson-Rubin 检验（ivmodel 包）
library(ivmodel)

# 准备数据（单内生变量）
iv_model <- ivmodel(
  Y  = df$outcome,
  D  = df$endogenous_x,
  Z  = as.matrix(df[, instruments]),
  X  = as.matrix(df[, exog_controls])
)

# AR 置信区间（弱工具稳健）
ar_result <- AR.test(iv_model)
cat("Anderson-Rubin 置信区间（弱 IV 稳健）:\n")
print(ar_result$ci)
# 报告 AR CI 而非传统 Wald CI（当 F < 104.7 时）

# 若工具弱，用 AR CI 替代 2SLS 标准 CI
if (f_stat < 104.7) {
  cat("⚠️  建议报告 AR 置信区间（F < 104.7，弱工具稳健推断）\n")
}
```

### Step 4c：Jackknife 影响力诊断

逐一剔除聚类（或个体），检验结果是否对特定聚类高度敏感。

```r
# R: Jackknife 聚类影响力诊断
jackknife_cluster_iv <- function(data, formula_iv, cluster_var, treated_var) {
  clusters <- unique(data[[cluster_var]])
  main_res <- feols(formula_iv, data = data, cluster = as.formula(paste0("~", cluster_var)))
  beta_main <- coef(main_res)[treated_var]
  
  jk_results <- lapply(clusters, function(cl) {
    df_sub <- data[data[[cluster_var]] != cl, ]
    tryCatch({
      res <- feols(formula_iv, data = df_sub,
                   cluster = as.formula(paste0("~", cluster_var)))
      data.frame(
        excluded_cluster = cl,
        beta             = coef(res)[treated_var],
        change_pct       = (coef(res)[treated_var] - beta_main) / abs(beta_main) * 100
      )
    }, error = function(e) NULL)
  })
  
  jk_df <- do.call(rbind, Filter(Negate(is.null), jk_results))
  jk_df <- jk_df[order(abs(jk_df$change_pct), decreasing = TRUE), ]
  
  # 波动 > 20% 时警告
  flagged <- jk_df[abs(jk_df$change_pct) > 20, ]
  if (nrow(flagged) > 0) {
    cat(sprintf("⚠️  %d 个聚类剔除后系数变化 > 20%%：\n", nrow(flagged)))
    print(flagged[, c("excluded_cluster", "beta", "change_pct")])
  } else {
    cat("✓  Jackknife 诊断：所有聚类剔除后系数变化 < 20%，结果稳定\n")
  }
  
  return(invisible(jk_df))
}

# 使用示例
jk_diag <- jackknife_cluster_iv(
  data        = df,
  formula_iv  = outcome ~ exog_ctrl | unit_fe + year_fe | endogenous_x ~ instrument_z,
  cluster_var = "province_id",
  treated_var = "fit_endogenous_x"   # feols 内生变量拟合名
)
```

---

## Step 5：排他性约束论证 + plausexog

### 5a：Conley-Hansen-Rossi (2012) UCI 方法

允许工具变量对结果有微小的直接影响（δ），检验结论在δ范围内的稳健性。

```r
# R: plausexog 包（UCI 方法）
# install.packages("plausexog")  # 若未安装
library(plausexog)

# 参数设定：δ 的先验范围
# 假设工具变量 Z 对 Y 的直接效应 δ ∈ [-delta_max, delta_max]
delta_max <- 0.1 * abs(coef(iv_main)["fit_endogenous_x"])  # 最大直接效应 = 主效应的 10%

# UCI（union of confidence intervals）方法
# 需要构造数据矩阵
plaus_res <- uci(
  y      = df$outcome,
  x      = df$endogenous_x,
  z      = df[, instruments],
  w      = df[, exog_controls],
  delta  = delta_max,    # 先验约束
  level  = 0.95
)
print(plaus_res)
# 若在 delta 范围内 CI 仍不包含 0，排他性约束不那么严格也能保持结论

# sensemakr 类比（连续处理变量的稳健性）
library(sensemakr)
# 注意：sensemakr 主要用于 OLS，作为 IV 排他性论证的补充参考
ols_sens <- lm(outcome ~ endogenous_x + exog_ctrl + instrument_z, data = df)
sens <- sensemakr(
  model      = ols_sens,
  treatment  = "instrument_z",   # 检验 Z 的直接影响
  benchmark_covariates = exog_controls[1]
)
summary(sens)
```

**结果解读指南：**
- `uci()` 输出：在不同 δ 值下的 β 置信区间
- 若 δ = 0（完全满足排他性），恢复标准 2SLS CI
- 若 δ 增大后 CI 仍不含 0，说明轻度违反排他性不影响结论
- 论文中报告：δ 的经济意义解释 + CI 如何随 δ 变化

### 5b：Lee Bounds（作为替代方法）

当排他性难以完全论证时（如 Fuzzy IV），Lee bounds 提供处理效应上下界。

```r
# R: Lee bounds（trteff 包或手动计算）
# install.packages("trteff")
library(trteff)  # 或手动

# Lee bounds 适用于：结果变量有选择性缺失（selection into measurement）
# 此处用于：IV 排他性约束违反程度的界定
lee_bounds <- function(y, d, z) {
  # z: 工具变量（二元）, d: 处理（二元）
  # 计算两侧 trimming 后的 Wald 估计上下界
  data_lb <- data.frame(y = y, d = d, z = z)
  
  # 处理组（z=1）与控制组（z=0）的结果分布
  y1 <- y[z == 1]; d1 <- d[z == 1]
  y0 <- y[z == 0]; d0 <- d[z == 0]
  
  # 服从率（compliance rate）
  p1 <- mean(d1); p0 <- mean(d0)
  trim_frac <- (p1 - p0) / p1  # Always-takers 比例
  
  if (trim_frac < 0 | trim_frac > 1) {
    cat("⚠️  Monotonicity 可能不满足（trim_frac 超出[0,1]范围）\n")
    return(NULL)
  }
  
  # Upper bound：移除 y1 中最低 trim_frac 比例
  y1_upper <- y1[y1 >= quantile(y1, trim_frac)]
  # Lower bound：移除 y1 中最高 trim_frac 比例
  y1_lower <- y1[y1 <= quantile(y1, 1 - trim_frac)]
  
  ub <- mean(y1_upper) - mean(y0)
  lb <- mean(y1_lower) - mean(y0)
  
  cat(sprintf("Lee Bounds: [%.4f, %.4f]\n", lb, ub))
  cat(sprintf("若界不含 0，处理效应在单调性下可识别\n"))
  return(list(lb = lb, ub = ub))
}
```

---

## Step 6：同行业剩余均值型 IV

剔除自身后的行业/地区/组别均值，用作内生变量的工具。

```python
# Python: leave-one-out 行业均值 IV
import pandas as pd
import numpy as np

def leave_one_out_iv(df, outcome_col, endog_col, group_col, 
                     time_col=None, weight_col=None):
    """
    构造 leave-one-out 行业均值工具变量。
    
    Logic:  Z_i = mean(X_j | j ≠ i, j ∈ group(i))
    
    Parameters
    ----------
    df         : DataFrame
    outcome_col: str，结果变量
    endog_col  : str，内生变量 X（被平均的变量）
    group_col  : str，分组变量（行业/地区）
    time_col   : str or None，时间列（面板数据）
    weight_col : str or None，加权均值（如企业规模）
    
    Returns
    -------
    df 加入新列 'loo_iv'
    """
    df = df.copy()
    group_keys = [group_col] + ([time_col] if time_col else [])
    
    # 分组求和和计数
    group_sum   = df.groupby(group_keys)[endog_col].transform('sum')
    group_count = df.groupby(group_keys)[endog_col].transform('count')
    
    if weight_col:
        # 加权均值版：Z_i = (Σ_{j≠i} w_j X_j) / (Σ_{j≠i} w_j)
        group_wsum = df.groupby(group_keys).apply(
            lambda g: g[weight_col] * g[endog_col]
        ).groupby(level=0).transform('sum')
        group_wtot = df.groupby(group_keys)[weight_col].transform('sum')
        df['loo_iv'] = (group_wsum - df[weight_col] * df[endog_col]) / \
                       (group_wtot - df[weight_col])
    else:
        # 简单均值：Z_i = (group_sum - X_i) / (group_count - 1)
        df['loo_iv'] = (group_sum - df[endog_col]) / (group_count - 1)
    
    # 检查：单人行业会产生 NaN
    n_nan = df['loo_iv'].isna().sum()
    if n_nan > 0:
        print(f"⚠️  {n_nan} 个观测所在组仅有 1 个单位，LOO IV = NaN，已自动排除")
    
    return df

# 使用示例
df = leave_one_out_iv(df, 
                      outcome_col='ln_revenue', 
                      endog_col='ln_wage',
                      group_col='industry_code',
                      time_col='year')

# 查看 IV 相关性
corr = df[['ln_wage', 'loo_iv']].corr().iloc[0, 1]
print(f"LOO IV 与内生变量相关性: {corr:.4f}")
```

```r
# R: leave-one-out 行业均值 IV
library(dplyr)

construct_loo_iv <- function(df, endog_col, group_col, time_col = NULL) {
  # 构造 leave-one-out 行业均值工具变量
  group_keys <- c(group_col, time_col)
  
  df %>%
    group_by(across(all_of(group_keys))) %>%
    mutate(
      group_sum   = sum(.data[[endog_col]], na.rm = TRUE),
      group_count = sum(!is.na(.data[[endog_col]])),
      # Z_i = (Σ_{j≠i} X_j) / (n - 1)
      loo_iv      = (group_sum - .data[[endog_col]]) / (group_count - 1)
    ) %>%
    ungroup() %>%
    select(-group_sum, -group_count)
}

df <- construct_loo_iv(df, endog_col = "ln_wage",
                       group_col = "industry_code", time_col = "year")

# 检验 IV 强度
fs_loo <- feols(ln_wage ~ exog_ctrl | unit_fe + year_fe | ln_wage ~ loo_iv,
                data = df, cluster = ~industry_code)
# 注意：loo_iv 用作 instrument，直接放入 IV 公式
fs_check <- feols(ln_wage ~ loo_iv + exog_ctrl | unit_fe + year_fe,
                  data = df, cluster = ~industry_code)
fitstat(fs_check, "ivf")
```

---

## LATE 识别群体与论文声明

### Complier 特征分析（Abadie κ-weighting）

IV 识别的是 **LATE（Local Average Treatment Effect）**，即 Complier 群体（因工具变量由未处理变为处理的个体）的平均处理效应。**必须在论文中声明 Complier 特征。**

```r
# R: Abadie (2003) κ-weighting 自动生成 Complier 特征描述
library(dplyr)

abadie_kappa_weights <- function(df, treatment, instrument, prob_d1z1 = NULL,
                                  prob_d1z0 = NULL) {
  # κ-weighting: 每个个体的权重反映其为 Complier 的概率
  # κ_i = 1 - D_i(1-Z_i)/(1-P(D=1|Z=0)) - (1-D_i)Z_i/P(D=1|Z=1)
  D <- df[[treatment]]
  Z <- df[[instrument]]
  
  # 估计第一阶段概率
  if (is.null(prob_d1z1)) prob_d1z1 <- mean(D[Z == 1])
  if (is.null(prob_d1z0)) prob_d1z0 <- mean(D[Z == 0])
  
  kappa <- 1 - D * (1 - Z) / (1 - prob_d1z0) - (1 - D) * Z / prob_d1z1
  
  # Complier 特征：用 κ 权重的加权均值 vs 总体均值
  return(kappa)
}

describe_compliers <- function(df, treatment, instrument, covariates) {
  kappa <- abadie_kappa_weights(df, treatment, instrument)
  
  # Complier 特征 vs 全样本
  complier_chars <- sapply(covariates, function(cov) {
    cov_vals <- df[[cov]]
    complier_mean <- weighted.mean(cov_vals, pmax(kappa, 0), na.rm = TRUE)
    overall_mean  <- mean(cov_vals, na.rm = TRUE)
    c(complier_mean = complier_mean, 
      overall_mean  = overall_mean,
      ratio         = complier_mean / overall_mean)
  })
  
  result <- as.data.frame(t(complier_chars))
  cat("\n=== Complier 特征描述（Abadie κ-weighting）===\n")
  print(round(result, 3))
  cat("\n论文声明建议：\n")
  cat("IV 估计量识别的 LATE 对应 Complier 群体，即因工具变量 Z 从 0 变为 1 而\n")
  cat("相应改变处理状态的个体。Complier 群体在以下特征上与全样本有所差异：\n")
  for (cov in covariates) {
    ratio <- result[cov, "ratio"]
    if (abs(ratio - 1) > 0.1) {
      direction <- ifelse(ratio > 1, "高于", "低于")
      cat(sprintf("  - %s：Complier 均值 %s 全样本 %.0f%%\n",
                  cov, direction, abs(ratio - 1) * 100))
    }
  }
  return(invisible(result))
}

# 使用示例
describe_compliers(
  df          = df,
  treatment   = "actual_treatment",
  instrument  = "binary_instrument",
  covariates  = c("age", "education", "income", "urban")
)
```

### Estimand 声明模板

| 情形 | 识别量 | 必须声明内容 |
|------|--------|-------------|
| 二元内生变量 | LATE（Compliers 的 ITT / 第一阶段） | Complier 比例 + 主要特征 |
| 连续内生变量 | 每单位 X 变化的 LATE（加权平均） | 哪些单位 X 受 Z 影响最大 |
| 多 IV | 不同 IV → 不同 LATE | 说明每个 IV 识别的 Complier 群体可能不同 |

**论文声明标准模板：**
```
本文 IV 估计量识别的是 LATE，即 Complier 群体（因工具变量 [Z 的经济含义] 
从 [Z=0 状态] 变为 [Z=1 状态] 而相应改变 [处理变量 X] 的个体）的平均处理效应。
Complier 占总样本约 [第一阶段跳跃比例 × 100]%。
根据 Abadie (2003) κ-weighting 分析，Complier 群体在 [特征1] 和 [特征2] 
上与全样本存在差异（[具体描述]），因此本文结论的外推范围限于类似特征的群体。
```

**连续 vs 二元内生变量的解释差异：**
- **二元 X**：β_2SLS = ITT 效应 / 第一阶段概率跳跃，解释为 Complier 的 ATT
- **连续 X**：β_2SLS = E[dY/dX | Complier]，解释为 Complier 中 X 每增加一单位对 Y 的效应，但"Complier"定义依赖连续化近似

---

## 常见错误

> **错误 1：只报告 F > 10 而不报告 Lee et al. 标准**
> 传统 F > 10 标准低估了弱工具带来的尺寸扭曲。当 F ∈ (10, 104.7) 时，应同时报告 AR 置信区间或 LIML 对照。

> **错误 2：过度识别时不做 Hansen J 检验**
> 当工具变量数 > 内生变量数时，必须报告 Hansen J 检验结果。使用 `ivreg(..., diagnostics=TRUE)` 而非手写。

> **错误 3：Fuzzy IV 第一阶段规格与主回归不一致**
> 第一阶段（报告 F 统计量）和第二阶段（主 2SLS）必须使用相同的固定效应和聚类方案。feols 自动确保一致性。

> **错误 4：忽略倍增比异常（|β_2SLS / β_OLS| > 5）**
> 倍增比 > 5 通常提示排他性约束可能被违反，或 Complier 群体极小（弱工具）。必须论证。

> **错误 5：恰好识别时尝试 Hansen J 检验**
> 恰好识别时 Hansen J 无自由度，无法计算。此时只能逻辑论证排他性。

> **错误 6：面板 IV 忘记在两个阶段加入相同固定效应**
> OLS 和 2SLS 规格必须包含相同的固定效应；feols 的 | FE | endog ~ IV 语法自动处理。

> **错误 7：不说明 IV 识别谁的效应（忽略 LATE）**
> 2SLS 估计量是 LATE，而非 ATE。不同工具变量识别不同 Complier 群体的 LATE；比较多个 IV 的结果时必须考虑 Complier 异质性。必须在论文中声明 Complier 特征并解读外推局限性。

---

## 检验清单

| 检验 | 方法 | 通过标准 |
|------|------|----------|
| 工具相关性（弱工具） | 第一阶段 F 统计量 | F > 10（基本），F > 104.7（严格） |
| 多内生变量条件 F | Sanderson-Windmeijer | 每个内生变量的条件 F > 10 |
| 过度识别（仅过度识别） | Hansen J 检验 | p > 0.05（无法拒绝排他性） |
| 弱工具稳健推断 | Anderson-Rubin CI | 报告 AR CI（当 F < 104.7 时） |
| LIML 对照 | IVLIML vs IV2SLS | 系数接近（过度识别 + 弱工具下尤重要） |
| 排他性约束 | plausexog/sensemakr | δ 范围内结论稳定 |
| Jackknife 影响力 | 逐一剔除聚类 | 单聚类剔除后变化 < 20% |
| 倍增比 | \|β_2SLS/β_OLS\| | 报告并解释（> 5 需重点论证） |
| LATE 声明 | Abadie κ-weighting | 论文中明确 Complier 特征 |

---

## 输出规范

### 主结果表格格式

```r
# R: 汇总对比表（OLS、2SLS、LIML）
library(fixest)
library(modelsummary)

# 输出三列对比：OLS、2SLS（主）、LIML（稳健性）
models_list <- list(
  "OLS"  = ols_res,
  "2SLS" = iv_main,
  "LIML" = liml_res
)

modelsummary(
  models_list,
  coef_map    = c("endogenous_x" = "内生变量 X", "fit_endogenous_x" = "内生变量 X (IV)"),
  gof_map     = c("nobs", "r.squared", "FE: unit_fe", "FE: year_fe"),
  stars       = c("*" = 0.1, "**" = 0.05, "***" = 0.01),
  add_rows    = data.frame(
    term = c("First-stage F", "Hansen J p-val"),
    OLS  = c("—", "—"),
    `2SLS` = c(sprintf("%.1f", f_stat), sprintf("%.3f", hansen_j_pval)),
    LIML = c(sprintf("%.1f", f_stat), "—")
  ),
  output = "output/iv_main_table.tex"
)
```

**表格必须包含：**
1. OLS 对照列（含内生性方向对比）
2. 2SLS 主估计（聚类标准误）
3. LIML 稳健性（若过度识别或弱工具）
4. 第一阶段 F 统计量
5. Hansen J p 值（仅过度识别时）
6. 固定效应结构说明

### 文件命名

```
output/
  iv_first_stage.csv         # 第一阶段 F 统计量
  iv_main_table.tex          # 主回归结果（OLS/2SLS/LIML）
  iv_amplification_ratio.csv # 倍增比
  iv_overid_test.csv         # Hansen J 检验（过度识别时）
  iv_ar_ci.csv               # Anderson-Rubin 置信区间
  iv_jackknife_diag.csv      # Jackknife 影响力
  iv_complier_chars.csv      # Complier 特征（κ-weighting）
  iv_plausexog.csv           # 排他性约束稳健性
```

---

## Estimand 声明

**IV 2SLS → LATE（Local Average Treatment Effect，Complier 群体）**

| 声明项目 | 内容要求 |
|----------|----------|
| 估计量定义 | 明确标注"本文 IV 估计量为 LATE，识别 Complier 群体的平均处理效应" |
| Complier 特征 | 报告 Abadie κ-weighting 分析的 Complier vs 全样本特征对比 |
| 外推局限 | 声明结论不适用于 Always-taker 和 Never-taker |
| 多 IV 比较 | 若使用多个 IV，说明不同 IV 可能识别不同 Complier 群体 |
| 连续内生变量 | 额外说明"Complier"的近似性质及局限 |
