# 动态面板 GMM

> **加载条件：** 模型中包含滞后因变量（Y_{t-1}），或理论上存在调整成本/状态依赖，或诊断发现残差序列相关。

---

## 触发条件决策树

```
是否需要动态面板GMM？
│
├─ [1] 理论上：存在调整成本 / 习惯形成 / 状态依赖？
│       （企业投资受上期影响；消费存在习惯形成）
│
├─ [2] 实证上：静态FE残差存在序列相关（AR(1)显著）？
│       pbgtest(fe_static, order=1)，p < 0.05
│
├─ [3] 数据上：T < 15-20，Nickell偏误显著？
│       Nickell偏误 ≈ 1/T（T=8 → 约12.5%偏误）
│
├─ 满足 ≥ 2 条 → 使用 System GMM
├─ 满足 0-1 条 → 静态FE即可
└─ 不确定 → 同时跑FE和GMM，验证 ρ_FE < ρ_GMM < ρ_OLS
```

**Nickell偏误大小：**

| T | 偏误估算 | 建议 |
|---|---------|------|
| 5 | ~20% | 强烈建议GMM |
| 8 | ~12.5% | 建议GMM |
| 15 | ~7% | 边界情形，两者都跑 |
| 30 | ~3% | 静态FE基本可接受 |
| > 30 | < 3% | 静态FE即可 |

---

## 前置检验

```r
# R: 诊断是否需要动态模型
library(plm)
library(lmtest)

df_panel <- pdata.frame(df, index = c("entity_id", "year"))

# 静态FE基准
fe_static <- plm(Y ~ X + control1 + control2,
                 data   = df_panel,
                 model  = "within",
                 effect = "twoways")

# 残差序列相关检验（触发条件[2]）
pbgtest(fe_static, order = 1)
# p < 0.05 → 序列相关显著 → 支持动态模型

# OLS含滞后项（Nickell偏误上界）
ols_dyn <- lm(Y ~ lag(Y, 1) + X + control1 + control2, data = as.data.frame(df_panel))
rho_ols <- coef(ols_dyn)["lag(Y, 1)"]
cat(sprintf("ρ_OLS = %.4f（上界）\n", rho_ols))
cat(sprintf("ρ_FE  = ? （下界，用下方gmm估计前先跑含lagy的FE）\n"))
```

---

## System GMM 代码模板（R）

```r
library(plm)
library(lmtest)

# 确保pdata.frame格式
df_panel <- pdata.frame(df, index = c("entity_id", "year"))

# ============================================================
# System GMM（Blundell & Bond, 1998）
# 差分方程 + 水平方程联立，利用水平工具变量
# ============================================================
gmm_sys <- pgmm(
  Y ~ lag(Y, 1) + X + control1 + control2     # 回归方程（含滞后Y）
  | lag(Y, 2:4) + lag(X, 1:2),                # 工具变量（差分方程）
  data            = df_panel,
  effect          = "twoways",        # 双向固定效应
  model           = "twosteps",       # 两步GMM（更有效率，推荐）
  collapse        = TRUE,             # ⚠️ 重要：减少工具变量数量，防止过多
  transformation  = "ld"              # "ld"=System GMM（差分+水平）
)

summary(gmm_sys, robust = TRUE)       # 使用稳健标准误（Windmeijer修正）

# ============================================================
# 差分GMM对比（Arellano-Bond，仅差分方程）
# ============================================================
gmm_diff <- pgmm(
  Y ~ lag(Y, 1) + X + control1 + control2
  | lag(Y, 2:4) + lag(X, 1:2),
  data            = df_panel,
  effect          = "twoways",
  model           = "twosteps",
  collapse        = TRUE,
  transformation  = "d"               # "d"=差分GMM
)
summary(gmm_diff, robust = TRUE)
```

---

## 强制检验清单（自动执行）

估计完成后必须运行以下完整检验函数：

```r
check_gmm <- function(gmm_model, fe_rho = NULL, ols_rho = NULL, N = NULL) {
  cat("=== GMM强制检验清单 ===\n\n")
  sum_gmm <- summary(gmm_model, robust = TRUE)

  # ---- [1] AR检验（⚠️ Fix：mtest分开调用order=1和order=2）----
  # 错误写法：mtest(gmm_model, order=1:2)  ← 可能只返回最后一个
  ar1 <- mtest(gmm_model, order = 1, robust = TRUE)
  ar2 <- mtest(gmm_model, order = 2, robust = TRUE)

  cat(sprintf("[1] AR(1) p = %.4f", ar1$p.value))
  cat(if (ar1$p.value < 0.05) " ✓ 预期中（差分残差应显著AR(1)）\n"
      else " ⚠️ AR(1)不显著，检查差分阶数\n")

  cat(sprintf("[2] AR(2) p = %.4f", ar2$p.value))
  if (ar2$p.value > 0.10) {
    cat(" ✓ 工具变量有效（不显著，支持识别假设）\n")
  } else {
    cat(" ⚠️ 严重：AR(2)显著！增加滞后阶数（如lag(Y, 3:6)）\n")
    cat("   → AR(2)显著意味着误差存在二阶相关，滞后2期的工具变量无效\n")
  }

  # ---- [2] Hansen J检验 ----
  # pgmm的summary中sargan即Hansen（twosteps时报告Hansen）
  hansen_stat <- sum_gmm$sargan
  if (!is.null(hansen_stat)) {
    hansen_p <- hansen_stat$p.value
    cat(sprintf("[3] Hansen p = %.4f", hansen_p))
    if (hansen_p < 0.10) {
      cat(" ⚠️ 工具变量可能无效！检查排他性约束\n")
    } else if (hansen_p > 0.90) {
      cat(" ⚠️ p > 0.90：工具变量过多（Hansen统计量失效），强制collapse=TRUE\n")
    } else {
      cat(" ✓ 工具变量联合有效（p在0.10-0.90之间）\n")
    }
  }

  # ---- [3] 工具变量数量 ----
  n_instr <- length(gmm_model$instruments)
  cat(sprintf("[4] 工具变量数 = %d", n_instr))
  if (!is.null(N)) {
    cat(sprintf("，样本N = %d", N))
    if (n_instr > N) {
      cat(" ⚠️ 必须减少！工具数 > N → 强制collapse=TRUE或缩短滞后阶数\n")
    } else if (n_instr > N * 0.5) {
      cat(" ⚠️ 工具较多（> N/2），建议collapse\n")
    } else {
      cat(" ✓ 在安全范围内\n")
    }
  } else {
    cat(" （建议提供N进行检查）\n")
  }

  # ---- [4] ρ区间检查 ----
  rho_gmm <- tryCatch(coef(gmm_model)["lag(Y, 1)"], error=function(e) NA)
  if (!is.na(rho_gmm)) {
    cat(sprintf("[5] ρ_GMM = %.4f", rho_gmm))
    if (!is.null(fe_rho) && !is.null(ols_rho)) {
      cat(sprintf(" （ρ_FE=%.4f, ρ_OLS=%.4f）", fe_rho, ols_rho))
      if (rho_gmm > fe_rho && rho_gmm < ols_rho) {
        cat(" ✓ GMM在FE-OLS合理区间内\n")
      } else {
        cat(" ⚠️ GMM不在合理区间！检查工具变量设定或滞后阶数\n")
      }
    } else {
      cat("\n   （建议同时提供FE和OLS的ρ进行三点比较）\n")
    }
  }

  # ---- [5] 长期效应计算 ----
  beta_x  <- tryCatch(coef(gmm_model)["X"], error=function(e) NA)
  if (!is.na(rho_gmm) && !is.na(beta_x) && abs(rho_gmm) < 1) {
    lr_effect <- beta_x / (1 - rho_gmm)
    cat(sprintf("\n[6] 短期效应 β = %.4f\n", beta_x))
    cat(sprintf("    长期效应 β/(1-ρ) = %.4f / (1 - %.4f) = %.4f\n",
                beta_x, rho_gmm, lr_effect))
    cat("    ⚠️ 论文中必须区分短期和长期效应（Estimand要求）\n")
  } else if (!is.na(rho_gmm) && abs(rho_gmm) >= 1) {
    cat("\n⚠️ |ρ| ≥ 1，过程不平稳！不能计算长期效应，检查模型设定\n")
  }

  # ---- [6] 样本量警告 ----
  if (!is.null(N) && N < 50) {
    cat("\n⚠️ N < 50：样本量可能不足以支撑GMM（需至少50个个体）\n")
  }

  invisible(list(ar1=ar1, ar2=ar2, rho=rho_gmm,
                 lr_effect=if(!is.na(beta_x) && abs(rho_gmm)<1) beta_x/(1-rho_gmm) else NA))
}

# 使用示例
N_entities <- n_distinct(df$entity_id)
check_result <- check_gmm(gmm_sys, fe_rho = rho_fe, ols_rho = rho_ols, N = N_entities)
```

---

## 警告规则汇总

| 条件 | 警告 | 处理方式 |
|------|------|---------|
| N < 50 | 样本量不足 | 谨慎使用，说明局限 |
| 工具数 > N | 严重过度拟合 | 强制 `collapse=TRUE` |
| 工具数 > N/2 | 工具较多 | 建议 `collapse=TRUE` |
| AR(2) p < 0.10 | 二阶序列相关 | 增加滞后阶数 `lag(Y, 3:6)` |
| Hansen p < 0.10 | 工具变量无效 | 检查排他性约束 |
| Hansen p > 0.90 | Hansen失效 | 减少工具变量 |
| ρ 不在 [ρ_FE, ρ_OLS] | 模型设定问题 | 重新设定工具变量 |
| \|ρ\| ≥ 1 | 不平稳 | 检查变量是否需要差分 |

---

## Estimand 声明

| 效应 | 公式 | 经济含义 |
|------|------|---------|
| 短期效应 | \(\hat{\beta}\) | X变化1单位后，当期Y的即时变化 |
| 长期效应（稳态乘数） | \(\hat{\beta}/(1-\hat{\rho})\) | X持续变化后，Y达到新稳态的总变化 |
| 调整速度 | \(1-\hat{\rho}\) | 每期向稳态调整的比例 |

**声明模板：**
> "本文采用系统GMM（Blundell & Bond, 1998）估计动态面板模型。短期效应 \(\hat{\beta}=X\)，长期效应（稳态乘数）为 \(\hat{\beta}/(1-\hat{\rho})=Y\)，说明处理效应会通过调整过程逐渐放大/收缩。AR(2)检验（p=...）和Hansen J检验（p=...）支持工具变量的有效性。"

---

## 参考文献

- Arellano, M. & Bond, S. (1991). Some tests of specification for panel data. *Review of Economic Studies*, 58(2), 277–297.
- Blundell, R. & Bond, S. (1998). Initial conditions and moment restrictions in dynamic panel data models. *Journal of Econometrics*, 87(1), 115–143.
- Nickell, S. (1981). Biases in dynamic models with fixed effects. *Econometrica*, 49(6), 1417–1426.
- Roodman, D. (2009). How to do xtabond2. *Stata Journal*, 9(1), 86–136.（方法论参考，非Stata使用）
