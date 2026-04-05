# DID 高级工具

本文件在 SKILL.md 基础上提供高级估计量、多估计量比较、合成 DID 及矩阵补全等工具。**按需加载**，不作为默认流程。

**适用场景：** TWFE 负权重 ≥ 10%，或需要向顶刊审稿人展示多估计量一致性。

---

## Stacked DID

**原理：** 为每个处理队列（cohort）构建一个干净的 2×2 数据集，每个 cohort 的对照组仅使用"尚未处理"或"从未处理"的单位。然后将所有 cohort 数据纵向拼接，在 cohort×个体 固定效应下做联合回归。

相比 CS/SA，Stacked DID 直觉更强，与标准 OLS 框架兼容，适合没有 `did` 包的计算环境。

```r
# ============================================================
# Stacked DID — R（fixest + 手动 stacking）
# ============================================================
library(fixest)
library(dplyr)

stack_did <- function(df, outcome, unit_id = "unit_id", time = "time",
                       first_treat = "first_treat",
                       window = c(-4, 4), controls = NULL) {
  """
  为每个 cohort 构建干净 2×2，纵向拼接后估计 Stacked DID。

  Parameters
  ----------
  df          : 面板数据，first_treat = 0 表示从未处理
  window      : c(pre, post)，事件窗口
  controls    : 控制变量向量

  Returns
  -------
  list: feols 结果 + stacked 数据集
  """
  cohorts <- sort(unique(df[[first_treat]]))
  cohorts <- cohorts[cohorts > 0 & is.finite(cohorts)]  # 排除从未处理

  pre_win  <- abs(window[1])
  post_win <- window[2]

  stacked_list <- lapply(cohorts, function(g) {
    t_start <- g - pre_win
    t_end   <- g + post_win

    # 处理组：该 cohort 的单位
    treated_units <- df %>%
      filter(.data[[first_treat]] == g) %>%
      pull(.data[[unit_id]]) %>%
      unique()

    # 对照组：时间窗口内从未处理 + 尚未处理（first_treat > t_end）
    control_units <- df %>%
      filter(.data[[first_treat]] == 0 |
               .data[[first_treat]] > t_end) %>%
      pull(.data[[unit_id]]) %>%
      unique()

    df_cohort <- df %>%
      filter(.data[[unit_id]] %in% c(treated_units, control_units),
             .data[[time]] >= t_start,
             .data[[time]] <= t_end) %>%
      mutate(
        cohort_id  = g,
        # 组合 ID 保证 stacked 数据中同一单位在不同 cohort 视为不同"单位"
        stack_id   = paste0(.data[[unit_id]], "_c", g),
        event_time = .data[[time]] - g,
        post_stack = as.integer(.data[[time]] >= g),
        treat_stack = as.integer(.data[[unit_id]] %in% treated_units),
        did_stack   = post_stack * treat_stack
      )
    df_cohort
  })

  df_stacked <- bind_rows(stacked_list)
  cat(sprintf("Stacked 数据：%d 行，%d 个 cohort\n",
              nrow(df_stacked), length(cohorts)))

  # ── 回归：双向 FE（stack_id + cohort×time）───────────────
  ctrl_str <- if (length(controls) > 0)
    paste("+", paste(controls, collapse = "+")) else ""

  res_stacked <- feols(
    as.formula(sprintf("%s ~ did_stack %s | stack_id + cohort_id^%s",
                        outcome, ctrl_str, time)),
    data    = df_stacked,
    cluster = ~stack_id
  )

  cat("Stacked DID 结果：\n")
  etable(res_stacked,
         title = "Stacked DID 估计量",
         notes = "双向固定效应：个体（cohort 内）+ cohort×时间。聚类：stack_id。")

  # ── 动态效应图 ───────────────────────────────────────────
  res_dyn <- feols(
    as.formula(sprintf("%s ~ i(event_time, treat_stack, ref=-1) %s | stack_id + cohort_id^%s",
                        outcome, ctrl_str, time)),
    data    = df_stacked,
    cluster = ~stack_id
  )
  iplot(res_dyn, main = "Stacked DID 动态效应",
        xlab = "事件时间", ylab = "ATT", col = "#20808D")
  ggsave("output/stacked_did_event_study.png", width = 8, height = 5, dpi = 150)

  return(list(result = res_stacked, dyn_result = res_dyn,
              stacked_df = df_stacked))
}
```

---

## Borusyak, Jaravel & Spiess (2024) 插补估计量

**原理（BJS）：** 利用未处理期的数据估计反事实（固定效应），再将处理后实际结果与反事实之差作为处理效应。比 TWFE 更抗负权重。

**R 包：** `did2s`（基于 Gardner 2021 两步插补，与 BJS 2024 方法相近）

```r
# ============================================================
# Borusyak 插补估计量（BJS）— R（did2s 包）
# install.packages("did2s")
# ============================================================
library(did2s)
library(ggplot2)

run_bjs <- function(df, outcome, unit_id = "unit_id", time = "time",
                     first_treat = "first_treat", controls = NULL) {
  """
  BJS 插补估计量。
  df 中 first_treat = 0 表示从未处理；处理单位 = 首次处理年份。
  """
  ctrl_str  <- if (length(controls) > 0) paste(controls, collapse = "+") else "1"
  first_stage_fml <- as.formula(
    sprintf("~ 0 + %s | %s + %s", ctrl_str, unit_id, time)
  )

  res_bjs <- did2s(
    data          = df,
    yname         = outcome,
    first_stage   = first_stage_fml,
    second_stage  = ~ i(event_time, ref = -1),
    treatment     = "treat_indicator",  # 0/1 在某期是否受处理
    cluster_var   = unit_id
  )

  cat("BJS 估计量结果（动态效应）：\n")
  iplot(res_bjs,
        main = "BJS 插补估计量：动态处理效应",
        xlab = "事件时间", ylab = "ATT")
  ggsave("output/bjs_dynamic.png", width = 8, height = 5, dpi = 150)

  return(res_bjs)
}
```

**注：** `did2s` 需要数据中有 `treat_indicator`（0/1，逐观测处理状态）和 `event_time`（相对处理期，从未处理填 NA）。

---

## de Chaisemartin & D'Haultfoeuille (dCDH)

**原理：** 局部化估计"switchers"（从未处理到处理）的瞬时效应，对效应异质性更稳健。  
**适用：** 处理状态随时间可逆（处理后可回到对照）；对效应动态不做约束。

```r
# ============================================================
# de Chaisemartin & D'Haultfoeuille — R（DIDmultiplegt 包）
# install.packages("DIDmultiplegt")
# ============================================================
library(DIDmultiplegt)

run_dcdh <- function(df, outcome, unit_id = "unit_id", time = "time",
                      treatment = "treated", controls = NULL,
                      n_bootstrap = 100) {
  """
  dCDH 估计量。

  注意：DIDmultiplegt 运行较慢（Bootstrap），建议 n_bootstrap >= 100。
  返回处理期效应 + 事件研究系数。
  """
  ctrl_vec <- if (length(controls) > 0) controls else character(0)

  res_dcdh <- did_multiplegt(
    df        = df,
    Y         = outcome,
    G         = unit_id,
    T         = time,
    D         = treatment,
    controls  = ctrl_vec,
    brep      = n_bootstrap,
    placebo   = 3,      # 处理前 3 期安慰剂
    dynamic   = 3,      # 处理后 3 期动态效应
    cluster   = unit_id
  )

  cat("dCDH 估计量摘要：\n")
  print(res_dcdh)

  # 可视化（手动提取）
  # res_dcdh$effect / res_dcdh$se / res_dcdh$p_value
  cat("\n[提示] dCDH 计算密集，在大样本下可减少 brep 至 50。\n")
  return(res_dcdh)
}
```

---

## 多估计量对比可视化

**使用场景：** 向审稿人展示 TWFE / CS / SA / Stacked / BJS / dCDH 六种估计量系数方向和幅度一致，增强可信度。

```r
# ============================================================
# 多估计量并排系数图 — R（ggplot2）
# ============================================================
library(ggplot2)
library(dplyr)

plot_estimator_comparison <- function(estimates_list) {
  """
  Parameters
  ----------
  estimates_list : named list，每个元素为
    list(coef = numeric, se = numeric, label = character)
    例：
    list(
      TWFE    = list(coef = 0.12, se = 0.03, label = "TWFE"),
      CS      = list(coef = 0.14, se = 0.04, label = "Callaway-Sant'Anna"),
      SA      = list(coef = 0.13, se = 0.03, label = "Sun-Abraham"),
      Stacked = list(coef = 0.15, se = 0.04, label = "Stacked DID"),
      BJS     = list(coef = 0.13, se = 0.03, label = "BJS (did2s)"),
      dCDH    = list(coef = 0.11, se = 0.04, label = "dCDH")
    )
  """
  df_plot <- bind_rows(lapply(names(estimates_list), function(nm) {
    e <- estimates_list[[nm]]
    data.frame(
      estimator = e$label,
      coef      = e$coef,
      ci_lo     = e$coef - 1.96 * e$se,
      ci_hi     = e$coef + 1.96 * e$se,
      order     = which(names(estimates_list) == nm)
    )
  })) %>% arrange(order)

  df_plot$estimator <- factor(df_plot$estimator,
                               levels = rev(df_plot$estimator))

  # 颜色：TWFE 用灰色（可能有偏），其他估计量用蓝绿色
  df_plot$color_group <- ifelse(df_plot$estimator == "TWFE",
                                 "TWFE（参考）", "稳健估计量")

  p <- ggplot(df_plot, aes(x = coef, y = estimator,
                            color = color_group, shape = color_group)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    geom_errorbarh(aes(xmin = ci_lo, xmax = ci_hi),
                   height = 0.2, linewidth = 1) +
    geom_point(size = 4) +
    scale_color_manual(values = c("TWFE（参考）" = "#7A7974",
                                   "稳健估计量"   = "#20808D")) +
    scale_shape_manual(values = c("TWFE（参考）" = 17,
                                   "稳健估计量"   = 19)) +
    labs(
      x       = "处理效应系数（ATT）",
      y       = NULL,
      title   = "多估计量对比：交错 DID 稳健性",
      caption = "水平线为 95% CI；各估计量均控制个体+时间固定效应。",
      color   = NULL, shape = NULL
    ) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom",
          panel.grid.major.y = element_blank())

  ggsave("output/estimator_comparison.png", p,
         width = 8, height = 5, dpi = 150)
  print(p)
  return(p)
}

# 示例调用
# estimates <- list(
#   TWFE    = list(coef = 0.12, se = 0.03, label = "TWFE"),
#   CS      = list(coef = 0.14, se = 0.04, label = "Callaway-Sant'Anna"),
#   SA      = list(coef = 0.13, se = 0.03, label = "Sun-Abraham"),
#   Stacked = list(coef = 0.15, se = 0.04, label = "Stacked DID"),
#   BJS     = list(coef = 0.13, se = 0.03, label = "BJS (did2s)"),
#   dCDH    = list(coef = 0.11, se = 0.04, label = "dCDH")
# )
# plot_estimator_comparison(estimates)
```

---

## Synthetic DID（synthdid）

**原理：** 结合合成控制（加权对照单位）与 DID（加权时间期），同时对单位和时间进行加权，减少对平行趋势假设的依赖。适合处理单位数量少的场景（如少数省份/国家接受政策）。

```r
# ============================================================
# Synthetic DID — R（synthdid 包）
# install.packages("synthdid")
# ============================================================
library(synthdid)

run_synthdid <- function(df, outcome, unit_id = "unit_id",
                          time = "time", treatment = "treated") {
  """
  Synthdid 估计量。
  适用：处理单位数量较少（< 20）；有足够长的处理前时期。

  注意：synthdid 需要宽格式（单位 × 时间矩阵）。
  """
  # ── 转换为矩阵格式 ────────────────────────────────────────
  setup_obj <- panel.matrices(
    as.data.frame(df),
    unit    = unit_id,
    time    = time,
    outcome = outcome,
    treatment = treatment
  )

  # ── 三种估计量 ──────────────────────────────────────────
  tau_sdid  <- synthdid_estimate(setup_obj$Y, setup_obj$N0,
                                  setup_obj$T0)
  tau_sc    <- sc_estimate(setup_obj$Y, setup_obj$N0, setup_obj$T0)
  tau_did   <- did_estimate(setup_obj$Y, setup_obj$N0, setup_obj$T0)

  estimates <- list(synthdid = tau_sdid, sc = tau_sc, did = tau_did)
  se_vals   <- lapply(estimates, function(e) sqrt(vcov(e, method = "placebo")))

  cat("Synthetic DID 结果对比：\n")
  cat(sprintf("  Synthdid : %.4f  SE: %.4f\n", tau_sdid, se_vals$synthdid))
  cat(sprintf("  SC       : %.4f  SE: %.4f\n", tau_sc,   se_vals$sc))
  cat(sprintf("  DID      : %.4f  SE: %.4f\n", tau_did,  se_vals$did))

  # ── 可视化 ───────────────────────────────────────────────
  png("output/synthdid_plot.png", width = 900, height = 500, res = 120)
  synthdid_plot(tau_sdid, se.method = "placebo",
                title = "Synthetic DID 估计量")
  dev.off()

  return(list(synthdid = tau_sdid, sc = tau_sc, did = tau_did,
              se = se_vals))
}
```

---

## fect（矩阵补全反事实）

**原理：** 用因子模型估计处理单位的反事实，适合处理存在较强时间趋势或因子结构的场景。比合成控制更灵活，支持多处理单位。

```r
# ============================================================
# fect 矩阵补全 — R（fect 包）
# install.packages("fect")
# ============================================================
library(fect)
library(ggplot2)

run_fect <- function(df, outcome, unit_id = "unit_id", time = "time",
                      treatment = "treated", controls = NULL,
                      method = "mc") {
  """
  fect 反事实估计量。

  Parameters
  ----------
  method : 'ife'（交互固定效应）/ 'mc'（矩阵补全，默认）/ 'fe'（双向 FE）

  Returns
  -------
  fect 结果对象
  """
  ctrl_fml <- if (length(controls) > 0)
    as.formula(paste(outcome, "~", treatment, "+",
                     paste(controls, collapse = "+")))
  else
    as.formula(paste(outcome, "~", treatment))

  res_fect <- fect(
    formula  = ctrl_fml,
    data     = df,
    index    = c(unit_id, time),
    force    = "two-way",
    method   = method,
    CV       = TRUE,     # 交叉验证选 r（因子数）
    r        = c(0, 5),  # 因子数范围
    se       = TRUE,
    nboots   = 200,
    parallel = TRUE,
    cores    = 4
  )

  cat(sprintf("fect 估计量（方法：%s）：\n", method))
  print(res_fect)

  # ── 可视化 ───────────────────────────────────────────────
  plot(res_fect,
       type     = "gap",
       main     = sprintf("fect（%s）：处理效应动态图", method),
       ylab     = "ATT")

  # 平行趋势检验（fect 内置）
  plot(res_fect,
       type     = "equiv",
       main     = "fect 等价性检验（平行趋势）",
       bound    = "equiv")

  return(res_fect)
}
```

---

## 交错 DID 专项安慰剂（CS 框架下）

在 Callaway-Sant'Anna 框架内做安慰剂：打乱 `first_treat`（而非 `treated`），保持 cohort 结构，更适合交错 DID 的内部逻辑。

```r
# ============================================================
# CS 框架安慰剂 — R
# ============================================================
library(did)
library(dplyr)

cs_placebo <- function(df, outcome, unit_id = "unit_id", time = "time",
                        first_treat = "first_treat", controls = NULL,
                        n_perm = 200, seed = 42) {
  """
  在 CS 框架下打乱 first_treat（cohort 归属），估计安慰剂 ATT。
  仅打乱处理单位的 cohort 归属，从未处理单位保持不变。
  """
  set.seed(seed)

  # 真实 CS 估计
  out_true <- att_gt(
    yname         = outcome,
    gname         = first_treat,
    idname        = unit_id,
    tname         = time,
    xformla       = if (!is.null(controls))
                      as.formula(paste("~", paste(controls, collapse="+")))
                    else ~ 1,
    data          = df,
    control_group = "nevertreated"
  )
  true_att <- aggte(out_true, type = "simple")$overall.att
  cat(sprintf("真实 CS ATT：%.4f\n", true_att))

  # 处理单位的 cohort 值
  treat_units <- df %>%
    filter(.data[[first_treat]] > 0 & is.finite(.data[[first_treat]])) %>%
    distinct(.data[[unit_id]], .data[[first_treat]])

  fake_atts <- numeric(n_perm)

  for (i in seq_len(n_perm)) {
    # 打乱处理单位的 cohort 归属
    shuffled_ft <- treat_units %>%
      mutate(fake_ft = sample(.data[[first_treat]]))

    df_fake <- df %>%
      left_join(shuffled_ft %>% select(.data[[unit_id]], fake_ft),
                by = unit_id) %>%
      mutate(!!first_treat := ifelse(!is.na(fake_ft), fake_ft,
                                      .data[[first_treat]])) %>%
      select(-fake_ft)

    out_fake <- tryCatch(
      att_gt(yname = outcome, gname = first_treat, idname = unit_id,
             tname = time, xformla = ~ 1, data = df_fake,
             control_group = "nevertreated"),
      error = function(e) NULL
    )
    fake_atts[i] <- if (!is.null(out_fake))
      aggte(out_fake, type = "simple")$overall.att
    else NA_real_
  }

  fake_atts <- na.omit(fake_atts)
  p_val     <- mean(abs(fake_atts) >= abs(true_att))
  cat(sprintf("CS 安慰剂经验 p 值：%.4f（%d 次置换）\n",
              p_val, length(fake_atts)))

  # 核密度图
  p <- ggplot(data.frame(x = fake_atts), aes(x = x)) +
    geom_density(fill = "#20808D", alpha = 0.3, color = "#1B474D") +
    geom_vline(xintercept = true_att, color = "#A84B2F",
               linewidth = 1.5, linetype = "dashed") +
    labs(title = sprintf("CS 框架安慰剂（%d 次置换）", length(fake_atts)),
         x = "安慰剂 ATT", y = "核密度",
         caption = sprintf("经验 p = %.4f", p_val)) +
    theme_minimal(base_size = 12)

  ggsave("output/cs_placebo.png", p, width = 7, height = 4, dpi = 150)
  print(p)

  return(list(fake_atts = fake_atts, p_value = p_val))
}
```

---

## 连续处理 DID（Intensity DID）

**警告：** 标准 TWFE 不适用于连续处理量（treatment intensity）。连续处理的识别假设更强，文献中对方法论的争议尚未完全解决。

**参考文献：**
- Callaway, Goodman-Bacon & Sant'Anna (2021). "Difference-in-Differences with a Continuous Treatment." *NBER WP 30108*
- de Chaisemartin & D'Haultfoeuille (2024). "Two-Way Fixed Effects and Differences-in-Differences Estimators with Several Treatments." *Journal of Econometrics*

**处理建议：**

| 情形 | 建议做法 |
|------|----------|
| 处理强度有自然阈值 | 离散化为 0/1（高强度 vs 低强度），声明切割点 |
| 需要量化剂量-响应关系 | Callaway et al. (2021) 连续处理扩展版（R: 暂无稳定包，参考作者 GitHub） |
| 连续处理来自随机分配 | 直接 OLS，无需 DID 框架 |
| 连续处理 + 交错时间 | 高难度：需声明更强假设，必须在论文中充分讨论识别策略 |

```r
# ============================================================
# 连续处理强度（分组离散化方案）— R
# 将连续处理离散化后使用标准 CS 估计量
# ============================================================
library(dplyr)

discretize_treatment <- function(df, intensity_var,
                                  threshold = NULL,
                                  method = "median") {
  """
  将连续处理强度离散化。

  Parameters
  ----------
  intensity_var : 连续处理变量列名
  threshold     : 手动设定阈值（NULL = 自动计算）
  method        : 'median'（中位数二值化）/
                  'tertile'（三分位，低/中/高）/
                  'quartile'（四分位）

  注：离散化会损失信息，且切割点选择主观。
      在论文中必须报告多个切割点的稳健性（敏感性分析）。
  """
  vals <- df[[intensity_var]]

  if (method == "median") {
    thr <- threshold %||% median(vals, na.rm = TRUE)
    df$treat_discrete <- as.integer(vals >= thr)
    cat(sprintf("[离散化] 中位数切割：%.4f\n", thr))
    cat("[声明] 请在论文中报告 25th 和 75th 百分位切割点的稳健性结果。\n")

  } else if (method == "tertile") {
    q33 <- quantile(vals, 1/3, na.rm = TRUE)
    q67 <- quantile(vals, 2/3, na.rm = TRUE)
    df$treat_discrete <- case_when(
      vals <= q33 ~ 0L,
      vals <= q67 ~ 1L,
      TRUE        ~ 2L
    )
    cat("[离散化] 三分位：低=0, 中=1, 高=2\n")
    cat("[注意] 三分位分组后需修改 CS 估计量为多处理组版本。\n")

  } else if (method == "quartile") {
    df$treat_discrete <- ntile(vals, 4)
    cat("[离散化] 四分位分组（1-4）\n")
  }

  cat("[警告] 连续处理识别假设更强。请在论文中充分讨论以下问题：\n")
  cat("  1. 为什么处理强度可以视为外生？\n")
  cat("  2. 高强度组和低强度组的平行趋势是否合理？\n")
  cat("  3. 参考：Callaway, Goodman-Bacon & Sant'Anna (2021)\n")

  return(df)
}
```

---

## 估计量选择速查表

| 估计量 | 方法论要求 | 计算复杂度 | R 包 | 适用期刊风格 |
|--------|-----------|-----------|------|-------------|
| **TWFE** | 效应时间不变（强假设） | 低 | fixest | 所有期刊（基准） |
| **CS** | 平行趋势 + 从未处理对照 | 中 | did | 顶刊（AER/QJE/ReStat）|
| **SA** | 平行趋势 + 不含负权重 | 低 | fixest | 顶刊（与 CS 等价）|
| **Stacked** | 干净 2×2 平行趋势 | 中 | fixest（手动）| 顶刊附录（直觉强）|
| **BJS** | 平行趋势 + 插补有效 | 中 | did2s | 顶刊附录 |
| **dCDH** | 仅 switchers 平行趋势 | 高 | DIDmultiplegt | 方法论 / 顶刊 |
| **Synthdid** | 单位权重收敛 | 中 | synthdid | 处理单位少时（< 20）|
| **fect** | 因子模型稳定 | 高 | fect | 政治学 / 社会学顶刊 |
