#!/usr/bin/env Rscript
# =============================================================================
# DID 分析模板脚本 (R 版)
# 核心包：fixest (TWFE/事件研究) + did (Callaway-Sant'Anna 交错 DID)
# 对标：did_analysis.py（Python 版）
#
# 使用方式：修改"配置区"中的变量名后直接运行
#   Rscript did_analysis.R
# 或在 RStudio 中 Source
# =============================================================================

# =============================================================================
# 包依赖（首次运行自动安装）
# =============================================================================
packages <- c("fixest", "did", "tidyverse", "modelsummary", "ggplot2",
              "patchwork", "scales", "haven", "data.table")

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("Installing %s ...", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

suppressPackageStartupMessages({
  library(fixest)       # TWFE 主力包：feols, i(), etable
  library(did)          # Callaway-Sant'Anna att_gt
  library(tidyverse)    # 数据清洗 dplyr + 绘图 ggplot2
  library(modelsummary) # 模型汇总表（可选，etable 已足够）
  library(scales)       # 坐标轴美化
  library(data.table)   # fread 快速读取
})

# =============================================================================
# 配置区 — 根据实际研究修改以下变量名
# =============================================================================

DATA_PATH      <- "data.csv"        # 数据文件路径（支持 .csv / .dta）
Y_VAR          <- "outcome"         # 因变量（连续变量）
TREAT_VAR      <- "treated"         # 处理组虚拟变量（0/1，时不变）
POST_VAR       <- "post"            # 政策后虚拟变量（0/1）
ENTITY_VAR     <- "entity_id"       # 个体标识（用于固定效应和聚类）
TIME_VAR       <- "year"            # 时间变量（数值型）
EVENT_TIME_VAR <- "event_time"      # 相对事件时间（year - treatment_year）
TREAT_YEAR_VAR <- "treat_year"      # 处理年份变量（交错DID使用；对照组设0或Inf）
CLUSTER_VAR    <- ENTITY_VAR        # 聚类层级（一般与 ENTITY_VAR 相同）

CONTROLS <- c("control1", "control2", "control3")  # 控制变量列表

OUTPUT_DIR <- "output"

# 事件研究窗口期
LEADS <- 4   # 政策前看多少期（pre-treatment leads）
LAGS  <- 4   # 政策后看多少期（post-treatment lags）
REF   <- -1  # 基准期（omitted period），一般取 -1

# =============================================================================
# 辅助函数
# =============================================================================

#' 创建输出目录
setup_dirs <- function() {
  dirs <- c(
    file.path(OUTPUT_DIR, "tables"),
    file.path(OUTPUT_DIR, "figures")
  )
  for (d in dirs) dir.create(d, showWarnings = FALSE, recursive = TRUE)
}

#' 打印分隔线
sep <- function(title = "") {
  cat("\n", strrep("=", 60), "\n", title, "\n", strrep("=", 60), "\n", sep = "")
}

# 控制变量 formula 字符串（fixest 语法）
controls_str <- if (length(CONTROLS) > 0) paste(CONTROLS, collapse = " + ") else "1"

# =============================================================================
# 1. 数据加载与准备
# =============================================================================

load_and_prepare <- function() {
  sep("1. 数据加载")

  # 支持 CSV 和 Stata dta 文件
  if (grepl("\\.dta$", DATA_PATH)) {
    df <- haven::read_dta(DATA_PATH)
  } else {
    df <- data.table::fread(DATA_PATH) |> as_tibble()
  }

  # 构建交乘项
  df <- df |>
    mutate(
      treat_post = .data[[TREAT_VAR]] * .data[[POST_VAR]]
    )

  # 基本信息
  cat(sprintf("样本量:       %d 行\n", nrow(df)))
  cat(sprintf("变量数:       %d 列\n", ncol(df)))
  cat(sprintf("时间范围:     %s — %s\n",
              min(df[[TIME_VAR]]), max(df[[TIME_VAR]])))
  cat(sprintf("个体数:       %d\n",
              n_distinct(df[[ENTITY_VAR]])))
  cat(sprintf("处理组 obs:   %d (%.1f%%)\n",
              sum(df[[TREAT_VAR]]), mean(df[[TREAT_VAR]]) * 100))

  return(df)
}

# =============================================================================
# 2. Table 1: 描述性统计
# =============================================================================

descriptive_stats <- function(df) {
  sep("2. 描述性统计")

  vars <- c(Y_VAR, CONTROLS)

  # 分组均值、标准差
  stats <- df |>
    select(all_of(c(vars, TREAT_VAR))) |>
    pivot_longer(-all_of(TREAT_VAR), names_to = "variable") |>
    group_by(variable, group = .data[[TREAT_VAR]]) |>
    summarise(
      N    = n(),
      Mean = mean(value, na.rm = TRUE),
      SD   = sd(value, na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(group = ifelse(group == 1, "Treated", "Control")) |>
    pivot_wider(names_from = group,
                values_from = c(N, Mean, SD))

  # 均值差 t 检验
  t_results <- sapply(vars, function(v) {
    x1 <- df[df[[TREAT_VAR]] == 1, v, drop = TRUE]
    x0 <- df[df[[TREAT_VAR]] == 0, v, drop = TRUE]
    t.test(x1, x0)$p.value
  })

  stats$`Diff p-val` <- round(t_results, 3)
  print(stats)

  # 保存
  write_csv(stats, file.path(OUTPUT_DIR, "tables", "table1_descriptive.csv"))
  cat("\n=> Saved: output/tables/table1_descriptive.csv\n")

  return(stats)
}

# =============================================================================
# 3. 主回归: TWFE DID（fixest::feols）
# =============================================================================

main_regression <- function(df) {
  sep("3. 主回归 TWFE DID")

  # Formula 构建（fixest 语法）
  # `| entity_id + year` 表示固定效应
  # `cluster = ~entity_id` 聚类标准误

  fml_base <- as.formula(
    sprintf("%s ~ treat_post | %s + %s",
            Y_VAR, ENTITY_VAR, TIME_VAR)
  )
  fml_ctrl <- as.formula(
    sprintf("%s ~ treat_post + %s | %s + %s",
            Y_VAR, controls_str, ENTITY_VAR, TIME_VAR)
  )

  # 拟合（双向聚类：个体+时间）
  m1 <- feols(fml_base, data = df, cluster = as.formula(paste0("~", CLUSTER_VAR)))
  m2 <- feols(fml_ctrl, data = df, cluster = as.formula(paste0("~", CLUSTER_VAR)))

  # 汇总输出（etable 格式，类似 esttab）
  cat("\n--- Baseline DID ---\n")
  etable(m1, m2,
         dict = c(treat_post = "Treat × Post"),
         fitstat = ~ r2 + n,
         stars = c("*" = 0.10, "**" = 0.05, "***" = 0.01))

  # 保存为 LaTeX（可选）
  etable(m1, m2,
         dict = c(treat_post = "Treat × Post"),
         fitstat = ~ r2 + n,
         stars = c("*" = 0.10, "**" = 0.05, "***" = 0.01),
         file = file.path(OUTPUT_DIR, "tables", "table2_main_results.tex"),
         replace = TRUE)

  cat("=> Saved: output/tables/table2_main_results.tex\n")
  return(list(m1 = m1, m2 = m2))
}

# =============================================================================
# 4. 事件研究（平行趋势检验）
# =============================================================================

event_study <- function(df) {
  sep("4. 事件研究（平行趋势检验）")

  # fixest i() 语法：i(event_time, treated, ref = -1)
  # 自动生成交乘虚拟变量，ref 期系数归零
  fml_es <- as.formula(
    sprintf("%s ~ i(%s, %s, ref = %d) + %s | %s + %s",
            Y_VAR,
            EVENT_TIME_VAR, TREAT_VAR, REF,
            controls_str,
            ENTITY_VAR, TIME_VAR)
  )

  es_model <- feols(fml_es, data = df,
                    cluster = as.formula(paste0("~", CLUSTER_VAR)))

  # 提取系数和置信区间
  coef_df <- iplot(es_model, plot = FALSE)  # 返回数据框
  # 或手动提取：
  # coef_df <- broom::tidy(es_model, conf.int = TRUE) |>
  #   filter(str_detect(term, EVENT_TIME_VAR))

  # ── 绘图 ──────────────────────────────────────────────
  p_es <- iplot(
    es_model,
    xlab      = "Relative Time to Policy",
    ylab      = "Coefficient (95% CI)",
    main      = "Event Study: Parallel Trends Test",
    ref.line  = 0,
    ci_level  = 0.95,
    zero.line = TRUE,
    pt.join   = TRUE
  )
  # fixest iplot 返回 base R 图形；转存 ggplot 版本

  # ggplot2 版本（更灵活）
  coef_tidy <- as.data.frame(coef_df)
  p_gg <- ggplot(coef_tidy, aes(x = estimate, y = coef, ymin = ci_low, ymax = ci_high)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    geom_vline(xintercept = REF + 0.5, linetype = "dashed",
               color = "#e74c3c", alpha = 0.7) +
    geom_errorbar(width = 0.2, color = "#2c3e50", alpha = 0.8) +
    geom_point(size = 2.5, color = "#2c3e50") +
    geom_line(color = "#2c3e50", alpha = 0.6) +
    labs(
      x = "Relative Time to Policy",
      y = "Coefficient",
      title = "Event Study: Parallel Trends Test",
      caption = "Notes: 95% confidence intervals. Standard errors clustered at entity level.\nDashed vertical line = policy implementation. Reference period = -1."
    ) +
    theme_bw(base_size = 12) +
    theme(plot.caption = element_text(hjust = 0, size = 9))

  # fixest 内置 iplot 直接保存 PDF
  pdf(file.path(OUTPUT_DIR, "figures", "fig2_event_study.pdf"),
      width = 9, height = 5.5)
  iplot(es_model,
        xlab = "Relative Time to Policy",
        ylab = "Coefficient (95% CI)",
        main = "Event Study: Parallel Trends Test")
  dev.off()

  ggsave(file.path(OUTPUT_DIR, "figures", "fig2_event_study.png"),
         plot = p_gg, width = 9, height = 5.5, dpi = 300)

  cat("=> Saved: output/figures/fig2_event_study.png\n")

  # ── 预处理系数联合显著性检验 ──────────────────────────
  # 提取所有 event_time < 0 的系数
  pre_vars <- grep(sprintf("%s::%s::", EVENT_TIME_VAR,
                            paste(seq(-LEADS, -2), collapse = "|")),
                   names(coef(es_model)), value = TRUE)
  if (length(pre_vars) > 0) {
    wt <- wald(es_model, pre_vars, print = FALSE)
    cat(sprintf("\n预处理期联合检验: F = %.3f, p = %.4f\n",
                wt$stat, wt$p))
    cat("（p > 0.10 表明平行趋势检验通过）\n")
  }

  return(es_model)
}

# =============================================================================
# 5. 交错 DID（Callaway & Sant'Anna 2021 / Sun & Abraham 2021）
# =============================================================================

staggered_did <- function(df) {
  sep("5. 交错 DID")

  # ── 方法1：fixest sunab（Sun & Abraham 2021）──────────────
  cat("\n--- 5a. Sun & Abraham (2021) via fixest ---\n")

  # 需要 cohort 变量：处理年份（从未处理 → 0 或 Inf）
  if (TREAT_YEAR_VAR %in% names(df)) {
    fml_sa <- as.formula(
      sprintf("%s ~ sunab(%s, %s) + %s | %s + %s",
              Y_VAR,
              TREAT_YEAR_VAR, TIME_VAR,
              controls_str,
              ENTITY_VAR, TIME_VAR)
    )

    sa_model <- feols(fml_sa, data = df,
                      cluster = as.formula(paste0("~", CLUSTER_VAR)))

    cat("Sun & Abraham 聚合处理效应:\n")
    print(aggregate(sa_model, agg = "period"))  # 按处理后相对时间聚合

    iplot(sa_model, main = "Sun & Abraham Event Study")

  } else {
    cat("(跳过：数据中未找到 TREAT_YEAR_VAR = '", TREAT_YEAR_VAR, "')\n")
    sa_model <- NULL
  }

  # ── 方法2：did 包（Callaway & Sant'Anna 2021）──────────────
  cat("\n--- 5b. Callaway & Sant'Anna (2021) via did package ---\n")

  if (TREAT_YEAR_VAR %in% names(df)) {
    # att_gt 要求：
    #   idname = 个体变量
    #   tname  = 时间变量
    #   gname  = 处理组别（首次处理年份，从未处理=0）
    #   yname  = 因变量

    df_cs <- df |>
      mutate(
        gname = ifelse(is.na(.data[[TREAT_YEAR_VAR]]) |
                         .data[[TREAT_YEAR_VAR]] == 0,
                       0, .data[[TREAT_YEAR_VAR]])
      )

    cs_out <- att_gt(
      yname         = Y_VAR,
      tname         = TIME_VAR,
      idname        = ENTITY_VAR,
      gname         = "gname",
      xformla       = if (length(CONTROLS) > 0)
        as.formula(paste("~", controls_str)) else ~1,
      data          = df_cs,
      est_method    = "reg",      # "reg" = 回归调整; "ipw" = IPW; "dr" = 双重稳健
      control_group = "nevertreated",  # 对照组：从未处理 or "notyettreated"
      clustervars   = ENTITY_VAR,
      print_details = FALSE
    )

    # 聚合：简单聚合 ATT
    agg_simple <- aggte(cs_out, type = "simple")
    cat(sprintf("CS ATT (overall): %.4f (SE: %.4f, p = %.4f)\n",
                agg_simple$overall.att,
                agg_simple$overall.se,
                2 * pnorm(-abs(agg_simple$overall.att / agg_simple$overall.se))))

    # 动态聚合（事件研究）
    agg_dynamic <- aggte(cs_out, type = "dynamic")
    ggdid(agg_dynamic,
          title = "Callaway & Sant'Anna: Dynamic ATT",
          ylab  = "ATT",
          xlab  = "Relative Time") +
      theme_bw()
    ggsave(file.path(OUTPUT_DIR, "figures", "fig2b_cs_event_study.png"),
           width = 9, height = 5.5, dpi = 300)

    cat("=> Saved: output/figures/fig2b_cs_event_study.png\n")
    return(list(sa = sa_model, cs = cs_out))
  } else {
    cat("(跳过：数据中未找到 TREAT_YEAR_VAR)\n")
    return(NULL)
  }
}

# =============================================================================
# 6. 安慰剂检验
# =============================================================================

placebo_test <- function(df, true_event_year, placebo_years) {
  sep("6. 安慰剂检验")

  # 仅使用政策实施前的数据
  df_pre <- df |> filter(.data[[TIME_VAR]] < true_event_year)

  results <- list()
  for (py in placebo_years) {
    df_pre <- df_pre |>
      mutate(
        placebo_post = as.integer(.data[[TIME_VAR]] >= py),
        placebo_tp   = .data[[TREAT_VAR]] * placebo_post
      )

    fml_p <- as.formula(
      sprintf("%s ~ placebo_tp + %s | %s + %s",
              Y_VAR, controls_str, ENTITY_VAR, TIME_VAR)
    )
    m_p <- feols(fml_p, data = df_pre,
                 cluster = as.formula(paste0("~", CLUSTER_VAR)))

    coef_p   <- coef(m_p)["placebo_tp"]
    se_p     <- se(m_p)["placebo_tp"]
    pval_p   <- pvalue(m_p)["placebo_tp"]
    stars_p  <- ifelse(pval_p < 0.01, "***",
                       ifelse(pval_p < 0.05, "**",
                              ifelse(pval_p < 0.10, "*", "")))

    cat(sprintf("Placebo year = %d: β = %.4f%s (SE = %.4f, p = %.4f)\n",
                py, coef_p, stars_p, se_p, pval_p))

    results[[as.character(py)]] <- list(coef = coef_p, se = se_p, pval = pval_p)
  }

  cat("\n（安慰剂期系数均不显著则支持主结果的可信性）\n")
  return(results)
}

# =============================================================================
# 7. 异质性分析
# =============================================================================

heterogeneity_analysis <- function(df, het_var, group_labels = NULL) {
  sep(sprintf("7. 异质性分析 (by %s)", het_var))

  groups  <- sort(unique(df[[het_var]]))
  results <- list()

  for (g in groups) {
    df_g <- df |> filter(.data[[het_var]] == g)
    fml_g <- as.formula(
      sprintf("%s ~ treat_post + %s | %s + %s",
              Y_VAR, controls_str, ENTITY_VAR, TIME_VAR)
    )
    m_g <- feols(fml_g, data = df_g,
                 cluster = as.formula(paste0("~", CLUSTER_VAR)))
    results[[as.character(g)]] <- m_g
  }

  # etable 对比输出
  cat("\n--- 异质性回归对比 ---\n")
  do.call(etable, c(results,
                    list(dict  = c(treat_post = "Treat × Post"),
                         heads = if (!is.null(group_labels)) group_labels
                                 else paste0(het_var, "=", groups))))

  return(results)
}

# =============================================================================
# 主程序
# =============================================================================

main <- function() {
  setup_dirs()

  sep("DID Analysis Pipeline (R / fixest)")

  # 1. 加载数据
  df <- load_and_prepare()

  # 2. 描述性统计
  table1 <- descriptive_stats(df)

  # 3. 主回归
  main_res <- main_regression(df)

  # 4. 事件研究（平行趋势）
  es_model <- event_study(df)

  # 5. 交错 DID（若数据含处理年份变量）
  # staggered_res <- staggered_did(df)

  # 6. 安慰剂检验（取消注释并修改参数）
  # placebo_res <- placebo_test(df,
  #   true_event_year = 2015,
  #   placebo_years   = c(2012, 2013))

  # 7. 异质性分析（取消注释并修改 het_var）
  # het_res <- heterogeneity_analysis(df, het_var = "size_group")

  sep("Analysis Complete")
  cat("输出文件位于:", file.path(getwd(), OUTPUT_DIR), "\n")
}

# 执行主程序
main()
