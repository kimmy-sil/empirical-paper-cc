---
name: synthetic-control
description: "合成控制法，含 SCM 权重优化、SDID、安慰剂检验、批量结果变量、现代推断"
---

# 合成控制法 (SCM / SDID)

## 探索模式说明（Agent 自动路由）

当用户上传数据但**未指定识别策略**时，Agent 应：
1. 运行「开始前 5 问」自检
2. 根据自检结果走方法决策树，自动选择 SCM / SDID / augsynth / gsynth
3. 先跑主流程，再提示用户选择稳健性选项

---

## 概述

合成控制法（Abadie, Diamond & Hainmueller 2010, 2015）通过将多个对照单位的加权组合构造"合成控制组"，与处理单位对比，识别政策或事件的因果效应。特别适合**只有一个或少数几个处理单位**（如一个省份、国家、企业）的情形。

**SCM 的三条核心假设（不是"没有假设"）：**
1. 处理单位在处理前的结果路径和 predictors 能被供体池的**加权凸组合**较好逼近（凸包假设）
2. 干预后**没有其他只影响处理单位的同步冲击**（exclusion restriction）
3. 供体池单位在整个研究期间**未受到相似政策干扰**（clean donor pool）

SCM 不要求 DID 式的平行趋势假设，但上述三条假设同样需要论证，不能省略。

---

## 开始前 5 问（必做自检）

拿到数据，先回答这 5 个问题，决定能否用 SCM 以及用哪种变体：

```
Q1. 处理单位是否在供体池的凸包内？
    → 检验：预处理期 RMSPE < 10%，且最大供体权重 < 80%
    → 若否：换 SDID 或 augsynth（放松凸包约束）

Q2. 处理前期是否足够长，且明显大于 predictor 数？
    → 规则：pre_T > max(5, 2 × n_predictors)
    → 若否：精简 predictor set，或换 augsynth

Q3. 处理期间是否存在同期其他重大冲击？
    → 例：政策实施年同时发生经济危机、自然灾害
    → 若是：无法用 SCM 干净识别，需在论文中明确说明局限

Q4. 供体池是否被相似政策污染？
    → 例：研究某省限购令，供体池中是否有其他省在研究期也实施了限购
    → 若是：将受污染单位移出供体池

Q5. 供体池数量是否足够支持置换推断？
    → 规则：J ≥ 10（最小 p = 1/(J+1)）；J ≥ 20 才能达 5% 显著水平
    → 若否：改用 scpi 预测区间（不依赖供体数量）
```

---

## 方法选择决策树

```
单个处理单位 + 供体池质量高（凸包内，RMSPE < 10%）+ 预处理期充足
    → SCM（tidysynth / scpi）

单个/少数处理单位 + 想放松凸包约束 / 提高稳健性
    → SDID（synthdid）

供体池较小（< 20）或预处理拟合一般（RMSPE 10-20%）
    → augsynth（偏误修正 + 自带标准误）

多个处理单位 / 交错处理
    → gsynth / fect / scpi(staggered)

处理单位很多 + 平行趋势更可信
    → DID / event study（跳出本 skill，改用 did-analysis）
```

---

## 前置条件

### 数据结构要求

```
必须包含：
  - 个体 ID 列（unit_id）
  - 时间列（time），数值型，年份/期数
  - 结果变量（outcome）
  - 预测变量（predictors）：处理前特征，用于构造合成控制权重
  - 处理单位标识（如 unit_id = "GuangDong" 或 treated = 1）
  - 处理时间（treatment_year）：政策实施的第一期

数据必须为平衡面板（每个单位每期均有观测）
处理前期数 >> 预测变量数（避免过拟合合成控制）
供体池单位不得在研究期间受到相似政策干扰
```

### Predictor 选择原则（方法论）

三类合法 predictor：
1. **结构性协变量**：如道路面积、人口、产业结构（处理前稳定特征）
2. **干预前 outcome 的分段均值**：如处理前早期、中期、晚期各段的 outcome 均值
3. **不受政策影响的预处理特征**：地理、历史数据

⚠️ **Bad control 警告**：
- 不要把处理后变量放进 predictor set
- 不要把潜在中介变量（mechanism 变量）放进 predictor set
- 不要把处理期内的协变量放进去

### 前置检查代码

```r
library(dplyr)
library(tidyr)

treated_unit   <- "GuangDong"
treatment_year <- 2010
pre_years      <- min(df$time):(treatment_year - 1)
post_years     <- treatment_year:max(df$time)
donor_pool     <- unique(df$unit_id[df$unit_id != treated_unit])

# 确认平衡面板
balance_check <- df %>% count(unit_id, time) %>% filter(n > 1)
if (nrow(balance_check) > 0) stop("存在重复观测，数据非平衡面板")

cat(sprintf("处理单位: %s\n", treated_unit))
cat(sprintf("供体池: %d 个单位\n", length(donor_pool)))
cat(sprintf("处理前期数: %d\n", length(pre_years)))
cat(sprintf("处理后期数: %d\n", length(post_years)))

if (length(pre_years) < 5)
  cat("⚠️  处理前期数 < 5，合成控制权重不可靠\n")
if (length(donor_pool) < 10)
  cat("⚠️  供体池 < 10，置换检验 p 值分辨率低（最小 p ≈ 1/J）\n")
if (length(donor_pool) < 20)
  cat("⚠️  供体 < 20，无法达到 5% 显著水平，推荐改用 scpi 预测区间\n")
```

---

## Step 1：描述性分析

```python
# Python: 处理单位 vs 供体池趋势图
import matplotlib.pyplot as plt
import pandas as pd

def plot_raw_trends(df, unit_col, time_col, outcome_col,
                    treated_unit, treatment_year):
    fig, ax = plt.subplots(figsize=(10, 5))
    for unit in df[unit_col].unique():
        if unit == treated_unit:
            continue
        sub = df[df[unit_col] == unit].sort_values(time_col)
        ax.plot(sub[time_col], sub[outcome_col],
                color='grey', alpha=0.3, linewidth=0.8)
    treated = df[df[unit_col] == treated_unit].sort_values(time_col)
    ax.plot(treated[time_col], treated[outcome_col],
            color='steelblue', linewidth=2.5, label=str(treated_unit))
    ax.axvline(treatment_year - 0.5, color='tomato',
               linestyle='--', linewidth=1.5, label='Treatment')
    ax.set_xlabel('Year'); ax.set_ylabel('Outcome')
    ax.set_title('Raw Trends: Treated vs Donor Pool')
    ax.legend(); plt.tight_layout()
    return fig

fig = plot_raw_trends(df, 'unit_id', 'year', 'outcome',
                      treated_unit='GuangDong', treatment_year=2010)
fig.savefig('output/sc_raw_trends.png', dpi=150)
```

```r
# R: ggplot2 趋势图
library(ggplot2)
ggplot(df, aes(x = time, y = outcome, group = unit_id)) +
  geom_line(data = df %>% filter(unit_id != treated_unit),
            color = "grey70", alpha = 0.5, linewidth = 0.7) +
  geom_line(data = df %>% filter(unit_id == treated_unit),
            color = "steelblue", linewidth = 1.8) +
  geom_vline(xintercept = treatment_year - 0.5,
             linetype = "dashed", color = "tomato", linewidth = 1) +
  labs(title = paste("Raw Trends:", treated_unit, "vs Donor Pool"),
       x = "Year", y = "Outcome") +
  theme_minimal()
```

---

## Step 2：SCM 估计

### R：tidysynth（推荐，pipe-friendly）

```r
library(tidysynth)
library(dplyr)

sc_out <- df %>%
  synthetic_control(
    outcome           = outcome,
    unit              = unit_id,
    time              = time,
    i_unit            = treated_unit,
    i_time            = treatment_year,
    generate_placebos = TRUE
  ) %>%
  generate_predictor(
    time_window = pre_years,
    gdp_pc      = mean(gdp_pc, na.rm = TRUE),
    population  = mean(population, na.rm = TRUE),
    trade_share = mean(trade_share, na.rm = TRUE)
  ) %>%
  generate_predictor(
    time_window = seq(min(pre_years), treatment_year - 1, by = 3),
    outcome_lag = mean(outcome, na.rm = TRUE)
  ) %>%
  generate_weights(
    optimization_window = pre_years,
    margin_ipop = 0.02, sigf_ipop = 7, bound_ipop = 6
  ) %>%
  generate_control()

sc_out %>% grab_unit_weights()
sc_out %>% grab_predictor_weights()
sc_out %>% plot_trends()
sc_out %>% plot_differences()
```

### Python：scpi_pkg

```python
from scpi_pkg.scest import scest
from scpi_pkg.scdata import scdata
import pandas as pd

def run_scm_python(df, outcome_col, unit_col, time_col,
                   treated_unit, treatment_year, predictors=None):
    df_sorted = df.sort_values([unit_col, time_col]).reset_index(drop=True)
    sc_data = scdata(
        df          = df_sorted,
        id_var      = unit_col,
        time_var    = time_col,
        outcome_var = outcome_col,
        period_pre  = list(range(df_sorted[time_col].min(), treatment_year)),
        period_post = list(range(treatment_year, df_sorted[time_col].max() + 1)),
        unit_tr     = treated_unit,
        unit_co     = [u for u in df_sorted[unit_col].unique() if u != treated_unit],
        features    = predictors if predictors else [outcome_col],
        cov_adj     = predictors
    )
    sc_result = scest(data=sc_data, w_constr={'name': 'simplex'})
    print(sc_result.summary())
    return {"data": sc_data, "result": sc_result}
```

---

## Step 3：拟合评估 + Balance Table

### RMSPE 检验

```r
sc_synth    <- sc_out %>% grab_synthetic_control()
pre_data    <- sc_synth %>% filter(time_unit < treatment_year)
pre_rmspe   <- sqrt(mean((pre_data$real_y - pre_data$synth_y)^2))
rel_rmspe   <- pre_rmspe / mean(pre_data$real_y) * 100

cat(sprintf("Pre-treatment RMSPE = %.4f\n", pre_rmspe))
cat(sprintf("相对 RMSPE = %.2f%%\n", rel_rmspe))

if (rel_rmspe > 10)
  cat("⚠️  拟合偏差 > 10%，建议换 SDID 或 augsynth\n")

weights     <- sc_out %>% grab_unit_weights()
max_weight  <- max(weights$weight)
if (max_weight > 0.80)
  cat(sprintf("⚠️  最大供体权重 = %.2f，过度依赖单一供体\n", max_weight))
```

### Balance Table（论文表格必备）

```r
# 输出标准 balance table：真实处理组 vs 合成处理组 vs 差值
library(dplyr)

predictors_list <- c("gdp_pc", "population", "trade_share")  # 替换为实际变量

# 处理单位处理前均值
treated_means <- df %>%
  filter(unit_id == treated_unit, time %in% pre_years) %>%
  summarise(across(all_of(predictors_list), mean, na.rm = TRUE)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "treated")

# 合成控制的处理前均值（供体加权）
unit_wts <- sc_out %>% grab_unit_weights() %>%
  rename(unit_id = unit_name)

donor_means <- df %>%
  filter(unit_id %in% donor_pool, time %in% pre_years) %>%
  group_by(unit_id) %>%
  summarise(across(all_of(predictors_list), mean, na.rm = TRUE)) %>%
  left_join(unit_wts, by = "unit_id") %>%
  summarise(across(all_of(predictors_list),
                   ~ sum(. * weight, na.rm = TRUE))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "synthetic")

balance_tbl <- treated_means %>%
  left_join(donor_means, by = "variable") %>%
  mutate(
    diff     = treated - synthetic,
    std_diff = diff / treated * 100
  )

print(balance_tbl)
write.csv(balance_tbl, "output/sc_balance_table.csv", row.names = FALSE)
```

---

## Step 4：推断

### Step 4a：置换推断（Fisher Exact Test）

```r
sc_out %>% plot_placebos(
  time_window = c(min(df$time), max(df$time)),
  prune       = TRUE
)

mspe_ratio  <- sc_out %>% grab_significance()
treated_pval <- mspe_ratio %>%
  filter(unit_name == treated_unit) %>%
  pull(fishers_exact_pvalue)

cat(sprintf("置换检验 p 值 = %.4f\n", treated_pval))
cat(sprintf("（供体池 %d 个单位，最小可能 p = %.4f）\n",
            length(donor_pool), 1 / (length(donor_pool) + 1)))
```

### Step 4b：scpi 预测区间（现代标准）

```r
library(scpi)

sc_data_r <- scdata(
  df          = df,
  id.var      = "unit_id",
  time.var    = "time",
  outcome.var = "outcome",
  period.pre  = pre_years,
  period.post = post_years,
  unit.tr     = treated_unit,
  unit.co     = donor_pool,
  features    = c("outcome"),
  cov.adj     = c("gdp_pc", "population")
)

sc_est <- scest(data = sc_data_r, w.constr = list(name = "simplex"))
sc_pi  <- scpi(data = sc_data_r, w.constr = list(name = "simplex"),
               sims = 200, cores = 4, CI.level = 0.95)
scplot(sc_pi)
```

---

## Step 5：稳健性检验

### Step 5a：Leave-One-Out

```r
top_donors <- sc_out %>% grab_unit_weights() %>%
  filter(weight > 0.05) %>% pull(unit_name)

loo_results <- lapply(top_donors, function(excl) {
  tryCatch({
    loo_sc <- df %>% filter(unit_id != excl) %>%
      synthetic_control(outcome = outcome, unit = unit_id, time = time,
                        i_unit = treated_unit, i_time = treatment_year,
                        generate_placebos = FALSE) %>%
      generate_predictor(time_window = pre_years,
                         gdp_pc = mean(gdp_pc), population = mean(population)) %>%
      generate_predictor(time_window = seq(min(pre_years), treatment_year - 1, 3),
                         outcome_lag = mean(outcome)) %>%
      generate_weights(optimization_window = pre_years) %>%
      generate_control()
    loo_sc %>% grab_synthetic_control() %>%
      mutate(excluded = excl, gap = real_y - synth_y)
  }, error = function(e) { cat(sprintf("⚠️ LOO %s 失败\n", excl)); NULL })
})

loo_df   <- do.call(rbind, Filter(Negate(is.null), loo_results))
main_gap <- sc_out %>% grab_synthetic_control() %>%
  mutate(excluded = "Main", gap = real_y - synth_y)

ggplot(mapping = aes(x = time_unit, y = gap)) +
  geom_line(data = loo_df, aes(group = excluded), color = "grey60", alpha = 0.7) +
  geom_line(data = main_gap, linewidth = 1.5, color = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_vline(xintercept = treatment_year - 0.5, linetype = "dashed", color = "tomato") +
  labs(title = "Leave-One-Out Robustness", x = "Year", y = "Gap") +
  theme_minimal()
ggsave("output/sc_loo_robustness.png", dpi = 150)
```

### Step 5b：In-Time Placebo

```r
fake_treatment_year <- treatment_year - 5
df_pre_only <- df %>% filter(time < treatment_year)

sc_placebo_time <- df_pre_only %>%
  synthetic_control(outcome = outcome, unit = unit_id, time = time,
                    i_unit = treated_unit, i_time = fake_treatment_year,
                    generate_placebos = FALSE) %>%
  generate_predictor(
    time_window = min(df_pre_only$time):(fake_treatment_year - 1),
    gdp_pc = mean(gdp_pc), population = mean(population)) %>%
  generate_predictor(
    time_window = seq(min(df_pre_only$time), fake_treatment_year - 1, 2),
    outcome_lag = mean(outcome)) %>%
  generate_weights(
    optimization_window = min(df_pre_only$time):(fake_treatment_year - 1)) %>%
  generate_control()

sc_placebo_time %>% plot_differences()
```

### Step 5c：Treated-Unit Reassignment Placebo

```r
# 将"处理单位"换成权重最高的供体，验证效应是否消失
top_donor <- sc_out %>% grab_unit_weights() %>%
  arrange(desc(weight)) %>% slice(1) %>% pull(unit_name)

# 用去掉真实处理单位的供体池，对 top_donor 跑 SCM
donor_pool_excl_treated <- df %>%
  filter(unit_id != treated_unit) %>% pull(unit_id) %>% unique()

sc_reassign <- df %>%
  filter(unit_id != treated_unit) %>%
  synthetic_control(outcome = outcome, unit = unit_id, time = time,
                    i_unit = top_donor, i_time = treatment_year,
                    generate_placebos = FALSE) %>%
  generate_predictor(time_window = pre_years,
                     gdp_pc = mean(gdp_pc), population = mean(population)) %>%
  generate_predictor(time_window = seq(min(pre_years), treatment_year - 1, 3),
                     outcome_lag = mean(outcome)) %>%
  generate_weights(optimization_window = pre_years) %>%
  generate_control()

sc_reassign %>% plot_differences()
cat("期望：对权重最高供体做 SCM，处理后 gap 不显著\n")
```

---

## Step 6：扩展方法

### Step 6a：augsynth（偏误修正 SCM）

当供体池较小（< 20）或预处理拟合一般时，augsynth 通过结果模型修正偏误并提供标准误。

```r
library(augsynth)

df_aug <- df %>%
  mutate(treated_post = as.integer(unit_id == treated_unit & time >= treatment_year))

aug_out <- augsynth(
  form     = outcome ~ treated_post,
  unit     = unit_id, time = time, data = df_aug,
  progfunc = "Ridge",   # 或 "GSYN", "EN"
  scm      = TRUE
)

summary(aug_out)
plot(aug_out, inf = TRUE)  # 含置信区间
att_by_period <- summary(aug_out)$average_att
write.csv(att_by_period, "output/sc_att_by_period.csv", row.names = FALSE)
```

### Step 6b：SDID（synthdid）

SDID 连接 SCM 与 DID，放松凸包假设，在许多场景下比经典 SCM 更稳健。**推荐作为标准稳健性方法，而非可选扩展。**

**SDID vs SCM vs DID 核心区别：**

| | SCM | SDID | DID |
|---|---|---|---|
| 处理单位数 | 1 | 1–少数 | 多 |
| 凸包要求 | 严格 | 放松 | 无 |
| 时间权重 | 无 | 有（对齐预处理趋势） | 无 |
| 推断 | 置换 | 置换/bootstrap | 聚类标准误 |
| 依赖假设 | 预处理拟合 | 加权平行趋势 | 平行趋势 |

```r
# install.packages("synthdid")
library(synthdid)
library(dplyr)

# synthdid 需要宽格式矩阵：行=单位，列=时间
# 准备处理矩阵
panel_mat <- df %>%
  select(unit_id, time, outcome) %>%
  pivot_wider(names_from = time, values_from = outcome) %>%
  column_to_rownames("unit_id") %>%
  as.matrix()

# 处理向量：哪些单位在哪些时期被处理
N_co   <- length(donor_pool)
N_tr   <- 1
T_pre  <- length(pre_years)
T_post <- length(post_years)

# panel.matrices() 自动拆分
setup  <- panel.matrices(panel_mat,
                         unit    = which(rownames(panel_mat) == treated_unit),
                         time    = which(colnames(panel_mat) == as.character(treatment_year)),
                         treated = 1)

# 三种估计量（对比报告）
tau_sdid <- synthdid_estimate(setup$Y, setup$N0, setup$T0)
tau_sc   <- sc_estimate(setup$Y, setup$N0, setup$T0)
tau_did  <- did_estimate(setup$Y, setup$N0, setup$T0)

cat(sprintf("SDID ATT = %.4f\n", tau_sdid))
cat(sprintf("SCM  ATT = %.4f\n", tau_sc))
cat(sprintf("DID  ATT = %.4f\n", tau_did))

# 标准误（置换推断）
se_sdid <- sqrt(vcov(tau_sdid, method = "placebo"))
se_sc   <- sqrt(vcov(tau_sc,   method = "placebo"))
cat(sprintf("SDID SE = %.4f, 95%% CI: [%.4f, %.4f]\n",
            se_sdid, tau_sdid - 1.96*se_sdid, tau_sdid + 1.96*se_sdid))

# 可视化
synthdid_plot(tau_sdid)

# 单位权重
synthdid_controls(tau_sdid)
```

---

### Step 6c：多 Outcome / 异质性 / 机制 批量模板

真实论文中需要对多个结果变量、子样本、机制变量批量运行 SCM/SDID。

```r
# ========================================
# 批量运行 SCM：多个结果变量
# ========================================
library(tidysynth)
library(dplyr)
library(purrr)

# 定义所有需要跑的结果变量
outcomes_list <- c(
  "peak_congestion",    # 主结果
  "all_day_congestion", # 稳健性
  "pm25",              # 机制：空气质量
  "pm10",
  "aqi",
  "speed_peak"         # 机制：车速
)

# 批量函数
run_scm_outcome <- function(outcome_var, df, treated_unit, treatment_year,
                             pre_years, predictor_vars) {
  tryCatch({
    sc <- df %>%
      rename(outcome_temp = !!sym(outcome_var)) %>%
      synthetic_control(
        outcome = outcome_temp, unit = unit_id, time = time,
        i_unit  = treated_unit, i_time = treatment_year,
        generate_placebos = TRUE
      ) %>%
      generate_predictor(
        time_window = pre_years,
        across(all_of(predictor_vars), ~ mean(.x, na.rm = TRUE))
      ) %>%
      generate_predictor(
        time_window = seq(min(pre_years), treatment_year - 1, by = 3),
        outcome_lag = mean(outcome_temp, na.rm = TRUE)
      ) %>%
      generate_weights(optimization_window = pre_years) %>%
      generate_control()

    # 提取处理后平均 ATT
    synth_data <- sc %>% grab_synthetic_control()
    post_data  <- synth_data %>% filter(time_unit >= treatment_year)
    att_avg    <- mean(post_data$real_y - post_data$synth_y, na.rm = TRUE)

    # RMSPE
    pre_data  <- synth_data %>% filter(time_unit < treatment_year)
    rmspe     <- sqrt(mean((pre_data$real_y - pre_data$synth_y)^2))
    rel_rmspe <- rmspe / mean(pre_data$real_y) * 100

    # p 值
    pval <- sc %>% grab_significance() %>%
      filter(unit_name == treated_unit) %>%
      pull(fishers_exact_pvalue)

    # 保存图
    sc %>% plot_differences() +
      labs(title = paste("SCM Gap:", outcome_var))
    ggsave(sprintf("output/sc_gap_%s.png", outcome_var), dpi = 150)

    list(outcome = outcome_var, att = att_avg,
         rmspe = rmspe, rel_rmspe = rel_rmspe, pval = pval,
         status = "success")
  }, error = function(e) {
    list(outcome = outcome_var, att = NA, rmspe = NA,
         rel_rmspe = NA, pval = NA, status = e$message)
  })
}

# 执行批量运行
predictor_vars <- c("gdp_pc", "population")  # 替换为实际变量

results_list <- map(outcomes_list, ~ run_scm_outcome(
  .x, df, treated_unit, treatment_year, pre_years, predictor_vars
))

results_df <- bind_rows(results_list)
print(results_df)
write.csv(results_df, "output/sc_batch_results.csv", row.names = FALSE)
```

```r
# ========================================
# 批量运行 SDID：多结果变量（更简洁）
# ========================================
library(synthdid)
library(dplyr)
library(tidyr)

run_sdid_outcome <- function(outcome_var, df, setup_fun, ...) {
  tryCatch({
    df_wide <- df %>%
      select(unit_id, time, !!sym(outcome_var)) %>%
      pivot_wider(names_from = time, values_from = !!sym(outcome_var)) %>%
      column_to_rownames("unit_id") %>%
      as.matrix()
    setup <- setup_fun(df_wide, ...)
    tau   <- synthdid_estimate(setup$Y, setup$N0, setup$T0)
    se    <- sqrt(vcov(tau, method = "placebo"))
    data.frame(outcome = outcome_var, att = as.numeric(tau),
               se = se, ci_lo = tau - 1.96*se, ci_hi = tau + 1.96*se)
  }, error = function(e) {
    data.frame(outcome = outcome_var, att = NA, se = NA,
               ci_lo = NA, ci_hi = NA)
  })
}
```

---

### Step 6d：gsynth / fect（多处理单位，非经典 SCM）

⚠️ **注意**：gsynth/fect 是**交互固定效应/矩阵完成类方法**，与经典 SCM 有亲缘关系，但不是同一个估计器。适用于多处理单位场景。

```r
library(gsynth)
gs_out <- gsynth(
  outcome ~ treated + control1 + control2,
  data     = df, index = c("unit_id", "time"),
  force    = "two-way", CV = TRUE, r = c(0, 5),
  se       = TRUE, nboots = 500, parallel = TRUE, cores = 4
)
plot(gs_out, type = "gap")
plot(gs_out, type = "counterfactual")

# fect：gsynth 现代替代（Liu-Wang-Xu 2022）
library(fect)
fe_out <- fect(
  outcome ~ treated + control1 + control2,
  data   = df, index = c("unit_id", "time"),
  method = "ife", CV = TRUE, r = c(0, 5),
  se     = TRUE, nboots = 500, parallel = TRUE, cores = 4
)
plot(fe_out)
```

---

### Step 6e：Penalized SCM（放松 simplex 权重约束）

经典 SCM 使用 simplex 约束（非负 + 和为 1）。当供体池凸包问题明显时，可尝试惩罚型权重：

```r
# scpi 包支持多种权重约束
library(scpi)

# Ridge 惩罚（放松凸包约束）
sc_ridge <- scest(
  data     = sc_data_r,
  w.constr = list(name = "lasso")   # 或 "ridge", "ols", "simplex"
)
print(sc_ridge)

# 对比不同权重约束下的 ATT
constraints <- c("simplex", "lasso", "ridge", "ols")
att_by_constr <- lapply(constraints, function(c) {
  est <- scest(data = sc_data_r, w.constr = list(name = c))
  data.frame(constraint = c,
             att_post = mean(est$est.results$Y.post -
                             est$est.results$Y.hat.post, na.rm = TRUE))
})
print(do.call(rbind, att_by_constr))
# 若各约束下 ATT 接近，结果稳健
```

---

## 检验清单（分层）

### 🔴 红线（任何分析都不可省）

| 检验 | 方法 | 通过标准 |
|------|------|----------|
| 开始前 5 问 | 自检 | 全部通过，否则调整方案 |
| 预处理拟合度 | RMSPE | < 10% × 结果均值 |
| 供体权重合理性 | `grab_unit_weights()` | 无极端单一供体（< 80%） |
| 推断方法 | 置换检验 **或** scpi 预测区间 | p < 0.10 或 CI 不含 0 |
| Estimand 声明 | 文本 | 明确写出 ATT + 凸包假设说明 |

### 🟡 条件触发（满足前提才激活）

| 条件 | 激活的检验 |
|------|-----------|
| 供体 < 20 | 改用 scpi 预测区间（置换 p 值分辨率不足） |
| 凸包拟合差（RMSPE > 10%）| 加跑 SDID 或 augsynth |
| 有多个结果变量 | 跑批量 outcome 模板（Step 6c） |
| 有重要机制假说 | 对机制变量批量跑 SCM |
| 供体池可能被污染 | 移除受污染单位后重跑（敏感性分析） |

### ⚪ 推荐（正式投稿前补）

| 检验 | 方法 |
|------|------|
| LOO 稳健性 | Step 5a |
| In-time placebo | Step 5b |
| Reassignment placebo | Step 5c |
| Balance table | Step 3 balance table 代码 |
| SDID 对比 | Step 6b，与 SCM 结果并排报告 |

---

## 常见错误

> **错误 1：供体池包含受到相似政策干扰的单位**
> 供体必须是从未受处理的单位。受干扰单位会污染合成控制，必须移除。

> **错误 2：处理前期太短（< 5 期）**
> 权重通过最小化处理前残差得到，预处理期太短极易过拟合。规则：pre_T >> n_predictors。

> **错误 3：只报告视觉结果，不报告推断**
> 必须报告置换检验 p 值或 scpi 预测区间。

> **错误 4：使用 grab_significance(time_window=...) 参数**
> tidysynth 的 `grab_significance()` 不接受 `time_window` 参数，直接调用即可。

> **错误 5：RMSPE ratio 安慰剂不剪枝**
> 用 `prune = TRUE` 移除 RMSPE > 处理单位 × 2 的供体，否则 p 值被扭曲。

> **错误 6：混淆 ATT 和 ATE**
> SCM 识别的是处理单位的 ATT，不能外推。gsynth/fect 多处理单位估计量是 ATT 平均。

> **错误 7：把中介变量放进 predictor set**
> 中介变量是 bad control，会产生过控制偏误（collider bias）。

---

## Estimand 声明

**SCM → ATT（处理单位的处理效应）**

| 情形 | 估计量 | 必须声明内容 |
|------|--------|------------|
| 单处理单位（SCM/SDID） | ATT | 权重分布 + 凸包假设是否满足 |
| 多处理单位（gsynth/fect） | ATT 平均 | 处理单位数 + 方法（IFE/MC） |

**标准声明模板：**
```
本文合成控制法估计量识别的是 [处理单位名] 的处理效应（ATT），
即与假设该单位从未受政策干预的反事实情形相比的因果效应。
合成控制由以下供体单位按权重构成：[主要供体 + 权重]。
预处理期 RMSPE = [值]（相对偏差 [值]%），[良好/一般]。
SCM 不要求 DID 式平行趋势假设，但要求处理单位在供体池凸包内，
且干预后无其他只影响处理组的同步冲击。
本文通过置换检验（p = [值]）和 LOO 稳健性检验对上述假设进行了验证。
```

---

## Few-Shot 示例：杭州城市大脑（交通拥堵）

```r
# 场景：杭州 2016 年实施"城市大脑"AI 交通系统，研究对拥堵的影响
# 数据：30 个城市 × 2012-2020 年，结果变量：peak_congestion_index

treated_unit   <- "Hangzhou"
treatment_year <- 2016
pre_years      <- 2012:2015
post_years     <- 2016:2020
donor_pool     <- c("Beijing", "Shanghai", "Guangzhou", "Chengdu",
                    "Wuhan", "Nanjing", "Zhengzhou", ...)  # 未实施城市大脑的城市

# Step 0: 5 问自检
# Q1: 杭州是否在供体池凸包内？→ 跑 SCM 看 RMSPE
# Q2: pre_T = 4 期，predictor 数控制在 ≤ 3 个
# Q3: 2016 年杭州无其他重大冲击？→ 查文献确认
# Q4: 供体池城市在 2016-2020 是否有类似 AI 交通系统？→ 逐一检查
# Q5: 供体池 J = 29 ≥ 20，置换推断可达 5% 显著水平

# 主要结果变量
sc_congestion <- df %>%
  synthetic_control(
    outcome = peak_congestion_index, unit = city_id, time = year,
    i_unit = "Hangzhou", i_time = 2016, generate_placebos = TRUE
  ) %>%
  generate_predictor(time_window = 2012:2015,
                     road_density = mean(road_density),
                     population   = mean(population)) %>%
  generate_predictor(time_window = c(2012, 2014),
                     congestion_lag = mean(peak_congestion_index)) %>%
  generate_weights(optimization_window = 2012:2015) %>%
  generate_control()

# 批量跑稳健性结果变量
outcomes_to_run <- c("peak_congestion_index", "all_day_congestion",
                     "pm25", "speed_peak")
batch_results <- map(outcomes_to_run, ~ run_scm_outcome(
  .x, df, "Hangzhou", 2016, 2012:2015, c("road_density", "population")
))
```

---

## 输出规范

| 图 | 内容 |
|----|------|
| 图1 | 处理单位 vs 合成控制（主趋势图）|
| 图2 | Gap 图（ATT 随时间）|
| 图3 | 安慰剂图（置换推断）|
| 图4 | LOO 稳健性图 |
| 图5 | SDID 对比图（可选）|

```
output/
  sc_raw_trends.png
  sc_main_results.png
  sc_placebo_inference.png
  sc_loo_robustness.png
  sc_intime_placebo.png
  sc_reassignment_placebo.png
  sc_sdid_comparison.png
  sc_unit_weights.csv
  sc_predictor_weights.csv
  sc_balance_table.csv
  sc_att_by_period.csv
  sc_batch_results.csv
  sc_prediction_intervals.csv
```
