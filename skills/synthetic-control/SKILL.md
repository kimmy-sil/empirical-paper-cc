# 合成控制法 (SCM)

## 概述

合成控制法（Abadie, Diamond & Hainmueller 2010, 2015）通过将多个对照单位的加权组合构造"合成控制组"，与处理单位对比，识别政策或事件的因果效应。特别适合**只有一个或少数几个处理单位**（如一个省份、国家、企业）的情形。

**适用场景：**
- 只有 1 个或极少数（< 5 个）处理单位
- 有多个未被处理的对照单位（供体池，donor pool），推荐 10–40 个
- 面板结构，处理前有足够时间期（至少 5–10 期）
- 处理后无法轻易找到自然对照（DID 对照组不合适）
- 示例：某省政策试点、某国经济危机、某城市禁烟令

**与 DID 的关键区别：** SCM 在预处理期通过优化权重"构造"对照组，不依赖平行趋势假设，而依赖预处理期拟合优度。

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

### 前置检查

```r
# R: 检查数据结构
library(dplyr)
library(tidyr)

treated_unit   <- "GuangDong"
treatment_year <- 2010
pre_years      <- min(df$time):(treatment_year - 1)
post_years     <- treatment_year:max(df$time)
donor_pool     <- unique(df$unit_id[df$unit_id != treated_unit])

# 确认平衡面板
balance_check <- df %>%
  count(unit_id, time) %>%
  filter(n > 1)
if (nrow(balance_check) > 0) stop("存在重复观测，数据非平衡面板")

cat(sprintf("处理单位: %s\n", treated_unit))
cat(sprintf("供体池: %d 个单位\n", length(donor_pool)))
cat(sprintf("处理前期数: %d\n", length(pre_years)))
cat(sprintf("处理后期数: %d\n", length(post_years)))

# 推荐：处理前期数 > 5，供体池 >= 10 个单位
if (length(pre_years) < 5)
  cat("⚠️  处理前期数 < 5，合成控制权重不可靠\n")
if (length(donor_pool) < 10)
  cat("⚠️  供体池 < 10，置换检验 p 值分辨率低（最小 p ≈ 1/J）\n")
```

---

## Step 1：描述性分析

```python
# Python: 处理单位 vs 供体池趋势图
import matplotlib.pyplot as plt
import pandas as pd

def plot_raw_trends(df, unit_col, time_col, outcome_col, 
                    treated_unit, treatment_year):
    """绘制处理单位 vs 供体池的原始趋势图。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 对照单位（灰色细线）
    for unit in df[unit_col].unique():
        if unit == treated_unit:
            continue
        sub = df[df[unit_col] == unit].sort_values(time_col)
        ax.plot(sub[time_col], sub[outcome_col],
                color='grey', alpha=0.3, linewidth=0.8)
    
    # 处理单位（蓝色粗线）
    treated = df[df[unit_col] == treated_unit].sort_values(time_col)
    ax.plot(treated[time_col], treated[outcome_col],
            color='steelblue', linewidth=2.5, label=str(treated_unit))
    
    ax.axvline(treatment_year - 0.5, color='tomato',
               linestyle='--', linewidth=1.5, label='Treatment')
    ax.set_xlabel('Year')
    ax.set_ylabel('Outcome')
    ax.set_title('Raw Trends: Treated vs Donor Pool')
    ax.legend()
    plt.tight_layout()
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
# R: tidysynth（推荐，与 dplyr 兼容）
library(tidysynth)
library(dplyr)

sc_out <- df %>%
  synthetic_control(
    outcome          = outcome,
    unit             = unit_id,
    time             = time,
    i_unit           = treated_unit,   # 处理单位名（字符型）
    i_time           = treatment_year, # 处理年份
    generate_placebos = TRUE           # 同时生成安慰剂（推断用）
  ) %>%
  # 添加预测变量（处理前均值）
  generate_predictor(
    time_window = pre_years,
    gdp_pc     = mean(gdp_pc, na.rm = TRUE),
    population = mean(population, na.rm = TRUE),
    trade_share = mean(trade_share, na.rm = TRUE)
  ) %>%
  # 结果变量的处理前多期拟合（推荐）
  generate_predictor(
    time_window = seq(min(pre_years), treatment_year - 1, by = 3),
    outcome_lag = mean(outcome, na.rm = TRUE)
  ) %>%
  # 生成权重（优化）
  generate_weights(
    optimization_window = pre_years,
    margin_ipop = 0.02, sigf_ipop = 7, bound_ipop = 6
  ) %>%
  # 生成合成控制
  generate_control()

# 查看结果
sc_out %>% grab_unit_weights()       # 供体权重（稀疏性检查）
sc_out %>% grab_predictor_weights()  # 预测变量权重

# 可视化
sc_out %>% plot_trends()             # 处理单位 vs 合成控制
sc_out %>% plot_differences()        # ATT 随时间变化（gap 图）
```

### Python：scpi_pkg.scest（替代 SparseSC）

```python
# Python: scpi_pkg（替换已废弃的 SparseSC！）
# pip install scpi-pkg
from scpi_pkg.scest import scest
from scpi_pkg.scdata import scdata
import pandas as pd
import numpy as np

def run_scm_python(df, outcome_col, unit_col, time_col,
                   treated_unit, treatment_year, predictors=None):
    """
    使用 scpi_pkg 运行合成控制。
    
    Parameters
    ----------
    df             : pandas DataFrame（长格式平衡面板）
    outcome_col    : str，结果变量列名
    unit_col       : str，单位 ID 列名
    time_col       : str，时间列名
    treated_unit   : str/int，处理单位标识
    treatment_year : int，处理年份
    predictors     : list or None，预测变量列名
    
    Returns
    -------
    dict 含 scdata_obj, scest_obj
    """
    # 准备 scdata 对象
    df_sorted = df.sort_values([unit_col, time_col]).reset_index(drop=True)
    
    # scdata 格式：设置处理单位和供体
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
        cov_adj     = predictors  # 协变量调整
    )
    
    # 估计合成控制
    sc_result = scest(
        data    = sc_data,
        w_constr = {'name': 'simplex'}  # 权重约束（非负 + 加和为 1）
    )
    
    print(sc_result.summary())
    return {"data": sc_data, "result": sc_result}

# 使用示例
scm_res = run_scm_python(
    df            = df,
    outcome_col   = 'outcome',
    unit_col      = 'unit_id',
    time_col      = 'year',
    treated_unit  = 'GuangDong',
    treatment_year = 2010,
    predictors    = ['gdp_pc', 'population', 'trade_share']
)
```

---

## Step 3：拟合评估

**RMSPE（Root Mean Squared Prediction Error）** 是衡量合成控制拟合质量的核心指标，越小越好。

```r
# R: 预处理 RMSPE（tidysynth）
# 注意修复：grab_loss() 返回 tibble，需要 filter() + pull()
pre_rmspe <- sc_out %>%
  grab_loss() %>%
  # grab_loss() 返回 tibble，含 unit_id 和 loss 列
  filter(unit_type == "treated") %>%   # 仅保留处理单位
  pull(loss) %>%                       # 提取 MSPE
  sqrt()                               # RMSPE = sqrt(MSPE)

# 若 grab_loss() 结构不同，手动计算：
sc_synth <- sc_out %>% grab_synthetic_control()
pre_data  <- sc_synth %>% filter(time_unit < treatment_year)
pre_rmspe_manual <- sqrt(mean((pre_data$real_y - pre_data$synth_y)^2))

cat(sprintf("Pre-treatment RMSPE = %.4f\n", pre_rmspe_manual))
cat(sprintf("相对 RMSPE（/结果均值）= %.2f%%\n",
            pre_rmspe_manual / mean(pre_data$real_y) * 100))

# 通过标准：RMSPE < 5-10% × 结果变量均值
if (pre_rmspe_manual / mean(pre_data$real_y) > 0.10) {
  cat("⚠️  预处理拟合偏差 > 10%，合成控制质量较低\n")
  cat("   建议：增加预测变量、调整 optimization_window、或检查供体池\n")
}

# 权重分布检查
weights <- sc_out %>% grab_unit_weights()
cat("\n=== 供体权重分布 ===\n")
print(weights %>% filter(weight > 0.01) %>% arrange(desc(weight)))

max_weight <- max(weights$weight)
if (max_weight > 0.80) {
  cat(sprintf("⚠️  最大供体权重 = %.2f，过度依赖单一供体\n", max_weight))
}
```

```python
# Python: scpi_pkg 拟合评估
import numpy as np

def evaluate_scm_fit(sc_result, treatment_year):
    """评估合成控制预处理期拟合质量。"""
    # 提取合成控制路径
    synth_path = sc_result["result"].est_results
    
    # 处理前期
    pre_mask = synth_path['Time'] < treatment_year
    real_pre  = synth_path.loc[pre_mask, 'Y.pre']
    synth_pre = synth_path.loc[pre_mask, 'Y.hat.pre']
    
    rmspe = np.sqrt(np.mean((real_pre.values - synth_pre.values) ** 2))
    rel_rmspe = rmspe / real_pre.mean() * 100
    
    print(f"Pre-treatment RMSPE = {rmspe:.4f}")
    print(f"相对 RMSPE = {rel_rmspe:.2f}%")
    
    if rel_rmspe > 10:
        print("⚠️  拟合偏差 > 10%，建议增加预测变量")
    
    return {"rmspe": rmspe, "rel_rmspe": rel_rmspe}
```

---

## Step 4：推断

### Step 4a：置换推断（经典，Fisher Exact Test）

```r
# R: tidysynth 置换推断（generate_placebos = TRUE 时已自动生成）

# 安慰剂图（灰色：各供体 gap；黑色：处理单位 gap）
sc_out %>% plot_placebos(
  time_window = c(min(df$time), max(df$time)),
  prune       = TRUE   # 排除预处理 RMSPE > 处理单位 × 2 的供体
)

# Rank-based p 值
# 注意修复：grab_significance() 无 time_window 参数（直接调用）
mspe_ratio <- sc_out %>%
  grab_significance()   # 不传 time_window 参数

print(mspe_ratio)

# 提取处理单位的 p 值
treated_pval <- mspe_ratio %>%
  filter(unit_name == treated_unit) %>%
  pull(fishers_exact_pvalue)
cat(sprintf("置换检验 p 值 = %.4f\n", treated_pval))
cat(sprintf("（供体池 %d 个单位，最小可能 p = %.4f）\n",
            length(donor_pool), 1 / (length(donor_pool) + 1)))

# 显著性：p ≤ 0.05 需要 ≥ 20 个供体单位
if (length(donor_pool) < 20) {
  cat("⚠️  供体 < 20，置换 p 值分辨率不足达到 5% 水平\n")
}
```

### Step 4b：scpi 预测区间（现代标准）

**scpi 预测区间 vs 置信区间：**
- 传统置换推断给出 Fisher Exact p 值（离散，分辨率受供体数限制）
- scpi 的**预测区间（prediction intervals）** 考虑随机性来源（不确定性量化），推断更精准
- 推荐：两种都报告

```r
# R: scpi 包（Cattaneo-Feng-Palomba-Titiunik 2021+）
# install.packages("scpi")
library(scpi)

# Step 1: 准备 scdata 对象
sc_data_r <- scdata(
  df            = df,
  id.var        = "unit_id",
  time.var      = "time",
  outcome.var   = "outcome",
  period.pre    = pre_years,
  period.post   = post_years,
  unit.tr       = treated_unit,
  unit.co       = donor_pool,
  features      = c("outcome"),       # 可加协变量
  cov.adj       = c("gdp_pc", "population")  # 调整协变量
)

# Step 2: 点估计（scest）
sc_est <- scest(
  data     = sc_data_r,
  w.constr = list(name = "simplex")   # 权重：非负 + 加和为 1
)
print(sc_est)

# Step 3: 预测区间（scpi）
sc_pi <- scpi(
  data        = sc_data_r,
  w.constr    = list(name = "simplex"),
  sims        = 200,    # 模拟次数（500-1000 更精确）
  cores       = 4,      # 并行
  CI.level    = 0.95
)
print(sc_pi)

# Step 4: 可视化（含预测区间）
scplot(sc_pi, 
       main    = paste("Synthetic Control:", treated_unit),
       x.label = "Year",
       y.label = "Outcome")
```

```python
# Python: scpi_pkg（与 R scpi 包对应）
from scpi_pkg.scest  import scest
from scpi_pkg.scpi   import scpi
from scpi_pkg.scdata import scdata
from scpi_pkg.scplot import scplot

# Step 1: scdata 已在 Step 2 构造
# Step 2: 点估计（已完成）

# Step 3: 预测区间
sc_pi_py = scpi(
    data     = scm_res["data"],
    w_constr = {'name': 'simplex'},
    sims     = 200,
    cores    = 4,
    ci_level = 0.95
)
print(sc_pi_py.summary())

# Step 4: 可视化
scplot(sc_pi_py)
```

---

## Step 5：稳健性检验

### Step 5a：Leave-One-Out（移除主要供体）

```r
# R: Leave-One-Out
library(tidysynth)
library(dplyr)
library(ggplot2)

# 主要供体（权重 > 5%）
top_donors <- sc_out %>%
  grab_unit_weights() %>%
  filter(weight > 0.05) %>%
  pull(unit_name)
cat(sprintf("主要供体（权重>5%%）: %s\n", paste(top_donors, collapse = ", ")))

loo_results <- lapply(top_donors, function(excl) {
  tryCatch({
    loo_sc <- df %>%
      filter(unit_id != excl) %>%
      synthetic_control(
        outcome = outcome, unit = unit_id, time = time,
        i_unit  = treated_unit, i_time = treatment_year,
        generate_placebos = FALSE
      ) %>%
      generate_predictor(time_window = pre_years,
                         gdp_pc = mean(gdp_pc),
                         population = mean(population)) %>%
      generate_predictor(time_window = seq(min(pre_years), treatment_year - 1, 3),
                         outcome_lag = mean(outcome)) %>%
      generate_weights(optimization_window = pre_years) %>%
      generate_control()
    
    loo_sc %>% grab_synthetic_control() %>%
      mutate(excluded = excl,
             gap      = real_y - synth_y)
  }, error = function(e) {
    cat(sprintf("⚠️  LOO（排除 %s）失败：%s\n", excl, e$message)); NULL
  })
})

loo_df <- do.call(rbind, Filter(Negate(is.null), loo_results))

main_gap <- sc_out %>% grab_synthetic_control() %>%
  mutate(excluded = "Main", gap = real_y - synth_y)

ggplot(mapping = aes(x = time_unit, y = gap)) +
  geom_line(data = loo_df,  aes(group = excluded), color = "grey60", alpha = 0.7) +
  geom_line(data = main_gap, linewidth = 1.5, color = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray30") +
  geom_vline(xintercept = treatment_year - 0.5, linetype = "dashed", color = "tomato") +
  labs(title = "Leave-One-Out Robustness", x = "Year", y = "Gap (Treated − Synthetic)") +
  theme_minimal()
ggsave('output/sc_loo_robustness.png', dpi = 150)
```

### Step 5b：In-Space Placebo

```r
# R: In-space placebo（对每个供体单位做 SCM，对比 gap 分布）
# 由 generate_placebos = TRUE + plot_placebos() 自动完成（见 Step 4a）
```

### Step 5c：In-Time Placebo（假处理期）

```r
# R: In-time placebo（假设政策在处理前某期发生）
fake_treatment_year <- treatment_year - 5  # 政策前 5 年

df_pre_only <- df %>% filter(time < treatment_year)

sc_placebo_time <- df_pre_only %>%
  synthetic_control(
    outcome = outcome, unit = unit_id, time = time,
    i_unit  = treated_unit, i_time = fake_treatment_year,
    generate_placebos = FALSE
  ) %>%
  generate_predictor(
    time_window = min(df_pre_only$time):(fake_treatment_year - 1),
    gdp_pc = mean(gdp_pc), population = mean(population)
  ) %>%
  generate_predictor(
    time_window = seq(min(df_pre_only$time), fake_treatment_year - 1, 2),
    outcome_lag = mean(outcome)
  ) %>%
  generate_weights(
    optimization_window = min(df_pre_only$time):(fake_treatment_year - 1)
  ) %>%
  generate_control()

# 处理后期（假处理期后、真处理期前）的 gap 应不显著
sc_placebo_time %>% plot_differences()
cat("期望：假处理期后 gap 不显著（平均接近 0）\n")
```

---

## Step 6：扩展

### Step 6a：augsynth（偏误修正 SCM）

当供体池较小（< 20 个单位）时，传统 SCM 权重估计不稳定，**augsynth 通过结果模型（outcome model）修正偏误**，同时提供置信区间（标准误）。

```r
# R: augsynth 包（Ben-Michael, Feller, Rothstein 2021）
# install.packages("augsynth")
library(augsynth)
library(dplyr)

# augsynth 格式：Y ~ treat | time | covariates
# 标记处理变量（post × treated）
df_aug <- df %>%
  mutate(treated_post = as.integer(unit_id == treated_unit & time >= treatment_year))

# 主估计（augsynth 自带标准误）
aug_out <- augsynth(
  form      = outcome ~ treated_post,   # 公式：结果 ~ 处理哑变量
  unit      = unit_id,
  time      = time,
  data      = df_aug,
  # 协变量调整（降低偏误）
  progfunc  = "Ridge",   # 结果模型：Ridge 回归（可改 "GSYN", "EN"）
  scm       = TRUE       # 是否同时用 SCM 权重（TRUE=augmented）
)

summary(aug_out)
# 输出：ATT 估计、置信区间（基于正则化bootstrap）

# 可视化
plot(aug_out)          # 主趋势图
plot(aug_out, inf = TRUE)  # 含置信区间

# 提取逐期 ATT
att_by_period <- summary(aug_out)$average_att
print(att_by_period)

# 小供体池推荐：augsynth 而非经典 Synth
cat(sprintf("\n供体池大小: %d 个单位\n", length(donor_pool)))
if (length(donor_pool) < 20) {
  cat("✓  供体 < 20：推荐使用 augsynth（偏误修正 + 自带标准误）\n")
} else {
  cat("ℹ️  供体充足：经典 SCM（tidysynth）和 augsynth 均可，建议均报告\n")
}
```

### Step 6b：gsynth / fect（多处理单位）

当有少量多个处理单位时，使用交互固定效应（IFE）类方法。

```r
# R: gsynth（Xu 2017，矩阵完成方法）
library(gsynth)

gs_out <- gsynth(
  outcome ~ treated + control1 + control2,
  data       = df,
  index      = c("unit_id", "time"),
  force      = "two-way",  # 双向固定效应
  CV         = TRUE,       # 交叉验证选因子数
  r          = c(0, 5),    # 候选因子数范围
  se         = TRUE,       # 置信区间（bootstrap）
  nboots     = 500,
  parallel   = TRUE,
  cores      = 4
)

plot(gs_out, type = "gap")           # ATT（gap 图）
plot(gs_out, type = "counterfactual") # 反事实趋势图
print(gs_out)                         # ATT 摘要

# fect（Liu-Wang-Xu 2022，Factor-Augmented IFE，gsynth 现代替代）
library(fect)
fe_out <- fect(
  outcome ~ treated + control1 + control2,
  data   = df,
  index  = c("unit_id", "time"),
  method = "ife",     # 交互固定效应（或 "mc"=矩阵完成）
  CV     = TRUE,
  r      = c(0, 5),
  se     = TRUE,
  nboots = 500,
  parallel = TRUE,
  cores  = 4
)
plot(fe_out)

# gsynth vs fect 选择：
# - gsynth：经典，引用多
# - fect：现代替代，支持更多方法（IFE, MC, FEct, GSC）
```

---

## 检验清单（6 项）

| 检验 | 方法 | 通过标准 |
|------|------|----------|
| 预处理拟合度 | RMSPE（`grab_loss()` + sqrt） | RMSPE < 5-10% × 结果均值 |
| 供体权重合理性 | `grab_unit_weights()` | 无极端单一供体（< 80%） |
| 置换推断 | Fisher Exact（`grab_significance()`） | rank-based p < 0.10 |
| scpi 预测区间 | `scpi()` 包 | 处理后期 CI 不含 0 |
| Leave-One-Out | 逐个移除主要供体 | Gap 图形态稳定 |
| In-time placebo | 假处理期 SCM | 假处理后 gap 不显著 |

---

## 常见错误（6 条）

> **错误 1：供体池包含受到相似政策干扰的单位**
> 合成控制的供体必须是从未受处理的单位。相关政策干扰会污染合成控制，应将受干扰单位从供体池中移除。

> **错误 2：处理前期太短（< 5 期）**
> 权重通过最小化处理前残差得到。预处理期太短极易过拟合，权重不可靠。规则：处理前期数 >> 预测变量数。

> **错误 3：只报告视觉结果，不报告推断**
> 必须报告置换检验 p 值（或 scpi 预测区间），不能只说"图上看起来有效果"。

> **错误 4：使用 grab_significance(time_window=...) 参数**
> tidysynth 的 `grab_significance()` 不接受 `time_window` 参数，直接调用即可。传入该参数会报错。

> **错误 5：RMSPE ratio 安慰剂不剪枝**
> 在置换检验中，预处理拟合极差的供体（RMSPE 远大于处理单位）会扭曲 p 值。使用 `prune=TRUE`（`plot_placebos`）移除 RMSPE > 处理单位 × 2 的供体。

> **错误 6：混淆 ATT 和 ATE**
> SCM 识别的是处理单位的 ATT（单个单位），不能外推到其他单位或总体。多处理单位时（gsynth/fect），估计量是参与单位的 ATT 平均，与全体 ATE 不同。

---

## Estimand 声明

**SCM → ATT（处理单位的处理效应）**

| 情形 | 估计量 | 必须声明内容 |
|------|--------|------------|
| 单处理单位 | ATT 单位（单个单位的因果效应） | 合成控制权重分布 + 主要供体 |
| 多处理单位（gsynth/fect） | ATT（处理单位的平均处理效应） | 处理单位数量 + 权重来源 |

**标准声明模板：**
```
本文合成控制法估计量识别的是 [处理单位名] 的处理效应（ATT），
即与假设该单位从未受政策干预的反事实情形相比的因果效应。
合成控制由以下供体单位按权重构成：[主要供体 + 权重]。
预处理期（[起始年]–[处理前一年]）RMSPE = [值]，
相对偏差 = [值]%，表明合成控制对处理单位预处理期的拟合
[良好/一般]（< 5% / 5-10% / > 10%）。
该估计量不能外推到其他单位。
```

---

## 输出规范

**图表规范：**

| 图 | 内容 |
|----|------|
| 图1：主趋势图 | 处理单位（实线）vs 合成控制（虚线）+ 垂直处理线 |
| 图2：Gap 图 | (处理单位 - 合成控制) 随时间变化，含处理线 |
| 图3：安慰剂图 | 灰色=各供体 gap，黑色=处理单位 gap |
| 图4：LOO 图 | 各条线=移除一个供体后的 gap |

### 文件命名

```
output/
  sc_raw_trends.png            # 原始趋势图
  sc_main_results.png          # 主图（trends + gap）
  sc_placebo_inference.png     # 置换推断图
  sc_loo_robustness.png        # Leave-one-out 图
  sc_intime_placebo.png        # 时间安慰剂
  sc_unit_weights.csv          # 供体权重
  sc_predictor_weights.csv     # 预测变量权重
  sc_balance_table.csv         # 预处理期特征对比
  sc_att_by_period.csv         # 逐期 ATT 估计
  sc_prediction_intervals.csv  # scpi 预测区间
```
