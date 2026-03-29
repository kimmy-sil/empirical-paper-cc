# 合成控制法 (Synthetic Control Method, SCM)

## 概述

合成控制法（Abadie, Diamond & Hainmueller 2010, 2015）通过将多个对照单位的加权组合构造出"合成控制组"，与处理单位进行比较，识别政策/事件的因果效应。特别适合**只有一个或少数几个处理单位**（如一个省份、一个国家、一家企业）的情形，是 DID 在少处理单位场景下的替代方案。

**适用场景：**
- 只有 1 个或极少数（< 5 个）处理单位
- 有多个未被处理的对照单位（供体池，donor pool）
- 数据为面板结构，处理前有足够多的时间期（至少 5-10 期）
- 处理后无法轻易找到自然对照（DID 对照组不合适）
- 示例：某省的政策试点、某国的经济危机、某城市禁烟令

---

## 前置条件

### 数据结构要求

```
必须包含：
  - 个体 ID 列（unit_id）：各单位名称/编码
  - 时间列（time）：年份/期数，数值型
  - 结果变量（outcome）：Y
  - 预测变量（predictors）：处理前特征，用于构造合成控制权重
  - 处理单位标识（treated_unit）：如 "GuangDong" 或 unit_id = 1
  - 处理时间（treatment_year）：政策实施的第一期

数据必须为平衡面板（每个单位每期均有观测）
处理前期数 >> 协变量数（避免过拟合合成控制）
```

### 前置准备

```r
# R: 检查数据结构
library(tidyr)

# 确认平衡面板
panel_check <- df %>%
  count(unit_id, time) %>%
  filter(n > 1)  # 若有记录则存在重复
stopifnot(nrow(panel_check) == 0)

# 处理单位 vs 供体池
treated_unit <- "GuangDong"
donor_pool   <- unique(df$unit_id[df$unit_id != treated_unit])
cat("Treated unit:", treated_unit, "\n")
cat("Donor pool size:", length(donor_pool), "\n")
cat("Pre-treatment periods:", sum(df$time < treatment_year) / n_distinct(df$unit_id), "\n")
```

---

## 分析步骤

### Step 1：描述性分析 — 原始趋势图

```python
# Python: 处理单位 vs 所有对照单位趋势图
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

fig, ax = plt.subplots(figsize=(10, 5))

# 对照单位（灰色细线）
for unit in donor_pool:
    sub = df[df['unit_id'] == unit].sort_values('time')
    ax.plot(sub['time'], sub['outcome'], color='grey', alpha=0.3, linewidth=0.8)

# 处理单位（蓝色粗线）
treated_data = df[df['unit_id'] == treated_unit].sort_values('time')
ax.plot(treated_data['time'], treated_data['outcome'],
        color='blue', linewidth=2.5, label=treated_unit)

ax.axvline(treatment_year - 0.5, color='red', linestyle='--', label='Treatment')
ax.set_xlabel('Year')
ax.set_ylabel('Outcome')
ax.set_title('Raw Trends: Treated vs Donor Pool')
ax.legend()
plt.tight_layout()
plt.savefig('output/sc_raw_trends.png', dpi=150)
```

```r
# R: ggplot2 趋势图
library(ggplot2)

ggplot(df, aes(x = time, y = outcome, group = unit_id)) +
  geom_line(data = df %>% filter(unit_id != treated_unit), color = "grey70", alpha = 0.5) +
  geom_line(data = df %>% filter(unit_id == treated_unit), color = "steelblue", linewidth = 1.5) +
  geom_vline(xintercept = treatment_year - 0.5, linetype = "dashed", color = "red") +
  labs(title = paste("Raw Trends:", treated_unit, "vs Donor Pool")) +
  theme_minimal()
```

---

### Step 2：经典 SCM 估计（Abadie et al.）

```r
# R: Synth 包（经典实现）
library(Synth)

# 数据转换为 Synth 需要的矩阵格式
dataprep_out <- dataprep(
  foo            = df,
  predictors     = c("gdp_pc", "population", "trade_share", "education"),  # 预测变量
  predictors.op  = "mean",   # 取处理前均值
  dependent      = "outcome",
  unit.variable  = "unit_id_numeric",   # 必须是数值型
  time.variable  = "time",
  treatment.identifier  = treated_unit_id,   # 数值型 ID
  controls.identifier   = donor_ids,         # 数值型 ID 向量
  time.predictors.prior = pre_treatment_years,   # 预处理期
  time.optimize.ssr     = pre_treatment_years,   # 优化期（用于结果变量拟合）
  unit.names.variable   = "unit_name",
  time.plot             = all_years
)

# 估计合成控制权重
synth_out <- synth(dataprep_out)

# 结果路径
synth_tables <- synth.tab(
  dataprep.res = dataprep_out,
  synth.res    = synth_out
)
print(synth_tables$tab.w)   # 供体权重
print(synth_tables$tab.v)   # 预测变量权重
print(synth_tables$tab.pred)  # 预处理期拟合对比
```

```r
# R: tidysynth（更现代的接口，与 dplyr 兼容）
library(tidysynth)

sc_out <- df %>%
  synthetic_control(
    outcome   = outcome,
    unit      = unit_id,
    time      = time,
    i_unit    = treated_unit,   # 处理单位名（字符型）
    i_time    = treatment_year, # 处理年份
    generate_placebos = TRUE    # 同时生成安慰剂（用于推断）
  ) %>%
  # 添加预测变量（处理前均值）
  generate_predictor(time_window = pre_years,
                     gdp_pc     = mean(gdp_pc),
                     population = mean(population),
                     trade_share = mean(trade_share)) %>%
  # 添加结果变量的预处理期拟合（关键！）
  generate_predictor(time_window = seq(first_year, treatment_year-1, by=2),
                     outcome_lag = mean(outcome)) %>%
  # 生成权重
  generate_weights(optimization_window = pre_years,
                   margin_ipop = .02, sigf_ipop = 7, bound_ipop = 6) %>%
  # 生成合成控制
  generate_control()

# 查看权重
sc_out %>% grab_unit_weights()       # 供体权重
sc_out %>% grab_predictor_weights()  # 预测变量权重

# 主图：处理单位 vs 合成控制
sc_out %>% plot_trends(time_window = all_years)
sc_out %>% plot_differences(time_window = all_years)  # 处理效应（gap 图）
```

```stata
* Stata: synth（经典）
ssc install synth

synth outcome gdp_pc population trade_share outcome(1995) outcome(2000) outcome(2003), ///
    trunit(1) trperiod(2005) unitnames(province) ///
    counit(2 3 4 5 6 7 8) ///
    xperiod(1990(1)2004) mspeperiod(1990(1)2004) ///
    figure keep(synth_results) replace

* Stata: synth_runner（批量安慰剂）
ssc install synth_runner
synth_runner outcome gdp_pc population, trunit(1) trperiod(2005) ///
    gen_vars  // 生成效应变量
```

---

### Step 3：预处理拟合度评估（Pre-treatment Fit）

**RMSPE（Root Mean Squared Prediction Error）**是衡量合成控制拟合质量的核心指标，越小越好。

```r
# R: 计算预处理 RMSPE（tidysynth）
mspe_pre <- sc_out %>%
  grab_loss()  # MSPE（MSE of synthetic vs treated in pre-period）

cat("Pre-treatment RMSPE:", sqrt(mspe_pre), "\n")

# 预处理期对比表（处理单位 vs 合成控制 vs 样本均值）
sc_out %>% grab_balance_table()
# 检查: 合成控制与处理单位的预处理期特征应非常接近
```

**拟合度判断标准：**
- Pre-treatment RMSPE 相对于处理单位结果变量均值应 < 5-10%
- 预处理期趋势图中合成控制曲线应与处理单位高度重叠

---

### Step 4：推断——置换检验（Permutation / Placebo Inference）

SCM 无法使用传统标准误，通过**依次将每个供体单位作为"处理单位"**重复估计，构造经验分布。

```r
# R: tidysynth 安慰剂推断
# （generate_placebos = TRUE 时已自动生成）

# 安慰剂图（Placebos plot）
sc_out %>% plot_placebos(
  time_window   = all_years,
  prune         = TRUE   # 排除预处理 RMSPE 过大的供体（>2x 处理单位）
)

# 计算 p-value
# p = 处理后 RMSPE / 处理前 RMSPE（ratio）中，处理单位排第几名
mspe_ratio <- sc_out %>%
  grab_significance(time_window = post_years)
print(mspe_ratio)
# 若处理单位的 RMSPE ratio 在所有单位中排第 1，p = 1/(J+1)，J = 供体数

# 显著性：p ≤ 0.05 需要 ≥ 20 个供体单位
cat("Rank-based p-value:", mspe_ratio %>% filter(unit == treated_unit) %>% pull(fishers_exact_pvalue), "\n")
```

```python
# Python: SparseSC（含推断）
import SparseSC as sc
import numpy as np

# 构造矩阵：N_donors × T
Y = df.pivot(index='unit_id', columns='time', values='outcome').values
# treated 行排最后
Y_pre  = Y[:, :pre_periods]
Y_post = Y[:, pre_periods:]

fit = sc.fit_fast(
    features        = Y_pre[:-1],   # 供体预处理期
    targets         = Y_pre[-1:],   # 处理单位预处理期
    treated_units   = [Y.shape[0]-1]
)
Y_counterfactual = fit.predict(Y_pre)
effect = Y_post[-1] - fit.predict(Y_post[-1:])  # 处理效应
```

---

### Step 5：Leave-One-Out 稳健性

逐个移除权重最大的供体，检验合成控制结论是否依赖某一特定供体。

```r
# R: Leave-one-out（手动循环）
top_donors <- sc_out %>%
  grab_unit_weights() %>%
  filter(weight > 0.05) %>%
  pull(unit)

loo_results <- list()
for (exclude_unit in top_donors) {
  loo_sc <- df %>%
    filter(unit_id != exclude_unit) %>%  # 移除该供体
    synthetic_control(
      outcome = outcome, unit = unit_id, time = time,
      i_unit = treated_unit, i_time = treatment_year,
      generate_placebos = FALSE
    ) %>%
    generate_predictor(...) %>%
    generate_weights() %>%
    generate_control()

  loo_results[[exclude_unit]] <- loo_sc %>%
    grab_synthetic_control() %>%
    mutate(excluded = exclude_unit)
}

# 绘制 LOO 图：各条线代表移除一个供体后的合成控制
loo_df <- bind_rows(loo_results)
ggplot(loo_df, aes(x = time, y = real_y - synth_y, group = excluded, color = excluded)) +
  geom_line(alpha = 0.5) +
  geom_line(data = sc_out %>% grab_synthetic_control() %>% mutate(excluded = "Main"),
            aes(y = real_y - synth_y), color = "black", linewidth = 1.5) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_vline(xintercept = treatment_year - 0.5, linetype = "dashed", color = "red") +
  labs(title = "Leave-One-Out Robustness") +
  theme_minimal()
```

---

### Step 6：广义合成控制（gsynth，含多处理单位）

当有少量多个处理单位时，使用 `gsynth`（Xu 2017，Matrix Completion 方法）。

```r
# R: gsynth（多处理单位 + 置信区间）
library(gsynth)

gs_out <- gsynth(
  outcome ~ treated + control1 + control2,
  data         = df,
  index        = c("unit_id", "time"),
  force        = "two-way",   # 双向固定效应
  CV           = TRUE,        # 交叉验证选择因子数
  r            = c(0, 5),     # 因子数候选范围
  se           = TRUE,        # 置信区间（bootstrap）
  nboots       = 1000,
  parallel     = TRUE,        # 并行加速
  cores        = 4
)

plot(gs_out, type = "gap")      # 处理效应（gap）图
plot(gs_out, type = "counterfactual")  # 反事实趋势图
print(gs_out)                   # ATT 摘要
```

---

## 必做检验清单

| 检验 | 方法 | 通过标准 |
|------|------|----------|
| 预处理拟合度 | RMSPE | < 5-10% 结果均值 |
| 供体权重合理性 | 查看 tab.w | 无极端单一供体（< 80%）|
| 安慰剂推断 | Permutation test | rank-based p < 0.1 |
| Leave-one-out | 逐个移除主要供体 | Gap 图形态稳定 |
| 随时间变化 placebo | 假处理期 | 处理前 gap 不显著 |
| 供体池剔除不良单位 | 预处理期结构异常 | 移除受到干扰的潜在对照 |

---

## 常见错误提醒

> **错误 1：供体池包含受到相似政策干扰的单位**
> 合成控制的供体池必须是"从未受处理"的单位。如果某些供体在研究期间也受到相关政策影响，会污染合成控制，应从供体池中移除。

> **错误 2：预处理期太短**
> 合成控制的权重通过最小化预处理期残差得到。如果预处理期太短（< 5 期），权重不可靠，极易过拟合。规则：预处理期 >> 预测变量数。

> **错误 3：只报告视觉结果，不报告推断**
> SCM 的标准推断依赖置换检验（permutation inference），必须报告 rank-based p-value，不能只说"图上看起来有效果"。

> **错误 4：忽略预测变量选择的任意性**
> 预测变量的选择会影响合成控制权重，存在"规格搜索"（specification fishing）风险。推荐：事先在论文中注册预测变量选择逻辑，或进行预测变量选择的敏感性分析。

> **错误 5：RMSPE ratio 安慰剂剪枝不当**
> 在置换检验中，预处理拟合很差（RMSPE 极大）的供体会扭曲 p-value。标准做法是移除预处理 RMSPE > 处理单位 × 2（或 × 5）的供体，然后计算 p-value。

> **错误 6：混淆 ATT 和 ATE**
> SCM 识别的是处理单位的处理效应（单个单位的因果效应），不能外推到其他单位。

---

## 输出规范

### 图表规范

**图1：主趋势图（Trends Plot）**
- X 轴：时间
- Y 轴：结果变量
- 实线：处理单位；虚线：合成控制
- 垂直虚线：处理时间点

**图2：Gap 图（Difference Plot）**
- 显示处理单位 - 合成控制的差值（ATT随时间变化）
- 包含置信区间（如果用 gsynth）

**图3：安慰剂图（Placebo Plot）**
- 灰色细线：各供体的 gap
- 黑色粗线：处理单位 gap
- 视觉上处理单位 gap 应在处理后明显偏离安慰剂分布

**图4：供体权重表**
- 非零权重供体的权重和特征对比

### 文件命名

```
output/
  sc_raw_trends.png            # 原始趋势图
  sc_main_results.png          # 合成控制主图（trends + gap）
  sc_placebo_inference.png     # 安慰剂推断图
  sc_loo_robustness.png        # Leave-one-out 图
  sc_unit_weights.csv          # 供体权重
  sc_predictor_weights.csv     # 预测变量权重
  sc_balance_table.csv         # 预处理期特征对比
  sc_att_by_period.csv         # 逐期 ATT 估计
```
