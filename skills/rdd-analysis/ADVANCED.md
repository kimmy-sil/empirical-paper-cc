# RDD 变体与拓展

> 按需加载。本文档覆盖五种断点类型、RDiT（时间 RDD）、地理 RDD、多门槛 RDD、RD-DD，以及前沿方法标注。

---

## 五种断点类型分类表

| 断点类型 | 代表场景 | 适用条件 | 核心假设 | 常见陷阱 | 代表文献 |
|----------|----------|----------|----------|----------|----------|
| **分数/声誉断点** | 高考分数线、录取分数线、信用评级阈值 | 评分规则公开，断点唯一且确定 | Sharp RDD: D = 1(R ≥ c)，无精确操控评分 | 分数堆积（最常见）；跨年分数比较 | Lee (2008) 选举；Hoekstra (2009) 旗舰大学 |
| **年龄断点** | 退休年龄、养老金参保年龄、医保资格年龄 | 处理和取消分段实施；年龄测量稳定 | 年龄多为 Fuzzy RDD（选择性提前/延后参保） | 递归现象（特定年龄堆积）；年龄测量误差 | Card et al. (2008) 医保；Angrist & Krueger (1991) 教育 |
| **时间断点（RDiT）** | 政策实施日、法律修改日、交易开放日 | 时间序列数据；政策在确定日期生效 | 潜在趋势可控；无其他政策同期交叉 | 时间序列自相关；季节性控制不当；同期其他政策 | Hausman & Rapson (2018)；Davis (2008) 污染 |
| **地理断点** | 行政边界、气候分界线、历史分割线 | 地理边界干净；边界两侧可比 | 到边界距离为评分变量；人口特征在边界处连续 | 边界附近选择性居住/迁移；空间溢出效应 | Dell (2010) 内战；Huang et al. (2014) 秦岭-淮河 |
| **指标阈值** | AQI 阈值、资产负值线、财务门槛 | 阈值公开且清晰已知；个体无法精确操控 | 指标评分无操控；多阈值不产生交互 | 多阈值（带宽不够）；报告性操控（如 AQI 附近) | Chen et al. (2013) AQI；Grembi et al. (2016) 财务规则 |

---

## RDiT（时间断点回归，Regression Discontinuity in Time）

### 与标准 RDD 的关键区别

| 维度 | 标准截面 RDD | RDiT（时间 RDD） |
|------|--------------|--------------------|
| 评分变量 | 截面指标（考分、指标值） | 时间（日/周/月） |
| 操控检验 | McCrary 密度检验必需 | 不可直接应用（时间不能"堆积"） |
| 主要威胁 | 操控、协变量不平衡 | 时间序列自相关、季节性、同时政策 |
| 内部效度假设 | 断点两侧潜在结果连续 | 政策日前后的趋势可控（无同期其他变化） |
| 额外检验 | 协变量平衡、Donut hole | 自相关检验、季节性控制、预期效应检验 |

> **方法论警告：RDiT 的内部效度通常弱于标准截面 RDD。**
>
> **核心原因：** 时间无法被视为"随机分配"（Hausman & Rapson 2018），
> 因此 RDiT **不适用 local randomization 解释框架**，
> 只能依赖 continuity-based 框架
> （即潜在结果和混杂因素在政策日附近连续变化）。
>
> **RDiT 存在的理由**正是因为所有单位同时经历政策变化、缺乏截面变异，
> 导致 DID 等需要处理组-对照组对比的方法不可用。
> 这不是 RDiT 的内部效度缺陷，而是它被需要的场景特征。
>
> 相对于标准截面 RDD，RDiT 面临以下**额外威胁**：
>
> | 威胁 | 说明 | 应对 |
> |------|------|------|
> | 无法检验操控 | 时间密度均匀，McCrary 检验不可用 | Donut RD 作为替代 |
> | 预期效应 | 个体可能提前调整行为 | 预期效应检验（假断点） |
> | 同期冲击 | 政策日可能伴随其他变化 | 协变量连续性检验 + 安慰剂地区 |
> | 序列相关 | 时间序列残差非独立（截面 RDD 不存在此问题） | HAC/Newey-West 标准误 |
> | 自回归效应 | Y_t 依赖 Y_{t-1}，污染处理效应估计 | 纳入滞后项控制 |
> | 带宽过宽 | 为增加样本量被迫扩大时间窗口，捕获时间趋势 | Augmented local linear 方法 |
>
> 因此，RDiT 需要比标准 RDD **更多的稳健性检验**才能建立可信度。

### R 代码模板（Hausman-Rapson 2018 框架）

```r
# R: RDiT 完整流程
library(rdrobust)
library(fixest)
library(dplyr)
library(lubridate)   # 日期处理
library(tseries)     # 自相关检验

# ---- Step 1：构造时间评分变量 ----
policy_date <- as.Date("2015-01-01")

df <- df %>%
  mutate(
    t_centered   = as.numeric(date - policy_date),   # 以天为单位
    post_policy  = as.integer(t_centered >= 0),
    month_of_year = month(date),
    day_of_week   = wday(date),
    yday          = yday(date)
  )

# ---- Step 2：自相关检验（内部效度关键）----
outcome_ts <- ts(df$outcome[order(df$date)], frequency = 365)
ljung_test <- Box.test(outcome_ts, lag = 20, type = "Ljung-Box")
cat(sprintf("Ljung-Box 自相关检验 p = %.4f\n", ljung_test$p.value))
if (ljung_test$p.value < 0.05) {
  cat("⚠️  存在显著自相关，需在模型中控制（加入滞后项 or Fourier 项）\n")
}

# ---- Step 3：Fourier 季节性控制 ----
df <- df %>%
  mutate(
    sin_yearly = sin(2 * pi * yday / 365),
    cos_yearly = cos(2 * pi * yday / 365),
    sin_2yr    = sin(4 * pi * yday / 365),
    cos_2yr    = cos(4 * pi * yday / 365),
    sin_weekly = sin(2 * pi * day_of_week / 7),
    cos_weekly = cos(2 * pi * day_of_week / 7)
  )

# ---- Step 4：RDiT 主估计（rdrobust 时间维度）----
rdit_main <- rdrobust(
  y        = df$outcome,
  x        = df$t_centered,
  c        = 0,
  p        = 1,
  kernel   = "triangular",
  bwselect = "mserd"
)
summary(rdit_main)
h_days <- rdit_main$bws["h", 1]
cat(sprintf("RDiT 带宽 = %.0f 天\n", h_days))

# ---- Step 5：含季节性控制的 RDiT（feols）----
# 限制在最优带宽内
df_bw <- df %>% filter(abs(t_centered) <= h_days)

res_rdit_fe <- feols(
  outcome ~ t_centered + i(post_policy, t_centered) +   # 断点两侧不同斜率
    sin_yearly + cos_yearly + sin_2yr + cos_2yr +        # Fourier 季节控制
    sin_weekly + cos_weekly |                             # 周内模式
    month_of_year + day_of_week,                          # 月份/周固定效应
  data    = df_bw,
  cluster = ~week_id   # 按周聚类（时间序列推荐）
)
summary(res_rdit_fe)

# ---- Step 6：预期效应检验（应不显著）----
df_pre_window <- df %>%
  filter(t_centered >= -60, t_centered < 0) %>%
  mutate(placebo_post = as.integer(t_centered >= -30))

res_anticipation <- feols(
  outcome ~ t_centered + i(placebo_post, t_centered) +
    sin_yearly + cos_yearly | month_of_year,
  data = df_pre_window
)
cat("预期效应检验（placebo_post 系数应不显著）：\n")
summary(res_anticipation)

# ---- Step 7：带宽敏感性 ----
bw_days <- c(30, 60, 90, 180, 365)

rdit_bw_res <- lapply(bw_days, function(h) {
  res <- rdrobust(y = df$outcome, x = df$t_centered, c = 0, p = 1, h = h)
  data.frame(bw_days = h,
             coef_bc = res$coef["Bias-Corrected", 1],
             ci_lo   = res$ci["Robust", 1],
             ci_hi   = res$ci["Robust", 2])
}) %>% bind_rows()
print(rdit_bw_res)
```

**RDiT 额外检验清单：**
1. 自相关检验（Ljung-Box / ADF）
2. 季节性控制（Fourier 项 + 月/周固定效应）
3. 预期效应检验（政策日前窗口假断点）
4. 同期政策识别（记录研究期间同日期实施的其他政策）
5. 自回归效应控制（在 feols 中加入滞后项 `lag(outcome, 1)`）
6. HAC 标准误（若残差存在序列相关，使用 Newey-West 标准误替代聚类标准误）

---

## 地理 RDD

### 方法框架

以地理边界为断点，个体到边界的（符号）距离为评分变量：

$$R_i = \text{sign}(\text{side}) \times \text{dist}(i, \text{boundary})$$

**额外识别挑战：**
- 边界附近的选择性居住（人口、企业在边界两侧分布不均）
- 空间溢出效应（边界两侧潜在结果不独立）
- 地理近邻内生性

### R 代码（sf + rdrobust）

```r
# R: 地理 RDD 完整流程
library(sf)
library(rdrobust)
library(dplyr)
library(ggplot2)

# ---- Step 1：读入地理数据 ----
df_sf <- st_as_sf(df_points, coords = c("longitude", "latitude"), crs = 4326)

boundary_line <- st_read("data/boundary_line.shp") %>%
  st_transform(crs = 4326)

# ---- Step 2：计算到边界的符号距离 ----
# 无符号距离
df_sf$dist_km <- as.numeric(
  st_distance(df_sf, st_union(boundary_line))
) / 1000

# 确定方向（处理侧为正）
treatment_area <- st_read("data/treatment_region.shp") %>%
  st_transform(crs = 4326)
df_sf$treated <- as.integer(
  st_within(df_sf, treatment_area, sparse = FALSE)[, 1]
)
df_sf <- df_sf %>%
  mutate(r_geo = ifelse(treated == 1, dist_km, -dist_km))

df <- df_sf %>% st_drop_geometry()

# ---- Step 3：地理 RDD 可视化 ----
ggplot(df_sf) +
  geom_sf(aes(color = as.factor(treated)), alpha = 0.4, size = 0.5) +
  geom_sf(data = boundary_line, color = "black", linewidth = 1.2) +
  scale_color_manual(values = c("steelblue", "tomato"),
                     labels = c("Control", "Treated")) +
  labs(title = "Geographic RDD: Sample Distribution", color = "") +
  theme_minimal()

rdplot(y = df$outcome, x = df$r_geo, c = 0,
       x.label = "Distance to Boundary (km, signed)",
       y.label = "Outcome",
       title = "Geographic RDD Plot")

# ---- Step 4：主估计 ----
rdd_geo <- rdrobust(y = df$outcome, x = df$r_geo, c = 0,
                    p = 1, kernel = "triangular", bwselect = "mserd")
summary(rdd_geo)

# ---- Step 5：边界两侧协变量连续性检验 ----
geo_covariates <- c("age_share", "income_log", "education_yr", "urban_share")
geo_balance <- lapply(geo_covariates, function(cov) {
  res <- rdrobust(y = df[[cov]], x = df$r_geo, c = 0, p = 1, bwselect = "mserd")
  data.frame(covariate = cov,
             coef_bc   = res$coef["Bias-Corrected", 1],
             p_robust  = res$pv["Robust", 1])
}) %>% bind_rows()
print(geo_balance)

# ---- Step 6：空间溢出检验 ----
# 取带宽内样本，检验控制侧是否存在溢出
df_bw_geo <- df %>% filter(abs(r_geo) <= rdd_geo$bws["h", 1])

# 在控制侧（r_geo < 0）内部设置虚假断点，检验溢出
df_ctrl <- df_bw_geo %>% filter(r_geo < 0)
spillover_test <- rdrobust(y = df_ctrl$outcome, x = df_ctrl$r_geo,
                            c = -rdd_geo$bws["h", 1] / 2, p = 1)
cat("溢出检验（控制侧内部假断点，应不显著）:\n")
summary(spillover_test)
```

**地理 RDD 额外检验：**

| 检验 | 方法 | 目的 |
|------|------|------|
| 人口特征连续性 | 协变量 RDD | 证明边界两侧人口特征连续 |
| 空间溢出评估 | 控制侧内假断点 | 边界效应是否渗透到控制侧 |
| 选择性迁居 | Donut hole（排除边界最近个体） | 排除选择性居住偏误 |
| 安慰剂边界 | 用附近平行边界做安慰剂 | 边界不应显示虚假断点 |

**前沿替代：rd2d（Cattaneo, Titiunik, Vazquez-Bare 2025）**

> `rd2d` 包解决了传统地理 RDD 中**"将二维距离压缩为一维"的信息损失问题**。
> 标准做法是计算每个观测到边界的最短距离，将二维位置信息压缩为一个标量，
> 这会丢失沿边界方向的变异信息。`rd2d` 直接在二维空间中进行非参数估计，
> 支持任意方向的边界和二维核函数，是地理 RDD 的正确前进方向。

```r
# rd2d 包（Cattaneo, Titiunik, Vazquez-Bare 2025，前沿方法）
# devtools::install_github("rdpackages/rd2d")
# library(rd2d)
# rdd_2d <- rd2d(y = df$outcome,
#                x1 = df$longitude,
#                x2 = df$latitude,
#                c1 = boundary_x_coords,
#                c2 = boundary_y_coords)
# 优势：避免一维压缩的信息损失，捕获沿边界方向的异质效应
```

---

## 多门槛 RDD

### 适用场景

- 全国不同地区/学校有不同录取分数线
- 同一政策在不同年份变动阈值
- 指标类行业（如 AQI 预警级别多档阈值）

### rdmc pooling 方法

```r
# R: rdmc（Cattaneo-Titiunik-Vazquez-Bare，rdrobust 包内置）
library(rdrobust)
library(dplyr)

# ---- 方法1：标准化 + 合并（最简单）----
df_norm <- df %>%
  mutate(r_normalized = running_var - cutoff,  # 各自对阈值中心化
         above_norm   = as.integer(r_normalized >= 0))

rdd_pooled <- rdrobust(y = df_norm$outcome, x = df_norm$r_normalized, c = 0,
                       p = 1, bwselect = "mserd")
summary(rdd_pooled)

# ---- 方法2：rdmc（允许各断点带宽不同）----
cutoffs_vec <- sort(unique(df$cutoff))
cat(sprintf("共 %d 个断点: %s\n", length(cutoffs_vec), 
            paste(cutoffs_vec, collapse = ", ")))

rdd_mc <- rdmc(
  y = df$outcome,
  x = df$running_var,
  c = cutoffs_vec        # 向量：各个断点值
)
summary(rdd_mc)
# 输出：每个断点的 RDD 估计 + 加权平均 pooled estimate

# ---- 方法3：分断点单独估计 + 异质性检验 ----
by_cutoff <- lapply(cutoffs_vec, function(c_val) {
  df_c <- df %>% filter(cutoff == c_val) %>%
    mutate(r_c = running_var - c_val)

  if (min(sum(df_c$r_c >= 0), sum(df_c$r_c < 0)) < 20) return(NULL)

  res <- rdrobust(y = df_c$outcome, x = df_c$r_c, c = 0, p = 1)
  data.frame(cutoff  = c_val,
             coef_bc = res$coef["Bias-Corrected", 1],
             ci_lo   = res$ci["Robust", 1],
             ci_hi   = res$ci["Robust", 2],
             n_left  = res$N_h[1],
             n_right = res$N_h[2])
}) %>% bind_rows()
print(by_cutoff)

# 各断点系数图
ggplot(by_cutoff, aes(x = as.factor(cutoff), y = coef_bc)) +
  geom_point(size = 3, color = "steelblue") +
  geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.2, color = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  coord_flip() +
  labs(title = "Multi-Cutoff RDD Estimates", x = "Cutoff Value",
       y = "RDD Estimate (Bias-corrected, Robust CI)") +
  theme_minimal()
```

---

## RD-DD（断点 + 双重差分结合）

**适用场景：** 断点附近存在随时间变化的混杂因素，仅靠截面 RDD 无法控制，需要结合 DID 吸收时间趋势。

**识别假设：**
1. **RDD 假设**：阈值附近连续性（无精确操控评分变量）
2. **DID 假设**：阈值两侧的时间趋势在政策前平行（条件平行趋势）

### Estimand 声明

> **RD-DD 的估计对象（estimand）与纯 RDD 不同，需要单独声明。**
>
> - **纯 RDD** 的 estimand 是 **LATE at cutoff**（断点处局部平均处理效应）
> - **RD-DD** 的 estimand 是 **断点附近 + 政策前后的差中差效应**，
>   即"阈值两侧的结果差异在政策前后的变化量"
> - RD-DD 同时利用了截面维度（断点两侧）和时间维度（政策前后）的变异
> - 因此 RD-DD 的外部效度范围不同于纯 RDD：
>   它不仅局限于 R ≈ c 的个体，还受制于特定的政策时间窗口
>
> **标准声明模板：**
> ```
> 本文 RD-DD 估计量识别的是：评分变量在阈值 [c] 附近的个体，
> 在政策实施前后，处于阈值以上（处理组）与以下（对照组）之间
> 的结果差异变化。该估计量同时要求 RDD 的连续性假设
> 和 DID 的条件平行趋势假设成立。
> ```

### 带宽选择：必须基于政策前截面

> ⚠️ **RD-DD 的带宽选择应基于政策前截面 RDD，而非全样本。**
>
> **原因：** 政策后数据已包含处理效应，用政策后数据选带宽会导致选择偏误——
> 带宽算法会将处理效应误认为是函数形状的特征，从而系统性地选择过窄或过宽的带宽。
> 政策前截面不含处理效应，能提供无偏的带宽估计。

```r
# R: RD-DD 代码模板（rdrobust + fixest 结合）
library(rdrobust)
library(fixest)
library(dplyr)

# 数据要求：面板数据（单元 × 时间），含评分变量、时间虚拟、处理虚拟
df <- df %>%
  mutate(
    r_centered   = running_var - cutoff,
    above_cutoff = as.integer(r_centered >= 0),
    rddd_treat   = above_cutoff * post          # 核心识别变量
  )

# ---- Step 1：确定最优带宽（用政策前截面 RDD）----
# ⚠️  必须用政策前数据，避免处理效应污染带宽选择
bw_pre <- rdrobust(
  y = df$outcome[df$post == 0],
  x = df$r_centered[df$post == 0],
  c = 0, p = 1, bwselect = "mserd"
)
h_opt <- bw_pre$bws["h", 1]
cat(sprintf("RD-DD 带宽（基于政策前 RDD）= %.4f\n", h_opt))

# ---- Step 2：限制带宽内样本 ----
df_bw <- df %>% filter(abs(r_centered) <= h_opt)

# ---- Step 3：RD-DD 主估计 ----
res_rddd <- feols(
  outcome ~ rddd_treat +
    r_centered + above_cutoff + post +            # 主效应
    r_centered:post + r_centered:above_cutoff |   # 评分 × 时间/处理交互
    unit_fe + year_fe,
  data    = df_bw,
  cluster = ~unit_id
)
summary(res_rddd)
cat("rddd_treat 系数 = RD-DD 估计量\n")

# ---- Step 4：预趋势检验（条件平行趋势）----
df_pre <- df_bw %>% filter(post == 0)

res_pretrend <- feols(
  outcome ~ i(year, above_cutoff, ref = base_year) + r_centered | unit_fe,
  data    = df_pre,
  cluster = ~unit_id
)
iplot(res_pretrend, main = "Pre-trend Check: Threshold × Year")
# 期望：所有政策前年份系数在 0 附近，联合检验不显著

# ---- Step 5：带宽敏感性 ----
rddd_bw_sens <- lapply(c(0.5, 0.75, 1.0, 1.25, 1.5) * h_opt, function(h) {
  df_h <- df %>% filter(abs(r_centered) <= h)
  res  <- feols(
    outcome ~ rddd_treat + r_centered + above_cutoff + post +
      r_centered:post + r_centered:above_cutoff | unit_fe + year_fe,
    data = df_h, cluster = ~unit_id
  )
  data.frame(bw_pct = h / h_opt, bw_val = h,
             coef   = coef(res)["rddd_treat"],
             se     = se(res)["rddd_treat"],
             n      = nrow(df_h))
}) %>% bind_rows()
print(rddd_bw_sens)
```

---

## 前沿标注

### rdhte（RDD 原生异质性效应）

```r
# rdhte 包：RDD 框架内直接估计异质处理效应（Cattaneo et al.）
# install.packages("rdhte")
# library(rdhte)
# 
# rdd_hte <- rdhte(
#   y        = df$outcome,
#   x        = df$r_centered,
#   w        = df[, heterogeneity_vars],   # 异质性维度
#   c        = 0,
#   p        = 1,
#   bwselect = "mserd"
# )
# summary(rdd_hte)
# 
# 优势：在局部多项式框架内直接估计 CATE at cutoff
# 不需要分子样本（子样本 RDD 有带宽减半问题）
```

### rd2d（地理 RDD 专用，Cattaneo 2025）

参见地理 RDD 部分。rd2d 支持二维边界的非参数 RDD 估计，是地理 RDD 的现代替代方法。

### RDD 溢出效应

当处理存在空间/网络溢出时（如政策影响邻近地区），标准 RDD 的 LATE 被污染：

```r
# R: 溢出效应诊断（修改版地理 RDD）
# 方法：同时纳入处理距离和溢出距离
# - r_treat: 到处理边界的距离（处理侧正）
# - r_spill: 到处理边界的距离（对控制侧的溢出路径）

# 检验：控制侧内部是否随距离边界越近效应越大
df_ctrl_side <- df %>% filter(r_geo < 0)

# 控制侧的结果是否随距离边界变化（溢出的特征）
spillover_gradient <- feols(
  outcome ~ r_geo + exog_ctrl | unit_fe + year_fe,
  data    = df_ctrl_side,
  cluster = ~unit_id
)
# r_geo 系数（控制侧）应为 0；若显著为正，提示向控制侧溢出

# 参考文献：
# Butts (2021) - RDD with spillovers（SUTVA 放松）
# Cattaneo, Idrobo, Titiunik (2024) - RDD 教材最新版
```

### 推荐阅读

| 主题 | 文献 | 核心贡献 |
|------|------|----------|
| RDD 基础 | Hahn, Todd, Van der Klaauw (2001) *Econometrica* | 局部多项式 RDD 理论基础 |
| 带宽选择 | Calonico, Cattaneo, Titiunik (2014) *Econometrica* | CCT 带宽、Bias-corrected Robust CI |
| 密度检验 | Cattaneo, Jansson, Ma (2018) *JASA* | rddensity，优于 McCrary 2008 |
| 高阶多项式 | Gelman & Imbens (2019) *JBES* | p ≥ 3 的危险性 |
| Donut RD | Cattaneo, Titiunik, Vazquez-Bare (2023) | Donut 精度警告 |
| 多断点 | Cattaneo, Titiunik, Vazquez-Bare (2016/2020) | rdmc 理论与实现 |
| RDiT | Hausman & Rapson (2018) *QJE* | 时间 RDD 框架 |
| 地理 RDD | Dell (2010) *Econometrica* | Mita 制度边界经典应用 |
| rd2d | Cattaneo, Titiunik, Vazquez-Bare (2025) | 二维地理 RDD |
| Fuzzy RDD | Imbens & Lemieux (2008) *JE* | Fuzzy RDD 综述 |
