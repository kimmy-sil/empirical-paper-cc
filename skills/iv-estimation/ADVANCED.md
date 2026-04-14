# IV 高级方法

> 按需加载。本文档覆盖：LATE/MTE 框架、IV 四大流派、Bartik/Shift-Share、DID 型 IV、Jackknife 诊断、Complier 分析、前沿进展。

---

## LATE / MTE 框架

### LATE 定义（Imbens-Angrist 1994）

$$\text{LATE} = E[Y_i(1) - Y_i(0) \mid D_i(1) > D_i(0)] = \frac{\text{Cov}(Y, Z)}{\text{Cov}(D, Z)}$$

**四类个体：**

| 类型 | D(Z=0) | D(Z=1) | IV 识别？ |
|------|--------|--------|----------|
| **Complier** | 0 | 1 | ✓ |
| **Always-taker** | 1 | 1 | ✗ |
| **Never-taker** | 0 | 0 | ✗ |
| **Defier**（单调性排除） | 1 | 0 | ✗ |

**Complier 比例 = 第一阶段系数。** 不同 IV 识别不同 Complier 群体的 LATE。

### Abadie κ-weighting（Complier 特征描述）

```r
# R: κ-weighting 描述 Complier 特征
library(fixest); library(dplyr)

# Step 1: 估计倾向分数 P(Z=1|W)
ps_model <- glm(instrument_z ~ exog_ctrl1 + exog_ctrl2,
                data = df, family = binomial)
df$ps <- predict(ps_model, type = "response")

# Step 2: 构造 κ 权重
df <- df %>%
  mutate(
    kappa = endog_x - ps * endog_x / ps -
            (1 - endog_x) * (1 - ps) * endog_x / (1 - ps)
  )
# 简化版：二元 Z 二元 D
df <- df %>%
  mutate(kappa = 1 - (endog_x * (1 - instrument_z)) / (1 - ps) -
                     ((1 - endog_x) * instrument_z) / ps)

# Step 3: 对比 Complier vs 全样本特征
chars <- c("age", "education", "income_baseline")
complier_chars <- sapply(chars, function(v) {
  c(full_sample = mean(df[[v]], na.rm = TRUE),
    complier    = weighted.mean(df[[v]], w = pmax(df$kappa, 0), na.rm = TRUE))
})
print(t(complier_chars))
```

### MTE 估计（localIV 包）

MTE 将异质性处理效应按"抵制处理倾向" u_D 连续展开：

$$\text{MTE}(x, u_D) = E[Y(1) - Y(0) \mid X = x, U_D = u_D]$$

```r
# R: localIV 包
library(localIV)
ps_model <- glm(treatment ~ instrument + exog_ctrl1 + exog_ctrl2,
                data = df, family = binomial(link = "probit"))
df$prop_score <- predict(ps_model, type = "response")

mte_fit <- mte(
  selection = treatment ~ prop_score,
  outcome   = outcome ~ exog_ctrl1 + exog_ctrl2 + prop_score,
  data = df, bw = 0.1
)
plot(mte_fit)
mte_weights <- mte_weights(mte_fit)
print(mte_weights)  # ATE、ATT、ATU 的 MTE 加权
```

**ivmte 包（bounds/外推）：** 见 Mogstad-Santos-Torgovitsky (2018)，设定 MTE 函数空间求目标估计量的部分识别区间。

```r
# R: ivmte（点识别到集识别）
library(ivmte)
mte_ivmte <- ivmte(
  ivlike = outcome ~ treatment + exog_ctrl | instrument + exog_ctrl,
  target = "ate",
  data   = df,
  propensity = treatment ~ instrument + exog_ctrl,
  m0 = ~ uSplines(degree = 2, knots = c(0.3, 0.6), intercept = TRUE),
  m1 = ~ uSplines(degree = 2, knots = c(0.3, 0.6), intercept = TRUE)
)
result <- ivmtesolve(mte_ivmte)
cat(sprintf("ATE bounds: [%.4f, %.4f]\n", result$bounds[1], result$bounds[2]))
```

**Python 注意：** Python 无成熟 MTE 包，建议用 R。

---

## IV 四大流派

| 流派 | IV 来源 | 典型论证 | 代表文献 |
|------|--------|---------|---------|
| **结构型** | 成本端变量 | 供给冲击不直接影响需求 | Berry-Levinsohn-Pakes (1995) |
| **自然实验** | 外生事件（征兵/降雨） | 随机或准随机分配 | Angrist (1990) |
| **Shift-Share** | 历史份额 × 全国变动 | 份额/变动各自外生 | Bartik (1991); GPSS (2020) |
| **法官/官员设计** | 随机分配机制 | 分配独立于个体特征 | Kling (2006) |

---

## Bartik / Shift-Share IV

$$Z_l = \sum_k s_{lk,t_0} \times g_{k,t}$$

### 两种识别路径

| 框架 | 假设 | 检验 |
|------|------|------|
| **GPSS（份额外生）** | 历史份额 s_lk 外生 | Rotemberg 权重 + 多重检验 |
| **BHJ（变动外生）** | 行业变动 g_k 外生 | ssaggregate 行业级检验 |

```r
# R: Bartik IV 构造
library(dplyr); library(tidyr)

construct_bartik_iv <- function(df_long, unit_col, time_col, industry_col,
                                 employment_col, base_year, end_year) {
  base_shares <- df_long %>%
    filter(.data[[time_col]] == base_year) %>%
    group_by(.data[[unit_col]]) %>%
    mutate(share_lk = .data[[employment_col]] / sum(.data[[employment_col]], na.rm = TRUE)) %>%
    ungroup() %>%
    select(all_of(c(unit_col, industry_col)), share_lk)

  national_growth <- df_long %>%
    filter(.data[[time_col]] %in% c(base_year, end_year)) %>%
    group_by(.data[[industry_col]], .data[[time_col]]) %>%
    summarise(nat_emp = sum(.data[[employment_col]], na.rm = TRUE), .groups = "drop") %>%
    pivot_wider(names_from = all_of(time_col), values_from = nat_emp) %>%
    mutate(g_k = (.data[[as.character(end_year)]] - .data[[as.character(base_year)]]) /
                  .data[[as.character(base_year)]]) %>%
    select(all_of(industry_col), g_k)

  base_shares %>%
    left_join(national_growth, by = industry_col) %>%
    group_by(.data[[unit_col]]) %>%
    summarise(bartik_iv = sum(share_lk * g_k, na.rm = TRUE), .groups = "drop")
}

bartik_df <- construct_bartik_iv(emp_panel, "city_id", "year", "industry_code",
                                  "employment", 1990, 2000)
df_main <- df_main %>% left_join(bartik_df, by = "city_id")

iv_bartik <- feols(outcome ~ exog_ctrl | city_fe + year_fe | endog_x ~ bartik_iv,
                   data = df_main, cluster = ~city_id)
```

### GPSS 诊断

```r
# R: Rotemberg 权重（哪些行业贡献最大识别力）
# devtools::install_github("paulgp/bartik-weight")
# 手动：各行业历史份额分别作 IV 做过度识别检验
```

### BHJ 诊断

```r
# R: ssaggregate（转为行业冲击级别 IV 问题）
library(ssaggregate)
bartik_agg <- ssaggregate(data = df_main, n = "n_units", s = "share_lk",
                          t = "year", l = "unit_id", k = "industry_code",
                          y = "outcome", x = "endog_x")
# 行业冲击级别：检验 g_k 与基期特征无关
bhj_balance <- feols(g_k ~ base_char1 + base_char2 | year,
                     data = bartik_agg$industry_data,
                     weights = ~bartik_agg$industry_data$weight)
```

---

## DID 型 IV

**构造：** $Z_{it} = \text{Post}_t \times X_i^{(0)}$（政策冲击 × 基期特征）

```r
# R: DID 型 IV
library(fixest)
df <- df %>% mutate(iv_did = post * baseline_char)

iv_did_main <- feols(
  outcome ~ exog_ctrl | unit_fe + year_fe | endog_policy ~ iv_did,
  data = df, cluster = ~unit_id
)

# 排他性检验：基期特征是否预测处理前趋势？
pre_trend_check <- feols(
  outcome ~ baseline_char:year | unit_fe + year_fe,
  data = df %>% filter(post == 0), cluster = ~unit_id
)
# 期望：交互系数不显著

# plausexog UCI
library(plausexog)
uci_result <- uci(y = df$outcome, x = df$endog_policy, z = df$iv_did,
                  w = as.matrix(df[, exog_controls]),
                  delta = 0.05 * abs(coef(iv_did_main)["fit_endog_policy"]))
```

**DID IV 排他性三要素：**
1. 基期特征 X_0 不预测处理前趋势
2. 政策冲击 Post 外生（如全国统一推行时间）
3. X_0 影响政策强度但不直接影响政策后结果

---

## Jackknife 聚类影响力诊断

```r
# R: 逐一剔除聚类
jackknife_cluster_iv <- function(data, formula_iv, cluster_var, treated_var) {
  clusters <- unique(data[[cluster_var]])
  main_res <- feols(formula_iv, data = data,
                    cluster = as.formula(paste0("~", cluster_var)))
  beta_main <- coef(main_res)[treated_var]

  jk_results <- lapply(clusters, function(cl) {
    df_sub <- data[data[[cluster_var]] != cl, ]
    tryCatch({
      res <- feols(formula_iv, data = df_sub,
                   cluster = as.formula(paste0("~", cluster_var)))
      data.frame(excluded = cl, beta = coef(res)[treated_var],
                 change_pct = (coef(res)[treated_var] - beta_main) / abs(beta_main) * 100)
    }, error = function(e) NULL)
  })

  jk_df <- do.call(rbind, Filter(Negate(is.null), jk_results))
  jk_df <- jk_df[order(abs(jk_df$change_pct), decreasing = TRUE), ]
  flagged <- jk_df[abs(jk_df$change_pct) > 20, ]
  if (nrow(flagged) > 0) {
    cat(sprintf("⚠️  %d 个聚类剔除后变化 > 20%%\n", nrow(flagged)))
    print(flagged)
  } else {
    cat("✓  所有聚类剔除后变化 < 20%\n")
  }
  invisible(jk_df)
}
```

---

## 多 IV 比较 + Hausman 检验

```r
# R: 不同 IV 估计量比较（Complier 异质性检验）
library(fixest)
iv1 <- feols(outcome ~ exog | FE | endog ~ instrument_1, data = df, cluster = ~id)
iv2 <- feols(outcome ~ exog | FE | endog ~ instrument_2, data = df, cluster = ~id)
iv3 <- feols(outcome ~ exog | FE | endog ~ instrument_3, data = df, cluster = ~id)

etable(iv1, iv2, iv3, headers = c("IV1", "IV2", "IV3"),
       title = "不同 IV 估计量比较（不同 Complier 群体的 LATE）")
# 差异显著 → 处理效应异质性显著
```

---

## 设计层决策案例速查

### 何时用 IV

| 场景 | IV 来源 | 为什么需要 | 核心论证 |
|------|--------|-----------|---------|
| RCT 中分离机制（如"知道"vs"种子"效应） | 随机分组（开放 vs 双盲） | 内生变量 = 是否知道种子类型 | 外生性由随机化保证；F=70.5 |
| 信息实验中的不依从 | 随机分配信息 | 内生变量 = 预期更新幅度 | 收到≠相信，IV 修正不依从 |
| 观察性研究中的反向因果 | leave-one-out 均值 / Bartik | 内生变量 = 企业智能化水平 | 同行扩散 / 教育供给冲击 |

### 何时不用 IV

| 场景 | 替代策略 | 为什么不需要 |
|------|---------|-------------|
| RCT 完全依从 | ATE 直接比较 | 无不依从，ITT = ATE |
| 关心政策意图效果 | ITT / reduced form | 不需要修正到 LATE |
| 需要深层参数（风险偏好等） | 结构模型 | IV 只给 LATE，无法识别结构参数 |
| 有可信的政策断点 | RDD | 比 IV 假设更弱 |

---

## 前沿标注

### 部分单调性（Partial Monotonicity）

多值处理时标准单调性过强，需放松为部分单调性。

```r
# R: 多值处理的分段 LATE
library(ivreg)
# D ∈ {0, 1, 2}：分别估计 LATE(0→1) 和 LATE(1→2)
late_01 <- ivreg(outcome ~ treatment | instrument, data = df %>% filter(treatment %in% c(0,1)))
late_12 <- ivreg(outcome ~ treatment | instrument, data = df %>% filter(treatment %in% c(1,2)))
```

### 连续内生变量 LATE 拓展

β_2SLS 是异质效应 dY/dX 按 |dX/dZ| 加权的平均，权重大的是"更服从工具"的个体。

```r
# R: Z → X 响应曲线（识别谁是 Complier）
library(ggplot2)
df <- df %>% mutate(z_quartile = ntile(instrument_z, 4))
responsiveness <- df %>%
  group_by(z_quartile) %>%
  summarise(mean_z = mean(instrument_z), mean_x = mean(endog_x), n = n())

ggplot(responsiveness, aes(x = mean_z, y = mean_x)) +
  geom_point(aes(size = n)) + geom_smooth(method = "lm") +
  labs(title = "Z → X 响应曲线", x = "IV (分位数均值)", y = "内生变量 (分位数均值)") +
  theme_minimal()
```

### 推荐阅读

| 主题 | 文献 | 核心贡献 |
|------|------|---------|
| LATE 基础 | Imbens & Angrist (1994) *Econometrica* | LATE 定义 |
| 弱工具新标准 | Lee et al. (2022) *AER* | F > 104.7 |
| MTE | Heckman & Vytlacil (2005) *Econometrica* | MTE 统一框架 |
| GPSS Bartik | Goldsmith-Pinkham et al. (2020) *AER* | 份额外生性 |
| BHJ Bartik | Borusyak et al. (2022) *RES* | 变动外生性 |
| 排他性约束 | Conley et al. (2012) *RES* | UCI 方法 |
| Complier 描述 | Abadie (2003) *RES* | κ-weighting |
