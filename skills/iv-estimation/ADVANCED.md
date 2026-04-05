# IV 高级方法

> 按需加载。本文档覆盖 LATE/MTE 框架、IV 四大流派、Bartik/Shift-Share、DID 变量作 IV、以及前沿进展。

---

## LATE / MTE 框架

### LATE 定义（Imbens-Angrist 1994）

**局部平均处理效应（LATE）：**

$$\text{LATE} = E[Y_i(1) - Y_i(0) \mid D_i(1) > D_i(0)] = \frac{\text{Cov}(Y, Z)}{\text{Cov}(D, Z)}$$

其中 D_i(z) 是个体 i 在工具 Z = z 下的潜在处理状态。

**成立条件（Imbens-Angrist 框架）：**
1. **相关性（Relevance）**：E[D(1)] ≠ E[D(0)]，即工具改变处理概率
2. **独立性（Independence）**：Z ⊥ {Y(0), Y(1), D(0), D(1)}，工具外生
3. **排他性（Exclusion）**：Z 只通过 D 影响 Y
4. **单调性（Monotonicity）**：D_i(1) ≥ D_i(0)，∀i（无 Defier）

### 四类个体

| 类型 | 定义 | D(Z=0) | D(Z=1) | IV 识别？ |
|------|------|---------|---------|-----------|
| **Complier** | 受工具影响改变处理状态 | 0 | 1 | ✓ 被 LATE 识别 |
| **Always-taker** | 无论工具如何都接受处理 | 1 | 1 | ✗ |
| **Never-taker** | 无论工具如何都拒绝处理 | 0 | 0 | ✗ |
| **Defier** | 与工具方向相反（单调性假设排除） | 1 | 0 | ✗（假设不存在） |

**Complier 比例估计：**
$$P(\text{Complier}) = E[D(1)] - E[D(0)] = \text{第一阶段系数}$$

即第一阶段系数（阈值处或 Z 变化引起的处理概率变化）直接估计 Complier 比例。

### 为什么不同 IV 给出不同估计

不同工具变量识别不同 Complier 群体的 LATE，当处理效应异质时（ITE 因人而异），不同 IV 的估计量天然不同。

```r
# R: 检验多个 IV 估计量差异（是否来自 Complier 异质性）
library(fixest)

# 三个工具变量分别估计
iv1 <- feols(outcome ~ exog | FE | endog ~ instrument_1, data = df, cluster = ~id)
iv2 <- feols(outcome ~ exog | FE | endog ~ instrument_2, data = df, cluster = ~id)
iv3 <- feols(outcome ~ exog | FE | endog ~ instrument_3, data = df, cluster = ~id)

etable(iv1, iv2, iv3, headers = c("IV1 (Z1)", "IV2 (Z2)", "IV3 (Z3)"),
       title = "不同工具变量估计量比较（不同 Complier 群体的 LATE）")

# 检验估计量差异是否显著（Hausman 风格）
# 若差异显著，说明处理效应异质性显著，LATE 对应 Complier 群体特定
```

### 设计本位 vs 结构性传统争论

| 维度 | 设计本位（Design-Based） | 结构性（Structural）|
|------|--------------------------|---------------------|
| 目标 | 识别特定 Complier 的 LATE | 识别可外推的结构参数 |
| IV 选择 | 来自自然实验，IV 本身即研究对象 | IV 为识别工具，经济机制为核心 |
| 异质性处理 | 接受 LATE 的局部性 | 尝试从 LATE 推断 MTE，外推 ATE |
| 代表文献 | Angrist & Pischke (2008) | Heckman & Vytlacil (2005) |

---

## MTE 估计（Marginal Treatment Effect）

MTE 框架将异质性处理效应按个体"抵制处理倾向"（unobservable resistance）连续展开：

$$\text{MTE}(x, u_D) = E[Y(1) - Y(0) \mid X = x, U_D = u_D]$$

其中 u_D = P(D=1|Z) 是处理倾向分位数。不同 IV 的 LATE 是 MTE 的不同加权平均。

### localIV R 包（推荐）

```r
# R: localIV 包（默认推荐，~30行）
# install.packages("localIV")
library(localIV)
library(dplyr)

# 数据要求：二元处理 D，连续工具 Z（或可转为倾向分数）
# Step 1: 估计倾向分数 P(D=1|Z, X)
ps_model <- glm(treatment ~ instrument + exog_ctrl1 + exog_ctrl2,
                data = df, family = binomial(link = "probit"))
df$prop_score <- predict(ps_model, type = "response")

# Step 2: 估计 LIV（Local Instrumental Variable = MTE）
mte_fit <- mte(
  selection = treatment ~ prop_score,   # D ~ P(Z,X)
  outcome   = outcome ~ exog_ctrl1 + exog_ctrl2 + prop_score,
  data      = df,
  bw        = 0.1   # 核带宽（可交叉验证选择）
)

# Step 3: 绘制 MTE 曲线
plot(mte_fit)
# X轴：u_D（抵制处理倾向，0=总是接受，1=总是拒绝）
# Y轴：MTE(u_D)

# Step 4: 从 MTE 计算各估计量
# ATE = ∫ MTE(u) du（在[0,1]上积分）
# ATT = ∫ MTE(u) f(u|D=1) du（按 D=1 加权）
# LATE = ∫ MTE(u) 1[u ∈ (p0, p1)] du（IV 对应区间）
mte_weights <- mte_weights(mte_fit)
print(mte_weights)  # ATE、ATT、ATU 的 MTE 加权

# Step 5: 外推 ATE（需外推假设，务必注明）
ate_from_mte <- integrate(function(u) predict(mte_fit, newdata = u), 0, 1)$value
cat(sprintf("MTE 外推 ATE = %.4f（需外推假设，局限性大）\n", ate_from_mte))
```

### ivmte R 包（bounds/外推，~40行）

```r
# R: ivmte 包（Mogstad-Santos-Torgovitsky 2018，点识别到集识别）
# install.packages("ivmte")
library(ivmte)

# ivmte 框架：设定 MTE 函数空间（多项式或分段线性），
# 求 LATE/ATE 等目标估计量的部分识别区间

# Step 1: 定义模型
mte_ivmte <- ivmte(
  ivlike      = outcome ~ treatment + exog_ctrl | instrument + exog_ctrl,
  target      = "ate",    # 目标：ATE（或"late", "att", "atu"）
  data        = df,
  components  = lateComponents(d0 = 0, d1 = 1),  # 目标处理范围
  propensity  = treatment ~ instrument + exog_ctrl,
  saturate    = FALSE,
  # MTE 函数空间：Bernstein 多项式（阶数越高，约束越弱）
  m0          = ~ uSplines(degree = 2, knots = c(0.3, 0.6), intercept = TRUE),
  m1          = ~ uSplines(degree = 2, knots = c(0.3, 0.6), intercept = TRUE)
)

# Step 2: 求解（线性规划）
result_ivmte <- ivmtesolve(mte_ivmte)

cat("目标估计量（ATE）的部分识别区间：\n")
cat(sprintf("  下界：%.4f\n", result_ivmte$bounds[1]))
cat(sprintf("  上界：%.4f\n", result_ivmte$bounds[2]))

# 若上下界均不含 0，处理效应方向可识别

# Step 3: 点识别（加额外约束）
mte_point <- ivmte(
  ivlike      = outcome ~ treatment + exog_ctrl | instrument + exog_ctrl,
  target      = "late",
  targetWeight0 = function(u) (u >= 0.3) * (u <= 0.7),
  targetWeight1 = function(u) (u >= 0.3) * (u <= 0.7),
  data        = df,
  propensity  = treatment ~ instrument + exog_ctrl
)
```

**Python 注意：** Python 无成熟 MTE 包，建议用 R 的 `localIV` 或 `ivmte`。若必须在 Python 中分析，可手动实现核回归版 LIV，但准确性和稳健性不如 R 包。

---

## IV 四大流派

| 流派 | 代表方法 | IV 来源 | 典型外生性论证 | 代表文献 |
|------|----------|---------|----------------|---------|
| **结构型** | 生产函数、需求系统 IV | 成本端变量（价格 → 数量） | 供给侧冲击不直接影响需求 | Berry-Levinsohn-Pakes (1995) |
| **自然实验** | 征兵抽签、出生日期、降雨 | 外生事件 | 随机或准随机分配 | Angrist (1990) 越战征兵 |
| **Shift-Share（Bartik）** | 行业就业冲击 | 历史份额 × 全国行业变动 | 份额/变动各自外生 | Bartik (1991), Goldsmith-Pinkham et al. (2020) |
| **法官/官员设计** | 随机分配的法官、检察官 | 随机分配机制 | 法官风格外生于当事人特征 | Kling (2006), Dobbie et al. (2018) |

---

## Bartik / Shift-Share IV

Bartik IV 将行业总量变动（shift）乘以地区历史行业结构（share）构造工具变量：

$$Z_l = \sum_k \underbrace{s_{lk,t_0}}_{\text{份额}} \times \underbrace{g_{k,t}}_{\text{变动}}$$

### 两种识别路径

| 框架 | 外生性假设 | 适用场景 | 实现 |
|------|------------|---------|------|
| **GPSS（份额外生）** | 历史份额 s_lk 外生（Goldsmith-Pinkham, Sorkin, Swift 2020） | 地区历史结构反映外生差异 | 强调份额的多重检验 |
| **BHJ（变动外生）** | 行业变动 g_k 外生（Borusyak, Hull, Jaravel 2022） | 全国行业冲击外生 | 强调行业冲击的随机化 |

```r
# R: Bartik IV 构造（通用代码）
library(dplyr)
library(tidyr)

construct_bartik_iv <- function(df_long, 
                                 unit_col, time_col, industry_col,
                                 employment_col, outcome_col,
                                 base_year, start_year, end_year) {
  # df_long: 单位×行业×时间 长格式面板
  # 步骤1：构造基期行业份额
  base_shares <- df_long %>%
    filter(.data[[time_col]] == base_year) %>%
    group_by(.data[[unit_col]]) %>%
    mutate(
      total_emp   = sum(.data[[employment_col]], na.rm = TRUE),
      share_lk    = .data[[employment_col]] / total_emp
    ) %>%
    ungroup() %>%
    select(all_of(c(unit_col, industry_col)), share_lk)
  
  # 步骤2：构造全国行业变动（剔除本地区）
  national_growth <- df_long %>%
    filter(.data[[time_col]] %in% c(base_year, end_year)) %>%
    group_by(.data[[industry_col]], .data[[time_col]]) %>%
    summarise(nat_emp = sum(.data[[employment_col]], na.rm = TRUE), .groups = "drop") %>%
    pivot_wider(names_from = all_of(time_col), values_from = nat_emp) %>%
    mutate(g_k = (.data[[as.character(end_year)]] - .data[[as.character(base_year)]]) /
                 .data[[as.character(base_year)]]) %>%
    select(all_of(industry_col), g_k)
  
  # 步骤3：合成 Bartik IV = Σ_k s_lk * g_k
  bartik_iv <- base_shares %>%
    left_join(national_growth, by = industry_col) %>%
    group_by(.data[[unit_col]]) %>%
    summarise(bartik_iv = sum(share_lk * g_k, na.rm = TRUE), .groups = "drop")
  
  return(bartik_iv)
}

# 使用示例
bartik_df <- construct_bartik_iv(
  df_long      = emp_panel,
  unit_col     = "city_id",
  time_col     = "year",
  industry_col = "industry_code",
  employment_col = "employment",
  outcome_col  = "log_wage",
  base_year    = 1990,
  start_year   = 1990,
  end_year     = 2000
)

# 合并 Bartik IV 到主数据
df_main <- df_main %>% left_join(bartik_df, by = "city_id")

# 主估计
iv_bartik <- feols(
  outcome ~ exog_ctrl | city_fe + year_fe | endogenous_x ~ bartik_iv,
  data    = df_main,
  cluster = ~city_id
)
summary(iv_bartik)
```

### GPSS 诊断（份额外生性检验）

```r
# R: GPSS 过度识别检验框架（将每个行业的历史份额作为单独 IV）
# Goldsmith-Pinkham et al. (2020) 思路

# 每个行业 k 的 Rotemberg 权重（衡量哪些行业贡献最多）
# 安装：devtools::install_github("paulgp/bartik-weight")
# library(bartik.weight)   # 或手动实现

# 手动：计算每个行业的识别贡献
rotemberg_weights <- function(base_shares, national_growth, fs_result) {
  # 各行业的 Rotemberg 权重 ≈ 对整体 Bartik IV 的贡献度
  contrib <- base_shares %>%
    left_join(national_growth, by = "industry_code") %>%
    group_by(industry_code) %>%
    summarise(
      alpha_k = cor(share_lk * g_k, base_shares$bartik_total) *
                sd(share_lk * g_k) / sd(base_shares$bartik_total),
      .groups = "drop"
    )
  return(contrib)
}

# 过度识别检验：以各行业份额为独立 IV 做 Hansen J 检验
# 实现参考 Goldsmith-Pinkham et al. (2020) Online Appendix
```

### BHJ 诊断（变动外生性检验）

```r
# R: ssaggregate（BHJ 框架，将 Bartik 问题转化为行业冲击级别）
# install.packages("ssaggregate")
library(ssaggregate)

# ssaggregate 将样本 n 个单位的 Shift-Share IV 问题
# 转化为 K 个行业冲击的 IV 问题（K << n 时效率更高）
bartik_agg <- ssaggregate(
  data      = df_main,
  n         = "n_units",         # 单位数量权重
  s         = "share_lk",        # 行业份额
  t         = "year",
  l         = "unit_id",
  k         = "industry_code",
  y         = "outcome",
  x         = "endogenous_x"
)

# 在行业冲击级别检验外生性（协变量平衡）
# 检验行业冲击 g_k 是否与行业基期特征无关
bhj_balance <- feols(g_k ~ base_char1 + base_char2 | year,
                     data    = bartik_agg$industry_data,
                     weights = ~bartik_agg$industry_data$weight)
summary(bhj_balance)
# 期望：g_k 与基期行业特征不相关
```

---

## DID 变量作 IV 的范式

**适用场景：** 政策冲击 × 个体特征构造工具变量，解决政策处理的内生性（如政策执行力度与潜在结果相关）。

**经典构造：**
$$Z_{it} = \underbrace{\text{Post}_t}_{\text{时间冲击}} \times \underbrace{X_i^{(0)}}_{\text{基期特征（外生）}}$$

```r
# R: DID 变量作 IV + plausexog 排他性检验
library(fixest)
library(plausexog)

# Step 1: 构造 DID 型 IV
df <- df %>%
  mutate(
    # 处理前特征（固定，外生）
    baseline_char = first_period_char,   # 如：基期城市化率、教育水平
    # DID 型 IV：政策后 × 基期特征
    iv_did        = post * baseline_char
  )

# Step 2: 主估计（DID 变量作 IV）
iv_did_main <- feols(
  outcome ~ exog_ctrl | unit_fe + year_fe | 
    endogenous_policy_intensity ~ iv_did,
  data    = df,
  cluster = ~unit_id
)
summary(iv_did_main)

# Step 3: 排他性约束检验（plausexog）
# 排他性：Z = Post × X_0 只通过处理变量 D 影响 Y
# 潜在违反：X_0 本身可能直接影响政策后的结果（趋势异质性）
# → 如果 X_0 与处理前趋势相关，DID IV 可能违反排他性

# 检验：处理前期 Y 的趋势是否与 X_0 相关
pre_trend_check <- feols(
  outcome ~ baseline_char:year | unit_fe + year_fe,
  data    = df %>% filter(post == 0),
  cluster = ~unit_id
)
summary(pre_trend_check)
# 期望：baseline_char × year 交互系数不显著（无差异趋势）

# plausexog 排他性约束稳健性
# δ：允许 iv_did 对 Y 的微小直接效应
uci_result <- uci(
  y     = df$outcome,
  x     = df$endogenous_policy_intensity,
  z     = df$iv_did,
  w     = as.matrix(df[, c("exog_ctrl", "unit_fe_dummy", "year_dummy")]),
  delta = 0.05 * abs(coef(iv_did_main)["fit_endogenous_policy_intensity"])
)
print(uci_result)

# Step 4: 平行趋势验证（DID IV 的前提）
df_pre <- df %>%
  filter(post == 0) %>%
  mutate(year_rel = year - treatment_year)

pre_trend_iv <- feols(
  outcome ~ i(year_rel, baseline_char) | unit_fe,
  data    = df_pre,
  cluster = ~unit_id
)
iplot(pre_trend_iv, main = "Pre-trend: DID IV 排他性条件检验")
```

**DID IV 排他性的关键论证：**
1. 基期特征 X_0 不预测处理前趋势（平行趋势检验通过）
2. 政策冲击（Post）是外生的（如全国统一政策推行时间）
3. X_0 影响政策强度但不直接影响政策后结果（需经济逻辑支撑）

---

## 前沿标注

### 部分单调性（Partial Monotonicity）

标准 LATE 假设所有个体满足单调性（D_i(1) ≥ D_i(0)）。当工具变量或处理为多值时，需要放松为**部分单调性**：

- **Angrist-Imbens (1995)**：多值处理（D ∈ {0, 1, 2, ...}），每个值对之间的局部效应
- **Heckman-Pinto (2018)**：放松单调性的 LATE 界限
- **王也等 (2025)**：中国经济政策中的部分单调性实践与检验方法

```r
# R: 多值处理的 LATE（ivreg + 分段处理）
# 将多值处理拆分为一系列二元比较
library(ivreg)

# 示例：D ∈ {0, 1, 2}，IV = Z（二元）
# LATE(0→1) = 2SLS 限于 D ∈ {0,1} 样本
df_01 <- df %>% filter(treatment %in% c(0, 1))
late_01 <- ivreg(outcome ~ treatment | instrument, data = df_01)

# LATE(1→2) = 2SLS 限于 D ∈ {1,2} 样本
df_12 <- df %>% filter(treatment %in% c(1, 2))
late_12 <- ivreg(outcome ~ treatment | instrument, data = df_12)

cat("LATE(0→1):", coef(late_01)["treatment"], "\n")
cat("LATE(1→2):", coef(late_12)["treatment"], "\n")
```

### 非二元内生变量 LATE 拓展

当内生变量为连续时，LATE 的"Complier"定义变为对工具变量反应最强的个体群体：

$$\text{LATE}^{cont} = \frac{E[dY | dZ]}{E[dX | dZ]} = \beta_{2SLS}$$

此时 β_2SLS 是异质效应 dY/dX 按 |dX/dZ| 加权的加权平均，权重大的是"更服从工具"的个体。

**实践建议（王也等 2025 框架）：**
1. 估计 E[X | Z = z] 的导数（第一阶段斜率），识别哪类个体 X 对 Z 更敏感
2. 分析高反应性 vs 低反应性子组的异质处理效应
3. 在论文中明确说明"连续内生变量 LATE 的加权性质"

```r
# R: 连续内生变量的 Complier 分析（近似）
# 将 Z 分组，分析不同 Z 值下 X 的反应性
df <- df %>%
  mutate(z_quartile = ntile(instrument_z, 4))

responsiveness <- df %>%
  group_by(z_quartile) %>%
  summarise(
    mean_z = mean(instrument_z),
    mean_x = mean(endogenous_x),
    mean_y = mean(outcome),
    n      = n()
  )

# 绘制 Z → X 的响应曲线（斜率代表"服从度"）
library(ggplot2)
ggplot(responsiveness, aes(x = mean_z, y = mean_x)) +
  geom_point(aes(size = n)) +
  geom_smooth(method = "lm", se = TRUE) +
  labs(title = "工具变量 Z → 内生变量 X 的响应曲线（非线性可能说明异质 Complier）",
       x = "工具变量 Z（分位数均值）",
       y = "内生变量 X（分位数均值）") +
  theme_minimal()
```

### 推荐阅读

| 主题 | 文献 | 核心贡献 |
|------|------|---------|
| LATE 基础 | Imbens & Angrist (1994) *Econometrica* | LATE 定义与四类个体 |
| 弱工具 | Staiger & Stock (1997) *Econometrica* | F > 10 规则 |
| 弱工具（新） | Lee, McCrary, Moreira, Porter (2022) *AER* | F > 104.7 tF 程序 |
| MTE 框架 | Heckman & Vytlacil (2005) *Econometrica* | MTE 与 IV 统一框架 |
| MTE 软件 | Mogstad, Santos, Torgovitsky (2018) *Econometrica* | ivmte 包理论基础 |
| GPSS Bartik | Goldsmith-Pinkham, Sorkin, Swift (2020) *AER* | 份额外生性 |
| BHJ Bartik | Borusyak, Hull, Jaravel (2022) *RES* | 变动外生性 |
| 排他性约束 | Conley, Hansen, Rossi (2012) *RES* | UCI 方法 |
| Complier 描述 | Abadie (2003) *RES* | κ-weighting |
| 部分单调性 | 王也等 (2025) | 中国政策评估应用 |
