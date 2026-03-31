# DID 分析 — 双重差分法 (Difference-in-Differences)

## 概述

双重差分法通过比较"处理组前后变化"与"对照组前后变化"来识别因果效应，是政策评估中最常用的准自然实验方法。本 skill 涵盖经典 TWFE、交错 DID（staggered）、平行趋势检验、安慰剂检验及 PSM-DID 等全套流程。

**适用场景：**
- 政策/事件具有明确的处理时间点
- 可以找到合理的对照组
- 数据为面板结构（同一单位跨多期观测）
- 处理是二元的（treated/untreated），或者处理时间因单位而异（交错）

---

## 前置条件

### 数据结构要求

```
面板数据：长格式（long format）
必须包含：
  - 个体 ID 列（unit_id）：企业/地区/个人编码
  - 时间列（time）：年份/季度/月份
  - 结果变量（outcome）：被解释变量 Y
  - 处理变量（treated）：0/1 或处理年份
  - 协变量（controls）：可选

如果是交错 DID，还需要：
  - 首次处理时间列（first_treat / cohort）：未处理单位填 0 或 Inf
```

### 前置检查

```python
# Python: 检查面板平衡性
df.groupby('unit_id')['time'].count().describe()
# 检查处理变量分布
df.groupby(['treated', 'time']).size()
# 缺失值检查
df[['outcome', 'treated', 'time', 'unit_id']].isnull().sum()
```

---

## 分析步骤

### Step 1：描述性统计与初步可视化

```python
# Python: 处理组 vs 对照组的结果变量趋势图
import matplotlib.pyplot as plt
import pandas as pd

mean_by_group = df.groupby(['time', 'treated'])['outcome'].mean().reset_index()
fig, ax = plt.subplots()
for grp, data in mean_by_group.groupby('treated'):
    label = 'Treated' if grp == 1 else 'Control'
    ax.plot(data['time'], data['outcome'], marker='o', label=label)
ax.axvline(x=treatment_year, color='red', linestyle='--', label='Treatment')
ax.legend()
ax.set_title('Parallel Trends: Pre-treatment Visual Check')
plt.tight_layout()
plt.savefig('output/pre_trend_check.png', dpi=150)
```

```r
# R: 趋势图
library(ggplot2)
df_mean <- df %>%
  group_by(time, treated) %>%
  summarise(mean_y = mean(outcome, na.rm = TRUE))

ggplot(df_mean, aes(x = time, y = mean_y, color = factor(treated))) +
  geom_line() + geom_point() +
  geom_vline(xintercept = treatment_year, linetype = "dashed", color = "red") +
  labs(title = "Pre-treatment Trend Check", color = "Group") +
  theme_minimal()
```

---

### Step 2：经典 TWFE DID（适用于单一处理时间点）

**警告：** 如果处理时间因单位而异（staggered adoption），不能直接使用 TWFE，需跳至 Step 4。

#### 基础模型

$$Y_{it} = \alpha + \beta \cdot \text{Post}_t \times \text{Treated}_i + \mu_i + \lambda_t + \varepsilon_{it}$$

其中 $\beta$ 为 ATT（Average Treatment Effect on the Treated）。

```python
# Python: linearmodels TWFE
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

df_panel = df.set_index(['unit_id', 'time'])
df_panel['did'] = df_panel['post'] * df_panel['treated']

# TWFE: entity FE + time FE
mod = PanelOLS(
    dependent=df_panel['outcome'],
    exog=sm.add_constant(df_panel[['did'] + control_vars]),
    entity_effects=True,
    time_effects=True
)
res = mod.fit(cov_type='clustered', cluster_entity=True)
print(res.summary)
```

```r
# R: fixest（推荐，最快）
library(fixest)

res_twfe <- feols(
  outcome ~ i(post, treated, ref = 0) | unit_id + time,
  data = df,
  cluster = ~unit_id  # 聚类标准误
)
etable(res_twfe)
```

```stata
* Stata: reghdfe
ssc install reghdfe

reghdfe outcome did i.time, absorb(unit_id time) cluster(unit_id)
```

---

### Step 3：事件研究图（平行趋势检验）

事件研究图是验证平行趋势假设的标准方法。在处理前各期系数应不显著（联合检验 F 也不显著）。

```python
# Python: 手动构造事件研究变量
df['event_time'] = df['time'] - df['first_treat']
# 对从未处理单位设为 -999 或删除（取决于estimand）
df = df[df['event_time'].between(-5, 5) | df['treated'] == 0]

# 创建哑变量（omit period = -1 as reference）
event_dummies = pd.get_dummies(df['event_time'], prefix='evt')
event_dummies = event_dummies.drop(columns=['evt_-1'], errors='ignore')
df = pd.concat([df, event_dummies], axis=1)
```

```r
# R: fixest 事件研究图（最简洁）
library(fixest)

res_es <- feols(
  outcome ~ i(event_time, treated, ref = -1) | unit_id + time,
  data = df,
  cluster = ~unit_id
)

# 绘制系数图
iplot(res_es,
      xlab = "Periods relative to treatment",
      main = "Event Study: Parallel Trends Test")
```

```stata
* Stata: 事件研究
gen event_time = time - first_treat
forvalues t = -5(1)5 {
    if `t' != -1 {
        gen evt_`t' = (event_time == `t') * treated
    }
}
reghdfe outcome evt_*, absorb(unit_id time) cluster(unit_id)

* 使用 coefplot 绘图
coefplot, keep(evt_*) vertical yline(0) xline(5.5, lpattern(dash))
```

**判断标准：**
- Pre-period 系数（t < 0）不显著（|t| < 1.96）
- Pre-period 系数联合 F 检验 p > 0.1
- 系数在处理前期围绕 0 波动，无单调趋势

---

### Step 4：交错 DID（Staggered Adoption）

**必读：** Callaway & Sant'Anna (2021) 和 Sun & Abraham (2021) 指出，当处理时间因单位而异时，TWFE 的 β 是各期各组 ATT 的加权平均，权重可能为**负值**，导致估计量符号甚至都可能错误。

#### 4a. Callaway-Sant'Anna (2021) — ATT(g,t)

```r
# R: did 包
library(did)

# 数据要求：first_treat = 0 表示从未处理
cs_res <- att_gt(
  yname   = "outcome",
  tname   = "time",
  idname  = "unit_id",
  gname   = "first_treat",   # 首次处理年份，未处理 = 0
  xformla = ~control1 + control2,  # 协变量
  data    = df,
  est_method = "reg",         # 或 "ipw", "dr"（doubly robust 推荐）
  control_group = "nevertreated"  # 或 "notyettreated"
)

# 汇总：事件研究风格
es <- aggte(cs_res, type = "dynamic", na.rm = TRUE)
ggdid(es, title = "CS2021: Event Study")

# 汇总：总体 ATT
att_overall <- aggte(cs_res, type = "simple")
summary(att_overall)
```

#### 4b. Sun-Abraham (2021) — interaction-weighted estimator

```r
# R: fixest sunab()
library(fixest)

res_sa <- feols(
  outcome ~ sunab(first_treat, time) | unit_id + time,
  data = df,
  cluster = ~unit_id
)
iplot(res_sa, main = "Sun-Abraham (2021) Event Study")
```

```stata
* Stata: csdid (Callaway-Sant'Anna)
ssc install csdid
ssc install drdid

csdid outcome control1 control2, ivar(unit_id) time(time) gvar(first_treat) ///
    method(dripw) wboot(reps(999)) cluster(cluster_var)

* Aggregation
csdid_stats simple
csdid_stats dynamic

* Stata: eventstudyinteract (Sun-Abraham)
ssc install eventstudyinteract
eventstudyinteract outcome evt_*, cohort(first_treat) control_cohort(never_treated) ///
    absorb(unit_id time) vce(cluster unit_id)
```

---

### Step 5：PSM-DID（倾向得分匹配 + DID）

在基线期用倾向得分匹配，使处理组和对照组在观测协变量上更平衡，再做 DID。

```python
# Python: PSM 匹配
from sklearn.linear_model import LogisticRegression
import numpy as np

# 1. 用基线期数据估计倾向得分
baseline = df[df['time'] == base_year].copy()
X = baseline[control_vars].values
T = baseline['treated'].values

lr = LogisticRegression(max_iter=1000)
lr.fit(X, T)
baseline['pscore'] = lr.predict_proba(X)[:, 1]

# 2. 1:1 最近邻匹配（caliper = 0.05 * std(pscore)）
from sklearn.neighbors import NearestNeighbors

treated_ps  = baseline[baseline['treated']==1]['pscore'].values.reshape(-1,1)
control_ps  = baseline[baseline['treated']==0]['pscore'].values.reshape(-1,1)
caliper     = 0.05 * baseline['pscore'].std()

nn = NearestNeighbors(n_neighbors=1, radius=caliper)
nn.fit(control_ps)
dist, idx = nn.kneighbors(treated_ps)

valid_match = dist.flatten() < caliper
matched_control_ids = baseline[baseline['treated']==0].iloc[idx[valid_match,0]]['unit_id'].values
matched_treated_ids = baseline[baseline['treated']==1][valid_match]['unit_id'].values
matched_ids = np.concatenate([matched_treated_ids, matched_control_ids])

df_matched = df[df['unit_id'].isin(matched_ids)]
# 3. 在匹配样本上做 TWFE DID
```

```r
# R: MatchIt + DID
library(MatchIt)
library(did)

# 基线期匹配
baseline <- df %>% filter(time == base_year)
m.out <- matchit(
  treated ~ control1 + control2 + control3,
  data    = baseline,
  method  = "nearest",
  distance = "logit",
  caliper = 0.05,
  ratio   = 1
)
summary(m.out)  # 检查匹配平衡

matched_ids <- match.data(m.out)$unit_id
df_matched   <- df %>% filter(unit_id %in% matched_ids)

# 在匹配样本上做 DID
res_psm_did <- feols(outcome ~ i(post, treated, ref=0) | unit_id + time,
                     data = df_matched, cluster = ~unit_id)
```

---

### Step 6：三重差分（DDD, Difference-in-Difference-in-Differences）

DDD 利用额外维度（如行业、年龄组）进一步控制混淆，在只有处理组+对照组+两期的情况下仍可识别。

$$Y_{igt} = \alpha + \beta_1 D_g \times \text{Post}_t + \beta_2 D_g \times X_i + \beta_3 \text{Post}_t \times X_i + \delta \cdot D_g \times \text{Post}_t \times X_i + \text{FEs} + \varepsilon$$

其中 $\delta$ 为 DDD 估计量。

```r
# R: DDD
res_ddd <- feols(
  outcome ~ treated:post:subgroup + treated:post + treated:subgroup + post:subgroup |
    unit_id + time,
  data = df, cluster = ~unit_id
)
```

```stata
* Stata: DDD
gen ddd = treated * post * subgroup
reghdfe outcome ddd treated##post treated##subgroup post##subgroup, ///
    absorb(unit_id time) cluster(unit_id)
```

---

### Step 7：安慰剂检验

#### 7a. 假时间点检验（Fake Treatment Timing）

```r
# R: 在纯控制组中随机分配虚假处理时间
set.seed(42)
control_only <- df %>% filter(treated == 0)
fake_treat_units <- sample(unique(control_only$unit_id), size = n_fake)
control_only$fake_treated <- as.integer(control_only$unit_id %in% fake_treat_units)
control_only$fake_post    <- as.integer(control_only$time >= fake_treatment_year)

res_placebo <- feols(outcome ~ i(fake_post, fake_treated, ref=0) | unit_id + time,
                     data = control_only, cluster = ~unit_id)
# 期望：系数不显著
```

#### 7b. 结果变量替换检验（Fake Outcome）

用不应受政策影响的变量作为被解释变量，期望系数不显著。

```r
res_fake_y <- feols(unrelated_outcome ~ i(post, treated, ref=0) | unit_id + time,
                    data = df, cluster = ~unit_id)
```

---

## 必做检验清单

| 检验 | 方法 | 通过标准 |
|------|------|----------|
| 平行趋势（视觉） | 事件研究图 | Pre-period 系数围绕 0 |
| 平行趋势（联合检验） | F-test on pre-period coefs | p > 0.10 |
| 安慰剂：假时间点 | 对照组中随机分配 | 系数不显著 |
| 安慰剂：假结果变量 | 用不相关 Y 重跑 | 系数不显著 |
| 交错 DID 异质性 | Bacon decomposition | 负权重比例 < 10% |
| 带宽/窗口敏感性 | ±2 期样本窗口变化 | 结论稳定 |
| 协变量平衡 | 基线期 t-test / 标准化差异 | |p均值差| / std < 0.1 |

### Bacon Decomposition（交错 DID 诊断）

```r
# R: 检查 TWFE 中负权重问题
library(bacondecomp)
bacon_res <- bacon(outcome ~ did, data = df, id_var = "unit_id", time_var = "time")
print(bacon_res)
# 如果 "Later vs Earlier Treated" 权重很大或为负，须改用 CS/SA 估计量
```

---

## 标准误聚类建议

| 数据结构 | 推荐聚类方式 |
|----------|-------------|
| 处理在个体层面 | 聚类到个体 |
| 处理在省/行业层面 | 聚类到省/行业 |
| 少于 30 个聚类 | Wild Bootstrap（`boottest` in Stata，`wildboottest` in R） |
| 空间相关性 | Conley SE（`conleyreg` in R）|

```r
# R: Wild bootstrap（少聚类）
library(fwildclusterboot)
res_wb <- boottest(res_twfe, clustid = "province_id", param = "did", B = 9999)
```

---

## 常见错误提醒

> **错误 1：交错 DID 直接用 TWFE**
> 当不同单位在不同时间被处理（staggered adoption），TWFE 的 β 是各组 ATT 的加权平均，权重可能为负，导致估计偏误甚至符号相反。**必须改用 Callaway-Sant'Anna 或 Sun-Abraham。**

> **错误 2：事件研究图仅"目测"平行趋势**
> 必须同时报告 Pre-period 系数的联合显著性检验（F-test）。视觉上"差不多平行"不足以支撑因果推断。

> **错误 3：处理变量随时间可逆（treatment reversal）**
> DID 假设处理是永久的。如果企业可以退出政策，需要用 stacked DID 或单独处理。

> **错误 4：忽略 Anticipation Effect**
> 如果主体在政策正式实施前就调整了行为（如预期效应），基准期（pre-period）末期系数会显著，导致 ATT 低估。通过 `control_anticipation_periods` 参数处理。

> **错误 5：聚类层级太细**
> 聚类标准误应在处理分配层级聚类（如处理在省级，则聚类到省）。聚类到个体会低估标准误。

> **错误 6：对"从未处理"和"尚未处理"的混淆**
> CS2021 的 `control_group` 参数：`"nevertreated"` 仅用从未被处理的单位作对照；`"notyettreated"` 还包括尚未被处理的单位。后者样本更大但可能引入偏误。

---

## 输出规范

### 主回归结果表

```r
# R: fixest 输出三线表
library(fixest)
etable(res_twfe, res_sa, res_cs,
       title  = "DID Estimation Results",
       headers = c("TWFE", "Sun-Abraham", "CS2021"),
       tex    = FALSE)   # tex=TRUE 生成 LaTeX
```

**必须在表格中注明：**
1. 固定效应类型（Entity FE, Time FE）
2. 标准误类型（Clustered at xxx level）
3. 观测数 N，聚类数 G
4. 控制变量 Yes/No

### 事件研究图规范

- X 轴：相对于处理期的时间（以 -1 为基准归一化）
- Y 轴：ATT 估计值及 95% 置信区间
- 垂直虚线：处理时间点
- 水平虚线：y = 0
- 图注说明：估计量来源（CS2021 / SA2021）、SE 聚类方式

### 文件命名

```
output/
  did_main_results.csv        # 主回归系数表
  did_event_study.png         # 事件研究图
  did_placebo_results.csv     # 安慰剂检验
  did_bacon_decomp.csv        # Bacon decomposition（交错DID）
  did_robustness.csv          # 稳健性检验汇总
```

---

## 高级工具一：Stacked DID

### 为什么需要 Stacked DID？

TWFE（双向固定效应）在交错处理设计中存在两类根本性问题：

1. **"禁止比较"（Forbidden Comparison）**：早处理队列（earlier-treated cohort）在后处理时期被用作晚处理队列的对照组，即便该队列已处于政策处理之中，污染干净的 2×2 对比。
2. **负权重（Negative Weights）**：某些队列-时期组合获得负的聚合权重，导致 TWFE 估计量可能不对应任何有经济意义的加权平均处理效应。

**Stacked DID 的核心思想**：为每个处理队列（cohort）单独构造干净的 2×2 DID 数据集——仅使用该队列的处理前后期以及"干净"对照组（尚未处理或从未处理）——然后将所有队列的数据纵向叠加，加入队列固定效应后联合估计。

### R 代码（fixest + 手动 stacking）

```r
# ============================================================
# Stacked DID — R（手动构造 + fixest 估计）
# 参考：Baker et al. (2022) "How Much Should We Trust Staggered DID Estimates?"
# ============================================================

library(dplyr)
library(fixest)

# 假设数据结构：
#   unit_id    : 个体 ID
#   time       : 时间（年份）
#   outcome    : 结果变量
#   first_treat: 首次处理年份（从未处理 = Inf 或 9999）
#   treated    : 当期是否受处理（0/1）

# ── Step 1：识别所有处理队列 ─────────────────────────────────
cohorts <- df %>%
  filter(!is.infinite(first_treat), !is.na(first_treat)) %>%
  distinct(first_treat) %>%
  pull(first_treat) %>%
  sort()

cat("处理队列:", cohorts, "\n")

# ── Step 2：为每个队列构造干净数据集 ────────────────────────
# 窗口：处理前 k_pre 期 到 处理后 k_post 期
k_pre  <- 3   # 处理前窗口
k_post <- 3   # 处理后窗口

stack_list <- list()

for (g in cohorts) {
  # 该队列的时间窗口
  t_min <- g - k_pre
  t_max <- g + k_post

  # 干净对照组：从未处理（Inf）OR 在此窗口内尚未处理
  # 排除在窗口内已处理的其他队列（"forbidden" comparisons）
  control_units <- df %>%
    filter(first_treat > t_max | is.infinite(first_treat)) %>%
    distinct(unit_id) %>%
    pull()

  treated_units <- df %>%
    filter(first_treat == g) %>%
    distinct(unit_id) %>%
    pull()

  sub_df <- df %>%
    filter(
      time >= t_min, time <= t_max,
      unit_id %in% c(treated_units, control_units)
    ) %>%
    mutate(
      cohort    = g,                           # 队列标签
      rel_time  = time - g,                    # 相对时间
      treat_cohort = as.integer(unit_id %in% treated_units)  # 该队列的处理指示
    )

  stack_list[[as.character(g)]] <- sub_df
}

# ── Step 3：叠加所有队列数据 ────────────────────────────────
df_stacked <- bind_rows(stack_list) %>%
  mutate(cohort = factor(cohort))

cat("Stacked 数据量:", nrow(df_stacked),
    "（原始:", nrow(df), "）\n")

# ── Step 4：估计（加入队列×时间、队列×个体固定效应）────────
# 方法1：简化版 TWFE（unit + time FE，加入 cohort 交互）
res_stacked <- feols(
  outcome ~ i(rel_time, treat_cohort, ref = -1) |
    unit_id^cohort + time^cohort,    # 队列内的个体和时间 FE
  data    = df_stacked,
  cluster = ~unit_id
)

# 事件研究图
iplot(res_stacked,
      xlab  = "Periods relative to treatment",
      main  = "Stacked DID: Event Study",
      col   = "steelblue")
abline(h = 0, lty = 2, col = "gray")
abline(v = -0.5, lty = 2, col = "red")

# 总体 ATT（仅取处理后系数的加权平均）
post_coefs <- coef(res_stacked)[grepl("rel_time::", names(coef(res_stacked)))]
post_coefs <- post_coefs[as.numeric(gsub(".*::(\\-?[0-9]+).*", "\\1",
                                         names(post_coefs))) >= 0]
cat("Stacked DID ATT（处理后系数均值）:", mean(post_coefs), "\n")
```

### Stata 代码（stackedev 或手动构造）

```stata
* ============================================================
* Stacked DID — Stata（stackedev 命令）
* ssc install stackedev
* ============================================================

* 准备数据：first_treat = 0 表示从未处理
* 生成相对时间（event time）
gen rel_time = time - first_treat
replace rel_time = -999 if first_treat == 0   * 从未处理单位

* 方法1：使用 stackedev 自动构造
stackedev outcome, time(time) unit(unit_id) ///
    treated(treated) ///
    ever_treated(ever_treated) ///       * 曾被处理的指示变量
    cohort(first_treat) ///
    cluster(unit_id) ///
    pre(3) post(3)                       * 窗口大小

* 输出事件研究图
stackedev_plot, yline(0) xline(-0.5)


* ──────────────────────────────────────────────────────────
* 方法2：手动构造（更灵活）
* ──────────────────────────────────────────────────────────

* 清空并重建
tempfile stacked_data
local g_list 2005 2008 2012    * 修改为实际队列年份

foreach g of local g_list {
    preserve
        local t_min = `g' - 3
        local t_max = `g' + 3

        * 保留该队列处理组 + 干净对照组（处理时间 > t_max）
        keep if (first_treat == `g') | ///
                (first_treat > `t_max') | ///
                (first_treat == 0)

        keep if time >= `t_min' & time <= `t_max'

        gen cohort   = `g'
        gen rel_time = time - `g'
        gen treat_cohort = (first_treat == `g')

        if "`g'" == "`= word("`g_list'", 1)'" {
            save `stacked_data', replace
        }
        else {
            append using `stacked_data'
            save `stacked_data', replace
        }
    restore
}

use `stacked_data', clear

* 估计（cohort×unit、cohort×time 联合固定效应）
reghdfe outcome i.rel_time#i.treat_cohort, ///
    absorb(cohort#unit_id cohort#time) ///
    cluster(unit_id) ///
    noconstant

coefplot, keep(*treat_cohort*) ///
    vertical yline(0) xline(3.5, lpattern(dash)) ///
    title("Stacked DID Event Study")
```

---

## 高级工具二：TWFE 诊断工具

### twowayfeweights（de Chaisemartin & D'Haultfoeuille 2020）

**用途**：量化 TWFE 估计量中各个处理效应的聚合权重，诊断负权重比例，评估 TWFE 偏误严重程度。

**核心输出**：
- 权重为正的 (unit, time) 对占比
- 权重为负的 (unit, time) 对占比
- TWFE 系数可被分解为 $\hat{\beta}_{TWFE} = \sum_{i,t} w_{it} \cdot ATT(i,t)$

**Stata 代码**：

```stata
* ============================================================
* twowayfeweights — Stata
* ssc install twowayfeweights
* ============================================================

* 基本用法（binary treatment）
twowayfeweights outcome unit_id time treated, ///
    type(feTR)          * feTR = FE with Treatment Regressor

* 输出说明：
*   - % of positive weights: 正权重 (unit,time) 对比例
*   - % of negative weights: 负权重比例（> 20% 则 TWFE 结论存疑）
*   - Minimum variance of effects compatible with TWFE sign

* 连续处理变量（如处理强度）
twowayfeweights outcome unit_id time treatment_intensity, ///
    type(feS)           * feS = FE with continuous treatment (Stayers)

* 进阶：加入控制变量
twowayfeweights outcome unit_id time treated ///
    control1 control2, ///
    type(feTR) other_treatments(treat2 treat3)
```

**R 代码**：

```r
# install.packages("TwoWayFEWeights")
library(TwoWayFEWeights)

weights <- twowayfeweights(
  df, "outcome", "unit_id", "time", "treated",
  type = "feTR"
)
summary(weights)
# 重点关注：negative_weight_share（负权重比例）
# < 5%：TWFE 结论相对可靠
# 5%–20%：谨慎，建议同时报告 CS2021 / SA2021
# > 20%：TWFE 严重误导，必须改用稳健估计量
```

**Bacon Decomposition（已在主体部分介绍）**：

```r
# 补充：Bacon 分解可视化
library(bacondecomp)
library(ggplot2)

bacon_res <- bacon(outcome ~ treated, data = df,
                   id_var = "unit_id", time_var = "time")

# 散点图：各比较类型的权重 vs 估计值
ggplot(bacon_res, aes(x = weight, y = estimate,
                       color = type, size = weight)) +
  geom_point(alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  scale_size_continuous(range = c(1, 8)) +
  labs(x = "权重", y = "2×2 DID 估计值",
       title = "Bacon Decomposition",
       subtitle = paste0("负权重组合: ",
           sum(bacon_res$weight[bacon_res$weight < 0]), "个"),
       color = "比较类型") +
  theme_minimal()
```

---

## 高级工具三：平行趋势假设检验与敏感性分析

### pretrends（Roth 2022）：功效分析

**核心问题**：事件研究图中 pre-period 系数"不显著"，可能仅仅是因为**统计功效不足**，而非平行趋势真正成立。

**pretrends 包**的功效分析回答：在给定违反程度（线性趋势大小）下，标准事件研究检验能以多大概率检测到违反？

```r
# ============================================================
# pretrends — R（功效分析）
# install.packages("pretrends")
# ============================================================

library(pretrends)
library(fixest)

# 步骤1：估计事件研究模型
res_es <- feols(
  outcome ~ i(rel_time, ref = -1) | unit_id + time,
  data    = df[df$treated == 1 | df$first_treat == 0, ],
  cluster = ~unit_id
)

# 提取 pre-period 系数和方差-协方差矩阵
coefs   <- coef(res_es)
vcov_es <- vcov(res_es)

# 只保留 pre-period 系数（rel_time < 0，排除 ref = -1）
pre_idx <- grep("rel_time::-[2-9]|rel_time::-[1-9][0-9]", names(coefs))
coefs_pre  <- coefs[pre_idx]
vcov_pre   <- vcov_es[pre_idx, pre_idx]

# 步骤2：设定违反假设（线性趋势）
# delta = 每期趋势差异的大小（以系数单位计）
# 通常用 post-period 效应的 1/3 或 1/2 作为基准
delta_hypothetical <- 0.02  # 每期 0.02 单位的线性趋势违反

pt_result <- pretrends(
  coefs  = coefs_pre,
  vcov   = vcov_pre,
  nPre   = length(coefs_pre),
  deltas = seq(-delta_hypothetical * 3, delta_hypothetical * 3,
               length.out = 100)
)

# 步骤3：可视化功效曲线
plot(pt_result,
     xlab = "假定违反幅度（每期线性趋势）",
     ylab = "检测到违反的概率（功效）",
     main = "pretrends: 平行趋势检验功效分析")
abline(h = 0.8, lty = 2, col = "blue")   # 80% 功效基准线
abline(v = 0,   lty = 2, col = "gray")

# 计算特定违反幅度下的检测概率
power_at_delta <- pt_result$power[which.min(abs(pt_result$deltas - delta_hypothetical))]
cat(sprintf("违反幅度 = %.3f 时，检测概率 = %.1f%%\n",
            delta_hypothetical, power_at_delta * 100))
# 若检测概率 < 50%，pre-period 不显著不能证明平行趋势成立
```

### honestdid（Rambachan & Roth 2023）：敏感性边界

**核心思想**：不强求平行趋势完全成立，而是允许它有有限度的违反。在给定违反幅度上界 $\bar{M}$ 下，构造部分识别区间（sensitivity CI），检验结论对假设违反的稳健性。

**两类敏感性限制**：
- `"smoothness"`：平行趋势违反随时间平滑变化（最常用）
- `"rm"`：相对趋势稳定（Relative Magnitude 限制）

```r
# ============================================================
# HonestDiD — R（敏感性边界）
# install.packages("HonestDiD")
# ============================================================

library(HonestDiD)
library(fixest)
library(ggplot2)

# 步骤1：获取事件研究结果（含完整 VCV）
res_es <- feols(
  outcome ~ i(rel_time, ref = -1) | unit_id + time,
  data    = df, cluster = ~unit_id
)

# 提取系数和方差协方差矩阵（全部相对时间）
betahat  <- coef(res_es)
sigma    <- vcov(res_es)

# 识别 pre 和 post period 索引
pre_period_idx  <- which(as.numeric(gsub("rel_time::(\\-?[0-9]+)", "\\1",
                    names(betahat)[grep("rel_time::", names(betahat))])) < 0)
post_period_idx <- which(as.numeric(gsub("rel_time::(\\-?[0-9]+)", "\\1",
                    names(betahat)[grep("rel_time::", names(betahat))])) >= 0)

es_coefs <- betahat[grep("rel_time::", names(betahat))]
es_vcov  <- sigma[grep("rel_time::", rownames(sigma)),
                  grep("rel_time::", colnames(sigma))]

# 步骤2：构造敏感性 CI（Smoothness 限制）
# Mbar：允许违反的幅度上界（以估计标准误单位表示）
Mbar_seq <- seq(0, 1, by = 0.1)  # 从 0 到 1 倍标准误

delta_sd_result <- createSensitivityResults(
  betahat   = es_coefs[post_period_idx],
  sigma     = es_vcov[post_period_idx, post_period_idx],
  numPrePeriods  = length(pre_period_idx),
  numPostPeriods = length(post_period_idx),
  Mvec      = Mbar_seq * sd(df$outcome, na.rm = TRUE),
  l_vec     = rep(1 / length(post_period_idx), length(post_period_idx)),  # 均等权重
  alpha     = 0.05,
  type      = "smoothness"  # 或 "rm"
)

# 步骤3：可视化敏感性 CI
createSensitivityPlot(
  robustResults    = delta_sd_result,
  originalResults  = OBcoeffs(res_es)  # 原始 95% CI
)

# 步骤4：报告最大可容忍违反幅度
# 找到 CI 恰好包含 0 时的 Mbar（"breakdown"点）
breakdown <- delta_sd_result$lb[min(which(delta_sd_result$lb <= 0 | delta_sd_result$ub >= 0))]
cat(sprintf("结论稳健的最大违反幅度: M = %.4f\n", breakdown))
cat("（M 越大，平行趋势假设越宽松，结论越稳健）\n")
```

**报告建议**：在论文表格中额外增加一列，报告 `M=0.5` 和 `M=1.0` 时的 sensitivity CI，向读者展示结论对平行趋势假设的容忍度。

---

## 高级工具四：专项拓展方法

### sdid — Synthetic DID（Arkhangelsky et al. 2021）

**核心思想**：结合 DID 和合成控制的双重优势——
- 像合成控制那样，用对照组的加权组合构造反事实（权重体现单位相似性）
- 像 DID 那样，同时使用时间维度权重，消除时间固定效应

**优点**：
1. 比传统 DID 更好地处理处理组和对照组基线差异
2. 比合成控制更灵活，适用于多处理组
3. 提供有效的推断（bootstrap SE）

```r
# ============================================================
# Synthetic DID — R（synthdid 包）
# install.packages("synthdid")
# ============================================================

library(synthdid)

# 数据需要转换为矩阵格式（行=单位，列=时间）
# panel.matrices() 辅助转换
panel_mats <- panel.matrices(
  df,
  unit    = "unit_id",
  time    = "time",
  outcome = "outcome",
  treatment = "treated"  # 0/1 的处理指示
)

# 估计 Synthetic DID
tau_sdid <- synthdid_estimate(
  Y = panel_mats$Y,    # 结果矩阵
  N0 = panel_mats$N0,  # 控制组单位数
  T0 = panel_mats$T0   # 处理前时期数
)

cat(sprintf("SDID ATT 估计: %.4f\n", tau_sdid))

# Bootstrap 标准误（推荐 placebo bootstrap）
se_sdid <- sqrt(vcov(tau_sdid, method = "placebo"))
cat(sprintf("SE: %.4f,  95%% CI: [%.4f, %.4f]\n",
            se_sdid,
            tau_sdid - 1.96 * se_sdid,
            tau_sdid + 1.96 * se_sdid))

# 可视化：合成控制 vs DID 路径
plot(tau_sdid)

# 与经典 SC 和 DID 对比
tau_sc  <- sc_estimate(panel_mats$Y, panel_mats$N0, panel_mats$T0)
tau_did <- did_estimate(panel_mats$Y, panel_mats$N0, panel_mats$T0)

cat(sprintf("DID  ATT: %.4f\n", tau_did))
cat(sprintf("SC   ATT: %.4f\n", tau_sc))
cat(sprintf("SDID ATT: %.4f\n", tau_sdid))
```

### fect — Fixed Effects Counterfactual Estimator（Liu et al. 2022）

**核心思想**：用矩阵补全（Matrix Completion）方法估计反事实结果，**不依赖平行趋势假设**。适合处理趋势差异明显、传统 DID 假设难以成立的情形。

```r
# ============================================================
# fect — R（fect 包）
# install.packages("fect")
# ============================================================

library(fect)

# 方法选择：
#   "ife"  : 交互固定效应（Interactive Fixed Effects，类似 Bai 2009）
#   "mc"   : 矩阵补全（Matrix Completion）
#   "fe"   : 经典双向固定效应（用于对比）

# 方法1：交互固定效应（IFE）
out_ife <- fect(
  Y = "outcome",
  D = "treated",
  X = c("control1", "control2"),   # 时变协变量
  index = c("unit_id", "time"),
  data  = df,
  method = "ife",
  r     = 2,          # 因子数量（可用 CV 选择）
  se    = TRUE,
  nboots = 200,       # Bootstrap 次数
  seed  = 42
)

print(out_ife)        # ATT 和 SE
plot(out_ife)         # 事件研究图（自动显示）

# 方法2：矩阵补全（MC）—— 不需要指定因子数量
out_mc <- fect(
  Y = "outcome", D = "treated",
  X = c("control1", "control2"),
  index = c("unit_id", "time"),
  data  = df,
  method = "mc",
  se    = TRUE,
  nboots = 200,
  seed  = 42
)
print(out_mc)

# 方法3：诊断——"提前处理"检验（测试模型对反事实的预测能力）
# fect 自带"hold-out"检验，随机遮盖部分处理前观测，测试预测误差
out_test <- fect(
  Y = "outcome", D = "treated",
  index = c("unit_id", "time"),
  data  = df,
  method  = "ife",
  r       = 2,
  placebo.period = c(-3, -1),  # 用处理前 3 期做安慰剂
  placebo.run    = TRUE
)
plot(out_test, type = "placebo")  # 安慰剂检验可视化
```

---

## 高级工具五：Balance Table（基线平衡检验表）

准自然实验的"门面表格"，用于证明处理组和对照组在基线期特征上可比。

### 核心统计量

| 统计量 | 定义 | 通过标准 |
|--------|------|----------|
| 均值差 | $\bar{X}_{treat} - \bar{X}_{control}$ | — |
| t 检验 p 值 | H0: 均值相等 | p > 0.10 |
| **标准化差异** | $d = \frac{\bar{X}_{treat} - \bar{X}_{control}}{\sqrt{(s^2_{treat} + s^2_{control})/2}}$ | **|d| < 0.1**（严格），< 0.25（宽松） |
| 方差比 | $s^2_{treat} / s^2_{control}$ | 0.5 ~ 2.0 为可接受 |

> **为什么用标准化差异而非 t 检验？**
> t 检验受样本量影响（大样本下微小差异也显著），标准化差异是纯粹的效应量，不受 n 影响，更适合判断实质平衡性。

### Python 代码

```python
# ============================================================
# Balance Table — Python（手动 + tableone 包）
# pip install tableone
# ============================================================

import pandas as pd
import numpy as np
from scipy import stats

# ── 方法1：手动构造（更灵活）────────────────────────────────
def standardized_diff(x1, x2):
    """计算标准化差异 (Standardized Difference)"""
    mean_diff = np.mean(x1) - np.mean(x2)
    pooled_sd = np.sqrt((np.var(x1, ddof=1) + np.var(x2, ddof=1)) / 2)
    return mean_diff / pooled_sd if pooled_sd > 0 else np.nan

baseline = df[df['time'] == base_year].copy()
treated  = baseline[baseline['treated'] == 1]
control  = baseline[baseline['treated'] == 0]

balance_vars = ['age', 'size', 'leverage', 'roa', 'asset']  # 修改为实际变量

rows = []
for var in balance_vars:
    x1 = treated[var].dropna()
    x2 = control[var].dropna()

    mean_t = x1.mean()
    mean_c = x2.mean()
    sd_t   = x1.std()
    sd_c   = x2.std()

    # t 检验（假设方差不等）
    t_stat, p_val = stats.ttest_ind(x1, x2, equal_var=False)
    std_diff = standardized_diff(x1, x2)

    rows.append({
        '变量': var,
        '处理组均值': f"{mean_t:.3f}",
        '处理组标准差': f"({sd_t:.3f})",
        '对照组均值': f"{mean_c:.3f}",
        '对照组标准差': f"({sd_c:.3f})",
        '均值差': f"{mean_t - mean_c:.3f}",
        't统计量': f"{t_stat:.2f}",
        'p值': f"{p_val:.3f}",
        '标准化差异': f"{std_diff:.3f}",
        '平衡': '✓' if abs(std_diff) < 0.1 else ('△' if abs(std_diff) < 0.25 else '✗')
    })

balance_df = pd.DataFrame(rows)
print("基线平衡检验表")
print("=" * 80)
print(balance_df.to_string(index=False))
print(f"\n不平衡变量数（|d|≥0.1）: {(balance_df['标准化差异'].astype(float).abs() >= 0.1).sum()}")

balance_df.to_csv("output/balance_table.csv", index=False)

# ── 方法2：tableone 包（更标准化）────────────────────────────
from tableone import TableOne

table1 = TableOne(
    baseline,
    columns    = balance_vars,
    groupby    = 'treated',
    pval       = True,
    smd        = True,        # 自动计算标准化均值差（SMD）
    overall    = False
)
print(table1.tabulate(tablefmt='fancy_grid'))
table1.to_csv("output/tableone_balance.csv")
```

### R 代码（cobalt 包）

```r
# ============================================================
# Balance Table — R（cobalt 包）
# install.packages("cobalt")
# ============================================================

library(cobalt)
library(MatchIt)
library(dplyr)

# 准备基线期数据
baseline <- df %>% filter(time == base_year)

# ── 未匹配时的平衡性 ────────────────────────────────────────
bal_unmatched <- bal.tab(
  treated ~ age + size + leverage + roa + asset,
  data    = baseline,
  thresholds = c(m = 0.1),   # 标准化差异阈值
  un      = TRUE              # 显示未加权结果
)
print(bal_unmatched)

# ── 匹配后平衡性对比 ────────────────────────────────────────
m.out <- matchit(
  treated ~ age + size + leverage + roa + asset,
  data    = baseline,
  method  = "nearest",
  distance = "logit",
  caliper = 0.05
)

bal_matched <- bal.tab(
  m.out,
  thresholds = c(m = 0.1)
)
print(bal_matched)

# ── 可视化：Love Plot（标准化差异点图）────────────────────────
love.plot(
  m.out,
  threshold   = 0.1,
  abs         = TRUE,
  colors      = c("coral", "steelblue"),
  shapes      = c("triangle", "circle"),
  var.order   = "unadjusted",
  title       = "Covariate Balance: Before and After Matching",
  sample.names = c("Unmatched", "Matched")
)

# ── 输出 LaTeX 表格 ──────────────────────────────────────────
# install.packages("flextable")  # 或用 knitr::kable
library(knitr)
bal_data <- bal_unmatched$Balance[, c("Diff.Un", "Diff.Adj", "M.Threshold")]
kable(bal_data, format = "latex", booktabs = TRUE,
      caption = "Covariate Balance Before and After Matching",
      col.names = c("Std. Diff (Before)", "Std. Diff (After)", "Threshold"),
      digits = 3)
```

### Stata 代码（iebaltab / balancetable）

```stata
* ============================================================
* Balance Table — Stata
* ssc install iebaltab
* ssc install balancetable
* ============================================================

* 筛选基线期
keep if time == `base_year'

* ── 方法1：iebaltab（推荐，World Bank DIME 团队开发）────────
iebaltab age size leverage roa asset, ///
    grpvar(treated) ///
    rowvarlabels ///             * 使用变量标签
    stdev ///                    * 显示标准差
    ttest ///                    * t 检验
    normdiff ///                 * 标准化差异（核心！）
    savecsv("output/balance_table.csv") ///
    savexlsx("output/balance_table.xlsx") ///
    savetex("output/balance_table.tex") ///
    replace


* ── 方法2：balancetable（输出格式更美观）────────────────────
balancetable treated age size leverage roa asset, ///
    varlabels ///
    format(%9.3f) ///
    onerow ///                   * 均值和 SD 合并一行
    pval ///
    normdiff ///
    saving("output/balance_table_v2.xlsx", replace)


* ── 手动计算标准化差异 ─────────────────────────────────────
foreach var of varlist age size leverage roa asset {
    qui sum `var' if treated == 1
    local mean_t = r(mean)
    local sd_t   = r(sd)

    qui sum `var' if treated == 0
    local mean_c = r(mean)
    local sd_c   = r(sd)

    local pooled_sd = sqrt((`sd_t'^2 + `sd_c'^2) / 2)
    local std_diff  = (`mean_t' - `mean_c') / `pooled_sd'

    di "`var': Std.Diff = " %6.3f `std_diff' ///
        " (|d| " cond(abs(`std_diff') < 0.1, "< 0.1 ✓", ///
                 cond(abs(`std_diff') < 0.25, "< 0.25 △", ">= 0.25 ✗")) ")"
}
```

### LaTeX 输出模板

```latex
% ============================================================
% Balance Table LaTeX 模板（三线表）
% ============================================================

\begin{table}[htbp]
\centering
\caption{Baseline Characteristics: Treated vs. Control Group}
\label{tab:balance}
\begin{threeparttable}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{\textbf{Treated}} & \multicolumn{2}{c}{\textbf{Control}} & & \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
\textbf{Variable} & Mean & SD & Mean & SD & \textbf{Std. Diff} & \textbf{p-value} \\
\midrule
Age (years)      & 8.23 & (3.41) & 7.98 & (3.52) & 0.072 & 0.312 \\
Firm Size (log)  & 10.45 & (1.23) & 10.38 & (1.31) & 0.055 & 0.421 \\
Leverage         & 0.42 & (0.18) & 0.41 & (0.17) & 0.057 & 0.398 \\
ROA              & 0.058 & (0.042) & 0.061 & (0.045) & -0.069 & 0.183 \\
Total Assets (log) & 12.31 & (1.45) & 12.28 & (1.51) & 0.020 & 0.712 \\
\midrule
\textit{N}       & \multicolumn{2}{c}{342} & \multicolumn{2}{c}{1,256} & & \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes}: Balance table reports mean and standard deviation for
      treated and control firms in the baseline year (year $t-1$).
      Std. Diff = standardized difference = $(\bar{X}_{treat} - \bar{X}_{ctrl})
      / \sqrt{(s^2_{treat} + s^2_{ctrl})/2}$. Values $|d| < 0.1$ indicate
      good balance. The $p$-value is from a two-sample $t$-test with unequal
      variances. All variables are measured at the end of the fiscal year
      prior to treatment assignment.
\end{tablenotes}
\end{threeparttable}
\end{table}
```

---

## 文献补充（高级工具）

| 文献 | 贡献 |
|------|------|
| Baker et al. (2022) | Stacked DID 系统论述 |
| de Chaisemartin & D'Haultfoeuille (2020) | TWFE 负权重诊断（twowayfeweights）|
| Roth (2022) | pre-trends 功效分析（pretrends 包）|
| Rambachan & Roth (2023) | 平行趋势敏感性边界（HonestDiD 包）|
| Arkhangelsky et al. (2021) | Synthetic DID（synthdid 包）|
| Liu, Wang & Xu (2022) | Fixed Effects Counterfactual（fect 包）|
| Sant'Anna & Zhao (2020) | Doubly Robust DID（drdid 包）|

```bibtex
@article{baker2022stacked,
  author  = {Baker, Andrew C. and Larcker, David F. and Wang, Charles C.Y.},
  title   = {How Much Should We Trust Staggered Difference-in-Differences Estimates?},
  journal = {Journal of Financial Economics},
  year    = {2022}, volume = {144}, number = {2}, pages = {370--395}
}
@article{dechaisemartin2020twfe,
  author  = {de Chaisemartin, Cl\'{e}ment and D'Haultfoeuille, Xavier},
  title   = {Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects},
  journal = {American Economic Review},
  year    = {2020}, volume = {110}, number = {9}, pages = {2964--2996}
}
@article{roth2022pretrends,
  author  = {Roth, Jonathan},
  title   = {Pre-test with Caution: Event-Study Estimates after Testing for Parallel Trends},
  journal = {American Economic Review: Insights},
  year    = {2022}, volume = {4}, number = {3}, pages = {305--322}
}
@article{rambachan2023honestdid,
  author  = {Rambachan, Ashesh and Roth, Jonathan},
  title   = {A More Credible Approach to Parallel Trends},
  journal = {The Review of Economic Studies},
  year    = {2023}, volume = {90}, number = {5}, pages = {2555--2591}
}
@article{arkhangelsky2021sdid,
  author  = {Arkhangelsky, Dmitry and Athey, Susan and Hirshberg, David A.
             and Imbens, Guido W. and Wager, Stefan},
  title   = {Synthetic Difference-in-Differences},
  journal = {American Economic Review},
  year    = {2021}, volume = {111}, number = {12}, pages = {4088--4118}
}
@article{liu2022fect,
  author  = {Liu, Licheng and Wang, Ye and Xu, Yiqing},
  title   = {A Practical Guide to Counterfactual Estimators for Causal Inference with Time-Series Cross-Sectional Data},
  journal = {American Journal of Political Science},
  year    = {2022}, volume = {68}, number = {1}, pages = {160--176}
}
```

---

## 完整安慰剂检验（空间安慰剂）

随机置换处理组身份，重复估计 DID 系数，通过经验分布判断真实效应的统计显著性。相比传统 t 检验，置换检验不依赖正态分布假设，在小样本或非对称分布下更可靠。

**步骤：**
1. 将处理组身份（treated=1）随机打乱到同等数量的单位上，重复 500 次
2. 每次重新估计 TWFE DID 系数
3. 绘制 500 个假系数的核密度分布图（真实系数用红色竖线标注）
4. 经验 p 值 = (假系数绝对值 ≥ 真实系数绝对值的次数) / 500

### Python 代码（linearmodels + matplotlib）

```python
# ============================================================
# 空间安慰剂检验 — Python（linearmodels PanelOLS 循环 500 次）
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
from tqdm import tqdm  # pip install tqdm

# ── 前置：先跑一次真实回归，获得真实系数 ─────────────────────
df_panel = df.set_index(['unit_id', 'time'])
df_panel['did'] = df_panel['post'] * df_panel['treated']

mod_real = PanelOLS(
    dependent=df_panel['outcome'],
    exog=sm.add_constant(df_panel[['did'] + control_vars]),
    entity_effects=True,
    time_effects=True
)
res_real = mod_real.fit(cov_type='clustered', cluster_entity=True)
beta_real = res_real.params['did']
print(f"真实 DID 系数: {beta_real:.4f}")

# ── 置换检验主循环 ────────────────────────────────────────────
N_PERM = 500
np.random.seed(42)

# 取基线期单位列表（每次打乱处理组身份）
units = df['unit_id'].unique()
n_treated = df.groupby('unit_id')['treated'].max().sum()  # 处理组单位数

fake_betas = []

for _ in tqdm(range(N_PERM), desc="空间安慰剂置换"):
    # 随机选取同等数量的"假处理组"单位
    fake_treated_units = np.random.choice(units, size=int(n_treated), replace=False)

    df_perm = df.copy()
    df_perm['fake_treated'] = df_perm['unit_id'].isin(fake_treated_units).astype(int)
    df_perm['fake_did'] = df_perm['post'] * df_perm['fake_treated']

    df_p = df_perm.set_index(['unit_id', 'time'])
    try:
        mod_p = PanelOLS(
            dependent=df_p['outcome'],
            exog=sm.add_constant(df_p[['fake_did'] + control_vars]),
            entity_effects=True,
            time_effects=True
        )
        res_p = mod_p.fit(cov_type='clustered', cluster_entity=True)
        fake_betas.append(res_p.params['fake_did'])
    except Exception:
        continue  # 跳过奇异矩阵等异常

fake_betas = np.array(fake_betas)

# ── 计算经验 p 值 ─────────────────────────────────────────────
emp_pval = np.mean(np.abs(fake_betas) >= np.abs(beta_real))
print(f"经验 p 值: {emp_pval:.4f}  （有效置换次数: {len(fake_betas)}）")

# ── 核密度分布图 ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

# 核密度曲线
kde = gaussian_kde(fake_betas, bw_method='scott')
x_grid = np.linspace(fake_betas.min() - 0.1, fake_betas.max() + 0.1, 500)
ax.plot(x_grid, kde(x_grid), color='steelblue', lw=2, label='假系数核密度 (N=500)')
ax.fill_between(x_grid, kde(x_grid), alpha=0.2, color='steelblue')

# 真实系数标注（红色竖线）
ax.axvline(x=beta_real, color='red', lw=2, linestyle='-',
           label=f'真实系数 = {beta_real:.4f}')
ax.axvline(x=-beta_real, color='red', lw=1.5, linestyle='--',
           label=f'对称位置 = {-beta_real:.4f}', alpha=0.6)

# 着色：|假系数| ≥ |真实系数| 的区域
mask_pos = x_grid >= np.abs(beta_real)
mask_neg = x_grid <= -np.abs(beta_real)
for mask in [mask_pos, mask_neg]:
    ax.fill_between(x_grid[mask], kde(x_grid[mask]), alpha=0.4,
                    color='red', label='_nolegend_')

ax.set_xlabel('DID 系数', fontsize=12)
ax.set_ylabel('核密度', fontsize=12)
ax.set_title(f'空间安慰剂检验（置换次数=500）\n经验 p 值 = {emp_pval:.4f}',
             fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('output/did_spatial_placebo.png', dpi=150)
plt.show()
print("图片已保存到 output/did_spatial_placebo.png")
```

### R 代码（fixest 循环 + ggplot2 密度图）

```r
# ============================================================
# 空间安慰剂检验 — R（fixest feols 循环 + ggplot2）
# ============================================================
library(fixest)
library(ggplot2)
library(dplyr)

set.seed(42)
N_PERM <- 500

# ── 真实系数 ──────────────────────────────────────────────────
res_real <- feols(
  outcome ~ i(post, treated, ref = 0) | unit_id + time,
  data = df, cluster = ~unit_id
)
beta_real <- coef(res_real)["post::1:treated"]
cat(sprintf("真实 DID 系数: %.4f\n", beta_real))

# ── 置换主循环 ────────────────────────────────────────────────
units     <- unique(df$unit_id)
n_treated <- df %>% group_by(unit_id) %>% summarise(t = max(treated)) %>%
             pull(t) %>% sum()

fake_betas <- numeric(N_PERM)
for (i in seq_len(N_PERM)) {
  fake_units <- sample(units, size = n_treated, replace = FALSE)

  df_perm <- df %>%
    mutate(
      fake_treated = as.integer(unit_id %in% fake_units),
      fake_did     = post * fake_treated
    )

  res_p <- tryCatch(
    feols(outcome ~ fake_did | unit_id + time,
          data = df_perm, cluster = ~unit_id),
    error = function(e) NULL
  )

  fake_betas[i] <- if (!is.null(res_p)) coef(res_p)["fake_did"] else NA_real_
}

fake_betas <- na.omit(fake_betas)

# ── 经验 p 值 ─────────────────────────────────────────────────
emp_pval <- mean(abs(fake_betas) >= abs(beta_real))
cat(sprintf("经验 p 值: %.4f  （有效置换次数: %d）\n",
            emp_pval, length(fake_betas)))

# ── ggplot2 核密度图 ──────────────────────────────────────────
df_fake <- data.frame(beta = fake_betas)

p_placebo <- ggplot(df_fake, aes(x = beta)) +
  geom_density(fill = "steelblue", alpha = 0.25, color = "steelblue", linewidth = 1) +
  # 真实系数竖线
  geom_vline(xintercept = beta_real, color = "red", linewidth = 1.2,
             linetype = "solid") +
  geom_vline(xintercept = -beta_real, color = "red", linewidth = 0.9,
             linetype = "dashed", alpha = 0.7) +
  annotate("text", x = beta_real, y = Inf,
           label = sprintf("真实系数\n%.4f", beta_real),
           hjust = -0.1, vjust = 1.5, color = "red", size = 3.5) +
  # 着色尾部区域
  stat_function(
    fun = function(x) {
      d <- density(fake_betas)
      approx(d$x, d$y, xout = x)$y
    },
    xlim = c(abs(beta_real), max(fake_betas)),
    geom = "area", fill = "red", alpha = 0.35
  ) +
  stat_function(
    fun = function(x) {
      d <- density(fake_betas)
      approx(d$x, d$y, xout = x)$y
    },
    xlim = c(min(fake_betas), -abs(beta_real)),
    geom = "area", fill = "red", alpha = 0.35
  ) +
  labs(
    title    = sprintf("空间安慰剂检验（置换次数 = %d）", length(fake_betas)),
    subtitle = sprintf("经验 p 值 = %.4f", emp_pval),
    x        = "假 DID 系数",
    y        = "核密度"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("output/did_spatial_placebo.png", p_placebo, width = 8, height = 5, dpi = 150)
print(p_placebo)
```

---

## 交叠DID专项安慰剂

在 Staggered DID（交叠 DID）框架下，安慰剂检验需要打乱 **首次处理年份（first_treat / gname）**，而不仅仅是处理组身份。

**逻辑：** Callaway-Sant'Anna（CS）框架的识别依赖于"处理时机是外生的"这一假设。通过随机 shuffle gname，验证真实处理时机下得到的效应是否显著超出随机情形下的效应分布。

### R 代码（did 包 + 随机 shuffle gname）

```r
# ============================================================
# 交叠 DID 安慰剂检验 — R（CS 框架打乱 first_treat 年份）
# ============================================================
library(did)
library(dplyr)
library(ggplot2)

set.seed(42)
N_PERM <- 500

# ── 真实 CS 估计 ──────────────────────────────────────────────
# 注意：gname = 0 表示从未处理
cs_real <- att_gt(
  yname   = "outcome",
  tname   = "time",
  idname  = "unit_id",
  gname   = "first_treat",
  data    = df,
  est_method     = "dr",
  control_group  = "nevertreated",
  bstrap = FALSE  # 置换循环内关闭 bootstrap 加速
)
att_real <- aggte(cs_real, type = "simple")$overall.att
cat(sprintf("真实 CS ATT: %.4f\n", att_real))

# ── 获取有处理的单位（first_treat > 0）─────────────────────
treated_units <- df %>%
  filter(first_treat > 0) %>%
  select(unit_id, first_treat) %>%
  distinct()

fake_atts <- numeric(N_PERM)
for (i in seq_len(N_PERM)) {
  # 随机打乱首次处理年份（只在处理单位之间重新分配）
  shuffled_treat <- treated_units %>%
    mutate(fake_first_treat = sample(first_treat))

  df_perm <- df %>%
    left_join(shuffled_treat %>% select(unit_id, fake_first_treat),
              by = "unit_id") %>%
    mutate(
      fake_first_treat = ifelse(is.na(fake_first_treat), 0, fake_first_treat)
    )

  cs_p <- tryCatch({
    att_gt(
      yname   = "outcome",
      tname   = "time",
      idname  = "unit_id",
      gname   = "fake_first_treat",
      data    = df_perm,
      est_method    = "dr",
      control_group = "nevertreated",
      bstrap = FALSE
    )
  }, error = function(e) NULL)

  if (!is.null(cs_p)) {
    agg_p <- tryCatch(aggte(cs_p, type = "simple"), error = function(e) NULL)
    fake_atts[i] <- if (!is.null(agg_p)) agg_p$overall.att else NA_real_
  } else {
    fake_atts[i] <- NA_real_
  }
}

fake_atts <- na.omit(fake_atts)
emp_pval  <- mean(abs(fake_atts) >= abs(att_real))
cat(sprintf("Staggered DID 安慰剂经验 p 值: %.4f\n", emp_pval))

# ── 可视化 ────────────────────────────────────────────────────
df_fake2 <- data.frame(att = fake_atts)

ggplot(df_fake2, aes(x = att)) +
  geom_density(fill = "steelblue", alpha = 0.25, color = "steelblue") +
  geom_vline(xintercept = att_real, color = "red", linewidth = 1.2) +
  annotate("text", x = att_real, y = Inf,
           label = sprintf("真实 ATT\n%.4f", att_real),
           hjust = -0.15, vjust = 2, color = "red") +
  labs(
    title    = "Staggered DID 安慰剂检验（打乱 first_treat 年份）",
    subtitle = sprintf("经验 p 值 = %.4f，有效置换 %d 次", emp_pval, length(fake_atts)),
    x = "假 ATT", y = "核密度"
  ) +
  theme_minimal()
```

---

## TWFE自动Bacon分解（必做，非可选）

**背景：** Goodman-Bacon (2021) 证明，TWFE 的 $\hat\beta$ 等于所有可能的 2×2 DID 子组比较的加权平均，权重由各子组样本量决定，并可能为负。负权重出现时，$\hat\beta$ 不再对应任何有意义的 ATT。

**决策树：**

```
TWFE 负权重比例？
├── 负权重 = 0
│   └── TWFE 可信，可作主回归
├── 负权重 > 0 但 < 10%
│   └── 标注风险，TWFE 作主回归 + CS/SA 作稳健性并排报告
├── 负权重 > 10%
│   └── 强烈建议切换 CS/SA 作主回归（TWFE 降级为参考列）
│       例外：如有强理论理由相信效应时间不变（homogeneous effects）
│             → TWFE 仍可用，但必须在脚注声明该假设
└── 负权重比例 = N/A（单一处理时间点）
    └── 无 Bacon 分解问题，TWFE 直接可信
```

### R 代码（bacondecomp + 可视化）

```r
# ============================================================
# TWFE 自动 Bacon 分解 — R
# install.packages("bacondecomp")
# ============================================================
library(bacondecomp)
library(ggplot2)
library(dplyr)

# ── Step 1：运行 Bacon 分解 ──────────────────────────────────
# 注意：outcome 必须是连续变量；treated 是 0/1 的 binary treatment
bacon_res <- bacon(
  outcome ~ treated,
  data     = df,
  id_var   = "unit_id",
  time_var = "time"
)

print(bacon_res)
# 输出列说明：
#   type        : 比较类型（"Earlier vs Later Treated", "Later vs Earlier",
#                           "Treated vs Untreated"）
#   weight      : 该子组比较在 TWFE β 中的权重
#   estimate    : 该子组 2×2 DID 估计值

# ── Step 2：汇总负权重信息 ──────────────────────────────────
total_neg_weight <- sum(bacon_res$weight[bacon_res$weight < 0])
neg_weight_share <- sum(bacon_res$weight[bacon_res$weight < 0]) /
                    sum(abs(bacon_res$weight))

cat(sprintf("=== Bacon 分解诊断 ===\n"))
cat(sprintf("TWFE 估计量中负权重比例: %.2f%%\n", neg_weight_share * 100))
cat(sprintf("负权重总量: %.4f\n", total_neg_weight))

# ── Step 3：自动决策 ─────────────────────────────────────────
if (neg_weight_share == 0) {
  cat("决策：负权重 = 0，TWFE 可信，无需切换估计量。\n")
} else if (neg_weight_share < 0.10) {
  cat("决策：负权重 > 0 但 < 10%，标注风险。\n",
      "建议：TWFE 作主回归，同时跑 CS/SA 作稳健性检验。\n")
} else {
  cat("决策：负权重 > 10%，TWFE 存在严重偏误风险。\n",
      "强烈建议：改用 CS2021 或 SA2021 作主回归！\n")
}

# ── Step 4：Bacon 分解可视化 ─────────────────────────────────
# 散点图：权重（x）vs 2×2 DID 估计值（y），点大小 = 权重绝对值
p_bacon <- ggplot(bacon_res, aes(x = weight, y = estimate,
                                  color = type, size = abs(weight))) +
  geom_point(alpha = 0.75) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red",
             alpha = 0.6) +
  scale_size_continuous(range = c(1, 10)) +
  scale_color_brewer(palette = "Set1") +
  annotate("text", x = Inf, y = Inf,
           label = sprintf("负权重比例: %.1f%%", neg_weight_share * 100),
           hjust = 1.1, vjust = 1.5, size = 4, color = "red") +
  labs(
    x        = "权重（Weight in TWFE）",
    y        = "2×2 DID 估计值",
    color    = "比较类型",
    size     = "权重绝对值",
    title    = "Bacon Decomposition：TWFE 权重分布诊断",
    subtitle = sprintf("负权重比例 = %.2f%%  |  TWFE β̂ 加权和 = %.4f",
                       neg_weight_share * 100,
                       weighted.mean(bacon_res$estimate, bacon_res$weight))
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

ggsave("output/did_bacon_decomp.png", p_bacon, width = 9, height = 6, dpi = 150)
print(p_bacon)
```

---

## Borusyak et al. (2024) 插补估计量

Borusyak, Jaravel & Spiess (2024) 提出的**插补估计量（Imputation Estimator）**，通过先用从未处理/尚未处理单位估计反事实，再用实际结果减去预测反事实得到处理效应。属于"BJS 估计量"或"did2s 方法"。

**优点：**
- 计算效率高（先跑一次固定效应，再插补）
- 自动处理负权重问题
- 事件研究图天然满足 TWFE 一致性

### R 代码（did2s 包完整流程）

```r
# ============================================================
# Borusyak et al. (2024) 插补估计量 — R（did2s 包）
# install.packages("did2s")
# ============================================================
library(did2s)
library(ggplot2)
library(fixest)

# ── 数据要求 ──────────────────────────────────────────────────
# first_treat: 首次处理年份；从未处理单位 = 0（或 NA）
# treated    : 当期是否处于处理状态（0/1）

# ── Step 1：第一阶段——估计反事实（只用未处理观测）────────────
# first_stage: 仅使用 treated == 0 的观测估计个体 + 时间固定效应
# second_stage: 用估计的 FE 插补反事实，再对残差做 WLS

res_bjs <- did2s(
  data          = df,
  yname         = "outcome",
  first_stage   = ~ 0 | unit_id + time,      # 第一阶段固定效应
  second_stage  = ~ i(rel_time, ref = -1),    # 第二阶段：事件研究
  treatment     = "treated",
  cluster_var   = "unit_id"                  # 聚类标准误
)

summary(res_bjs)

# ── Step 2：提取总体 ATT（处理后期系数均值）──────────────────
# 若只需单一 ATT 而非完整事件研究，second_stage 改为 ~treated
res_bjs_att <- did2s(
  data         = df,
  yname        = "outcome",
  first_stage  = ~ 0 | unit_id + time,
  second_stage = ~ treated,
  treatment    = "treated",
  cluster_var  = "unit_id"
)
cat(sprintf("BJS ATT: %.4f  SE: %.4f\n",
            coef(res_bjs_att)["treated"],
            sqrt(vcov(res_bjs_att)["treated", "treated"])))

# ── Step 3：事件研究图 ────────────────────────────────────────
# did2s 返回 fixest 对象，直接用 iplot
iplot(res_bjs,
      main = "Borusyak et al. (2024) 插补估计量：事件研究图",
      xlab = "相对处理期",
      col  = "darkgreen")

# ── Step 4：与其他估计量对比（并排系数图，见"多估计量对比"节）
```

---

## did_multiplegt（de Chaisemartin & D'Haultfoeuille）

**适用场景：** 连续处理变量（treatment intensity）、处理可逆（switchers-in and switchers-out）、或需要分解处理强度变化的 DID。

### R 代码（DIDmultiplegt 包）

```r
# ============================================================
# did_multiplegt — R（DIDmultiplegt 包）
# install.packages("DIDmultiplegt")
# ============================================================
library(DIDmultiplegt)

# ── 基础用法（二元处理）──────────────────────────────────────
# 参数说明：
#   df         : 数据框
#   "outcome"  : 结果变量名
#   "unit_id"  : 个体 ID 名
#   "time"     : 时间变量名
#   "treated"  : 处理变量（0/1 或连续）
#   controls   : 控制变量向量（字符串）
#   placebo    : 安慰剂期数（处理前检验）
#   dynamic    : 动态效应期数（处理后）
#   breps      : Bootstrap 次数

res_dcdh <- did_multiplegt(
  df         = df,
  Y          = "outcome",
  G          = "unit_id",
  T          = "time",
  D          = "treated",
  controls   = c("control1", "control2"),  # 可选
  placebo    = 3,      # 检验处理前 3 期
  dynamic    = 4,      # 估计处理后 4 期动态效应
  breps      = 200,    # Bootstrap 置信区间
  cluster    = "unit_id",
  average_effect = "simple"  # 或 "dynamic"
)

# 打印结果
print(res_dcdh)

# ── 连续处理（Intensity DID）────────────────────────────────
# 当处理变量是连续的（如补贴金额、污染指数），使用 did_multiplegt_dyn
# install.packages("DIDmultiplegt")
res_dcdh_cont <- did_multiplegt_dyn(
  df       = df,
  outcome  = "outcome",
  group    = "unit_id",
  time     = "time",
  treatment = "treatment_intensity",   # 连续处理变量
  effects   = 4,    # 估计 4 期动态效应
  placebo   = 3,    # 3 期安慰剂
  cluster   = "unit_id"
)

summary(res_dcdh_cont)
plot(res_dcdh_cont)
```

---

## 多估计量对比可视化

在同一张图上并排展示 TWFE、CS、SA、Stacked、BJS、dCDH 等多个估计量的系数和置信区间，直观比较各方法的一致性。

### R ggplot2 代码模板

```r
# ============================================================
# 多估计量对比图 — R（ggplot2 并排系数图）
# 假设已经跑完：res_twfe, res_sa, res_cs, res_stacked, res_bjs, res_dcdh
# ============================================================
library(ggplot2)
library(dplyr)

# ── 辅助函数：提取估计量摘要 ─────────────────────────────────
extract_att <- function(method_name, coef_val, se_val) {
  data.frame(
    method  = method_name,
    coef    = coef_val,
    se      = se_val,
    ci_lo   = coef_val - 1.96 * se_val,
    ci_hi   = coef_val + 1.96 * se_val
  )
}

# ── 从各模型提取总体 ATT ──────────────────────────────────────
# TWFE
coef_twfe <- coef(res_twfe)["post::1:treated"]
se_twfe   <- sqrt(vcov(res_twfe)["post::1:treated", "post::1:treated"])

# Sun-Abraham（从 fixest 聚合）
sa_agg  <- aggregate(res_sa, agg = "ATT")
coef_sa <- coef(sa_agg)["ATT"]
se_sa   <- sqrt(diag(vcov(sa_agg))["ATT"])

# Callaway-Sant'Anna
cs_agg  <- aggte(cs_real, type = "simple")
coef_cs <- cs_agg$overall.att
se_cs   <- cs_agg$overall.se

# Stacked DID（用处理后系数均值）
post_coefs_stacked <- coef(res_stacked)[grepl("rel_time::[^-]", names(coef(res_stacked)))]
coef_stacked <- mean(post_coefs_stacked)
se_stacked   <- mean(sqrt(diag(vcov(res_stacked))[names(post_coefs_stacked)]))

# BJS（did2s）
coef_bjs <- coef(res_bjs_att)["treated"]
se_bjs   <- sqrt(vcov(res_bjs_att)["treated","treated"])

# dCDH（did_multiplegt，从 res_dcdh 提取 effect_average）
coef_dcdh <- res_dcdh$effect_average
se_dcdh   <- res_dcdh$se_effect_average

# ── 合并数据 ──────────────────────────────────────────────────
df_compare <- bind_rows(
  extract_att("TWFE",          coef_twfe,    se_twfe),
  extract_att("Sun-Abraham",   coef_sa,      se_sa),
  extract_att("CS2021",        coef_cs,      se_cs),
  extract_att("Stacked DID",   coef_stacked, se_stacked),
  extract_att("BJS (did2s)",   coef_bjs,     se_bjs),
  extract_att("dCDH",          coef_dcdh,    se_dcdh)
) %>%
  mutate(method = factor(method, levels = c("TWFE","Sun-Abraham","CS2021",
                                             "Stacked DID","BJS (did2s)","dCDH")))

# ── ggplot2 并排系数图 ────────────────────────────────────────
p_compare <- ggplot(df_compare, aes(x = method, y = coef, color = method)) +
  geom_point(size = 3.5) +
  geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.25, linewidth = 1) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  scale_color_brewer(palette = "Dark2") +
  labs(
    title    = "多估计量对比：各方法 ATT 及 95% 置信区间",
    subtitle = "误差棒 = 95% CI（基于聚类标准误）",
    x        = "估计方法",
    y        = "ATT 估计值",
    caption  = "TWFE 若存在负权重，以 CS/SA/BJS 为主回归"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "none",
    axis.text.x     = element_text(angle = 20, hjust = 1)
  )

ggsave("output/did_estimator_comparison.png", p_compare,
       width = 9, height = 5, dpi = 150)
print(p_compare)
```

---

## 组间系数差异检验

检验处理组和对照组的 DID 系数是否显著不同，或比较两个子组（如高/低杠杆企业）的异质性处理效应。

### Fisher Permutation Test（随机交换组标签 1000 次）

**逻辑：** 如果两组系数之差是偶然的，随机交换组标签后得到的差异分布应覆盖观测到的真实差异。若真实差异在分布尾部（经验 p < 0.05），则认为两组系数显著不同。

### R 代码

```r
# ============================================================
# 组间系数差异检验 — R（Fisher 置换检验）
# 场景：比较高组（subgroup=1）vs 低组（subgroup=0）的 DID 系数
# ============================================================
library(fixest)
library(dplyr)

set.seed(42)
N_PERM <- 1000

# ── 真实两组系数之差 ─────────────────────────────────────────
# 高组
res_hi <- feols(outcome ~ i(post, treated, ref=0) | unit_id + time,
                data = df %>% filter(subgroup == 1),
                cluster = ~unit_id)
# 低组
res_lo <- feols(outcome ~ i(post, treated, ref=0) | unit_id + time,
                data = df %>% filter(subgroup == 0),
                cluster = ~unit_id)

beta_hi   <- coef(res_hi)["post::1:treated"]
beta_lo   <- coef(res_lo)["post::1:treated"]
diff_real <- beta_hi - beta_lo
cat(sprintf("真实系数差: %.4f  (高组=%.4f, 低组=%.4f)\n",
            diff_real, beta_hi, beta_lo))

# ── 置换主循环：随机重新分配 subgroup 标签 ───────────────────
units_sub <- df %>% select(unit_id, subgroup) %>% distinct()

fake_diffs <- numeric(N_PERM)
for (i in seq_len(N_PERM)) {
  # 在单位级别随机打乱 subgroup
  shuffled <- units_sub %>%
    mutate(fake_sub = sample(subgroup))

  df_p <- df %>%
    left_join(shuffled %>% select(unit_id, fake_sub), by = "unit_id")

  r_hi <- tryCatch(
    feols(outcome ~ i(post, treated, ref=0) | unit_id + time,
          data = df_p %>% filter(fake_sub == 1), cluster = ~unit_id),
    error = function(e) NULL
  )
  r_lo <- tryCatch(
    feols(outcome ~ i(post, treated, ref=0) | unit_id + time,
          data = df_p %>% filter(fake_sub == 0), cluster = ~unit_id),
    error = function(e) NULL
  )

  if (!is.null(r_hi) && !is.null(r_lo)) {
    fake_diffs[i] <- coef(r_hi)["post::1:treated"] -
                     coef(r_lo)["post::1:treated"]
  } else {
    fake_diffs[i] <- NA_real_
  }
}

fake_diffs <- na.omit(fake_diffs)
emp_pval   <- mean(abs(fake_diffs) >= abs(diff_real))
cat(sprintf("置换检验 p 值: %.4f\n", emp_pval))
```

### Python 代码

```python
# ============================================================
# 组间系数差异检验 — Python（Fisher 置换检验）
# ============================================================
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
from tqdm import tqdm

np.random.seed(42)
N_PERM = 1000

def run_did(data, post_col, treated_col, controls=[]):
    """运行单次 TWFE DID，返回交互项系数"""
    data = data.copy()
    data['did'] = data[post_col] * data[treated_col]
    panel = data.set_index(['unit_id', 'time'])
    try:
        mod = PanelOLS(
            dependent=panel['outcome'],
            exog=sm.add_constant(panel[['did'] + controls]),
            entity_effects=True,
            time_effects=True
        )
        res = mod.fit(cov_type='clustered', cluster_entity=True)
        return res.params['did']
    except Exception:
        return np.nan

# ── 真实两组系数之差 ─────────────────────────────────────────
beta_hi_real = run_did(df[df['subgroup'] == 1], 'post', 'treated')
beta_lo_real = run_did(df[df['subgroup'] == 0], 'post', 'treated')
diff_real = beta_hi_real - beta_lo_real
print(f"真实系数差: {diff_real:.4f}")

# ── 置换循环 ─────────────────────────────────────────────────
units_sub = df[['unit_id', 'subgroup']].drop_duplicates()
fake_diffs = []

for _ in tqdm(range(N_PERM), desc="组间置换检验"):
    shuffled = units_sub.copy()
    shuffled['fake_sub'] = np.random.permutation(shuffled['subgroup'].values)
    df_p = df.merge(shuffled[['unit_id','fake_sub']], on='unit_id', how='left')

    b_hi = run_did(df_p[df_p['fake_sub'] == 1], 'post', 'treated')
    b_lo = run_did(df_p[df_p['fake_sub'] == 0], 'post', 'treated')
    if not (np.isnan(b_hi) or np.isnan(b_lo)):
        fake_diffs.append(b_hi - b_lo)

fake_diffs = np.array(fake_diffs)
emp_pval = np.mean(np.abs(fake_diffs) >= np.abs(diff_real))
print(f"置换检验 p 值: {emp_pval:.4f}  （有效置换次数: {len(fake_diffs)}）")
```

---

## 稳健性补全

### 排除其他政策干扰：加入同期政策虚拟变量

当样本期内存在其他可能影响结果变量的政策，需在主回归中加入同期政策虚拟变量，以防混淆。

```r
# ============================================================
# 排除同期政策干扰 — R
# ============================================================
library(fixest)

# 假设在样本期内还有 policy2（另一个政策的指示变量 0/1）
# 将其加入控制，验证主系数稳定性

res_main <- feols(
  outcome ~ i(post, treated, ref = 0) | unit_id + time,
  data = df, cluster = ~unit_id
)

# 加入同期其他政策指示变量
res_excl_policy <- feols(
  outcome ~ i(post, treated, ref = 0) + policy2 + policy3 | unit_id + time,
  data = df, cluster = ~unit_id
)

# 对比：主系数变化幅度 < 10% 则稳健
etable(res_main, res_excl_policy,
       headers = c("基准", "控制同期政策"),
       keep    = "post")
```

### 滞后效应检验：加入滞后 1-3 期处理变量

检验处理效应是否存在传导时滞（先锋与非先锋效应），同时排除预期效应的干扰。

```r
# ============================================================
# 滞后效应检验 — R（加入 lag1/lag2/lag3 处理变量）
# ============================================================
library(fixest)
library(dplyr)

df_lag <- df %>%
  arrange(unit_id, time) %>%
  group_by(unit_id) %>%
  mutate(
    did_lag1 = lag(post * treated, 1),  # 滞后 1 期处理
    did_lag2 = lag(post * treated, 2),  # 滞后 2 期
    did_lag3 = lag(post * treated, 3)   # 滞后 3 期
  ) %>%
  ungroup()

# 逐步加入滞后项
res_lag1 <- feols(
  outcome ~ i(post, treated, ref=0) + did_lag1 | unit_id + time,
  data = df_lag, cluster = ~unit_id
)
res_lag3 <- feols(
  outcome ~ i(post, treated, ref=0) + did_lag1 + did_lag2 + did_lag3 |
    unit_id + time,
  data = df_lag, cluster = ~unit_id
)

etable(res_main, res_lag1, res_lag3,
       headers = c("基准", "Lag1", "Lag1-3"),
       keep    = "post|lag")
```

### 调节效应模板：X×M 交互项 + 边际效应图

```r
# ============================================================
# 调节效应 + 边际效应图 — R（fixest + ggplot2）
# ============================================================
library(fixest)
library(ggplot2)
library(marginaleffects)  # install.packages("marginaleffects")

# 调节变量 M（连续，如企业规模对数）先做中心化
df <- df %>% mutate(M_c = M - mean(M, na.rm = TRUE))

# 交互项回归：DID × 调节变量
res_mod <- feols(
  outcome ~ i(post, treated, ref=0) * M_c | unit_id + time,
  data = df, cluster = ~unit_id
)
etable(res_mod)

# ── 边际效应图：DID 效应随 M 变化 ──────────────────────────
# 使用 marginaleffects 包自动计算
slopes_df <- marginaleffects::slopes(
  res_mod,
  variables  = "treated",          # 调节 DID 的 treated 方向
  newdata    = datagrid(
    M_c = seq(min(df$M_c, na.rm=TRUE), max(df$M_c, na.rm=TRUE), length.out = 50),
    post = 1
  )
)

ggplot(slopes_df, aes(x = M_c, y = estimate)) +
  geom_line(color = "steelblue", linewidth = 1.2) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.2, fill = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  labs(
    x        = "调节变量 M（中心化后）",
    y        = "处理效应（边际效应）",
    title    = "DID 处理效应随调节变量 M 变化的边际效应图",
    subtitle = "阴影区域 = 95% CI"
  ) +
  theme_minimal()
```

---

## DID处理变量类型自动检测

在运行 DID 之前，系统自动检测处理变量的类型，根据类型选择合适的估计策略。

```python
# ============================================================
# 处理变量类型自动检测 — Python
# ============================================================
import pandas as pd
import numpy as np

def detect_treatment_type(df, treatment_col, unit_col='unit_id', time_col='time'):
    """
    自动检测处理变量类型并给出估计策略建议。

    Returns:
        dict: 包含 type、unique_values、recommendation 等字段
    """
    treatment = df[treatment_col].dropna()
    unique_vals = treatment.unique()
    n_unique = len(unique_vals)

    result = {
        'col'          : treatment_col,
        'n_unique'     : n_unique,
        'unique_vals'  : sorted(unique_vals[:20]),  # 只显示前20个
        'has_missing'  : df[treatment_col].isnull().any()
    }

    # ── 检测逻辑 ──────────────────────────────────────────────
    if set(unique_vals).issubset({0, 1}):
        result['type'] = 'binary'
        result['recommendation'] = (
            "处理变量为二元（0/1）。\n"
            "推荐方法：\n"
            "  1. 单一处理时间点 → 经典 TWFE（Step 2）\n"
            "  2. 交错处理时间 → Callaway-Sant'Anna 或 Sun-Abraham（Step 4）\n"
            "  3. 对照 → Bacon 分解检查负权重（必做）"
        )
    elif n_unique <= 10 and all(isinstance(v, (int, np.integer)) for v in unique_vals):
        result['type'] = 'ordinal_discrete'
        result['recommendation'] = (
            "警告：处理变量为有序离散（多值），非严格二元。\n"
            "建议：\n"
            "  - 如果各值代表处理强度（doses），使用 Intensity DID\n"
            "  - 推荐 did_multiplegt_dyn（DIDmultiplegt 包）\n"
            "  - 或将其二值化（>0 = treated）后跑标准 DID（需说明截断理由）"
        )
    else:
        result['type'] = 'continuous'
        result['recommendation'] = (
            "警告：处理变量为连续型（Intensity / Dosage DID）！\n"
            "连续处理需要不同的估计策略：\n"
            "  1. 使用 did_multiplegt_dyn（dCDH 2024，支持连续处理）\n"
            "  2. 或将其离散化为分组 DID（四分位），但需理论支撑\n"
            "  3. 不能直接套用二元 TWFE，否则系数解释为弹性而非 ATT\n"
            "注意：需额外检验处理强度的外生性！"
        )

    # ── 额外检查：单调性（处理是否可逆）─────────────────────
    if time_col in df.columns and unit_col in df.columns:
        reversals = (
            df.sort_values([unit_col, time_col])
              .groupby(unit_col)[treatment_col]
              .apply(lambda x: (x.diff() < 0).any())
              .sum()
        )
        result['n_reversals'] = int(reversals)
        if reversals > 0:
            result['recommendation'] += (
                f"\n\n额外警告：检测到 {reversals} 个单位存在处理逆转（treatment reversal）！\n"
                "标准 DID 假设处理是永久的。如存在逆转，请使用：\n"
                "  - did_multiplegt（支持处理逆转）\n"
                "  - 或单独处理逆转单位（样本删除/敏感性分析）"
            )

    return result


# ── 使用示例 ──────────────────────────────────────────────────
detect_result = detect_treatment_type(df, treatment_col='treated')
print(f"处理变量类型: {detect_result['type']}")
print(f"唯一值数量  : {detect_result['n_unique']}")
print(f"处理逆转单位: {detect_result.get('n_reversals', 'N/A')}")
print("\n建议策略:")
print(detect_result['recommendation'])
```

```r
# ============================================================
# 处理变量类型自动检测 — R
# ============================================================
detect_treatment_type <- function(df, treatment_col,
                                   unit_col = "unit_id",
                                   time_col = "time") {
  treatment  <- df[[treatment_col]]
  unique_vals <- sort(na.omit(unique(treatment)))
  n_unique    <- length(unique_vals)

  # ── 类型判断 ─────────────────────────────────────────────
  if (all(unique_vals %in% c(0, 1))) {
    type <- "binary"
    rec  <- paste0(
      "处理变量为二元（0/1）。\n",
      "推荐：标准 TWFE / CS2021 / SA2021（见主流程 Step 2-4）\n",
      "必须先运行 Bacon 分解检查负权重。"
    )
  } else if (n_unique <= 10 && all(unique_vals == floor(unique_vals))) {
    type <- "ordinal_discrete"
    rec  <- paste0(
      "警告：处理为有序离散（多值），非严格二元。\n",
      "建议：使用 did_multiplegt_dyn 或二值化后跑标准 DID。"
    )
  } else {
    type <- "continuous"
    rec  <- paste0(
      "警告：处理变量为连续型（Intensity DID）！\n",
      "连续处理需要 did_multiplegt_dyn 或 Callaway-Sant'Anna（连续版）。\n",
      "不能直接套用二元 TWFE，系数解释将变为弹性而非 ATT。"
    )
  }

  # ── 检测处理逆转 ─────────────────────────────────────────
  reversals <- df |>
    dplyr::arrange(.data[[unit_col]], .data[[time_col]]) |>
    dplyr::group_by(.data[[unit_col]]) |>
    dplyr::summarise(reversed = any(diff(.data[[treatment_col]]) < 0,
                                    na.rm = TRUE)) |>
    dplyr::pull(reversed) |>
    sum()

  if (reversals > 0) {
    rec <- paste0(rec, sprintf(
      "\n\n额外警告：检测到 %d 个单位存在处理逆转！\n",
      "标准 DID 假设处理永久，建议使用 did_multiplegt。",
      reversals
    ))
  }

  list(type = type, n_unique = n_unique,
       unique_vals = unique_vals,
       n_reversals = reversals,
       recommendation = rec)
}

# ── 使用 ──────────────────────────────────────────────────────
result <- detect_treatment_type(df, "treated")
cat("类型:", result$type, "\n")
cat("逆转单位数:", result$n_reversals, "\n")
cat("建议策略:\n", result$recommendation, "\n")
```

---

## Estimand声明

在使用 DID 方法时，必须在论文方法部分和结果表格脚注中明确声明估计量（Estimand）的性质。

| 方法 | 默认 Estimand | 必须声明 |
|------|--------------|---------|
| DID（TWFE） | Weighted ATT（加权平均处理效应，权重可能为负） | 负权重比例（Bacon 分解结果） |
| DID（CS2021 / SA2021） | Group-time ATT → 聚合后的 ATT | 聚合权重选择理由（simple / dynamic / calendar） |
| PSM-DID | ATT（匹配样本上的平均处理效应） | 共同支撑样本损失比例 |
| Stacked DID | 各队列 ATT 的等权平均 | 窗口宽度选择依据 |
| BJS（did2s） | ATT（插补估计，理论上等于 CS/SA 的加权平均） | 同 CS/SA |
| dCDH（did_multiplegt） | 处理强度 × 效应的加权平均 | 处理逆转单位处理方式 |

**DID (TWFE)：** 在样本中报告的 $\hat\beta_{TWFE}$ 是 Weighted ATT，其权重由各 (unit, time) 对的处理状态方差决定。**必须报告负权重比例**（通过 Bacon 分解或 twowayfeweights）。若负权重比例 > 10%，TWFE 估计不可信，须切换 CS/SA。

**DID (CS/SA)：** 报告的是 Group-time ATT $ATT(g,t)$ 的聚合值。聚合方式影响解释：
- `type = "simple"`：等权平均，对应总体 ATT
- `type = "dynamic"`：按事件时间聚合，对应平均动态效应（用于事件研究图）
- `type = "calendar"`：按日历时间聚合，对应某时期截面 ATT

**必须在脚注中说明聚合权重选择理由**，例如：*"We aggregate group-time ATTs using simple averaging weights, which give equal weight to each cohort-period cell and are robust to heterogeneous treatment timing. Dynamic aggregation is reported in Figure X as the event study plot."*
