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
