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
