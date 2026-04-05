# 面板数据分析 — 固定效应与随机效应

适用场景：面板结构数据的FE/RE估计、Hausman检验、高维固定效应、标准误选择、诊断检验。触发关键词：面板数据、固定效应、随机效应、TWFE、linearmodels、plm、fixest、pyfixest、Hausman、组内变异。

> **动态面板GMM（System GMM）** 已迁移至 `panel-data/GMM-DYNAMIC.md`，按需加载。

---

## 概述与前置条件

面板数据允许控制不随时间变化的个体异质性（个体FE）和共同时间趋势（时间FE）。FE估计识别的是**个体内（within）变异**，不受个体层面不可观测异质性影响。

**数据结构要求：**
```
长格式（long format）：
  - 个体ID列（entity_id）：企业/地区编码
  - 时间列（time）：年份/季度，数值型
  - 结果变量 Y，核心解释变量 X，控制变量 W
  - 平衡/非平衡面板均可
```

---

## Step 1: 面板描述性统计

包含 overall / between / within 标准差分解，诊断FE识别的来源。

```python
# Python: 面板描述性统计 + 方差分解
import pandas as pd
import numpy as np

def panel_describe(df, vars_list, entity_id='entity_id', time_id='time'):
    """overall/between/within方差分解，类似Stata xtsum"""
    n_entities = df[entity_id].nunique()
    n_periods  = df[time_id].nunique()
    n_obs      = len(df)

    print(f"面板维度: N={n_entities}, T={n_periods}, NT={n_obs}")
    obs_per = df.groupby(entity_id)[time_id].count()
    print(f"平衡面板: {obs_per.std() == 0}（每实体观测数 {obs_per.min()}–{obs_per.max()}）\n")

    records = []
    for var in vars_list:
        overall_sd  = df[var].std()
        between_sd  = df.groupby(entity_id)[var].mean().std()
        within_vals = df[var] - df.groupby(entity_id)[var].transform('mean') + df[var].mean()
        within_sd   = within_vals.std()
        records.append({
            'variable':   var,
            'mean':       round(df[var].mean(), 4),
            'overall_sd': round(overall_sd, 4),
            'between_sd': round(between_sd, 4),
            'within_sd':  round(within_sd, 4),
            'within/total': round(within_sd / overall_sd, 3) if overall_sd > 0 else np.nan
        })
        if within_sd < 0.01 * overall_sd:
            print(f"⚠️ {var}: within SD极小（{within_sd:.4f}），FE可能无法识别")

    return pd.DataFrame(records)

desc = panel_describe(df, [outcome, core_var] + controls)
print(desc.to_string(index=False))
```

```r
# R: 面板方差分解（pyfixest作为Python替代；R用以下代码）
library(dplyr)

panel_describe_r <- function(df, vars, entity_id = "entity_id", time_id = "time") {
  cat(sprintf("N=%d, T=%d, NT=%d\n",
              n_distinct(df[[entity_id]]),
              n_distinct(df[[time_id]]),
              nrow(df)))
  results <- lapply(vars, function(v) {
    overall_sd  <- sd(df[[v]], na.rm = TRUE)
    between_sd  <- sd(tapply(df[[v]], df[[entity_id]], mean, na.rm=TRUE), na.rm=TRUE)
    within_vals <- df[[v]] - ave(df[[v]], df[[entity_id]], FUN=mean) + mean(df[[v]], na.rm=TRUE)
    within_sd   <- sd(within_vals, na.rm=TRUE)
    list(variable=v, mean=mean(df[[v]],na.rm=TRUE),
         overall_sd=overall_sd, between_sd=between_sd, within_sd=within_sd,
         within_ratio=within_sd/overall_sd)
  })
  do.call(rbind, lapply(results, as.data.frame))
}

desc_r <- panel_describe_r(df, c(outcome, core_var, controls))
print(desc_r)
```

> **pyfixest（Python替代linearmodels）：** Python中可用 `pyfixest` 替代 `linearmodels`，API更接近R的fixest，速度更快：`pip install pyfixest`。

---

## Step 2: Hausman 检验（FE vs RE）

**⚠️ Python的linearmodels没有内置Hausman检验，必须手写。**

H₀：随机效应一致（个体效应与X不相关）→ RE有效率；拒绝H₀ → 用FE。

```python
# Python: 手写Hausman检验（linearmodels无内置！）
import numpy as np
from scipy import stats
from linearmodels.panel import PanelOLS, RandomEffects
import statsmodels.api as sm

df_panel = df.set_index([entity_id, time_id])
exog = sm.add_constant(df_panel[[core_var] + controls])

# FE 和 RE 估计
fe_res = PanelOLS(df_panel[outcome], exog, entity_effects=True).fit(
    cov_type='clustered', cluster_entity=True)
re_res = RandomEffects(df_panel[outcome], exog).fit()

# ⚠️ RE必须包含add_constant（截距），否则协方差矩阵维度不匹配

def hausman_test(fe_res, re_res):
    """手写Hausman检验（排除截距项）"""
    b_fe = fe_res.params
    b_re = re_res.params

    # 公共变量（排除截距）
    common = [v for v in b_fe.index if v in b_re.index and v != 'const']
    b_fe_c = b_fe[common].values
    b_re_c = b_re[common].values

    # 协方差差矩阵
    V_fe = fe_res.cov.loc[common, common].values
    V_re = re_res.cov.loc[common, common].values
    V_diff = V_fe - V_re

    # 处理数值不稳定（V_diff可能不正定）
    try:
        from numpy.linalg import inv, matrix_rank
        if matrix_rank(V_diff) < len(common):
            print("⚠️ V_diff奇异，使用广义逆")
            V_inv = np.linalg.pinv(V_diff)
        else:
            V_inv = inv(V_diff)

        d = b_fe_c - b_re_c
        H = float(d @ V_inv @ d)
        dof = len(common)
        p_val = 1 - stats.chi2.cdf(H, dof)
        return {'H_stat': round(H, 4), 'dof': dof, 'p_value': round(p_val, 4),
                'conclusion': 'FE' if p_val < 0.05 else 'RE'}
    except Exception as e:
        return {'error': str(e)}

hausman_result = hausman_test(fe_res, re_res)
print(f"Hausman H={hausman_result['H_stat']:.3f}, p={hausman_result['p_value']:.4f}")
print(f"结论: {'拒绝H0 → 用FE' if hausman_result['p_value'] < 0.05 else '不拒绝H0 → RE有效率'}")
```

```r
# R: Hausman检验（plm）
library(plm)
library(fixest)

fe_plm <- plm(Y ~ X + control1 + control2,
              data = df, index = c("entity_id", "time"), model = "within")
re_plm <- plm(Y ~ X + control1 + control2,
              data = df, index = c("entity_id", "time"), model = "random")

phtest(fe_plm, re_plm)
# p < 0.05 → 拒绝H0 → 用FE
# 实践提示：大多数经管研究默认FE（更保守），RE仅在理论有依据时使用
```

> **实践提示：** 大多数经管研究默认使用FE，因为FE对个体异质性更保守。Hausman检验只是形式确认。

---

## Step 3: FE 模型估计

### 3a: Entity FE（个体固定效应）

\[Y_{it} = \alpha_i + \beta X_{it} + \gamma W_{it} + \varepsilon_{it}\]

```python
# Python: Entity FE（linearmodels）
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

df_panel = df.set_index([entity_id, time_id])
mod = PanelOLS(
    dependent      = df_panel[outcome],
    exog           = sm.add_constant(df_panel[[core_var] + controls]),
    entity_effects = True,
    time_effects   = False
)
res_efe = mod.fit(cov_type='clustered', cluster_entity=True)
print(res_efe.summary)

# pyfixest替代方案（API更简洁）
# import pyfixest as pf
# res_efe_pf = pf.feols(f"{outcome} ~ {core_var} + {'+'.join(controls)} | {entity_id}", df)
# pf.etable([res_efe_pf])
```

```r
# R: Entity FE（fixest）
library(fixest)
res_efe <- feols(Y ~ X + control1 + control2 | entity_id,
                 data = df, cluster = ~entity_id)
etable(res_efe)
```

---

### 3b: TWFE（双向固定效应）

\[Y_{it} = \mu_i + \lambda_t + \beta X_{it} + \gamma W_{it} + \varepsilon_{it}\]

```python
# Python: TWFE（linearmodels）
mod_twfe = PanelOLS(
    dependent      = df_panel[outcome],
    exog           = sm.add_constant(df_panel[[core_var] + controls]),
    entity_effects = True,
    time_effects   = True
)
res_twfe = mod_twfe.fit(cov_type='clustered', cluster_entity=True)
```

```r
# R: TWFE（fixest）
res_twfe <- feols(Y ~ X + control1 + control2 | entity_id + time,
                  data = df, cluster = ~entity_id)
```

---

### 3c: 高维FE（省份×年份等交互FE）

```python
# Python: 高维FE（pyfixest）
import pyfixest as pf

# 高维FE：个体 + 时间 + 行业×年份
res_hdfe = pf.feols(
    f"{outcome} ~ {core_var} + control1 | {entity_id} + {time_id} + industry^year",
    data = df
)
pf.etable([res_hdfe])
```

```r
# R: 高维FE（fixest，速度极快）
res_hdfe <- feols(
    Y ~ X + control1 + control2 | entity_id + time + industry^year,
    data    = df,
    cluster = ~entity_id
)

# 逐步加入FE（展示识别来源）
res1 <- feols(Y ~ X + control1 | entity_id, df, cluster=~entity_id)
res2 <- feols(Y ~ X + control1 | entity_id + time, df, cluster=~entity_id)
res3 <- feols(Y ~ X + control1 | entity_id + time + industry^year, df, cluster=~entity_id)
etable(res1, res2, res3, fitstat = ~ r2_within + n)
```

---

## Step 4: 标准误选择

| 场景 | 推荐标准误 | R代码 | Python代码 |
|------|-----------|-------|-----------|
| 标准面板 | 聚类（个体层面） | `cluster = ~entity_id` | `cov_type='clustered'` |
| 处理在组层面 | 聚类（组层面） | `cluster = ~province` | `cluster_entity=False` |
| < 50个聚类 | Wild Bootstrap | `boottest(...)` | `wildboottest` |
| 截面相关（宏观） | Driscoll-Kraay | `vcovSCC(...)` | — |
| 时序相关 | Newey-West HAC | `vcovNW(...)` | — |

**聚类层级规则：**
- 聚类到**处理分配层级**（处理在省级 → 聚类到省）
- 不确定时 → 报告多种层级，选最保守的
- 聚类数 < 50 → Wild Bootstrap

```r
# R: Driscoll-Kraay（宏观面板，截面相关）
library(lmtest); library(sandwich); library(plm)

res_plm <- plm(Y ~ X + control1, data=df,
               index=c("entity_id","time"), model="within")
coeftest(res_plm, vcov = vcovSCC(res_plm, type="HC3", maxlag=2))

# R: Wild Bootstrap（<50个聚类）
library(fwildclusterboot)
res_fe <- feols(Y ~ X + control1 | entity_id + time, data=df)
boottest(res_fe, clustid="province", param="X", B=9999)
```

---

## Step 5: 诊断检验

### 5a: 序列相关（Wooldridge Test）

```r
# R: 面板序列相关检验（plm）
library(plm)
pbgtest(res_plm, order=1)   # Breusch-Godfrey for panels
# 若显著（p<0.05）→ 需用HAC SE或Driscoll-Kraay
```

---

### 5b: 截面相关（Pesaran CD）

```r
# R: Pesaran CD检验
library(plm)
pcdtest(res_plm, test="cd")
# 若显著 → 用Driscoll-Kraay SE；宏观国家面板尤其需要
```

---

### 5c: 面板单位根（长面板T>10必做）

```r
# R: Im-Pesaran-Shin单位根检验
library(plm)
purtest(Y ~ 1, data=df, index=c("entity_id","time"),
        pmax=2, test="ips", exo="trend")
# H0: 所有个体均有单位根
# p < 0.05 → 拒绝单位根（平稳），可直接用FE
# p > 0.05 → 考虑一阶差分或协整检验

# Hadri检验（H0: 平稳）
purtest(Y ~ 1, data=df, index=c("entity_id","time"),
        test="hadri", exo="intercept")
```

---

## Step 6: Mundlak / Correlated RE（新增）

当需要保留不随时间变化的变量（如性别、地区）但FE会吸收时，使用Mundlak/CRE方法：在RE中加入个体均值作为额外控制变量。

```r
# R: Mundlak/CRE — 在RE中加入组均值
library(plm)
library(dplyr)

# 计算时变变量的个体均值（"between"部分）
df <- df %>%
  group_by(entity_id) %>%
  mutate(
    X_mean       = mean(X, na.rm = TRUE),
    control1_mean = mean(control1, na.rm = TRUE),
    control2_mean = mean(control2, na.rm = TRUE)
  ) %>%
  ungroup()

# Mundlak RE（含组均值）
re_mundlak <- plm(
  Y ~ X + control1 + control2 +          # 时变变量
      time_invariant_var +                # 不随时间变化的变量（FE无法估计！）
      X_mean + control1_mean + control2_mean,  # Mundlak项（组均值）
  data  = df,
  index = c("entity_id", "time"),
  model = "random"
)
summary(re_mundlak)

# 检验Mundlak项是否联合显著（等价于Hausman检验）
mundlak_vars <- c("X_mean", "control1_mean", "control2_mean")
linearHypothesis(re_mundlak, paste0(mundlak_vars, " = 0"))
# 若联合显著 → FE和RE有差异（等价Hausman拒绝H0）

# 注：Mundlak RE的X系数 ≈ FE系数（within效应）
#     time_invariant_var的系数来自between变异，FE模型无法估计
```

**Mundlak方法优势：**
- 保留不随时间变化的变量（如地区虚拟变量、个人性别）
- 通过检验Mundlak项联合显著性替代Hausman检验
- 提供within效应和between效应的分解

---

## 检验清单

| 步骤 | 检验 | 通过标准 |
|------|------|---------|
| Step 1 | 组内变异分解 | within SD > 0；within/total > 0.1 |
| Step 2 | Hausman检验 | 默认FE；RE需有理论依据 |
| Step 3 | FE模型逐步加入 | 核心系数方向稳定 |
| Step 4 | 标准误类型 | 匹配处理分配层级 |
| Step 5a | 序列相关 | 若显著 → 升级SE |
| Step 5b | 截面相关 | 若显著 → DK SE |
| Step 5c | 单位根（T>10） | 平稳或差分处理 |

---

## 常见错误

> **错误1：FE后无法估计不随时间变化的变量**
> 个体FE会吸收所有不随时间变化的变量（如地区、性别、行业代码）。若这些是研究核心，用Mundlak/CRE方法（Step 6）。

> **错误2：TWFE不等于已控制所有混淆**
> TWFE只控制加性个体效应和时间效应，时变的遗漏变量（如企业当年新政策）仍引起内生性。

> **错误3：聚类层级选错**
> 聚类过细（如个体层面）会低估SE；聚类过粗会高估SE。聚到**处理分配层级**。

> **错误4：忽略序列相关**
> 同一个体跨期误差项通常正相关。不处理序列相关会严重低估SE（过度拒绝H0）。

> **错误5：报告overall R²**
> FE模型应报告**within R²**（组内拟合优度），不要报告overall R²（被FE虚高）。

> **错误6：Python linearmodels的Hausman内置函数**
> `linearmodels`没有内置Hausman检验。必须手写（见Step 2），否则会报错或给出错误结果。

---

## Estimand 声明

| 方法 | Estimand | 声明要点 |
|------|----------|---------|
| Entity FE | 组内变异识别的效应 | 效应由个体内时间变异识别；不随时间变化的变量被吸收 |
| TWFE | 加性ATT（可能含负权重） | 处理时机异质时TWFE加权可能产生负权重，见DID skill |
| RE | GLS加权平均（within + between） | 须通过Hausman检验；报告个体效应假设 |
| Mundlak RE | within效应（X系数）+ between效应 | 时不变变量系数来自between变异 |

**声明模板：**
> "本文采用双向固定效应（TWFE）估计，通过个体内跨期变异识别处理效应，控制了不随时间变化的个体异质性（个体FE）和共同时间趋势（年份FE）。Estimand为以组内变异加权的平均处理效应。标准误聚类到[处理分配层级]。"

---

## 输出规范

```r
# R: fixest etable输出
etable(res1, res2, res3, res4,
       title   = "Panel FE Results",
       headers = c("Entity FE", "+Controls", "TWFE", "+HDFE"),
       fitstat = ~ r2_within + n + G,   # G=聚类数
       tex     = FALSE)
```

**必须注明：**
1. 固定效应类型（个体FE、年份FE、行业×年份FE）
2. 标准误类型（Clustered at entity level）
3. Within R²、N、聚类数 G

```
output/
  panel_variance_decomp.csv    # 组内/组间方差分解
  panel_hausman.txt            # Hausman检验结果
  panel_main_results.csv       # 主回归结果表
  panel_diagnostics.txt        # 序列/截面相关检验
  panel_unit_root.txt          # 单位根检验（长面板）
  panel_robustness.csv         # 稳健性（不同SE/FE规格）
```
