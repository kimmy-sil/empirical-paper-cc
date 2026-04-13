---
name: panel-data
description: "面板数据分析，含面板分类路由、FE决策树、FE/RE选择、聚类标准误、面板诊断"
---

# 面板数据分析 — 从数据结构到模型选择

适用场景：面板结构数据的分类路由、FE/RE估计、Hausman检验、高维固定效应、标准误选择、诊断检验。触发关键词：面板数据、固定效应、随机效应、TWFE、linearmodels、plm、fixest、pyfixest、Hausman、组内变异。

> **子文件按需加载：**
> - 动态面板GMM → `panel-data/GMM-DYNAMIC.md`
> - 长面板（T >> N）→ `panel-data/LONG-PANEL.md`
> - 离散因变量面板 → `panel-data/DISCRETE.md`
> - 非平衡面板诊断 → `panel-data/UNBALANCED.md`

---

## Step 0: 面板分类与路由

**数据进来先回答四个问题，再进入具体流程。**

### 0a: 面板类型判断（短 vs 长）

```
N 和 T 的相对大小？
│
├─ N >> T（如 500企业 × 10年）→ 短面板
│   └─ 留在本文件，继续 Step 0b
│
├─ T >> N（如 30国 × 50年）→ 长面板
│   └─ ⚠️ 跳转 LONG-PANEL.md（先做单位根，再决定模型）
│
└─ N ≈ T（如 50省 × 20年）→ 边界情形
    ├─ 先做单位根检验（Step 5c）
    ├─ 平稳 → 按短面板流程 + 注意截面相关
    └─ 非平稳 → 按长面板流程
```

**判断标准：** T > 20–30 且 N < T → 长面板特征显著，必须先检验平稳性。

### 0b: 面板平衡性检查

```
面板是否平衡？
│
├─ 平衡（所有个体观测期数相同）→ 继续 Step 0c
│
└─ 非平衡 → 先跳 UNBALANCED.md 诊断缺失机制
    ├─ 随机缺失（MCAR/MAR）→ 回到主流程，FE/RE 可直接跑
    └─ 非随机缺失（MNAR）→ 需要选择模型修正，见 UNBALANCED.md
```

### 0c: 因变量类型判断

```
Y 的类型？
│
├─ 连续变量 → 继续 Step 0d
│
├─ 二元（0/1）→ 跳转 DISCRETE.md
│
├─ 计数（非负整数）→ 跳转 DISCRETE.md
│
└─ 有序离散 → 跳转 DISCRETE.md
```

### 0d: X 变异来源诊断 → FE 结构选择 → 聚类层级确认

**固定效应不是默认操作，而是需要理由的识别选择。**

#### X 变异来源

核心问题：β 要从哪种变异中识别？

```
X 的变异主要来自哪里？
│
├─ 主要是 within（个体内跨期变化）
│   → Entity FE 可以识别 ✓
│   → 例：企业逐年研发投入变化 → 对利润的影响
│
├─ 主要是 between（个体间差异）
│   → Entity FE 会吸收 X → 无法识别 ✗
│   → 例：企业所有制（国企/民企）→ 对效率的影响
│   → 解决：Mundlak/CRE（Step 6）、RE、或截面回归
│
├─ 混合（within + between 都有）
│   → FE 只识别 within 部分，between 部分被吸收
│   → 须在论文中声明 estimand 是 within 效应
│
└─ X 是政策冲击（某时点突然变化）
    → 这是 DID 设定，跳转 did-analysis skill
```

**判断依据：** Step 1 方差分解的 `within/total` 比值：
- `> 0.5` → within 变异为主，FE 可识别
- `0.1–0.5` → 可用 FE 但效率较低，须讨论
- `< 0.1` → FE 下 X 近似被吸收，不宜用 FE

#### FE 结构选择

```
需要控制什么不可观测因素？
│
├─ 个体层面不可观测异质性（如企业文化、管理能力）
│   → 加 Entity FE
│
├─ 共同时间趋势（如宏观经济周期、政策环境）
│   → 加 Time FE
│
├─ 行业/地区特定时间趋势
│   → 加交互 FE（industry × year）
│   → ⚠️ 每加一层 FE 都吸收变异，须验证 X 还有足够 within 变异
│
├─ 模型含滞后因变量 Y_{t-1}
│   → FE 有 Nickell 偏误，跳转 GMM-DYNAMIC.md
│
└─ 不确定 → 逐步加入，观察系数变化（见 Step 3d）
```

#### 聚类层级确认

```
处理/冲击在哪个层级分配？
│
├─ 个体层面（每个企业独立受处理）→ 聚类到个体
├─ 省/地区层面（政策在省级实施）→ 聚类到省
├─ 行业层面（行业监管变化）→ 聚类到行业
├─ 不确定 → 报告多层级，取最保守
└─ 聚类数 < 50 → Wild Bootstrap（见 Step 4）
```

---

## Step 1: 数据准备与描述性统计

### 1a: 长格式检查与转换

```python
# Python: 长格式检查
import pandas as pd

def check_panel_structure(df, entity_id='entity_id', time_id='time'):
    """检查面板数据基本结构"""
    n_entities = df[entity_id].nunique()
    n_periods  = df[time_id].nunique()
    n_obs      = len(df)

    # 唯一性检查
    dup = df.duplicated(subset=[entity_id, time_id], keep=False)
    if dup.any():
        print(f"⚠️ 存在 {dup.sum()} 行重复的 (entity, time) 组合，必须先去重！")
        return False

    # 平衡性检查
    obs_per = df.groupby(entity_id)[time_id].count()
    balanced = obs_per.std() == 0

    # N vs T 判断
    panel_type = "短面板" if n_entities > 3 * n_periods else (
                 "长面板" if n_periods > 3 * n_entities else "边界面板")

    print(f"面板维度: N={n_entities}, T={n_periods}, NT={n_obs}")
    print(f"面板类型: {panel_type}")
    print(f"平衡面板: {balanced}（每实体观测数 {obs_per.min()}–{obs_per.max()}）")

    if panel_type == "长面板":
        print("⚠️ T >> N，建议先做单位根检验 → 跳转 LONG-PANEL.md")
    if not balanced:
        print("⚠️ 非平衡面板，建议先诊断缺失机制 → 跳转 UNBALANCED.md")

    return True
```

```python
# Python: 宽转长格式
def wide_to_long(df_wide, entity_id='entity_id', time_vars=None, stub_names=None):
    """宽格式 → 长格式转换"""
    df_long = pd.wide_to_long(df_wide, stubnames=stub_names,
                               i=entity_id, j='time', sep='_').reset_index()
    return df_long
```

### 1b: 方差分解（overall / between / within）

```python
# Python: 面板描述性统计 + 方差分解
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
        ratio       = round(within_sd / overall_sd, 3) if overall_sd > 0 else np.nan
        records.append({
            'variable':   var,
            'mean':       round(df[var].mean(), 4),
            'overall_sd': round(overall_sd, 4),
            'between_sd': round(between_sd, 4),
            'within_sd':  round(within_sd, 4),
            'within/total': ratio
        })
        # Step 0d 联动：X 变异来源判断
        if within_sd < 0.01 * overall_sd:
            print(f"⚠️ {var}: within SD 极小（{within_sd:.4f}），FE 无法识别此变量")
        elif ratio < 0.1:
            print(f"⚠️ {var}: within/total={ratio}，FE 识别力弱，考虑 RE 或 Mundlak")

    return pd.DataFrame(records)
```

```r
# R: 面板方差分解
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
    ratio       <- within_sd / overall_sd
    if (ratio < 0.1) cat(sprintf("⚠️ %s: within/total=%.3f，FE识别力弱\n", v, ratio))
    list(variable=v, mean=mean(df[[v]],na.rm=TRUE),
         overall_sd=overall_sd, between_sd=between_sd, within_sd=within_sd,
         within_ratio=ratio)
  })
  do.call(rbind, lapply(results, as.data.frame))
}
```

---

## Step 2: 模型选择三步检验

### 2a: F 检验（Pooled OLS vs FE）

H₀：所有个体效应 αᵢ = 0（不需要 FE）。拒绝 → 用 FE。

```python
# Python: F检验（pyfixest）
import pyfixest as pf

def test_pooled_vs_fe(df, outcome, core_var, controls, entity_id='entity_id'):
    """比较 Pooled OLS 和 Entity FE，报告 FE 联合显著性"""
    ctrl_str = ' + '.join([core_var] + controls)
    res_pool = pf.feols(f"{outcome} ~ {ctrl_str}", data=df)
    res_fe   = pf.feols(f"{outcome} ~ {ctrl_str} | {entity_id}", data=df)
    pf.etable([res_pool, res_fe])
    # pyfixest 的 FE 模型自动报告 F-stat for FE significance
    return res_pool, res_fe
```

```r
# R: F检验（plm）
library(plm)

fe_mod <- plm(Y ~ X + control1 + control2,
              data = df, index = c("entity_id", "time"), model = "within")
pFtest(fe_mod, plm(Y ~ X + control1 + control2,
                    data = df, index = c("entity_id", "time"), model = "pooling"))
# p < 0.05 → 拒绝 H0 → 个体效应显著 → 不应用 Pooled OLS
```

### 2b: Breusch-Pagan LM 检验（Pooled OLS vs RE）

H₀：个体方差 σ²ᵤ = 0（不需要 RE）。拒绝 → RE 优于 Pooled。

```r
# R: BP-LM检验（plm）
re_mod <- plm(Y ~ X + control1 + control2,
              data = df, index = c("entity_id", "time"), model = "random")
plmtest(re_mod, type = "bp")
# p < 0.05 → 拒绝 H0 → 个体方差显著 → RE 优于 Pooled
```

### 2c: Hausman 检验（FE vs RE）

H₀：个体效应与 X 不相关（RE 一致）。拒绝 → 用 FE。

```python
# Python: 手写Hausman检验（linearmodels无内置！）
import numpy as np
from scipy import stats
from linearmodels.panel import PanelOLS, RandomEffects
import statsmodels.api as sm

def hausman_test(df, outcome, core_var, controls,
                 entity_id='entity_id', time_id='time'):
    df_panel = df.set_index([entity_id, time_id])
    exog = sm.add_constant(df_panel[[core_var] + controls])

    fe_res = PanelOLS(df_panel[outcome], exog, entity_effects=True).fit(
        cov_type='clustered', cluster_entity=True)
    re_res = RandomEffects(df_panel[outcome], exog).fit()

    b_fe = fe_res.params
    b_re = re_res.params
    common = [v for v in b_fe.index if v in b_re.index and v != 'const']
    b_fe_c = b_fe[common].values
    b_re_c = b_re[common].values

    V_diff = fe_res.cov.loc[common, common].values - re_res.cov.loc[common, common].values
    try:
        if np.linalg.matrix_rank(V_diff) < len(common):
            V_inv = np.linalg.pinv(V_diff)
        else:
            V_inv = np.linalg.inv(V_diff)
        d = b_fe_c - b_re_c
        H = float(d @ V_inv @ d)
        p_val = 1 - stats.chi2.cdf(H, len(common))
        conclusion = 'FE' if p_val < 0.05 else 'RE'
        print(f"Hausman H={H:.3f}, p={p_val:.4f} → {conclusion}")
        return {'H_stat': round(H, 4), 'p_value': round(p_val, 4), 'conclusion': conclusion}
    except Exception as e:
        return {'error': str(e)}
```

```r
# R: Hausman检验（plm）
library(plm)
fe_plm <- plm(Y ~ X + control1 + control2,
              data = df, index = c("entity_id", "time"), model = "within")
re_plm <- plm(Y ~ X + control1 + control2,
              data = df, index = c("entity_id", "time"), model = "random")
phtest(fe_plm, re_plm)
# p < 0.05 → FE；p >= 0.05 → RE
```

### 模型选择决策总结

```
Pooled OLS → FE/RE → FE vs RE 三步：

[1] F检验: Pooled vs FE
│   ├─ p < 0.05 → FE 优于 Pooled
│   └─ p ≥ 0.05 → 可能不需要个体效应
│
[2] BP-LM: Pooled vs RE
│   ├─ p < 0.05 → RE 优于 Pooled
│   └─ p ≥ 0.05 → Pooled OLS 即可
│
[3] Hausman: FE vs RE（仅在 [1] 和 [2] 都拒绝时需要）
│   ├─ p < 0.05 → FE
│   └─ p ≥ 0.05 → RE
│
⚠️ 实践提示：大多数经管研究默认 FE（更保守），
   RE 仅在理论有依据时使用（如需要估计不随时间变化的变量）。
```

---

## Step 3: FE / RE 模型估计

### 3a: Entity FE（个体固定效应）

\[Y_{it} = \alpha_i + \beta X_{it} + \gamma W_{it} + \varepsilon_{it}\]

```python
# Python: Entity FE（pyfixest，推荐）
import pyfixest as pf

res_efe = pf.feols(f"{outcome} ~ {core_var} + {'+'.join(controls)} | {entity_id}",
                   data=df, vcov={entity_id: "hetero"})
pf.etable([res_efe])
```

```r
# R: Entity FE（fixest）
library(fixest)
res_efe <- feols(Y ~ X + control1 + control2 | entity_id,
                 data = df, cluster = ~entity_id)
etable(res_efe)
```

### 3b: TWFE（双向固定效应）

\[Y_{it} = \mu_i + \lambda_t + \beta X_{it} + \gamma W_{it} + \varepsilon_{it}\]

```python
# Python: TWFE（pyfixest）
res_twfe = pf.feols(
    f"{outcome} ~ {core_var} + {'+'.join(controls)} | {entity_id} + {time_id}",
    data=df, vcov={entity_id: "hetero"})
```

```r
# R: TWFE（fixest）
res_twfe <- feols(Y ~ X + control1 + control2 | entity_id + time,
                  data = df, cluster = ~entity_id)
```

### 3c: 高维FE（省份×年份等交互FE）

```python
# Python: 高维FE（pyfixest）
res_hdfe = pf.feols(
    f"{outcome} ~ {core_var} + control1 | {entity_id} + {time_id} + industry^year",
    data=df)
```

```r
# R: 高维FE（fixest，速度极快）
res_hdfe <- feols(
    Y ~ X + control1 + control2 | entity_id + time + industry^year,
    data = df, cluster = ~entity_id)
```

### 3d: 逐步加入 FE（展示识别来源变化）

```python
# Python: 逐步加入FE
res1 = pf.feols(f"{outcome} ~ {core_var} + {'+'.join(controls)} | {entity_id}", data=df)
res2 = pf.feols(f"{outcome} ~ {core_var} + {'+'.join(controls)} | {entity_id} + {time_id}", data=df)
res3 = pf.feols(f"{outcome} ~ {core_var} + {'+'.join(controls)} | {entity_id} + {time_id} + industry^year", data=df)
pf.etable([res1, res2, res3])
# 观察核心系数方向和显著性是否稳定
```

```r
# R: 逐步加入FE
res1 <- feols(Y ~ X + control1 | entity_id, df, cluster=~entity_id)
res2 <- feols(Y ~ X + control1 | entity_id + time, df, cluster=~entity_id)
res3 <- feols(Y ~ X + control1 | entity_id + time + industry^year, df, cluster=~entity_id)
etable(res1, res2, res3, fitstat = ~ r2_within + n)
```

### 3e: RE 模型估计

```python
# Python: RE（linearmodels）
from linearmodels.panel import RandomEffects
import statsmodels.api as sm

df_panel = df.set_index([entity_id, time_id])
exog = sm.add_constant(df_panel[[core_var] + controls])
res_re = RandomEffects(df_panel[outcome], exog).fit()
print(res_re.summary)
```

```r
# R: RE（plm）
res_re <- plm(Y ~ X + control1 + control2,
              data = df, index = c("entity_id", "time"), model = "random")
summary(res_re)
```

> **RE 使用条件：** Hausman 不拒绝 H₀ + 理论上个体效应与 X 不相关 + 需要估计不随时间变化的变量。三者同时满足时才用 RE。

---

## Step 4: 标准误选择

| 场景 | 推荐标准误 | R代码 | Python代码 |
|------|-----------|-------|-----------| 
| 标准面板 | 聚类（个体层面） | `cluster = ~entity_id` | `vcov={entity_id: "hetero"}` |
| 处理在组层面 | 聚类（组层面） | `cluster = ~province` | `vcov={"CRV1": "province"}` |
| < 50个聚类 | Wild Bootstrap | `boottest(...)` | `wildboottest` |
| 截面相关（宏观） | Driscoll-Kraay | `vcovSCC(...)` | — |
| 时序相关 | Newey-West HAC | `vcovNW(...)` | — |

**聚类层级规则（Step 0c 的执行）：**
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
pbgtest(res_plm, order=1)
# p < 0.05 → 序列相关显著 → 需用 HAC SE 或 Driscoll-Kraay
```

### 5b: 截面相关（Pesaran CD）

```r
# R: Pesaran CD检验
pcdtest(res_plm, test="cd")
# p < 0.05 → 截面相关显著 → 用 Driscoll-Kraay SE；宏观面板尤其需要
```

### 5c: 组间异方差（Modified Wald）

```python
# Python: 组间异方差 Wald 检验
from scipy import stats

def wald_heteroskedasticity(df, outcome, core_var, controls,
                             entity_id='entity_id', time_id='time'):
    """Modified Wald test: H0 = 所有个体的残差方差相同"""
    import pyfixest as pf
    fml = f"{outcome} ~ {core_var} + {'+'.join(controls)} | {entity_id} + {time_id}"
    res = pf.feols(fml, data=df)
    df_res = df.copy()
    df_res['_resid'] = res.resid()
    group_var = df_res.groupby(entity_id)['_resid'].var()
    sigma2_pool = df_res['_resid'].var()
    N = group_var.shape[0]
    W = N * ((group_var - sigma2_pool)**2).sum() / (2 * sigma2_pool**2)
    p_val = 1 - stats.chi2.cdf(W, N - 1)
    print(f"Modified Wald: W={W:.2f}, df={N-1}, p={p_val:.4f}")
    if p_val < 0.05:
        print("⚠️ 异方差显著 → 必须使用聚类稳健标准误或 HC 标准误")
    return {'W': W, 'p_value': p_val}
```

```r
# R: Modified Wald（手写，plm无直接命令）
library(plm)
fe_res <- plm(Y ~ X + control1, data=df, index=c("entity_id","time"), model="within")
residuals_by_id <- tapply(residuals(fe_res), df$entity_id, var)
sigma2_pool <- var(residuals(fe_res))
N <- length(residuals_by_id)
W <- N * sum((residuals_by_id - sigma2_pool)^2) / (2 * sigma2_pool^2)
pchisq(W, df = N-1, lower.tail = FALSE)
# p < 0.05 → 异方差显著
```

### 5d: 面板单位根（长面板 T>10 必做）

```r
# R: Im-Pesaran-Shin 单位根检验
library(plm)
purtest(Y ~ 1, data=df, index=c("entity_id","time"),
        pmax=2, test="ips", exo="trend")
# H0: 所有个体均有单位根
# p < 0.05 → 拒绝（平稳），可直接用 FE
# p > 0.05 → 考虑一阶差分或跳转 LONG-PANEL.md
```

### 诊断结果 → 标准误升级路径

```
序列相关显著 + 截面相关不显著 → HAC / Newey-West
截面相关显著 → Driscoll-Kraay
异方差显著 → 聚类稳健SE（默认应已使用）
序列 + 截面都显著 → Driscoll-Kraay（同时处理两者）
```

---

## Step 6: Mundlak / Correlated RE

当需要保留不随时间变化的变量但 FE 会吸收时，在 RE 中加入个体均值控制。

```r
# R: Mundlak/CRE
library(plm); library(dplyr)

df <- df %>%
  group_by(entity_id) %>%
  mutate(
    X_mean       = mean(X, na.rm = TRUE),
    control1_mean = mean(control1, na.rm = TRUE)
  ) %>% ungroup()

re_mundlak <- plm(
  Y ~ X + control1 + time_invariant_var +
      X_mean + control1_mean,
  data = df, index = c("entity_id", "time"), model = "random")
summary(re_mundlak)

# Mundlak项联合显著性检验（等价 Hausman）
library(car)
linearHypothesis(re_mundlak, c("X_mean = 0", "control1_mean = 0"))
```

**Mundlak 优势：**
- 保留不随时间变化的变量（地区、性别等）
- X 系数 ≈ FE 的 within 效应
- Mundlak 项联合显著性 ≈ Hausman 检验

---

## Step 7: 面板可视化

### 7a: 个体轨迹图（Spaghetti Plot）

```python
# Python: 个体轨迹图
import matplotlib.pyplot as plt

def spaghetti_plot(df, outcome, entity_id='entity_id', time_id='time',
                   sample_n=30, seed=42):
    """随机抽取 N 个个体，绘制 Y 的时间轨迹"""
    np.random.seed(seed)
    ids = np.random.choice(df[entity_id].unique(), min(sample_n, df[entity_id].nunique()), replace=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    for eid in ids:
        sub = df[df[entity_id] == eid].sort_values(time_id)
        ax.plot(sub[time_id], sub[outcome], alpha=0.3, linewidth=0.8)
    ax.set_xlabel('Time'); ax.set_ylabel(outcome)
    ax.set_title('Individual Trajectories (Spaghetti Plot)')
    fig.savefig('output/panel_spaghetti.png', dpi=150, bbox_inches='tight')
    return fig
```

### 7b: Within Variation 散点图

```python
# Python: 去均值散点图（FE 识别的变异）
def within_scatter(df, outcome, x_var, entity_id='entity_id'):
    """去均值后的散点图，直观展示 FE 在估计什么"""
    df_w = df.copy()
    for v in [outcome, x_var]:
        df_w[f'{v}_within'] = df_w[v] - df_w.groupby(entity_id)[v].transform('mean')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(df[x_var], df[outcome], alpha=0.15, s=8)
    axes[0].set_title('Raw Data (OLS 看到的)')
    axes[0].set_xlabel(x_var); axes[0].set_ylabel(outcome)
    axes[1].scatter(df_w[f'{x_var}_within'], df_w[f'{outcome}_within'], alpha=0.15, s=8)
    axes[1].set_title('Within-Transformed (FE 看到的)')
    axes[1].set_xlabel(f'{x_var} (demeaned)'); axes[1].set_ylabel(f'{outcome} (demeaned)')
    fig.tight_layout()
    fig.savefig('output/panel_within_scatter.png', dpi=150, bbox_inches='tight')
    return fig
```

### 7c: 组均值趋势图

```python
# Python: 处理组 vs 对照组均值趋势
def group_trend_plot(df, outcome, group_var='treated', time_id='time'):
    """分组均值随时间的趋势（通往 DID 的桥梁）"""
    trend = df.groupby([time_id, group_var])[outcome].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    for g, sub in trend.groupby(group_var):
        label = f'Group {g}' if isinstance(g, int) else str(g)
        ax.plot(sub[time_id], sub[outcome], marker='o', label=label)
    ax.legend(); ax.set_xlabel('Time'); ax.set_ylabel(f'Mean {outcome}')
    ax.set_title('Group Mean Trends')
    fig.savefig('output/panel_group_trend.png', dpi=150, bbox_inches='tight')
    return fig
```

---

## Step 8: 稳健性检验模板

```python
# Python: 系统化稳健性检验
import pyfixest as pf

def robustness_battery(df, outcome, core_var, controls,
                        entity_id='entity_id', time_id='time'):
    """一次性跑多种规格，输出对比表"""
    ctrl_str = ' + '.join(controls) if controls else '1'
    results = {}

    # (1) 不同 FE 结构
    results['Entity FE'] = pf.feols(
        f"{outcome} ~ {core_var} + {ctrl_str} | {entity_id}", data=df)
    results['TWFE'] = pf.feols(
        f"{outcome} ~ {core_var} + {ctrl_str} | {entity_id} + {time_id}", data=df)

    # (2) 不同聚类层级（若数据有 province 列）
    if 'province' in df.columns:
        results['Cluster-Province'] = pf.feols(
            f"{outcome} ~ {core_var} + {ctrl_str} | {entity_id} + {time_id}",
            data=df, vcov={"CRV1": "province"})

    # (3) 缩尾处理
    df_w = df.copy()
    for v in [outcome, core_var]:
        lo, hi = df_w[v].quantile([0.01, 0.99])
        df_w[v] = df_w[v].clip(lo, hi)
    results['Winsorized 1%'] = pf.feols(
        f"{outcome} ~ {core_var} + {ctrl_str} | {entity_id} + {time_id}", data=df_w)

    pf.etable(list(results.values()), labels=list(results.keys()))
    return results
```

```r
# R: 系统化稳健性
library(fixest)
res_robust <- list(
  "Entity FE"  = feols(Y ~ X + ctrl | entity_id, df, cluster=~entity_id),
  "TWFE"       = feols(Y ~ X + ctrl | entity_id + time, df, cluster=~entity_id),
  "Cluster-Prov" = feols(Y ~ X + ctrl | entity_id + time, df, cluster=~province),
  "No Controls" = feols(Y ~ X | entity_id + time, df, cluster=~entity_id)
)
etable(res_robust, fitstat = ~ r2_within + n + G)
```

---

## 常见错误

> **错误1：不做 Step 0 直接跑 FE**
> FE 会吸收所有不随时间变化的变量。如果核心 X 的 within 变异极小，FE 下系数不显著不代表 X 无效，而是 FE 吃掉了变异。先看方差分解。

> **错误2：FE 后无法估计不随时间变化的变量**
> 个体 FE 会吸收性别、地区、行业代码等。若这些是研究核心，用 Mundlak/CRE（Step 6）。

> **错误3：TWFE 不等于已控制所有混淆**
> TWFE 只控制加性个体效应和时间效应，时变遗漏变量仍引起内生性。

> **错误4：聚类层级选错**
> 聚类到处理分配层级。过细低估 SE，过粗高估 SE。

> **错误5：忽略序列相关**
> 同一个体跨期误差项通常正相关。不处理会严重低估 SE（过度拒绝 H₀）。

> **错误6：报告 overall R²**
> FE 模型应报告 **within R²**，不要报告 overall R²（被 FE 虚高）。

> **错误7：短面板套长面板**
> T > 20–30 时，不检验单位根就跑 FE 可能得到伪回归结果。先跳 LONG-PANEL.md。

> **错误8：跳过 F 检验直接 Hausman**
> 如果个体效应本身不显著（F 检验不拒绝），Pooled OLS 就够了，Hausman 都不需要。

---

## Estimand 声明

| 方法 | Estimand | 声明要点 |
|------|----------|---------| 
| Entity FE | 组内变异识别的效应 | 效应由个体内时间变异识别；不随时间变化的变量被吸收 |
| TWFE | 加性 ATT（可能含负权重） | 处理时机异质时 TWFE 加权可能产生负权重，见 DID skill |
| RE | GLS 加权平均（within + between） | 须通过 Hausman 检验；报告个体效应假设 |
| Mundlak RE | within 效应（X 系数）+ between 效应 | 时不变变量系数来自 between 变异 |

**声明模板：**
> "本文采用双向固定效应（TWFE）估计，通过个体内跨期变异识别处理效应，控制了不随时间变化的个体异质性（个体FE）和共同时间趋势（年份FE）。Estimand为以组内变异加权的平均处理效应。标准误聚类到[处理分配层级]。"

---

## 输出规范

```r
# R: fixest etable输出
etable(res1, res2, res3, res4,
       title   = "Panel FE Results",
       headers = c("Entity FE", "+Controls", "TWFE", "+HDFE"),
       fitstat = ~ r2_within + n + G,
       tex     = FALSE)
```

**必须注明：**
1. 固定效应类型（个体FE、年份FE、行业×年份FE）
2. 标准误类型（Clustered at entity level）
3. Within R²、N、聚类数 G

```
output/
  panel_structure_check.txt     # 面板结构诊断（Step 0）
  panel_variance_decomp.csv     # 组内/组间方差分解
  panel_model_selection.txt     # F检验 + BP-LM + Hausman
  panel_main_results.csv        # 主回归结果表
  panel_diagnostics.txt         # 序列/截面相关/异方差检验
  panel_unit_root.txt           # 单位根检验（长面板）
  panel_robustness.csv          # 稳健性（不同SE/FE规格）
  panel_spaghetti.png           # 个体轨迹图
  panel_within_scatter.png      # Within散点图
  panel_group_trend.png         # 组均值趋势图
```
