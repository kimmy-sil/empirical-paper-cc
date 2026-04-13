# 离散因变量面板模型

> **加载条件：** 因变量 Y 为二元（0/1）、计数（非负整数）、有序离散或多分类变量。
> 连续 Y 的面板分析见 `SKILL.md`。

---

## 触发条件

```
Y 的类型？
│
├─ 二元（0/1）→ Panel Logit / Probit（本文件 Step 1–3）
│
├─ 计数（0, 1, 2, ...）→ Panel Poisson / NB（本文件 Step 4）
│
├─ 有序离散（如 1–5 评分）→ Ordered Panel（本文件 Step 5）
│
└─ 多分类（无序）→ Conditional Logit / Mixed Logit（本文件 Step 5）
```

---

## Step 1: 二元因变量 — FE Logit vs RE Logit/Probit

### 核心陷阱：Incidental Parameters Problem

线性 FE 中，个体 FE 数量随 N 增大不影响 β 的一致性。但非线性模型（Logit/Probit）中，每个 αᵢ 都是待估参数，当 T 固定而 N → ∞ 时，αᵢ 的估计不一致，**连带污染 β 的一致性**。

**后果：**
- FE Probit / FE 非线性最小二乘：β 不一致，**禁止使用**
- FE Logit（条件 Logit）：利用充分统计量消除 αᵢ → β 一致 ✓
- RE Probit：假设 αᵢ 与 X 不相关 → 如果假设成立，β 一致且可估计边际效应

### FE Logit（Conditional Logit，Chamberlain 1980）

```r
# R: Conditional Logit（survival 包的 clogit 或 fixest）
library(fixest)

# fixest 的 feglm 支持面板 FE Logit
res_fe_logit <- feglm(
  Y_binary ~ X + control1 + control2 | entity_id,
  data   = df,
  family = binomial(link = "logit"),
  cluster = ~entity_id
)
summary(res_fe_logit)
# 系数是 log-odds ratio，不是边际效应

# ⚠️ 只有组内有 Y 变化的个体（既有 0 又有 1）才被保留
# 若大量个体始终 Y=0 或 Y=1，样本会大幅缩减
n_vary <- sum(tapply(df$Y_binary, df$entity_id, function(x) length(unique(x)) > 1))
cat(sprintf("Y 有变化的个体: %d / %d (%.1f%%)\n",
            n_vary, length(unique(df$entity_id)),
            100 * n_vary / length(unique(df$entity_id))))
```

```python
# Python: FE Logit（statsmodels conditional logit）
import statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit

# 条件 Logit
exog = df[[core_var] + controls]
res_clogit = ConditionalLogit(
    endog  = df[outcome],
    exog   = exog,
    groups = df[entity_id]
).fit()
print(res_clogit.summary())
```

### RE Probit / RE Logit

```r
# R: RE Probit（glmer，lme4包）
library(lme4)

res_re_probit <- glmer(
  Y_binary ~ X + control1 + control2 + (1 | entity_id),
  data   = df,
  family = binomial(link = "probit")
)
summary(res_re_probit)
# 可以计算边际效应（AME）
# 可以估计不随时间变化的变量的系数
```

```python
# Python: RE Logit（statsmodels BinomialBayesMixedGLM 或用 R）
# Python 对面板 RE 非线性模型支持较弱，建议用 R
```

### FE Logit vs RE Probit 选择

```
FE Logit vs RE Probit？
│
├─ 核心关注：一致性（担心 αᵢ 与 X 相关）
│   → FE Logit（Conditional Logit）
│   → 代价：丢失始终 Y=0/1 的个体；无法估计时不变变量
│
├─ 核心关注：效率 + 边际效应 + 时不变变量
│   → RE Probit
│   → 风险：若 αᵢ 与 X 相关则不一致
│
└─ 折中：Mundlak RE Probit（Step 3）
    → 在 RE 中加入组均值，放松 αᵢ 与 X 不相关的假设
```

---

## Step 2: FE Logit 的边际效应

FE Logit 的系数是 log-odds ratio，需要额外步骤得到边际效应。

```r
# R: FE Logit 的 APE（Average Partial Effect）
# 方法1: 直接计算 APE（需要还原个体效应）
# FE Logit 的个体效应可通过迭代得到，但样本大时计算昂贵

# 方法2: Fernández-Val & Weidner (2016) 偏差校正 APE
# install.packages("bife")
library(bife)

res_bife <- bife(Y_binary ~ X + control1 + control2 | entity_id, data = df)
summary(res_bife)
# bife 直接输出偏差校正后的 APE
apes_bife <- get_APEs(res_bife)
summary(apes_bife)
```

---

## Step 3: Mundlak / Correlated RE Probit

放松 RE 的核心假设（αᵢ 与 X 不相关），同时保留边际效应和时不变变量。

```r
# R: CRE Probit
library(lme4); library(dplyr)

# 计算组均值
df <- df %>%
  group_by(entity_id) %>%
  mutate(X_mean       = mean(X, na.rm = TRUE),
         control1_mean = mean(control1, na.rm = TRUE)) %>%
  ungroup()

res_cre_probit <- glmer(
  Y_binary ~ X + control1 + time_invariant_var +
    X_mean + control1_mean +           # Mundlak 项
    (1 | entity_id),
  data   = df,
  family = binomial(link = "probit")
)
summary(res_cre_probit)

# X 系数 ≈ FE Logit 的效应方向
# time_invariant_var 可被估计
# Mundlak 项联合检验 → 等价 Hausman
```

```python
# Python: CRE Probit（statsmodels）
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable

# 加入组均值
for v in [core_var] + controls:
    df[f'{v}_mean'] = df.groupby(entity_id)[v].transform('mean')

mean_vars = [f'{v}_mean' for v in [core_var] + controls]
exog = sm.add_constant(df[[core_var] + controls + mean_vars])

res_gee = GEE(
    endog       = df[outcome],
    exog        = exog,
    groups      = df[entity_id],
    family      = Binomial(),
    cov_struct  = Exchangeable()
).fit()
print(res_gee.summary())
```

---

## Step 4: 计数因变量 — Panel Poisson / Negative Binomial

### FE Poisson

```r
# R: FE Poisson（fixest）
library(fixest)

res_fe_poisson <- fepois(
  Y_count ~ X + control1 + control2 | entity_id + time,
  data    = df,
  cluster = ~entity_id
)
summary(res_fe_poisson)
# 系数解读：exp(β) = incidence rate ratio (IRR)
cat(sprintf("IRR of X: %.4f (X每增1单位，Y计数变化%.1f%%)\n",
            exp(coef(res_fe_poisson)["X"]),
            (exp(coef(res_fe_poisson)["X"]) - 1) * 100))
```

```python
# Python: FE Poisson（pyfixest）
import pyfixest as pf

res_pois = pf.fepois(
    f"{outcome} ~ {core_var} + {'+'.join(controls)} | {entity_id} + {time_id}",
    data = df
)
pf.etable([res_pois])
```

### RE Negative Binomial（过度离散时）

```r
# R: RE Negative Binomial（lme4 不支持 NB，用 glmmTMB）
# install.packages("glmmTMB")
library(glmmTMB)

res_re_nb <- glmmTMB(
  Y_count ~ X + control1 + control2 + (1 | entity_id),
  data   = df,
  family = nbinom2  # NB2 参数化
)
summary(res_re_nb)
```

### Poisson vs NB 选择

```
过度离散检验：Var(Y) >> E(Y)？
│
├─ Var/Mean ≈ 1 → Poisson 合适
│
├─ Var/Mean >> 1 → 过度离散
│   ├─ FE: 依然可用 FE Poisson（Poisson PML 对过度离散稳健）
│   └─ RE: 用 RE Negative Binomial
│
└─ 大量零值（zero-inflation）→ Zero-Inflated Poisson/NB
    → glmmTMB(... , ziformula = ~1, family = nbinom2)
```

---

## Step 5: 有序离散与多分类

### 有序面板模型（Ordered Logit / Probit）

```r
# R: RE Ordered Probit（ordinal包）
# install.packages("ordinal")
library(ordinal)

df$Y_ordered <- factor(df$Y_ordered, ordered = TRUE)
res_ord <- clmm(
  Y_ordered ~ X + control1 + control2 + (1 | entity_id),
  data = df,
  link = "probit"
)
summary(res_ord)
```

### 多分类面板模型（Conditional Logit）

```r
# R: Mixed Logit / Conditional Logit for panel choice data
# install.packages("mlogit")
library(mlogit)

df_ml <- dfidx(df, choice = "chosen", idx = list("entity_id", "alternative_id"))
res_mlogit <- mlogit(
  chosen ~ price + quality | income,  # | 后为个体特征
  data = df_ml
)
summary(res_mlogit)
```

---

## 检验清单

| 步骤 | 检验 | 通过标准 |
|------|------|---------| 
| Y 类型 | 连续 / 二元 / 计数 / 有序 | 决定模型族 |
| FE 样本缩减 | Y 有变化的个体比例 | < 50% → 考虑 RE 或 CRE |
| 过度离散 | Var(Y)/E(Y) | > 2 → NB 替代 Poisson |
| Mundlak 项 | 联合显著性 | 显著 → αᵢ 与 X 相关 |

---

## 常见错误

> **错误1：跑 FE Probit**
> 非线性模型中 FE Probit 有 incidental parameters problem，β 不一致。只有 FE Logit（条件 Logit）通过充分统计量消除了这个问题。

> **错误2：FE Logit 报告边际效应时直接用系数**
> FE Logit 系数是 log-odds ratio，不是概率变化。需要额外计算 APE（用 bife 包）。

> **错误3：忽略 FE Logit 的样本缩减**
> 始终 Y=0 或 Y=1 的个体会被丢弃。如果这些个体占比很高，结果只代表"有变化的子样本"，外部效度存疑。

> **错误4：计数模型不报告 IRR**
> Poisson 系数是 log-IRR，直接报告系数不直观。须报告 exp(β) 并解释为"X 每增1单位，Y 计数变化 (exp(β)-1)×100%"。

> **错误5：零膨胀不处理**
> 如果 Y 有大量结构性零（如企业专利数，大多数企业为0），标准 Poisson 会低估零值概率。用 Zero-Inflated 模型。

---

## Estimand 声明

| 方法 | Estimand | 声明要点 |
|------|----------|---------| 
| FE Logit | 条件 odds ratio | 由组内变异识别；丢弃无变化个体 |
| RE Probit | 总体 odds ratio / 边际效应 | 假设 αᵢ ⊥ X；可估计时不变变量 |
| CRE Probit | 条件边际效应（放松 RE 假设） | Mundlak 项控制 αᵢ 与 X 的相关 |
| FE Poisson | Incidence Rate Ratio | 对过度离散稳健（Poisson PML） |

**声明模板（二元 Y）：**
> "本文使用条件固定效应 Logit 模型（Chamberlain, 1980）估计 X 对 Y 的影响。该方法通过充分统计量消除个体异质性，避免 incidental parameters 偏误。样本限于 Y 在观测期内存在变化的个体（N_eff=[有效个体数]，占总样本[比例]%）。报告 odds ratio 及偏差校正后的平均偏效应（APE）。"
