---
name: unbalanced-panel
description: >
  非平衡面板诊断与处理。用于个体观测期数不一致、样本流失、
  晚进入、间隔缺失等场景。覆盖缺失模式诊断、MCAR/MAR/MNAR
  判别、流失回归、IPW 修正、Heckman 选择模型、平衡子样本稳健性比较。
---

# 非平衡面板诊断与处理

> **加载条件：** Step 0 判定面板为非平衡（个体观测期数不一致），或已知存在样本流失/进入退出。  
> 非平衡面板的核心风险不是“能不能跑”，而是“缺失是不是随机的”。

---

## 触发条件

```text
面板是否平衡？
│
├─ 平衡（所有个体 T_i 相同）→ 不需要本文件
│
└─ 非平衡
    │
    ├─ 轻微不平衡（> 90% 个体完整）→ Step 1 快速诊断后回主流程
    │
    ├─ 中度不平衡（50–90% 完整）→ Step 1–3 完整诊断
    │
    └─ 严重不平衡（< 50% 完整）→ 必须走 Step 1–5 完整流程
```

---

## Step 1: 缺失模式诊断

### 1a: 缺失概况统计

```python
# Python: 非平衡面板缺失诊断
import pandas as pd
import numpy as np

def diagnose_unbalanced(df, entity_id='entity_id', time_id='time'):
    """非平衡面板缺失概况"""
    obs_per = df.groupby(entity_id)[time_id].count()
    T_max   = df[time_id].nunique()
    N       = df[entity_id].nunique()

    n_complete   = (obs_per == T_max).sum()
    pct_complete = n_complete / N * 100
    n_obs_actual = len(df)
    n_obs_full   = N * T_max
    pct_obs      = n_obs_actual / n_obs_full * 100

    print(f"个体数 N = {N}, 最大期数 T = {T_max}")
    print(f"实际观测 {n_obs_actual} / 完全平衡 {n_obs_full} ({pct_obs:.1f}%)")
    print(f"完整个体: {n_complete} / {N} ({pct_complete:.1f}%)")
    print(f"每个体观测期数: min={obs_per.min()}, median={obs_per.median():.0f}, max={obs_per.max()}")

    if pct_complete > 90:
        print("→ 轻微不平衡，快速诊断后可回主流程")
    elif pct_complete > 50:
        print("→ 中度不平衡，建议完整诊断")
    else:
        print("⚠️ 严重不平衡，必须诊断缺失机制")

    return obs_per
```

```r
# R: 缺失概况
library(dplyr)

diagnose_unbalanced_r <- function(df, entity_id = "entity_id", time_id = "time") {
  obs_per <- df %>% count(.data[[entity_id]], name = "n_obs")
  T_max   <- n_distinct(df[[time_id]])
  N       <- nrow(obs_per)

  cat(sprintf("N=%d, T_max=%d, 完整个体: %d (%.1f%%)\n",
              N, T_max,
              sum(obs_per$n_obs == T_max),
              100 * mean(obs_per$n_obs == T_max)))
  cat(sprintf("观测期数: min=%d, median=%.0f, max=%d\n",
              min(obs_per$n_obs), median(obs_per$n_obs), max(obs_per$n_obs)))

  return(obs_per)
}
```

### 1b: 缺失模式可视化

```python
# Python: 缺失热力图
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def missing_heatmap(df, entity_id='entity_id', time_id='time',
                    sample_n=100, seed=42):
    """可视化哪些个体在哪些时期缺失"""
    np.random.seed(seed)
    all_times = sorted(df[time_id].unique())
    all_ids   = df[entity_id].unique()

    if len(all_ids) > sample_n:
        all_ids = np.random.choice(all_ids, sample_n, replace=False)

    presence = pd.DataFrame(0, index=all_ids, columns=all_times)
    for _, row in df[df[entity_id].isin(all_ids)].iterrows():
        presence.loc[row[entity_id], row[time_id]] = 1

    fig, ax = plt.subplots(figsize=(12, max(6, sample_n * 0.08)))
    ax.imshow(presence.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Entity')
    ax.set_title('Panel Presence (green=observed, red=missing)')
    ax.set_xticks(range(len(all_times)))
    ax.set_xticklabels(all_times, rotation=45, fontsize=7)

    fig.tight_layout()
    fig.savefig('output/panel_missing_heatmap.png', dpi=150, bbox_inches='tight')
    return fig
```

### 1c: 缺失模式分类

```python
# Python: 判断缺失是入口/退出/间隔缺失
def classify_missing_pattern(df, entity_id='entity_id', time_id='time'):
    """分类缺失模式：attrition / late entry / intermittent"""
    all_times = sorted(df[time_id].unique())
    T_max = len(all_times)
    time_rank = {t: i for i, t in enumerate(all_times)}

    patterns = {
        'complete': 0,
        'attrition': 0,
        'late_entry': 0,
        'intermittent': 0,
        'other': 0
    }

    for eid, grp in df.groupby(entity_id):
        obs_times = sorted(grp[time_id].unique())
        n_obs = len(obs_times)

        if n_obs == T_max:
            patterns['complete'] += 1
        else:
            ranks = [time_rank[t] for t in obs_times]
            is_consecutive = (max(ranks) - min(ranks) + 1) == n_obs

            if is_consecutive:
                if min(ranks) == 0:
                    patterns['attrition'] += 1
                elif max(ranks) == T_max - 1:
                    patterns['late_entry'] += 1
                else:
                    patterns['other'] += 1
            else:
                patterns['intermittent'] += 1

    total = sum(patterns.values())
    for k, v in patterns.items():
        print(f"  {k}: {v} ({v/total*100:.1f}%)")

    if patterns['attrition'] > 0.2 * total:
        print("⚠️ 样本流失（attrition）占比高 → 强烈建议做 Step 2 流失检验")
    if patterns['intermittent'] > 0.1 * total:
        print("⚠️ 间隔缺失占比高 → 可能是数据质量问题，检查原始数据")

    return patterns
```

---

## Step 2: 缺失机制检验

### 缺失机制三分类

| 类型 | 全称 | 含义 | 后果 |
|------|------|------|------|
| MCAR | Missing Completely At Random | 缺失与任何变量无关 | FE/RE 无偏 |
| MAR | Missing At Random | 缺失与可观测变量有关 | FE/RE 可能有偏，但可通过控制变量修正 |
| MNAR | Missing Not At Random | 缺失与不可观测因素有关 | FE/RE 有偏，需选择模型修正 |

### 2a: MCAR 检验（Little's Test 思路）

```python
# Python: 简化版 MCAR 检验
def test_mcar(df, entity_id='entity_id', time_id='time', test_vars=None):
    """比较完整个体 vs 不完整个体的基期特征"""
    import numpy as np
    from scipy import stats

    T_max = df[time_id].nunique()
    obs_per = df.groupby(entity_id)[time_id].count()
    complete_ids = obs_per[obs_per == T_max].index
    incomplete_ids = obs_per[obs_per < T_max].index

    t0 = df[time_id].min()
    df_t0 = df[df[time_id] == t0].copy()
    df_t0['_complete'] = df_t0[entity_id].isin(complete_ids).astype(int)

    if test_vars is None:
        test_vars = [
            c for c in df_t0.select_dtypes(include=[np.number]).columns
            if c not in [entity_id, time_id, '_complete']
        ]

    print("=== MCAR 检验：完整 vs 不完整个体的基期特征比较 ===")
    any_sig = False

    for v in test_vars:
        g1 = df_t0[df_t0['_complete'] == 1][v].dropna()
        g0 = df_t0[df_t0['_complete'] == 0][v].dropna()
        if len(g1) < 5 or len(g0) < 5:
            continue

        _, p_val = stats.ttest_ind(g1, g0, equal_var=False)
        sig = "⚠️ *" if p_val < 0.05 else ""
        if p_val < 0.05:
            any_sig = True

        print(f"  {v}: complete={g1.mean():.3f}, incomplete={g0.mean():.3f}, p={p_val:.4f} {sig}")

    if any_sig:
        print("\\n⚠️ 部分变量显著不同 → 缺失可能非随机（MAR 或 MNAR）")
        print("   → 建议做 Step 3 流失回归 或 Step 4 选择模型修正")
    else:
        print("\\n✓ 未发现显著差异 → 暂时支持 MCAR，可回主流程")

    return any_sig
```

### 2b: 流失回归（Attrition Probit）

```python
# Python: 流失回归
from statsmodels.discrete.discrete_model import Probit

def attrition_probit(df, entity_id='entity_id', time_id='time', predictors=None):
    """Probit: 退出 = f(可观测变量)"""
    import numpy as np
    import statsmodels.api as sm

    T_max = df[time_id].max()
    obs_per = df.groupby(entity_id)[time_id].max()
    attrited = (obs_per < T_max).astype(int)
    attrited.name = 'attrited'

    t0 = df[time_id].min()
    df_t0 = df[df[time_id] == t0].set_index(entity_id)
    df_t0 = df_t0.join(attrited)

    if predictors is None:
        predictors = [
            c for c in df_t0.select_dtypes(include=[np.number]).columns
            if c not in [time_id, 'attrited']
        ]

    exog = sm.add_constant(df_t0[predictors].dropna())
    endog = df_t0.loc[exog.index, 'attrited']

    res = Probit(endog, exog).fit(disp=0)
    print(res.summary2())
    return res
```

```r
# R: 流失回归
library(dplyr)

df_attrition <- df %>%
  group_by(entity_id) %>%
  summarise(last_obs = max(time), .groups = "drop") %>%
  mutate(attrited = as.integer(last_obs < max(df$time)))

df_t0 <- df %>%
  filter(time == min(time)) %>%
  left_join(df_attrition, by = "entity_id")

attrition_probit <- glm(
  attrited ~ Y + X + control1 + control2,
  data = df_t0,
  family = binomial(link = "probit")
)

summary(attrition_probit)
```

---

## Step 3: 缺失机制 → 处理方案路由

```text
Step 2 诊断结果：
│
├─ MCAR（完整 vs 不完整无差异）
│   → 直接回 SKILL.md，FE/RE 可安全使用
│   → 非平衡面板的 FE 估计在 MCAR 下无偏
│
├─ MAR（缺失可由可观测变量预测，但不由 Y 本身预测）
│   ├─ 加入预测流失的变量作为控制变量
│   ├─ IPW 加权（Step 4a）
│   └─ 报告缩减样本（仅完整个体）作为稳健性检验
│
└─ MNAR（缺失与 Y 或不可观测因素相关）
    ├─ Heckman 选择模型（Step 4b）
    ├─ 上下界分析（Lee Bounds / Horowitz-Manski）
    └─ ⚠️ 在论文中明确声明潜在选择偏误
```

---

## Step 4: 修正方法

### 4a: IPW 加权（Inverse Probability Weighting）

```python
import statsmodels.api as sm

def ipw_correction(df, outcome, core_var, controls,
                   entity_id='entity_id', time_id='time'):
    """用留存概率的逆作为权重"""
    import pyfixest as pf

    df = df.sort_values([entity_id, time_id])
    df['_next_obs'] = df.groupby(entity_id)[time_id].shift(-1)
    df['_survived'] = (~df['_next_obs'].isna()).astype(int)

    df_surv = df[df[time_id] < df[time_id].max()].copy()
    exog = sm.add_constant(df_surv[[core_var] + controls])
    surv_model = sm.Probit(df_surv['_survived'], exog).fit(disp=0)
    df_surv['_p_survive'] = surv_model.predict(exog)

    df_surv['_ipw'] = 1 / df_surv['_p_survive'].clip(0.05, 0.95)
    df_surv['_wt'] = df_surv['_ipw']

    res = pf.feols(
        f"{outcome} ~ {core_var} + {'+'.join(controls)} | {entity_id} + {time_id}",
        data=df_surv,
        weights='_wt'
    )
    pf.etable([res])
    return res
```

```r
library(fixest)
library(dplyr)

df <- df %>%
  group_by(entity_id) %>%
  mutate(survived = as.integer(!is.na(lead(time)))) %>%
  ungroup()

surv_mod <- glm(
  survived ~ Y + X + control1,
  data = df %>% filter(time < max(time)),
  family = binomial
)

df$p_survive <- predict(surv_mod, newdata = df, type = "response")
df$ipw <- 1 / pmax(df$p_survive, 0.05)

res_ipw <- feols(
  Y ~ X + control1 | entity_id + time,
  data = df,
  weights = ~ipw,
  cluster = ~entity_id
)

etable(res_ipw)
```

### 4b: Heckman 选择模型（面板版）

```r
library(plm)
library(sampleSelection)

df$in_sample <- 1

selection_eq <- in_sample ~ X + control1 + exclusion_var
outcome_eq   <- Y ~ X + control1 + control2

heckman_res <- selection(selection_eq, outcome_eq, data = df, method = "2step")
summary(heckman_res)
```

### 4c: 上下界分析（极端情况）

```python
def lee_bounds(df, outcome, core_var, controls,
               entity_id='entity_id', time_id='time'):
    """极端假设下的效应上下界"""
    import pyfixest as pf

    res_base = pf.feols(
        f"{outcome} ~ {core_var} + {'+'.join(controls)} | {entity_id} + {time_id}",
        data=df
    )
    base_coef = res_base.coef()[core_var]

    print(f"基准估计: β = {base_coef:.4f}")
    print("注：完整 Lee Bounds 需要根据具体缺失机制构造，此处为简化版")
    return base_coef
```

---

## Step 5: 稳健性——平衡子样本比较

```python
import pyfixest as pf

def balanced_subsample_check(df, outcome, core_var, controls,
                             entity_id='entity_id', time_id='time'):
    """对比全样本（非平衡）和平衡子样本的结果"""
    T_max = df[time_id].nunique()
    obs_per = df.groupby(entity_id)[time_id].count()
    balanced_ids = obs_per[obs_per == T_max].index
    df_balanced = df[df[entity_id].isin(balanced_ids)]

    fml = f"{outcome} ~ {core_var} + {'+'.join(controls)} | {entity_id} + {time_id}"
    res_full = pf.feols(fml, data=df)
    res_bal  = pf.feols(fml, data=df_balanced)

    pf.etable([res_full, res_bal], labels=['Full (Unbalanced)', 'Balanced Subsample'])

    coef_full = res_full.coef()[core_var]
    coef_bal  = res_bal.coef()[core_var]
    diff_pct = abs(coef_full - coef_bal) / abs(coef_full) * 100

    print(f"\\n系数差异: {diff_pct:.1f}%")
    if diff_pct > 20:
        print("⚠️ 差异 > 20%，选择偏误风险高，建议 IPW 或 Heckman 修正")

    return res_full, res_bal
```

```r
library(fixest)
library(dplyr)

balanced_ids <- df %>%
  count(entity_id) %>%
  filter(n == max(n)) %>%
  pull(entity_id)

res_full <- feols(Y ~ X + control1 | entity_id + time, data = df, cluster = ~entity_id)
res_bal  <- feols(Y ~ X + control1 | entity_id + time,
                  data = df %>% filter(entity_id %in% balanced_ids),
                  cluster = ~entity_id)

etable(res_full, res_bal, headers = c("Full", "Balanced"))
```

---

## 检验清单

| 步骤 | 检验 | 通过标准 |
|------|------|---------|
| Step 1a | 缺失概况 | 完整个体 > 90% → 轻微不平衡 |
| Step 1b | 缺失热力图 | 无系统性模式 |
| Step 1c | 缺失分类 | attrition < 20% |
| Step 2a | MCAR 检验 | 完整 vs 不完整无显著差异 |
| Step 2b | 流失回归 | Y 和核心 X 不显著预测流失 |
| Step 5 | 平衡子样本 | 系数差异 < 20% |

---

## 常见错误

> **错误 1：非平衡面板直接跑 FE 不诊断**  
> FE 在 MCAR 下无偏，但 MNAR 下有偏。不诊断就假设 MCAR 是赌博。

> **错误 2：删除不完整个体“制造”平衡面板**  
> 如果缺失是非随机的，删除不完整个体会加重选择偏误而非消除。应先诊断再决定。

> **错误 3：IPW 权重不截断**  
> 留存概率极小的个体会获得极大权重，导致方差爆炸。必须截断（如 [0.05, 0.95]）。

> **错误 4：Heckman 没有排他性约束**  
> 选择方程和结果方程用相同变量时，识别完全依赖函数形式假设，结果不可靠。至少需要一个排他性工具。

> **错误 5：只报告平衡子样本结果**  
> 平衡子样本是稳健性检验，不是基准。基准应是全样本结果（可能加 IPW 修正）。

---

## Estimand 声明

**声明模板：**

> “本文使用非平衡面板数据。流失回归（Attrition Probit）显示[结果变量/核心解释变量]不显著预测样本退出（p = [值]），支持缺失随机假设（MAR/MCAR）。主回归使用全样本 FE 估计，稳健性检验包括：(1) 限制为平衡子样本，系数变化[值]%；(2) IPW 加权修正，结果一致。”

---

## 输出规范

```text
output/
  panel_missing_summary.txt       # 缺失概况统计
  panel_missing_heatmap.png       # 缺失模式热力图
  panel_missing_pattern.txt       # 缺失分类（attrition / entry / intermittent）
  panel_mcar_test.txt             # MCAR 检验结果
  panel_attrition_probit.txt      # 流失回归
  panel_ipw_results.csv           # IPW 修正后结果
  panel_balanced_comparison.csv   # 全样本 vs 平衡子样本对比
```