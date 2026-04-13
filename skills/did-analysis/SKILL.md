---
name: did-analysis
description: >
  双重差分法（DID）分析。标准 2×2 DID 与交错 DID（Staggered）。
  TWFE 基准回归 + 自动 Bacon 分解（负权重三档决策）。
  事件研究图（平行趋势检验）+ HonestDiD 敏感性分析。
  Callaway-Sant'Anna / Sun-Abraham 稳健估计量。DDD 三重差分。
  安慰剂检验（空间 500 次置换 + 时间安慰剂）。SUTVA 缓冲区检验。
  高级工具见 ADVANCED.md：Stacked DID / BJS / dCDH / Synthdid / fect /
  连续处理 Intensity DID / 交错 DDD。
  语言：Python（pyfixest 为主，linearmodels 为辅）、R（fixest / did / bacondecomp）。
---

# DID 分析（双重差分法）

**适用场景：** 标准 2×2 DID；交错 DID（Staggered）；DDD 三重差分。

**语言：** Python（pyfixest）+ R（fixest / did / bacondecomp）。

---

## 前置条件

```
面板数据（长格式）必须包含：
  unit_id    : 个体编码
  time       : 时间（年/季/月）
  outcome    : 结果变量 Y
  treated    : 是否受处理（0/1，不随时间变化）
  post       : 处理后时期虚拟变量（标准 2×2）
  first_treat: 首次处理时间（交错 DID；从未处理 → 0 或 Inf）
```

**关键假设：** 平行趋势（事件研究检验）、无预期效应（pre 系数不显著）、SUTVA（无溢出）、无同期干扰政策。

**注：** 加入双向 FE 后，Treat_i 被个体 FE 吸收，Post_t 被时间 FE 吸收，交互项 Treat×Post 的系数即为 ATT。

---

## DID 变体路由

| 场景特征 | 典型描述 | DID 变体 | 跳转 |
|----------|---------|----------|------|
| 一次性政策、两组两期 | "减税试点城市 vs 非试点" | 标准 2×2 DID | Step 2 |
| 一次性政策、多期面板 | "开发区设立对制造业升级" | TWFE（面板 DID）| Step 2→3 |
| 分批推行政策 | "各省分批实施碳交易" | 交错 DID（CS/SA）| Step 2→4 |
| 处理组选择偏差 | "碳交易试点企业 vs 非试点" | PSM-DID | → matching skill + Step 2 |
| 政策针对子群体 / 溢出 | "医保只针对 B 组，溢出 A 组" | DDD 三重差分 | Step 4b |
| 处理强度连续 | "不同地区补贴金额不同" | Intensity DID | ADVANCED.md |

**PSM-DID：** 先调用 `matching` skill 做倾向得分匹配获得匹配后样本，再回到本 skill Step 2 跑 TWFE。

---

## Step 1：数据诊断

```python
# Python
import pandas as pd, numpy as np

def diagnose_did(df, unit_id='unit_id', time='time',
                 treatment='treated', first_treat=None):
    treat_vals = df[treatment].dropna().unique()
    if set(treat_vals).issubset({0, 1}):
        print("[OK] 二元处理（0/1），适合标准 DID。")
    else:
        print("[WARNING] 非二元处理 → 连续处理，见 ADVANCED.md Intensity DID。")

    obs_per_unit = df.groupby(unit_id)[time].count()
    is_balanced  = obs_per_unit.nunique() == 1
    print(f"面板平衡性：{'平衡' if is_balanced else '非平衡'}")

    if first_treat is not None:
        cohort = (df.drop_duplicates(subset=[unit_id])
                    .groupby(first_treat)[unit_id].count().reset_index())
        print("Cohort 结构："); print(cohort.to_string(index=False))
        n_never = df[df[first_treat].isin([0, np.inf])][unit_id].nunique()
        if n_never == 0:
            print("[WARNING] 无从未处理单位 → CS 以最晚处理组为对照，效力下降。")
    return {'balanced': is_balanced}
```

```r
# R
library(dplyr)
diagnose_did <- function(df, unit_id="unit_id", time="time",
                          treatment="treated", first_treat=NULL) {
  tv <- unique(na.omit(df[[treatment]]))
  if (!all(tv %in% c(0,1))) cat("[WARNING] 非二元处理。\n") else cat("[OK] 二元处理。\n")
  obs <- df %>% count(.data[[unit_id]]) %>% pull(n)
  cat(sprintf("面板平衡性：%s\n", ifelse(length(unique(obs))==1,"平衡","非平衡")))
  if (!is.null(first_treat)) {
    df %>% distinct(.data[[unit_id]], .data[[first_treat]]) %>%
      count(.data[[first_treat]], name="n_units") %>% print()
  }
}
```

---

## Step 2：TWFE 基准回归 + Bacon 分解

**规则：** TWFE 跑完后 **自动** 运行 Bacon 分解，根据负权重比例决定后续策略。

```python
# Python — pyfixest
import pyfixest as pf

def run_twfe(df, outcome, did_var, controls=None,
             unit_id='unit_id', time='time'):
    controls = controls or []
    ctrl = (" + " + " + ".join(controls)) if controls else ""
    fml = f"{outcome} ~ {did_var}{ctrl} | {unit_id} + {time}"
    res = pf.feols(fml, data=df, vcov={"CRV1": unit_id})
    pf.etable(res)
    print("[提示] 交错 DID 须运行 Bacon 分解（见 R 代码）。")
    return res
```

```r
# R — fixest + bacondecomp
library(fixest); library(bacondecomp); library(ggplot2)

run_twfe_bacon <- function(df, outcome, did_var, controls=NULL,
                            unit_id="unit_id", time="time", treatment="treated") {
  ctrl_str <- if (length(controls)>0) paste("+",paste(controls,collapse="+")) else ""
  fml <- as.formula(sprintf("%s ~ %s %s | %s + %s",
                             outcome, did_var, ctrl_str, unit_id, time))
  res <- feols(fml, data=df, cluster=as.formula(paste0("~",unit_id)))
  etable(res)

  # Bacon 分解
  bacon_out <- bacon(as.formula(paste(outcome,"~",treatment)),
                     data=df, id_var=unit_id, time_var=time)
  neg_wt <- with(bacon_out,
    sum(abs(weight[type=="Later vs Always Treated"])) / sum(abs(weight)))
  cat(sprintf("负权重比例：%.1f%%\n", neg_wt*100))

  # 三档决策
  if      (neg_wt == 0)   cat("[决策] 负权重=0 → TWFE 可信。\n")
  else if (neg_wt < 0.10) cat("[决策] <10% → TWFE 为主，补充 CS/SA。\n")
  else                     cat("[决策] ≥10% → 切换 CS/SA 为主，TWFE 降附录。\n")

  p <- ggplot(bacon_out, aes(x=weight, y=estimate, color=type)) +
    geom_point(size=3, alpha=0.8) + geom_hline(yintercept=0, lty=2) +
    labs(x="权重", y="系数", title="Bacon 分解") + theme_minimal()
  ggsave("output/bacon_decomposition.png", p, width=8, height=5, dpi=150)
  return(list(twfe=res, bacon=bacon_out, neg_wt=neg_wt))
}
```

| 负权重比例 | 主策略 |
|------------|--------|
| = 0 | TWFE 可信 |
| < 10% | TWFE 为主 + CS/SA 稳健性 |
| ≥ 10% | 切换 CS/SA 为主回归 |

---

## Step 3：事件研究图（平行趋势检验）

**注意：** pre-period 系数不显著 ≠ 平行趋势假设成立，建议配合 HonestDiD（Step 6）。

```python
# Python — pyfixest（3 行核心代码）
import pyfixest as pf

def event_study(df, outcome, unit_id='unit_id', time='time',
                treatment='treated', ref=-1, controls=None):
    controls = controls or []
    ctrl = (" + " + " + ".join(controls)) if controls else ""
    fml = f"{outcome} ~ i(event_time, {treatment}, ref={ref}){ctrl} | {unit_id} + {time}"
    res = pf.feols(fml, data=df, vcov={"CRV1": unit_id})
    pf.iplot(res, alpha=0.05, figsize=(9,5),
             title="事件研究图（平行趋势检验）")
    return res
```

```r
# R — fixest i() + iplot
library(fixest)

run_event_study <- function(df, outcome, unit_id="unit_id", time="time",
                             treatment="treated", ref=-1, controls=NULL) {
  ctrl_str <- if (length(controls)>0) paste("+",paste(controls,collapse="+")) else ""
  fml <- as.formula(sprintf("%s ~ i(event_time,%s,ref=%d) %s | %s + %s",
                             outcome, treatment, ref, ctrl_str, unit_id, time))
  res <- feols(fml, data=df, cluster=as.formula(paste0("~",unit_id)))
  iplot(res, main="事件研究图", xlab="事件时间", ylab="ATT", col="#20808D")
  abline(h=0, lty=2)

  # Pre-period 联合 F 检验
  pre <- grep("event_time::-[2-9]", names(coef(res)), value=TRUE)
  if (length(pre)>0) {
    wt <- wald(res, keep=pre)
    cat(sprintf("Pre-period 联合 F=%.3f, p=%.4f  通过: %s\n",
                wt$stat, wt$p, wt$p>0.10))
  }
  return(res)
}
```

---

## Step 4：交错 DID 稳健估计量

### 4a Callaway-Sant'Anna (CS)

```python
# Python — diff-diff 包
# pip install diff-diff
from diff_diff import ATTgt

def run_cs_python(df, outcome, unit_id='unit_id', time='time',
                   first_treat='first_treat', controls=None):
    controls = controls or []
    att = ATTgt(data=df, cohort_name=first_treat, period_name=time,
                unit_name=unit_id, outcome_name=outcome,
                control_group='nevertreated', estimation_method='dr',
                xformla=controls if controls else None)
    att.fit()
    att.aggregate('simple')
    att.aggregate('dynamic', plot=True)
    return att
```

```r
# R — did 包
library(did)

run_cs <- function(df, outcome, unit_id="unit_id", time="time",
                   first_treat="first_treat", controls=NULL) {
  out <- att_gt(yname=outcome, gname=first_treat, idname=unit_id, tname=time,
                xformla=if (!is.null(controls))
                  as.formula(paste("~",paste(controls,collapse="+"))) else ~1,
                data=df, est_method="dr", control_group="nevertreated")
  agg <- aggte(out, type="simple")
  cat(sprintf("CS ATT: %.4f  SE: %.4f\n", agg$overall.att, agg$overall.se))
  agg_dyn <- aggte(out, type="dynamic")
  ggdid(agg_dyn); ggsave("output/cs_dynamic.png", width=8, height=5, dpi=150)
  return(list(att_gt=out, simple=agg, dynamic=agg_dyn))
}
```

### 4b Sun-Abraham (SA)

```python
# Python — pyfixest sunab
def run_sa_python(df, outcome, unit_id='unit_id', time='time',
                   first_treat='first_treat', controls=None):
    controls = controls or []
    ctrl = (" + " + " + ".join(controls)) if controls else ""
    fml = f"{outcome} ~ sunab({first_treat}, {time}){ctrl} | {unit_id} + {time}"
    res = pf.feols(fml, data=df, vcov={"CRV1": unit_id})
    pf.etable(res); pf.iplot(res)
    return res
```

```r
# R — fixest sunab
run_sa <- function(df, outcome, unit_id="unit_id", time="time",
                   first_treat="first_treat", controls=NULL) {
  ctrl_str <- if (length(controls)>0) paste("+",paste(controls,collapse="+")) else ""
  res <- feols(as.formula(sprintf("%s ~ sunab(%s,%s) %s | %s + %s",
               outcome, first_treat, time, ctrl_str, unit_id, time)),
               data=df, cluster=as.formula(paste0("~",unit_id)))
  iplot(res, main="SA 动态效应"); return(res)
}
```

---

## Step 4c：DDD（三重差分）

**何时使用：**
- 政策针对特定子群体（B 组），但组内对照有溢出，跨组对照违反平行趋势
- DDD 不需要两个平行趋势同时成立，只需两个 DID 的偏差相同可相减抵消

**Estimand：** β₇ = (DID_B组) − (DID_A组)  (Olden & Moen, 2022)

```python
# Python — diff-diff TripleDifference (Ortiz-Villavicencio & Sant'Anna 2025)
# pip install diff-diff
from diff_diff import TripleDifference

def run_ddd_python(df, outcome, group='treated', partition='target_group',
                    time_var='post', controls=None):
    """
    group     : 处理州=1 / 对照州=0 (T)
    partition : 政策针对群体=1 / 非针对=0 (B)
    time_var  : 政策后=1 / 政策前=0 (Post)
    """
    ddd = TripleDifference(estimation_method='dr')  # 双稳健
    results = ddd.fit(df, outcome=outcome, group=group,
                      partition=partition, time=time_var,
                      covariates=controls)
    return results
```

```r
# R — fixest 手动三重交互（非交错场景）
library(fixest)

run_ddd <- function(df, outcome, treat="treated", group="target_group",
                     post="post", controls=NULL, unit_id="unit_id", time="time") {
  # 完整三重交互 + 所有低阶交互
  ctrl_str <- if (length(controls)>0) paste("+",paste(controls,collapse="+")) else ""
  fml <- as.formula(sprintf(
    "%s ~ %s:%s:%s + %s:%s + %s:%s + %s:%s %s | %s + %s",
    outcome, treat, group, post,
    treat, group, treat, post, group, post,
    ctrl_str, unit_id, time))
  res <- feols(fml, data=df, cluster=as.formula(paste0("~",unit_id)))
  etable(res); return(res)
}

# R — triplediff 包（交错 DDD，Sant'Anna 官方）
# install.packages("triplediff")
library(triplediff)

run_ddd_staggered <- function(df, outcome, unit_id="unit_id", time="time",
                                gname="first_treat", pname="target_group",
                                controls=NULL) {
  out <- ddd(yname=outcome, tname=time, idname=unit_id,
             gname=gname, pname=pname,
             xformla=if (!is.null(controls))
               as.formula(paste("~",paste(controls,collapse="+"))) else ~1,
             data=df, control_group="nevertreated", est_method="dr")
  return(out)
}
```

---

## Step 5：安慰剂检验

### 5a 空间安慰剂（500 次置换）

**关键：** 置换在 unit 级别进行。经验 p 值 = |假系数| ≥ |真实系数| 的比例。

```python
# Python — pyfixest
import numpy as np, matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def spatial_placebo(df, outcome, did_var, unit_id='unit_id', time='time',
                    controls=None, n_perm=500, seed=42):
    controls = controls or []
    ctrl = (" + " + " + ".join([did_var]+controls)) if controls else did_var
    fml = f"{outcome} ~ {ctrl} | {unit_id} + {time}"
    res_true = pf.feols(fml, data=df, vcov={"CRV1": unit_id})
    true_coef = res_true.coef()[did_var]

    np.random.seed(seed)
    units = df[unit_id].unique()
    treat_map = df.drop_duplicates(unit_id).set_index(unit_id)['treated']
    fake_coefs = []

    for _ in range(n_perm):
        df_f = df.copy()
        perm = np.random.permutation(treat_map.values)
        df_f['_fake'] = df_f[unit_id].map(dict(zip(units, perm)))
        df_f['_fake_did'] = df_f['_fake'] * df_f['post']
        fml_f = f"{outcome} ~ _fake_did | {unit_id} + {time}"
        try:
            rf = pf.feols(fml_f, data=df_f, vcov={"CRV1": unit_id})
            fake_coefs.append(rf.coef()['_fake_did'])
        except: pass

    fake_coefs = np.array(fake_coefs)
    p_val = np.mean(np.abs(fake_coefs) >= np.abs(true_coef))
    print(f"真实系数：{true_coef:.4f}  经验 p={p_val:.4f}  (n={len(fake_coefs)})")

    fig, ax = plt.subplots(figsize=(8,5))
    kde = gaussian_kde(fake_coefs); x = np.linspace(fake_coefs.min(), fake_coefs.max(), 200)
    ax.fill_between(x, kde(x), alpha=0.35, color='#20808D', label='安慰剂分布')
    ax.axvline(true_coef, color='#A84B2F', lw=2, ls='--', label=f'真实={true_coef:.4f}')
    ax.set(xlabel='DID 系数', ylabel='核密度',
           title=f'空间安慰剂 (n={len(fake_coefs)})  p={p_val:.4f}')
    ax.legend(); plt.tight_layout()
    plt.savefig('output/placebo_spatial.png', dpi=150); return {'p': p_val}
```

### 5b 时间安慰剂

**关键：** 严格去掉处理后样本（post==0 子样本），用处理前各期做假处理时间。

```r
# R
time_placebo <- function(df, outcome, unit_id="unit_id", time="time",
                          treatment="treated", controls=NULL, n_fake=5) {
  ctrl_str <- if (length(controls)>0) paste("+",paste(controls,collapse="+")) else ""
  df_pre <- df[df$post == 0, ]
  fake_years <- tail(sort(unique(df_pre[[time]]))[-c(1,2)], n_fake)
  results <- lapply(fake_years, function(fy) {
    df_pre$fake_post <- as.integer(df_pre[[time]] >= fy)
    df_pre$fake_did  <- df_pre$fake_post * df_pre[[treatment]]
    res <- tryCatch(
      feols(as.formula(sprintf("%s ~ fake_did %s | %s + %s",
            outcome, ctrl_str, unit_id, time)),
            data=df_pre, cluster=as.formula(paste0("~",unit_id))),
      error=function(e) NULL)
    if (!is.null(res))
      data.frame(fake_year=fy, coef=coef(res)["fake_did"],
                 se=se(res)["fake_did"], pval=pvalue(res)["fake_did"])
  })
  res_df <- do.call(rbind, Filter(Negate(is.null), results))
  print(res_df); cat("所有假处理期系数应不显著（p > 0.10）。\n")
  return(res_df)
}
```

---

## Step 6：HonestDiD 敏感性分析

默认 Relative Magnitudes（RM）。Mbar=1 时 CI 不含零 → 结论稳健。

```r
# R
library(HonestDiD)

run_honest_did <- function(res_es, Mbarvec=seq(0,2,0.5)) {
  cf <- coef(res_es)[grep("event_time", names(coef(res_es)))]
  vc <- vcov(res_es)[grep("event_time",rownames(vcov(res_es))),
                      grep("event_time",colnames(vcov(res_es)))]
  t_vals <- as.numeric(gsub(".*::(-?[0-9]+).*","\\1",names(cf)))
  pre <- which(t_vals < 0); pst <- which(t_vals > 0)

  hd <- createSensitivityResults_relativeMagnitudes(
    betahat=cf[pst], sigma=vc[pst,pst],
    numPrePeriods=length(pre), numPostPeriods=length(pst),
    Mbarvec=Mbarvec, alpha=0.05)
  p <- createSensitivityPlot_relativeMagnitudes(hd, alpha=0.05)
  ggsave("output/honest_did.png", p, width=8, height=5, dpi=150)

  m1 <- hd[hd$Mbar==1,]
  if (nrow(m1)>0) cat(sprintf("Mbar=1 CI: [%.4f, %.4f]  稳健: %s\n",
                                m1$lb[1], m1$ub[1], m1$lb[1]>0 | m1$ub[1]<0))
  return(hd)
}
```

---

## 检验清单

### 必做

| 检验 | 步骤 | 判断标准 |
|------|------|----------|
| 事件研究图 | Step 3 | Pre-period 联合 p > 0.10 |
| HonestDiD | Step 6 | Mbar=1 时 CI 不含零 |
| 安慰剂 500 次 | Step 5a | 经验 p < 0.05 |
| Bacon 分解 | Step 2 | 记录负权重比例 + 决策 |

### 推荐

- 替换结果变量代理指标
- 缩短事件窗口（±2 期）
- 排除同期政策（文献 + 数据双验证）
- 异质性检验（行业 / 规模 / 地区分组）
- 仅用平衡子样本重跑

---

## 常见错误（8 条）

| # | 错误 | 正确做法 |
|---|------|----------|
| 1 | 交错 DID 盲目用 TWFE | 先 Bacon 分解；负权重 ≥10% → CS/SA |
| 2 | 聚类层级低于处理层级 | 聚类层级 = 处理分配层级 |
| 3 | 安慰剂未去掉处理后样本 | 严格限制于 post==0 子样本 |
| 4 | 事件研究参照期选错 | ref = -1（处理前最后一期）|
| 5 | pre 不显著 = 平行趋势成立 | 配合 HonestDiD 报告 Mbar |
| 6 | 只看 TWFE 系数忽视 Bacon | 明确报告权重分布 |
| 7 | 连续处理直接套 TWFE | 先离散化或用 contdid 包（见 ADVANCED.md）|
| 8 | 事件研究窗口仅 ±1 期 | 至少 ±3 期 |

---

## Estimand 声明

| 方法 | Estimand | 必须声明 |
|------|----------|---------| 
| **TWFE** | Weighted ATT（可能含负权重）| 负权重比例（Bacon）；> 0 须声明 |
| **CS** | Group-time ATT，聚合为整体 ATT | 聚合权重类型；从未处理对照合理性 |
| **SA** | Interaction-weighted ATT | 线性假设下等价于 CS |
| **DDD** | β₇ = DID_B − DID_A | "偏差相同"假设的合理性论证 |

**声明模板：**
```
本文 TWFE 估计量识别 Weighted ATT。根据 Bacon (2021) 分解，
负权重比例为 [Y]%。鉴于负权重 [低于/高于] 10%，本文 [以 TWFE 为主 /
切换 CS 为主]。CS 识别 Group-time ATT，以 simple 加权聚合。
```

> 高级工具见 ADVANCED.md。

---

## 输出文件

```
output/
  bacon_decomposition.png
  event_study.png
  cs_dynamic.png / sa_dynamic.png
  placebo_spatial.png
  honest_did.png
  did_main_results.csv
  did_diagnostics.txt
  did_robustness.csv
```
