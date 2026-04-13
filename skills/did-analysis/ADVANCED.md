# DID 高级工具

本文件在 SKILL.md 基础上提供高级估计量、DDD 详细说明、连续处理 DID、
多估计量比较、合成 DID 及矩阵补全等工具。**按需加载**，不作为默认流程。

**适用场景：** TWFE 负权重 ≥ 10%；需多估计量一致性；DDD / Intensity DID；顶刊审稿应对。

---

## DDD 详细说明

### 使用场景

**场景 1：组内溢出 + 跨组趋势差异**

研究某州医疗保健改革（仅针对 B 组人群）对健康支出的影响。
- 组内对照（A vs B）→ 医疗改革可能从 B 组溢出到 A 组 → DID 无效
- 跨州对照（处理州 B vs 对照州 B）→ 两州经济趋势不同 → 平行趋势不成立
- DDD 假设：经济差异不影响 A/B 组间的**相对差别** → 用相对差别估计反事实

**场景 2：处理非平行趋势**

研究某省最低工资政策对女性工资的影响。
- 仅该省 女性 vs 男性 DID → 无法排除女性工资自身趋势差异
- 仅两省 女性 DID → 无法排除省份间趋势差异
- DDD："省份(T) × 性别(B) × 时间(Post)"三重交互

### 完整模型（Olden & Moen, 2022）

```
Y_sit = β₀ + β₁·T + β₂·B + β₃·Post + β₄·T×B + β₅·T×Post
      + β₆·B×Post + β₇·T×B×Post + ε_sit
```

β₇ = [(Y_{T=1,B=1,Post=1} − Y_{T=1,B=1,Post=0}) − (Y_{T=0,B=1,Post=1} − Y_{T=0,B=1,Post=0})]
   − [(Y_{T=1,B=0,Post=1} − Y_{T=1,B=0,Post=0}) − (Y_{T=0,B=0,Post=1} − Y_{T=0,B=0,Post=0})]

即 **B 组的 DID** 减去 **A 组的 DID**。

加入双向 FE 后，β₁(T)、β₃(Post) 被 FE 吸收，模型简化为：
```
Y_sit = β₂·B + β₄·T×B + β₅·T×Post + β₆·B×Post + β₇·T×B×Post + θ_i + λ_t + ε_sit
```

### DDD 的假设

DDD **不需要**两个平行趋势同时成立，只需两个 DID 估计中的偏差相同，
相减后即得无偏估计。这是比标准 DID 更弱的假设。

### 交错 DDD（triplediff 包详细用法）

```r
# R — triplediff 完整用法（Sant'Anna 官方包）
library(triplediff)
library(ggplot2)

run_ddd_staggered <- function(df, outcome, unit_id="unit_id", time="time",
                                gname="first_treat", pname="target_group",
                                controls=NULL) {
  out <- ddd(
    yname         = outcome,
    tname         = time,
    idname        = unit_id,
    gname         = gname,      # 首次处理时间（0=从未处理）
    pname         = pname,      # 政策针对子群体（0/1）
    xformla       = if (!is.null(controls))
                      as.formula(paste("~",paste(controls,collapse="+"))) else ~1,
    data          = df,
    control_group = "nevertreated",
    est_method    = "dr"  # 双稳健
  )
  # 聚合
  agg_simple  <- aggte(out, type="simple")
  agg_dynamic <- aggte(out, type="dynamic")
  cat(sprintf("DDD ATT: %.4f  SE: %.4f\n",
              agg_simple$overall.att, agg_simple$overall.se))
  ggdid(agg_dynamic, title="交错 DDD 动态效应")
  ggsave("output/ddd_staggered_dynamic.png", width=8, height=5, dpi=150)
  return(list(att_gt=out, simple=agg_simple, dynamic=agg_dynamic))
}
```

### DDD 事件研究图（平行趋势检验）

```r
# R — fixest 三重交互事件研究
library(fixest)

ddd_event_study <- function(df, outcome, treat="treated", group="target_group",
                              unit_id="unit_id", time="time", ref=-1) {
  # 构造 event_time × treat × group 交互
  fml <- as.formula(sprintf(
    "%s ~ i(event_time, %s:%s, ref=%d) + i(event_time, %s, ref=%d) + i(event_time, %s, ref=%d) | %s + %s",
    outcome, treat, group, ref, treat, ref, group, ref, unit_id, time))
  res <- feols(fml, data=df, cluster=as.formula(paste0("~",unit_id)))
  # 三重交互项的事件研究图
  iplot(res, main="DDD 事件研究图", xlab="事件时间", ylab="DDD 系数",
        keep=sprintf("%s:%s", treat, group))
  return(res)
}
```

### DDD 安慰剂检验

```python
# Python — 对 DDD 做空间安慰剂
import pyfixest as pf
import numpy as np

def ddd_placebo(df, outcome, treat='treated', group='target_group',
                unit_id='unit_id', time='time', n_perm=500, seed=42):
    np.random.seed(seed)
    df['ddd_var'] = df[treat] * df[group] * df['post']
    fml = f"{outcome} ~ ddd_var + {treat}:{group} + {treat}:post + {group}:post | {unit_id} + {time}"
    true_res = pf.feols(fml, data=df, vcov={"CRV1": unit_id})
    true_coef = true_res.coef()['ddd_var']

    units = df[unit_id].unique()
    treat_map = df.drop_duplicates(unit_id).set_index(unit_id)[treat]
    fake = []
    for _ in range(n_perm):
        df_f = df.copy()
        perm = np.random.permutation(treat_map.values)
        df_f['_ft'] = df_f[unit_id].map(dict(zip(units, perm)))
        df_f['_fddd'] = df_f['_ft'] * df_f[group] * df_f['post']
        fml_f = f"{outcome} ~ _fddd + _ft:{group} + _ft:post + {group}:post | {unit_id} + {time}"
        try:
            rf = pf.feols(fml_f, data=df_f, vcov={"CRV1": unit_id})
            fake.append(rf.coef()['_fddd'])
        except: pass
    fake = np.array(fake)
    p = np.mean(np.abs(fake) >= np.abs(true_coef))
    print(f"DDD 真实系数: {true_coef:.4f}  经验 p={p:.4f}")
    return {'true_coef': true_coef, 'p_value': p, 'fake_coefs': fake}
```

---

## 连续处理 Intensity DID

### 基础 Intensity × Post 回归

```python
# Python — pyfixest
import pyfixest as pf

def run_intensity_did(df, outcome, intensity='intensity', post='post',
                       unit_id='unit_id', time='time', controls=None):
    """
    Y_st = β(Intensity_s × Post_t) + θ_s + λ_t + ε_st
    intensity 为连续处理强度（如补贴金额、监管人员数量），跨单位变化。
    """
    controls = controls or []
    ctrl = (" + " + " + ".join(controls)) if controls else ""
    df['intensity_post'] = df[intensity] * df[post]
    fml = f"{outcome} ~ intensity_post{ctrl} | {unit_id} + {time}"
    res = pf.feols(fml, data=df, vcov={"CRV1": unit_id})
    pf.etable(res)
    return res
```

```r
# R — fixest
run_intensity_did_r <- function(df, outcome, intensity="intensity",
                                 unit_id="unit_id", time="time",
                                 post="post", controls=NULL) {
  ctrl_str <- if (length(controls)>0) paste("+",paste(controls,collapse="+")) else ""
  fml <- as.formula(sprintf("%s ~ i(%s,%s) %s | %s + %s",
                             outcome, intensity, post, ctrl_str, unit_id, time))
  res <- feols(fml, data=df, cluster=as.formula(paste0("~",unit_id)))
  etable(res); return(res)
}
```

### Callaway et al. (2024) 连续处理正式方法

```r
# R — contdid 包（Callaway, Goodman-Bacon & Sant'Anna 官方）
# install.packages("contdid")
library(contdid)

run_contdid <- function(df, outcome, unit_id="unit_id", time="time",
                          dose="intensity", first_treat="first_treat") {
  res <- cont_did(
    yname            = outcome,
    tname            = time,
    idname           = unit_id,
    dname            = dose,        # 连续处理变量
    gname            = first_treat, # 首次处理时间
    data             = df,
    target_parameter = "slope",     # 剂量-响应斜率
    treatment_type   = "continuous",
    control_group    = "notyettreated"
  )
  summary(res); plot(res)
  ggsave("output/contdid_dose_response.png", width=8, height=5, dpi=150)
  return(res)
}
```

### 连续处理离散化（替代方案）

```r
# R — 将连续强度离散化后用标准 CS
discretize_treatment <- function(df, intensity_var, method="median") {
  vals <- df[[intensity_var]]
  if (method == "median") {
    thr <- median(vals, na.rm=TRUE)
    df$treat_discrete <- as.integer(vals >= thr)
    cat(sprintf("中位数切割：%.4f\n", thr))
    cat("[声明] 须报告 25th/75th 切割点的稳健性。\n")
  } else if (method == "tertile") {
    q33 <- quantile(vals, 1/3, na.rm=TRUE)
    q67 <- quantile(vals, 2/3, na.rm=TRUE)
    df$treat_discrete <- dplyr::case_when(
      vals <= q33 ~ 0L, vals <= q67 ~ 1L, TRUE ~ 2L)
    cat("三分位分组。\n")
  }
  cat("[警告] 连续处理识别假设更强，须在论文中讨论外生性。\n")
  return(df)
}
```

---

## Stacked DID

为每个 cohort 构建干净 2×2 数据集，纵向拼接后联合回归。
直觉更强，与标准 OLS 框架兼容。

```r
library(fixest); library(dplyr)

stack_did <- function(df, outcome, unit_id="unit_id", time="time",
                       first_treat="first_treat", window=c(-4,4), controls=NULL) {
  cohorts <- sort(unique(df[[first_treat]]))
  cohorts <- cohorts[cohorts > 0 & is.finite(cohorts)]

  stacked_list <- lapply(cohorts, function(g) {
    t_start <- g + window[1]; t_end <- g + window[2]
    tr_units <- df %>% filter(.data[[first_treat]]==g) %>%
      pull(.data[[unit_id]]) %>% unique()
    co_units <- df %>% filter(.data[[first_treat]]==0 | .data[[first_treat]] > t_end) %>%
      pull(.data[[unit_id]]) %>% unique()
    df %>%
      filter(.data[[unit_id]] %in% c(tr_units, co_units),
             .data[[time]] >= t_start, .data[[time]] <= t_end) %>%
      mutate(cohort_id=g, stack_id=paste0(.data[[unit_id]],"_c",g),
             event_time=.data[[time]]-g,
             post_stack=as.integer(.data[[time]]>=g),
             treat_stack=as.integer(.data[[unit_id]] %in% tr_units),
             did_stack=post_stack*treat_stack)
  })
  df_s <- bind_rows(stacked_list)

  ctrl_str <- if (length(controls)>0) paste("+",paste(controls,collapse="+")) else ""
  res <- feols(as.formula(sprintf("%s ~ did_stack %s | stack_id + cohort_id^%s",
               outcome, ctrl_str, time)), data=df_s, cluster=~stack_id)
  etable(res)

  # 动态效应
  res_dyn <- feols(as.formula(sprintf(
    "%s ~ i(event_time, treat_stack, ref=-1) %s | stack_id + cohort_id^%s",
    outcome, ctrl_str, time)), data=df_s, cluster=~stack_id)
  iplot(res_dyn, main="Stacked DID 动态效应")
  ggsave("output/stacked_did_event_study.png", width=8, height=5, dpi=150)
  return(list(result=res, dyn=res_dyn, data=df_s))
}
```

---

## BJS 插补估计量（Borusyak, Jaravel & Spiess 2024）

```r
# install.packages("did2s")
library(did2s)

run_bjs <- function(df, outcome, unit_id="unit_id", time="time",
                     first_treat="first_treat", controls=NULL) {
  ctrl_str <- if (length(controls)>0) paste(controls, collapse="+") else "1"
  res <- did2s(data=df, yname=outcome,
               first_stage=as.formula(sprintf("~ 0 + %s | %s + %s",
                                               ctrl_str, unit_id, time)),
               second_stage=~ i(event_time, ref=-1),
               treatment="treat_indicator", cluster_var=unit_id)
  iplot(res, main="BJS 插补估计量")
  ggsave("output/bjs_dynamic.png", width=8, height=5, dpi=150)
  return(res)
}
```

---

## dCDH（de Chaisemartin & D'Haultfoeuille）

局部化估计 switchers 的瞬时效应，处理状态可逆时适用。

```r
# install.packages("DIDmultiplegt")
library(DIDmultiplegt)

run_dcdh <- function(df, outcome, unit_id="unit_id", time="time",
                      treatment="treated", controls=NULL, n_boot=100) {
  res <- did_multiplegt(df=df, Y=outcome, G=unit_id, T=time, D=treatment,
                        controls=if (length(controls)>0) controls else character(0),
                        brep=n_boot, placebo=3, dynamic=3, cluster=unit_id)
  print(res); return(res)
}
```

---

## Synthetic DID（synthdid）

结合合成控制 + DID，同时对单位和时间加权。处理单位少（< 20）时适用。

```r
# install.packages("synthdid")
library(synthdid)

run_synthdid <- function(df, outcome, unit_id="unit_id",
                          time="time", treatment="treated") {
  setup <- panel.matrices(as.data.frame(df), unit=unit_id,
                          time=time, outcome=outcome, treatment=treatment)
  tau_sdid <- synthdid_estimate(setup$Y, setup$N0, setup$T0)
  tau_sc   <- sc_estimate(setup$Y, setup$N0, setup$T0)
  tau_did  <- did_estimate(setup$Y, setup$N0, setup$T0)
  se_vals  <- lapply(list(sdid=tau_sdid, sc=tau_sc, did=tau_did),
                     function(e) sqrt(vcov(e, method="placebo")))
  cat(sprintf("Synthdid: %.4f (%.4f)\nSC: %.4f (%.4f)\nDID: %.4f (%.4f)\n",
              tau_sdid, se_vals$sdid, tau_sc, se_vals$sc, tau_did, se_vals$did))
  png("output/synthdid_plot.png", width=900, height=500, res=120)
  synthdid_plot(tau_sdid, se.method="placebo"); dev.off()
  return(list(synthdid=tau_sdid, sc=tau_sc, did=tau_did))
}
```

---

## fect（矩阵补全反事实）

因子模型估计反事实，支持多处理单位，适合强时间趋势 / 因子结构。

```r
# install.packages("fect")
library(fect)

run_fect <- function(df, outcome, unit_id="unit_id", time="time",
                      treatment="treated", controls=NULL, method="mc") {
  ctrl_fml <- if (length(controls)>0)
    as.formula(paste(outcome,"~",treatment,"+",paste(controls,collapse="+")))
  else as.formula(paste(outcome,"~",treatment))
  res <- fect(formula=ctrl_fml, data=df, index=c(unit_id,time),
              force="two-way", method=method, CV=TRUE,
              r=c(0,5), se=TRUE, nboots=200, parallel=TRUE, cores=4)
  print(res)
  plot(res, type="gap", main=sprintf("fect(%s) 处理效应", method))
  plot(res, type="equiv", main="fect 等价性检验")
  return(res)
}
```

---

## 多估计量对比可视化

向审稿人展示多种估计量系数方向和幅度一致。

```r
library(ggplot2); library(dplyr)

plot_estimator_comparison <- function(estimates_list) {
  # estimates_list: named list, each = list(coef, se, label)
  df_p <- bind_rows(lapply(names(estimates_list), function(nm) {
    e <- estimates_list[[nm]]
    data.frame(estimator=e$label, coef=e$coef,
               ci_lo=e$coef-1.96*e$se, ci_hi=e$coef+1.96*e$se,
               order=which(names(estimates_list)==nm))
  })) %>% arrange(order)
  df_p$estimator <- factor(df_p$estimator, levels=rev(df_p$estimator))
  df_p$grp <- ifelse(df_p$estimator=="TWFE","TWFE","稳健估计量")

  p <- ggplot(df_p, aes(x=coef, y=estimator, color=grp)) +
    geom_vline(xintercept=0, lty=2, color="gray50") +
    geom_errorbarh(aes(xmin=ci_lo, xmax=ci_hi), height=0.2, linewidth=1) +
    geom_point(size=4) +
    scale_color_manual(values=c("TWFE"="#7A7974","稳健估计量"="#20808D")) +
    labs(x="ATT", y=NULL, title="多估计量对比", color=NULL) +
    theme_minimal() + theme(legend.position="bottom")
  ggsave("output/estimator_comparison.png", p, width=8, height=5, dpi=150)
  return(p)
}
```

---

## CS 框架安慰剂（交错 DID）

在 CS 框架内打乱 `first_treat`（cohort 归属），保持结构。

```r
library(did); library(dplyr); library(ggplot2)

cs_placebo <- function(df, outcome, unit_id="unit_id", time="time",
                        first_treat="first_treat", n_perm=200, seed=42) {
  set.seed(seed)
  out_true <- att_gt(yname=outcome, gname=first_treat, idname=unit_id,
                     tname=time, xformla=~1, data=df, control_group="nevertreated")
  true_att <- aggte(out_true, type="simple")$overall.att

  treat_units <- df %>%
    filter(.data[[first_treat]]>0 & is.finite(.data[[first_treat]])) %>%
    distinct(.data[[unit_id]], .data[[first_treat]])
  fake_atts <- numeric(n_perm)
  for (i in seq_len(n_perm)) {
    shuffled <- treat_units %>% mutate(fake_ft=sample(.data[[first_treat]]))
    df_f <- df %>% left_join(shuffled %>% select(.data[[unit_id]], fake_ft), by=unit_id) %>%
      mutate(!!first_treat := ifelse(!is.na(fake_ft), fake_ft, .data[[first_treat]])) %>%
      select(-fake_ft)
    out_f <- tryCatch(att_gt(yname=outcome, gname=first_treat, idname=unit_id,
                             tname=time, xformla=~1, data=df_f,
                             control_group="nevertreated"), error=function(e) NULL)
    fake_atts[i] <- if (!is.null(out_f)) aggte(out_f, type="simple")$overall.att else NA
  }
  fake_atts <- na.omit(fake_atts)
  p_val <- mean(abs(fake_atts) >= abs(true_att))
  cat(sprintf("CS 安慰剂 p=%.4f (%d 次)\n", p_val, length(fake_atts)))
  return(list(fake_atts=fake_atts, p_value=p_val))
}
```

---

## SUTVA / 溢出效应（缓冲区检验）

```r
# 缓冲区检验思路（需 geosphere 包）：
# 1. 计算每个对照单位与最近处理单位的球面距离（distHaversine）
# 2. 排除距离 ≤ buffer_km（50/100/200 km）的对照单位
# 3. 在缩减样本上重跑基准回归，比较系数变化
# 建议报告三档缓冲区（50/100/200 km），距离需有经济学理由。

library(geosphere); library(fixest); library(dplyr)

buffer_test <- function(df, outcome, did_var, unit_id="unit_id", time="time",
                          lat="lat", lon="lon", treatment="treated",
                          buffers=c(50,100,200), controls=NULL) {
  ctrl_str <- if (length(controls)>0) paste("+",paste(controls,collapse="+")) else ""
  treated_locs <- df %>% filter(.data[[treatment]]==1) %>%
    distinct(.data[[unit_id]], .data[[lat]], .data[[lon]])
  control_units <- df %>% filter(.data[[treatment]]==0) %>%
    distinct(.data[[unit_id]], .data[[lat]], .data[[lon]])

  results <- lapply(buffers, function(buf_km) {
    control_units$min_dist <- sapply(seq_len(nrow(control_units)), function(i) {
      min(distHaversine(
        control_units[i, c(lon,lat)],
        treated_locs[, c(lon,lat)])) / 1000
    })
    keep <- control_units %>% filter(min_dist > buf_km) %>% pull(.data[[unit_id]])
    df_sub <- df %>% filter(.data[[treatment]]==1 | .data[[unit_id]] %in% keep)
    res <- feols(as.formula(sprintf("%s ~ %s %s | %s + %s",
                  outcome, did_var, ctrl_str, unit_id, time)),
                  data=df_sub, cluster=as.formula(paste0("~",unit_id)))
    data.frame(buffer_km=buf_km, coef=coef(res)[did_var],
               se=se(res)[did_var], n_control=length(keep))
  })
  res_df <- do.call(rbind, results)
  print(res_df); return(res_df)
}
```

---

## 估计量选择速查表

| 估计量 | 方法论要求 | 复杂度 | R 包 | Python 包 | 期刊风格 |
|--------|-----------|--------|------|-----------|---------|
| TWFE | 效应时间不变（强） | 低 | fixest | pyfixest | 所有（基准）|
| CS | 平行趋势 + 从未处理 | 中 | did | diff-diff | 顶刊 |
| SA | 平行趋势 | 低 | fixest | pyfixest | 顶刊 |
| Stacked | 干净 2×2 | 中 | fixest | — | 顶刊附录 |
| BJS | 平行趋势 + 插补 | 中 | did2s | — | 顶刊附录 |
| dCDH | switchers 平行趋势 | 高 | DIDmultiplegt | — | 方法论刊 |
| Synthdid | 单位权重收敛 | 中 | synthdid | — | 处理单位少 |
| fect | 因子模型稳定 | 高 | fect | — | 政治学 / 社会学 |
| DDD | 偏差相同 | 中 | triplediff / fixest | diff-diff | 管理学高频 |
| Intensity | 连续处理外生 | 中 | contdid / fixest | pyfixest | 政策评估 |
