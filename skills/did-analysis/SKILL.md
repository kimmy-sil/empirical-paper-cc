# DID 分析（双重差分法）

双重差分法通过比较处理组与对照组在处理前后的变化差异来识别因果效应，是政策评估中最常用的准自然实验方法。

**适用场景：** 标准 2×2 DID；交错 DID（Staggered，不同单位在不同时间接受处理）。政策在不同地区 / 时间实施时触发。

**语言：** Python（linearmodels）+ R（fixest / did / bacondecomp）。无 Stata。

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

---

## Step 1：数据诊断

```python
# ============================================================
# 数据诊断 — Python
# ============================================================
import pandas as pd
import numpy as np

def diagnose_did_data(df, unit_id='unit_id', time='time',
                      treatment='treated', first_treat=None):
    """
    检测处理变量类型 + 面板平衡性 + 处理时间分布。
    Returns: dict 含诊断结果
    """
    treat_vals = df[treatment].dropna().unique()
    # ── 处理变量类型检测 ──────────────────────────────────
    if set(treat_vals).issubset({0, 1}):
        print("[OK] 处理变量为二元（0/1），适合标准 DID。")
    else:
        print("[WARNING] 处理变量非二元 → 连续处理，需不同识别策略。")
        print("          参考：Callaway, Goodman-Bacon & Sant'Anna (2021)")

    # ── 面板平衡性 ────────────────────────────────────────
    obs_per_unit = df.groupby(unit_id)[time].count()
    is_balanced  = obs_per_unit.nunique() == 1
    print(f"\n面板平衡性：{'平衡' if is_balanced else '非平衡'}")

    # ── 处理时间分布表（交错 DID）────────────────────────
    if first_treat is not None:
        cohort_dist = (df.drop_duplicates(subset=[unit_id])
                         .groupby(first_treat)[unit_id].count()
                         .reset_index())
        print("\n处理时间分布（Cohort 结构）：")
        print(cohort_dist.to_string(index=False))
        n_never = df[df[first_treat].isin([0, np.inf, 9999])
                    ][unit_id].nunique()
        if n_never == 0:
            print("[WARNING] 无从未处理单位 → CS/SA 以最晚处理组为对照，效力下降。")
    return {'balanced': is_balanced}
```

```r
# ============================================================
# 数据诊断 — R
# ============================================================
library(dplyr)

diagnose_did_data <- function(df, unit_id="unit_id", time="time",
                               treatment="treated", first_treat=NULL) {
  treat_vals <- unique(na.omit(df[[treatment]]))
  if (!all(treat_vals %in% c(0,1)))
    cat("[WARNING] 处理变量非二元，可能为连续处理，需 CS(2021) 连续扩展版。\n")
  else
    cat("[OK] 处理变量为二元（0/1）。\n")

  obs <- df %>% count(.data[[unit_id]]) %>% pull(n)
  cat(sprintf("面板平衡性：%s\n", ifelse(length(unique(obs))==1,"平衡","非平衡")))

  if (!is.null(first_treat)) {
    cohort_dist <- df %>%
      distinct(.data[[unit_id]], .data[[first_treat]]) %>%
      count(.data[[first_treat]], name="n_units")
    cat("\n处理时间分布（Cohort）：\n"); print(cohort_dist)
  }
}
```

---

## Step 2：TWFE 基准回归 + 自动 Bacon 分解

**规则：** TWFE 跑完后**自动**运行 Bacon 分解。根据负权重比例决定后续策略。

```python
# ============================================================
# TWFE 基准回归 — Python（linearmodels PanelOLS）
# ============================================================
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

def run_twfe(df, outcome, treatment_var, controls=None,
             unit_id='unit_id', time='time'):
    """
    TWFE 双向固定效应。treatment_var 为 post×treated 交互项。
    Returns: dict{'result', 'coef', 'se', 'pval'}
    """
    controls  = controls or []
    df_panel  = df.set_index([unit_id, time])
    X = sm.add_constant(df_panel[[treatment_var] + controls])
    res = PanelOLS(df_panel[outcome], X,
                   entity_effects=True, time_effects=True
                  ).fit(cov_type='clustered', cluster_entity=True)
    print(f"TWFE ATT: {res.params[treatment_var]:.4f}  "
          f"SE: {res.std_errors[treatment_var]:.4f}  "
          f"p: {res.pvalues[treatment_var]:.4f}")
    print("[提示] 交错 DID 须自动运行 Bacon 分解（见 R 代码）。")
    return {'result': res, 'coef': res.params[treatment_var],
            'se': res.std_errors[treatment_var],
            'pval': res.pvalues[treatment_var]}
```

```r
# ============================================================
# TWFE + 自动 Bacon 分解 — R（fixest + bacondecomp）
# ============================================================
library(fixest)
# install.packages("bacondecomp")
library(bacondecomp)
library(ggplot2)

run_twfe_bacon <- function(df, outcome, did_var, controls=NULL,
                            unit_id="unit_id", time="time",
                            treatment="treated") {
  ctrl_str <- if (length(controls)>0) paste("+",paste(controls,collapse="+")) else ""
  fml      <- as.formula(sprintf("%s ~ %s %s | %s + %s",
                                  outcome, did_var, ctrl_str, unit_id, time))
  res_twfe <- feols(fml, data=df,
                    cluster=as.formula(paste0("~",unit_id)))
  etable(res_twfe, title="TWFE 基准回归",
         notes="聚类至个体；双向 FE。")

  # ── 自动 Bacon 分解 ──────────────────────────────────────
  bacon_out <- bacon(as.formula(paste(outcome,"~",treatment)),
                     data=df, id_var=unit_id, time_var=time)

  # 负权重比例
  neg_wt <- with(bacon_out,
    sum(abs(weight[type=="Later vs Always Treated"])) / sum(abs(weight))
  )
  cat(sprintf("\n负权重（Later vs Always Treated）：%.1f%%\n", neg_wt*100))

  # 三档决策
  if      (neg_wt == 0)    cat("[决策] 负权重=0 → TWFE 可信。\n")
  else if (neg_wt < 0.10)  cat("[决策] 负权重 < 10% → TWFE 为主，补充 CS/SA 稳健性。\n")
  else                      cat("[决策] 负权重 ≥ 10% → 切换 CS/SA 为主回归，TWFE 降附录。\n")

  # Bacon 可视化
  p <- ggplot(bacon_out, aes(x=weight, y=estimate, color=type)) +
    geom_point(size=3, alpha=0.8) +
    geom_hline(yintercept=0, linetype="dashed") +
    labs(x="权重", y="系数", title="Bacon 分解",
         caption=sprintf("负权重比例：%.1f%%", neg_wt*100),
         color="对比类型") +
    theme_minimal(base_size=12)
  ggsave("output/bacon_decomposition.png", p, width=8, height=5, dpi=150)

  return(list(twfe=res_twfe, bacon=bacon_out, neg_wt_share=neg_wt))
}
```

**三档决策：**

| 负权重比例 | 主策略 |
|------------|--------|
| = 0 | TWFE 可信 |
| < 10% | TWFE 为主 + CS/SA 稳健性 |
| ≥ 10% | 切换 CS 或 SA 为主回归 |
| 有理论依据认为效应时间不变 | 可用 TWFE，必须声明 + 报告分解 |

---

## Step 3：事件研究图（平行趋势检验）

**注意：** pre-period 系数不显著 ≠ 平行趋势假设成立，建议配合 HonestDiD。

```python
# ============================================================
# 事件研究图 — Python（PanelOLS + matplotlib）
# 需要 df 中已有 event_time 列（= time - first_treat）
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def event_study_python(df, outcome, unit_id='unit_id', time='time',
                        treatment='treated', ref_period=-1,
                        window=(-4,4), controls=None):
    """
    构造事件时间哑变量，估计并绘制事件研究图。
    Returns: dict{'fig', 'coefs': pd.DataFrame, 'result'}
    """
    controls  = controls or []
    df        = df.copy()
    t_min, t_max = window

    # 构造哑变量（排除参照期）
    dummy_cols = []
    for t in range(t_min, t_max+1):
        if t == ref_period: continue
        col = f'et_{t}'
        df[col] = ((df[treatment]==1) & (df['event_time']==t)).astype(int)
        dummy_cols.append(col)

    df_panel = df.set_index([unit_id, time])
    X = sm.add_constant(df_panel[dummy_cols + controls])
    res = PanelOLS(df_panel[outcome], X,
                   entity_effects=True, time_effects=True
                  ).fit(cov_type='clustered', cluster_entity=True)

    # 提取系数
    import pandas as pd, numpy as np
    records = [{'event_time': int(c.replace('et_','')),
                'coef': res.params.get(c, np.nan),
                'ci_lo': res.params.get(c,np.nan) - 1.96*res.std_errors.get(c,np.nan),
                'ci_hi': res.params.get(c,np.nan) + 1.96*res.std_errors.get(c,np.nan)}
               for c in dummy_cols]
    records.append({'event_time': ref_period, 'coef':0., 'ci_lo':0., 'ci_hi':0.})
    coef_df = pd.DataFrame(records).sort_values('event_time')

    # 绘图
    fig, ax = plt.subplots(figsize=(9,5))
    pre  = coef_df[coef_df['event_time'] < 0]
    post = coef_df[coef_df['event_time'] >= 0]
    for sub, col in [(pre,'#1B474D'), (post,'#20808D')]:
        ax.plot(sub['event_time'], sub['coef'], marker='o', color=col, lw=1.6)
        ax.fill_between(sub['event_time'], sub['ci_lo'], sub['ci_hi'],
                        alpha=0.15, color=col)
    ax.axhline(0, color='gray', linestyle='--', lw=0.9)
    ax.axvline(-0.5, color='#A84B2F', linestyle=':', lw=1.2)
    ax.set_xlabel('事件时间（相对处理期）'); ax.set_ylabel(f'{outcome} 系数')
    ax.set_title('事件研究图（平行趋势检验）')
    ax.annotate('注：pre-period 系数不显著≠假设成立，建议配合 HonestDiD。',
                xy=(0.01,0.02), xycoords='axes fraction',
                fontsize=8, color='gray', style='italic')
    plt.tight_layout()
    plt.savefig('output/event_study.png', dpi=150)
    return {'fig': fig, 'coefs': coef_df, 'result': res}
```

```r
# ============================================================
# 事件研究图 — R（fixest i() + iplot）
# ============================================================
library(fixest)

run_event_study_r <- function(df, outcome, unit_id="unit_id", time="time",
                               treatment="treated", ref_period=-1,
                               controls=NULL) {
  ctrl_str <- if (length(controls)>0) paste("+",paste(controls,collapse="+")) else ""
  fml <- as.formula(sprintf("%s ~ i(event_time,%s,ref=%d) %s | %s + %s",
                              outcome, treatment, ref_period,
                              ctrl_str, unit_id, time))
  res_es <- feols(fml, data=df,
                  cluster=as.formula(paste0("~",unit_id)))

  iplot(res_es, main="事件研究图（平行趋势检验）",
        xlab="事件时间（相对处理期）", ylab="系数（ATT）",
        col="#20808D", pt.pch=19)
  abline(h=0, lty=2, col="gray50")
  mtext("注：pre-period 系数不显著≠假设成立，建议配合 HonestDiD。",
        side=1, line=4, cex=0.75, col="gray40", font=3)

  # Pre-period 联合显著性检验
  pre <- grep("event_time::-[2-9]", names(coef(res_es)), value=TRUE)
  if (length(pre)>0) {
    wt <- wald(res_es, keep=pre)
    cat(sprintf("Pre-period 联合 F: %.3f, p=%.4f  通过(p>0.10): %s\n",
                wt$stat, wt$p, wt$p>0.10))
  }
  return(res_es)
}
```

---

## Step 4：交错 DID 稳健估计量

| 估计量 | 适用场景 | R 包 |
|--------|----------|------|
| **Callaway-Sant'Anna (CS)** | 有从未处理单位；效应可能随时间变化 | `did` |
| **Sun-Abraham (SA)** | 希望留在 fixest 生态；快速交互加权 | `fixest` sunab |

**原则：** 有从未处理单位 → 优先 CS；希望与 feols 集成 → SA 同样可信。

```r
# ============================================================
# Callaway-Sant'Anna — R（did 包）
# ============================================================
# install.packages("did")
library(did)

run_cs <- function(df, outcome, unit_id="unit_id", time="time",
                   first_treat="first_treat", controls=NULL) {
  out <- att_gt(
    yname         = outcome,
    gname         = first_treat,
    idname        = unit_id,
    tname         = time,
    xformla       = if (!is.null(controls))
                      as.formula(paste("~",paste(controls,collapse="+"))) else ~1,
    data          = df,
    est_method    = "dr",           # 双稳健（推荐）
    control_group = "nevertreated"
  )
  agg_att <- aggte(out, type="simple")
  cat(sprintf("CS 整体 ATT: %.4f  SE: %.4f\n",
              agg_att$overall.att, agg_att$overall.se))

  agg_dyn <- aggte(out, type="dynamic")
  ggdid(agg_dyn, title="CS 动态处理效应",
        xlab="相对处理期", ylab="ATT")
  ggsave("output/cs_dynamic.png", width=8, height=5, dpi=150)

  return(list(att_gt=out, agg_simple=agg_att, agg_dynamic=agg_dyn))
}

# ============================================================
# Sun-Abraham — R（fixest sunab）
# ============================================================
run_sa <- function(df, outcome, unit_id="unit_id", time="time",
                   first_treat="first_treat", controls=NULL) {
  ctrl_str <- if (length(controls)>0) paste("+",paste(controls,collapse="+")) else ""
  res_sa <- feols(
    as.formula(sprintf("%s ~ sunab(%s,%s) %s | %s + %s",
                        outcome, first_treat, time,
                        ctrl_str, unit_id, time)),
    data=df, cluster=as.formula(paste0("~",unit_id))
  )
  cat(sprintf("SA 聚合 ATT: %.4f\n",
              aggregate(res_sa, agg="att")[1]))
  iplot(res_sa, main="SA 动态处理效应", col="#20808D")
  ggsave("output/sa_dynamic.png", width=8, height=5, dpi=150)
  return(res_sa)
}
```

---

## Step 5：安慰剂检验（完整流程）

### 5a 空间安慰剂（500次置换）

**关键：置换在 unit 级别进行，不是观测级别。**

```python
# ============================================================
# 空间安慰剂 500 次置换 — Python（完整含图）
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def spatial_placebo(df, outcome, did_var, controls=None,
                    unit_id='unit_id', time='time',
                    n_perm=500, seed=42, true_coef=None):
    """
    随机打乱 unit 级处理状态 500 次，估计经验 p 值。
    经验 p 值 = |假系数| ≥ |真实系数| 的比例。
    """
    np.random.seed(seed)
    controls = controls or []

    # 估计真实系数
    df_panel  = df.set_index([unit_id, time])
    X         = sm.add_constant(df_panel[[did_var]+controls])
    res_true  = PanelOLS(df_panel[outcome], X,
                          entity_effects=True, time_effects=True
                         ).fit(cov_type='clustered', cluster_entity=True)
    true_coef = true_coef or res_true.params[did_var]

    units      = df[unit_id].unique()
    treat_vals = df.drop_duplicates(unit_id)['treated'].values
    fake_coefs = []

    for _ in range(n_perm):
        df_f = df.copy()
        perm  = np.random.permutation(treat_vals)
        df_f['_fake_treat'] = df_f[unit_id].map(dict(zip(units, perm)))
        df_f['_fake_did']   = df_f['_fake_treat'] * df_f['post']
        try:
            pf = df_f.set_index([unit_id, time])
            Xf = sm.add_constant(pf[['_fake_did']+controls])
            rf = PanelOLS(pf[outcome], Xf,
                          entity_effects=True, time_effects=True
                         ).fit(cov_type='clustered', cluster_entity=True)
            fake_coefs.append(rf.params['_fake_did'])
        except Exception:
            pass

    fake_coefs = np.array(fake_coefs)
    p_val      = np.mean(np.abs(fake_coefs) >= np.abs(true_coef))
    print(f"真实系数：{true_coef:.4f}  经验 p 值：{p_val:.4f}  "
          f"置换次数：{len(fake_coefs)}")

    # 核密度图
    fig, ax = plt.subplots(figsize=(8,5))
    kde = gaussian_kde(fake_coefs)
    x   = np.linspace(fake_coefs.min(), fake_coefs.max(), 200)
    ax.fill_between(x, kde(x), alpha=0.35, color='#20808D',
                    label='安慰剂分布（500次）')
    ax.plot(x, kde(x), color='#1B474D', lw=1.5)
    ax.axvline(true_coef, color='#A84B2F', lw=2, linestyle='--',
               label=f'真实系数={true_coef:.4f}')
    ax.set(xlabel='DID 系数', ylabel='核密度',
           title=f'空间安慰剂（n={len(fake_coefs)}）  经验 p={p_val:.4f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig('output/placebo_spatial.png', dpi=150)
    return {'fake_coefs': fake_coefs, 'p_value': p_val}
```

```r
# ============================================================
# 空间安慰剂 500 次置换 — R（完整含图）
# ============================================================
library(fixest); library(dplyr); library(ggplot2)

spatial_placebo_r <- function(df, outcome, controls=NULL,
                               unit_id="unit_id", time="time",
                               treatment="treated", post="post",
                               n_perm=500, seed=42, true_coef=NULL) {
  set.seed(seed)
  ctrl_str <- if (length(controls)>0) paste("+",paste(controls,collapse="+")) else ""
  df$did   <- df[[post]] * df[[treatment]]

  if (is.null(true_coef)) {
    res_true  <- feols(as.formula(sprintf("%s ~ did %s | %s + %s",
                                           outcome, ctrl_str, unit_id, time)),
                       data=df, cluster=as.formula(paste0("~",unit_id)))
    true_coef <- coef(res_true)["did"]
  }
  cat(sprintf("真实系数：%.4f\n", true_coef))

  units      <- unique(df[[unit_id]])
  treat_vals <- df %>% distinct(.data[[unit_id]], .data[[treatment]]) %>%
    pull(.data[[treatment]])
  fake_coefs <- numeric(n_perm)

  for (i in seq_len(n_perm)) {
    perm_map <- setNames(sample(treat_vals), units)
    df_f     <- df %>%
      mutate(fake_treat = perm_map[as.character(.data[[unit_id]])],
             fake_did   = .data[[post]] * fake_treat)
    res_f <- tryCatch(
      feols(as.formula(sprintf("%s ~ fake_did %s | %s + %s",
                                outcome, ctrl_str, unit_id, time)),
            data=df_f, cluster=as.formula(paste0("~",unit_id))),
      error=function(e) NULL)
    fake_coefs[i] <- if (!is.null(res_f)) coef(res_f)["fake_did"] else NA_real_
  }

  fake_coefs <- na.omit(fake_coefs)
  p_val      <- mean(abs(fake_coefs) >= abs(true_coef))
  cat(sprintf("经验 p 值：%.4f（%d 次）\n", p_val, length(fake_coefs)))

  p <- ggplot(data.frame(x=fake_coefs), aes(x=x)) +
    geom_density(fill="#20808D", alpha=0.35, color="#1B474D", lw=1) +
    geom_vline(xintercept=true_coef, color="#A84B2F", lw=1.5, lty="dashed") +
    labs(x="DID 系数（安慰剂）", y="核密度",
         title=sprintf("空间安慰剂（%d 次）", length(fake_coefs)),
         caption=sprintf("经验 p = %.4f", p_val)) +
    theme_minimal(base_size=12)
  ggsave("output/placebo_spatial.png", p, width=8, height=5, dpi=150)
  return(list(fake_coefs=fake_coefs, p_value=p_val))
}
```

### 5b 时间安慰剂

**关键：去掉处理后样本（post == 0 的子样本），用处理前各期做假处理时间。**

```r
# ============================================================
# 时间安慰剂 — R
# ============================================================
time_placebo_r <- function(df, outcome, unit_id="unit_id", time="time",
                            treatment="treated", controls=NULL, n_fake=5) {
  ctrl_str <- if (length(controls)>0) paste("+",paste(controls,collapse="+")) else ""
  df_pre    <- df[df$post == 0, ]  # 严格去掉处理后样本！
  cat(sprintf("使用处理前样本：%d 行（已去除处理后数据）\n", nrow(df_pre)))

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
  print(res_df)
  cat("解读：所有假处理期系数应不显著（p > 0.10）。\n")
  return(res_df)
}
```

---

## Step 6：HonestDiD 敏感性分析

默认使用 Relative Magnitudes（RM），Smoothness 作补充。

```r
# ============================================================
# HonestDiD — R（install.packages("HonestDiD")）
# ============================================================
library(HonestDiD)

run_honest_did <- function(res_es, Mbarvec=seq(0,2,0.5)) {
  """
  结果解读：Mbar=1 时 CI 不含零 → 在处理前偏移 1 倍的容忍度下结论稳健。
  """
  cf    <- coef(res_es)[grep("event_time", names(coef(res_es)))]
  vc    <- vcov(res_es)[grep("event_time",rownames(vcov(res_es))),
                         grep("event_time",colnames(vcov(res_es)))]
  t_vals <- as.numeric(gsub(".*::(-?[0-9]+).*","\\1",names(cf)))
  pre    <- which(t_vals < 0)
  pst    <- which(t_vals > 0)

  hd_out <- createSensitivityResults_relativeMagnitudes(
    betahat        = cf[pst],
    sigma          = vc[pst,pst],
    numPrePeriods  = length(pre),
    numPostPeriods = length(pst),
    Mbarvec        = Mbarvec,
    alpha          = 0.05
  )

  p_hd <- createSensitivityPlot_relativeMagnitudes(hd_out, alpha=0.05)
  ggsave("output/honest_did.png", p_hd, width=8, height=5, dpi=150)

  mbar1 <- hd_out[hd_out$Mbar==1,]
  if (nrow(mbar1)>0) {
    cat(sprintf("Mbar=1 CI: [%.4f, %.4f]  通过（不含零）: %s\n",
                mbar1$lb[1], mbar1$ub[1],
                (mbar1$lb[1]>0 | mbar1$ub[1]<0)))
  }
  return(hd_out)
}
```

---

## Step 7：SUTVA / 溢出效应

DID 假设无溢出（SUTVA）。若政策可能影响对照组（如产业迁移、贸易溢出），需做**地理缓冲区检验**：排除处理地区周边 N 公里内的对照单位后重跑基准回归。若系数基本不变，溢出效应不构成威胁。

```r
# 缓冲区检验（需 geosphere 包计算距离）：
# library(geosphere)
# 1. 计算每个对照单位与最近处理单位的球面距离（distHaversine）
# 2. 排除距离 ≤ buffer_km（50/100/200 km）的对照单位
# 3. 在缩减样本上重跑基准回归，比较系数变化
# 建议报告三档缓冲区结果（50/100/200 公里）。
# 缓冲区距离需要有经济学理由（如政策传导的地理范围）。
```

---

## 检验清单

### 必做（缺一项影响发表）

| 检验 | 对应步骤 | 判断标准 |
|------|----------|----------|
| 事件研究图 | Step 3 | Pre-period 联合 p > 0.10 |
| HonestDiD | Step 6 | Mbar=1 时 CI 不含零 |
| 安慰剂 500 次 | Step 5a | 经验 p 值 < 0.05 |
| Bacon 分解 | Step 2 | 记录负权重比例 + 决策 |

### 推荐（应对审稿人）

- 替换结果变量代理指标
- 缩短事件窗口（±2 期）
- 排除同期政策（文献 + 数据双验证）
- 异质性检验（行业 / 规模 / 地区分组）
- 仅用平衡子样本重跑

---

## 常见错误（8条）

| # | 错误 | 后果 | 正确做法 |
|---|------|------|----------|
| 1 | **交错 DID 盲目用 TWFE** | 负权重导致符号错误 | 先 Bacon 分解；负权重≥10% → CS/SA |
| 2 | **聚类层级低于处理层级** | 标准误严重低估 | 聚类层级 = 处理分配层级 |
| 3 | **安慰剂未去掉处理后样本** | 时间安慰剂 p 值虚低 | 严格限制于 post==0 子样本 |
| 4 | **事件研究参照期选错** | 系数基准漂移 | ref = -1（处理前最后一期）|
| 5 | **pre 不显著 = 平行趋势成立** | 功效不足无法区分 | 配合 HonestDiD 报告 Mbar |
| 6 | **只看 TWFE 系数忽视 Bacon** | 忽视负权重风险 | 明确报告权重分布 |
| 7 | **连续处理直接套 TWFE** | 识别假设失效 | 先离散化或 CS(2021) 连续扩展 |
| 8 | **事件研究窗口仅 ±1 期** | 无法检验平行趋势 | 至少 ±3 期 |

---

## Estimand 声明

| 方法 | Estimand | 必须声明 |
|------|----------|---------|
| **TWFE** | Weighted ATT（可能含负权重）| 负权重比例（来自 Bacon 分解）；若 > 0 须声明不等于简单 ATT |
| **Callaway-Sant'Anna** | Group-time ATT，聚合为整体 ATT | 聚合权重类型（simple / dynamic）；从未处理对照合理性 |
| **Sun-Abraham** | Interaction-weighted ATT | TWFE 的交互加权修正；线性假设下等价于 CS |

**论文声明模板：**

```
本文 TWFE 估计量识别加权平均处理效应（Weighted ATT）。根据 Bacon (2021) 分解，
"Later vs Always Treated" 权重占比为 [X]%，负权重比例为 [Y]%。
鉴于负权重比例 [低于/高于] 10%，本文 [以 TWFE 为主，补充 CS 稳健性 /
切换为 Callaway-Sant'Anna (2021) 作主回归，TWFE 降附录]。
CS 估计量识别 Group-time ATT，以 simple 加权聚合，对照组为从未处理单位。
```

> 高级工具（Stacked DID / BJS / dCDH / Synthdid / fect）见 ADVANCED.md。
