# 机制分析

机制分析回答"为什么 X 影响 Y"，通过引入中介变量或调节变量揭示因果链条。在建立基准因果效应之后进行，是顶刊论文从"好到卓越"的关键步骤。

**语言：** Python（statsmodels / linearmodels）+ R（fixest / mediation / marginaleffects）。无 Stata。所有函数参数传入，不使用 `input()`。

---

## 0. 强制前置：DAG 绘制

**在跑任何机制回归之前，必须画因果图（DAG）。** 确认 M 是中介、调节还是对撞变量，才能决定后续策略。

### 文字描述 DAG 模板

```
本文提出如下因果链条：
  处理变量 X（政策/事件）→ 中介变量 M（企业融资约束）→ 结果变量 Y（投资规模）
直接效应：X 同时对 Y 存在直接影响（X → Y 直接路径）。
混淆因子：宏观经济冲击 Z 同时影响 X 和 Y，已通过时间固定效应控制。
M 满足中介变量三大条件：
  (1) M 在 X 之后、Y 之前发生（时序逻辑成立）
  (2) X 显著影响 M（第一步回归验证）
  (3) M 显著影响 Y，X→Y 效应在控制 M 后发生变化
```

### dagitty R 代码（核心 10 行）

```r
# ============================================================
# DAG — R（dagitty，核心功能）
# install.packages("dagitty")
# ============================================================
library(dagitty)

dag_spec <- dagitty('
  dag {
    X [exposure, pos="0,1"]; Y [outcome, pos="2,1"]
    M [pos="1,1"];            Z [pos="1,2"]
    X -> M -> Y;  X -> Y;    Z -> X;  Z -> Y
  }
')

# 最小充分调整集：若 M 出现 → M 是混淆因子（非中介！）
adjustmentSets(dag_spec, exposure="X", outcome="Y")

# 隐含条件独立性（可检验假设，与数据不符则需修改 DAG）
impliedConditionalIndependencies(dag_spec)

# M 角色判断
cat("M 是 Y 的祖先（中介路径存在）:", isAncestorOf(dag_spec,"M","Y"), "\n")
cat("M 是 X 的后代（受 X 影响）:",    isDescendantOf(dag_spec,"X","M"), "\n")
```

### Python networkx DAG 可视化

```python
# ============================================================
# DAG 可视化 — Python（networkx，简单版）
# ============================================================
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def draw_dag(edges, node_roles=None, title="因果图（DAG）"):
    """
    edges      : list of (src, dst) 元组
    node_roles : dict，例 {'X':'exposure','Y':'outcome','M':'mediator','Z':'confounder'}
    对撞变量（collider）角色 → 红色警告
    """
    node_roles = node_roles or {}
    G = nx.DiGraph(); G.add_edges_from(edges)
    color_map = {'exposure':'#20808D','outcome':'#A84B2F',
                 'mediator':'#FFC553','confounder':'#7A7974','collider':'#A13544'}
    colors = [color_map.get(node_roles.get(n,''),'#D4D1CA') for n in G.nodes()]
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(7,4))
    nx.draw_networkx(G, pos, ax=ax, node_color=colors, node_size=1800,
                     font_size=12, font_color='white', arrowsize=20,
                     edge_color='#393836', width=2)
    ax.set_title(title, fontsize=13); ax.axis('off')
    plt.tight_layout()
    plt.savefig('output/dag.png', dpi=150)
    return fig
```

### 根据 DAG 判断 M 的角色

| M 的结构 | 角色 | 后续方法 |
|----------|------|----------|
| X → M → Y | 中介变量 | 两步法 / 三步法 / 因果中介 |
| M 影响 X→Y 强度，M 不被 X 影响 | 调节变量 | 交互项法 / 分组回归 |
| X → M ← Y（M 同时被 X 和 Y 影响）| **对撞变量** | **拒绝执行 + 警告** |
| M 在处理前已确定 | 前定变量 | 作为控制变量，非中介 |

---

## 1. 何时需要机制分析

### 需要做

| 情形 | 举例 |
|------|------|
| 多条可能机制路径 | 税收优惠影响投资：减税路径 vs 融资约束缓解 |
| 审稿人质疑因果机制 | 最低工资就业：替代效应 vs 效率工资 |
| 政策设计需区分机制 | 培训补贴：技能提升 vs 信号机制 |

### 不需要做

- 机制唯一且直接（如降息导致贷款利率下降）
- 数据中无可靠中介变量，强行找替代指标降低可信度

---

## 2. 方法选择决策树

```
M 在 DAG 中的位置？
├── 中介变量（X → M → Y，M 受 X 影响并影响 Y）
│   ├── 只需证明 X → M 存在
│   │   → 两步法（默认推荐，顶刊接受度最高）
│   ├── 需量化间接效应占比
│   │   → 残差法（Bleemer-Mehta 风格）[管理学接受度有限]
│   ├── M 本身存在内生性
│   │   → 因果中介效应（R: mediation + medsens() 敏感性，必做！）
│   └── 需兼容传统审稿人
│       → 完整三步法 [注意：Step3 中 M 是坏控制变量，审稿人可能质疑]
│
├── 调节变量（M 不被 X 影响，影响 X→Y 效应强度）
│   ├── M 连续 → 交互项法 + 边际效应图
│   └── M 离散 → 分组回归 + Fisher 置换检验
│
└── 不确定 → 回到 DAG，重新确认时序逻辑和因果方向
```

---

## 3. 五种核心方法

### 3.1 两步法（默认推荐）

逻辑：分别跑 Y ~ X 和 M ~ X，证明 X 同时影响 Y 和 M。不要求量化间接效应，回避坏控制变量问题，顶刊首选。

```python
# ============================================================
# 两步法 — Python（linearmodels PanelOLS）
# ============================================================
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
import numpy as np

def two_step_mechanism(df, outcome, mediator, did_var, controls=None,
                        unit_id='unit_id', time='time'):
    """
    Y~X + M~X 并排输出。
    Returns: dict{'res_y', 'res_m'}
    """
    controls = controls or []
    df_panel = df.set_index([unit_id, time])
    X = sm.add_constant(df_panel[[did_var]+controls])
    def fit(dep):
        return PanelOLS(df_panel[dep], X,
                        entity_effects=True, time_effects=True
                       ).fit(cov_type='clustered', cluster_entity=True)
    res_y, res_m = fit(outcome), fit(mediator)

    print(f"{'':22} {'Y（结果）':>14} {'M（机制）':>14}")
    print("-"*52)
    for var in [did_var]+controls:
        cy, cm = res_y.params.get(var,np.nan), res_m.params.get(var,np.nan)
        sy, sm_ = res_y.std_errors.get(var,np.nan), res_m.std_errors.get(var,np.nan)
        print(f"{var:22} {cy:>14.4f} {cm:>14.4f}")
        print(f"{'':22} {'('+f'{sy:.4f}'+')':>14} {'('+f'{sm_:.4f}'+')':>14}")
    print(f"{'N':22} {int(res_y.nobs):>14} {int(res_m.nobs):>14}")
    print("解读：DID 系数在两列均显著 → M 为传导渠道。")
    return {'res_y': res_y, 'res_m': res_m}
```

```r
# ============================================================
# 两步法 — R（fixest，推荐）
# ============================================================
library(fixest)

two_step_r <- function(df, outcome, mediator, did_var, controls=NULL,
                        unit_id="unit_id", time="time") {
  ctrl_str <- if (length(controls)>0) paste("+",paste(controls,collapse="+")) else ""
  clust    <- as.formula(paste0("~",unit_id))
  res_y <- feols(as.formula(sprintf("%s ~ %s %s | %s + %s",
                                     outcome,  did_var, ctrl_str, unit_id, time)),
                 data=df, cluster=clust)
  res_m <- feols(as.formula(sprintf("%s ~ %s %s | %s + %s",
                                     mediator, did_var, ctrl_str, unit_id, time)),
                 data=df, cluster=clust)
  etable(res_y, res_m,
         headers=c("Y（结果变量）","M（机制变量）"), keep=did_var,
         title="机制分析：两步法",
         notes="两列 DID 系数均显著支持 M 为 X→Y 传导渠道。双向 FE；聚类至个体。")
  return(list(res_y=res_y, res_m=res_m))
}
```

### 3.2 完整三步法

逻辑：Baron & Kenny (1986)：(1) Y~X，(2) M~X，(3) Y~X+M。关注 X 系数在加入 M 后的变化幅度（部分/完全中介）。

**审稿人可能质疑：** Step3 中 M 属潜在坏控制变量（M 受 X 影响），解释需谨慎。

```r
# ============================================================
# 完整三步法 — R（fixest）
# ============================================================
three_step_r <- function(df, outcome, mediator, did_var, controls=NULL,
                          unit_id="unit_id", time="time") {
  ctrl_str <- if (length(controls)>0) paste("+",paste(controls,collapse="+")) else ""
  clust    <- as.formula(paste0("~",unit_id))
  res_s1 <- feols(as.formula(sprintf("%s ~ %s %s | %s + %s",
                                      outcome, did_var, ctrl_str, unit_id, time)),
                  data=df, cluster=clust)
  res_s2 <- feols(as.formula(sprintf("%s ~ %s %s | %s + %s",
                                      mediator, did_var, ctrl_str, unit_id, time)),
                  data=df, cluster=clust)
  res_s3 <- feols(as.formula(sprintf("%s ~ %s + %s %s | %s + %s",
                                      outcome, did_var, mediator, ctrl_str, unit_id, time)),
                  data=df, cluster=clust)
  pct <- (coef(res_s1)[did_var]-coef(res_s3)[did_var])/coef(res_s1)[did_var]*100
  etable(res_s1, res_s2, res_s3,
         headers=c("Step1: Y~X","Step2: M~X","Step3: Y~X+M"),
         keep=c(did_var, mediator),
         title="机制分析：完整三步法（Baron & Kenny）",
         notes=paste0(sprintf("Step3 X系数变化：%.1f%%。",pct),
                      "变化>20%且M显著→部分中介。",
                      "\n[警告] Step3中M属潜在坏控制变量，解释需谨慎。"))
  return(list(s1=res_s1, s2=res_s2, s3=res_s3, pct_change=pct))
}
```

```python
# 完整三步法 — Python
def three_step_mechanism(df, outcome, mediator, did_var, controls=None,
                          unit_id='unit_id', time='time'):
    controls = controls or []
    df_panel = df.set_index([unit_id, time])
    def fit(dep, regs):
        X = sm.add_constant(df_panel[regs])
        return PanelOLS(df_panel[dep], X, entity_effects=True,
                        time_effects=True).fit(cov_type='clustered', cluster_entity=True)
    res_s1 = fit(outcome,  [did_var]+controls)
    res_s2 = fit(mediator, [did_var]+controls)
    res_s3 = fit(outcome,  [did_var, mediator]+controls)
    pct = (res_s1.params[did_var]-res_s3.params[did_var])/res_s1.params[did_var]*100
    print(f"X系数变化：{pct:.1f}%  [警告] Step3 中 M 属潜在坏控制变量。")
    return {'s1':res_s1,'s2':res_s2,'s3':res_s3,'pct_change':pct}
```

### 3.3 分组回归

逻辑：按 M 中位数分为高 / 低组，分别跑主回归。高组效应更强说明 M 是传导渠道。组间差异用 Fisher 置换检验（单位级别打乱）。

**补充：三分位 / 四分位分组 + 多切割点敏感性分析图。**

```r
# ============================================================
# 分组回归 + Fisher 置换检验 — R（完整版）
# ============================================================
library(fixest); library(dplyr); library(ggplot2)

subgroup_mechanism_r <- function(df, outcome, mediator, did_var, controls=NULL,
                                  unit_id="unit_id", time="time",
                                  n_quantiles=2, n_perm=1000, seed=42) {
  set.seed(seed)
  ctrl_str <- if (length(controls)>0) paste("+",paste(controls,collapse="+")) else ""
  clust    <- as.formula(paste0("~",unit_id))
  df       <- df %>% mutate(m_group=ntile(.data[[mediator]], n_quantiles))
  fml      <- as.formula(sprintf("%s ~ %s %s | %s + %s",
                                  outcome, did_var, ctrl_str, unit_id, time))

  # 分组回归
  res_df <- bind_rows(lapply(1:n_quantiles, function(g) {
    res <- tryCatch(feols(fml, data=df[df$m_group==g,], cluster=clust),
                    error=function(e) NULL)
    if (!is.null(res)) data.frame(group=g, coef=coef(res)[did_var],
                                   se=se(res)[did_var], pval=pvalue(res)[did_var],
                                   ci_lo=coef(res)[did_var]-1.96*se(res)[did_var],
                                   ci_hi=coef(res)[did_var]+1.96*se(res)[did_var])
  }))
  print(res_df)
  diff_obs <- max(res_df$coef) - min(res_df$coef)

  # Fisher 置换（单位级别）
  ug <- df %>% distinct(.data[[unit_id]], m_group)
  fake_diffs <- sapply(seq_len(n_perm), function(i) {
    shuf <- ug %>% mutate(m_group=sample(m_group))
    df_p <- df %>% select(-.data$m_group) %>%
      left_join(shuf %>% rename(m_group=m_group), by=unit_id)
    gc <- sapply(1:n_quantiles, function(g) {
      rf <- tryCatch(feols(fml,data=df_p[df_p$m_group==g,],cluster=clust),
                     error=function(e) NULL)
      if (!is.null(rf)) coef(rf)[did_var] else NA_real_
    })
    if (all(!is.na(gc))) max(gc)-min(gc) else NA_real_
  })
  fake_diffs <- na.omit(fake_diffs)
  pval_perm  <- mean(abs(fake_diffs) >= abs(diff_obs))
  cat(sprintf("Fisher 置换 p 值：%.4f（%d 次）\n", pval_perm, length(fake_diffs)))

  # 系数图
  labels <- switch(as.character(n_quantiles),
                   "2"=c("低 M 组","高 M 组"), "3"=c("低 M","中 M","高 M"),
                   paste0("Q", 1:n_quantiles))
  res_df$label <- labels[res_df$group]
  p <- ggplot(res_df, aes(x=label, y=coef, ymin=ci_lo, ymax=ci_hi)) +
    geom_col(fill="#20808D", alpha=0.7, width=0.5) +
    geom_errorbar(width=0.2, color="#1B474D", lw=1) +
    geom_hline(yintercept=0, lty="dashed", color="gray50") +
    labs(x=sprintf("M 分组（%d分位）",n_quantiles), y="处理效应系数",
         title="分组回归：M 分组的处理效应",
         caption=sprintf("Fisher p=%.4f（%d次）",pval_perm,length(fake_diffs))) +
    theme_minimal(base_size=12)
  ggsave("output/mechanism_subgroup.png", p, width=7, height=4, dpi=150)
  return(list(results=res_df, pval_perm=pval_perm))
}
```

```python
# 分组回归 + Fisher 置换 — Python
def subgroup_mechanism(df, outcome, mediator, did_var, controls=None,
                        unit_id='unit_id', time='time',
                        n_quantiles=2, n_perm=1000, seed=42):
    np.random.seed(seed)
    controls = controls or []
    df = df.copy()
    df['m_group'] = pd.qcut(df[mediator], q=n_quantiles, labels=False)
    def fit_g(data_g):
        dp = data_g.set_index([unit_id, time])
        try:
            return PanelOLS(dp[outcome],
                            sm.add_constant(dp[[did_var]+controls]),
                            entity_effects=True, time_effects=True
                           ).fit(cov_type='clustered', cluster_entity=True)
        except: return None
    group_res = {}
    for g in range(n_quantiles):
        res = fit_g(df[df['m_group']==g])
        if res: group_res[g] = {'coef':res.params[did_var],'pval':res.pvalues[did_var]}
        print(f"组{g}: {group_res.get(g,{})}")
    diff_obs = max(r['coef'] for r in group_res.values()) - \
               min(r['coef'] for r in group_res.values())
    units = df[unit_id].unique()
    gmap  = df.drop_duplicates(unit_id).set_index(unit_id)['m_group']
    fake_diffs = []
    for _ in range(n_perm):
        perm = np.random.permutation(gmap.values)
        df['fg'] = df[unit_id].map(dict(zip(gmap.index, perm)))
        gc = [fit_g(df[df['fg']==g]) for g in range(n_quantiles)]
        gc = [r.params[did_var] for r in gc if r]
        if len(gc)==n_quantiles: fake_diffs.append(max(gc)-min(gc))
    p_val = np.mean(np.abs(fake_diffs) >= np.abs(diff_obs))
    print(f"Fisher p={p_val:.4f}（{len(fake_diffs)}次）")
    return {'group_results':group_res,'p_value':p_val}
```

### 3.4 交互项法

逻辑：在主回归中加入 X×M 交互项，系数 β₃ 代表 M 对 X→Y 效应的调节强度。绘制边际效应图。

**适用：** M 是调节变量（M 不被 X 影响）。

```r
# ============================================================
# 交互项法 + 边际效应图 — R（fixest + marginaleffects）
# ============================================================
library(fixest); library(marginaleffects); library(ggplot2); library(dplyr)

interaction_mechanism_r <- function(df, outcome, moderator, did_var, controls=NULL,
                                     unit_id="unit_id", time="time") {
  df      <- df %>% mutate(M_c=as.numeric(scale(.data[[moderator]], center=TRUE, scale=FALSE)))
  ctrl_str <- if (length(controls)>0) paste("+",paste(controls,collapse="+")) else ""
  res_int <- feols(as.formula(sprintf("%s ~ %s * M_c %s | %s + %s",
                                       outcome, did_var, ctrl_str, unit_id, time)),
                   data=df, cluster=as.formula(paste0("~",unit_id)))
  etable(res_int, keep=c(did_var,"M_c",paste0(did_var,":M_c")),
         title="调节效应：X × M 交互项",
         notes="M 已中心化；交互项系数>0=正向调节。")

  M_rng <- seq(quantile(df$M_c,.05,na.rm=TRUE), quantile(df$M_c,.95,na.rm=TRUE), l=50)
  me_df <- marginaleffects::slopes(res_int, variables=did_var,
                                    newdata=datagrid(M_c=M_rng))
  p <- ggplot(me_df, aes(x=M_c, y=estimate)) +
    geom_ribbon(aes(ymin=conf.low, ymax=conf.high), alpha=0.2, fill="#20808D") +
    geom_line(color="#20808D", lw=1.3) +
    geom_hline(yintercept=0, lty="dashed", color="gray50") +
    labs(x=sprintf("%s（中心化）",moderator), y="处理效应（边际效应）",
         title="交互项法：处理效应随 M 变化") +
    theme_minimal(base_size=12)
  ggsave("output/mechanism_marginal.png", p, width=8, height=5, dpi=150)
  return(list(result=res_int, plot=p))
}
```

```python
# 交互项法 — Python（within demean + delta method SE）
def interaction_mechanism_python(df, outcome, moderator, did_var, controls=None,
                                   unit_id='unit_id', time='time'):
    controls = controls or []
    df = df.copy()
    df['M_c'] = df[moderator] - df[moderator].mean()
    df[f'{did_var}_x_M'] = df[did_var] * df['M_c']
    for col in [outcome, did_var, 'M_c', f'{did_var}_x_M']+controls:
        df[col+'_dm'] = df[col] - df.groupby(unit_id)[col].transform('mean')
    regs = [f'{did_var}_dm','M_c_dm',f'{did_var}_x_M_dm']+[c+'_dm' for c in controls]
    mod = sm.OLS(df[f'{outcome}_dm'], sm.add_constant(df[regs])).fit(
        cov_type='cluster', cov_kwds={'groups':df[unit_id]})
    int_coef = mod.params[f'{did_var}_x_M_dm']
    print(f"交互项系数：{int_coef:.4f}（>0 = 正向调节）")
    return {'result': mod}
```

### 3.5 因果中介效应

适用于 M 存在内生性时。**必须做 `medsens()` 敏感性分析，报告 ρ 临界值。**

```r
# ============================================================
# 因果中介效应 — R（mediation 包）
# medsens() 为必做步骤！
# ============================================================
library(mediation); library(dplyr)

causal_mediation_r <- function(df, outcome, mediator, treatment, controls=NULL,
                                unit_id="unit_id", n_sims=1000, seed=42) {
  """
  声明：Sequential Ignorability 假设（类似中介随机化），必须在论文中讨论合理性。
  """
  set.seed(seed)
  controls <- controls %||% character(0)
  # 面板 Within 去均值（模拟 FE）
  df_dm <- df %>% group_by(.data[[unit_id]]) %>%
    mutate(across(all_of(c(outcome,mediator,treatment,controls)),
                  ~.-mean(.,na.rm=TRUE))) %>% ungroup()
  ctrl_str <- if (length(controls)>0) paste("+",paste(controls,collapse="+")) else ""
  med_fit <- lm(as.formula(sprintf("%s ~ %s %s",mediator,treatment,ctrl_str)), data=df_dm)
  out_fit <- lm(as.formula(sprintf("%s ~ %s + %s %s",outcome,treatment,mediator,ctrl_str)), data=df_dm)

  med_out <- mediate(model.m=med_fit, model.y=out_fit, treat=treatment,
                     mediator=mediator, sims=n_sims, boot=TRUE, boot.ci.type="perc")
  summary(med_out)  # ACME / ADE / Total / Prop.Mediated

  # ── 必做：medsens() 敏感性分析 ──────────────────────────
  sens_out <- medsens(med_out, rho.by=0.05, effect.type="indirect", sims=200)
  rho_crit <- sens_out$rho[which.min(abs(sens_out$d0))]
  cat(sprintf("[敏感性] ρ 临界值=%.2f。|ρ|<0.1时结论对假设偏离非常敏感，需在论文中说明。\n",
              rho_crit))

  return(list(med_out=med_out, sens_out=sens_out, rho_crit=rho_crit))
}
```

```python
# ============================================================
# 因果中介效应 — Python（Cluster Bootstrap，按 unit_id 重抽）
# 注意：Bootstrap 必须在 unit_id 层面重抽，不是观测级别！
# ============================================================
def causal_mediation_python(df, outcome, mediator, treatment, controls=None,
                              unit_id='unit_id', n_boot=1000, seed=42):
    """
    Sequential Ignorability 假设，需在论文中讨论合理性。
    Returns: dict{ACME, ADE, Total_Effect, Prop_Mediated} 含 95% CI
    """
    np.random.seed(seed)
    controls = controls or []
    def fit_med(data):
        Xm = sm.add_constant(data[[treatment]+controls])
        mm = sm.OLS(data[mediator], Xm).fit()
        Xy = sm.add_constant(data[[treatment, mediator]+controls])
        my = sm.OLS(data[outcome], Xy).fit()
        a  = mm.params[treatment]; b = my.params[mediator]; g = my.params[treatment]
        acme = a*b; ade = g; tot = acme+ade
        return acme, ade, tot, (acme/tot if abs(tot)>1e-10 else np.nan)

    obs = fit_med(df)
    units = df[unit_id].unique()
    boot_res = []
    for _ in range(n_boot):
        # Cluster Bootstrap：按 unit_id 有放回抽样
        su = np.random.choice(units, size=len(units), replace=True)
        bd = pd.concat([df[df[unit_id]==u] for u in su], ignore_index=True)
        try: boot_res.append(fit_med(bd))
        except: pass
    boot = np.array(boot_res)
    ci_lo, ci_hi = np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)
    for i, k in enumerate(['ACME','ADE','Total_Effect','Prop_Mediated']):
        print(f"{k:18}: {obs[i]:>8.4f}  95%CI=[{ci_lo[i]:.4f}, {ci_hi[i]:.4f}]")
    print("[声明] Sequential Ignorability 假设需在论文中讨论。Cluster Bootstrap 按 unit_id 重抽。")
    return {k: (obs[i], ci_lo[i], ci_hi[i])
            for i, k in enumerate(['ACME','ADE','Total_Effect','Prop_Mediated'])}
```

---

## 4. 残差法（Bleemer-Mehta 风格）

**标注：经济学顶刊前沿（Bleemer & Mehta 2022, AER），管理学期刊接受度有限。**

```r
# ============================================================
# 残差法 — R（含 Cluster Bootstrap SE）
# ============================================================
library(fixest); library(dplyr)

residual_mechanism_r <- function(df, outcome, mediator, did_var, controls=NULL,
                                  unit_id="unit_id", time="time",
                                  n_boot=500, seed=42) {
  set.seed(seed)
  ctrl_str <- if (length(controls)>0) paste("+",paste(controls,collapse="+")) else ""
  clust    <- as.formula(paste0("~",unit_id))

  res_total <- feols(as.formula(sprintf("%s ~ %s %s | %s+%s",
                                         outcome,did_var,ctrl_str,unit_id,time)),
                     data=df, cluster=clust)
  beta_total <- coef(res_total)[did_var]

  # 用控制组（post=0 | treated=0）估计 M→Y，取残差
  df_pre <- df %>% filter(post==0 | treated==0)
  res_m_y <- feols(as.formula(sprintf("%s ~ %s %s | %s+%s",
                                       outcome,mediator,ctrl_str,unit_id,time)),
                   data=df_pre)
  df$resid_ym <- df[[outcome]] - predict(res_m_y, newdata=df)
  res_dir <- feols(as.formula(sprintf("resid_ym ~ %s %s | %s+%s",
                                       did_var,ctrl_str,unit_id,time)),
                   data=df, cluster=clust)
  beta_dir   <- coef(res_dir)[did_var]
  beta_indir <- beta_total - beta_dir
  cat(sprintf("总效应：%.4f  直接：%.4f(%.1f%%)  间接：%.4f(%.1f%%)\n",
              beta_total, beta_dir, beta_dir/beta_total*100,
              beta_indir, beta_indir/beta_total*100))

  # Cluster Bootstrap SE
  units <- unique(df[[unit_id]])
  boot_i <- sapply(seq_len(n_boot), function(b) {
    bu <- sample(units, replace=TRUE)
    df_b <- do.call(rbind, lapply(bu, function(u) df[df[[unit_id]]==u,]))
    rt <- tryCatch(feols(as.formula(sprintf("%s~%s|%s+%s",
                                            outcome,did_var,unit_id,time)),data=df_b),error=function(e)NULL)
    df_pre_b <- df_b %>% filter(post==0|treated==0)
    rmy <- tryCatch(feols(as.formula(sprintf("%s~%s|%s+%s",
                                             outcome,mediator,unit_id,time)),data=df_pre_b),error=function(e)NULL)
    if (!is.null(rt)&&!is.null(rmy)){
      df_b$resid_b <- df_b[[outcome]]-predict(rmy,newdata=df_b)
      rd <- tryCatch(feols(resid_b~get(did_var)|unit_id+time,data=df_b),error=function(e)NULL)
      if (!is.null(rd)) coef(rt)[did_var]-coef(rd)[did_var] else NA_real_
    } else NA_real_
  })
  boot_i <- na.omit(boot_i)
  cat(sprintf("间接效应 SE(Bootstrap)：%.4f  95%%CI：[%.4f, %.4f]\n",
              sd(boot_i), quantile(boot_i,.025), quantile(boot_i,.975)))
  cat("[标注] 残差法属经济学顶刊前沿方法（Bleemer & Mehta 2022, AER），管理学接受度有限。\n")
  return(list(beta_total=beta_total, beta_indirect=beta_indir,
              se_indirect=sd(boot_i)))
}
```

---

## 5. 多中介变量竞争性检验

分别跑 M1 ~ X, M2 ~ X, M3 ~ X 并排比较，验证哪条机制路径更强。

```r
# ============================================================
# 多中介竞争性检验 — R
# ============================================================
competitive_mediators_r <- function(df, outcome, mediators, did_var,
                                     controls=NULL, unit_id="unit_id", time="time") {
  # mediators: named list，例 list(融资约束='sa_index', 信息不对称='bid_ask')
  ctrl_str <- if (length(controls)>0) paste("+",paste(controls,collapse="+")) else ""
  clust    <- as.formula(paste0("~",unit_id))
  res_y <- feols(as.formula(sprintf("%s~%s%s|%s+%s",outcome,did_var,ctrl_str,unit_id,time)),
                 data=df, cluster=clust)
  res_list  <- list(Y=res_y)
  hdr_names <- c("Y（结果变量）")
  for (nm in names(mediators)) {
    res_list[[nm]] <- feols(as.formula(sprintf("%s~%s%s|%s+%s",
                                                mediators[[nm]],did_var,ctrl_str,unit_id,time)),
                            data=df, cluster=clust)
    hdr_names <- c(hdr_names, nm)
  }
  do.call(etable, c(res_list, list(headers=hdr_names, keep=did_var,
                                    title="多中介变量竞争性检验",
                                    notes="DID 系数显著的中介变量为优先支持的机制路径。")))
  return(res_list)
}
```

---

## 6. 坏控制变量警告

```python
# ============================================================
# 坏控制变量检查 — Python（函数参数传入，不用 input()）
# ============================================================
def check_bad_control(M_name, X_name, Y_name,
                       M_affected_by_X=None,
                       M_before_X=None,
                       Y_affects_M=None):
    """
    Parameters
    ----------
    M_affected_by_X : bool，M 是否受 X 影响
    M_before_X      : bool，M 是否在 X 处理前就已确定
    Y_affects_M     : bool，Y 是否反向影响 M（对撞变量判断）

    Returns: str（角色：'mediator'/'moderator'/'collider'/'predetermined'/'unclear'）
    """
    if Y_affects_M is True:
        print(f"[拒绝执行] {M_name} 是对撞变量：{X_name}→{M_name}←{Y_name}")
        print("加入回归将引入虚假关联。请重新检查 DAG，修改假设。")
        return 'collider'
    if M_before_X is True:
        print(f"[警告] {M_name} 在处理前已确定 → 前定变量（非中介）。推荐用交互项法。")
        return 'predetermined'
    if M_affected_by_X is False:
        print(f"[信息] {M_name} 不受 {X_name} 影响 → 调节变量。推荐交互项法。")
        return 'moderator'
    if M_affected_by_X is True and M_before_X is False:
        print(f"[通过] {M_name} 符合中介变量条件。推荐两步法或因果中介效应。")
        return 'mediator'
    print("[不确定] 回到 DAG 重新梳理因果关系。")
    return 'unclear'

# 调用示例：
# role = check_bad_control("融资约束指数","政策处理","企业投资",
#                           M_affected_by_X=True, M_before_X=False, Y_affects_M=False)
```

---

## 7. 机制变量平行趋势

若主回归使用 DID，机制变量 M 也需满足平行趋势假设，否则 X→M 的证据本身存在内生性问题。

```r
# ============================================================
# 机制变量事件研究图（与主回归并排）— R
# ============================================================
library(fixest); library(ggplot2); library(dplyr)

mechanism_parallel_trend_r <- function(df, outcome, mediator,
                                        unit_id="unit_id", time="time",
                                        treatment="treated", ref_period=-1) {
  clust <- as.formula(paste0("~",unit_id))
  es_fml <- function(dep)
    as.formula(sprintf("%s ~ i(event_time,%s,ref=%d) | %s + %s",
                        dep, treatment, ref_period, unit_id, time))

  res_y_es <- feols(es_fml(outcome),  data=df, cluster=clust)
  res_m_es <- feols(es_fml(mediator), data=df, cluster=clust)

  extract_es <- function(res, label) {
    cf <- coef(res); ci <- confint(res)
    idx <- grep("event_time", names(cf))
    tt  <- as.numeric(gsub(".*::(\\-?[0-9]+).*","\\1",names(cf)[idx]))
    rbind(data.frame(event_time=c(tt,ref_period),
                     coef=c(cf[idx],0), ci_lo=c(ci[idx,1],0), ci_hi=c(ci[idx,2],0),
                     variable=label))
  }

  df_es <- rbind(extract_es(res_y_es,"Y（结果变量）"),
                 extract_es(res_m_es,"M（机制变量）"))

  p <- ggplot(df_es, aes(x=event_time, y=coef, color=variable, fill=variable)) +
    geom_hline(yintercept=0, lty="dashed", color="gray50") +
    geom_ribbon(aes(ymin=ci_lo, ymax=ci_hi), alpha=0.15, color=NA) +
    geom_line(lw=1.1) + geom_point(size=2) +
    facet_wrap(~variable, scales="free_y") +
    scale_color_manual(values=c("Y（结果变量）"="#20808D","M（机制变量）"="#A84B2F")) +
    scale_fill_manual(values =c("Y（结果变量）"="#20808D","M（机制变量）"="#A84B2F")) +
    labs(x="相对处理期（ref=-1）", y="系数（ATT）",
         title="主回归（Y）与机制变量（M）事件研究对比",
         caption="M 在处理前平行趋势成立 → 支持 X→M 的因果解释") +
    theme_minimal(base_size=12) + theme(legend.position="none")
  ggsave("output/mechanism_event_study.png", p, width=10, height=5, dpi=150)

  # M 的 Pre-period 联合检验
  pre_m <- grep("event_time::-[2-9]", names(coef(res_m_es)), value=TRUE)
  if (length(pre_m)>0) {
    wt <- wald(res_m_es, keep=pre_m)
    cat(sprintf("M Pre-period 联合 F=%.3f, p=%.4f  通过(p>0.10): %s\n",
                wt$stat, wt$p, wt$p>0.10))
  }
  return(list(res_y=res_y_es, res_m=res_m_es, plot=p))
}
```

---

## 方法对照表

| 方法 | 适用场景 | 优势 | 局限 | 适用期刊范围 |
|------|----------|------|------|-------------|
| **两步法** | 只需证明 X→M 存在 | 无坏控制变量问题；直觉强 | 无法量化间接效应占比 | 经济学顶刊（AER/QJE/JFE）|
| **完整三步法** | 兼容传统审稿人 | 展示中介系数变化幅度 | Step3 M 是坏控制变量 | 管理学（AMJ/SMJ）|
| **分组回归** | M 可二值化（高/低）| 无函数形式假设 | 分组切割点主观 | 各类实证期刊 |
| **交互项法** | M 是调节变量（不受 X 影响）| 正式估计调节效应 | 需要 M 外生 | 顶刊 + 管理学 |
| **因果中介** | M 内生，需 Sequential Ignorability | 正式分解 IE/DE + 敏感性 | 假设强，需 medsens() | 方法论期刊 / 顶刊附录 |
| **残差法** | 量化间接效应比例 | 不引入坏控制变量 | 审稿人不熟悉 | 经济学顶刊前沿 |
