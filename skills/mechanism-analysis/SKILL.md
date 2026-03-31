# 机制分析

机制分析（Mechanism Analysis）回答"为什么 X 影响 Y"，通过引入中介变量或调节变量揭示因果链条。在建立基准因果效应之后进行，是顶刊论文从"好到卓越"的关键步骤。

---

## 0. 强制前置：DAG绘制

**在跑任何机制回归之前，必须画因果图（Directed Acyclic Graph, DAG）。** 因果图明确 X、Y、M 之间的方向关系，决定 M 是中介、调节还是对撞变量，从而决定后续回归策略。

### 为什么 DAG 是强制的？

将后门路径上的变量（对撞变量）加入回归会**引入新的偏误**，甚至让无关变量看起来显著。DAG 先行可防止"坏控制变量（Bad Controls）"问题。

### dagitty R 语法示例

```r
# ============================================================
# DAG 绘制 — R（dagitty 包）
# install.packages("dagitty")
# install.packages("ggdag")      # 可视化
# ============================================================
library(dagitty)
library(ggdag)

# ── 示例：X（政策处理）→ M（融资约束）→ Y（企业投资）
# 同时 Z（宏观冲击）影响 X 和 Y（需控制）
dag_spec <- dagitty('
  dag {
    X [exposure, pos="0,1"]
    Y [outcome,  pos="2,1"]
    M [pos="1,1"]
    Z [pos="1,2"]

    X -> M -> Y        # 中介路径（机制）
    X -> Y             # 直接效应
    Z -> X             # Z 影响处理
    Z -> Y             # Z 影响结果（混淆因子，需控制）
  }
')

# 检查 M 在 DAG 中的角色
# isAncestorOf(): M 是 Y 的祖先节点 → 中介变量
# isDescendantOf(): M 是 X 的后代 → 受 X 影响（潜在中介）
cat("M 是 Y 的祖先:", isAncestorOf(dag_spec, "M", "Y"), "\n")
cat("M 是 X 的后代:", isDescendantOf(dag_spec, "X", "M"), "\n")

# 可视化 DAG
ggdag(dag_spec, layout = "circle") +
  geom_dag_point(color = "steelblue") +
  geom_dag_text(color = "white") +
  geom_dag_edges(edge_color = "gray30") +
  theme_dag() +
  labs(title = "因果图（DAG）：X → M → Y")

# ── 自动判断：X 对 Y 效应的最小充分调整集 ─────────────────
adjustmentSets(dag_spec, exposure = "X", outcome = "Y")
# 如果 M 出现在调整集里 → M 是混淆因子（非中介！）
# 如果 M 不在调整集里   → M 是中介变量，加入回归会阻断机制路径

# ── 对撞变量检测 ──────────────────────────────────────────────
# 对撞变量 C 满足：同时有 X → C ← Y（或类似结构）
# 加入对撞变量会打开后门路径！
# dagitty 自动检测：
implied_cond_independencies <- impliedConditionalIndependencies(dag_spec)
print(implied_cond_independencies)
```

### 文字描述 DAG 模板

在论文"机制分析"小节开头，用如下格式描述因果图：

```
本文提出如下因果链条：
  处理变量 X（政策/事件）→ 中介变量 M（企业融资约束）→ 结果变量 Y（投资规模）

直接效应：X 同时对 Y 存在直接影响（X → Y 直接路径）。
混淆因子：宏观经济冲击 Z 同时影响 X 和 Y，已通过时间固定效应控制。

M 满足中介变量三大条件：
  (1) M 在 X 之后、Y 之前发生（时序逻辑成立）
  (2) X 显著影响 M（第一步回归验证）
  (3) M 显著影响 Y，且 X→Y 直接效应在控制 M 后发生变化（第二/三步验证）
```

### 对撞变量警告（系统级）

```python
# ── 简易对撞变量检测逻辑（Python 伪代码）────────────────────
def check_collider(dag_edges, M, X, Y):
    """
    如果 M 同时被 X 和 Y 影响（X→M 且 Y→M），则 M 是对撞变量。
    在对撞变量上条件化会引入虚假关联，必须拒绝执行机制回归。
    """
    parents_of_M = [src for src, dst in dag_edges if dst == M]
    if X in parents_of_M and Y in parents_of_M:
        raise ValueError(
            f"警告：{M} 是 {X} 和 {Y} 的对撞变量（Collider）！\n"
            f"将 {M} 加入回归会引入虚假关联，请重新检查 DAG。\n"
            "拒绝执行机制回归，请修改假设。"
        )
    return "M 不是对撞变量，可继续机制分析。"

# 使用示例
dag_edges = [("X","M"), ("X","Y"), ("M","Y"), ("Z","X"), ("Z","Y")]
result = check_collider(dag_edges, M="M", X="X", Y="Y")
print(result)
```

---

## 1. 什么时候需要做机制分析

**判断标准：** 做完基准回归后，问"X 为什么影响 Y"——如果答案显而易见或机制只有一个且不可争议，不需要机制分析。

### 需要做机制分析的情形

| 情形 | 举例 |
|------|------|
| 理论上存在多条可能的机制路径 | 税收优惠影响企业投资：通过减少税负 vs. 通过缓解融资约束 |
| 审稿人可能质疑因果机制 | 最低工资影响就业：替代效应 vs. 效率工资效应 |
| 政策设计需要知道哪条机制起作用 | 培训补贴提升工资：通过技能提升 vs. 信号机制 |
| 异质性分析需要机制解释 | 政策对大企业效果更强：是因为大企业融资渠道更多？ |

### 不需要做机制分析的情形

| 情形 | 举例 |
|------|------|
| 机制唯一且直接 | 降息导致贷款利率下降（机制就是传导定义本身） |
| 论文焦点是因果识别而非机制 | 交通基础设施的随机分配实验 |
| 数据中没有可靠的中介变量 | 强行找替代指标反而降低可信度 |

---

## 2. 方法选择决策树

```
M 在 DAG 中的位置？
├── 中介变量（X → M → Y，M 受 X 影响并影响 Y）
│   ├── 只需证明 X → M 存在（不量化间接效应占比）
│   │   → 两步法（默认推荐，顶刊接受度最高）
│   │
│   ├── 需要量化间接效应占比（间接效应 / 总效应）
│   │   → 残差法（Bleemer-Mehta 风格）
│   │   [标注：经济学顶刊前沿方法，管理学期刊接受度有限]
│   │
│   ├── M 本身存在内生性（M 受遗漏变量影响）
│   │   → 因果中介效应（mediation 包，需额外 IV 或随机化假设）
│   │
│   └── 需兼容传统审稿人预期（完整三步法）
│       → 完整三步法或三步法变体
│       [标注：三步法第三步的 M 属于坏控制变量，部分审稿人会质疑]
│
├── 调节变量（M 影响 X→Y 的效应强度，但 M 不被 X 影响）
│   ├── M 是连续变量
│   │   → 交互项法（Y ~ X + M + X×M）+ 边际效应图
│   └── M 是离散/二元变量
│       → 分组回归（按 M 分组，组间系数差异检验）
│
└── 不确定 M 的角色
    → 回到 DAG，重新确认时序逻辑和因果方向
```

---

## 3. 七种方法代码模板

### 3.1 两步法（默认推荐）

逻辑：分别跑 Y ~ X 和 M ~ X，证明 X 同时影响 Y 和 M，从而支持"M 是 X 影响 Y 的渠道"这一说法。**不要求量化间接效应，回避了坏控制变量问题，是顶刊首选。**

```python
# ============================================================
# 两步法 — Python（linearmodels / statsmodels）
# ============================================================
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

# ── 第一步：主回归 Y ~ X（验证主效应）──────────────────────
df_panel = df.set_index(['unit_id', 'time'])
df_panel['did'] = df_panel['post'] * df_panel['treated']

mod_y = PanelOLS(
    dependent=df_panel['outcome'],           # 结果变量 Y
    exog=sm.add_constant(df_panel[['did'] + control_vars]),
    entity_effects=True, time_effects=True
)
res_y = mod_y.fit(cov_type='clustered', cluster_entity=True)

# ── 第二步：机制回归 M ~ X（验证 X → M）───────────────────
mod_m = PanelOLS(
    dependent=df_panel['mediator'],          # 中介变量 M
    exog=sm.add_constant(df_panel[['did'] + control_vars]),
    entity_effects=True, time_effects=True
)
res_m = mod_m.fit(cov_type='clustered', cluster_entity=True)

# ── 并排输出系数（手动格式化）──────────────────────────────
print("=" * 50)
print(f"{'':20} {'Y（结果）':>12} {'M（机制）':>12}")
print("=" * 50)
for var in ['did'] + control_vars:
    coef_y = res_y.params.get(var, np.nan)
    coef_m = res_m.params.get(var, np.nan)
    se_y   = res_y.std_errors.get(var, np.nan)
    se_m   = res_m.std_errors.get(var, np.nan)
    print(f"{var:20} {coef_y:>12.4f} {coef_m:>12.4f}")
    print(f"{'':20} {'(' + f'{se_y:.4f}' + ')':>12} {'(' + f'{se_m:.4f}' + ')':>12}")
print("=" * 50)
print(f"{'Obs':20} {int(res_y.nobs):>12} {int(res_m.nobs):>12}")
print("两步法解读：DID 系数在两列均显著，支持 M 为机制渠道。")
```

```r
# ============================================================
# 两步法 — R（fixest，推荐）
# ============================================================
library(fixest)

# 第一步：主效应
res_step1_y <- feols(
  outcome  ~ i(post, treated, ref = 0) + control1 + control2 | unit_id + time,
  data = df, cluster = ~unit_id
)

# 第二步：机制变量
res_step1_m <- feols(
  mediator ~ i(post, treated, ref = 0) + control1 + control2 | unit_id + time,
  data = df, cluster = ~unit_id
)

# 并排输出（两步法标准表格）
etable(res_step1_y, res_step1_m,
       headers    = c("Y（结果变量）", "M（机制变量）"),
       keep       = "post",      # 只展示 DID 系数
       title      = "机制分析：两步法",
       notes      = "两列 DID 系数均显著支持 M 为 X→Y 的传导渠道。固定效应：个体+时间。聚类标准误至个体层面。")
```

---

### 3.2 完整三步法

逻辑：Baron & Kenny (1986) 经典步骤：(1) X→Y，(2) X→M，(3) X+M→Y。关注第三步中 X 系数在加入 M 后的变化幅度（部分中介 vs. 完全中介）。

**注意：** 第三步中同时放入 X 和 M，M 属于潜在的坏控制变量（可能阻断感兴趣的路径），部分审稿人会质疑。如果 X→M 路径是核心机制，加入 M 会导致 X 系数低估直接效应。

```r
# ============================================================
# 完整三步法 — R（fixest）
# ============================================================
library(fixest)

# Step 1: Y ~ X（总效应）
res_s1 <- feols(
  outcome ~ i(post, treated, ref=0) + control1 + control2 | unit_id + time,
  data = df, cluster = ~unit_id
)

# Step 2: M ~ X（X → M 路径）
res_s2 <- feols(
  mediator ~ i(post, treated, ref=0) + control1 + control2 | unit_id + time,
  data = df, cluster = ~unit_id
)

# Step 3: Y ~ X + M（加入中介后 X 系数变化）
res_s3 <- feols(
  outcome ~ i(post, treated, ref=0) + mediator + control1 + control2 | unit_id + time,
  data = df, cluster = ~unit_id
)

# 三列表格，标注 X 系数变化幅度
etable(res_s1, res_s2, res_s3,
       headers  = c("Step1: Y~X", "Step2: M~X", "Step3: Y~X+M"),
       keep     = "post|mediator",
       title    = "机制分析：完整三步法（Baron & Kenny）",
       notes    = paste0(
         "Step3 中 X 系数变化：",
         round((coef(res_s1)["post::1:treated"] -
                coef(res_s3)["post::1:treated"]) /
               coef(res_s1)["post::1:treated"] * 100, 1),
         "%。若变化 > 20% 且 M 系数显著，支持部分中介。",
         "\n警告：Step3 中 M 属于潜在坏控制变量（M 受 X 影响），解释需谨慎。"
       ))
```

---

### 3.3 三步法变体

逻辑：回避第三步中 X 和 M 共存的共线性问题——Step2 跑 M~X，Step3 跑 Y~M（不放 X）。适合 X 和 M 高度相关、第三步多重共线性严重的场景。

```r
# ============================================================
# 三步法变体 — R（M~X + Y~M，不在第三步放 X）
# ============================================================
library(fixest)

# Step 1: Y ~ X（总效应，同完整三步法）
res_v1 <- feols(
  outcome ~ i(post, treated, ref=0) + control1 + control2 | unit_id + time,
  data = df, cluster = ~unit_id
)

# Step 2: M ~ X（X 对机制变量的影响）
res_v2 <- feols(
  mediator ~ i(post, treated, ref=0) + control1 + control2 | unit_id + time,
  data = df, cluster = ~unit_id
)

# Step 3（变体）: Y ~ M（不放 X，避免共线性）
res_v3 <- feols(
  outcome ~ mediator + control1 + control2 | unit_id + time,
  data = df, cluster = ~unit_id
)

etable(res_v1, res_v2, res_v3,
       headers = c("Step1: Y~X（总效应）",
                   "Step2: M~X",
                   "Step3变体: Y~M"),
       keep    = "post|mediator",
       title   = "机制分析：三步法变体（回避共线性）",
       notes   = "Step3 不放 X，避免 X 与 M 高度相关导致的多重共线性。适合 X 和 M 相关性 > 0.7 的场景。")
```

---

### 3.4 分组回归做机制

逻辑：按中介变量 M 的中位数将样本分为高 M 组和低 M 组，分别跑主回归。若 X 对 Y 的影响在高 M 组更强（或仅在高 M 组显著），说明 M 是传导渠道之一。组间系数差异用 Fisher 置换检验检验统计显著性。

**适用场景：** M 是可分组的离散/有序变量，或可按中位数二值化。

```r
# ============================================================
# 分组回归做机制 — R（fixest + Fisher 置换检验）
# ============================================================
library(fixest)
library(dplyr)

# ── Step 1：按 M 中位数分组 ──────────────────────────────────
m_median <- median(df$mediator, na.rm = TRUE)
df <- df %>%
  mutate(m_group = ifelse(mediator >= m_median, "high", "low"))

# ── Step 2：分组回归 ──────────────────────────────────────────
res_hi <- feols(
  outcome ~ i(post, treated, ref=0) + control1 + control2 | unit_id + time,
  data = df %>% filter(m_group == "high"), cluster = ~unit_id
)
res_lo <- feols(
  outcome ~ i(post, treated, ref=0) + control1 + control2 | unit_id + time,
  data = df %>% filter(m_group == "low"),  cluster = ~unit_id
)

beta_hi <- coef(res_hi)["post::1:treated"]
beta_lo <- coef(res_lo)["post::1:treated"]
diff_obs <- beta_hi - beta_lo
cat(sprintf("高 M 组系数: %.4f，低 M 组系数: %.4f，差值: %.4f\n",
            beta_hi, beta_lo, diff_obs))

etable(res_hi, res_lo,
       headers = c("高 M 组", "低 M 组"),
       keep    = "post",
       title   = "分组回归：按中介变量中位数分组")

# ── Step 3：Fisher 置换检验（组间系数差异）───────────────────
set.seed(42)
N_PERM <- 1000

units_group <- df %>%
  select(unit_id, m_group) %>%
  distinct()

fake_diffs <- numeric(N_PERM)
for (i in seq_len(N_PERM)) {
  # 在单位级别随机打乱组别标签
  shuffled <- units_group %>%
    mutate(fake_group = sample(m_group))

  df_p <- df %>%
    left_join(shuffled %>% select(unit_id, fake_group), by = "unit_id")

  r_hi <- tryCatch(
    feols(outcome ~ i(post,treated,ref=0) | unit_id+time,
          data=df_p %>% filter(fake_group=="high"), cluster=~unit_id),
    error = function(e) NULL
  )
  r_lo <- tryCatch(
    feols(outcome ~ i(post,treated,ref=0) | unit_id+time,
          data=df_p %>% filter(fake_group=="low"),  cluster=~unit_id),
    error = function(e) NULL
  )

  if (!is.null(r_hi) && !is.null(r_lo)) {
    fake_diffs[i] <- coef(r_hi)["post::1:treated"] -
                     coef(r_lo)["post::1:treated"]
  } else {
    fake_diffs[i] <- NA_real_
  }
}

fake_diffs <- na.omit(fake_diffs)
pval_perm  <- mean(abs(fake_diffs) >= abs(diff_obs))
cat(sprintf("组间系数差异 Fisher 置换检验 p 值: %.4f\n", pval_perm))
```

---

### 3.5 交互项法

逻辑：在主回归中加入处理变量 X 与调节变量 M 的交互项（X×M），系数 $\beta_3$ 代表 M 对 X→Y 效应的调节强度。再绘制边际效应图，展示不同 M 值下 X 的效应大小和置信区间。

**适用场景：** M 是调节变量（M 不被 X 影响，M 影响 X 对 Y 的效应强度）。

```r
# ============================================================
# 交互项法 + 边际效应图 — R（fixest + marginaleffects + ggplot2）
# ============================================================
library(fixest)
library(marginaleffects)  # install.packages("marginaleffects")
library(ggplot2)

# 对调节变量做中心化（便于解读主效应）
df <- df %>%
  mutate(M_c = as.numeric(scale(mediator, center = TRUE, scale = FALSE)))

# ── 交互项回归 ─────────────────────────────────────────────
# 注意：fixest 中 post 和 treated 已通过 i() 交互，此处做三阶交互
df$did <- df$post * df$treated
res_interact <- feols(
  outcome ~ did * M_c + control1 + control2 | unit_id + time,
  data = df, cluster = ~unit_id
)
etable(res_interact,
       keep  = "did|M_c",
       title = "调节效应：DID × 调节变量 M 的交互项")

cat("交互项系数:", coef(res_interact)["did:M_c"], "\n")
cat("解读：交互项系数 > 0 表示 M 越高，处理效应越强（正向调节）\n")

# ── 边际效应图：DID 效应随 M 变化 ─────────────────────────
M_range <- seq(
  quantile(df$M_c, 0.05, na.rm = TRUE),
  quantile(df$M_c, 0.95, na.rm = TRUE),
  length.out = 50
)

me_df <- marginaleffects::slopes(
  res_interact,
  variables = "did",
  newdata   = datagrid(M_c = M_range, post = 1, treated = 1)
)

p_me <- ggplot(me_df, aes(x = M_c, y = estimate)) +
  geom_line(color = "steelblue", linewidth = 1.2) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high),
              alpha = 0.2, fill = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  labs(
    x        = "调节变量 M（中心化后，原始单位）",
    y        = "处理效应（边际效应 dy/dx at did=1）",
    title    = "交互项法：处理效应随调节变量 M 变化的边际效应图",
    subtitle = "阴影区域 = 95% CI；水平虚线 = 零效应"
  ) +
  theme_minimal(base_size = 12)

ggsave("output/mechanism_marginal_effect.png", p_me,
       width = 8, height = 5, dpi = 150)
print(p_me)
```

---

### 3.6 因果中介效应

适用于 M 本身存在内生性时，通过 **顺序无混淆假设（Sequential Ignorability）** 或借助工具变量，正式识别直接效应（Direct Effect, DE）和间接效应（Indirect Effect via M, IE）。

**警告：** Sequential Ignorability 是强假设（类似于 M 的随机化），需要在论文中明确声明并讨论其合理性。

```r
# ============================================================
# 因果中介效应 — R（mediation 包）
# install.packages("mediation")
# ============================================================
library(mediation)
library(lme4)  # 或直接用 lm/plm

# ── 第一步：中介方程（M ~ X + 控制变量）──────────────────────
# 注意：mediation 包与 fixest 不直接兼容，改用 lm 加控制固定效应
# 如需面板 FE，可 demean 后再用 lm

# 对面板数据去均值（within-transformation）
df_dm <- df %>%
  group_by(unit_id) %>%
  mutate(across(c(outcome, mediator, did, control1, control2),
                ~ . - mean(., na.rm=TRUE) + mean(df$outcome, na.rm=TRUE))) %>%
  ungroup()

# 中介方程
med_fit <- lm(
  mediator ~ did + control1 + control2,
  data = df_dm
)

# 结果方程（含 M）
out_fit <- lm(
  outcome ~ did + mediator + control1 + control2,
  data = df_dm
)

# ── 因果中介分析 ──────────────────────────────────────────────
med_out <- mediate(
  model.m  = med_fit,           # 中介方程
  model.y  = out_fit,           # 结果方程
  treat    = "did",             # 处理变量名
  mediator = "mediator",        # 中介变量名
  sims     = 1000,              # Bootstrap 次数
  boot     = TRUE,              # 非参数 Bootstrap
  boot.ci.type = "perc"         # 百分位数 CI
)

summary(med_out)
# 输出：
#   ACME (Average Causal Mediation Effect) = 间接效应
#   ADE  (Average Direct Effect)           = 直接效应
#   Total Effect                           = 总效应
#   Prop. Mediated                         = 间接效应占总效应比例

plot(med_out,
     main   = "因果中介效应分解",
     labels = c("ACME\n(间接效应)", "ADE\n(直接效应)", "Total\n(总效应)"))
```

```python
# ============================================================
# 因果中介效应手动实现框架 — Python
# 适合：线性模型 + OLS，无需额外包
# ============================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.utils import resample

def causal_mediation_bootstrap(df, outcome, mediator, treatment,
                                controls=[], n_boot=1000, seed=42):
    """
    线性因果中介效应（Bootstrap 置信区间）
    假设：Sequential Ignorability（中介的顺序无混淆）
    
    Returns: dict 含 ACME, ADE, Total, Prop_Mediated 及其 95% CI
    """
    np.random.seed(seed)
    X_controls = [treatment, mediator] + controls
    X_med      = [treatment] + controls

    def fit_effects(data):
        # 中介方程：M ~ X + controls
        X_m   = sm.add_constant(data[[treatment] + controls])
        mod_m = sm.OLS(data[mediator], X_m).fit()
        alpha = mod_m.params[treatment]   # X → M 系数

        # 结果方程：Y ~ X + M + controls
        X_y   = sm.add_constant(data[[treatment, mediator] + controls])
        mod_y = sm.OLS(data[outcome], X_y).fit()
        beta  = mod_y.params[mediator]   # M → Y 系数（控制 X）
        gamma = mod_y.params[treatment]  # X → Y 直接效应

        acme  = alpha * beta             # 间接效应 (ACME)
        ade   = gamma                    # 直接效应 (ADE)
        total = acme + ade               # 总效应
        prop  = acme / total if total != 0 else np.nan
        return acme, ade, total, prop

    # 点估计
    acme_obs, ade_obs, total_obs, prop_obs = fit_effects(df)

    # Bootstrap CI
    boot_results = []
    for _ in range(n_boot):
        boot_df = resample(df, replace=True, random_state=None)
        boot_results.append(fit_effects(boot_df))

    boot_arr  = np.array(boot_results)
    ci_lo     = np.percentile(boot_arr, 2.5, axis=0)
    ci_hi     = np.percentile(boot_arr, 97.5, axis=0)

    result = {
        "ACME"          : (acme_obs,  ci_lo[0], ci_hi[0]),
        "ADE"           : (ade_obs,   ci_lo[1], ci_hi[1]),
        "Total_Effect"  : (total_obs, ci_lo[2], ci_hi[2]),
        "Prop_Mediated" : (prop_obs,  ci_lo[3], ci_hi[3]),
    }

    print("=" * 55)
    print(f"{'指标':<20} {'点估计':>10} {'95% CI':>20}")
    print("=" * 55)
    for k, (est, lo, hi) in result.items():
        print(f"{k:<20} {est:>10.4f} [{lo:>8.4f}, {hi:>8.4f}]")
    print("=" * 55)
    print(f"解读：ACME = 间接效应（通过 {mediator} 传导的部分）")
    return result

# 使用
med_result = causal_mediation_bootstrap(
    df       = df_matched,
    outcome  = 'outcome',
    mediator = 'mediator',
    treatment= 'did',
    controls = ['control1', 'control2'],
    n_boot   = 1000
)
```

---

### 3.7 残差法（Bleemer-Mehta 风格）

**方法来源：** Bleemer & Mehta (2022, AER) 在研究大学录取对薪资影响的机制时，通过"用 M 预测 Y 的残差再做主回归"来量化间接效应的比例。

**逻辑：**
1. 用中介变量 M 预测结果变量 Y（控制固定效应），取残差 $\hat{\varepsilon}_{Y|M}$
2. 用残差对处理变量 X 做主回归
3. 残差回归系数捕捉"X 对 Y 中无法被 M 解释的部分"的效应 → 即直接效应
4. 间接效应 = 总效应 - 残差法估计的直接效应

**标注：** 此方法属于经济学顶刊前沿做法，管理学期刊（如 SMJ、AMJ）审稿人接受度有限，使用前需评估期刊偏好。

```r
# ============================================================
# 残差法（Bleemer-Mehta 风格）— R
# ============================================================
library(fixest)
library(dplyr)

set.seed(42)
B <- 500  # Bootstrap 次数

# ── Step 1：总效应（主回归）──────────────────────────────────
res_total <- feols(
  outcome ~ i(post, treated, ref=0) + control1 + control2 | unit_id + time,
  data = df, cluster = ~unit_id
)
beta_total <- coef(res_total)["post::1:treated"]
cat(sprintf("总效应: %.4f\n", beta_total))

# ── Step 2：用 M 预测 Y，取残差 ─────────────────────────────
# 注意：只用从未处理单位（or 处理前期）估计 M→Y 的参数，
#       避免 X 通过 M 影响残差（Bleemer-Mehta 建议）
df_pre <- df %>% filter(post == 0 | treated == 0)

res_m_y <- feols(
  outcome ~ mediator + control1 + control2 | unit_id + time,
  data = df_pre
)

# 将系数应用到全样本，计算残差
df <- df %>%
  mutate(
    y_hat_by_m = predict(res_m_y, newdata = df),
    resid_y_m  = outcome - y_hat_by_m
  )

# ── Step 3：对残差做主回归（估计直接效应）───────────────────
res_direct <- feols(
  resid_y_m ~ i(post, treated, ref=0) + control1 + control2 | unit_id + time,
  data = df, cluster = ~unit_id
)
beta_direct <- coef(res_direct)["post::1:treated"]

# ── 间接效应（M 渠道贡献）────────────────────────────────────
beta_indirect  <- beta_total - beta_direct
prop_mediated  <- beta_indirect / beta_total

cat(sprintf("总效应:   %.4f\n", beta_total))
cat(sprintf("直接效应: %.4f  (%.1f%%)\n", beta_direct,  beta_direct/beta_total*100))
cat(sprintf("间接效应: %.4f  (%.1f%%)  ← M 渠道贡献\n",
            beta_indirect, prop_mediated*100))

# ── Bootstrap 标准误（间接效应 SE）──────────────────────────
boot_indirect <- numeric(B)

for (b in seq_len(B)) {
  # 对单位 ID 做 cluster bootstrap
  boot_units <- sample(unique(df$unit_id), replace = TRUE)
  df_b <- do.call(rbind, lapply(boot_units, function(u) df[df$unit_id == u, ]))

  r_total  <- tryCatch(
    feols(outcome ~ i(post,treated,ref=0)+control1+control2|unit_id+time,
          data=df_b), error=function(e) NULL)
  df_pre_b <- df_b %>% filter(post==0|treated==0)
  r_m_y    <- tryCatch(
    feols(outcome ~ mediator+control1+control2|unit_id+time,
          data=df_pre_b), error=function(e) NULL)

  if (!is.null(r_total) && !is.null(r_m_y)) {
    df_b <- df_b %>%
      mutate(resid_b = outcome - predict(r_m_y, newdata=df_b))
    r_direct <- tryCatch(
      feols(resid_b ~ i(post,treated,ref=0)|unit_id+time, data=df_b),
      error=function(e) NULL)
    if (!is.null(r_direct)) {
      bt <- coef(r_total)["post::1:treated"]
      bd <- coef(r_direct)["post::1:treated"]
      boot_indirect[b] <- bt - bd
    } else boot_indirect[b] <- NA_real_
  } else boot_indirect[b] <- NA_real_
}

boot_indirect <- na.omit(boot_indirect)
se_indirect   <- sd(boot_indirect)
ci_lo <- quantile(boot_indirect, 0.025)
ci_hi <- quantile(boot_indirect, 0.975)
cat(sprintf("间接效应 SE（Bootstrap）: %.4f\n", se_indirect))
cat(sprintf("间接效应 95%% CI: [%.4f, %.4f]\n", ci_lo, ci_hi))
cat("\n[标注] 残差法属经济学顶刊前沿方法（参考 Bleemer & Mehta 2022, AER），\n",
    "      管理学期刊接受度有限，请评估目标期刊偏好。\n")
```

---

## 4. 方法对照表

| 方法 | 适用场景 | 优势 | 局限 | 适用期刊 |
|------|----------|------|------|---------|
| **两步法** | 只需证明 X→M 存在 | 无坏控制变量问题；解释直觉 | 无法量化间接效应占比 | 顶刊首选（AER/QJE/JFE）|
| **完整三步法** | 兼容传统审稿人 | 展示中介系数变化 | 第三步 M 是坏控制变量 | 管理学期刊（AMJ/SMJ）|
| **三步法变体** | X 与 M 高度相关 | 回避多重共线性 | 第三步解释力弱化 | 管理学 / 经济学均可 |
| **分组回归** | M 可二值化（高/低） | 无函数形式假设；直觉强 | 分组切割点主观 | 各类实证期刊 |
| **交互项法** | M 是调节变量（非受 X 影响）| 正式估计调节效应 | 需要 M 外生 | 顶刊 + 管理学 |
| **因果中介** | M 内生，需 Sequential Ignorability | 正式分解 IE/DE | 假设强，难以验证 | 方法论期刊 / 顶刊附录 |
| **残差法** | 量化间接效应比例 | 不引入坏控制变量 | 需额外假设；审稿人不熟悉 | 经济学顶刊前沿 |

---

## 5. 坏控制变量警告

**核心问题：** 如果中介变量 M 受到处理变量 X 的影响（X→M），那么在 Y~X+M 的回归中加入 M 会"阻断"X 通过 M 影响 Y 的路径，导致 X 系数低估总效应，且 M 系数不能解释为纯粹的 M→Y 因果效应。

### 用户指定 M 时的确认提示

```python
# ============================================================
# 坏控制变量确认流程 — Python
# ============================================================

def check_bad_control(M_name, X_name, Y_name):
    """
    在执行机制回归之前，提示用户确认 M 的性质。
    """
    print(f"{'='*60}")
    print(f"机制变量安全检查：{M_name}")
    print(f"{'='*60}")
    print(f"\n请回答以下问题以确认 {M_name} 的角色：\n")

    q1 = input(f"1. {M_name} 是否受 {X_name}（处理变量）的影响？\n"
               "   (y = 是，n = 否，u = 不确定): ").strip().lower()

    q2 = input(f"\n2. {M_name} 是否在 {X_name} 处理发生之前就已确定？\n"
               "   （时序逻辑：M 应在处理发生后才受影响）\n"
               "   (y = M 在处理前确定，n = M 在处理后变化): ").strip().lower()

    q3 = input(f"\n3. {M_name} 是否同时被 {Y_name}（结果变量）影响（即 Y→M）？\n"
               "   (y = 是，n = 否): ").strip().lower()

    print(f"\n{'='*60}")

    if q3 == 'y':
        print(f"⛔ 警告：{M_name} 可能是对撞变量（同时受 X 和 Y 影响）！")
        print("加入回归将引入虚假关联。建议重新检查 DAG，拒绝执行此机制回归。")
        return "collider"

    if q2 == 'y':
        print(f"⚠️  警告：{M_name} 在处理发生前已确定，可能是前定变量，而非中介变量！")
        print("前定变量应作为控制变量（调节变量），而非中介。")
        return "predetermined"

    if q1 == 'n':
        print(f"ℹ️  信息：{M_name} 不受处理影响 → 可能是调节变量，使用交互项法。")
        return "moderator"

    if q1 == 'y':
        print(f"✅ {M_name} 受处理影响、在处理后变化、不被 Y 反向影响 → 符合中介变量条件。")
        print("推荐使用两步法或因果中介效应方法。")
        return "mediator"

    print("不确定，建议回到 DAG 重新梳理因果关系。")
    return "unclear"

# 使用
role = check_bad_control("融资约束指数", "政策处理", "企业投资")
```

### 时序逻辑检查

```r
# ============================================================
# 时序逻辑检查 — R
# 确保 M 的变化发生在 X 处理之后，而非之前
# ============================================================
library(dplyr)

# 检验：处理前（pre-treatment）M 的变化趋势
# 如果 M 在处理前就已经在变化，说明 M 不是由 X 导致的（反向因果或共同趋势）

pre_trend_m <- df %>%
  filter(post == 0) %>%                    # 仅处理前期
  group_by(time, treated) %>%
  summarise(mean_m = mean(mediator, na.rm = TRUE), .groups = "drop")

library(ggplot2)
ggplot(pre_trend_m, aes(x = time, y = mean_m, color = factor(treated))) +
  geom_line() + geom_point() +
  labs(title = "处理前 M 的时间趋势（时序逻辑检查）",
       subtitle = "若处理组与对照组处理前趋势已分叉，M 可能不是 X 的结果",
       color = "组别") +
  theme_minimal()
```

---

## 6. 机制变量的平行趋势

如果主回归使用了 DID 方法，机制变量 M 也应当满足平行趋势假设——即在处理前，处理组和对照组的 M 趋势平行。否则，X→M 的证据本身也存在内生性问题。

### 机制变量事件研究图

```r
# ============================================================
# 机制变量平行趋势检验 — R（fixest 事件研究图）
# 与主回归的事件研究图并排报告
# ============================================================
library(fixest)
library(ggplot2)

# ── 机制变量 M 的事件研究 ────────────────────────────────────
res_m_es <- feols(
  mediator ~ i(event_time, treated, ref = -1) | unit_id + time,
  data    = df %>% filter(event_time %in% -4:4),
  cluster = ~unit_id
)

# ── 可视化（与主回归 Y 的事件研究并排）──────────────────────
# 提取系数和 CI
extract_es <- function(res, label) {
  cf   <- coef(res)
  ci   <- confint(res)
  idx  <- grep("event_time", names(cf))
  times <- as.numeric(gsub(".*event_time::(-?[0-9]+).*", "\\1", names(cf)[idx]))
  data.frame(
    event_time = times,
    coef       = cf[idx],
    ci_lo      = ci[idx, 1],
    ci_hi      = ci[idx, 2],
    variable   = label
  )
}

df_es_y <- extract_es(res_es,   "Y（结果变量）")      # res_es 来自主回归
df_es_m <- extract_es(res_m_es, "M（机制变量）")

df_es_all <- rbind(df_es_y, df_es_m)

ggplot(df_es_all, aes(x = event_time, y = coef,
                       color = variable, fill = variable)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_vline(xintercept = -0.5, linetype = "dashed", color = "red", alpha = 0.7) +
  geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), alpha = 0.15, color = NA) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  facet_wrap(~variable, scales = "free_y") +
  scale_color_manual(values = c("steelblue", "darkorange")) +
  scale_fill_manual(values  = c("steelblue", "darkorange")) +
  labs(
    x        = "相对处理期（ref = -1）",
    y        = "系数（ATT）",
    title    = "主回归（Y）与机制变量（M）的事件研究图对比",
    subtitle = "若 M 在处理前平行趋势成立，支持 X → M 的因果解释",
    caption  = "固定效应：个体+时间；聚类标准误：个体层面"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

ggsave("output/mechanism_event_study.png",
       width = 10, height = 5, dpi = 150)

# ── 联合显著性检验（Pre-period M 平行趋势）──────────────────
pre_coefs_m  <- grep("event_time::-[2-9]", names(coef(res_m_es)), value=TRUE)
waldtest_m   <- wald(res_m_es, keep = pre_coefs_m)
cat(sprintf("M 的 Pre-period 联合检验：F = %.3f，p = %.4f\n",
            waldtest_m$stat, waldtest_m$p))
cat("通过标准（p > 0.10）：", waldtest_m$p > 0.10, "\n")
```

### 机制分析输出清单

```
output/
  mechanism_two_step.csv          # 两步法回归结果
  mechanism_three_step.csv        # 三步法/变体回归结果
  mechanism_subgroup.csv          # 分组回归结果（高/低 M 组）
  mechanism_interaction.png       # 交互项边际效应图
  mechanism_mediation.csv         # 因果中介效应分解（ACME/ADE）
  mechanism_residual.csv          # 残差法间接效应估计
  mechanism_event_study.png       # 机制变量平行趋势检验图
  mechanism_permtest_pval.txt     # 组间系数差异 Fisher 置换检验 p 值
```
