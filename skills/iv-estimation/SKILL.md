# IV 估计 — 工具变量法 (Instrumental Variables)

## 概述

工具变量法通过寻找只影响内生解释变量、但不直接影响结果变量的外生变量（工具变量），解决内生性问题（遗漏变量、测量误差、反向因果）。标准方法是两阶段最小二乘（2SLS）。

**适用场景：**
- 核心解释变量 X 与误差项相关（内生性）
- 能找到满足相关性（relevance）和排他性（exclusion restriction）的工具变量
- 常见工具变量来源：地理/气候特征、政策变化、历史数据、随机抽签

---

## 前置条件

### 数据结构要求

```
截面数据：一行一观测
面板数据：长格式，含个体 ID 和时间列

必须包含：
  - 结果变量 Y（被解释变量）
  - 内生变量 X（endogenous variable）
  - 工具变量 Z（可以多个）
  - 外生控制变量 W（exogenous controls，可选）
  - 聚类变量（FE/cluster，可选）

工具变量数量必须 ≥ 内生变量数量（恰好识别 or 过度识别）
```

### 识别条件核查清单（必须在论文中论证）

1. **相关性（Relevance）**：Cov(Z, X) ≠ 0 → 用第一阶段 F 检验验证
2. **排他性（Exclusion Restriction）**：Cov(Z, ε) = 0 → 逻辑论证，无法统计检验（恰好识别时）
3. **单调性（Monotonicity）**：对 LATE 解释（如 Imbens-Angrist 框架）的额外假设

---

## 分析步骤

### Step 1：工具变量相关性预检

在做 2SLS 前，先检查工具变量与内生变量的相关性（散点图 + 相关系数）。

```python
# Python: 相关性检查
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(instruments), figsize=(5*len(instruments), 4))
for ax, z in zip(axes, instruments):
    ax.scatter(df[z], df['endogenous_x'], alpha=0.3)
    corr = df[[z, 'endogenous_x']].corr().iloc[0,1]
    ax.set_title(f'{z} vs X  (r={corr:.3f})')
    ax.set_xlabel(z)
    ax.set_ylabel('Endogenous X')
plt.tight_layout()
plt.savefig('output/iv_relevance_scatter.png', dpi=150)
```

```r
# R: 相关性矩阵
cor_mat <- df[, c('endogenous_x', instruments)] |> cor(use = "complete.obs")
print(round(cor_mat, 3))
```

---

### Step 2：第一阶段回归（First Stage）

$$X_i = \pi_0 + \pi_1 Z_i + \gamma W_i + \nu_i$$

**关键指标：第一阶段 F 统计量**

```python
# Python: linearmodels IV2SLS
from linearmodels.iv import IV2SLS
import statsmodels.api as sm

formula = f"outcome ~ 1 + {' + '.join(controls)} + [{endogenous} ~ {' + '.join(instruments)}]"
res_iv = IV2SLS.from_formula(formula, data=df).fit(cov_type='robust')

# 手动运行第一阶段
import statsmodels.formula.api as smf
first_stage_formula = f"{endogenous} ~ {' + '.join(instruments)} + {' + '.join(controls)}"
res_fs = smf.ols(first_stage_formula, data=df).fit()

# 第一阶段 F 统计量（限制工具变量系数）
from statsmodels.stats.anova import anova_lm
res_fs_noz = smf.ols(f"{endogenous} ~ {' + '.join(controls)}", data=df).fit()
f_stat = anova_lm(res_fs_noz, res_fs)
print(f"First Stage F-stat on excluded instruments: {f_stat['F'].iloc[1]:.2f}")
print(f"Partial R²: {res_fs.rsquared - res_fs_noz.rsquared:.4f}")
```

```r
# R: fixest feols IV（推荐）
library(fixest)

# 格式：outcome ~ controls | FE | endogenous ~ instruments
res_iv <- feols(
  outcome ~ control1 + control2 | unit_fe + time_fe | endogenous_x ~ z1 + z2,
  data    = df,
  cluster = ~cluster_var
)

# 第一阶段结果
fitstat(res_iv, type = c("ivf", "ivwald", "kpr"))  # Kleibergen-Paap rk
summary(res_iv, stage = 1)  # 查看第一阶段
```

```stata
* Stata: ivreghdfe（推荐，含高维FE）
ssc install ivreghdfe
ssc install ivreg2

ivreghdfe outcome control1 control2 (endogenous_x = z1 z2), ///
    absorb(unit_id time) cluster(cluster_var) first

* 或 ivreg2（输出更详细）
ivreg2 outcome control1 control2 (endogenous_x = z1 z2), ///
    cluster(cluster_var) first savefirst
```

---

### Step 3：弱工具变量诊断

#### F 统计量标准

| 标准 | 阈值 | 来源 |
|------|------|------|
| 经典规则 | F > 10 | Staiger & Stock (1997) |
| 5% 偏误上界 | F > 10 | Stock & Yogo (2005) 临界值 |
| 严格标准（单工具） | F > 104.7 | Lee, McCrary, Moreira & Porter (2022) |
| 多工具：Kleibergen-Paap | KP rk Wald F > 查临界值表 | Kleibergen & Paap (2006) |

**实践建议：**
- 单个工具变量：报告 F > 104.7 才能用 t-test 推断；否则用 Anderson-Rubin 置信区间
- 多个工具变量：报告 Kleibergen-Paap rk Wald F 统计量

```r
# R: 报告各类 F 统计量
fitstat(res_iv, type = c("ivf",        # 经典 F（恰好识别时等于 KP）
                          "kpr",        # Kleibergen-Paap rk
                          "ef",         # Effective F（Montiel-Olea & Pflueger 2013）
                          "cd"))        # Cragg-Donald F（IID误差假设下）
```

```stata
* Stata: ivreg2 输出 KP 统计量
ivreg2 y controls (x = z1 z2), robust first
* 重点看: Kleibergen-Paap rk Wald F statistic
* 与 Stock-Yogo critical values 比较（输出中会显示）
```

---

### Step 4：弱工具变量稳健推断（Anderson-Rubin）

当工具变量可能较弱时，Anderson-Rubin 置信区间在弱工具下仍有正确的覆盖率。

```r
# R: AR 置信区间
library(ivmodel)

# 恰好识别（单工具）
iv_mod <- ivmodel(
  Y = df$outcome,
  D = df$endogenous_x,
  Z = df$instrument,
  X = as.matrix(df[, controls])
)
summary(iv_mod)  # 包含 AR, CLR, LIML 等稳健推断
AR.test(iv_mod)  # Anderson-Rubin test
```

```stata
* Stata: 弱工具稳健推断（CONDITIONAL置信区间）
ssc install weakiv

ivregress 2sls y controls (x = z), robust
weakiv   * 计算 CLR（Conditional Likelihood Ratio）置信区间
```

---

### Step 5：过度识别检验（Overidentification Test）

当工具变量数量 > 内生变量数量时，可以检验工具变量的外生性（Hansen J test）。

**注意：** 恰好识别（Z 数量 = X 数量）时无法进行过度识别检验。

```python
# Python: linearmodels 输出 J 统计量
res_iv = IV2SLS.from_formula(formula, data=df).fit(cov_type='unadjusted')
print(res_iv.wooldridge_overid)      # Wooldridge's overidentification test
print(res_iv.anderson_rubin)         # AR test
```

```r
# R: fixest 过度识别（需安装 lmtest）
library(lmtest)

# 手动 Hansen J test（2SLS 残差对工具变量回归）
res_fs   <- lm(as.formula(paste(endogenous, "~", paste(c(instruments, controls), collapse="+"))), df)
x_hat    <- fitted(res_fs)
res_2sls <- lm(as.formula(paste("outcome ~", endogenous_hat, "+", paste(controls, collapse="+"))), df)
resid_2sls <- residuals(res_2sls)
j_reg    <- lm(as.formula(paste("resid_2sls ~", paste(c(instruments, controls), collapse="+"))), df)
j_stat   <- summary(j_reg)$r.squared * nrow(df)
j_pval   <- pchisq(j_stat, df = length(instruments) - 1, lower.tail = FALSE)
cat(sprintf("Hansen J: chi2(%d) = %.3f, p = %.4f\n", length(instruments)-1, j_stat, j_pval))
```

```stata
* Stata: ivreg2 自动报告 Hansen J
ivreg2 y controls (x = z1 z2 z3), robust
* 输出中包含: Hansen J statistic (overidentification test of all instruments)
* H0: 所有工具变量均外生 → p > 0.1 才通过
```

**解释：** Hansen J 显著（p < 0.1）说明至少一个工具变量可能不满足排他性约束。但注意：Hansen J 的功效（power）有限，通过检验≠工具变量一定有效。

---

### Step 6：内生性检验（Hausman-Wu Durbin-Wu Test）

检验是否真的存在内生性，如不存在则 OLS 更有效率。

```python
# Python: Hausman 检验（Wu-Hausman）
res_ols = IV2SLS.from_formula(f"outcome ~ 1 + {endogenous} + {' + '.join(controls)}", df).fit()
res_iv  = IV2SLS.from_formula(formula, df).fit()
print(res_iv.wu_hausman())   # H0: OLS 是一致的（无内生性）
```

```r
# R: Durbin-Wu-Hausman
library(ivreg)
res_ivreg <- ivreg(
  outcome ~ endogenous_x + control1 + control2 |
            instrument1 + instrument2 + control1 + control2,
  data = df
)
summary(res_ivreg, diagnostics = TRUE)
# Diagnostics 包含: Wu-Hausman (endogeneity), Sargan (overid), Weak instruments
```

```stata
* Stata: C-statistic 或 Durbin-Wu-Hausman
ivreg2 y controls (x = z1 z2), robust endog(x)
* 输出: C statistic (endogeneity test)
```

---

## 排他性约束论证框架

排他性约束（Z 只通过 X 影响 Y，不直接影响 Y）是 IV 的核心假设，**无法统计检验**，必须依赖逻辑和领域知识论证。

### 标准论证结构

```
1. 机制论证：
   "Z 影响 Y 的唯一通道是通过 X，理由是……（经济学/管理学机制）"

2. 反驳潜在违约渠道：
   "有人可能认为 Z 通过路径 A 直接影响 Y，但……（数据/文献/逻辑反驳）"

3. 历史/地理工具变量特别说明：
   "该工具变量建立在历史/地理特征上，在当前时期不直接影响结果，因为……"

4. 间接支持证据：
   - 对不应受影响的子样本/结果变量做安慰剂检验
   - 控制可能的直接渠道后系数稳定
```

### 安慰剂检验：排他性约束

```r
# 用 Z 解释不相关的结果变量（若 Z 显著则排他性存疑）
res_placebo_z <- feols(
  unrelated_outcome ~ instrument + control1 + control2 | unit_fe,
  data = df, cluster = ~cluster_var
)
# 期望：coefficient on instrument ≈ 0, p > 0.1
```

---

## 必做检验清单

| 检验 | 指标 | 标准 |
|------|------|------|
| 第一阶段 F（单工具） | F-stat | > 104.7（Lee et al.）/ > 10（宽松） |
| 第一阶段 F（多工具） | Kleibergen-Paap rk F | > Stock-Yogo 临界值 |
| 过度识别（多工具） | Hansen J p-value | p > 0.10 |
| 内生性检验 | Durbin-Wu-Hausman p | p < 0.10（确认内生性） |
| 弱工具稳健推断 | AR / CLR 置信区间 | 与 2SLS 置信区间比较 |
| 排他性约束安慰剂 | 用 Z 解释无关结果 | 系数不显著 |
| 2SLS vs OLS 比较 | 系数方向和大小 | IV 系数通常更大（衰减偏误被纠正） |

---

## 常见错误提醒

> **错误 1：F > 10 就万事大吉**
> Staiger-Stock 的 F > 10 是 2SLS 偏误 < 10% OLS 偏误的条件，但不保证 t-test 推断有效。Lee et al. (2022) 指出，对单工具变量，需要 F > 104.7 才能用标准 t-test，否则置信区间覆盖率不足。**建议始终报告 Anderson-Rubin 或 CLR 置信区间。**

> **错误 2：过度识别通过 = 工具变量均有效**
> Hansen J 检验的原假设是"所有工具变量均外生"，但其功效有限。特别是当多个工具变量以相同方式违反排他性时，检验无法发现问题。

> **错误 3：用 OLS 残差替代 2SLS 第二阶段**
> 手动做两阶段时，第二阶段必须用 X̂（第一阶段拟合值）替换 X，且**标准误必须用 2SLS 公式计算**，不能直接用第二阶段 OLS 标准误（会严重低估）。始终用包（IV2SLS/feols/ivreg2）而非手动操作。

> **错误 4：Cragg-Donald F vs Kleibergen-Paap F**
> Cragg-Donald F 假设 IID 误差，当使用聚类/异方差稳健标准误时，应报告 Kleibergen-Paap rk Wald F 统计量。

> **错误 5：LATE vs ATE 混淆**
> 2SLS 估计的是 LATE（Local Average Treatment Effect），即 Compliers 的平均处理效应，不是总体 ATE。LATE 的外部效度有限，需明确说明。

> **错误 6：工具变量选择的"撒网"策略**
> 不要机械地选多个工具变量来提高 F 统计量。每个工具变量都需要独立的经济逻辑支撑，并通过过度识别检验。

---

## 输出规范

### 标准 IV 结果表（含第一阶段）

```r
# R: fixest 输出 IV 表（含第一阶段）
library(fixest)

etable(
  res_ols,   # OLS baseline
  res_iv,    # 2SLS
  stage = 1:2,   # 同时显示第一阶段和第二阶段
  fitstat = ~ ivf + kpr + n,
  title   = "IV Estimation Results",
  headers = c("OLS", "2SLS Stage 1", "2SLS Stage 2")
)
```

**表格必须包含：**
1. OLS 对比列（显示内生性偏误方向）
2. 第一阶段系数和 F 统计量
3. 第二阶段 IV 估计量
4. 过度识别检验（如适用）
5. 固定效应、聚类方式、观测数

### 文件命名

```
output/
  iv_first_stage.csv          # 第一阶段结果
  iv_main_results.csv         # 主回归（OLS vs 2SLS）
  iv_diagnostics.txt          # 诊断统计量（F, J, AR）
  iv_placebo_exclusion.csv    # 排他性安慰剂检验
  iv_robustness.csv           # 用不同工具变量/样本的稳健性
```

---

## 方法论深度补充

### A. LATE 与 MTE 框架

#### A1. LATE 的精确定义（Imbens & Angrist 1994）

$$\text{LATE} = E[Y_i(1) - Y_i(0) \mid \text{Complier}]$$

**四类个体（当工具变量 Z 从 0 变为 1）：**

| 类型 | 定义 | D(Z=0) | D(Z=1) | 2SLS 能识别？ |
|------|------|--------|--------|---------------|
| Complier（顺从者） | Z 变化时 D 跟着变化 | 0 | 1 | ✓ LATE 识别对象 |
| Always-taker（总参与者） | 无论 Z 如何都参与 | 1 | 1 | ✗ |
| Never-taker（从不参与者） | 无论 Z 如何都不参与 | 0 | 0 | ✗ |
| Defier（反从者） | Z 变化时 D 反向变化 | 1 | 0 | 单调性假设排除 |

**为什么 Card (1995) 的 IV 估计比 OLS 高 25–60%？**

Card 使用"距大学距离"作为教育年限的工具变量。IV 估计的是 **Complier 的 LATE**——这群人是距离近才去上大学的边际个体。这些边际个体恰好是教育回报率更高的群体（往往来自低收入家庭，教育对他们的信贷约束释放效果更大），所以 LATE > OLS（OLS 还被反向选择偏误拉低）。

**LATE 的外部效度限制：**
- LATE ≠ ATE（总体平均处理效应）
- LATE ≠ ATT（处理组的平均处理效应）
- 不同工具变量识别不同 Complier 子群体，所以不同 IV 给出的估计自然不同——这不是偏误，是异质性的反映
- 政策推广到 Never-taker 群体时，LATE 不能直接作为预测依据

#### A2. MTE 框架（Heckman & Vytlacil 2005）

MTE（Marginal Treatment Effect）是对 LATE 框架的结构性扩展，将异质性处理效应统一在一条曲线上。

$$\text{MTE}(u) = E[Y_i(1) - Y_i(0) \mid U_{D,i} = u]$$

其中 $U_{D,i}$ 是个体 $i$ 参与处理的"抵抗程度"（unobserved resistance），$u \in [0,1]$，$u$ 越大表示越不愿意参与。

**统一框架：ATE、ATT、LATE 均为 MTE 的加权积分**

$$\text{ATE} = \int_0^1 \text{MTE}(u)\, du$$

$$\text{ATT} = \int_0^1 \text{MTE}(u) \cdot \omega_{ATT}(u)\, du$$

$$\text{LATE}(z, z') = \int_{p(z)}^{p(z')} \text{MTE}(u)\, du \Big/ [p(z') - p(z)]$$

其中 $p(z) = P(D=1 \mid Z=z)$ 是倾向得分，不同工具变量改变 $p(z)$ 的取值范围，对应 MTE 曲线的不同积分区间。

**核心含义：**
- 不同 IV 给出不同估计量，是因为它们对 MTE 曲线的不同区段积分
- 如果 MTE(u) 是平坦的（无异质性），所有估计量相同
- MTE 框架可以用于政策外推：用估计的 MTE 曲线预测政策对任意目标人群的效果

```r
# R: MTE 估计（polyreg / locpoly 方法）
# 需要 ivtools 或 heckman 手动实现
library(ivtools)  # 或 AER + locfit

# 第一步：估计倾向得分
ps_model <- glm(D ~ Z + controls, data = df, family = binomial())
df$p_hat  <- predict(ps_model, type = "response")

# 第二步：LIV（Local Instrumental Variable）估计 MTE
# MTE(u) = d/du E[Y | P(Z)=u]
# 用局部多项式对 E[Y|P] 关于 P 求偏导
library(locfit)
mte_fit <- locfit(outcome ~ lp(p_hat, deg = 2), data = df)
# 在 u 网格上求导即为 MTE 曲线
u_grid  <- seq(0.05, 0.95, by = 0.05)
mte_est <- predict(mte_fit, newdata = data.frame(p_hat = u_grid),
                   deriv = 1)  # 一阶导数 = MTE
plot(u_grid, mte_est, type = "l",
     xlab = "u (resistance to treatment)",
     ylab = "MTE(u)",
     main = "Marginal Treatment Effect Curve")
abline(h = 0, lty = 2)
```

#### A3. 设计本位 vs 结构性传统的争论

| 立场 | 代表人物 | 核心主张 | 对 LATE 的态度 |
|------|----------|----------|----------------|
| 设计本位（Design-based） | Angrist, Imbens | LATE 是最诚实的因果声称；利用"自然实验"做最小假设推断 | 充分拥抱 LATE，明确报告可识别的 Complier 群体 |
| 结构性（Structural） | Heckman, Vytlacil | 政策分析需要超越 LATE；用 MTE 重构反事实，外推到 Never-taker | LATE 是有限的，需要结构模型做政策预测 |

**实践建议：** 对经管论文而言，大多数顶刊接受 LATE 框架，但需在论文中明确说明：  
① 谁是 Complier（描述性特征分析）；  
② LATE 的外部效度边界；  
③ 若政策目标是 ATE，需额外论证或使用 MTE 框架。

---

### B. IV 的四大流派

| 流派 | 识别逻辑 | 代表文献 | 典型应用 | 外生性来源 | 优势 | 局限 |
|------|----------|----------|----------|------------|------|------|
| **结构型 IV** | 理论模型推导出排他性约束；供需方程联立 | Wright (1928), Cowles 委员会 | 需求/供给弹性估计，联立方程模型 | 经济理论假设 | 可直接回答结构参数 | 强依赖模型设定；误设即失效 |
| **自然实验 IV** | 外生制度冲击（抽签、政策截断、出生日期）创造随机变异 | Angrist (1990) 征兵抽签; AJR (2001) 殖民地死亡率; Card (1995) 大学距离 | 教育回报、制度对增长、越战老兵收入影响 | 制度随机性 / 地理随机性 | 排他性约束可信；内部效度高 | 外部效度有限；Complier 群体可能特殊 |
| **Shift-Share / Bartik IV** | 历史产业构成 × 全国行业冲击；个体差异来自历史初始结构 | Bartik (1991); Goldsmith-Pinkham et al. (2020); Borusyak et al. (2022) | 劳动力市场冲击、移民影响、贸易冲击（China shock） | 历史份额（shares）或全国冲击（shifts）的外生性 | 可利用大规模行政数据；适合地区级分析 | 三大识别来源争议未定；标准误须修正 |
| **法官 / 审查官设计** | 随机案件分配给偏好不同的决策者；决策者严厉程度作为 IV | Kling (2006); Doyle (2007); Maestas et al. (2013) | 监禁效应（就业/犯罪再犯）、儿童寄养、残障保险批准效应 | 案件随机分配的行政规则 | 排他性论证自然；第一阶段通常很强 | 需要真正随机分配；地理/时间覆盖限制 |

#### B1. 结构型 IV（Wright 1928, Cowles 委员会）

- **核心思路：** 从理论模型（供需方程、行为方程）推导识别条件；供给侧变量不影响需求方程（反之亦然）
- **经典例子：** 用天气/运输成本作为供给冲击识别需求弹性
- **现代遗产：** 动态随机一般均衡（DSGE）中的结构 IV；健康经济学中的价格工具变量
- **弱点：** 一旦模型设定错误，识别崩溃；难以向非结构经济学读者论证

#### B2. 自然实验 IV

**三篇经典文献对比：**

| 文献 | 工具变量 | 内生变量 | 结果变量 | 排他性论证 |
|------|----------|----------|----------|------------|
| Angrist (1990) | 越战征兵抽签号码 | 越战服役（D） | 长期收入 | 抽签号码对收入无直接影响（只通过服役） |
| AJR (2001) | 殖民者死亡率（1700s） | 当前制度质量 | 人均 GDP | 历史死亡率不直接影响当前经济（只通过定居型/掠夺型制度形成） |
| Card (1995) | 距最近四年制大学距离 | 受教育年限 | 工资 | 距离不影响工资（只通过减少上学成本） |

**排他性约束常见挑战与回应：**
```
AJR 批评：殖民者死亡率可能通过疾病环境直接影响今日生产力
→ 回应：控制地理/气候变量；对前殖民时代人口密度做安慰剂检验

Card 批评：离大学近的家庭本身收入更高/文化资本更强
→ 回应：控制家庭背景；对女性（更少受距离影响）做安慰剂检验
```

#### B3. Shift-Share / Bartik IV

**构造方法：**
$$Z_{it} = \sum_k \underbrace{s_{ik,t_0}}_{\text{历史份额（shares）}} \times \underbrace{g_{kt}}_{\text{全国行业增速（shifts）}}$$

- $s_{ik,t_0}$：地区 $i$ 在基期 $t_0$ 的行业 $k$ 就业份额
- $g_{kt}$：行业 $k$ 在 $t$ 期的全国（剔除本地区）就业增速

**三场方法论争论：**

| 论文 | 核心观点 | 外生性来源 | 识别假设 |
|------|----------|------------|----------|
| GPSS (Goldsmith-Pinkham, Sorkin & Swift 2020) | Bartik IV = 历史份额 $s_{ik}$ 加权的一组行业 IV | 历史份额外生（如不相关于未来需求冲击） | 不同行业的份额与误差项不相关 |
| BHJ (Borusyak, Hull & Jaravel 2022) | 真正的外生性来自全国冲击 $g_{kt}$ 的随机性 | 全国行业冲击近似随机分配给行业 | 全国冲击独立于地区特征 |
| AKM (Adão, Kolesár & Morales 2019) | 标准 Bartik 的 SE 低估了空间相关性 | — | 需要修正标准误（AKM SE） |

**诊断建议：**
1. 报告 GPSS 分解：前几个行业的 Rotemberg 权重（哪些行业驱动识别）
2. 检验关键行业的排他性（对关键权重行业分别做安慰剂检验）
3. 使用 AKM 或 BHJ 修正的标准误

#### B4. 法官 / 审查官设计

**核心机制：** 案件被随机分配给不同严厉程度的决策者（法官/审查员）。以决策者的历史平均严厉程度作为 IV。

```r
# R: 法官设计 IV（leave-one-out 严厉程度计算）
library(fixest)

# 第一步：计算 leave-one-out 判决率（避免机械性相关）
df <- df %>%
  group_by(judge_id) %>%
  mutate(
    # 排除当前案件的法官历史判决率
    judge_leniency = (sum(sentenced) - sentenced) / (n() - 1)
  ) %>%
  ungroup()

# 第二步：随机分配检验（法官严厉程度应与前定特征无关）
randomization_check <- feols(
  judge_leniency ~ defendant_age + defendant_prior + defendant_race | court_id + year,
  data = df
)
etable(randomization_check)  # 期望: 所有系数不显著

# 第三步：2SLS（法官严厉程度作为 IV）
res_judge_iv <- feols(
  outcome ~ controls | court_id + year | sentenced ~ judge_leniency,
  data = df, cluster = ~judge_id
)
summary(res_judge_iv)
```

---

### C. Plausibly Exogenous（Conley, Hansen & Rossi 2012）

当排他性约束可能存在小幅违反时（即工具变量可能对结果有直接但较小的影响），使用敏感性分析量化结论的稳健程度。

**模型设定：**
$$Y_i = \beta X_i + \delta Z_i + \varepsilon_i$$

标准 IV 假设 $\delta = 0$（排他性）。Conley 等允许 $\delta \neq 0$，但假设 $\delta$ 在 $[-\delta_0, \delta_0]$ 的范围内，并计算 $\beta$ 的 bounds（界限）。

**三种方法：**

| 方法 | 描述 | 适用场景 |
|------|------|----------|
| Union of Confidence Intervals (UCI) | 对每个可能的 $\delta$ 计算 CI，取并集 | $\delta$ 支撑已知（如领域知识确定上界） |
| Local to Zero (LTZ) | $\delta \sim N(0, \sigma^2_\delta)$ 的先验 | $\delta$ 连续不确定但集中在零附近 |
| Prior Interval | $\delta \in [-\delta_0, \delta_0]$ 均匀分布 | 对违反程度有合理上界的猜测 |

```r
# R: Plausibly Exogenous 敏感性分析
# 方法1：手动 UCI（union of confidence intervals）
library(AER)  # ivreg
library(tidyverse)

# 扫描 delta 的可能值（直接效应的范围假设）
# 假设 Z 对 Y 的直接效应 delta 在 [-0.1, 0.1] 之间
delta_grid <- seq(-0.1, 0.1, by = 0.01)

bounds_df <- map_dfr(delta_grid, function(d) {
  # 将直接效应从结果中"减去"，再做 IV
  df_adj <- df %>% mutate(y_adj = outcome - d * instrument)
  
  res <- ivreg(
    y_adj ~ endogenous_x + control1 + control2 |
             instrument + control1 + control2,
    data = df_adj
  )
  
  ci <- confint(res)["endogenous_x", ]
  tibble(
    delta    = d,
    beta_hat = coef(res)["endogenous_x"],
    ci_low   = ci[1],
    ci_hi    = ci[2]
  )
})

# 画 bounds 图
ggplot(bounds_df, aes(x = delta)) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_hi), alpha = 0.2, fill = "steelblue") +
  geom_line(aes(y = beta_hat), color = "steelblue") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title    = "Plausibly Exogenous: Sensitivity to Exclusion Restriction",
    subtitle = "Assumption: direct effect δ ∈ [-0.1, 0.1]",
    x        = "Assumed direct effect δ of Z on Y",
    y        = "IV Estimate β"
  ) +
  theme_minimal()

# 方法2: sensemakr（更系统，基于 Cinelli-Hazlett 框架）
# install.packages("sensemakr")
library(sensemakr)

# 先跑 OLS（作为基础），然后做对 IV 排他性违反的等价敏感性分析
base_ols <- lm(outcome ~ endogenous_x + instrument + control1 + control2, data = df)
sens <- sensemakr(
  model         = base_ols,
  treatment     = "instrument",  # 关注 Z 的直接效应
  benchmark_covariates = "control1",  # 用已知控制变量做基准
  kd            = 1:3            # 混淆因子强度倍数
)
plot(sens)     # Contour plot: 多强的 confounder 才能推翻结论
summary(sens)  # 最小影响力（robustness value）
```

```python
# Python: 手动 Plausibly Exogenous bounds
import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS
import matplotlib.pyplot as plt

delta_grid = np.linspace(-0.1, 0.1, 41)
results = []

for delta in delta_grid:
    df_adj = df.copy()
    df_adj['y_adj'] = df_adj['outcome'] - delta * df_adj['instrument']
    
    formula = f"y_adj ~ 1 + control1 + control2 + [endogenous_x ~ instrument]"
    try:
        res = IV2SLS.from_formula(formula, data=df_adj).fit(cov_type='robust')
        ci  = res.conf_int().loc['endogenous_x']
        results.append({
            'delta': delta,
            'beta':  res.params['endogenous_x'],
            'ci_lo': ci['lower'],
            'ci_hi': ci['upper']
        })
    except Exception:
        pass

bounds = pd.DataFrame(results)
fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(bounds['delta'], bounds['ci_lo'], bounds['ci_hi'], alpha=0.2)
ax.plot(bounds['delta'], bounds['beta'], label='IV estimate')
ax.axhline(0, linestyle='--', color='gray')
ax.set_xlabel('Assumed direct effect δ of Z on Y')
ax.set_ylabel('IV Estimate of β')
ax.set_title('Plausibly Exogenous Sensitivity Analysis')
plt.tight_layout()
plt.savefig('output/iv_plausibly_exogenous.png', dpi=150)
```

```stata
* Stata: Plausibly Exogenous（conley 包）
* ssc install plausexog

* UCI 方法
plausexog uci outcome endogenous_x control1 control2 (endogenous_x = instrument), ///
    gmin(-0.1) gmax(0.1) grid(0.01) level(95)

* LTZ 方法（Local to Zero，需指定 delta 的标准差）
plausexog ltz outcome endogenous_x control1 control2 (endogenous_x = instrument), ///
    omega(0.0025)   // delta ~ N(0, 0.05^2)，即 SD=0.05

* 结果：在不同 delta 假设下的 beta 区间
```

**何时需要做 Plausibly Exogenous 分析：**
- 所有自然实验 IV 都建议做（作为稳健性检验）
- 工具变量有明显的"直接效应渠道"争议时（如 AJR 的地理因素争议）
- 过度识别检验（Hansen J）显示 p 值接近 0.10 时
- 在 Top Journal 投稿时，审稿人常常要求这类敏感性分析

---

### D. Shift-Share IV 代码模板（三语言）

#### D1. 数据结构

```
分析单位：地区-年份面板

变量：
  df_panel: 地区-年份数据（含 outcome、controls、region_id、year）
  df_shares: 地区-行业基期份额（region_id × industry_id → share）
  df_shifts: 全国-行业-年份增速（industry_id × year → national_growth_rate）
```

#### D2. R 代码（推荐：bartik.weight 包）

```r
# 安装
# devtools::install_github("jrgcmu/bartik.weight")
library(bartik.weight)
library(fixest)
library(dplyr)

# ---- 第一步：构造 Bartik 工具变量 ----
# df_shares: region × industry 的基期份额矩阵
# df_shifts: industry × year 的全国增速（已剔除本地区）

bartik_iv <- bartik.weight(
  data_share = df_shares,   # 长格式: region_id, industry_id, share
  data_shift = df_shifts,   # 长格式: industry_id, year, national_growth
  id_share   = "region_id",
  id_shift   = "year",
  id_industry= "industry_id",
  share_var  = "share",
  shift_var  = "national_growth"
)

# 合并到主数据
df_panel <- df_panel %>%
  left_join(bartik_iv %>% select(region_id, year, bartik_z), 
            by = c("region_id", "year"))

# ---- 第二步：2SLS 估计 ----
res_bartik <- feols(
  outcome ~ control1 + control2 | region_fe + year_fe | endogenous_x ~ bartik_z,
  data    = df_panel,
  cluster = ~region_id  # 地区级聚类
)
summary(res_bartik)
fitstat(res_bartik, type = c("ivf", "kpr"))

# ---- 第三步：GPSS 诊断（Rotemberg 权重）----
# 识别哪些行业驱动工具变量的识别力
gpss_weights <- bartik.weight(
  data_share = df_shares,
  data_shift = df_shifts,
  id_share   = "region_id",
  id_shift   = "year",
  id_industry= "industry_id",
  share_var  = "share",
  shift_var  = "national_growth",
  y_var      = "outcome",       # 需要 outcome 才能算 Rotemberg 权重
  x_var      = "endogenous_x",
  controls   = c("control1", "control2"),
  weight_type = "rotemberg"     # GPSS 权重
)

# 报告前 5 个最重要行业
print(head(gpss_weights %>% arrange(desc(abs(rotemberg_weight))), 5))
# 期望：没有单一行业主导（权重分散），且权重最大的行业排他性可信

# ---- 第四步：BHJ 方法（冲击外生性）----
# 用 shifts 作为 IV，shares 作为 controls/weights
# 需要 shifts 在行业层面近似随机
bhj_res <- feols(
  outcome ~ control1 + control2 | region_fe + year_fe |
    endogenous_x ~ i(industry_id, national_growth, ref = "mfg"),  # 行业 × 冲击
  data    = df_long,   # 地区-行业-年份长格式
  weights = ~share,    # 用历史份额加权
  cluster = ~region_id
)
summary(bhj_res)

# ---- 第五步：AKM 标准误修正 ----
# 处理行业层面的空间相关性
# install.packages("ssaggregate")  # R 版本 AKM
# 或直接在行业×年份层面聚类
res_akm_se <- feols(
  outcome ~ control1 + control2 | region_fe + year_fe | endogenous_x ~ bartik_z,
  data    = df_panel,
  cluster = ~industry_id  # AKM 推荐：行业层面聚类
)
etable(res_bartik, res_akm_se,
       headers = c("Region cluster", "Industry cluster (AKM)"))
```

#### D3. Python 代码

```python
import pandas as pd
import numpy as np
from linearmodels.iv import IV2SLS
from linearmodels import PanelOLS

# ---- 构造 Bartik 工具变量 ----
def make_bartik(df_shares, df_shifts,
                region_col='region_id', industry_col='industry_id',
                year_col='year', share_col='share', shift_col='national_growth'):
    """
    构造 Bartik IV: Z_it = sum_k(s_{ik,t0} * g_{kt})
    
    df_shares: 基期份额，含 region_id, industry_id, share
    df_shifts: 全国行业增速（已剔除本地区），含 industry_id, year, national_growth
    """
    # 合并
    merged = df_shares.merge(df_shifts, on=industry_col, how='left')
    
    # 加权求和
    bartik = (
        merged
        .assign(weighted_shift=lambda x: x[share_col] * x[shift_col])
        .groupby([region_col, year_col])['weighted_shift']
        .sum()
        .reset_index()
        .rename(columns={'weighted_shift': 'bartik_z'})
    )
    return bartik

bartik_z = make_bartik(df_shares, df_shifts)
df_panel = df_panel.merge(bartik_z, on=['region_id', 'year'], how='left')

# ---- 2SLS with Panel FE ----
# linearmodels 需要 MultiIndex
df_iv = df_panel.set_index(['region_id', 'year'])

formula = "outcome ~ 1 + EntityEffects + TimeEffects + [endogenous_x ~ bartik_z]"
res = IV2SLS.from_formula(formula, data=df_iv).fit(cov_type='clustered',
                                                    clusters=df_iv.index.get_level_values('region_id'))
print(res.summary)

# ---- GPSS Rotemberg 权重（手动近似）----
# 对每个行业 k，计算 IV_{-k}（剔除行业 k 后的 Bartik）
industries = df_shares['industry_id'].unique()
rotemberg_list = []

for k in industries:
    shares_k  = df_shares[df_shares['industry_id'] != k]
    shifts_k  = df_shifts[df_shifts['industry_id'] != k]
    bartik_k  = make_bartik(shares_k, shifts_k)
    
    df_k = df_panel.merge(bartik_k.rename(columns={'bartik_z': 'bartik_nok'}),
                          on=['region_id', 'year'], how='left')
    df_k_idx = df_k.set_index(['region_id', 'year'])
    
    try:
        res_k = IV2SLS.from_formula(
            "outcome ~ 1 + EntityEffects + TimeEffects + [endogenous_x ~ bartik_nok]",
            data=df_k_idx
        ).fit(cov_type='unadjusted')
        beta_k = res_k.params['endogenous_x']
    except Exception:
        beta_k = np.nan
    
    rotemberg_list.append({'industry': k, 'beta_without_k': beta_k})

beta_main = res.params['endogenous_x']
rotemberg_df = pd.DataFrame(rotemberg_list)
rotemberg_df['rotemberg_weight'] = beta_main - rotemberg_df['beta_without_k']
print(rotemberg_df.sort_values('rotemberg_weight', key=abs, ascending=False).head(5))
```

#### D4. Stata 代码（ssaggregate 包）

```stata
* 安装
* net install ssaggregate, from("https://raw.githubusercontent.com/jrhuit/ssaggregate/master")
* ssc install ivreg2
* ssc install ivreghdfe

* ---- 第一步：用 ssaggregate 构造 Bartik IV ----
* 假设数据：地区-行业-年份长格式
* 变量：region_id, industry_id, year, share (基期份额), national_growth (全国增速)

* 在行业-年份层面聚合（AKM 方法）
ssaggregate outcome endogenous_x control1 control2, ///
    n(national_growth) s(share)          /// shifts & shares
    t(year) l(industry_id)               /// 时间变量 & 行业变量
    controls(i.year i.region_id)         /// 固定效应
    by(region_id year)                   /// 地区-年份聚合
    saveas(data_agg.dta)                 // 保存聚合数据

* ---- 第二步：在聚合数据上做 IV 回归（AKM 修正 SE）----
use data_agg.dta, clear

* BHJ 方法：在行业×年份层面做 IV，份额加权
ivreg2 outcome_agg control1_agg control2_agg (endogenous_agg = national_growth), ///
    absorb(industry_id year) cluster(industry_id) first
* 注意：这里的 cluster(industry_id) 即 AKM 标准误

* ---- 第三步：传统 Bartik（GPSS 方法）----
* 先在原始数据中构造 Bartik IV
bysort region_id year: gen bartik_z = sum(share * national_growth)

* 2SLS with HDFE
ivreghdfe outcome control1 control2 (endogenous_x = bartik_z), ///
    absorb(region_id year) cluster(region_id) first

* ---- 第四步：GPSS 诊断（Rotemberg 权重，手动循环）----
* 获取行业列表
levelsof industry_id, local(industries)

foreach k of local industries {
    * 排除行业 k，重新计算 Bartik
    bysort region_id year: gen bartik_nok = sum(share * national_growth) if industry_id != `k'
    bysort region_id year: replace bartik_nok = sum(bartik_nok)
    
    * 回归
    qui ivreghdfe outcome control1 control2 (endogenous_x = bartik_nok), ///
        absorb(region_id year) cluster(region_id)
    
    scalar beta_nok_`k' = _b[endogenous_x]
    drop bartik_nok
}

* Rotemberg 权重 = beta_full - beta_nok_k（贡献度）
```

**Shift-Share IV 诊断清单：**

| 诊断项 | 方法 | 通过标准 |
|--------|------|----------|
| 第一阶段强度 | Kleibergen-Paap rk F | > 10（或报告 AR CI） |
| GPSS Rotemberg 权重 | bartik.weight / 手动循环 | 无单一行业主导（< 30% 权重） |
| 关键行业排他性 | 对 top-weight 行业做安慰剂 | 直接效应不显著 |
| 预期效应方向 | 检验 shifts 与 shares 的独立性 | shares 与 shifts 在基期无相关 |
| AKM 标准误 | industry-level 聚类 | 报告并与 region 聚类比较 |
| BHJ 冲击随机性 | shifts 与行业前定特征回归 | 无显著预测关系 |
