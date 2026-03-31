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

---

### 倍增比自动计算

跑完 2SLS 后，自动计算 |β_2SLS / β_OLS|。若比值 > 5，说明 IV 估计量放大远超 OLS，通常意味着弱工具、LATE 群体特殊、或存在测量误差被放大，需要额外审慎解释。

**经验判断规则：**
- |β_2SLS / β_OLS| ≈ 1~2：正常，IV 修正了适度的衰减偏误
- |β_2SLS / β_OLS| ∈ (2, 5]：需论证 Complier 群体的特殊性
- |β_2SLS / β_OLS| > 5：⚠️ 警告——极可能存在弱工具问题或估计不稳定

```python
# Python: 自动计算倍增比（linearmodels + statsmodels）
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

# ---- OLS 估计 ----
ols_formula = f"{outcome} ~ 1 + {endogenous} + {' + '.join(controls)}"
res_ols = smf.ols(ols_formula, data=df).fit(cov_type='HC3')
beta_ols = res_ols.params[endogenous]

# ---- 2SLS 估计 ----
iv_formula = f"{outcome} ~ 1 + {' + '.join(controls)} + [{endogenous} ~ {' + '.join(instruments)}]"
res_iv = IV2SLS.from_formula(iv_formula, data=df).fit(cov_type='robust')
beta_iv = res_iv.params[endogenous]

# ---- 倍增比计算 ----
ratio = abs(beta_iv / beta_ols) if beta_ols != 0 else float('inf')
print(f"OLS  β = {beta_ols:.4f}")
print(f"2SLS β = {beta_iv:.4f}")
print(f"|β_2SLS / β_OLS| = {ratio:.2f}")

if ratio > 5:
    print("⚠️  警告：倍增比 > 5，请检查：")
    print("   1. 第一阶段 F 统计量是否充分（>104.7 for单工具）")
    print("   2. Complier 群体是否具有特殊高效应特征（需论文中明确讨论）")
    print("   3. 测量误差放大效应（Classical ME → attenuation in OLS → IV 反向放大）")
elif ratio > 2:
    print("⚠️  注意：倍增比 > 2，建议在论文中解释 LATE 超过 OLS 的经济学原因")
else:
    print("✓  倍增比在正常范围内")
```

```r
# R: fixest feols vs feols with IV 对比 + 倍增比
library(fixest)

# OLS
res_ols <- feols(
  outcome ~ endogenous_x + control1 + control2 | unit_fe + time_fe,
  data    = df,
  cluster = ~cluster_var
)

# 2SLS（IV）
res_iv <- feols(
  outcome ~ control1 + control2 | unit_fe + time_fe | endogenous_x ~ z1 + z2,
  data    = df,
  cluster = ~cluster_var
)

# 提取系数
beta_ols <- coef(res_ols)["endogenous_x"]
beta_iv  <- coef(res_iv)["fit_endogenous_x"]

ratio <- abs(beta_iv / beta_ols)
cat(sprintf("OLS  β = %.4f\n", beta_ols))
cat(sprintf("2SLS β = %.4f\n", beta_iv))
cat(sprintf("|β_2SLS / β_OLS| = %.2f\n", ratio))

if (ratio > 5) {
  warning("倍增比 > 5！检查弱工具、Complier 特殊性和测量误差放大问题。")
} else if (ratio > 2) {
  message("注意：倍增比 > 2，请在论文中论证 LATE 超过 OLS 的原因。")
}

# 对比表
etable(res_ols, res_iv,
       headers = c("OLS", "2SLS"),
       fitstat = ~ ivf + kpr + n)
```

---

### Jackknife 影响力诊断

逐一剔除每个聚类（或地区/行业单元），重新估计 2SLS 系数。若任何一次剔除导致系数变化 > 20%，说明结论对该聚类高度敏感，需在论文中讨论。

**使用场景：**
- 聚类数量较少（<50 个）时特别重要
- 识别主要由少数大聚类驱动时（需检验）
- 对 Shift-Share IV 结构尤其关键（某些行业权重过大）

```python
# Python: Jackknife 影响力诊断
import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS

clusters = df[cluster_var].unique()
iv_formula = f"{outcome} ~ 1 + {' + '.join(controls)} + [{endogenous} ~ {' + '.join(instruments)}]"

# 全样本基准估计
res_full = IV2SLS.from_formula(iv_formula, data=df).fit(cov_type='robust')
beta_full = res_full.params[endogenous]
print(f"Full sample 2SLS β = {beta_full:.4f}")

# Jackknife 循环
jk_results = []
for cl in clusters:
    df_drop = df[df[cluster_var] != cl]
    try:
        res_jk = IV2SLS.from_formula(iv_formula, data=df_drop).fit(cov_type='robust')
        beta_jk = res_jk.params[endogenous]
        pct_change = abs(beta_jk - beta_full) / abs(beta_full) * 100
        jk_results.append({
            'dropped_cluster': cl,
            'beta_jk': beta_jk,
            'pct_change': pct_change,
            'flag': '⚠️ >20%' if pct_change > 20 else ''
        })
    except Exception as e:
        jk_results.append({'dropped_cluster': cl, 'beta_jk': np.nan,
                           'pct_change': np.nan, 'flag': f'Error: {e}'})

jk_df = pd.DataFrame(jk_results).sort_values('pct_change', ascending=False)
print(jk_df.head(10).to_string(index=False))

# 报告警告
flagged = jk_df[jk_df['pct_change'] > 20]
if len(flagged) > 0:
    print(f"\n⚠️  警告：以下 {len(flagged)} 个聚类剔除后系数变化 >20%：")
    print(flagged[['dropped_cluster', 'beta_jk', 'pct_change']].to_string(index=False))
else:
    print("\n✓  所有聚类剔除后系数变化均 ≤20%，结果稳健")

jk_df.to_csv('output/iv_jackknife_cluster.csv', index=False)
```

```r
# R: Jackknife 影响力诊断
library(fixest)
library(dplyr)

# 全样本基准
res_full <- feols(
  outcome ~ control1 + control2 | unit_fe + time_fe | endogenous_x ~ z1 + z2,
  data = df, cluster = ~cluster_var
)
beta_full <- coef(res_full)["fit_endogenous_x"]
cat(sprintf("Full sample 2SLS β = %.4f\n", beta_full))

# Jackknife 循环
clusters <- unique(df$cluster_var)

jk_results <- lapply(clusters, function(cl) {
  df_drop <- df %>% filter(cluster_var != cl)
  tryCatch({
    res_jk <- feols(
      outcome ~ control1 + control2 | unit_fe + time_fe | endogenous_x ~ z1 + z2,
      data = df_drop, cluster = ~cluster_var
    )
    beta_jk    <- coef(res_jk)["fit_endogenous_x"]
    pct_change <- abs(beta_jk - beta_full) / abs(beta_full) * 100
    data.frame(dropped = cl, beta_jk = beta_jk, pct_change = pct_change,
               flag = ifelse(pct_change > 20, "⚠️ >20%", ""))
  }, error = function(e) {
    data.frame(dropped = cl, beta_jk = NA, pct_change = NA, flag = "Error")
  })
}) |> bind_rows() |> arrange(desc(pct_change))

print(head(jk_results, 10))

# 警告提示
flagged <- jk_results[!is.na(jk_results$pct_change) & jk_results$pct_change > 20, ]
if (nrow(flagged) > 0) {
  warning(sprintf("%d 个聚类剔除后系数变化 >20%%，需在论文中讨论", nrow(flagged)))
  print(flagged)
} else {
  cat("✓ 所有聚类剔除后系数变化均 ≤20%\n")
}

write.csv(jk_results, "output/iv_jackknife_cluster.csv", row.names = FALSE)
```

---

### DID 变量作 IV 的范式

**适用场景：** 找不到传统 IV（地理/历史工具变量）时，利用政策冲击（policy shock）与行业/地区前定特征的交互项构造工具变量。本质是 Shift-Share 的简化版本。

**构造方法：**

$$Z_{it} = \text{PolicyShock}_t \times \text{PreChar}_i$$

- $\text{PolicyShock}_t$：外生的政策时间冲击（如行业整体监管变化、全国性政策）
- $\text{PreChar}_i$：个体 $i$ 的基期特征（固定，不受政策影响）——如行业类别、历史规模、初始杠杆率

**识别逻辑：**
- 相关性：基期特征决定了政策冲击对内生变量 $D$ 的影响程度（不同特征 → 不同力度）
- 排他性：政策冲击对结果的影响只通过处理变量 $D$，不直接影响 $Y$（需论证）

**识别假设强化检验：**
1. 对不应受影响的结果变量做安慰剂检验
2. 检验 $\text{PreChar}_i$ 在政策前与结果趋势的独立性（pre-trend of PreChar × time interaction）
3. 配套 plausexog 检验（见下节）

```r
# R: DID变量作IV — 代码模板
library(fixest)
library(dplyr)

# ---- 构造 IV：政策冲击 × 基期特征 ----
# policy_shock: 政策年份哑变量（post=1），或更细粒度的政策力度
# pre_char: 基期行业/地区特征（在政策前确定，此后不变）
df <- df %>%
  mutate(
    iv_interact = policy_shock * pre_char_baseline  # 交互项 = IV
  )

# ---- 第一阶段检验 ----
first_stage <- feols(
  endogenous_x ~ iv_interact + control1 + control2 | unit_fe + time_fe,
  data    = df,
  cluster = ~cluster_var
)
cat("第一阶段 F 统计量（排除工具变量）:\n")
fitstat(first_stage, type = "ivf")

# ---- 2SLS 主估计 ----
res_iv_did <- feols(
  outcome ~ control1 + control2 | unit_fe + time_fe | endogenous_x ~ iv_interact,
  data    = df,
  cluster = ~cluster_var
)
summary(res_iv_did)
fitstat(res_iv_did, type = c("ivf", "kpr"))

# ---- 排他性安慰剂检验 ----
# 用 IV 解释不相关的结果变量（应不显著）
placebo_iv <- feols(
  unrelated_outcome ~ control1 + control2 | unit_fe + time_fe | unrelated_x ~ iv_interact,
  data    = df,
  cluster = ~cluster_var
)
# 期望：coefficient on fit_unrelated_x ≈ 0

# ---- 平行趋势检验（PreChar × Year 趋势）----
df <- df %>% mutate(pre_char_x_year = pre_char_baseline * year)
trend_check <- feols(
  outcome ~ pre_char_x_year + control1 | unit_fe + year_fe,
  data   = df %>% filter(post == 0),  # 仅政策前样本
  cluster = ~cluster_var
)
# 期望：pre_char_x_year 系数不显著（基期特征与结果趋势无关）
```

---

### plausexog 不完全外生 IV 检验

当排他性约束可能存在小幅违反时（$Z$ 对 $Y$ 有直接效应 $\delta$），Union of Confidence Intervals（UCI）方法在假定 $\delta \in [-\delta_0, \delta_0]$ 下计算 $\beta$ 的稳健区间。

**UCI 方法说明：**
1. 研究者基于领域知识设定直接效应的上界 $\delta_0$（关键假设，需论证）
2. 对 $\delta$ 网格中的每个值，计算条件 CI
3. 取所有 CI 的并集（union），得到稳健置信区间
4. 若并集 CI 仍不包含零，结论对排他性约束的小幅违反稳健

**参数设定指南：**
- $\delta_0$ 的选择：参考直接效应渠道的量级估计（如控制直接渠道后系数变化量）
- 从保守的大 $\delta_0$ 开始，逐步缩小，找到结论转折点（"临界 $\delta_0$"）

```r
# R: plausexog UCI 方法（手动实现 + sensemakr）
library(AER)
library(tidyverse)

# ---- 方法1：UCI 手动扫描 ----
# 假定 Z 对 Y 的直接效应 δ 在 [-δ₀, δ₀] 内
delta0 <- 0.05  # 根据领域知识设定
delta_grid <- seq(-delta0, delta0, length.out = 41)

bounds_df <- map_dfr(delta_grid, function(d) {
  # 从结果中减去直接效应，再做 IV
  df_adj <- df %>% mutate(y_adj = outcome - d * instrument)

  res <- ivreg(
    y_adj ~ endogenous_x + control1 + control2 |
              instrument + control1 + control2,
    data = df_adj
  )
  ci <- confint(res)["endogenous_x", ]
  tibble(delta = d,
         beta  = coef(res)["endogenous_x"],
         lo    = ci[1], hi = ci[2])
})

# 并集 CI（UCI）
uci_lo <- min(bounds_df$lo)
uci_hi <- max(bounds_df$hi)
cat(sprintf("UCI 稳健区间: [%.4f, %.4f]\n", uci_lo, uci_hi))

if (uci_lo > 0 | uci_hi < 0) {
  cat("✓ 在 δ ∈ [-δ₀, δ₀] 假设下，结论稳健（区间不含0）\n")
} else {
  cat("⚠️ 区间包含0，结论对排他性违反敏感\n")
}

# 可视化
ggplot(bounds_df, aes(x = delta)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.2, fill = "steelblue") +
  geom_line(aes(y = beta), color = "steelblue", linewidth = 1) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  geom_hline(yintercept = uci_lo, linetype = "dotted") +
  geom_hline(yintercept = uci_hi, linetype = "dotted") +
  labs(
    title    = "plausexog UCI：排他性约束敏感性分析",
    subtitle = sprintf("假设直接效应 δ ∈ [−%.2f, %.2f]", delta0, delta0),
    x        = "直接效应假设值 δ",
    y        = "IV 估计 β"
  ) +
  theme_minimal()
ggsave("output/iv_plausexog_uci.png", dpi = 150)

# ---- 方法2：sensemakr（更系统的敏感性分析）----
# install.packages("sensemakr")
library(sensemakr)

# 以 OLS 作为基础，分析工具变量直接效应的等价混淆强度
base_ols <- lm(outcome ~ endogenous_x + instrument + control1 + control2, data = df)
sens <- sensemakr(
  model               = base_ols,
  treatment           = "instrument",       # 关注 Z 的直接效应
  benchmark_covariates = "control1",        # 以已知控制变量为基准
  kd                  = 1:3
)
plot(sens)    # 等高线图：多强的混淆才能推翻结论
summary(sens) # Robustness value（最小影响力）

# ---- 参数设定结果解读 ----
# 1. 绘制"β 随 δ 变化"图，找到 β 转变符号的临界 δ_crit
delta_crit <- bounds_df %>%
  filter(sign(beta) != sign(bounds_df$beta[bounds_df$delta == 0])) %>%
  summarise(delta_crit = min(abs(delta))) %>%
  pull()
cat(sprintf("结论转折临界值 δ_crit ≈ %.4f\n", delta_crit))
cat(sprintf("即：若 Z 对 Y 的直接效应小于 %.4f，结论不变\n", delta_crit))
```

---

### Lee Bounds

Lee Bounds（Lee 2009）是 plausexog 的替代方案，适用于**样本选择性缺失（selective attrition）**导致排他性约束受损的情形——例如工具变量影响了"是否被观测到"，从而间接影响了结果。

**核心思想：**
- 在单调性假设下，通过对处理组和控制组的尾部删减（trimming）构造处理效应的上界和下界
- Upper bound：删减处理组最差结果后估计
- Lower bound：删减处理组最好结果后估计

**适用条件：**
- Fuzzy RDD 或 IV 设计中存在样本选择性流失（attrition）
- 单调性假设（monotonicity）成立

```r
# R: Lee Bounds 代码框架
# install.packages("leebounds")
library(leebounds)
library(dplyr)

# ---- 基本 Lee Bounds ----
# 需要变量：
#   S: 样本选择指示（1=被观测到，0=缺失）
#   Y: 结果变量（仅对 S=1 观测）
#   D: 处理变量（工具变量/随机分配）

lee_res <- leebounds(
  s = df$selected,     # 样本选择变量（1=进入样本）
  y = df$outcome,      # 结果变量
  d = df$treatment     # 处理/工具变量
)
print(lee_res)
# 输出：Lower bound, Upper bound, 点估计（假设同质效应时）

# ---- 带协变量的 Lee Bounds（tighter bounds）----
# 协变量可以收紧 bounds，但需假设协变量与选择机制独立
lee_cov <- leebounds(
  s      = df$selected,
  y      = df$outcome,
  d      = df$treatment,
  stratify_by = df$stratum  # 按子层做 bounds，再合并
)
print(lee_cov)

# ---- 结果报告 ----
cat(sprintf("Lee Bounds: [%.4f, %.4f]\n",
            lee_res$lower_bound, lee_res$upper_bound))
cat(sprintf("若两端均显著异于0，结论对选择性缺失稳健\n"))

# 与主 IV 估计对比
cat(sprintf("主 2SLS 估计: %.4f\n", beta_iv))
cat(sprintf("Lee Lower:    %.4f\n", lee_res$lower_bound))
cat(sprintf("Lee Upper:    %.4f\n", lee_res$upper_bound))
```

---

### 同行业剩余均值型 IV

**核心思想：** 剔除个体自身后，用同行业（或同地区）其他个体的均值作为工具变量。常用于处理**网络效应内生性**、**同伴效应**或**行业均值混淆**。

**识别逻辑：**
- 相关性：行业/地区均值与个体的内生变量高度相关（行业普遍受到外部冲击）
- 排他性：剔除自身后的均值不直接影响个体结果（无反向因果；需论证无明显溢出效应）

**常见应用：**
- 企业融资成本的行业同期均值（剔除本企业）→ 作为融资约束的 IV
- 地区工资均值（剔除本人）→ 作为个人工资方程的 IV
- 银行同业拆借利率均值（剔除本行）→ 作为银行风险的 IV

```python
# Python: 同行业剩余均值型 IV（leave-self-out mean）
import pandas as pd
import numpy as np
from linearmodels.iv import IV2SLS

# 假设数据：df 含 firm_id, industry_id, year, endogenous_x, outcome, controls

# ---- 构造 leave-self-out 行业均值 IV ----
# 方法：行业均值 - 本企业贡献
df['industry_sum']   = df.groupby(['industry_id', 'year'])['endogenous_x'].transform('sum')
df['industry_count'] = df.groupby(['industry_id', 'year'])['endogenous_x'].transform('count')

# leave-self-out mean = (行业总和 - 自身值) / (行业个数 - 1)
df['iv_leave_out'] = (df['industry_sum'] - df['endogenous_x']) / (df['industry_count'] - 1)

# 若行业只有自身（单独个体），leave-out mean 为 NaN → 剔除
df = df.dropna(subset=['iv_leave_out'])

# ---- 2SLS 估计 ----
df_iv = df.set_index(['firm_id', 'year'])
formula = f"{outcome} ~ 1 + EntityEffects + TimeEffects + {' + '.join(controls)} + [{endogenous} ~ iv_leave_out]"
res = IV2SLS.from_formula(formula, data=df_iv).fit(
    cov_type='clustered',
    clusters=df_iv.index.get_level_values('firm_id')
)
print(res.summary)

# ---- 诊断：第一阶段 F ----
print(f"First Stage F: {res.first_stage.diagnostics['f.stat']:.2f}")
```

```r
# R: 同行业剩余均值型 IV（通用代码）
library(fixest)
library(dplyr)

# ---- 构造 leave-self-out 均值 ----
df <- df %>%
  group_by(industry_id, year) %>%
  mutate(
    industry_sum   = sum(endogenous_x, na.rm = TRUE),
    industry_count = sum(!is.na(endogenous_x)),
    # Leave-self-out mean
    iv_leave_out   = (industry_sum - endogenous_x) / (industry_count - 1)
  ) %>%
  ungroup() %>%
  filter(industry_count > 1)  # 需要至少2家企业才能构造均值

# ---- 检验：IV 与内生变量的相关性 ----
cor_check <- cor(df$iv_leave_out, df$endogenous_x, use = "complete.obs")
cat(sprintf("IV 与内生变量相关系数: %.4f\n", cor_check))

# ---- 2SLS 主估计 ----
res_iv_leave <- feols(
  outcome ~ control1 + control2 | firm_fe + year_fe | endogenous_x ~ iv_leave_out,
  data    = df,
  cluster = ~industry_id  # 在行业层面聚类（考虑同行业相关性）
)
summary(res_iv_leave)
fitstat(res_iv_leave, type = c("ivf", "kpr", "n"))

# ---- 稳健性：按地区构造 leave-self-out ----
# （同理，只需将 group_by 中的 industry_id 改为 region_id）
df <- df %>%
  group_by(region_id, year) %>%
  mutate(iv_region_leave = (sum(endogenous_x) - endogenous_x) / (n() - 1)) %>%
  ungroup()
```

---

### Complier 特征描述模板

IV/2SLS 识别的是 **Complier 群体**（当工具变量 Z 发生变化时，处理变量 D 随之改变的个体）。论文中**必须明确描述 Complier 的特征**，否则读者无法评估 LATE 的外部效度。

**自动生成 Complier 描述文字的代码：**

```r
# R: 自动生成 Complier 特征描述
# 方法：Abadie (2003) κ-weighting — 用 IV 加权估计 Complier 群体的协变量均值

library(fixest)
library(dplyr)

# 步骤1：估计倾向得分 P(Z=1|X)（IV 取值=1 的概率）
ps_model <- glm(instrument ~ control1 + control2 + covariate1 + covariate2,
                data = df, family = binomial())
df$pz <- predict(ps_model, type = "response")

# 步骤2：构造 Abadie kappa 权重
# κ = 1 - D(1-Z)/(1-P(Z)) - (1-D)Z/P(Z)
df <- df %>%
  mutate(
    kappa = 1 -
      (endogenous_x * (1 - instrument)) / (1 - pz) -
      ((1 - endogenous_x) * instrument) / pz
  )

# 步骤3：用 κ 加权估计 Complier 均值（vs 总体均值）
covariates_to_describe <- c("age", "firm_size", "leverage", "region_dummy")

complier_profile <- sapply(covariates_to_describe, function(cov) {
  complier_mean <- weighted.mean(df[[cov]], w = df$kappa, na.rm = TRUE)
  overall_mean  <- mean(df[[cov]], na.rm = TRUE)
  ratio         <- complier_mean / overall_mean
  c(complier_mean = complier_mean, overall_mean = overall_mean, ratio = ratio)
})

print(round(t(complier_profile), 3))

# 步骤4：自动生成描述文字
cat("\n=== Complier 群体特征描述（可直接粘贴至论文）===\n\n")
cat(sprintf(
  "本文工具变量 [%s] 识别的是当 [%s] 发生变化时处理状态随之改变的 Complier 群体。\n",
  "IV名称", "Z变动方向描述"
))
cat(sprintf(
  "相较于总体样本，Complier 群体在以下方面存在系统性差异：\n"
))

for (cov in covariates_to_describe) {
  ratio <- complier_profile["ratio", cov]
  direction <- ifelse(ratio > 1, "高于", "低于")
  cat(sprintf("  - %s 均值比总体样本 %s %.1f%%（Complier: %.3f，总体: %.3f）\n",
              cov, direction, abs(ratio - 1) * 100,
              complier_profile["complier_mean", cov],
              complier_profile["overall_mean", cov]))
}

cat(sprintf(
  "\n因此，本文 2SLS 估计的 LATE 代表 [经济学解释Complier特征] 群体的处理效应，\n",
))
cat("  其政策含义主要适用于该群体，不能直接外推至 Never-taker 或 Always-taker。\n")
```

---

### 错误 7 补充

> **错误 7：不说明 IV 识别的是谁的效应**
>
> 2SLS / IV 估计的是 **LATE（Local Average Treatment Effect）**，即 Complier 群体的平均处理效应。在论文中**必须明确讨论**以下三点：
>
> 1. **谁是 Complier？** 当工具变量 Z 从 0 变为 1 时，哪类个体的处理状态 D 随之变化？用数据描述其可观测特征（年龄、规模、地区等）。
>
> 2. **LATE 与 ATE/ATT 的关系：** 如果研究问题关心总体 ATE，需要额外论证为何 Complier 的效应能代表总体——或明确声明 LATE 的局限性。
>
> 3. **不同 IV 识别不同 LATE：** 当文章使用多个工具变量时，每个 IV 识别的是不同的 Complier 子群体，估计量自然不同，这不是矛盾而是**异质性的证据**。

**论文写作模板：**
```
"本文工具变量 [Z] 识别的是 [Complier 定义，如：因地理距离降低而开始使用金融服务的企业]。
这一群体在 [特征维度] 上与总体样本存在 [差异描述]（见表 X），
因此本文 2SLS 估计量代表该 Complier 群体的 LATE，
其推广到 [Never-taker 群体] 需要额外假设。"
```

---

### Estimand 声明

**IV / 2SLS → LATE（Compliers 的效应）**

在论文中，每次报告 IV/2SLS 结果时，**必须**包含以下声明：

| 声明项目 | 内容要求 |
|----------|---------|
| 估计量定义 | 明确标注"本文 IV 估计量为 LATE（局部平均处理效应）" |
| Complier 群体 | 描述当 Z 变动时处理状态改变的个体特征（用数据刻画） |
| 外部效度边界 | 声明 LATE 不等于 ATE，不适用于 Always-taker 和 Never-taker |
| 不同 IV 的比较 | 若使用多个 IV，解释估计量差异来自不同 Complier 群体 |

**标准声明模板（论文脚注或正文）：**
```
本文 2SLS 估计量识别的是工具变量 [Z名称] 下的局部平均处理效应（LATE），
即当 [Z描述] 时处理状态随之改变的 Complier 群体的平均效应。
该群体的特征详见表 [X]（Complier 特征分析）。
LATE 不能直接外推至总体平均处理效应（ATE），
政策含义主要适用于类似 Complier 特征的群体。
```
