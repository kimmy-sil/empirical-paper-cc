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
