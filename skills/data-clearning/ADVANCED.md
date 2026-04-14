# 数据清洗 — 高级内容

本文件按需加载（ADVANCED.md）。包含：
- Rubin's rules Python 完整实现
- 面板平衡性 R 完整代码（fixest 并行对比）
- 插值操作前后完整诊断代码
- 文本变量编码
- GLM-QMLE 完整实现 + 模型检验套件
- 比率变量 fractional response model
- 差异分数 Wald 检验完整代码
- 有限混合模型参考
- 多层模型中心化详细指南
- Heckman 两阶段说明（分析阶段执行）

---

## Rubin's Rules 合并 MICE 结果（Python）

```python
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

def rubin_pool(coefs_list, ses_list):
    M = len(coefs_list)
    beta_bar = np.mean([c.values for c in coefs_list], axis=0)
    U_bar = np.mean([se.values**2 for se in ses_list], axis=0)
    B = np.var([c.values for c in coefs_list], axis=0, ddof=1)
    T = U_bar + (1 + 1/M) * B
    SE_pooled = np.sqrt(T)
    r = (1 + 1/M) * B / U_bar
    df_barnard = (M - 1) * (1 + 1/r)**2
    t_stats = beta_bar / SE_pooled
    p_vals = 2 * t_dist.sf(abs(t_stats), df=df_barnard)
    return pd.DataFrame({
        "coef": beta_bar.round(6), "se": SE_pooled.round(6),
        "t": t_stats.round(4), "p": p_vals.round(4),
    }, index=coefs_list[0].index)

# 使用示例
import statsmodels.formula.api as smf
datasets = mice_impute(df, IMP_VARS, M=5)
coefs_list, ses_list = [], []
for df_imp in datasets:
    model = smf.ols("y ~ revenue + assets + leverage + age", data=df_imp).fit()
    coefs_list.append(model.params)
    ses_list.append(model.bse)
pooled = rubin_pool(coefs_list, ses_list)
```

---

## 面板平衡性 R 完整代码

```r
library(dplyr); library(fixest)

T_total <- n_distinct(df$year)
panel_summary <- df %>%
  group_by(firm_id) %>%
  summarise(n_obs=n(), t_min=min(year), t_max=max(year),
            t_gap=(t_max-t_min+1)-n_obs)

cat("总个体:", n_distinct(df$firm_id),
    "| 平衡:", sum(panel_summary$n_obs==T_total),
    "| 有缺口:", sum(panel_summary$t_gap>0), "\n")

balanced_ids <- panel_summary %>% filter(n_obs==T_total) %>% pull(firm_id)
df_balanced <- df %>% filter(firm_id %in% balanced_ids)

res_unbal <- feols(y ~ x1+x2 | firm_id+year, data=df, cluster=~firm_id)
res_bal   <- feols(y ~ x1+x2 | firm_id+year, data=df_balanced, cluster=~firm_id)
etable(res_unbal, res_bal, headers=c("非平衡（主回归）","平衡（稳健性）"))
```

---

## 插值操作完整诊断代码

```python
def interpolate_with_diagnostics(df, entity_col, time_col, var, max_gap=2):
    df = df.sort_values([entity_col, time_col]).copy()
    df[f"{var}_orig"] = df[var]
    df[f"{var}_filled"] = (
        df.groupby(entity_col)[var]
        .transform(lambda x: x.interpolate(method="linear", limit=max_gap,
                                            limit_direction="forward"))
    )
    df[f"{var}_imputed"] = df[f"{var}_orig"].isnull() & df[f"{var}_filled"].notna()
    n_imp = df[f"{var}_imputed"].sum()
    n_tot = df[var].notna().sum() + n_imp
    pct = n_imp / n_tot * 100
    sd_orig = df[f"{var}_orig"].std()
    sd_fill = df[f"{var}_filled"].std()
    sd_shrink = (sd_orig - sd_fill) / sd_orig * 100

    print(f"=== 插值诊断: {var} ===")
    print(f"插值观测: {n_imp} ({pct:.1f}%)")
    print(f"SD 原始: {sd_orig:.4f} → 插值后: {sd_fill:.4f} (缩小{sd_shrink:.1f}%)")
    if pct > 10:
        print("⚠️ 插值>10%！主回归应用非平衡面板")
    elif sd_shrink > 20:
        print("⚠️ SD缩小>20%，慎用插值")
    return df
```

---

## 文本变量编码

```python
from sklearn.preprocessing import LabelEncoder

# Label encoding（有序分类）
le = LabelEncoder()
df["industry_code"] = le.fit_transform(df["industry_name"])

# One-hot encoding
industry_dummies = pd.get_dummies(df["industry_name"], prefix="ind", drop_first=True)
df = pd.concat([df, industry_dummies], axis=1)

# 手动映射（推荐：明确控制编码含义）
region_map = {"北京":1, "上海":2, "广东":3, "浙江":4}
df["region_code"] = df["region"].map(region_map)
```

```r
df$industry_code <- as.integer(factor(df$industry_name))
industry_dummies <- model.matrix(~industry_name - 1, data=df)
```

---

## GLM-QMLE 完整实现

> 替代 OLS + log(Y) 的推荐方案。在异方差下一致，自然处理零值。

### Python

```python
import statsmodels.api as sm
import numpy as np

def fit_glm_qmle(df, formula_y, formula_X_cols, cluster_col=None):
    y = df[formula_y].values
    X = sm.add_constant(df[formula_X_cols].values)
    col_names = ["const"] + formula_X_cols

    model = sm.GLM(y, X, family=sm.families.Poisson(link=sm.families.links.Log()))

    if cluster_col is not None:
        result = model.fit(cov_type="cluster",
                           cov_kwds={"groups": df[cluster_col].values})
    else:
        result = model.fit(cov_type="HC1")

    print("=== GLM-QMLE (Poisson, log link) ===")
    print(f"N = {result.nobs:.0f}")
    summary_df = pd.DataFrame({
        "coef": result.params, "se": result.bse,
        "z": result.tvalues, "p": result.pvalues,
    }, index=col_names)
    print(summary_df.round(4).to_string())

    # 系数解释: exp(β)-1 = X增加1单位时Y的百分比变化
    print("\n半弹性解释 (% change in Y per unit X):")
    for i, name in enumerate(col_names[1:], 1):
        pct = (np.exp(result.params[i]) - 1) * 100
        print(f"  {name}: {pct:+.2f}%")

    return result
```

### R

```r
# GLM-QMLE
fit_qmle <- glm(y ~ x1 + x2 + x3, family=quasipoisson(link="log"), data=df)
summary(fit_qmle)

# 聚类稳健标准误
library(sandwich); library(lmtest)
coeftest(fit_qmle, vcov=vcovCL(fit_qmle, cluster=df$firm_id))
```

### 模型检验套件

```r
library(lmtest)

# 1. RESET 检验（函数形式）
ols_fit <- lm(y ~ x1 + x2, data=df)
resettest(ols_fit, power=2:3, type="fitted")
# p<0.05 → 存在未识别的非线性关系 → 考虑 GLM

# 2. Pregibon Link 检验
glm_fit <- glm(y ~ x1 + x2, family=poisson(link="log"), data=df)
linktest_df <- data.frame(y=df$y, yhat=predict(glm_fit, type="link"),
                          yhat2=predict(glm_fit, type="link")^2)
link_check <- glm(y ~ yhat + yhat2, family=poisson(link="log"), data=linktest_df)
summary(link_check)  # yhat2 不显著 → link function 正确

# 3. 修正版 Park 检验（选择最优分布族）
resid_raw <- df$y - predict(glm_fit, type="response")
log_resid2 <- log(resid_raw^2)
log_mu <- log(predict(glm_fit, type="response"))
park_fit <- lm(log_resid2 ~ log_mu)
cat("Park检验斜率:", coef(park_fit)[2], "\n")
cat("  ≈0 → Gaussian; ≈1 → Poisson; ≈2 → Gamma; ≈3 → Inverse Gaussian\n")
```

---

## 比率变量：Fractional Response Model

当因变量是自然比例（0-1 范围），如"创新销售额占比"：

```python
# Fractional logit (Papke & Wooldridge, 1996)
import statsmodels.api as sm
y_frac = df["innovation_share"].values  # 0-1
X = sm.add_constant(df[["rd_intensity","firm_size","age"]].values)
model = sm.GLM(y_frac, X,
               family=sm.families.Binomial(link=sm.families.links.Logit()))
result = model.fit(cov_type="HC1")
print(result.summary())
```

```r
# R: fractional logit
fit_frac <- glm(innovation_share ~ rd_intensity + firm_size + age,
                family=quasibinomial(link="logit"), data=df)
coeftest(fit_frac, vcov=vcovCL(fit_frac, cluster=df$firm_id))
```

---

## 差异分数 Wald 检验

### 差异分数作为因变量

```python
import statsmodels.formula.api as smf
from scipy.stats import f as f_dist

# 分别估计 ya 和 yb
fit_a = smf.ols("ya ~ x1 + x2 + x3", data=df).fit()
fit_b = smf.ols("yb ~ x1 + x2 + x3", data=df).fit()

# 检验: β_a(x1) = β_b(x1)?
diff = fit_a.params["x1"] - fit_b.params["x1"]
se_diff = np.sqrt(fit_a.bse["x1"]**2 + fit_b.bse["x1"]**2)
# 注意: 若ya和yb相关（同一样本），需要SUR获得正确标准误
print(f"β_a - β_b = {diff:.4f}, SE = {se_diff:.4f}, t = {diff/se_diff:.4f}")
print("若不显著 → 差异分数约束成立，可使用 Δy")
print("若显著 → 应分别建模 ya 和 yb")
```

### 差异分数作为自变量

```r
# 检验 β_xa = -β_xb（等量反号条件）
fit_full <- lm(y ~ xa + xb + controls, data=df)
library(car)
linearHypothesis(fit_full, "xa + xb = 0")
# p>0.05 → 约束成立，可用差异分数 Δx = xa - xb
# p<0.05 → 约束不成立，应分别纳入 xa 和 xb
```

### SUR 精确检验（推荐）

```r
library(systemfit)
eq_a <- ya ~ x1 + x2 + x3
eq_b <- yb ~ x1 + x2 + x3
sur <- systemfit(list(eq_a=eq_a, eq_b=eq_b), method="SUR", data=df)
linearHypothesis(sur, "eq_a_x1 - eq_b_x1 = 0")
# 正确处理 ya 和 yb 的相关性
```

---

## 有限混合模型（替代中位数分割）

当理论上确需将连续变量分组时：

```r
library(flexmix)
# 数据驱动确定分组数（BIC选择）
bic_vals <- sapply(2:5, function(k) {
  fit <- flexmix(y ~ x1 + x2, data=df, k=k)
  BIC(fit)
})
best_k <- which.min(bic_vals) + 1
cat("最优分组数:", best_k, "\n")

# 最终模型
fit_mix <- flexmix(y ~ x1 + x2, data=df, k=best_k)
summary(fit_mix)
df$cluster <- clusters(fit_mix)
```

```python
from sklearn.mixture import GaussianMixture
bics = [GaussianMixture(n_components=k, random_state=42).fit(X).bic(X)
        for k in range(2, 6)]
best_k = range(2,6)[np.argmin(bics)]
gmm = GaussianMixture(n_components=best_k, random_state=42).fit(X)
df["cluster"] = gmm.predict(X)
```

---

## 多层模型中心化详细指南

```
中心化选择 → 取决于研究问题

问题: "个体偏离组均值对 Y 的影响？"（within 效应）
  → 组均值中心化 (Group-Mean Centering, GMC)
  x_ij_gmc = x_ij - x̄_j
  将组均值 x̄_j 加回 Level-2 分离 between 效应

问题: "X 的整体效应（不区分 within/between）？"
  → 不中心化 或 总体均值中心化
  但 Antonakis et al. (2021) 警告：
  总体均值中心化在 HLM 中通常无用，不改变斜率
  且可能误导对截距的解释

实际操作:
  1. 默认推荐 GMC + 组均值回加（Mundlak 装置）
  2. 明确报告选择依据
  3. 稳健性检验对比不同中心化方案
```

```r
library(lme4)

# 组均值中心化 + 组均值回加（Mundlak 装置）
df <- df %>%
  group_by(firm_id) %>%
  mutate(x_mean = mean(x, na.rm=TRUE),
         x_within = x - x_mean) %>%
  ungroup()

# within + between 分离
fit_mundlak <- lmer(y ~ x_within + x_mean + z + (1|firm_id), data=df)
summary(fit_mundlak)
# x_within = within 效应（个体时间变化）
# x_mean   = between 效应（组间差异）
```

---

## Heckman 两阶段说明

**本 skill 不执行 Heckman。** 仅说明调用方式：

- **适用**：MNAR 缺失（如只有特定企业被观测）
- **执行位置**：OLS / DID skill 稳健性检验
- **Python**：`statsmodels.Probit` 手动两阶段（**不要** `from heckman import Heckman`）
- **R**：`sampleSelection::selection()`（推荐，标准误自动纠正）
- **核心**：需至少一个排他性变量（选择方程显著，结果方程不显著）
