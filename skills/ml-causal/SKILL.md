# ml-causal — 机器学习因果推断技能指南

适用场景：高维控制变量选择、异质性处理效应探索、ML辅助识别策略。触发关键词：DML、因果森林、LASSO、doubleml、grf、econml、ML因果、异质性处理效应、HTE。

> **重要提示：ML因果推断是辅助工具，不能替代清晰的识别策略。** 无论使用何种ML方法，核心因果识别（外生变异来源、可忽略性假设等）必须先于ML方法成立。

---

## 1. Double/Debiased Machine Learning（DML）

### 1.1 方法背景

**来源：** Chernozhukov et al. (2018)，*The Econometrics Journal*，21(1)，C1–C68。

**适用场景：**
- 控制变量维度高（特征数量 > 样本量的10%以上）
- 需要在高维控制变量下一致估计低维参数（ATE、ATT）
- 担心传统OLS选变量时的数据挖掘问题

**核心思想（部分线性模型）：**

$$Y = D\theta_0 + g_0(X) + \varepsilon, \quad \mathbb{E}[\varepsilon | D, X] = 0$$
$$D = m_0(X) + \upsilon, \quad \mathbb{E}[\upsilon | X] = 0$$

其中 $X$ 为高维控制变量，$D$ 为处理变量，$\theta_0$ 为感兴趣的因果参数。

**Neyman正交化（消除正则化偏误）：**
1. 用ML模型（LASSO/随机森林/XGBoost等）预测 $\hat{g}_0(X) = \hat{E}[Y|X]$
2. 用ML模型预测 $\hat{m}_0(X) = \hat{E}[D|X]$
3. 用残差回归估计 $\theta_0$：

$$\hat{\theta}_0 = \frac{1}{n}\sum_{i} \tilde{D}_i \tilde{Y}_i \Big/ \frac{1}{n}\sum_{i} \tilde{D}_i^2$$

其中 $\tilde{Y}_i = Y_i - \hat{g}_0(X_i)$，$\tilde{D}_i = D_i - \hat{m}_0(X_i)$。

**Cross-fitting（交叉拟合，避免过拟合偏误）：**
将样本分为 K 折，用其他 K-1 折训练模型，在第 K 折预测残差，循环进行。

### 1.2 Python 代码（econml / doubleml）

```python
# ============================================================
# DML — Python（doubleml 包）
# pip install doubleml econml scikit-learn
# ============================================================

import numpy as np
import pandas as pd
from doubleml import DoubleMLPLR, DoubleMLData
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

# --- 数据准备 ---
df = pd.read_csv("data.csv")

# 定义变量（根据实际修改）
Y_VAR = "outcome"
D_VAR = "treatment"
X_VARS = [c for c in df.columns if c not in [Y_VAR, D_VAR, "entity_id", "year"]]

dml_data = DoubleMLData(df, y_col=Y_VAR, d_cols=D_VAR, x_cols=X_VARS)

# --- ML 学习器选择 ---
# 选项1：随机森林（非线性，高维友好）
ml_g = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
ml_m = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)

# 选项2：LASSO（稀疏高维）
ml_g = LassoCV(cv=5, random_state=42)
ml_m = LassoCV(cv=5, random_state=42)

# --- DML 部分线性模型 ---
dml_plr = DoubleMLPLR(
    obj_dml_data=dml_data,
    ml_l=ml_g,          # 预测 Y 的学习器
    ml_m=ml_m,          # 预测 D 的学习器
    n_folds=5,          # K折交叉拟合
    n_rep=3,            # 重复次数（取中位数，减少随机性）
    score='partialling out'  # 标准DML
)

dml_plr.fit()
print(dml_plr.summary)

# 输出示例：
# coef  std err   t      P>|t|  [0.025  0.975]
# d     0.082   0.021  3.90   0.000   0.041   0.123

# --- IV 扩展（DoubleMLIIVM）---
# from doubleml import DoubleMLIIVM
# dml_iv = DoubleMLIIVM(dml_data, ml_g, ml_m, ml_r=ml_m)
```

```python
# ============================================================
# DML — Python（econml 包，更灵活）
# pip install econml
# ============================================================

from econml.dml import LinearDML, CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

Y = df[Y_VAR].values
D = df[D_VAR].values
X = df[X_VARS].values  # 高维控制变量
W = None  # 异质性特征（如需HTE）

# 部分线性 DML
est = LinearDML(
    model_y=GradientBoostingRegressor(),
    model_t=GradientBoostingRegressor(),
    cv=5, random_state=42
)
est.fit(Y, D, X=W, W=X)
print(est.ate_inference().summary_frame())
```

### 1.3 R 代码（DoubleML 包）

```r
# ============================================================
# DML — R（DoubleML 包）
# install.packages("DoubleML")
# install.packages("mlr3"); install.packages("mlr3learners")
# ============================================================

library(DoubleML)
library(mlr3)
library(mlr3learners)
library(data.table)

df <- fread("data.csv")

# 定义变量
y_var   <- "outcome"
d_var   <- "treatment"
x_vars  <- setdiff(names(df), c(y_var, d_var, "entity_id", "year"))

# 创建 DoubleML 数据对象
dml_data <- DoubleMLData$new(
  data    = df,
  y_col   = y_var,
  d_cols  = d_var,
  x_cols  = x_vars
)

# 定义 ML 学习器
learner_g <- lrn("regr.ranger", num.trees = 200, max.depth = 5)  # 随机森林
learner_m <- lrn("regr.ranger", num.trees = 200, max.depth = 5)

# 或用 LASSO
# learner_g <- lrn("regr.cv_glmnet", s = "lambda.min", alpha = 1)
# learner_m <- lrn("regr.cv_glmnet", s = "lambda.min", alpha = 1)

# 拟合 DML 部分线性模型
dml_plr <- DoubleMLPLR$new(
  data      = dml_data,
  ml_l      = learner_g,
  ml_m      = learner_m,
  n_folds   = 5,
  n_rep     = 3
)
dml_plr$fit()
print(dml_plr$summary())
```

---

## 2. Causal Forest（因果森林）

### 2.1 方法背景

**来源：** Athey & Imbens (2019)；Wager & Athey (2018)，*JASA*。
**异质性处理效应综述：** Athey & Imbens (2017)。

**适用场景：**
- 探索**条件平均处理效应（CATE）**：$\tau(x) = \mathbb{E}[Y(1) - Y(0) | X = x]$
- 数据驱动地发现哪些子群体效应更强
- 在 RCT 或可信识别策略下的异质性探索

**核心思想：**
随机森林的每棵树都是一个局部估计量，通过"诚实树"（honest tree）分离样本分割和叶节点估计，获得有效的置信区间。

**与传统异质性分析的区别：**
- 传统：事先定义子组（如大/小企业），进行分样本回归（受多重检验问题影响）
- 因果森林：数据驱动，自动发现最优分割维度，提供均匀有效置信区间

### 2.2 Python 代码（econml）

```python
# ============================================================
# Causal Forest — Python（econml）
# pip install econml
# ============================================================

from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

Y = df["outcome"].values
D = df["treatment"].values
X = df[["age", "size", "region_code"]].values  # 异质性特征（低维，可解释）
W = df[["control1", "control2", "control3"]].values  # 高维控制变量

# 因果森林（内置DML用于nuisance估计）
cf = CausalForestDML(
    model_y=GradientBoostingRegressor(n_estimators=100),
    model_t=GradientBoostingRegressor(n_estimators=100),
    n_estimators=1000,
    min_samples_leaf=5,
    max_features="sqrt",
    cv=5,
    random_state=42
)
cf.fit(Y, D, X=X, W=W)

# 个体处理效应预测
tau_hat = cf.predict(X)
lb, ub = cf.predict_interval(X, alpha=0.05)

# 总结统计
print(f"ATE: {tau_hat.mean():.4f}")
print(f"CATE std: {tau_hat.std():.4f}")

# 特征重要性（哪些变量驱动异质性）
feat_imp = cf.feature_importances_
feature_names = ["age", "size", "region_code"]
for name, imp in sorted(zip(feature_names, feat_imp), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.4f}")

# 绘制 CATE 分布
plt.hist(tau_hat, bins=50, edgecolor='white')
plt.xlabel("Estimated CATE")
plt.ylabel("Count")
plt.title("Distribution of Conditional Average Treatment Effects")
plt.tight_layout()
plt.savefig("output/figures/cate_distribution.png", dpi=300)
plt.close()
```

### 2.3 R 代码（grf 包）

```r
# ============================================================
# Causal Forest — R（grf 包）
# install.packages("grf")
# ============================================================

library(grf)
library(ggplot2)

df <- read.csv("data.csv")

Y  <- as.matrix(df[["outcome"]])
D  <- as.matrix(df[["treatment"]])
X  <- as.matrix(df[, c("age", "size", "region_code")])   # 异质性特征
W  <- as.matrix(df[, c("control1", "control2", "control3")])  # 控制变量

# 合并所有特征（grf 用 X 做所有控制）
X_full <- cbind(X, W)

# 拟合因果森林
cf <- causal_forest(
  X            = X_full,
  Y            = Y,
  W            = D,
  num.trees    = 2000,
  min.node.size = 5,
  seed         = 42
)

# ATE（包含置信区间）
ate <- average_treatment_effect(cf, target.sample = "all")
cat(sprintf("ATE: %.4f (SE: %.4f)\n", ate["estimate"], ate["std.err"]))

# ATT
att <- average_treatment_effect(cf, target.sample = "treated")

# 个体 CATE
tau_hat <- predict(cf, estimate.variance = TRUE)
df$tau  <- tau_hat$predictions
df$tau_se <- sqrt(tau_hat$variance.estimates)

# 特征重要性
vim <- variable_importance(cf)
cat("Variable importance:\n")
print(data.frame(variable = colnames(X_full), importance = vim))

# 最优政策树（可解释的异质性规则）
dr_scores <- get_scores(cf)
policy_tree <- policy_tree(X, dr_scores, depth = 2)
plot(policy_tree, leaf.labels = c("Don't treat", "Treat"))

# 异质性检验（BLP）
blp <- best_linear_projection(cf, A = X)
print(blp)
```

---

## 3. LASSO 高维变量选择

### 3.1 适用场景

- 控制变量候选集远大于样本量（p >> n）
- 需要从大量候选变量中筛选有效控制变量
- Post-LASSO 双步骤估计：先LASSO选变量，再OLS估计系数

### 3.2 Python 代码

```python
# ============================================================
# LASSO 变量选择 + Post-LASSO OLS
# ============================================================

from sklearn.linear_model import LassoCV, Lasso
import statsmodels.api as sm
import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")

Y = df["outcome"].values
D = df["treatment"].values
X = df[[c for c in df.columns if c.startswith("control")]].values

# Step 1: 用 LASSO 对 Y ~ X 选变量
lasso_y = LassoCV(cv=5, random_state=42, max_iter=5000).fit(X, Y)
selected_y = np.where(lasso_y.coef_ != 0)[0]

# Step 2: 用 LASSO 对 D ~ X 选变量
lasso_d = LassoCV(cv=5, random_state=42, max_iter=5000).fit(X, D)
selected_d = np.where(lasso_d.coef_ != 0)[0]

# Step 3: 取并集作为最终控制变量
selected = list(set(selected_y) | set(selected_d))
X_selected = X[:, selected]
print(f"选择了 {len(selected)} 个控制变量（原始 {X.shape[1]} 个）")

# Step 4: Post-LASSO OLS
X_final = np.column_stack([D, X_selected, np.ones(len(Y))])
ols = sm.OLS(Y, X_final).fit(cov_type='HC3')
print(f"Post-LASSO ATE: {ols.params[0]:.4f} (SE: {ols.bse[0]:.4f})")
```

### 3.3 R 代码（glmnet）

```r
# ============================================================
# LASSO — R（glmnet 包）
# ============================================================

library(glmnet)
library(fixest)

df <- read.csv("data.csv")

Y <- df$outcome
D <- df$treatment
X <- as.matrix(df[, grep("^control", names(df))])

# LASSO 选变量（交叉验证）
cv_y <- cv.glmnet(X, Y, alpha = 1, nfolds = 5)
cv_d <- cv.glmnet(X, D, alpha = 1, nfolds = 5)

coef_y <- coef(cv_y, s = "lambda.min")[-1]  # 去掉截距
coef_d <- coef(cv_d, s = "lambda.min")[-1]

selected <- union(which(coef_y != 0), which(coef_d != 0))
cat(sprintf("选择了 %d 个控制变量\n", length(selected)))

# Post-LASSO OLS（用 fixest 含固定效应）
df_sel <- cbind(df[, c("outcome", "treatment", "entity_id", "year")],
                X[, selected])
controls_str <- paste(colnames(X)[selected], collapse = " + ")
fml <- as.formula(paste("outcome ~ treatment +", controls_str, "| entity_id + year"))

post_lasso <- feols(fml, data = df_sel, cluster = ~entity_id)
etable(post_lasso)
```

---

## 4. Matrix Completion（矩阵补全因果推断）

### 4.1 方法背景

**来源：** Athey, Bayati, Doudchenko, Imbens & Khosravi (2021)，*JASA*，116(536)，1716–1730。

**适用场景：**
- 面板数据，部分单元在某时期被处理（类似合成控制）
- 处理组数量较多，传统合成控制不适用
- 希望利用矩阵的低秩结构估计反事实

**核心思想：**
将面板结果矩阵 $\mathbf{M}$ 视为低秩矩阵加噪声，通过核范数正则化将未处理期结果矩阵"补全"到处理组处理后的反事实状态。

### 4.2 Python 代码（MCPanel）

```python
# pip install MCPanel
from mcpanel import MCPanel
import pandas as pd
import numpy as np

df = pd.read_csv("panel_data.csv")  # 宽格式：行=单元，列=时期

# 构建矩阵
# Y_obs: 观测到的结果矩阵（N × T）
# mask: 处理指示矩阵（1=未处理/控制期，0=处理后）
Y_obs = df.pivot(index="entity_id", columns="year", values="outcome").values
treat_matrix = df.pivot(index="entity_id", columns="year", values="treated").values
mask = 1 - treat_matrix  # 1=观测，0=缺失（处理后）

mc = MCPanel(Y=Y_obs, mask=mask)
mc.fit()

# 提取处理效应
tau = mc.tau_hat  # 个体-时期处理效应矩阵
ate = np.nanmean(tau)
print(f"ATT（矩阵补全）: {ate:.4f}")
```

---

## 5. 适用场景汇总

| 方法 | 适用场景 | 不适用场景 | 假设 |
|------|---------|-----------|------|
| DML | 高维控制，估计ATE | 处理变量内生（需IV） | CIA（可忽略性）|
| Causal Forest | 探索CATE异质性 | 替代主识别策略 | CIA |
| LASSO | 从候选集中选控制变量 | 直接估计因果效应 | — |
| Matrix Completion | 面板反事实，多处理组合成控制 | 短面板（T小） | 低秩结构 |

**方法选择建议：**
1. 主识别策略（DID/IV/RDD）先确定，用清晰的外生变异支撑因果
2. 若控制变量维度高（>50个候选），用 LASSO 或 DML 作为控制变量选择步骤
3. 异质性分析完成传统分样本回归后，用因果森林作为探索性补充
4. 报告时清晰区分"主要结果"（传统计量）和"探索性ML分析"

---

## 6. 注意事项与常见误区

### 6.1 ML 因果推断 ≠ 因果识别

ML方法估计的是**给定假设成立时**的因果效应。若 CIA（条件独立假设）不成立，ML方法依然有偏。选择 ML 方法不能替代论证为什么处理变量是外生的。

### 6.2 DML 的假设清单

- **可忽略性（CIA）：** $Y(1), Y(0) \perp D | X$
- **重叠性（Overlap）：** $0 < \Pr(D=1|X) < 1$（倾向得分不为0或1）
- **SUTVA：** 无溢出效应，处理版本唯一
- **ML模型一致性：** 学习器对 $g_0$ 和 $m_0$ 的估计收敛

### 6.3 因果森林的使用规范

- 因果森林适合**探索**异质性，不适合作为检验异质性的主要方法
- 报告异质性时，应先有经济理论支撑预期（why should heterogeneity exist?）
- 不要在论文主体中报告"森林自动发现了N种异质性分组"而没有经济解释

### 6.4 计算资源提示

| 方法 | 样本量 | 估计时间（参考） |
|------|--------|--------------|
| DML/LASSO | 5万以内 | 1–5分钟 |
| Causal Forest（1000棵树） | 10万以内 | 5–30分钟 |
| Matrix Completion | 5000×20面板 | 1–10分钟 |

---

## 7. 关键文献索引

| 文献 | 贡献 |
|------|------|
| Chernozhukov et al. (2018) | DML基础理论，Neyman正交化，交叉拟合 |
| Wager & Athey (2018) | 因果森林方法，置信区间理论 |
| Athey & Imbens (2019) | 机器学习因果推断综述 |
| Belloni, Chernozhukov & Hansen (2014) | Post-LASSO高维变量选择 |
| Athey et al. (2021) | 矩阵补全因果推断 |
| Künzel et al. (2019) | S/T/X-Learner异质性处理效应 |

**完整 BibTeX：**
```bibtex
@article{chernozhukov2018dml,
  author  = {Chernozhukov, Victor and others},
  title   = {Double/Debiased Machine Learning for Treatment and Structural Parameters},
  journal = {The Econometrics Journal},
  year    = {2018}, volume = {21}, number = {1}, pages = {C1--C68}
}
@article{wager2018causalforest,
  author  = {Wager, Stefan and Athey, Susan},
  title   = {Estimation and Inference of Heterogeneous Treatment Effects using Random Forests},
  journal = {Journal of the American Statistical Association},
  year    = {2018}, volume = {113}, number = {523}, pages = {1228--1242}
}
@article{athey2021mc,
  author  = {Athey, Susan and Bayati, Mohsen and Doudchenko, Nikolay
             and Imbens, Guido and Khosravi, Khashayar},
  title   = {Matrix Completion Methods for Causal Panel Data Models},
  journal = {Journal of the American Statistical Association},
  year    = {2021}, volume = {116}, number = {536}, pages = {1716--1730}
}
```
