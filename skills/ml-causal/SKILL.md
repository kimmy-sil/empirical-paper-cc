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

---

## 8. DML 的准确定位：估计工具，非识别策略

> **核心论点："先识别，后估计"**
>
> DML 是在既有识别框架内提升估计效率的工具，不能替代清晰的识别策略。
> 就像 Stata 软件本身不能替代研究设计，DML 也无法凭空解决内生性问题。

### 8.1 DML ≠ 识别策略

以下是**DML 做不到的事**：

- ❌ 不能替代 IV 来解决处理变量的内生性
- ❌ 不能免除 DID 中的平行趋势检验
- ❌ 不能在 RDD 中扩大带宽以"发现"更多数据
- ❌ 不能将相关性转化为因果关系

**DML 能做到的事**：

- ✅ 在 CIA 成立的前提下，用 ML 灵活控制高维、非线性混淆变量
- ✅ 通过 Neyman 正交化消除 ML 估计误差对因果参数的一阶污染
- ✅ 通过交叉拟合避免过拟合偏误，确保统计推断有效
- ✅ 在传统线性控制变量无法充分捕捉混淆结构时提升估计精度

### 8.2 两大理论支柱

#### 支柱一：Neyman 正交性（Neyman Orthogonality）

设因果参数为 $\theta_0$，扰动参数（nuisance parameters）为 $\eta_0$（如 $E[Y|X]$、$E[D|X]$）。

Neyman 正交条件要求得分函数 $\psi(W; \theta, \eta)$ 对扰动参数 $\eta$ 的方向导数为零：

$$\partial_\eta \mathbb{E}[\psi(W; \theta_0, \eta_0)][\eta - \eta_0] = 0$$

**经济学含义：** 即便 ML 估计扰动参数有一定误差（因为正则化、有限样本），这种误差对因果参数 $\theta_0$ 估计的影响是**二阶小量**而非一阶，因此 $\hat{\theta}_0$ 依然一致且渐近正态。

#### 支柱二：交叉拟合（Cross-fitting）

将样本随机分为 $K$ 折（通常 $K = 5$）：

1. 用第 $k$ 折**以外**的 $K-1$ 折数据训练 ML 模型，估计扰动参数 $\hat{\eta}^{(-k)}$
2. 在第 $k$ 折上用 $\hat{\eta}^{(-k)}$ 计算残差和推断因果参数
3. 循环所有 $K$ 折，聚合估计量

**经济学含义：** 训练扰动参数和推断因果参数使用不同子样本，切断"数据泄露"通道，防止 ML 模型过拟合导致的系统性偏误（类似样本外预测 vs 样本内拟合的区别）。

```
直觉类比：
折1 → 训练 ML → 残差估计 θ (在折2~5上)
折2 → 训练 ML → 残差估计 θ (在折1,3~5上)
...
最终 θ = 各折估计的加权平均
```

---

## 9. PLM vs IRM：两大基本模型

DML 框架下有两种核心模型结构，选择依据处理效应的同质性假设和样本特征。

### 9.1 对比表格

| 维度 | PLM（部分线性模型） | IRM（交互回归模型） |
|------|-------------------|-------------------|
| 模型结构 | $Y = D\theta + g_0(X) + \varepsilon$ | $Y = g_0(D, X) + \varepsilon$，不假设线性 |
| 适用效应类型 | **同质性**处理效应（$\theta$ 对所有个体相同） | **异质性**处理效应（CATE 随 X 变化） |
| 估计目标 | ATE 或 ATO（Overlap Weighting） | ATE、ATT、CATE（通过 DR-learner） |
| 扰动参数 | $\ell_0(X) = E[Y\|X]$，$m_0(X) = E[D\|X]$ | $\mu_0(1,X) = E[Y\|D=1,X]$，$\mu_0(0,X) = E[Y\|D=0,X]$，$e_0(X) = P(D=1\|X)$ |
| 共同支撑不足时 | 相对稳健（残差回归对重叠要求低） | RMSE 膨胀、倾向得分极端值放大偏误 |
| 样本量要求 | 较小（n ≥ 500 可接受） | 较大（建议 n ≥ 1000） |
| 何时选择 | 样本量小 / 共同支撑差 / 先验认为效应同质 | 样本量大 + 共同支撑充分 + 探索异质性 |

### 9.2 Python 代码对比（doubleml 包）

```python
# ============================================================
# PLM vs IRM — Python（doubleml 包）
# pip install doubleml scikit-learn
# ============================================================

import numpy as np
import pandas as pd
from doubleml import DoubleMLPLR, DoubleMLIRM, DoubleMLData
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegressionCV

df = pd.read_csv("data.csv")
Y_VAR = "outcome"
D_VAR = "treatment"   # 二元处理变量
X_VARS = [c for c in df.columns if c not in [Y_VAR, D_VAR, "unit_id", "year"]]

dml_data = DoubleMLData(df, y_col=Y_VAR, d_cols=D_VAR, x_cols=X_VARS)

# ── 方案 A：部分线性模型（PLM）──────────────────────────────
# 适合效应同质 / 样本量较小 / 共同支撑不充分
ml_l = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)  # 预测 Y
ml_m = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)  # 预测 D（连续）

dml_plm = DoubleMLPLR(
    obj_dml_data = dml_data,
    ml_l = ml_l,
    ml_m = ml_m,
    n_folds = 5,
    n_rep   = 3,
    score   = 'partialling out'
)
dml_plm.fit()
print("=== PLM 结果 ===")
print(dml_plm.summary)

# ── 方案 B：交互回归模型（IRM）──────────────────────────────
# 适合效应异质 / 样本量大 / 共同支撑充分
# IRM 需要：预测 Y(D=0|X)、Y(D=1|X)（连续→回归器），以及 P(D=1|X)（分类→分类器）
ml_g = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)   # 预测 E[Y|D,X]
ml_r = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)  # 预测 P(D=1|X)

dml_irm = DoubleMLIRM(
    obj_dml_data = dml_data,
    ml_g = ml_g,
    ml_m = ml_r,
    n_folds = 5,
    n_rep   = 3,
    score   = 'ATE',   # 或 'ATT'
    trimming_threshold = 0.01  # 修剪极端倾向得分
)
dml_irm.fit()
print("=== IRM 结果 ===")
print(dml_irm.summary)

# ── 诊断：比较两种模型的一致性 ──────────────────────────────
plm_coef = dml_plm.coef[0]
irm_coef = dml_irm.coef[0]
print(f"\nPLM ATE: {plm_coef:.4f}")
print(f"IRM ATE: {irm_coef:.4f}")
print(f"差异: {abs(plm_coef - irm_coef):.4f}（若差异大，说明存在显著效应异质性）")
```

### 9.3 倾向得分诊断（IRM 专项）

```python
# ── 诊断倾向得分分布（IRM 必做）──────────────────────────────
import matplotlib.pyplot as plt

# 提取各折倾向得分预测值
ps_scores = dml_irm.predictions['ml_m'].flatten()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 分布图
axes[0].hist(ps_scores[df[D_VAR]==1], bins=40, alpha=0.6, label='Treated', density=True)
axes[0].hist(ps_scores[df[D_VAR]==0], bins=40, alpha=0.6, label='Control', density=True)
axes[0].axvline(0.01, color='red', linestyle='--', label='Trim threshold')
axes[0].axvline(0.99, color='red', linestyle='--')
axes[0].set_xlabel('Propensity Score')
axes[0].set_title('Overlap Check')
axes[0].legend()

# 极端值统计
n_trim = ((ps_scores < 0.01) | (ps_scores > 0.99)).sum()
axes[1].text(0.1, 0.5,
    f"样本量: {len(ps_scores)}\n"
    f"被修剪样本: {n_trim} ({100*n_trim/len(ps_scores):.1f}%)\n"
    f"PS 均值 (treated): {ps_scores[df[D_VAR]==1].mean():.3f}\n"
    f"PS 均值 (control): {ps_scores[df[D_VAR]==0].mean():.3f}",
    transform=axes[1].transAxes, fontsize=12,
    verticalalignment='center')
axes[1].axis('off')
axes[1].set_title('Overlap Statistics')

plt.tight_layout()
plt.savefig("output/figures/dml_overlap_check.png", dpi=300)
plt.close()
print(f"重叠性检验图已保存。被修剪样本比例: {100*n_trim/len(ps_scores):.1f}%")
# 经验准则：被修剪比例 > 10% → 考虑改用 PLM 或限定估计对象为 ATT
```

---

## 10. DML 与经典识别框架的嵌套

DML 本身不提供识别，而是在各类识别框架下充当高效估计工具。

### 10.1 嵌套关系总览

| 识别框架 | DML 的角色 | 关键扰动参数（需 ML 估计） | 对应 DoubleML 类 |
|----------|-----------|--------------------------|----------------|
| **CIA / 可忽略性** | 主估计框架，高维控制替代 OLS | $\ell_0(X) = E[Y\|X]$，$m_0(X) = E[D\|X]$ | `DoubleMLPLR` / `DoubleMLIRM` |
| **IV（工具变量）** | 估计 LATE，同时处理高维控制 | $E[Y\|X]$，$E[D\|X]$，$E[Z\|X]$ | `DoubleMLIIVM` / `DoubleMLPLIV` |
| **DID** | 估计 ATT，放松平行趋势对控制的线性限制 | 倾向得分 $P(D=1\|X)$，控制组结果均值 $E[Y(0)\|X]$ | `DoubleMLDID`（需手动构造） |
| **RDD** | 在断点邻域内灵活控制协变量（非线性） | 协变量条件期望函数 $E[X\|R]$ | 手动构造 + 局部多项式 |
| **Staggered DID** | 结合 CS(2021)，处理异质性处理时间 | 各队列-时期的 IPW 权重、结果均值 | `did` + `DoubleML` 嵌套 |

### 10.2 DML-IV 代码（DoubleMLIIVM）

```python
# ============================================================
# DML + IV — Python（DoubleMLIIVM，二元处理 + 二元工具变量）
# 适用场景：工具变量估计 LATE，同时控制高维协变量
# ============================================================

from doubleml import DoubleMLIIVM, DoubleMLData
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# 数据需要包含工具变量 Z（二元）
dml_data_iv = DoubleMLData(
    df, y_col="outcome", d_cols="treatment",
    z_cols="instrument",   # 工具变量
    x_cols=X_VARS
)

ml_g = RandomForestRegressor(n_estimators=200, random_state=42)   # E[Y|D,X]
ml_m = RandomForestClassifier(n_estimators=200, random_state=42)  # P(D=1|X)
ml_r = RandomForestClassifier(n_estimators=200, random_state=42)  # P(Z=1|X)（first stage）

dml_iv = DoubleMLIIVM(
    obj_dml_data = dml_data_iv,
    ml_g = ml_g,
    ml_m = ml_m,
    ml_r = ml_r,
    n_folds = 5,
    n_rep   = 3,
    score   = 'LATE'
)
dml_iv.fit()
print("DML-IV LATE 估计:")
print(dml_iv.summary)
```

### 10.3 DML-DID 结合（条件 ATT 估计）

```python
# ============================================================
# DML + DID — 在 DID 框架内用 ML 估计倾向得分和结果函数
# 放松平行趋势对控制变量的线性限制
# 参考：Sant'Anna & Zhao (2020) Doubly Robust DID
# ============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_predict

# 假设双期数据：pre（t=0）和 post（t=1）
# 仅使用 pre 期数据估计 nuisance 参数
pre = df[df['time'] == 0].copy()
post = df[df['time'] == 1].copy()

X = pre[X_VARS].values
D = pre['treated'].values

# 步骤1：用 ML 估计倾向得分 P(D=1|X)
ps_model = RandomForestClassifier(n_estimators=200, random_state=42)
ps_hat = cross_val_predict(ps_model, X, D, cv=5, method='predict_proba')[:, 1]
ps_hat = np.clip(ps_hat, 0.01, 0.99)  # 修剪极端值

# 步骤2：用 ML 估计控制组结果均值 E[Y_post | D=0, X]
control_post = post[post['treated'] == 0]
outcome_model = RandomForestRegressor(n_estimators=200, random_state=42)
mu0_hat = cross_val_predict(outcome_model,
    control_post[X_VARS].values,
    control_post['outcome'].values, cv=5)

# 步骤3：DR-DID 估计量（Sant'Anna & Zhao 2020）
# ATT_DR = E[ (D/P(D=1)) * (ΔY - μ_ΔY(0|X)) ]
# 此处简化版本
delta_y = post['outcome'].values - pre['outcome'].values
w_att = D / ps_hat - (1 - D) * ps_hat / (1 - ps_hat)  # IPW 权重

# 直接实现可用 R 的 drdid 包（更完整）
print("建议使用 R 包 drdid 实现完整 DR-DID:")
print("  library(drdid)")
print("  drdid(yname='outcome', tname='time', idname='unit_id',")
print("        dname='treated', xformla=~X1+X2, data=df)")
```

### 10.4 R 代码：drdid（Doubly Robust DID）

```r
# ============================================================
# Doubly Robust DID (Sant'Anna & Zhao 2020)
# install.packages("drdid")
# ============================================================

library(drdid)

# 两期面板：time = 0（pre）或 1（post）
res_dr_did <- drdid(
  yname   = "outcome",
  tname   = "time",
  idname  = "unit_id",
  dname   = "treated",
  xformla = ~ control1 + control2 + control3,  # 控制变量
  data    = df,
  esttype = "panel"   # 面板数据（"repeated cross-section" 用 "rc"）
)
summary(res_dr_did)
# 输出 ATT、SE、95% CI
# DR-DID 同时对倾向得分模型和结果模型都"双重稳健"
```

---

## 11. DML 实践指南（六条建议）

参考胡诗蕴等 (2026) 的系统性建议，以下六条为实操核心：

### 建议一：识别优先，估计其次

在写 DML 代码之前，必须先回答：

> "为什么 $D$ 是外生的？外生变异来自哪里？"

若无法清晰回答，DML 估计结果不具有因果解释。先完成识别策略论证（IV 来源、随机化设计、自然实验），再引入 DML 提升估计效率。

### 建议二：依据样本特征选择 PLM 或 IRM

```
决策树：
n < 500 ──────────────────────────→ 用 PLM（或传统 OLS+LASSO 控制）
n ≥ 500 AND 共同支撑差 ──────────→ 用 PLM（IRM 倾向得分不稳定）
n ≥ 1000 AND 共同支撑充分 ────────→ 可用 IRM 探索异质性
先验认为效应同质 ─────────────────→ PLM（效率更高）
明确需要 ATT / 异质性 ────────────→ IRM
```

### 建议三：ML 算法选择指南

| 场景 | 推荐算法 | 理由 |
|------|---------|------|
| 稀疏高维（p >> n，少数变量真正重要） | **LASSO** / Elastic Net | 稀疏性先验，系数可解释 |
| 非线性混淆、中等维度（p < 50） | **随机森林** / XGBoost | 捕捉非线性交互，无需调参 |
| 超高维+非线性（p > 100） | **梯度提升树**（LightGBM）| 精度高，计算效率好 |
| 特征有自然层次结构 | **深度网络**（PyTorch MLP） | 可学习特征层次 |
| 稳健性检验 | **集成多个算法** | 验证结果对 ML 选择不敏感 |

```python
# ── 集成多算法进行稳健性检验 ───────────────────────────────
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

models = {
    'LASSO':  (LassoCV(cv=5), LassoCV(cv=5)),
    'RF':     (RandomForestRegressor(200, random_state=42),
               RandomForestRegressor(200, random_state=42)),
    'GBM':    (GradientBoostingRegressor(100, random_state=42),
               GradientBoostingRegressor(100, random_state=42)),
}

results = {}
for name, (ml_l, ml_m) in models.items():
    m = DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m, n_folds=5, n_rep=3)
    m.fit()
    results[name] = {'coef': m.coef[0], 'se': m.se[0]}
    print(f"{name:8s}: β={results[name]['coef']:.4f}  SE={results[name]['se']:.4f}")

# 若三种方法系数差异 < 0.01，结果对 ML 选择稳健
```

### 建议四：诊断扰动参数

```python
# ── 诊断1：扰动参数 RMSE（模型拟合质量）─────────────────────
dml_plm.fit()
rmse_l = np.sqrt(np.mean((dml_plm.residuals['ml_l']**2)))  # E[Y|X] 残差
rmse_m = np.sqrt(np.mean((dml_plm.residuals['ml_m']**2)))  # E[D|X] 残差
print(f"RMSE(Y|X): {rmse_l:.4f}  —  应低于 Y 的标准差 {np.std(df[Y_VAR]):.4f}")
print(f"RMSE(D|X): {rmse_m:.4f}  —  应低于 D 的标准差 {np.std(df[D_VAR]):.4f}")

# ── 诊断2：倾向得分（IRM 专项）──────────────────────────────
# 见 9.3 节代码

# ── 诊断3：Frisch-Waugh 残差图 ──────────────────────────────
import matplotlib.pyplot as plt
res_y = dml_plm.residuals['ml_l'].flatten()
res_d = dml_plm.residuals['ml_m'].flatten()
plt.scatter(res_d, res_y, alpha=0.3, s=5)
plt.xlabel('D residual (D - E[D|X])')
plt.ylabel('Y residual (Y - E[Y|X])')
plt.title('Frisch-Waugh-Lovell: DML Orthogonalized Regression')
# 斜率即为 θ 的 OLS 估计；散点若呈现系统非线性，说明 PLM 可能误设
plt.savefig("output/figures/dml_fwl_plot.png", dpi=300)
```

### 建议五：规范性检验

```python
# ── 检验1：协变量平衡（残差层面）──────────────────────────────
# 若 DML 充分控制了混淆，D 的残差应与 X 的每个维度不相关
from scipy import stats

res_d_vec = dml_plm.residuals['ml_m'].flatten()
balance_results = []
for col in X_VARS:
    corr, pval = stats.pearsonr(res_d_vec, df[col].values)
    balance_results.append({'variable': col, 'correlation': corr, 'p_value': pval})

balance_df = pd.DataFrame(balance_results)
n_sig = (balance_df['p_value'] < 0.05).sum()
print(f"残差-协变量相关性显著的变量数: {n_sig}/{len(X_VARS)}")
# 若 n_sig/len(X_VARS) > 0.1，说明 ML 模型未充分拟合，考虑更灵活的模型

# ── 检验2：多模型稳健性 ─────────────────────────────────────
# 见建议三代码

# ── 检验3：n_rep 稳定性 ─────────────────────────────────────
# 增加 n_rep（重复次数）观察系数是否收敛
for n_rep in [1, 3, 5]:
    m = DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m, n_folds=5, n_rep=n_rep)
    m.fit()
    print(f"n_rep={n_rep}: β={m.coef[0]:.4f} ± {m.se[0]:.4f}")
```

### 建议六：报告透明度

论文中汇报 DML 结果时，**必须在正文或附录中包含以下信息**：

1. **ML 模型选择依据**：为何选择该算法（稀疏性先验 / 非线性考虑 / 计算约束）
2. **超参数设置**：树的数量、深度、交叉验证折数
3. **扰动参数 RMSE**：展示 ML 模型对 $E[Y|X]$ 和 $E[D|X]$ 的样本外拟合质量
4. **交叉拟合设置**：$K$ 折数、重复次数 $n\_rep$
5. **多算法稳健性**：至少报告 2 种 ML 算法下结果的一致性
6. **倾向得分（IRM）**：重叠性诊断图，修剪比例

**报告模板（论文脚注或附表）：**

```
DML 估计采用部分线性模型（PLM），扰动参数由随机森林估计
（500棵树，最大深度5）。交叉拟合设置：K=5折，重复3次，取中位数。
扰动参数样本外RMSE：E[Y|X]为0.xxx，E[D|X]为0.xxx。
采用LASSO和梯度提升树进行稳健性检验，系数差异均小于0.01。
```

---

## 12. 何时不该用 DML

并非所有情形都需要 DML，以下场景应优先选用传统方法：

### 12.1 不适用场景

**场景一：混淆关系线性 + 效应同质**

```
诊断标准：
- 控制变量 < 20 个
- 理论上控制变量与处理/结果的关系为线性
- 无证据显示效应存在异质性

处理建议：直接用 OLS + 聚类标准误，无需 DML
代价：DML 在此情形下估计结果与 OLS 基本无差异，
      但增加了不必要的模型选择不确定性
```

**场景二：样本量太小（n < 500）**

```
问题：ML 模型在小样本下过拟合风险极高
     交叉拟合虽能缓解，但学习器本身估计不稳定
     导致扰动参数 RMSE 大，因果参数方差膨胀

处理建议：
- n < 200：回归 OLS，手动选择控制变量
- 200 ≤ n < 500：LASSO（比非参数 ML 更稳健）
- n ≥ 500：可谨慎使用 DML（推荐 LASSO 或浅层树）
```

**场景三：识别策略本身不可信**

```
症状：
- 工具变量的外生性存疑（弱工具、相关性不清晰）
- DID 的平行趋势假设有明显违反
- 没有自然实验，纯粹观察数据

后果：DML 不能"拯救"识别策略
      高效地估计一个有偏参数，偏误依然存在
      精确的错误答案比模糊的错误答案更危险

处理建议：先解决识别问题，DML 只是锦上添花
```

**场景四：结果需要直接可解释性**

```
情形：政策报告需要每个控制变量的系数
     监管合规要求模型可解释

处理建议：用 Post-LASSO OLS（见第3节），
          先 LASSO 选变量，再 OLS 估计所有系数
```

### 12.2 方法选择速查

```
Q1: 处理变量是外生的吗（有识别策略支撑）？
    ├─ 否 → 先解决内生性（IV / DID / RDD），DML 无法帮忙
    └─ 是 → Q2

Q2: 控制变量维度高（>20个候选）或存在非线性混淆？
    ├─ 否 → 传统 OLS/TWFE + 手动选择控制变量
    └─ 是 → Q3

Q3: 样本量是否足够（n ≥ 500）？
    ├─ 否（n < 500）→ LASSO 控制变量选择 + Post-LASSO OLS
    └─ 是 → Q4

Q4: 是否关心效应异质性且共同支撑充分？
    ├─ 否 → DML-PLM（估计 ATE/ATT）
    └─ 是 → DML-IRM 或 因果森林（探索 CATE）
```

---

## 13. 关键文献补充（新增）

| 文献 | 贡献 |
|------|------|
| Chernozhukov et al. (2018) | DML 基础理论：Neyman 正交化、交叉拟合 |
| Sant'Anna & Zhao (2020) | Doubly Robust DID，结合倾向得分和结果回归 |
| Callaway & Sant'Anna (2021) | 交错 DID 的 DR 估计，ATT(g,t) |
| Roth et al. (2023) | DID 文献综述，DML 在 DID 中的角色 |
| Newey & Robins (2018) | Cross-fitting 理论基础 |
|胡诗蕴等 (2026) | DML 在经济学中的实践指南（中文） |

```bibtex
@article{santanna2020drdid,
  author  = {Sant'Anna, Pedro H.C. and Zhao, Jun},
  title   = {Doubly Robust Difference-in-Differences Estimators},
  journal = {Journal of Econometrics},
  year    = {2020}, volume = {219}, number = {1}, pages = {101--122}
}
@article{newey2018crossfitting,
  author  = {Newey, Whitney K. and Robins, James R.},
  title   = {Cross-fitting and Fast Remainder Rates for Semiparametric Estimation},
  journal = {arXiv preprint arXiv:1801.09138},
  year    = {2018}
}
```

---

### SHAP 可解释性

SHAP（SHapley Additive exPlanations）提供对 ML 模型预测的局部和全局解释，适用于 DML 的扰动参数估计和 Causal Forest 的异质性分析。

**工具组合：**
- `shap.TreeExplainer`：针对树模型（RF / XGBoost / LightGBM）的高效 SHAP 计算
- `summary_plot`：全局特征重要性（蜂窝图）
- PDP（Partial Dependence Plot）：特征的边际效应
- ICE（Individual Conditional Expectation）：个体级别的效应曲线

```python
# ============================================================
# SHAP 可解释性 — 完整可运行 Python 代码
# pip install shap scikit-learn matplotlib pandas
# ============================================================
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split

# ---- 示例数据 ----
np.random.seed(42)
n = 2000
df = pd.DataFrame({
    'age':      np.random.normal(40, 10, n),
    'size':     np.random.lognormal(3, 1, n),
    'leverage': np.random.uniform(0, 1, n),
    'region':   np.random.randint(0, 5, n).astype(float),
    'control1': np.random.normal(0, 1, n),
    'control2': np.random.normal(0, 1, n),
})
df['outcome'] = (0.3 * df['age'] + 0.5 * df['size'] -
                 0.2 * df['leverage'] + np.random.normal(0, 1, n))

feature_cols = ['age', 'size', 'leverage', 'region', 'control1', 'control2']
X = df[feature_cols].values
y = df['outcome'].values

# ---- 训练模型（DML 中的扰动参数估计器，或独立预测模型）----
model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
print(f"Model R² (test): {model.score(X_test, y_test):.4f}")

# ---- SHAP TreeExplainer ----
explainer   = shap.TreeExplainer(model)
shap_values = explainer(X_test)  # 返回 Explanation 对象（含 shap_values 和 base_values）

# ---- 图1：Summary Plot（全局特征重要性，蜂窝图）----
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test,
                  feature_names=feature_cols,
                  show=False)
plt.title("SHAP Summary Plot: Global Feature Importance")
plt.tight_layout()
plt.savefig("output/shap_summary_plot.png", dpi=150, bbox_inches='tight')
plt.close()
print("SHAP summary plot saved.")

# ---- 图2：Bar Plot（平均 |SHAP| 排序）----
plt.figure(figsize=(8, 5))
shap.plots.bar(shap_values, max_display=10, show=False)
plt.title("SHAP Feature Importance (Mean |SHAP value|)")
plt.tight_layout()
plt.savefig("output/shap_bar_plot.png", dpi=150, bbox_inches='tight')
plt.close()

# ---- 图3：Waterfall Plot（单个观测的解释）----
plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values[0], show=False)  # 第一个测试样本
plt.title("SHAP Waterfall: Single Observation Explanation")
plt.tight_layout()
plt.savefig("output/shap_waterfall_plot.png", dpi=150, bbox_inches='tight')
plt.close()

# ---- 图4：PDP（Partial Dependence Plot）---- 
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
features_to_plot = [0, 1, 2]  # age, size, leverage 的列索引

for ax, feat_idx in zip(axes, features_to_plot):
    disp = PartialDependenceDisplay.from_estimator(
        model, X_test, features=[feat_idx],
        feature_names=feature_cols,
        ax=ax, grid_resolution=50
    )
    ax.set_title(f"PDP: {feature_cols[feat_idx]}")
    ax.set_xlabel(feature_cols[feat_idx])

plt.suptitle("Partial Dependence Plots (PDP)", fontsize=14)
plt.tight_layout()
plt.savefig("output/shap_pdp_plots.png", dpi=150, bbox_inches='tight')
plt.close()

# ---- 图5：ICE（Individual Conditional Expectation）----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, feat_idx in zip(axes, [0, 1]):  # age, size
    disp = PartialDependenceDisplay.from_estimator(
        model, X_test[:200], features=[feat_idx],  # 前200个样本（速度考虑）
        feature_names=feature_cols,
        kind='both',   # 'both' = ICE + PDP
        ax=ax, grid_resolution=30,
        subsample=100  # 展示100条ICE线
    )
    ax.set_title(f"ICE + PDP: {feature_cols[feat_idx]}")
    ax.set_alpha(0.1)

plt.suptitle("Individual Conditional Expectation (ICE) Plots", fontsize=14)
plt.tight_layout()
plt.savefig("output/shap_ice_plots.png", dpi=150, bbox_inches='tight')
plt.close()

# ---- SHAP Dependence Plot（特征 × 交互项）----
plt.figure(figsize=(8, 6))
shap.dependence_plot(
    "age", shap_values.values, X_test,
    feature_names=feature_cols,
    interaction_index="size",  # 颜色表示 size 的值（交互效应）
    show=False
)
plt.title("SHAP Dependence Plot: age (colored by size)")
plt.tight_layout()
plt.savefig("output/shap_dependence_plot.png", dpi=150, bbox_inches='tight')
plt.close()

print("所有 SHAP 图已保存至 output/ 目录")
print(f"SHAP 值矩阵形状: {shap_values.values.shape}")
print(f"平均 |SHAP| 排名:")
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
for name, val in sorted(zip(feature_cols, mean_abs_shap), key=lambda x: -x[1]):
    print(f"  {name}: {val:.4f}")
```

---

### 算法选择决策树加强

在 DML 和因果森林框架内选择 ML 算法时，需综合考虑数据特点、计算资源和可解释性需求：

| 数据特点 | 推荐算法 | 理由 | 典型超参数 |
|---------|---------|------|----------|
| 特征维度 >> 样本量（稀疏高维） | **LASSO** / Elastic Net | 稀疏性先验，收缩冗余特征，系数直接可读 | `alpha=1.0`，`cv=5` |
| 非线性强 + 样本大（n > 2000） | **Random Forest / XGBoost** | 自动捕捉非线性和交互，无需人工构造特征 | 200-500棵树，`max_depth=5` |
| 一般情况（不确定哪种好） | **交叉验证选超参** + 集成多算法 | 让数据说话，报告稳健性 | 见下方代码 |
| 需要可解释性 | **LASSO > RF > NN** | LASSO系数直读；RF用SHAP解释；NN黑箱 | — |
| 超高维 + 非线性（p > 100，大样本） | **LightGBM** | 计算效率最优，速度比XGBoost快3~5倍 | `num_leaves=31`，`min_data_in_leaf=20` |
| 处理变量为连续型 | **RF/XGBoost**（回归器） | 连续倾向得分需要回归，非分类 | 同上 |
| 处理变量为二元 | 倾向得分用**RF分类器** | 输出概率更稳定，trim极端值 | `n_estimators=200` |

```python
# 交叉验证自动选择最优算法（DML 稳健性检验标准流程）
from doubleml import DoubleMLPLR, DoubleMLData
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
import pandas as pd

# 候选算法配置
candidate_models = {
    'LASSO': {
        'ml_l': LassoCV(cv=5, random_state=42),
        'ml_m': LassoCV(cv=5, random_state=42)
    },
    'RandomForest': {
        'ml_l': RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42),
        'ml_m': RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42)
    },
    'XGBoost': {
        'ml_l': GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
        'ml_m': GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    },
}

results_table = []
for name, config in candidate_models.items():
    dml = DoubleMLPLR(
        obj_dml_data=dml_data,
        ml_l=config['ml_l'],
        ml_m=config['ml_m'],
        n_folds=5, n_rep=3
    )
    dml.fit()
    results_table.append({
        'algorithm': name,
        'coef': dml.coef[0],
        'se':   dml.se[0],
        'ci_lo': dml.coef[0] - 1.96 * dml.se[0],
        'ci_hi': dml.coef[0] + 1.96 * dml.se[0]
    })
    print(f"{name:15s}: β={dml.coef[0]:.4f}  SE={dml.se[0]:.4f}")

results_df = pd.DataFrame(results_table)
# 若所有算法的系数差异 < 0.01，结论对 ML 选择稳健
max_diff = results_df['coef'].max() - results_df['coef'].min()
print(f"\n算法间最大系数差异: {max_diff:.4f}（< 0.01 表示稳健）")
```

---

### GRF 异质性探索加强

grf 包的 `best_linear_projection` 和 `variable_importance` 提供系统性的异质性驱动因素分析。

**`best_linear_projection`（BLP）：**
- 对 CATE 做线性投影：$\tau(x) \approx \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ...$
- $\beta_j$ 显著 → 变量 $x_j$ 驱动异质性
- 基于半参数效率界限，提供有效推断

**`variable_importance`：**
- 基于分裂频率和深度的加权变量重要性
- 不提供统计检验，仅用于**探索性排序**
- 结合 BLP 进行解读

```r
# R: GRF 异质性探索加强（grf 包完整示例）
library(grf)
library(ggplot2)
library(dplyr)

# ---- 数据准备 ----
df <- read.csv("data.csv")
Y  <- as.matrix(df$outcome)
D  <- as.matrix(df$treatment)

# 异质性特征（低维，可解释）
X_hetero <- as.matrix(df[, c("age", "size", "leverage", "region")])

# 控制变量（高维）
X_controls <- as.matrix(df[, grep("^control", names(df))])
X_full     <- cbind(X_hetero, X_controls)

# ---- 训练因果森林 ----
cf <- causal_forest(
  X            = X_full,
  Y            = Y,
  W            = D,
  num.trees    = 4000,
  min.node.size = 5,
  tune.parameters = "all",  # 自动调参（min.node.size, sample.fraction 等）
  seed         = 42
)

cat(sprintf("ATE: %.4f (SE: %.4f)\n",
            average_treatment_effect(cf)[1],
            average_treatment_effect(cf)[2]))

# ---- best_linear_projection：检验哪些变量驱动异质性 ----
# 对 CATE 关于异质性特征做线性投影
blp <- best_linear_projection(
  forest = cf,
  A      = X_hetero,  # 只用低维解释性变量做投影
  target.sample = "all"
)
cat("\n=== BLP：异质性线性投影 ===\n")
print(blp)
# 解读：
#   - 截距 = ATE
#   - A.age 显著 → 年龄驱动异质性（年龄大效应更强/弱）
#   - 系数正负方向 = CATE 随该变量变化的方向

# ---- variable_importance：变量重要性排名 ----
vim <- variable_importance(cf)
vim_df <- data.frame(
  variable   = colnames(X_full),
  importance = as.vector(vim)
) %>% arrange(desc(importance))

cat("\n=== 变量重要性（基于分裂频率）===\n")
print(vim_df)

# 可视化
ggplot(vim_df %>% head(10), aes(x = reorder(variable, importance), y = importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Causal Forest: Variable Importance (Top 10)",
       x = "Variable", y = "Importance") +
  theme_minimal()
ggsave("output/grf_variable_importance.png", dpi = 150)

# ---- CATE 分布与异质性检验 ----
tau_hat <- predict(cf, estimate.variance = TRUE)
df$cate    <- tau_hat$predictions
df$cate_se <- sqrt(tau_hat$variance.estimates)

# 异质性检验：CATE 方差是否显著大于零
# 用 BLP 的联合 F 检验
cat("\n=== 异质性显著性检验 ===\n")
cat("BLP 中 A 变量的联合显著性（F 检验 p 值）:\n")
# 提取 BLP 统计量
blp_summary <- summary(blp)
print(blp_summary)

# ---- CATE 按关键变量分组 ----
# 高/低分位数 CATE 对比
tau_q75 <- quantile(df$cate, 0.75)
tau_q25 <- quantile(df$cate, 0.25)

cat(sprintf("\nCATE 分布: 均值=%.4f, 中位数=%.4f, SD=%.4f\n",
            mean(df$cate), median(df$cate), sd(df$cate)))
cat(sprintf("高效应组（>75%%分位）均值: %.4f\n", mean(df$cate[df$cate > tau_q75])))
cat(sprintf("低效应组（<25%%分位）均值: %.4f\n", mean(df$cate[df$cate < tau_q25])))

# ---- 高/低效应组特征比较（辅助解释）----
df$high_cate <- df$cate > tau_q75
cate_profile <- df %>%
  group_by(high_cate) %>%
  summarise(across(c("age", "size", "leverage"), mean, .names = "mean_{.col}"),
            n = n())
cat("\n=== 高效应组 vs 低效应组特征比较 ===\n")
print(cate_profile)

# ---- 输出 CATE 分布图 ----
ggplot(df, aes(x = cate)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white", alpha = 0.8) +
  geom_vline(xintercept = mean(df$cate), color = "red",
             linetype = "dashed", linewidth = 1) +
  labs(title = "CATE Distribution (Causal Forest)",
       subtitle = sprintf("Mean=%.4f, SD=%.4f", mean(df$cate), sd(df$cate)),
       x = "Estimated CATE", y = "Count") +
  theme_minimal()
ggsave("output/cate_distribution.png", dpi = 150)
```

---

### DynamicDML

`DynamicDML`（`econml.panel.dml.DynamicDML`）用于**平衡面板 + 多期处理 + 存在状态依赖**的情形，扩展标准 DML 到动态因果框架。

**马尔可夫因果图说明：**
```
T=0: X₀ → D₀ → Y₀
                ↓
T=1: X₁ → D₁ → Y₁
         ↑
        Y₀, D₀  ← 状态依赖
```
- 当期处理 $D_t$ 受历史结果 $Y_{t-1}$ 和历史处理 $D_{t-1}$ 影响（状态依赖）
- DynamicDML 通过马尔可夫假设，在每期残差化后逐期建立正交条件

**适用条件：**
- 平衡面板（所有个体观测完整的 T 期）
- 多期处理（非单一政策，处理状态随时间变化）
- 存在状态依赖（当期处理/结果受历史影响）
- CIA 假设在每期条件上成立（无未观测时变混淆）

```python
# ============================================================
# DynamicDML — Python（econml 完整代码）
# pip install econml scikit-learn
# ============================================================
import numpy as np
import pandas as pd
from econml.panel.dml import DynamicDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LassoCV

# ---- 数据准备（平衡面板，长格式）----
# 必须字段：unit_id, year, outcome Y, treatment D, controls X
np.random.seed(42)
n_units, n_periods = 500, 4
panel = pd.DataFrame({
    'unit_id': np.repeat(np.arange(n_units), n_periods),
    'year':    np.tile(np.arange(n_periods), n_units),
})
panel['X1'] = np.random.normal(0, 1, len(panel))
panel['X2'] = np.random.normal(0, 1, len(panel))
panel['D']  = (0.5 * panel['X1'] + np.random.normal(0, 1, len(panel)) > 0).astype(float)
panel['Y']  = (0.3 * panel['D'] + 0.2 * panel['X1'] +
               np.random.normal(0, 0.5, len(panel)))

# ---- 转换为 DynamicDML 所需格式 ----
# DynamicDML 需要：
#   Y: (n_units, n_periods) 矩阵 或 展开后的 (n_units * n_periods,) 向量
#   D: (n_units, n_periods) 矩阵
#   X: (n_units, n_X_features) 时不变特征（或扩展为面板）
#   groups: 每个观测所属的个体 ID

Y_arr = panel.pivot(index='unit_id', columns='year', values='Y').values
D_arr = panel.pivot(index='unit_id', columns='year', values='D').values
X_arr = panel.groupby('unit_id')[['X1', 'X2']].first().values  # 时不变特征取首期

# ---- 拟合 DynamicDML ----
dynamic_dml = DynamicDML(
    model_y   = GradientBoostingRegressor(n_estimators=100, random_state=42),
    model_t   = GradientBoostingClassifier(n_estimators=100, random_state=42),
    cv        = 3,
    mc_iters  = 3,   # Monte Carlo 交叉拟合迭代次数
    mc_agg    = 'median'
)

# fit 接口：Y(n×T), T(n×T), X(n×n_x), groups=None（单层面板）
dynamic_dml.fit(Y_arr, D_arr, X=X_arr)

# ---- 提取处理效应 ----
# effect(X, T0, T1)：从处理水平 T0 变为 T1 时的效应
# 对所有个体估计 D: 0→1 的效应
effect = dynamic_dml.effect(X_arr, T0=0, T1=1)
print(f"DynamicDML ATE (D: 0→1): {effect.mean():.4f} ± {effect.std():.4f}")

# 各期效应
for t in range(n_periods):
    eff_t = dynamic_dml.effect(X_arr, T0=np.zeros(n_units), T1=np.ones(n_units))
    print(f"  Period {t}: {eff_t.mean():.4f}")

# 置信区间
eff_inf = dynamic_dml.effect_inference(X_arr, T0=0, T1=1)
print(f"\nATE 置信区间（95%）: [{eff_inf.conf_int()[0].mean():.4f}, "
      f"{eff_inf.conf_int()[1].mean():.4f}]")
print(eff_inf.summary_frame().describe())
```

**与静态 DML 的对比：**

| 维度 | 静态 DML（PLM/IRM） | DynamicDML |
|------|-------------------|-----------|
| 时间维度 | 无（截面）或简单 DID 两期 | 多期面板，显式建模时间动态 |
| 状态依赖 | 不处理 | 通过马尔可夫假设显式处理 |
| 历史处理效应 | 不估计 | 可估计各期累积处理效应 |
| 数据要求 | 横截面 or 面板均可 | 必须平衡面板 + 多期（T≥3） |
| 识别假设 | CIA | CIA + 马尔可夫条件独立 |
| 计算复杂度 | 低 | 高（逐期残差化 × MC 迭代） |
| 适用场景 | 一次性政策冲击 | 多期干预、累积效应、政策路径 |

---

### Causal Forest 红线警告

> ⚠️ **Causal Forest 不是因果识别方法，是异质性发现工具。**
>
> **核心假设：** Causal Forest 假设 CIA（Unconfoundedness / Conditional Independence Assumption）——即给定协变量 X 后，处理变量 D 与潜在结果 (Y(0), Y(1)) 相互独立：
>
> $$Y(1), Y(0) \perp D \mid X$$
>
> **这意味着：**
> - Causal Forest **不解决内生性问题**——如果处理变量存在未被 X 控制的遗漏变量，Causal Forest 的 CATE 估计是有偏的
> - Causal Forest **不是 IV、DID、RDD 的替代**——它依然无法识别来自内生处理变量的因果效应
> - Causal Forest 只是在 **CIA 已经成立的前提下**，高效估计异质性处理效应

**正确使用流程：**

```
Step 1: 用 DID / IV / RDD 建立因果关系
        └→ 确认 X → Y 的因果效应存在且无偏
           ↓
Step 2: 在可信识别框架内，用 Causal Forest 探索异质性
        └→ 回答"效应对哪类个体大、对哪类小"
           ↓
Step 3: 基于 GRF 的 BLP / variable_importance 确认异质性维度
        └→ 结合经济理论解释异质性来源
```

**禁止的做法：**
```
❌ 直接在观察数据上跑 Causal Forest，宣称发现"因果"异质性效应
❌ 用 Causal Forest 替代 IV/DID/RDD 的识别策略
❌ 将 Causal Forest 的 CATE 作为论文主要贡献报告（应作为辅助分析）
```

---

### PLM 估计 ATO 说明

当样本的**共同支撑（Common Support）不充分**时，部分线性模型（PLM）的估计量不再收敛于 ATE（全样本平均处理效应），而是收敛于 **ATO（Average Treatment Effect on the Overlap，交叠加权平均处理效应）**。

**ATO 定义：**
$$\text{ATO} = \frac{E[e(X)(1-e(X))\tau(X)]}{E[e(X)(1-e(X))]}$$

其中 $e(X) = P(D=1|X)$ 为倾向得分。ATO 给处于倾向得分分布中间（共同支撑充分区域）的个体赋予更高权重，而给极端倾向得分个体赋予接近零的权重。

**研究者必须知晓的三点：**
1. **ATO ≠ ATE**：若处理组和控制组的协变量分布重叠不充分，PLM 估计量收敛于 ATO，不代表总体 ATE
2. **诊断方法**：检查倾向得分分布的重叠性（见 `dml_overlap_check.png`），若大量样本倾向得分 <0.05 或 >0.95，说明共同支撑不充分
3. **报告方式**：若诊断显示 ATO 而非 ATE，论文中必须明确声明估计量为 ATO，并解释目标估计量的经济含义

```r
# R: PLM 估计量诊断（ATE vs ATO）
library(DoubleML)
library(mlr3)

# 拟合 PLM
dml_plm <- DoubleMLPLR$new(dml_data, ml_l = learner_g, ml_m = learner_m,
                            n_folds = 5, n_rep = 3)
dml_plm$fit()

# 提取倾向得分预测（M模型对处理变量D的预测）
ps_hat <- dml_plm$predictions$ml_m[, 1, 1]  # 第一折第一重复

# 诊断重叠性
cat(sprintf("倾向得分分布:\n"))
cat(sprintf("  均值: %.4f, 中位数: %.4f\n", mean(ps_hat), median(ps_hat)))
cat(sprintf("  < 0.05: %d 个 (%.1f%%)\n",
            sum(ps_hat < 0.05), mean(ps_hat < 0.05) * 100))
cat(sprintf("  > 0.95: %d 个 (%.1f%%)\n",
            sum(ps_hat > 0.95), mean(ps_hat > 0.95) * 100))

# 判断 ATE vs ATO
extreme_ratio <- mean(ps_hat < 0.05 | ps_hat > 0.95)
if (extreme_ratio > 0.05) {
  warning(sprintf("%.1f%% 样本倾向得分极端（<0.05 或 >0.95），PLM 估计量趋向 ATO 而非 ATE。\n",
                  extreme_ratio * 100))
  cat("建议：(1) 在论文中明确声明估计量为 ATO；\n")
  cat("       (2) 考虑限制样本到共同支撑充分的子集；\n")
  cat("       (3) 或切换至 IRM 使用 trimming。\n")
} else {
  cat("✓ 重叠性良好，PLM 估计量近似 ATE\n")
}
```

---

### DML+DID 红线警告

> ⚠️ **严重方法论错误：不能将时间/个体固定效应作为高维协变量放入 PLM，再将系数解读为 DID 估计量。**
>
> **错误做法（常见于错误论文）：**
> ```python
> # 错误示例：将年份哑变量和个体哑变量全部作为 X 列加入 DML
> X_wrong = [year_dummy_2019, year_dummy_2020, ...,
>             firm_id_dummy_1, firm_id_dummy_2, ...]  # 高维固定效应哑变量
> dml_plm = DoubleMLPLR(data, ml_l=RF, ml_m=RF, ...)
> # 然后宣称估计了 DID 效应 → 这是错误的
> ```
>
> **为什么是错的：**
> 1. PLM 的设计假设是 $Y = D\theta + g_0(X) + \varepsilon$，其中 $g_0(X)$ 由 ML 非参数估计
> 2. 将固定效应哑变量放入 ML 模型中，ML 会用高维哑变量 "记住" 每个个体，导致 $m_0(X)$ 估计过拟合、残差化失效
> 3. 系数 $\theta$ 不再有 DID 的因果解释
>
> **正确做法（DML + DID）：**
> 1. 先用 `feols` 或 `lm` 对 Y 和 D 分别进行双向固定效应 demean（去除个体和时间固定效应）
> 2. 用 demean 后的残差 $\tilde{Y}, \tilde{D}$ 构造 DML 问题，控制变量 X 为**时变协变量**（不含固定效应）
> 3. 在 DML 的残差化步骤中进一步控制 X 的非线性效应

```r
# R: 正确的 DML + DID 实现方式
library(fixest)
library(DoubleML)
library(mlr3)
library(data.table)

# ---- Step 1：双向 FE demean（先去除固定效应）----
# 使用 fixest 的 demean 函数（或 feols 残差）
res_y_fe <- feols(outcome   ~ 1 | unit_fe + year_fe, data = df)
res_d_fe <- feols(treatment ~ 1 | unit_fe + year_fe, data = df)

df$Y_demeaned <- residuals(res_y_fe)  # 去除个体+时间FE后的残差
df$D_demeaned <- residuals(res_d_fe)

# ---- Step 2：DML 使用 demean 后残差 + 时变协变量 ----
# X 只包含时变协变量（不含固定效应哑变量！）
X_vars_timevarying <- c("time_vary_control1", "time_vary_control2")

dml_data_did <- DoubleMLData$new(
  data   = df,
  y_col  = "Y_demeaned",        # demean 后的 Y
  d_cols = "D_demeaned",        # demean 后的 D
  x_cols = X_vars_timevarying   # 仅时变协变量
)

learner_g <- lrn("regr.ranger", num.trees = 200)
learner_m <- lrn("regr.ranger", num.trees = 200)

dml_did <- DoubleMLPLR$new(dml_data_did, ml_l = learner_g, ml_m = learner_m,
                            n_folds = 5, n_rep = 3)
dml_did$fit()
cat("DML + DID 估计（正确方式）:\n")
print(dml_did$summary())
```

---

### DML+RDD 操作区分

在 RDD 框架内使用 DML 时，**操作规则与标准 DML + DID/IV 不同**：

**核心原则：只对 Y 做残差化，不对 D 做残差化。**

**原因：**
- RDD 中，处理变量 $D$ 由断点规则决定（$D = \mathbf{1}[R \geq c]$），没有内生性问题
- 对 $D$ 做残差化会破坏断点的识别结构，改变估计量的解释
- 只需对 $Y$ 去除协变量 $X$ 的非线性影响（降低残差方差，提高精度）

```r
# R: DML + RDD 正确操作
library(rdrobust)
library(ranger)
library(dplyr)

# ---- Step 1：仅对 Y 做 ML 残差化（控制协变量）----
# 在带宽内样本上训练
h_opt <- rdrobust(df$outcome, df$r_centered, c=0, p=1)$bws["h",1]
df_bw <- df %>% filter(abs(r_centered) <= h_opt)

# 用随机森林对 Y ~ X（协变量，不含评分变量R或处理D）做预测
rf_y <- ranger(
  outcome ~ control1 + control2 + control3,  # 不含 r_centered 和 above_cutoff！
  data        = df_bw,
  num.trees   = 500,
  sample.fraction = 0.7  # 非训练集预测（避免过拟合）
)
# 交叉拟合残差
df_bw$Y_residual <- df_bw$outcome - predict(rf_y, data = df_bw)$predictions

# ---- Step 2：用 Y 残差对原始 RDD 评分变量做 rdrobust ----
# D（above_cutoff）保持原样，不做残差化
rdd_dml <- rdrobust(
  y    = df_bw$Y_residual,  # 残差化后的 Y
  x    = df_bw$r_centered,  # 原始评分变量（不变）
  c    = 0, p = 1,
  h    = h_opt              # 固定带宽（已用原始 Y 确定）
)
summary(rdd_dml)

# ---- 对比：原始 RDD vs DML + RDD ----
rdd_original <- rdrobust(y = df_bw$outcome, x = df_bw$r_centered,
                          c = 0, p = 1, h = h_opt)
cat(sprintf("原始 RDD:     %.4f (SE: %.4f)\n",
            rdd_original$coef["Bias-Corrected",1], rdd_original$se["Robust",1]))
cat(sprintf("DML + RDD:    %.4f (SE: %.4f)\n",
            rdd_dml$coef["Bias-Corrected",1], rdd_dml$se["Robust",1]))
cat("注：点估计应相近（协变量外生），SE 应降低（ML 吸收了协变量噪音）\n")
```

---

### 阿森费尔特沉降警告

**阿森费尔特沉降（Ashenfelter's Dip）：** 处理组个体在接受处理之前，由于自我选择，往往在政策前期存在**系统性趋势偏离**（通常是结果变量下滑后参与培训/政策）。

**在 DML+DID 结合中的表现：**
- DML 拟合的 $E[Y|X]$ 如果未能控制处理前的临时冲击，会导致残差系统性非零
- 若处理组在政策前存在 Ashenfelter Dip（结果暂时下降），DML 残差化后 DID 估计量会高估政策效果

**检验方法：**

```r
# R: Ashenfelter Dip 检验（DML+DID 必做）
library(fixest)
library(dplyr)
library(ggplot2)

# 方法1：处理组 vs 控制组的政策前趋势图（Event Study）
# 在未加入 DML 的简单规格下检查
res_event <- feols(
  outcome ~ i(year, treated, ref = base_year) | unit_fe + year_fe,
  data    = df,
  cluster = ~unit_id
)
iplot(res_event,
      main = "Event Study: Ashenfelter Dip Check",
      xlab = "Year relative to policy",
      ref.line = 1)  # 政策前各年系数应在0附近

# 方法2：处理组政策前2~3期的 outcome 趋势
pre_periods <- df %>%
  filter(year < policy_year, treated == 1) %>%
  group_by(year) %>%
  summarise(mean_outcome = mean(outcome), .groups = "drop")

ggplot(pre_periods, aes(x = year, y = mean_outcome)) +
  geom_line(linewidth = 1, color = "steelblue") +
  geom_point(size = 3, color = "steelblue") +
  geom_vline(xintercept = policy_year - 0.5, linetype = "dashed", color = "red") +
  labs(title = "处理组政策前均值趋势（检验 Ashenfelter Dip）",
       subtitle = "若政策前 1~2 期明显下降，存在 Ashenfelter Dip",
       x = "年份", y = "结果变量均值") +
  theme_minimal()
ggsave("output/ashenfelter_dip_check.png", dpi = 150)

# 方法3：DML 残差的政策前分组检验
# 若 DML+DID 的 Y 残差在政策前仍存在趋势，说明控制不充分
df_pre <- df %>% filter(year < policy_year)
res_dml_pretrend <- feols(
  Y_demeaned ~ i(year, treated, ref = policy_year - 1) | unit_fe + year_fe,
  data    = df_pre,
  cluster = ~unit_id
)
iplot(res_dml_pretrend, main = "DML 残差政策前趋势（应均不显著）")
# 若仍有显著的政策前趋势，说明存在 Ashenfelter Dip 且 DML 未充分控制
```

**处理建议：**
- 若发现 Ashenfelter Dip：加入处理前 1~2 期的结果变量作为控制变量（lagged outcome）
- 或在 DML 的 X 中加入政策前期的结果变量，让 ML 吸收这一预处理趋势
- 或使用 Callaway-Sant'Anna 的 "not-yet-treated" 对照组（避免选用已知有类似 dip 的对照组）

---

### Estimand 声明

**DML 框架下各模型的估计量（Estimand）明确声明：**

| 模型 | 默认估计量 | 条件 | 必须声明 |
|------|---------|------|---------|
| **PLM（部分线性模型）** | **ATO**（共同支撑不足时） | CIA + 重叠性充分时近似 ATE | 明确声明不是 ATE，报告倾向得分重叠性诊断 |
| **IRM（交互回归模型）** | **ATE / ATT / CATE** | 由 `score` 参数控制 | 指定估计的是哪一个，说明目标人群 |
| **DML-IV（IIVM）** | **LATE** | 工具变量下的 Complier 效应 | 同 IV/2SLS 的 Complier 声明要求 |
| **因果森林（CausalForest）** | **CATE（条件 ATE）** | CIA，异质性 ATE | 声明是 CIA 下的 CATE，依赖于 X 的充分性 |
| **DynamicDML** | **累积 ATE（各期）** | 马尔可夫 CIA | 说明是哪些期的累积效应 |

**PLM → ATO 标准声明模板：**
```
本文 DML 部分线性模型（PLM）的估计量，在样本共同支撑充分（倾向得分均在 [0.05, 0.95] 范围内）
时近似总体平均处理效应（ATE）。倾向得分重叠性诊断显示 [X%] 的样本倾向得分极端，
因此估计量更接近交叠加权平均处理效应（ATO），其政策含义适用于处理状态有实质不确定性的个体子群。
```

**IRM → ATE/ATT/CATE 标准声明模板：**
```
本文 DML 交互回归模型（IRM）以 [ATE/ATT/CATE] 为目标估计量。
[若 ATE：]估计结果代表从控制组转换为处理组的总体平均效应，
假设共同支撑充分且无溢出效应。
[若 ATT：]估计结果代表处理组个体的平均处理效应，
倾向得分加权剔除了处理组和控制组的可比性缺口。
[若 CATE：]通过 CausalForest / DR-Learner 估计，
报告各协变量取值下的条件处理效应，并用 BLP 检验其显著性。
```
