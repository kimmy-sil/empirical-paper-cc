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
