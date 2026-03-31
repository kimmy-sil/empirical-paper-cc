# EconML 估计器索引（Microsoft EconML 完整参考）

**版本基准：** econml >= 0.14  
**安装：** `pip install econml`  
**官方文档：** https://econml.azurewebsites.net/

---

## 1. DML 系列（Double/Debiased Machine Learning）

### LinearDML
**一句话描述：** 部分线性 DML，第二阶段用 LASSO 估计低维 θ，假设效应同质。

```python
from econml.dml import LinearDML
from sklearn.ensemble import GradientBoostingRegressor

est = LinearDML(
    model_y = GradientBoostingRegressor(),  # 预测 E[Y|X,W]
    model_t = GradientBoostingRegressor(),  # 预测 E[D|X,W]
    cv      = 5,
    random_state = 42
)
est.fit(Y, D, X=X_hetero, W=X_controls)  # X=异质性特征, W=高维控制
print(est.ate_inference().summary_frame())
# 输出：ATE, SE, t, p, 95% CI

# 系数（θ 对 X 的线性投影）
print(est.coef_inference().summary_frame())
```

---

### SparseLinearDML
**一句话描述：** LinearDML 的稀疏版本，第二阶段用 LASSO 对高维 X 估计效应修正项，适合特征维度大的 CATE 线性近似。

```python
from econml.dml import SparseLinearDML

est = SparseLinearDML(
    model_y      = GradientBoostingRegressor(),
    model_t      = GradientBoostingRegressor(),
    featurizer   = None,   # 可传入特征变换器（如多项式）
    cv           = 5,
    random_state = 42
)
est.fit(Y, D, X=X_high_dim, W=None)
print(est.coef_inference().summary_frame())  # 每个 X 维度对 CATE 的稀疏系数
```

---

### NonParamDML
**一句话描述：** 完全非参数 DML，CATE 由 ML 模型非参数估计（无线性假设），适合探索任意形状的异质性。

```python
from econml.dml import NonParamDML
from sklearn.ensemble import RandomForestRegressor

est = NonParamDML(
    model_y      = RandomForestRegressor(n_estimators=200),
    model_t      = RandomForestRegressor(n_estimators=200),
    model_final  = RandomForestRegressor(n_estimators=200),  # CATE 估计器
    cv           = 5,
    random_state = 42
)
est.fit(Y, D, X=X_hetero, W=X_controls)
tau_hat = est.effect(X_test)  # 个体 CATE 预测
```

---

### CausalForestDML
**一句话描述：** DML 框架下的因果森林，结合 DML 残差化和 GRF 树结构，支持高维控制变量 W 和异质性特征 X 分离。

```python
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor

cf = CausalForestDML(
    model_y       = GradientBoostingRegressor(n_estimators=100),
    model_t       = GradientBoostingRegressor(n_estimators=100),
    n_estimators  = 1000,
    min_samples_leaf = 5,
    max_features  = "sqrt",
    cv            = 5,
    random_state  = 42
)
cf.fit(Y, D, X=X_hetero, W=X_controls)

# 个体 CATE
tau_hat = cf.predict(X_test)
lb, ub  = cf.predict_interval(X_test, alpha=0.05)

# ATE
print(cf.ate_inference().summary_frame())

# 特征重要性
feat_imp = cf.feature_importances_
```

---

### DynamicDML
**一句话描述：** 平衡面板多期处理的动态 DML，通过马尔可夫条件处理状态依赖和历史效应。

```python
from econml.panel.dml import DynamicDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

dyn_dml = DynamicDML(
    model_y  = GradientBoostingRegressor(n_estimators=100),
    model_t  = GradientBoostingClassifier(n_estimators=100),
    cv       = 3,
    mc_iters = 3,
    mc_agg   = 'median'
)
# Y, T: (n_units, n_periods) 矩阵; X: (n_units, n_features)
dyn_dml.fit(Y_matrix, T_matrix, X=X_static)

# 估计处理效应（T0→T1 的效应）
eff = dyn_dml.effect(X_static, T0=0, T1=1)
print(f"Dynamic ATE: {eff.mean():.4f}")

# 带置信区间
eff_inf = dyn_dml.effect_inference(X_static, T0=0, T1=1)
print(eff_inf.summary_frame().describe())
```

---

## 2. DR-Learner 系列（Doubly Robust Learner）

### DRLearner（基础版）
**一句话描述：** 基于双重稳健评分函数估计 CATE，对倾向得分或结果模型之一误设时仍然一致，适合二元处理下的 CATE 估计。

```python
from econml.dr import DRLearner
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

dr = DRLearner(
    model_propensity   = GradientBoostingClassifier(n_estimators=100),  # P(D=1|X)
    model_regression   = GradientBoostingRegressor(n_estimators=100),   # E[Y|D,X]
    model_final        = GradientBoostingRegressor(n_estimators=100),   # CATE 回归
    cv                 = 5,
    random_state       = 42
)
dr.fit(Y, D, X=X_hetero, W=X_controls)
tau_dr = dr.effect(X_test)
print(dr.ate_inference().summary_frame())  # ATE + 95% CI
```

---

### LinearDRLearner
**一句话描述：** DR-Learner 的线性 CATE 版本，第二阶段用 LASSO 线性近似 CATE(X)，提供系数推断。

```python
from econml.dr import LinearDRLearner
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

linear_dr = LinearDRLearner(
    model_propensity = GradientBoostingClassifier(),
    model_regression = GradientBoostingRegressor(),
    cv = 5, random_state = 42
)
linear_dr.fit(Y, D, X=X_hetero, W=X_controls)
print(linear_dr.coef_inference().summary_frame())  # CATE 的线性系数
print(linear_dr.ate_inference().summary_frame())
```

---

### SparseLinearDRLearner
**一句话描述：** DR-Learner + 稀疏线性 CATE，适合高维 X 的线性 CATE 近似。与 SparseLinearDML 相似，但使用 DR 评分函数。

```python
from econml.dr import SparseLinearDRLearner
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

sparse_dr = SparseLinearDRLearner(
    model_propensity = GradientBoostingClassifier(),
    model_regression = GradientBoostingRegressor(),
    cv = 5, random_state = 42
)
sparse_dr.fit(Y, D, X=X_high_dim, W=None)
# 稀疏系数：哪些 X 维度最重要
coef_df = sparse_dr.coef_inference().summary_frame()
print(coef_df[coef_df['pvalue'] < 0.05])  # 仅显示显著系数
```

---

### ForestDRLearner
**一句话描述：** DR-Learner 的随机森林版本，完全非参数 CATE，使用诚实树（honest trees）提供有效置信区间。类似 GRF，但基于 DR 评分函数。

```python
from econml.dr import ForestDRLearner
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

forest_dr = ForestDRLearner(
    model_propensity  = GradientBoostingClassifier(n_estimators=100),
    model_regression  = GradientBoostingRegressor(n_estimators=100),
    n_estimators      = 1000,
    min_samples_leaf  = 5,
    max_features      = "sqrt",
    cv                = 5,
    random_state      = 42
)
forest_dr.fit(Y, D, X=X_hetero, W=X_controls)
tau_forest_dr = forest_dr.effect(X_test)
lb_dr, ub_dr  = forest_dr.predict_interval(X_test, alpha=0.05)
print(f"DR Forest ATE: {tau_forest_dr.mean():.4f}")
```

---

## 3. 工具变量系列（IV / LATE）

### OrthoIV
**一句话描述：** 基于正交化的 IV 估计，高维控制变量下估计 LATE，适合二元工具变量 + 二元处理变量。

```python
from econml.iv.dml import OrthoIV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

ortho_iv = OrthoIV(
    model_y_xw = GradientBoostingRegressor(),    # E[Y|X,W]
    model_t_xw = GradientBoostingClassifier(),   # P(D=1|X,W)
    model_z_xw = GradientBoostingClassifier(),   # P(Z=1|X,W)（first stage）
    cv         = 5,
    random_state = 42
)
# Z: 二元工具变量列向量
ortho_iv.fit(Y, D, Z=Z_instrument, X=X_hetero, W=X_controls)
print(ortho_iv.ate_inference().summary_frame())  # LATE
```

---

### NonParamDMLIV（DMLIV）
**一句话描述：** DML 框架下的非参数 IV 估计，支持连续工具变量，CATE 非参数估计，适合弱化排他性约束的探索性分析。

```python
from econml.iv.dml import NonParamDMLIV
from sklearn.ensemble import GradientBoostingRegressor

npdml_iv = NonParamDMLIV(
    model_y_xw   = GradientBoostingRegressor(),
    model_t_xwz  = GradientBoostingRegressor(),  # E[D|X,W,Z]（first stage）
    model_t_xw   = GradientBoostingRegressor(),  # E[D|X,W]（用于正交化）
    model_final  = GradientBoostingRegressor(),  # CATE 估计器
    cv           = 5
)
npdml_iv.fit(Y, D, Z=Z_continuous, X=X_hetero, W=X_controls)
tau_iv_hat = npdml_iv.effect(X_test)
```

---

## 4. Meta-Learner 系列（S/T/X Learner）

### S-Learner（Single Learner）
**一句话描述：** 用单个 ML 模型拟合 E[Y|D,X]，然后通过对比 D=1 和 D=0 时的预测得到 CATE。简单但可能忽略处理效应。

```python
from econml.meta import SLearner
from sklearn.ensemble import GradientBoostingRegressor

s_learner = SLearner(
    overall_model = GradientBoostingRegressor(n_estimators=200, random_state=42)
)
s_learner.fit(Y, D, X=X_hetero)
tau_s = s_learner.effect(X_test)
print(f"S-Learner ATE: {tau_s.mean():.4f}")
```

---

### T-Learner（Two Learner）
**一句话描述：** 分别对处理组和控制组训练两个 ML 模型，CATE = μ₁(X) - μ₀(X)。比 S-Learner 更关注处理效应异质性，但可能在小样本下不稳定。

```python
from econml.meta import TLearner
from sklearn.ensemble import GradientBoostingRegressor

t_learner = TLearner(
    models = GradientBoostingRegressor(n_estimators=200, random_state=42)
    # 或传入列表：models=[model_for_D0, model_for_D1]
)
t_learner.fit(Y, D, X=X_hetero)
tau_t = t_learner.effect(X_test)

# 与 S-Learner 对比
print(f"T-Learner ATE: {tau_t.mean():.4f}")
print(f"S-Learner ATE: {tau_s.mean():.4f}")
```

---

### X-Learner（Cross Learner）
**一句话描述：** Künzel et al. (2019) 提出，通过交叉学习修正 T-Learner 在样本不均衡时的偏差。处理组：用控制组模型预测 τ₁；控制组：用处理组模型预测 τ₀；最后用倾向得分加权合并。

```python
from econml.meta import XLearner
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

x_learner = XLearner(
    models = GradientBoostingRegressor(n_estimators=200, random_state=42),
    propensity_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
)
x_learner.fit(Y, D, X=X_hetero)
tau_x = x_learner.effect(X_test)
print(f"X-Learner ATE: {tau_x.mean():.4f}")

# 注：X-Learner 在处理/控制组样本量相差较大时表现最好
print(f"样本量：处理={D.sum()}, 控制={len(D)-D.sum()}")
```

---

## 5. 策略树/森林（Policy Tree / Forest）

### DRPolicyTree
**一句话描述：** 基于 DR 评分函数学习最优处理分配规则，输出可解释决策树（深度可控），适合政策制定（谁该受处理）。

```python
from econml.policy import DRPolicyTree
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

# 第一步：获取 DR 评分（也可从 DRLearner 提取）
dr_scores = ForestDRLearner(
    model_propensity = GradientBoostingClassifier(),
    model_regression = GradientBoostingRegressor(),
    cv = 5
)
dr_scores.fit(Y, D, X=X_hetero, W=X_controls)
scores = dr_scores.score_  # DR 评分向量

# 第二步：学习策略树
policy_tree = DRPolicyTree(
    max_depth = 2,
    min_impurity_decrease = 0.001
)
policy_tree.fit(X_hetero, scores)

# 可视化策略树
from econml.policy import PolicyTreeVisualizer
PolicyTreeVisualizer.render(policy_tree, feature_names=hetero_names,
                             treatment_names=['Control', 'Treat'])

# 对新个体推断最优处理
optimal_treat = policy_tree.predict(X_test)
print(f"建议处理比例: {optimal_treat.mean():.1%}")
```

---

### DRPolicyForest
**一句话描述：** 策略森林（随机策略树集成），比单棵策略树更稳定，适合大样本下的最优处理分配估计。

```python
from econml.policy import DRPolicyForest

policy_forest = DRPolicyForest(
    n_estimators  = 1000,
    max_depth     = 4,
    min_samples_leaf = 10,
    random_state  = 42
)
policy_forest.fit(X_hetero, scores)
optimal_treat_forest = policy_forest.predict(X_test)

# 特征重要性（哪些特征决定处理分配）
vim_policy = policy_forest.feature_importances_
for name, imp in sorted(zip(hetero_names, vim_policy), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.4f}")
```

---

## 6. 评分函数工具

### RScorer
**一句话描述：** 用 R-Score（R²的处理效应版本）评估 CATE 估计器的质量，提供无偏的模型选择标准，可用于比较不同 CATE 方法。

```python
from econml.score import RScorer
from sklearn.ensemble import GradientBoostingRegressor

# 构造 RScorer
r_scorer = RScorer(
    model_y = GradientBoostingRegressor(n_estimators=100),
    model_t = GradientBoostingRegressor(n_estimators=100),
    cv      = 5,
    random_state = 42
)
r_scorer.fit(Y, D, X=X_hetero, W=X_controls)

# 评估不同 CATE 估计器（tau_hat 数组）
scores = {}
for name, tau in [('S-Learner', tau_s), ('T-Learner', tau_t),
                   ('X-Learner', tau_x), ('CausalForest', tau_hat)]:
    score = r_scorer.score(tau)
    scores[name] = score
    print(f"{name:20s}: R-Score = {score:.4f}")

# 选择 R-Score 最高的方法（推荐最优 CATE 估计器）
best_method = max(scores, key=scores.get)
print(f"\n推荐方法：{best_method}（R-Score = {scores[best_method]:.4f}）")
```

---

## 7. 方法选择速查表

| 场景 | 推荐方法 | 关键参数 |
|------|---------|---------|
| 同质效应 + 高维控制 | `LinearDML` | `model_y`, `model_t`, `cv` |
| 非线性异质效应 + 中等维度 | `CausalForestDML` | `n_estimators=1000` |
| 二元处理 + 双重稳健 | `ForestDRLearner` | `model_propensity`, `model_regression` |
| 工具变量 + 高维控制 | `OrthoIV` | `model_z_xw` |
| 多期面板 + 状态依赖 | `DynamicDML` | `mc_iters` |
| 处理/控制样本不均衡 | `XLearner` | `propensity_model` |
| 最优政策分配 | `DRPolicyTree` | `max_depth` |
| CATE 方法比较 | `RScorer` | — |

---

## 8. 通用输出接口

所有 EconML 估计器共享以下方法：

```python
# 拟合
est.fit(Y, D, X=X, W=W)

# 点估计（CATE）
tau = est.effect(X_test)           # 个体 CATE
ate = est.ate(X_test)              # ATE（对 X_test 平均）
att = est.att(X_test, D_test)      # ATT

# 置信区间
lb, ub = est.effect_interval(X_test, alpha=0.05)

# 统计推断对象（含 SE、t、p、CI）
inf = est.effect_inference(X_test)
print(inf.summary_frame().head())

# ATE 推断
print(est.ate_inference().summary_frame())

# 系数（线性方法）
print(est.coef_inference().summary_frame())
```
