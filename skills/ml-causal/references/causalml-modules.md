# CausalML 模块索引（Uber CausalML 完整参考）

**版本基准：** causalml >= 0.14  
**安装：** `pip install causalml`  
**官方文档：** https://causalml.readthedocs.io/  
**GitHub：** https://github.com/uber/causalml

---

## 1. 核心设计理念

CausalML 专注于**商业场景的 Uplift Modeling**（增益建模），与 EconML 的学术导向（LATE、CATE 理论）不同：
- 重点：谁是"说服型"用户（Persuadables）——即施加处理才会改变行为的个体
- 应用：营销优惠、医疗干预、推荐系统中的个体化策略
- 方法：Uplift Tree、Meta-Learner、Sensitivity Analysis

---

## 2. Uplift Tree 系列

### UpliftTreeClassifier
**一句话描述：** 单棵 Uplift 决策树，直接优化处理效应（而非传统分类精度），基于 KL 散度/卡方等 Uplift 分裂标准，输出每个叶节点的 uplift 值。

```python
from causalml.inference.tree import UpliftTreeClassifier
import numpy as np
import pandas as pd

# ---- 数据准备 ----
# 需要字段：X（特征矩阵）, treatment（处理组='treatment', 控制组='control'), y（二元结果）
clf = UpliftTreeClassifier(
    control_name  = 'control',      # 控制组标签
    max_depth     = 5,              # 树最大深度
    min_samples_leaf = 50,          # 叶节点最小样本量（防过拟合）
    min_samples_treatment = 10,     # 叶节点处理组最小样本
    n_reg         = 100,            # 正则化参数（越大越平滑）
    evaluationFunction = 'KL',      # 分裂标准：'KL', 'Chi', 'ED', 'CTS'
    random_state  = 42
)

clf.fit(X_train, treatment_train, y_train)

# 预测 uplift（个体增益）
uplift_scores = clf.predict(X_test)
print(f"平均 Uplift: {uplift_scores.mean():.4f}")
print(f"Uplift 分布: min={uplift_scores.min():.4f}, max={uplift_scores.max():.4f}")

# 可视化决策树
from causalml.inference.tree import uplift_tree_string
print(uplift_tree_string(clf, feature_names=feature_names))
```

---

### UpliftRandomForestClassifier
**一句话描述：** Uplift 随机森林，集成多棵 Uplift 树，比单棵树更稳定，是商业 Uplift 建模的推荐方法。支持特征重要性分析。

```python
from causalml.inference.tree import UpliftRandomForestClassifier

rf_uplift = UpliftRandomForestClassifier(
    control_name       = 'control',
    n_estimators       = 200,         # 树的数量
    max_depth          = 6,
    min_samples_leaf   = 50,
    min_samples_treatment = 10,
    n_reg              = 100,
    evaluationFunction = 'KL',
    n_jobs             = -1,          # 并行
    random_state       = 42
)

rf_uplift.fit(X_train, treatment_train, y_train)
uplift_rf = rf_uplift.predict(X_test)

# ---- 特征重要性 ----
feature_importance = pd.Series(
    rf_uplift.feature_importances_,
    index = feature_names
).sort_values(ascending=False)
print("Uplift Forest 特征重要性（Top 10）:")
print(feature_importance.head(10))

# ---- Uplift 曲线（Qini Curve）----
from causalml.metrics import plot_gain, plot_qini

# 准备评估数据（需要真实处理分配和结果）
plot_qini(
    df_test[['y', 'treatment', 'uplift_score']],
    outcome_col    = 'y',
    treatment_col  = 'treatment',
    treatment_effect_col = 'uplift_score'
)
# Qini 系数 > 0 且曲线在对角线之上 → Uplift 模型有效
```

---

### UpliftTreeRegressor（连续结果）
**一句话描述：** 针对连续结果变量（如消费金额、使用时长）的 Uplift 树，损失函数改为均方误差类。

```python
from causalml.inference.tree import UpliftTreeRegressor

reg_uplift = UpliftTreeRegressor(
    control_name    = 'control',
    max_depth       = 5,
    min_samples_leaf = 100,
    random_state    = 42
)
reg_uplift.fit(X_train, treatment_train, y_continuous_train)
uplift_reg = reg_uplift.predict(X_test)
```

---

## 3. Meta-Learner 系列

CausalML 提供与 EconML 类似的 Meta-Learner，但接口更贴近商业使用（直接传入 treatment 字符串数组）。

### S-Learner
```python
from causalml.inference.meta import LRSRegressor  # LR-based S-Learner
from causalml.inference.meta import XGBTRegressor  # XGBoost-based S-Learner

# XGBoost S-Learner
s_learner = XGBTRegressor(
    control_name = 'control',
    random_state = 42
)
s_learner.fit(X_train, treatment_train, y_train)
uplift_s = s_learner.predict(X_test)
print(f"S-Learner ATE: {uplift_s.mean():.4f}")
```

### T-Learner
```python
from causalml.inference.meta import BaseTRegressor
from sklearn.ensemble import GradientBoostingRegressor

t_learner = BaseTRegressor(
    learner      = GradientBoostingRegressor(n_estimators=100),
    control_name = 'control'
)
t_learner.fit(X_train, treatment_train, y_train)
uplift_t, uplift_t_lb, uplift_t_ub = t_learner.predict(
    X_test, return_ci=True, n_bootstraps=100
)
```

### X-Learner
```python
from causalml.inference.meta import BaseXRegressor
from sklearn.ensemble import GradientBoostingRegressor

x_learner = BaseXRegressor(
    learner      = GradientBoostingRegressor(n_estimators=100),
    control_name = 'control'
)
x_learner.fit(X_train, treatment_train, y_train)
uplift_x = x_learner.predict(X_test)
print(f"X-Learner ATE: {uplift_x.mean():.4f}")
```

### R-Learner
```python
from causalml.inference.meta import BaseRRegressor
from sklearn.ensemble import GradientBoostingRegressor

r_learner = BaseRRegressor(
    learner      = GradientBoostingRegressor(n_estimators=100),
    control_name = 'control'
)
r_learner.fit(X_train, treatment_train, y_train)
uplift_r = r_learner.predict(X_test)
```

---

## 4. 敏感性分析（Sensitivity Analysis / Rosenbaum Bounds）

### Sensitivity Analysis（Rosenbaum Bounds）
**一句话描述：** 评估未观测混淆因素对因果估计的影响范围。Rosenbaum (2002) 方法：给定隐藏偏差参数 Γ（gamma），计算处理效应在最坏情况下的 p 值上界。

**核心参数 Γ：**
- Γ = 1：无隐藏偏差（精确随机化）
- Γ = 2：允许处理分配概率在 [1/3, 2/3] 内变化（相差一倍）
- 若在 Γ = 2 时结论仍显著，说明对隐藏偏差相当稳健

```python
from causalml.inference.meta import LRSRegressor
import numpy as np

# ---- 使用 Sensitivity Analysis 模块 ----
# 方法1：内置敏感性分析（通过 estimate_ate 的置信区间变化）
s_learner = LRSRegressor(control_name='control')
s_learner.fit(X_train, treatment_train, y_train)

# Bootstrap 置信区间（基础）
ate, ate_lb, ate_ub = s_learner.estimate_ate(
    X_test, treatment_test, y_test,
    bootstrap_ci = True,
    n_bootstraps  = 500,
    bootstrap_size = 500
)
print(f"ATE: {ate:.4f}  95% CI: [{ate_lb:.4f}, {ate_ub:.4f}]")

# 方法2：Rosenbaum Bounds（需手动实现或使用 sensitivity 子模块）
# CausalML 的 sensitivity 模块（>= 0.13）
try:
    from causalml.inference.meta.sensitivity import Sensitivity

    sensitivity = Sensitivity(
        df           = df_train,
        inference_dict = {
            'X': X_train,
            'p': s_learner.propensity_scores_,  # 倾向得分
            'y': y_train,
            'w': treatment_train,
            'tau': s_learner.predict(X_train)
        },
        alpha = 0.05,    # 显著性水平
        n_sample_iter = 100
    )

    # 计算在不同 Gamma 值下的显著性
    for gamma in [1.0, 1.25, 1.5, 2.0, 2.5]:
        result = sensitivity.sensitivity_analysis(
            n_jobs = -1
        )
        print(f"Γ = {gamma:.2f}: p-value upper bound = ???")

except ImportError:
    # 手动 Rosenbaum Bounds（近似）
    print("手动 Rosenbaum Bounds（近似）:")
    from scipy import stats

    # 配对检验（需要匹配对）
    ate_estimate = uplift_s.mean()
    ate_se = uplift_s.std() / np.sqrt(len(uplift_s))

    gamma_grid = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    for gamma in gamma_grid:
        # Rosenbaum Bound：调整 odds 比后的 SE 上界（简化近似）
        se_adjusted = ate_se * np.sqrt(gamma)  # 保守估计
        t_stat = ate_estimate / se_adjusted
        p_ub   = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        print(f"  Γ = {gamma:.2f}: t = {t_stat:.3f}, p_upper = {p_ub:.4f}",
              "← 结论依然显著" if p_ub < 0.05 else "← 结论变为不显著")
```

---

## 5. 模型评估工具

### Uplift 评估指标

```python
from causalml.metrics import (
    plot_gain,           # Gain 曲线（累积 uplift）
    plot_qini,           # Qini 曲线（Qini 系数）
    auuc_score,          # AUUC 分数（Area Under Uplift Curve）
    qini_score           # Qini 系数
)
import pandas as pd

# 准备评估 DataFrame
eval_df = pd.DataFrame({
    'y':          y_test,
    'treatment':  treatment_test,
    'uplift':     uplift_rf
})

# AUUC 分数
auuc = auuc_score(eval_df, outcome_col='y', treatment_col='treatment',
                   treatment_effect_col='uplift')
print(f"AUUC: {auuc:.4f}（越高越好，随机模型=0）")

# Qini 系数
qini = qini_score(eval_df, outcome_col='y', treatment_col='treatment',
                   treatment_effect_col='uplift')
print(f"Qini 系数: {qini:.4f}")

# 可视化（Gain Curve + Qini Curve）
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_gain(eval_df, outcome_col='y', treatment_col='treatment',
           treatment_effect_col='uplift', ax=axes[0])
plot_qini(eval_df, outcome_col='y', treatment_col='treatment',
           treatment_effect_col='uplift', ax=axes[1])
plt.tight_layout()
plt.savefig("output/uplift_evaluation.png", dpi=150)
```

---

## 6. 方法选择速查

| 场景 | 推荐方法 | 关键参数 |
|------|---------|---------|
| 二元结果 + 可解释策略规则 | `UpliftTreeClassifier` | `max_depth`, `evaluationFunction` |
| 二元结果 + 稳定性要求 | `UpliftRandomForestClassifier` | `n_estimators=200` |
| 连续结果 | `UpliftTreeRegressor` | `min_samples_leaf` |
| 处理/控制不均衡 | `BaseXRegressor` | — |
| 快速基线对比 | `XGBTRegressor`（S-Learner） | — |
| 敏感性分析 | Rosenbaum Bounds | Γ 网格 |
| 模型比较 | `auuc_score` + `qini_score` | — |

---

## 7. CausalML vs EconML 对比

| 维度 | CausalML（Uber） | EconML（Microsoft） |
|------|----------------|-------------------|
| 主要用途 | 商业 Uplift 建模，营销/推荐 | 学术因果推断，政策评估 |
| 核心方法 | Uplift Tree/Forest | DML, DR-Learner, IV |
| 接口设计 | 面向工程（treatment 字符串数组） | 面向研究（Y/D/X/W 矩阵） |
| 统计推断 | Bootstrap 置信区间 | 半参数渐近理论 |
| 敏感性分析 | Rosenbaum Bounds | plausexog 类方法 |
| 评估工具 | Qini / AUUC / Gain Curve | R-Score |
| 适用场景 | A/B 测试后的异质性分析 | 自然实验、观察数据因果 |
