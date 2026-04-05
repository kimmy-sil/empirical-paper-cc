# SHAP 值解释指南

> 本文件为可选参考资料，在需要解释ML模型预测或DML中学习器贡献时加载。

SHAP（SHapley Additive exPlanations）基于博弈论Shapley值，将ML模型的预测分解为每个特征的贡献量，具有加法性和一致性保证。

---

## 核心概念

- **SHAP值**：特征 $x_i$ 对预测 $\hat{f}(x)$ 相对于基准（期望值）的边际贡献
- **加法性**：$\hat{f}(x) = \mathbb{E}[\hat{f}] + \sum_i \phi_i$
- **TreeSHAP**：树模型的精确快速计算（随机森林、XGBoost）
- **DeepSHAP**：神经网络近似

---

## Python 代码

### 基础SHAP分析

```python
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# 训练模型
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(df[features], df['Y'])

# 计算SHAP值（TreeExplainer对树模型最快）
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df[features])  # shape: (N, p)

# 全局重要性：平均|SHAP|
shap_importance = pd.DataFrame({
    'feature':    features,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)
print(shap_importance)
```

### 标准可视化图

```python
# 1. Summary Plot（蜂群图，全局 + 方向）
shap.summary_plot(shap_values, df[features], plot_type="dot", show=False)
plt.tight_layout()
plt.savefig('output/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Bar Plot（全局重要性排序）
shap.summary_plot(shap_values, df[features], plot_type="bar", show=False)
plt.savefig('output/shap_importance_bar.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Dependence Plot（单变量非线性关系）
shap.dependence_plot(
    'feature_name',           # 目标特征
    shap_values,
    df[features],
    interaction_index='auto', # 自动选择交互特征（显示为颜色）
    show=False
)
plt.savefig('output/shap_dependence_feature.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Waterfall Plot（单个样本解释）
shap.waterfall_plot(
    shap.Explanation(
        values       = shap_values[0],
        base_values  = explainer.expected_value,
        data         = df[features].iloc[0],
        feature_names = features
    ),
    show=False
)
plt.savefig('output/shap_waterfall_obs0.png', dpi=150, bbox_inches='tight')
plt.close()
```

### DML中的SHAP应用（解释nuisance学习器）

```python
# 解释Y方程学习器（了解哪些变量驱动了Y的变化）
from doubleml import DoubleMLPLR, DoubleMLData
from sklearn.ensemble import RandomForestRegressor

# 在全样本上重新训练（DML是交叉拟合，SHAP用全样本近似）
ml_y = RandomForestRegressor(n_estimators=200, random_state=42)
ml_y.fit(df[controls], df['Y'])

explainer_y = shap.TreeExplainer(ml_y)
shap_y = explainer_y.shap_values(df[controls])

shap.summary_plot(shap_y, df[controls], plot_type="bar",
                  title="Y方程：控制变量重要性（SHAP）", show=False)
plt.savefig('output/shap_nuisance_y.png', dpi=150, bbox_inches='tight')
plt.close()
```

---

## R 代码（treeshap / fastshap）

```r
# R: SHAP值计算（treeshap包，支持ranger/xgboost）
# install.packages("treeshap")
library(treeshap)
library(ranger)
library(ggplot2)

# 训练模型
rf_model <- ranger(
  Y ~ .,
  data        = df[, c("Y", features)],
  num.trees   = 200,
  importance  = "permutation"
)

# TreeSHAP
unified <- unify(rf_model, df[, features])
shap_obj <- treeshap(unified, df[, features], interactions = FALSE)

# 提取SHAP矩阵
shap_matrix <- shap_obj$shaps   # shape: (N, p)

# 全局重要性
mean_abs_shap <- colMeans(abs(shap_matrix))
imp_df <- data.frame(
  feature    = names(mean_abs_shap),
  importance = mean_abs_shap
) |> dplyr::arrange(dplyr::desc(importance))
print(imp_df)

# 可视化（treeshap内置）
plot_feature_importance(shap_obj, max_vars = 15)
plot_contribution(shap_obj, obs = 1)   # 单观测瀑布图
```

---

## ⚠️ SHAP在因果分析中的限制

1. **SHAP ≠ 因果效应**：SHAP反映预测贡献，不是处理效应。高SHAP值不意味着该变量有因果作用。

2. **SHAP解释的是nuisance，不是θ**：在DML框架中，SHAP用于理解ML学习器的行为（控制了什么），不能直接解释因果参数。

3. **关联性问题**：特征间高度相关时，SHAP值会在相关变量间分散，难以解释单变量效应。

4. **正确使用场景**：
   - 报告nuisance学习器的控制变量重要性（透明度要求）
   - 检查学习器是否合理（哪些变量在预测Y/D中起作用）
   - 辅助异质性分析（CATE与哪些变量相关）

---

## 报告模板

> "为提高估计透明度，图X展示了DML中Y方程学习器（随机森林）的SHAP值分布。[变量A]和[变量B]对Y预测贡献最大，与经济理论一致。注意SHAP值反映预测贡献，不代表因果效应。"
