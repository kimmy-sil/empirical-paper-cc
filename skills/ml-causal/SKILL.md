# 机器学习因果推断

适用场景：高维控制变量选择、异质性处理效应探索、ML辅助识别策略。触发关键词：DML、因果森林、LASSO、doubleml、grf、econml、ML因果、异质性处理效应、HTE。

---

## 1. 核心原则：先识别后估计

**DML是估计优化工具，不是识别策略。**

- CIA / 平行趋势 / 排他性等核心识别假设在引入DML之前必须成立
- DML解决的是高维控制变量下的正则化偏误，不能使内生变量变外生
- 识别不可信 → DML救不了；识别可信 → DML提升效率

**何时不该用DML：**
- 效应线性且同质 + 控制变量维度低 → 传统OLS够用
- N < 500 → 交叉拟合过拟合风险高，慎用
- 只有二值处理且样本小 → PLM更稳

**正确流程：先确认识别策略 → 再选DML做估计优化**

---

## 2. DML 基础

### 2.1 PLM vs IRM 选择

| 维度 | PLM（部分线性模型） | IRM（交互随机模型） |
|------|-------------------|--------------------|
| 效应设定 | 同质处理效应 | 允许异质处理效应 |
| Estimand | **ATO**（不是ATE！） | ATE / ATT / CATE |
| 适用条件 | 支撑弱/样本小 | 共同支撑充分/样本大 |
| 二值处理 | 支持 | 支持（推荐） |
| 连续处理 | 支持 | 不支持 |

> **注意**：PLM在共同支撑差时收敛于ATO（Average Treatment on Overlap），不是ATE。若目标是ATE，需IRM + 充分共同支撑。

**倾向得分诊断（PLM/IRM共用）：**

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 估计倾向得分
ps_model = LogisticRegression(max_iter=500)
ps_model.fit(df[controls], df['D'])
ps = ps_model.predict_proba(df[controls])[:, 1]

# 检查极端值（缩尾到0.01-0.99）
print(f"PS范围: [{ps.min():.3f}, {ps.max():.3f}]")
print(f"PS < 0.05 比例: {(ps < 0.05).mean():.3f}")
print(f"PS > 0.95 比例: {(ps > 0.95).mean():.3f}")

# 缩尾
ps_trimmed = np.clip(ps, 0.01, 0.99)
```

```r
# R: 倾向得分诊断
library(ggplot2)

ps_model <- glm(D ~ ., data = df[, c("D", controls)], family = binomial)
ps <- predict(ps_model, type = "response")
cat(sprintf("PS范围: [%.3f, %.3f]\n", min(ps), max(ps)))
cat(sprintf("极端值(<0.05 or >0.95): %.1f%%\n", mean(ps < 0.05 | ps > 0.95) * 100))
```

---

### 2.2 Python 代码

#### doubleml — PLR（同质效应）

```python
import numpy as np
import pandas as pd
from doubleml import DoubleMLPLR, DoubleMLData
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV

# 数据格式
dml_data = DoubleMLData(
    df,
    y_col    = 'Y',       # 结果变量
    d_cols   = 'D',       # 处理变量
    x_cols   = controls   # 高维控制变量（list）
)

# ML学习器
ml_l = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
ml_m = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)

# PLR估计（同质效应 → Estimand: ATO）
dml_plr = DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m, n_folds=5)
dml_plr.fit()
print(dml_plr.summary)

# 返回结果dict（不print）
def get_dml_results(model):
    s = model.summary
    return {
        'coef': s['coef'].iloc[0],
        'se':   s['std err'].iloc[0],
        'pval': s['P>|z|'].iloc[0],
        'ci_low':  s['2.5 %'].iloc[0],
        'ci_high': s['97.5 %'].iloc[0],
        'nuisance_rmse': model.params_nuisance  # 诊断
    }

results_plr = get_dml_results(dml_plr)
```

#### doubleml — IRM（异质效应，ATE/ATT）

```python
from doubleml import DoubleMLIRM

# IRM估计（允许异质效应 → Estimand: ATE/ATT/CATE）
ml_g = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
ml_m2 = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)

dml_irm = DoubleMLIRM(
    dml_data,
    ml_g = ml_g,
    ml_m = ml_m2,
    score = 'ATE',     # 'ATE' 或 'ATTE'（ATT）
    n_folds = 5
)
dml_irm.fit()
print(dml_irm.summary)
```

#### econml — LinearDML（X vs W 语义明确）

```python
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# ⚠️ econml的X/W语义：
# X = 异质性特征（低维，用于估计CATE，进入最终回归）
# W = 控制变量（高维，nuisance，不进入最终回归）

# 用法1: 估计ATE（无异质性）
est_ate = LinearDML(
    model_y = RandomForestRegressor(n_estimators=100, random_state=42),
    model_t = RandomForestClassifier(n_estimators=100, random_state=42),
    cv      = 5
)
est_ate.fit(Y=df['Y'], T=df['D'], X=None, W=df[controls])
print(est_ate.ate_)          # ATE点估计
print(est_ate.ate_interval_())  # 置信区间

# 用法2: 估计CATE（含异质性特征X）
hetero_features = ['age', 'size']   # 低维异质性变量
remaining_controls = [c for c in controls if c not in hetero_features]

est_cate = LinearDML(
    model_y = RandomForestRegressor(n_estimators=100, random_state=42),
    model_t = RandomForestClassifier(n_estimators=100, random_state=42),
    cv      = 5
)
est_cate.fit(
    Y = df['Y'],
    T = df['D'],
    X = df[hetero_features],      # 异质性特征
    W = df[remaining_controls]    # 高维控制
)
# CATE预测
cate_pred = est_cate.effect(df[hetero_features])
```

---

### 2.3 R 代码（DoubleML包）

```r
library(DoubleML)
library(mlr3)
library(mlr3learners)

# 数据格式
df_dml <- df[, c("Y", "D", controls)]
dml_data <- DoubleMLData$new(
  data      = df_dml,
  y_col     = "Y",
  d_cols    = "D",
  x_cols    = controls
)

# 学习器
lrn_l <- lrn("regr.ranger",  num.trees = 200, max.depth = 5)
lrn_m <- lrn("classif.ranger", num.trees = 200, max.depth = 5,
              predict_type = "prob")

# PLR（同质效应 → Estimand: ATO）
dml_plr <- DoubleMLPLR$new(dml_data, ml_l = lrn_l, ml_m = lrn_m, n_folds = 5)
dml_plr$fit()
print(dml_plr$summary())

# IRM（异质效应 → Estimand: ATE/ATT）
lrn_g <- lrn("regr.ranger", num.trees = 200, max.depth = 5)
dml_irm <- DoubleMLIRM$new(
  dml_data,
  ml_g   = lrn_g,
  ml_m   = lrn_m,
  score  = "ATE",    # 或 "ATTE"
  n_folds = 5
)
dml_irm$fit()
print(dml_irm$summary())
```

---

### 2.4 敏感性分析（必做）

DoubleML 内置 Chernozhukov et al. (2022) 敏感性分析，评估违反CIA假设时估计稳健性：

```python
# Python: DoubleML敏感性分析
# cf_y: 结果方程中未观测混淆占总变异的比例（0-1）
# cf_d: 处理方程中未观测混淆占总变异的比例（0-1）

dml_plr.sensitivity_analysis(cf_y=0.1, cf_d=0.1)
print(dml_plr.sensitivity_summary)
# 解读：若sensitivity_params中的effect bounds包含0，结论对未观测混淆敏感

# 扫描不同假设
for cf in [0.05, 0.10, 0.15, 0.20]:
    dml_plr.sensitivity_analysis(cf_y=cf, cf_d=cf)
    bounds = dml_plr.sensitivity_params['theta']
    print(f"cf={cf}: bounds=[{bounds['lower']:.3f}, {bounds['upper']:.3f}]")
```

```r
# R: DoubleML敏感性分析
dml_plr$sensitivity_analysis(cf_y = 0.1, cf_d = 0.1)
print(dml_plr$sensitivity_summary())

# 扫描
for (cf in c(0.05, 0.10, 0.15, 0.20)) {
  dml_plr$sensitivity_analysis(cf_y = cf, cf_d = cf)
  cat(sprintf("cf=%.2f: %s\n", cf, dml_plr$sensitivity_summary()$conclusion))
}
```

---

## 3. Causal Forest

### 3.1 Python — econml CausalForestDML

```python
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

cf = CausalForestDML(
    model_y         = RandomForestRegressor(n_estimators=200, random_state=42),
    model_t         = RandomForestClassifier(n_estimators=200, random_state=42),
    n_estimators    = 500,
    min_samples_leaf = 10,
    cv              = 5,
    random_state    = 42
)
cf.fit(Y=df['Y'], T=df['D'], X=df[hetero_features], W=df[remaining_controls])

# CATE预测 + 置信区间
cate       = cf.effect(df[hetero_features])
cate_lb, cate_ub = cf.effect_interval(df[hetero_features], alpha=0.05)

# 异质性特征重要性
feat_imp = cf.feature_importances_
imp_df = pd.DataFrame({'feature': hetero_features, 'importance': feat_imp})
imp_df = imp_df.sort_values('importance', ascending=False)
print(imp_df)
```

### 3.2 R — grf causal_forest

```r
library(grf)

# 准备矩阵
X_mat <- as.matrix(df[, hetero_features])
W_vec <- df$D
Y_vec <- df$Y

# 估计
cf_r <- causal_forest(
  X = X_mat, Y = Y_vec, W = W_vec,
  num.trees        = 2000,
  min.node.size    = 10,
  tune.parameters  = "all",   # 自动调参
  seed             = 42
)

# CATE预测
tau_hat <- predict(cf_r, X_mat, estimate.variance = TRUE)
df$cate      <- tau_hat$predictions
df$cate_se   <- sqrt(tau_hat$variance.estimates)

# 线性投影（检验异质性统计显著性）
blp <- best_linear_projection(cf_r, A = X_mat)
print(blp)
# 若任意系数显著 → 异质性存在

# 变量重要性
vi <- variable_importance(cf_r)
imp_r <- data.frame(feature = hetero_features, importance = vi)
imp_r <- imp_r[order(-imp_r$importance), ]
print(imp_r)
```

---

### 3.3 ⚠️ 面板数据警告

**标准CausalForest假设截面独立同分布数据。面板数据上直接运行会将个体固定效应误认为处理效应异质性，导致CATE估计严重偏误。**

**正确处理方式：先对Y和D做FE demean，再跑CF**

```python
# Python: 面板数据 FE demean → CausalForest
import pandas as pd
import numpy as np
from econml.dml import CausalForestDML

# Step 1: FE demean（去除个体均值）
df['Y_dm'] = df['Y'] - df.groupby('entity_id')['Y'].transform('mean')
df['D_dm'] = df['D'] - df.groupby('entity_id')['D'].transform('mean')
# 若有时间FE：再去除时间均值
df['Y_dm'] = df['Y_dm'] - df.groupby('time')['Y_dm'].transform('mean')
df['D_dm'] = df['D_dm'] - df.groupby('time')['D_dm'].transform('mean')

# Step 2: 在demean后的数据上跑CF（此时不需要再控制FE）
cf_panel = CausalForestDML(n_estimators=500, cv=5, random_state=42)
cf_panel.fit(
    Y = df['Y_dm'],
    T = df['D_dm'],
    X = df[hetero_features],
    W = df[time_varying_controls]   # 时变控制变量（也需demean）
)
```

```r
# R: 面板数据 FE demean → causal_forest
library(dplyr)
library(grf)

# FE demean
df <- df %>%
  group_by(entity_id) %>%
  mutate(Y_dm = Y - mean(Y), D_dm = D - mean(D)) %>%
  group_by(time) %>%
  mutate(Y_dm = Y_dm - mean(Y_dm), D_dm = D_dm - mean(D_dm)) %>%
  ungroup()

cf_panel <- causal_forest(
  X        = as.matrix(df[, hetero_features]),
  Y        = df$Y_dm,
  W        = df$D_dm,
  num.trees = 2000,
  seed      = 42
)
```

---

### 3.4 ⚠️ 红线：CF ≠ 因果识别

```
CF 只在 CIA（条件独立假设）成立时有因果解释。

正确流程：
  DID / IV / RDD 建立因果识别
      ↓
  确认存在处理效应（主效应显著）
      ↓
  CF 探索异质性（谁受益更多？）

错误做法：
  直接对面板数据跑CF，宣称发现了因果异质性
```

---

## 4. LASSO 变量选择

LASSO用于高维控制变量筛选，或作为DML中的ML学习器。

```python
# Python: LASSO变量选择
from sklearn.linear_model import LassoCV
import numpy as np
import pandas as pd

# CV选最优lambda
lasso = LassoCV(cv=5, max_iter=5000, random_state=42)
lasso.fit(df[candidate_controls], df['Y'])

# 选出非零系数变量
selected = [v for v, c in zip(candidate_controls, lasso.coef_) if abs(c) > 1e-8]
print(f"LASSO选出 {len(selected)}/{len(candidate_controls)} 个变量: {selected}")

# 返回结果dict
def lasso_selection(X, y, candidates):
    model = LassoCV(cv=5, max_iter=5000, random_state=42).fit(X, y)
    selected = [v for v, c in zip(candidates, model.coef_) if abs(c) > 1e-8]
    return {'selected': selected, 'lambda': model.alpha_, 'n_selected': len(selected)}
```

```r
# R: LASSO变量选择（glmnet）
library(glmnet)

X_mat <- as.matrix(df[, candidate_controls])
y_vec <- df$Y

# CV选lambda
cv_lasso <- cv.glmnet(X_mat, y_vec, alpha = 1, nfolds = 5)
lambda_1se <- cv_lasso$lambda.1se   # 更保守（推荐）

# 提取选中变量
coefs <- coef(cv_lasso, s = "lambda.1se")
selected_r <- rownames(coefs)[which(abs(coefs) > 1e-8)]
selected_r <- selected_r[selected_r != "(Intercept)"]
cat("选中变量:", paste(selected_r, collapse=", "), "\n")
cat("lambda.1se =", lambda_1se, "\n")
```

**算法选择决策表：**

| 场景 | 推荐学习器 |
|------|-----------|
| 维度远超样本量（p >> N） | LASSO |
| 非线性关系、变量交互多 | 随机森林 |
| 一般情形，维度适中 | 5折CV选择 LASSO vs RF |
| 样本量极大（N > 50k） | XGBoost（速度快） |

---

## 5. DML + 识别框架嵌套

### 5.1 DML + DID

**⚠️ 红线：不能把固定效应哑变量作为高维X放入PLM。**

错误做法：直接将 entity_dummies 放入 `x_cols` → FE哑变量会把真实处理效应吸收进nuisance项。

正确做法：先FE demean，再在demean数据上做DML。

```python
# Python: DML+DID 正确流程
import pandas as pd
from doubleml import DoubleMLPLR, DoubleMLData
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Step 1: FE demean（entity + time）
df['Y_dm'] = (df['Y']
    - df.groupby('entity_id')['Y'].transform('mean')
    - df.groupby('time')['Y'].transform('mean')
    + df['Y'].mean())

df['D_dm'] = (df['D']
    - df.groupby('entity_id')['D'].transform('mean')
    - df.groupby('time')['D'].transform('mean')
    + df['D'].mean())

# Step 2: DML on demeaned data（controls也需demean）
for c in high_dim_controls:
    df[f'{c}_dm'] = (df[c]
        - df.groupby('entity_id')[c].transform('mean')
        - df.groupby('time')[c].transform('mean')
        + df[c].mean())

controls_dm = [f'{c}_dm' for c in high_dim_controls]

dml_data_did = DoubleMLData(df, y_col='Y_dm', d_cols='D_dm', x_cols=controls_dm)
dml_plr_did = DoubleMLPLR(
    dml_data_did,
    ml_l = RandomForestRegressor(n_estimators=200, random_state=42),
    ml_m = RandomForestRegressor(n_estimators=200, random_state=42),
    n_folds = 5
)
dml_plr_did.fit()
print(dml_plr_did.summary)
```

**⚠️ Ashenfelter Dip警告：** DID前若处理组在处理前已有下降趋势（预期处理组提前调整行为），平行趋势可能不成立。需检验事件研究图中处理前系数。

---

### 5.2 DML + IV（DoubleMLIIVM）

```python
# Python: DML + IV（内生处理变量）
from doubleml import DoubleMLIIVM

dml_data_iv = DoubleMLData(
    df,
    y_col    = 'Y',
    d_cols   = 'D',       # 内生处理变量
    z_cols   = 'Z',       # 工具变量
    x_cols   = controls
)

ml_g = RandomForestRegressor(n_estimators=200, random_state=42)
ml_m = RandomForestClassifier(n_estimators=200, random_state=42)
ml_r = RandomForestClassifier(n_estimators=200, random_state=42)

dml_iivm = DoubleMLIIVM(
    dml_data_iv, ml_g=ml_g, ml_m=ml_m, ml_r=ml_r,
    score='LATE',    # Estimand: LATE（工具变量识别的局部平均处理效应）
    n_folds=5
)
dml_iivm.fit()
print(dml_iivm.summary)
```

---

### 5.3 DML + RDD

RDD场景中，DML只对Y做残差化，**不对D做残差化**（D在断点处的跳变是识别来源，不能被ML吸收）。

```python
# Python: DML + RDD（仅对Y残差化）
# 保留断点附近带宽内样本
df_rdd = df[(df['running_var'] >= cutoff - bandwidth) &
            (df['running_var'] <= cutoff + bandwidth)].copy()

# 只对Y做ML残差化
ml_y = RandomForestRegressor(n_estimators=200, random_state=42)
ml_y.fit(df_rdd[controls], df_rdd['Y'])
df_rdd['Y_resid'] = df_rdd['Y'] - ml_y.predict(df_rdd[controls])

# 保留D的跳变（treat = 1 if running_var >= cutoff）
df_rdd['treat'] = (df_rdd['running_var'] >= cutoff).astype(int)

# 在Y残差上做RDD
import statsmodels.formula.api as smf
rdd_dml = smf.ols(
    'Y_resid ~ treat + running_var + treat:running_var',
    data = df_rdd
).fit(cov_type='HC3')
print(rdd_dml.summary())
```

---

## 6. 实践指南

**① 识别优先，估计其次**
先确认因果识别策略（CIA/DID/IV/RDD），再决定是否用DML提升效率。

**② 诊断nuisance模型质量**
报告Y方程和D方程的out-of-fold RMSE。RMSE远高于基准（如均值预测）说明ML未能有效控制混淆。

```python
# 提取nuisance RMSE
rmse_info = dml_plr.params_nuisance
print("Y方程 RMSE:", rmse_info)
```

**③ 报告透明**
- 明确报告：n_folds、学习器类型、超参数
- 敏感性分析：不同学习器（LASSO vs RF）结论是否一致
- 样本量：交叉拟合后每折的有效样本

**④ 结论表述**
不要说"ML发现了因果效应"，应说"在[识别策略]的基础上，使用DML控制高维混淆后，估计的[Estimand]为..."

---

## 7. Estimand 声明

| 方法 | Estimand | 声明要点 |
|------|----------|---------|
| DML PLM | **ATO**（不是ATE！） | 共同支撑差时收敛于ATO；如需ATE须用IRM |
| DML IRM score='ATE' | ATE | 需充分共同支撑；报告倾向得分分布 |
| DML IRM score='ATTE' | ATT | 处理组在对照组支撑内 |
| DML IIVM | LATE（工具变量Compliers） | 同IV；描述Complier特征 |
| Causal Forest | CATE（条件ATE） | 仅CIA下有因果解释；先建立主效应 |

**声明模板：**
> "本文在[识别策略]成立的前提下，采用DML-PLR估计核心参数。DML利用随机森林控制高维控制变量，通过5折交叉拟合避免过拟合偏误。在共同支撑域上，该估计量收敛于ATO（Average Treatment on Overlap）。敏感性分析（Chernozhukov et al., 2022）显示，在cf_y=cf_d=0.10的未观测混淆假设下，结论依然稳健。"

---

## DynamicDML（进阶可选，按需加载）

> **注意：** 本节为进阶内容。仅在面板数据中处理变量随时间变化、需要估计动态处理效应时使用。

动态DML用于面板数据中的动态处理效应估计（Chernozhukov et al., 2022 Dynamic Panel）。

### API 关键修复：长格式向量 + groups参数

**❌ 错误做法（矩阵格式）：**
```python
# 错误：传入 (N×T) 矩阵
est.fit(Y=panel_matrix, T=treatment_matrix)
```

**✅ 正确做法（长格式向量 + groups）：**
```python
from econml.panel.dml import DynamicDML
from sklearn.ensemble import RandomForestRegressor

# 数据必须按 (entity_id, time) 排序
panel = df.sort_values(['unit_id', 'time']).reset_index(drop=True)

est = DynamicDML(
    model_y = RandomForestRegressor(n_estimators=200, random_state=42),
    model_t = RandomForestRegressor(n_estimators=200, random_state=42),
    cv      = 3
)

# 正确API：长格式一维向量 + groups标识个体
est.fit(
    Y      = panel['Y'].values,       # 长格式向量，shape=(N*T,)
    T      = panel['D'].values,       # 长格式向量，shape=(N*T,)
    X      = panel[hetero_features].values if hetero_features else None,
    W      = panel[controls].values,
    groups = panel['unit_id'].values   # ⚠️ 关键：个体ID标识
)

# 动态效应：lag=0当期，lag=1滞后一期，...
effects = est.const_marginal_effect_inference(
    X = panel[hetero_features].values[:panel['unit_id'].nunique()] if hetero_features else None
)
print(effects.summary_frame())
```

**groups参数说明：**
- 必须传入；标识哪些行属于同一个体
- 确保同一个体的所有观测连续排列
- 时间期数T必须对每个个体相同（平衡面板）
