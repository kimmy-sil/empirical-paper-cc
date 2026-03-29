# 其他计量方法参考

## 1. 断点回归 (RDD)

### Sharp RDD

处理分配完全由运行变量 X 在断点 c 处决定：T_i = 1{X_i ≥ c}

```python
from rdrobust import rdrobust, rdplot, rdbwselect

# 最优带宽选择 + 局部多项式估计
result = rdrobust(Y, X, c=cutoff)
print(result)

# RDD 图
rdplot(Y, X, c=cutoff, title="RDD Plot")
```

```r
library(rdrobust)

# 主估计
rd <- rdrobust(y, x, c = cutoff)
summary(rd)

# 可视化
rdplot(y, x, c = cutoff)

# McCrary 密度检验（检查操纵）
library(rddensity)
density_test <- rddensity(x, c = cutoff)
summary(density_test)
```

### RDD 检验清单

- [ ] McCrary 密度检验（无操纵）
- [ ] 协变量在断点处平衡检验
- [ ] 带宽敏感性（0.5h, h, 1.5h, 2h）
- [ ] 多项式阶数敏感性（1阶 vs 2阶）
- [ ] 安慰剂断点检验
- [ ] Donut hole 检验（排除断点附近观测）

### Fuzzy RDD

处理分配在断点处不完全遵从：

```r
# Fuzzy RDD = IV at the cutoff
rd_fuzzy <- rdrobust(y, x, c = cutoff, fuzzy = treatment)
summary(rd_fuzzy)
```

---

## 2. 工具变量 (IV / 2SLS)

### 估计方程

```
第一阶段: X_it = π Z_it + W'_it γ + ε_it
第二阶段: Y_it = β X̂_it + W'_it δ + u_it
```

### Python 实现

```python
from linearmodels.iv import IV2SLS

# 2SLS
model = IV2SLS(
    dependent=df['y'],
    exog=df[controls],
    endog=df['x_endog'],
    instruments=df['z_instrument']
)
result = model.fit(cov_type='clustered', clusters=df['cluster_id'])
print(result)

# 第一阶段 F 统计量
print(f"First-stage F: {result.first_stage.diagnostics['f.stat']['x_endog']:.2f}")
```

### R 实现

```r
library(fixest)

# 2SLS with fixest
iv_model <- feols(y ~ controls | entity + year | x_endog ~ z_instrument,
                  data = df, cluster = ~entity)
summary(iv_model)
fitstat(iv_model, "ivf")  # 第一阶段 F

# 经典 ivreg
library(ivreg)
iv2 <- ivreg(y ~ x_endog + controls | z_instrument + controls, data = df)
summary(iv2, diagnostics = TRUE)
```

### IV 检验清单

- [ ] 第一阶段 F > 10（Staiger-Stock 规则；更严格用 F > 104.7 for 5% bias）
- [ ] 排他性约束论证（理论层面）
- [ ] 过度识别检验（Hansen J，如有多个 IV）
- [ ] 弱工具变量稳健推断（Anderson-Rubin test）
- [ ] 对比 OLS 结果，讨论内生性方向

---

## 3. 倾向得分匹配 (PSM)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# 估计倾向得分
psm = LogisticRegression()
psm.fit(df[covariates], df['treatment'])
df['pscore'] = psm.predict_proba(df[covariates])[:, 1]

# 最近邻匹配
treated = df[df['treatment'] == 1]
control = df[df['treatment'] == 0]

nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(control[['pscore']])
distances, indices = nn.kneighbors(treated[['pscore']])

# 匹配后 ATT
matched_control = control.iloc[indices.flatten()]
att = treated['y'].mean() - matched_control['y'].values.mean()
```

### PSM 注意事项

- PSM 本身不解决内生性问题，仅控制可观测混杂
- 通常与 DID 结合使用（PSM-DID）
- 必须报告匹配前后协变量平衡表
- 建议使用多种匹配方法做稳健性（1:1, 1:k, caliper, kernel）

---

## 4. 合成控制法 (Synthetic Control)

```r
library(Synth)

# 准备数据
dataprep.out <- dataprep(
  foo = df,
  predictors = c("gdp", "pop", "education"),
  predictors.op = "mean",
  dependent = "outcome",
  unit.variable = "id",
  time.variable = "year",
  treatment.identifier = treated_unit,
  controls.identifier = control_units,
  time.predictors.prior = pre_years,
  time.optimize.ssr = pre_years,
  time.plot = all_years
)

synth.out <- synth(data.prep.obj = dataprep.out)
path.plot(synth.out, dataprep.out)
gaps.plot(synth.out, dataprep.out)
```

### 合成控制检验

- [ ] 合成控制组对处理前结果的拟合度（RMSPE）
- [ ] 安慰剂检验（对每个对照单位假设为处理单位）
- [ ] 权重分布（不应过度集中于少数对照单位）
- [ ] Leave-one-out 稳健性
