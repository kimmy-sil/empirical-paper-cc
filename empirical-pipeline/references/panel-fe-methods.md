# 面板数据固定效应方法指南

## 1. 双向固定效应 (Two-Way Fixed Effects, TWFE)

### 估计方程

```
Y_it = α_i + δ_t + β X_it + ε_it
```

- α_i: 个体固定效应（控制不随时间变化的不可观测因素）
- δ_t: 时间固定效应（控制共同的时间趋势）

### Python 实现

```python
from linearmodels.panel import PanelOLS, RandomEffects, compare

def run_panel_fe(df, y_var, x_vars, entity_var='id', time_var='year',
                 entity_effects=True, time_effects=True, cluster_var=None):
    """面板固定效应回归"""
    
    panel_df = df.set_index([entity_var, time_var])
    
    model = PanelOLS(
        dependent=panel_df[y_var],
        exog=panel_df[x_vars],
        entity_effects=entity_effects,
        time_effects=time_effects
    )
    
    if cluster_var:
        result = model.fit(cov_type='clustered', clusters=panel_df[cluster_var])
    else:
        result = model.fit(cov_type='clustered', cluster_entity=True)
    
    return result


def hausman_test(df, y_var, x_vars, entity_var='id', time_var='year'):
    """Hausman 检验：FE vs RE"""
    
    panel_df = df.set_index([entity_var, time_var])
    
    # 固定效应
    fe_model = PanelOLS(
        dependent=panel_df[y_var],
        exog=panel_df[x_vars],
        entity_effects=True
    )
    fe_result = fe_model.fit()
    
    # 随机效应
    re_model = RandomEffects(
        dependent=panel_df[y_var],
        exog=sm.add_constant(panel_df[x_vars])
    )
    re_result = re_model.fit()
    
    # Hausman 统计量
    b_fe = fe_result.params
    b_re = re_result.params[x_vars]
    
    diff = b_fe - b_re
    var_diff = fe_result.cov - re_result.cov.loc[x_vars, x_vars]
    
    from scipy import stats as scipy_stats
    hausman_stat = diff @ np.linalg.inv(var_diff) @ diff
    p_value = scipy_stats.chi2.sf(hausman_stat, len(x_vars))
    
    return {
        'statistic': hausman_stat,
        'p_value': p_value,
        'recommendation': 'Fixed Effects' if p_value < 0.05 else 'Random Effects'
    }


def within_variation_check(df, x_vars, entity_var='id'):
    """检查组内变异（within variation）"""
    
    results = {}
    for var in x_vars:
        total_var = df[var].var()
        within_var = df.groupby(entity_var)[var].transform(
            lambda x: x - x.mean()
        ).var()
        between_var = df.groupby(entity_var)[var].mean().var()
        
        results[var] = {
            'total_variance': total_var,
            'within_variance': within_var,
            'between_variance': between_var,
            'within_share': within_var / total_var if total_var > 0 else 0,
            'warning': '⚠️ 组内变异过低，FE估计可能不可靠' if within_var / total_var < 0.1 else '✅'
        }
    
    return pd.DataFrame(results).T
```

### R 实现

```r
library(fixest)
library(plm)

# 双向固定效应
fe_model <- feols(y ~ x1 + x2 | entity + year, data = df, cluster = ~entity)

# Hausman 检验
fe <- plm(y ~ x1 + x2, data = pdf, model = "within", index = c("entity", "year"))
re <- plm(y ~ x1 + x2, data = pdf, model = "random", index = c("entity", "year"))
phtest(fe, re)

# 逐步加固定效应
m1 <- feols(y ~ x1 + x2, data = df)                          # OLS
m2 <- feols(y ~ x1 + x2 | entity, data = df)                 # 个体FE
m3 <- feols(y ~ x1 + x2 | entity + year, data = df)          # 双向FE
m4 <- feols(y ~ x1 + x2 | entity + year + province^year, data = df)  # 加省份×年份

etable(m1, m2, m3, m4, cluster = ~entity)
```

## 2. 常用面板模型变体

### 2.1 带有个体时间趋势

```
Y_it = α_i + δ_t + λ_i · t + β X_it + ε_it
```

控制每个个体的线性时间趋势。

### 2.2 高维固定效应

```r
# 省份×年份固定效应（吸收省份层面的时变冲击）
feols(y ~ x | entity + province^year, data = df)

# 行业×年份固定效应
feols(y ~ x | firm + industry^year, data = df)
```

### 2.3 动态面板 (GMM)

适用于因变量的滞后项作为解释变量时：

```r
library(plm)
# Arellano-Bond GMM
gmm_model <- pgmm(
  y ~ lag(y, 1) + x1 + x2 | lag(y, 2:99),
  data = pdf, 
  effect = "twoways",
  model = "twosteps"
)
```

## 3. 面板数据诊断

```python
def panel_diagnostics(df, y_var, x_vars, entity_var='id', time_var='year'):
    """面板数据全面诊断"""
    
    diagnostics = {}
    
    # 1. 面板结构
    n_entities = df[entity_var].nunique()
    n_periods = df[time_var].nunique()
    n_obs = len(df)
    balanced = (n_obs == n_entities * n_periods)
    
    diagnostics['structure'] = {
        'n_entities': n_entities,
        'n_periods': n_periods,
        'n_obs': n_obs,
        'balanced': balanced,
        'avg_periods_per_entity': df.groupby(entity_var)[time_var].count().mean()
    }
    
    # 2. 缺失值模式
    missing = df[x_vars + [y_var]].isnull().sum() / len(df) * 100
    diagnostics['missing'] = missing.to_dict()
    
    # 3. 组内变异
    diagnostics['within_variation'] = {}
    for var in x_vars:
        within = df.groupby(entity_var)[var].transform(lambda x: x - x.mean()).var()
        total = df[var].var()
        diagnostics['within_variation'][var] = within / total if total > 0 else 0
    
    # 4. 序列相关检验提示
    diagnostics['notes'] = [
        "建议检验残差序列相关 (Wooldridge test)",
        "建议检验截面相关 (Pesaran CD test)",
        "如有异方差，使用聚类稳健标准误"
    ]
    
    return diagnostics
```

## 4. 标准误选择指南

| 场景 | 推荐标准误 | 代码示例 |
|------|-----------|---------|
| 一般面板 | 个体聚类 | `cluster = ~entity` |
| 政策在省级实施 | 省级聚类 | `cluster = ~province` |
| 聚类数 < 50 | Wild cluster bootstrap | `boottest::boottest()` |
| 时间序列相关 | Newey-West | `vcov = 'NW'` |
| 截面相关 | Driscoll-Kraay | `vcov = 'DK'` |
| 多维聚类 | 双向聚类 | `cluster = ~entity + year` |
