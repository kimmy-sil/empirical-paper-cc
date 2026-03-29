# DID 方法论详细指南

## 1. 经典双重差分 (Classic 2×2 DID)

### 估计方程

```
Y_it = α + β₁ Treat_i + β₂ Post_t + δ (Treat_i × Post_t) + X'_it γ + ε_it
```

其中 δ 是政策效应（ATT）。

### Python 实现

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt

def run_classic_did(df, y_var, treat_var, post_var, controls=None, 
                     entity_var='id', time_var='year', cluster_var=None):
    """
    经典 2×2 DID 分析
    
    Parameters:
    -----------
    df : DataFrame - 面板数据
    y_var : str - 因变量名
    treat_var : str - 处理组虚拟变量名
    post_var : str - 政策后虚拟变量名
    controls : list - 控制变量列表
    entity_var : str - 个体标识变量
    time_var : str - 时间变量
    cluster_var : str - 聚类变量（默认为 entity_var）
    """
    
    # 构造交互项
    df['treat_post'] = df[treat_var] * df[post_var]
    
    # 准备自变量
    exog_vars = ['treat_post']
    if controls:
        exog_vars += controls
    
    # 设置面板结构
    panel_df = df.set_index([entity_var, time_var])
    
    # TWFE 估计
    model = PanelOLS(
        dependent=panel_df[y_var],
        exog=panel_df[exog_vars],
        entity_effects=True,
        time_effects=True,
        check_rank=False
    )
    
    cluster = cluster_var if cluster_var else entity_var
    result = model.fit(cov_type='clustered', cluster_entity=True)
    
    return result


def run_event_study(df, y_var, treat_var, time_var, event_time_var,
                    entity_var='id', controls=None, 
                    leads=4, lags=4, omit=-1):
    """
    事件研究（平行趋势检验 + 动态效应）
    
    Parameters:
    -----------
    event_time_var : str - 相对事件时间变量（event_time = year - treatment_year）
    leads : int - 事前期数
    lags : int - 事后期数
    omit : int - 基准期（通常为 -1）
    """
    
    # 生成事件时间虚拟变量
    for t in range(-leads, lags + 1):
        if t == omit:
            continue
        col_name = f'event_t{t}' if t < 0 else f'event_t_plus_{t}'
        df[col_name] = ((df[event_time_var] == t) & (df[treat_var] == 1)).astype(int)
    
    # 事件时间变量列表
    event_vars = [c for c in df.columns if c.startswith('event_t')]
    
    exog_vars = event_vars
    if controls:
        exog_vars += controls
    
    panel_df = df.set_index([entity_var, time_var])
    
    model = PanelOLS(
        dependent=panel_df[y_var],
        exog=panel_df[exog_vars],
        entity_effects=True,
        time_effects=True
    )
    result = model.fit(cov_type='clustered', cluster_entity=True)
    
    return result


def plot_event_study(result, leads=4, lags=4, omit=-1, title="Event Study"):
    """绘制事件研究图"""
    
    coefs = []
    ses = []
    times = []
    
    for t in range(-leads, lags + 1):
        if t == omit:
            coefs.append(0)
            ses.append(0)
            times.append(t)
            continue
        
        var_name = f'event_t{t}' if t < 0 else f'event_t_plus_{t}'
        if var_name in result.params.index:
            coefs.append(result.params[var_name])
            ses.append(result.std_errors[var_name])
            times.append(t)
    
    coefs = np.array(coefs)
    ses = np.array(ses)
    times = np.array(times)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(times, coefs, yerr=1.96*ses, fmt='o-', capsize=3,
                color='#2c3e50', markersize=6)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(x=-0.5, color='red', linestyle='--', alpha=0.5, label='政策实施')
    ax.set_xlabel('相对事件时间', fontsize=12)
    ax.set_ylabel('估计系数', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig('figures/event_study.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig


def run_placebo_test(df, y_var, treat_var, entity_var, time_var,
                     true_event_year, placebo_years, controls=None):
    """
    安慰剂检验：在真实政策前的不同年份假设政策发生
    """
    results = {}
    
    for placebo_year in placebo_years:
        df_temp = df[df[time_var] < true_event_year].copy()
        df_temp['placebo_post'] = (df_temp[time_var] >= placebo_year).astype(int)
        df_temp['placebo_treat_post'] = df_temp[treat_var] * df_temp['placebo_post']
        
        exog_vars = ['placebo_treat_post']
        if controls:
            exog_vars += controls
        
        panel_df = df_temp.set_index([entity_var, time_var])
        
        model = PanelOLS(
            dependent=panel_df[y_var],
            exog=panel_df[exog_vars],
            entity_effects=True,
            time_effects=True
        )
        result = model.fit(cov_type='clustered', cluster_entity=True)
        
        results[placebo_year] = {
            'coef': result.params['placebo_treat_post'],
            'se': result.std_errors['placebo_treat_post'],
            'pvalue': result.pvalues['placebo_treat_post']
        }
    
    return results
```

### R 实现

```r
library(fixest)
library(did)
library(ggplot2)

# === 经典 TWFE DID ===
did_twfe <- feols(
  y ~ treat_post | entity + year,
  data = df,
  cluster = ~entity
)
summary(did_twfe)
etable(did_twfe)  # 格式化输出

# === 事件研究 ===
event_study <- feols(
  y ~ i(event_time, treat, ref = -1) | entity + year,
  data = df,
  cluster = ~entity
)
iplot(event_study, main = "Event Study")  # 自动绘图

# === Callaway & Sant'Anna (2021) 交错 DID ===
cs_result <- att_gt(
  yname = "y",
  tname = "year", 
  idname = "id",
  gname = "first_treat_year",  # 首次处理年份（未处理=0）
  data = df,
  control_group = "notyettreated",  # 推荐
  anticipation = 0,
  est_method = "dr"  # doubly robust
)

# 汇总结果
summary(cs_result)
ggdid(cs_result)  # 可视化

# 聚合为整体 ATT
agg_simple <- aggte(cs_result, type = "simple")
agg_dynamic <- aggte(cs_result, type = "dynamic")  # 动态效应
ggdid(agg_dynamic)

# === Sun & Abraham (2021) ===
library(sunab)
sa_result <- feols(
  y ~ sunab(first_treat_year, year) | entity + year,
  data = df,
  cluster = ~entity
)
iplot(sa_result)
```

## 2. 交错 DID 注意事项

当不同个体在不同时间接受处理（staggered adoption）时，经典 TWFE 估计可能有偏。

### 问题来源
- TWFE 使用已处理单元作为对照（bad comparison）
- 处理效应异质性导致权重为负
- Goodman-Bacon (2021) 分解揭示了这一问题

### 推荐方法

| 方法 | 论文 | R 包 | Python |
|------|------|------|--------|
| Callaway & Sant'Anna | CS (2021) | `did` | `csdid` |
| Sun & Abraham | SA (2021) | `fixest::sunab` | — |
| Borusyak et al. | BJS (2024) | `did2s` | — |
| de Chaisemartin & D'Haultfoeuille | dCDH (2020) | `DIDmultiplegt` | — |
| Gardner | Gardner (2022) | `did2s` | — |

### 选择建议
- **默认推荐**: Callaway & Sant'Anna (2021)，最灵活，支持 DR 估计
- **简洁报告**: Sun & Abraham (2021)，可直接用 `fixest` 实现
- **稳健性**: 同时报告 2-3 种方法的结果

## 3. 三重差分 (DDD / Triple Difference)

### 适用场景
- 政策只影响特定行业/群体
- 需要额外的对照维度增强识别
- 平行趋势假设在 DID 层面不够可信

### 估计方程

```
Y_ijt = β₀ + β₁(Treat_i × Affected_j × Post_t)
      + β₂(Treat_i × Affected_j)
      + β₃(Treat_i × Post_t)  
      + β₄(Affected_j × Post_t)
      + γ_i + δ_j + τ_t + ε_ijt
```

β₁ 是政策效应。

## 4. 回归结果表格生成

```python
def format_regression_table(results_list, model_names, dep_var_name,
                            coef_vars=None, stats=['nobs', 'r2_within']):
    """
    生成标准回归结果表（Markdown格式）
    
    Parameters:
    -----------
    results_list : list - 回归结果对象列表
    model_names : list - 各列模型名称
    dep_var_name : str - 因变量名称
    coef_vars : list - 要展示的系数（默认全部）
    """
    
    header = f"| | {'  |  '.join(model_names)} |\n"
    header += f"|{'---|' * (len(model_names) + 1)}\n"
    header = f"**因变量: {dep_var_name}**\n\n" + header
    
    rows = []
    
    # 确定要展示的变量
    if coef_vars is None:
        coef_vars = list(results_list[0].params.index)
    
    for var in coef_vars:
        coef_row = f"| {var} |"
        se_row = f"| |"
        for res in results_list:
            if var in res.params.index:
                coef = res.params[var]
                se = res.std_errors[var]
                pval = res.pvalues[var]
                stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
                coef_row += f" {coef:.4f}{stars} |"
                se_row += f" ({se:.4f}) |"
            else:
                coef_row += " — |"
                se_row += " |"
        rows.append(coef_row)
        rows.append(se_row)
    
    # 统计量
    rows.append(f"|{'---|' * (len(model_names) + 1)}")
    for stat in stats:
        stat_row = f"| {stat} |"
        for res in results_list:
            if stat == 'nobs':
                stat_row += f" {res.nobs:,} |"
            elif stat == 'r2_within':
                stat_row += f" {res.rsquared_within:.4f} |"
        rows.append(stat_row)
    
    table = header + '\n'.join(rows)
    return table
```

## 5. 常见陷阱

1. **不要在交错 DID 中盲目使用 TWFE** — 必须检查或使用新方法
2. **标准误聚类层级要匹配处理层级** — 如政策在省级实施，至少聚类到省级
3. **少数聚类（< 50）** — 使用 Wild cluster bootstrap
4. **平行趋势≠等水平** — 只需趋势平行，不需水平相同
5. **不要 p-hacking** — 预注册分析计划，报告所有预设检验
6. **处理"坏控制"** — 不要加入可能受处理影响的中介变量作为控制变量
7. **注意预期效应（Anticipation）** — 如果政策提前公布，处理组可能提前反应
