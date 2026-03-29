# 数据审计检查清单

## 1. 数据加载与基础信息

```python
import pandas as pd
import numpy as np

# 加载（根据格式调整）
# df = pd.read_csv("data.csv")
# df = pd.read_excel("data.xlsx")
# df = pd.read_stata("data.dta")

print(f"样本量: {len(df)}")
print(f"变量数: {len(df.columns)}")
print(f"内存占用: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"\n数据类型分布:")
print(df.dtypes.value_counts())
```

## 2. 面板结构诊断

```python
def diagnose_panel(df, entity_var, time_var):
    """面板结构全面诊断"""
    n_entities = df[entity_var].nunique()
    n_periods = df[time_var].nunique()
    n_obs = len(df)
    
    expected_balanced = n_entities * n_periods
    balanced = (n_obs == expected_balanced)
    
    periods_per_entity = df.groupby(entity_var)[time_var].count()
    
    print(f"=== 面板结构 ===")
    print(f"截面单位数: {n_entities}")
    print(f"时间期数: {n_periods}")
    print(f"总观测数: {n_obs}")
    print(f"平衡面板预期: {expected_balanced}")
    print(f"是否平衡: {balanced}")
    print(f"平均每单位期数: {periods_per_entity.mean():.1f}")
    print(f"最少期数: {periods_per_entity.min()}")
    print(f"最多期数: {periods_per_entity.max()}")
    
    if not balanced:
        print(f"\n不平衡原因:")
        print(f"  缺失观测: {expected_balanced - n_obs}")
        print(f"  期数<平均的单位比例: {(periods_per_entity < periods_per_entity.mean()).mean():.1%}")
    
    # 时间连续性检查
    time_gaps = df.groupby(entity_var)[time_var].apply(
        lambda x: x.sort_values().diff().dropna()
    )
    if time_gaps.nunique() > 1:
        print(f"\n⚠️ 存在不等间距的时间间隔")
    
    return {
        'n_entities': n_entities,
        'n_periods': n_periods,
        'balanced': balanced,
        'periods_per_entity': periods_per_entity
    }
```

## 3. 缺失值分析

```python
def analyze_missing(df, threshold=10):
    """缺失值分析"""
    missing = df.isnull().sum()
    missing_pct = missing / len(df) * 100
    
    print("=== 缺失值分析 ===")
    print(f"无缺失变量: {(missing == 0).sum()}")
    print(f"缺失率 < 5%: {((missing_pct > 0) & (missing_pct < 5)).sum()}")
    print(f"缺失率 5-10%: {((missing_pct >= 5) & (missing_pct < 10)).sum()}")
    print(f"缺失率 10-30%: {((missing_pct >= 10) & (missing_pct < 30)).sum()}")
    print(f"缺失率 ≥ 30%: {(missing_pct >= 30).sum()}")
    
    high_missing = missing_pct[missing_pct >= threshold].sort_values(ascending=False)
    if len(high_missing) > 0:
        print(f"\n⚠️ 缺失率 ≥ {threshold}% 的变量:")
        for var, pct in high_missing.items():
            print(f"  {var}: {pct:.1f}%")
    
    return missing_pct
```

## 4. 变量分布诊断

```python
def diagnose_distributions(df, numeric_vars):
    """数值变量分布诊断"""
    stats = []
    for col in numeric_vars:
        s = df[col].dropna()
        stats.append({
            'variable': col,
            'n': len(s),
            'mean': s.mean(),
            'sd': s.std(),
            'min': s.min(),
            'p25': s.quantile(0.25),
            'median': s.median(),
            'p75': s.quantile(0.75),
            'max': s.max(),
            'skewness': s.skew(),
            'kurtosis': s.kurtosis(),
            'zeros_pct': (s == 0).mean() * 100,
            'outliers_iqr': ((s < s.quantile(0.25) - 1.5*(s.quantile(0.75)-s.quantile(0.25))) |
                            (s > s.quantile(0.75) + 1.5*(s.quantile(0.75)-s.quantile(0.25)))).mean() * 100
        })
    
    return pd.DataFrame(stats).set_index('variable')
```

## 5. 组内变异检查（FE 可行性）

```python
def check_within_variation(df, vars_to_check, entity_var):
    """检查组内变异——决定固定效应模型可行性"""
    results = []
    for var in vars_to_check:
        total_var = df[var].var()
        within = df.groupby(entity_var)[var].transform(lambda x: x - x.mean()).var()
        between = df.groupby(entity_var)[var].mean().var()
        
        within_share = within / total_var if total_var > 0 else 0
        
        results.append({
            'variable': var,
            'total_var': total_var,
            'within_var': within,
            'between_var': between,
            'within_share': within_share,
            'fe_feasible': '✅' if within_share > 0.1 else '⚠️ 低组内变异'
        })
    
    return pd.DataFrame(results).set_index('variable')
```

## 6. 处理变量诊断

```python
def diagnose_treatment(df, treat_var, entity_var, time_var):
    """处理变量诊断"""
    print("=== 处理变量诊断 ===")
    
    # 处理/对照分布
    treat_counts = df[treat_var].value_counts()
    print(f"处理组: {treat_counts.get(1, 0)} obs")
    print(f"对照组: {treat_counts.get(0, 0)} obs")
    print(f"处理比例: {df[treat_var].mean():.1%}")
    
    # 按时间的处理比例
    treat_by_time = df.groupby(time_var)[treat_var].mean()
    print(f"\n各期处理比例:")
    for t, p in treat_by_time.items():
        marker = " ←" if p > 0 and treat_by_time.shift(1).get(t, 0) == 0 else ""
        print(f"  {t}: {p:.1%}{marker}")
    
    # 处理组单位数
    treat_entities = df[df[treat_var] == 1][entity_var].nunique()
    control_entities = df[df[treat_var] == 0][entity_var].nunique()
    print(f"\n处理组单位: {treat_entities}")
    print(f"对照组单位: {control_entities}")
```

## 7. 审计报告模板

生成的 `data_audit_report.md` 应包含以下章节：

```markdown
# 数据审计报告

## 1. 数据概览
- 来源、采集方式、时间范围
- 样本量、变量数

## 2. 面板结构
- 截面/时间维度
- 平衡性
- 时间连续性

## 3. 变量字典
| 变量名 | 类型 | 描述 | 缺失率 | 均值 | 标准差 | 最小值 | 最大值 |
|-------|------|------|--------|------|--------|-------|-------|

## 4. 数据质量问题
- 缺失值模式
- 异常值
- 重复观测

## 5. 研究潜力评估
- 适合的计量方法
- 可能的因变量/自变量
- 数据限制

## 6. 建议
- 数据清洗建议
- 变量转换建议
- 样本限制建议
```
