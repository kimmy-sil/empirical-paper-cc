#!/usr/bin/env python3
"""
DID 分析模板脚本
使用方法: 根据实际数据修改变量名和参数后执行
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置区 — 根据实际研究修改以下变量名
# ============================================================

DATA_PATH = "data.csv"           # 数据文件路径
Y_VAR = "outcome"                # 因变量
TREAT_VAR = "treated"            # 处理组虚拟变量 (0/1)
POST_VAR = "post"                # 政策后虚拟变量 (0/1)
ENTITY_VAR = "entity_id"         # 个体标识
TIME_VAR = "year"                # 时间变量
CLUSTER_VAR = "entity_id"        # 聚类变量
EVENT_TIME_VAR = "event_time"    # 相对事件时间 (year - treatment_year)
CONTROLS = ["control1", "control2", "control3"]  # 控制变量列表

OUTPUT_DIR = "output"

# ============================================================
# 数据加载与准备
# ============================================================

def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df['treat_post'] = df[TREAT_VAR] * df[POST_VAR]
    print(f"样本量: {len(df)}")
    print(f"处理组: {df[TREAT_VAR].sum()} obs ({df[TREAT_VAR].mean():.1%})")
    print(f"Post期: {df[POST_VAR].sum()} obs")
    return df

# ============================================================
# Table 1: 描述性统计
# ============================================================

def descriptive_stats(df):
    """分组描述性统计"""
    vars_to_describe = [Y_VAR] + CONTROLS
    
    treated = df[df[TREAT_VAR] == 1][vars_to_describe].describe().T
    control = df[df[TREAT_VAR] == 0][vars_to_describe].describe().T
    
    table1 = pd.DataFrame({
        'N(处理)': treated['count'],
        'Mean(处理)': treated['mean'],
        'SD(处理)': treated['std'],
        'N(对照)': control['count'],
        'Mean(对照)': control['mean'],
        'SD(对照)': control['std'],
    })
    
    # 均值差异检验
    from scipy import stats
    for var in vars_to_describe:
        t, p = stats.ttest_ind(
            df[df[TREAT_VAR]==1][var].dropna(),
            df[df[TREAT_VAR]==0][var].dropna()
        )
        table1.loc[var, 'Diff p-value'] = p
    
    table1.to_markdown(f"{OUTPUT_DIR}/tables/table1_descriptive.md")
    print("Table 1 saved.")
    return table1

# ============================================================
# 主回归: TWFE DID
# ============================================================

def main_regression(df):
    """主回归（逐步加固定效应和控制变量）"""
    panel_df = df.set_index([ENTITY_VAR, TIME_VAR])
    results = []
    
    # Model 1: 基础 DID (entity + time FE)
    m1 = PanelOLS(
        panel_df[Y_VAR],
        panel_df[['treat_post']],
        entity_effects=True, time_effects=True
    ).fit(cov_type='clustered', cluster_entity=True)
    results.append(('(1) Baseline', m1))
    
    # Model 2: + 控制变量
    m2 = PanelOLS(
        panel_df[Y_VAR],
        panel_df[['treat_post'] + CONTROLS],
        entity_effects=True, time_effects=True
    ).fit(cov_type='clustered', cluster_entity=True)
    results.append(('(2) + Controls', m2))
    
    # 输出结果
    print("\n=== 主回归结果 ===")
    for name, res in results:
        coef = res.params['treat_post']
        se = res.std_errors['treat_post']
        pval = res.pvalues['treat_post']
        stars = '***' if pval<0.01 else ('**' if pval<0.05 else ('*' if pval<0.1 else ''))
        print(f"{name}: β = {coef:.4f}{stars} (SE = {se:.4f}), N = {res.nobs}")
    
    return results

# ============================================================
# 事件研究 (平行趋势检验)
# ============================================================

def event_study(df, leads=4, lags=4, omit=-1):
    """事件研究图"""
    df = df.copy()
    
    # 生成事件时间虚拟变量
    event_vars = []
    for t in range(-leads, lags + 1):
        if t == omit:
            continue
        col = f'et_{t}' if t < 0 else f'et_p{t}'
        df[col] = ((df[EVENT_TIME_VAR] == t) & (df[TREAT_VAR] == 1)).astype(int)
        event_vars.append(col)
    
    panel_df = df.set_index([ENTITY_VAR, TIME_VAR])
    
    model = PanelOLS(
        panel_df[Y_VAR],
        panel_df[event_vars + CONTROLS],
        entity_effects=True, time_effects=True
    ).fit(cov_type='clustered', cluster_entity=True)
    
    # 提取系数和置信区间
    times, coefs, ci_low, ci_high = [], [], [], []
    for t in range(-leads, lags + 1):
        times.append(t)
        if t == omit:
            coefs.append(0); ci_low.append(0); ci_high.append(0)
            continue
        col = f'et_{t}' if t < 0 else f'et_p{t}'
        c = model.params[col]
        s = model.std_errors[col]
        coefs.append(c); ci_low.append(c - 1.96*s); ci_high.append(c + 1.96*s)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(times, ci_low, ci_high, alpha=0.15, color='#2c3e50')
    ax.plot(times, coefs, 'o-', color='#2c3e50', markersize=6, linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(omit + 0.5, color='#e74c3c', linestyle='--', alpha=0.5, label='Policy')
    ax.set_xlabel('Relative Time', fontsize=12)
    ax.set_ylabel('Coefficient', fontsize=12)
    ax.set_title('Event Study: Parallel Trends Test', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figures/fig2_event_study.png', dpi=300)
    plt.close()
    
    print("Event study plot saved.")
    
    # 预处理系数联合检验
    pre_vars = [v for v in event_vars if v.startswith('et_')]
    pre_coefs = [model.params[v] for v in pre_vars]
    pre_pvals = [model.pvalues[v] for v in pre_vars]
    print(f"Pre-trend coefficients: {[f'{c:.4f}' for c in pre_coefs]}")
    print(f"Pre-trend p-values: {[f'{p:.4f}' for p in pre_pvals]}")
    
    return model

# ============================================================
# 安慰剂检验
# ============================================================

def placebo_test(df, true_event_year, placebo_years):
    """安慰剂检验: 在真实政策前假设不同时点"""
    df_pre = df[df[TIME_VAR] < true_event_year].copy()
    
    results = {}
    for py in placebo_years:
        df_pre['placebo_post'] = (df_pre[TIME_VAR] >= py).astype(int)
        df_pre['placebo_tp'] = df_pre[TREAT_VAR] * df_pre['placebo_post']
        
        panel = df_pre.set_index([ENTITY_VAR, TIME_VAR])
        m = PanelOLS(
            panel[Y_VAR], panel[['placebo_tp'] + CONTROLS],
            entity_effects=True, time_effects=True
        ).fit(cov_type='clustered', cluster_entity=True)
        
        results[py] = {
            'coef': m.params['placebo_tp'],
            'se': m.std_errors['placebo_tp'],
            'pval': m.pvalues['placebo_tp']
        }
        stars = '***' if m.pvalues['placebo_tp']<0.01 else (
            '**' if m.pvalues['placebo_tp']<0.05 else (
            '*' if m.pvalues['placebo_tp']<0.1 else ''))
        print(f"Placebo {py}: β={m.params['placebo_tp']:.4f}{stars}")
    
    return results

# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    import os
    os.makedirs(f'{OUTPUT_DIR}/tables', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)
    
    print("=" * 60)
    print("DID Analysis Pipeline")
    print("=" * 60)
    
    # 1. 加载数据
    df = load_and_prepare()
    
    # 2. 描述性统计
    print("\n--- 描述性统计 ---")
    table1 = descriptive_stats(df)
    
    # 3. 主回归
    print("\n--- 主回归 ---")
    main_results = main_regression(df)
    
    # 4. 事件研究
    print("\n--- 事件研究 ---")
    es_result = event_study(df)
    
    # 5. 安慰剂检验
    # print("\n--- 安慰剂检验 ---")
    # placebo_result = placebo_test(df, true_event_year=2020, placebo_years=[2017, 2018])
    
    print("\n" + "=" * 60)
    print("Analysis complete. Check output/ directory.")
