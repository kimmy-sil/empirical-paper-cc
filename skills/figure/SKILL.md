# 学术图表生成 (Figure)

## 概述

本 skill 提供计量经济学实证研究中标准学术图表的生成规范与代码模板，涵盖事件研究图、平行趋势图、系数图、RDD 图、合成控制图等，符合 AER/QJE 期刊审美规范。

**适用场景**：
- DiD 平行趋势可视化（pre-trend 检验）
- 事件研究法（Event Study）系数图
- 多模型系数对比（Forest Plot）
- RDD 断点可视化
- 合成控制法路径图与 gap 图
- 分布可视化（Histogram / KDE）

---

## 前置条件

| 工具 | 依赖库 |
|------|--------|
| Python | `matplotlib >= 3.5`, `seaborn`, `numpy`, `pandas` |
| R | `ggplot2`, `ggthemes`, `coefplot`, `rdplot`（rdrobust） |
| Stata | `coefplot`（ssc install）, `event_plot`（ssc install）, `graph` 内置 |

**图表规范（全局）**：
- 字号：轴标签 ≥ 12pt，标题 ≥ 14pt，图例 ≥ 11pt
- 分辨率：DPI ≥ 300（期刊投稿），矢量图优先（PDF/EPS）
- 配色：黑白友好（区分用线型/标记，而非纯色差）
- 格式：期刊投稿用 `.pdf`（矢量），工作论文可用 `.png`（DPI=300）

---

## 分析步骤

### 步骤 1：事件研究图（Event Study / Coefficient Plot with CI）

事件研究法估计处理前后各期的动态效应，t=-1 通常为基准期（系数归零）。

**Python (matplotlib)**
```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_event_study(
    coefs,       # dict 或 DataFrame: {period: coef}
    cis_low,     # 置信区间下界 {period: ci_low}
    cis_high,    # 置信区间上界 {period: ci_high}
    baseline_period=-1,
    title="Event Study",
    xlabel="Periods Relative to Treatment",
    ylabel="Coefficient Estimate",
    save_path=None,
    figsize=(8, 5)
):
    periods = sorted(coefs.keys())
    y = [coefs[t] for t in periods]
    y_low = [cis_low[t] for t in periods]
    y_high = [cis_high[t] for t in periods]
    error_low  = [c - l for c, l in zip(y, y_low)]
    error_high = [h - c for h, c in zip(y_high, y)]

    fig, ax = plt.subplots(figsize=figsize)

    # 点估计 + 置信区间
    ax.errorbar(
        periods, y,
        yerr=[error_low, error_high],
        fmt="o",
        color="black",
        ecolor="black",
        capsize=4,
        capthick=1.2,
        linewidth=1.5,
        markersize=6,
        label="Coefficient (95% CI)"
    )

    # 基准期垂直线
    ax.axvline(x=baseline_period - 0.5, color="gray", linestyle="--",
               linewidth=1, alpha=0.7, label="Treatment onset")
    # 零效应水平线
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)

    # 灰色阴影：处理前期
    pre_periods = [t for t in periods if t < 0]
    if pre_periods:
        ax.axvspan(min(pre_periods) - 0.5, baseline_period - 0.5,
                   alpha=0.05, color="gray", label="Pre-treatment")

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, pad=10)
    ax.tick_params(labelsize=11)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=10, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)  # 去掉上、右边框

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"已保存: {save_path}")
    plt.show()
    return fig

# 示例数据（通常来自回归估计结果）
periods = list(range(-4, 5))
# 模拟：处理前无显著效应，处理后正效应递增
np.random.seed(42)
coefs   = {t: 0.02 * t if t >= 0 else 0.005 * t for t in periods}
se      = {t: 0.015 for t in periods}
cis_low  = {t: coefs[t] - 1.96 * se[t] for t in periods}
cis_high = {t: coefs[t] + 1.96 * se[t] for t in periods}

plot_event_study(
    coefs, cis_low, cis_high,
    baseline_period=-1,
    save_path="output/figures/event_study.pdf"
)
```

**R (ggplot2)**
```r
library(ggplot2)
library(dplyr)

# event_study_df: 含 period, coef, ci_low, ci_high 的 data.frame
plot_event_study_r <- function(event_study_df, baseline = -1) {
  ggplot(event_study_df, aes(x = period, y = coef)) +
    geom_hline(yintercept = 0, linetype = "solid", color = "gray50", linewidth = 0.6) +
    geom_vline(xintercept = baseline - 0.5, linetype = "dashed",
               color = "gray40", linewidth = 0.8) +
    geom_errorbar(aes(ymin = ci_low, ymax = ci_high),
                  width = 0.2, linewidth = 0.8) +
    geom_point(size = 2.5, color = "black") +
    scale_x_continuous(breaks = unique(event_study_df$period)) +
    labs(
      x = "Periods Relative to Treatment",
      y = "Coefficient Estimate",
      title = "Event Study"
    ) +
    theme_classic(base_size = 13) +
    theme(
      axis.title = element_text(size = 13),
      plot.title = element_text(size = 14, hjust = 0),
      panel.grid.major.y = element_line(color = "gray90", linewidth = 0.4)
    )
}

ggsave("output/figures/event_study.pdf", width = 8, height = 5)
```

**Stata (event_plot)**
```stata
* 需先安装: ssc install event_plot
* 假设已用 reghdfe 估计 DiD 动态效应，结果存在 e()

event_plot, default_look           ///
    graph_opt(                     ///
        xtitle("Periods Relative to Treatment")   ///
        ytitle("Coefficient")      ///
        yline(0, lp(dash) lc(gray))              ///
        xlabel(-4(1)4)             ///
    )
graph export "output/figures/event_study.pdf", replace
```

---

### 步骤 2：平行趋势图（DiD 处理组 vs 对照组时间趋势）

```python
def plot_parallel_trends(
    df, outcome_col, time_col, treat_col,
    treat_label="Treated", control_label="Control",
    vline_year=None, save_path=None, figsize=(8, 5)
):
    """
    绘制处理组与对照组的时间趋势均值图
    """
    trend = (
        df.groupby([time_col, treat_col])[outcome_col]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=figsize)

    for g, (marker, linestyle, label) in enumerate([
        ("o", "-",  treat_label),
        ("s", "--", control_label)
    ]):
        sub = trend[trend[treat_col] == (1 - g)]
        ax.plot(sub[time_col], sub[outcome_col],
                marker=marker, linestyle=linestyle,
                color="black", linewidth=1.8, markersize=6,
                label=label)

    if vline_year:
        ax.axvline(x=vline_year - 0.5, color="red", linestyle="--",
                   linewidth=1.2, alpha=0.8, label="Treatment year")

    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel(outcome_col, fontsize=13)
    ax.set_title("Parallel Trends", fontsize=14)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig

plot_parallel_trends(
    df, outcome_col="log_revenue", time_col="year",
    treat_col="treated", vline_year=2015,
    save_path="output/figures/parallel_trends.pdf"
)
```

---

### 步骤 3：系数图（多模型对比 / Forest Plot）

```python
def plot_forest(
    models_dict,   # {"Model 1": (coef, ci_low, ci_high), ...}
    title="Coefficient Comparison",
    xlabel="Coefficient Estimate",
    save_path=None,
    figsize=(7, 5)
):
    """
    多模型系数对比图（森林图风格）
    """
    labels = list(models_dict.keys())
    y_pos  = list(range(len(labels)))
    coefs  = [v[0] for v in models_dict.values()]
    lows   = [v[1] for v in models_dict.values()]
    highs  = [v[2] for v in models_dict.values()]
    err_lo = [c - l for c, l in zip(coefs, lows)]
    err_hi = [h - c for h, c in zip(highs, coefs)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(
        coefs, y_pos,
        xerr=[err_lo, err_hi],
        fmt="D", color="black",
        ecolor="black", capsize=4, capthick=1.2,
        linewidth=1.5, markersize=6
    )
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="x", labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig
```

---

### 步骤 4：分布图（Histogram + KDE）

```python
import seaborn as sns

def plot_distribution(df, var, group_col=None, save_path=None, figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize)

    if group_col is None:
        sns.histplot(df[var].dropna(), kde=True, ax=ax,
                     color="gray", edgecolor="white", bins=30)
    else:
        for g in sorted(df[group_col].unique()):
            sub = df.loc[df[group_col] == g, var].dropna()
            linestyle = "-" if g == 1 else "--"
            sns.kdeplot(sub, ax=ax, linewidth=2,
                        linestyle=linestyle, color="black",
                        label=f"Group {g}")
        ax.legend(fontsize=11, frameon=False)

    ax.set_xlabel(var, fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
```

---

### 步骤 5：散点图 + 拟合线

```python
def plot_scatter_fit(df, x_var, y_var, save_path=None, figsize=(7, 5)):
    from numpy.polynomial.polynomial import polyfit

    x = df[x_var].dropna()
    y = df[[x_var, y_var]].dropna()[y_var]
    x = df[[x_var, y_var]].dropna()[x_var]

    # OLS 拟合线
    from scipy.stats import linregress
    slope, intercept, r, p, se = linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, alpha=0.4, s=15, color="gray", edgecolors="none")
    ax.plot(x_line, y_line, color="black", linewidth=2,
            label=f"OLS fit (R²={r**2:.3f}, p={p:.3f})")
    ax.set_xlabel(x_var, fontsize=13)
    ax.set_ylabel(y_var, fontsize=13)
    ax.legend(fontsize=11, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
```

---

### 步骤 6：RDD 图（断点回归可视化）

```python
def plot_rdd(df, running_var, outcome_var, cutoff=0,
             bins=30, poly_order=1, save_path=None, figsize=(8, 5)):
    """
    RDD 图：断点两侧散点（bin 均值）+ 局部多项式拟合
    """
    from numpy.polynomial.polynomial import Polynomial

    fig, ax = plt.subplots(figsize=figsize)
    colors = {"left": "black", "right": "black"}
    markers = {"left": "o", "right": "s"}

    for side in ["left", "right"]:
        if side == "left":
            mask = df[running_var] < cutoff
        else:
            mask = df[running_var] >= cutoff

        sub = df[mask].copy()
        sub["bin"] = pd.cut(sub[running_var], bins=bins//2)
        bin_means = sub.groupby("bin")[outcome_var].mean()
        bin_centers = bin_means.index.map(lambda x: x.mid)

        ax.scatter(bin_centers, bin_means.values,
                   color=colors[side], marker=markers[side],
                   s=30, alpha=0.8, zorder=5)

        # 局部多项式拟合
        x_side = sub[running_var].values
        y_side = sub[outcome_var].values
        valid = ~np.isnan(x_side) & ~np.isnan(y_side)
        p = Polynomial.fit(x_side[valid], y_side[valid], deg=poly_order)
        x_fit = np.linspace(x_side[valid].min(), x_side[valid].max(), 200)
        ax.plot(x_fit, p(x_fit), color="black", linewidth=2)

    ax.axvline(x=cutoff, color="red", linestyle="--", linewidth=1.5,
               alpha=0.8, label=f"Cutoff = {cutoff}")
    ax.set_xlabel(running_var, fontsize=13)
    ax.set_ylabel(outcome_var, fontsize=13)
    ax.set_title("Regression Discontinuity Design", fontsize=14)
    ax.legend(fontsize=11, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
```

**R (rdplot)**
```r
library(rdrobust)

# rdplot 自动选择 bin 数和拟合阶数
rdplot(y = df$outcome, x = df$running_var, c = 0,
       title = "RDD Plot",
       x.label = "Running Variable",
       y.label = "Outcome")
```

---

### 步骤 7：合成控制法图（Path Plot + Gap Plot）

```python
def plot_synthetic_control(
    df_wide, unit_col, time_col, outcome_col,
    treated_unit, treatment_year,
    weights=None,   # {unit: weight}
    save_path_path=None, save_path_gap=None, figsize=(8, 5)
):
    """
    Path plot: 处理单元 vs 合成控制单元的时间趋势
    Gap plot:  两者差值（处理效应）
    """
    # 计算合成控制值
    if weights:
        donor_units = [u for u in weights.keys()]
        synth_values = (
            df_wide[df_wide[unit_col].isin(donor_units)]
            .assign(weighted_outcome=lambda x: x.apply(
                lambda row: row[outcome_col] * weights.get(row[unit_col], 0), axis=1))
            .groupby(time_col)["weighted_outcome"]
            .sum()
        )
    else:
        # 简化：未加权平均（仅演示）
        donor_mask = df_wide[unit_col] != treated_unit
        synth_values = df_wide[donor_mask].groupby(time_col)[outcome_col].mean()

    treated_values = (
        df_wide[df_wide[unit_col] == treated_unit]
        .set_index(time_col)[outcome_col]
    )
    gap = treated_values - synth_values

    # --- Path Plot ---
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(treated_values.index, treated_values.values,
             "-o", color="black", linewidth=2, markersize=5,
             label=f"{treated_unit} (Treated)")
    ax1.plot(synth_values.index, synth_values.values,
             "--s", color="gray", linewidth=2, markersize=5,
             label="Synthetic Control")
    ax1.axvline(x=treatment_year - 0.5, color="black",
                linestyle=":", linewidth=1.5, alpha=0.7)
    ax1.set_xlabel(time_col, fontsize=13)
    ax1.set_ylabel(outcome_col, fontsize=13)
    ax1.set_title("Synthetic Control: Path Plot", fontsize=14)
    ax1.legend(fontsize=11, frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save_path_path:
        fig1.savefig(save_path_path, dpi=300, bbox_inches="tight")
    plt.show()

    # --- Gap Plot ---
    fig2, ax2 = plt.subplots(figsize=figsize)
    ax2.plot(gap.index, gap.values, "-o", color="black",
             linewidth=2, markersize=5, label="Gap (Treated - Synthetic)")
    ax2.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
    ax2.axvline(x=treatment_year - 0.5, color="black",
                linestyle=":", linewidth=1.5, alpha=0.7)
    ax2.set_xlabel(time_col, fontsize=13)
    ax2.set_ylabel("Treatment Effect (Gap)", fontsize=13)
    ax2.set_title("Synthetic Control: Gap Plot", fontsize=14)
    ax2.legend(fontsize=11, frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save_path_gap:
        fig2.savefig(save_path_gap, dpi=300, bbox_inches="tight")
    plt.show()
    return fig1, fig2
```

---

### 步骤 8：全局图表样式设置

```python
# matplotlib 全局样式（建议在脚本开头设置）
plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.figsize": (8, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "serif",       # 期刊推荐 serif 字体
    "pdf.fonttype": 42,           # 嵌入字体（避免 reviewer 乱码）
    "ps.fonttype": 42,
})
```

**R (ggplot2 主题)**
```r
theme_academic <- function(base_size = 13) {
  theme_classic(base_size = base_size) +
    theme(
      axis.title = element_text(size = base_size),
      axis.text  = element_text(size = base_size - 2),
      legend.text = element_text(size = base_size - 2),
      legend.title = element_blank(),
      legend.background = element_blank(),
      plot.title = element_text(size = base_size + 1, hjust = 0),
      panel.grid.major.y = element_line(color = "gray90", linewidth = 0.3)
    )
}
# 使用: + theme_academic()
```

---

### LaTeX includegraphics 代码模板

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.85\textwidth]{output/figures/event_study.pdf}
    \caption{Event Study: Dynamic Treatment Effects}
    \label{fig:event_study}
    \begin{tablenotes}
        \small
        \item \textit{Notes}: This figure plots the coefficients and 95\% confidence
        intervals from equation (\ref{eq:eventstudy}). The dashed vertical line
        indicates the treatment onset. Period $t=-1$ is the omitted baseline.
        Standard errors are clustered at the firm level.
    \end{tablenotes}
\end{figure}
```

---

## 检验清单

- [ ] DPI ≥ 300，字号 ≥ 12pt
- [ ] 黑白打印仍可区分不同组（使用线型/标记形状，而非颜色）
- [ ] 坐标轴标签清晰，含单位（如 "(Billion USD)"）
- [ ] 置信区间来源已说明（95% CI？标准误类型？）
- [ ] 基准期系数（t=-1）归零已说明
- [ ] 图表注释（Notes）说明样本、估计方法、标准误类型
- [ ] PDF 格式图片已嵌入字体（`pdf.fonttype = 42`）
- [ ] 图片尺寸与期刊版面要求匹配（单栏 ≈ 3.5 英寸，双栏 ≈ 7 英寸）

---

## 常见错误提醒

1. **事件研究基准期未说明**：必须在图注中注明哪一期是 omitted baseline。
2. **置信区间宽度异常**：处理后某期 CI 突然变宽，可能是该期样本量小，需检查。
3. **RDD 图 bin 太少/太多**：bin 过少掩盖局部模式，bin 过多噪声大。推荐 rdplot 自动选择。
4. **合成控制未做 pre-period 拟合检验**：path plot 中处理前两线应几乎重合，若差距大说明合成质量差。
5. **保存为 JPEG**：JPEG 有损压缩会导致文字边缘模糊。期刊投稿请用 PDF 或 PNG（DPI≥300）。
6. **字体未嵌入 PDF**：投稿系统可能报错，设置 `pdf.fonttype=42` 解决。

---

## 输出规范

- 图片路径：`output/figures/[figure_name].pdf`（主图）
- 备用格式：`output/figures/[figure_name].png`（DPI=300）
- 图片引用：在 LaTeX 中用 `\ref{fig:xxx}` 交叉引用
- 命名规范：`fig1_event_study`, `fig2_parallel_trends`, `fig3_rdd`, `figa1_robustness`（附图加 a 前缀）
