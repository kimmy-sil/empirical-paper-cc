# 学术表格排版 (Table)

## 概述

本 skill 提供实证论文中标准学术表格的排版规范与代码模板，涵盖三线表格式、回归结果表、描述性统计表、异质性分析表和稳健性检验表，支持 LaTeX / Markdown / Python / R / Stata 多种输出方式。

**适用场景**：
- 回归结果主表（多列逐步加入控制变量/固定效应）
- 描述性统计表（Table 1）
- 异质性分析表（子样本对比）
- 稳健性检验汇总表（多种规格并列）
- 机制分析表（中介/渠道）

---

## 前置条件

| 工具 | 依赖库 |
|------|--------|
| Python | `pandas`, `statsmodels`, `linearmodels`, `tabulate` |
| R | `modelsummary`, `stargazer`, `fixest`（`etable`） |
| Stata | `estout`/`esttab`（ssc install）, `outreg2`（ssc install） |
| LaTeX | `booktabs`, `threeparttable`, `multirow`, `adjustbox` 宏包 |

---

## 三线表标准格式

三线表（Booktabs 风格）是经管类期刊主流表格格式：
- **顶线（toprule）**：最粗，表格最上方
- **中线（midrule）**：中等粗细，分隔表头与数据
- **底线（bottomrule）**：最粗，表格最下方
- **禁止竖线**
- **禁止单元格填色**（黑白打印友好）

```latex
% LaTeX 最小示例
\usepackage{booktabs}
\usepackage{threeparttable}

\begin{table}[htbp]
\centering
\caption{Main Results}
\label{tab:main}
\begin{threeparttable}
\begin{tabular}{lccc}
\toprule
            & (1)       & (2)       & (3)       \\
            & OLS       & FE        & FE+IV     \\
\midrule
Treatment   & 0.152***  & 0.134**   & 0.178***  \\
            & (0.042)   & (0.051)   & (0.063)   \\
Log(Assets) &           & 0.089*    & 0.091*    \\
            &           & (0.046)   & (0.047)   \\
Constant    & 0.312***  &           &           \\
            & (0.081)   &           &           \\
\midrule
Observations & 3,142    & 3,142     & 3,142     \\
R-squared   & 0.142     & 0.287     & 0.291     \\
Firm FE     & No        & Yes       & Yes       \\
Year FE     & No        & Yes       & Yes       \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes}: Standard errors clustered at the firm level in parentheses.
      ***, **, and * indicate significance at 1\%, 5\%, and 10\% levels.
\end{tablenotes}
\end{threeparttable}
\end{table}
```

---

## 分析步骤

### 步骤 1：回归结果表（主表，多列对比）

**Python (statsmodels + 手动格式化)**
```python
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

def format_coef(coef, se, stars=True):
    """格式化系数（保留3位小数）+ 括号内标准误"""
    if pd.isna(coef):
        return "", ""
    sig = ""
    if stars:
        t = abs(coef / se) if se > 0 else 0
        # 近似 p 值阈值（双侧，大样本）
        if t > 2.576: sig = "***"
        elif t > 1.960: sig = "**"
        elif t > 1.645: sig = "*"
    coef_str = f"{coef:.3f}{sig}"
    se_str   = f"({se:.3f})"
    return coef_str, se_str

def build_regression_table(models, var_order=None, fe_rows=None, extra_stats=None):
    """
    models: list of (label, result_object, se_type)
            result_object 需有 .params, .bse, .nobs, .rsquared
    var_order: 变量显示顺序（未列出的变量不显示）
    fe_rows: list of (row_label, [bool, bool, ...]) 固定效应行
    extra_stats: list of (label, [val, val, ...]) 附加统计行
    """
    if var_order is None:
        all_params = set()
        for _, res, _ in models:
            all_params.update(res.params.index)
        var_order = sorted(all_params)

    n_cols = len(models)
    rows = []

    # 系数行（每个变量占2行：系数 + 标准误）
    for v in var_order:
        coef_row = [v]
        se_row   = [""]
        for _, res, se_attr in models:
            coef = res.params.get(v, np.nan)
            se   = getattr(res, se_attr, res.bse).get(v, np.nan) if hasattr(res, se_attr) else res.bse.get(v, np.nan)
            c, s = format_coef(coef, se)
            coef_row.append(c)
            se_row.append(s)
        rows.append(coef_row)
        rows.append(se_row)

    # 分隔行
    rows.append([""] * (n_cols + 1))

    # N, R²
    obs_row = ["Observations"]
    r2_row  = ["R-squared"]
    for _, res, _ in models:
        obs_row.append(f"{int(res.nobs):,}" if hasattr(res, "nobs") else "")
        r2_row.append(f"{res.rsquared:.3f}" if hasattr(res, "rsquared") else "")
    rows.append(obs_row)
    rows.append(r2_row)

    # 固定效应行
    if fe_rows:
        for label, flags in fe_rows:
            row = [label] + ["Yes" if f else "No" for f in flags]
            rows.append(row)

    # 额外统计行
    if extra_stats:
        for label, vals in extra_stats:
            rows.append([label] + [str(v) for v in vals])

    col_headers = [""] + [f"({i+1})" for i in range(n_cols)]
    df_tbl = pd.DataFrame(rows, columns=col_headers)
    return df_tbl

# 使用示例
import statsmodels.formula.api as smf

m1 = smf.ols("log_revenue ~ treated", data=df).fit()
m2 = smf.ols("log_revenue ~ treated + log_assets + leverage", data=df).fit()
m3 = smf.ols("log_revenue ~ treated + log_assets + leverage + age", data=df).fit()

tbl = build_regression_table(
    models=[
        ("OLS (1)", m1, "bse"),
        ("OLS (2)", m2, "bse"),
        ("OLS (3)", m3, "bse"),
    ],
    var_order=["treated", "log_assets", "leverage", "age", "Intercept"],
    fe_rows=[
        ("Firm FE",  [False, False, False]),
        ("Year FE",  [False, False, False]),
    ]
)
print(tbl.to_string(index=False))
```

**R (modelsummary — 最推荐)**
```r
library(modelsummary)
library(fixest)

# 估计模型
m1 <- feols(log_revenue ~ treated, data = df, vcov = "iid")
m2 <- feols(log_revenue ~ treated + log_assets + leverage | firm_id, data = df)
m3 <- feols(log_revenue ~ treated + log_assets + leverage | firm_id + year, data = df)

# 生成表格
modelsummary(
  list("(1) OLS" = m1, "(2) FE" = m2, "(3) TWFE" = m3),
  stars = c("*" = 0.1, "**" = 0.05, "***" = 0.01),
  coef_rename = c(
    "treated"    = "Treatment",
    "log_assets" = "Log(Assets)",
    "leverage"   = "Leverage"
  ),
  gof_map = c("nobs", "r.squared", "adj.r.squared"),
  output = "output/tables/main_results.tex",
  booktabs = TRUE,
  title = "Main Results",
  notes = "Standard errors clustered at firm level. *p<0.1; **p<0.05; ***p<0.01."
)

# 也可输出为 Word
modelsummary(list(m1, m2, m3), output = "output/tables/main_results.docx")
```

**R (fixest::etable)**
```r
library(fixest)

etable(
  m1, m2, m3,
  se = "cluster",       # 聚类标准误
  cluster = ~firm_id,
  digits = 3,
  fitstat = c("n", "r2", "ar2"),
  tex = TRUE,           # 输出 LaTeX
  file = "output/tables/main_results_etable.tex"
)
```

**Stata (esttab)**
```stata
* 估计并存储结果
eststo m1: reg log_revenue treated, robust
eststo m2: reghdfe log_revenue treated log_assets leverage, absorb(firm_id) cluster(firm_id)
eststo m3: reghdfe log_revenue treated log_assets leverage, absorb(firm_id year) cluster(firm_id)

* 输出三线表
esttab m1 m2 m3,                      ///
    b(3) se(3) star(* 0.1 ** 0.05 *** 0.01) ///
    booktabs                            ///
    title("Main Results")               ///
    mtitles("OLS" "Firm FE" "TWFE")    ///
    keep(treated log_assets leverage)   ///
    order(treated log_assets leverage)  ///
    scalars("N Observations" "r2 R-squared") ///
    addnotes("Standard errors clustered at firm level." ///
             "* p<0.1, ** p<0.05, *** p<0.01")  ///
    using "output/tables/main_results.tex", replace

* outreg2 替代方案
outreg2 [m1 m2 m3] using "output/tables/main_outreg.doc", ///
    word bdec(3) sdec(3) addstat("Firm FE", "No", "Year FE", "No")
```

---

### 步骤 2：描述性统计表

（参见 `stats/SKILL.md` 中的详细代码）

**LaTeX 模板**
```latex
\begin{table}[htbp]
\centering
\caption{Descriptive Statistics}
\label{tab:descriptive}
\begin{threeparttable}
\begin{tabular}{lrrrrrr}
\toprule
Variable      & N      & Mean  & SD    & Min   & P75   & Max   \\
\midrule
Log Revenue   & 31,400 & 8.342 & 1.245 & 3.210 & 9.187 & 13.420 \\
Log Assets    & 31,400 & 9.154 & 1.312 & 4.120 & 10.023 & 14.350 \\
Leverage      & 31,234 & 0.421 & 0.195 & 0.001 & 0.573 & 0.921 \\
ROA           & 30,987 & 0.045 & 0.067 & -0.312 & 0.083 & 0.312 \\
Firm Age      & 31,400 & 12.3  & 6.8   & 1     & 16    & 45    \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes}: Sample period: 2010--2022. Continuous variables
are winsorized at the 1st and 99th percentiles.
\end{tablenotes}
\end{threeparttable}
\end{table}
```

---

### 步骤 3：异质性分析表

```latex
\begin{table}[htbp]
\centering
\caption{Heterogeneity Analysis}
\label{tab:heterogeneity}
\begin{threeparttable}
\begin{tabular}{lcccc}
\toprule
              & \multicolumn{2}{c}{SOE vs Private} & \multicolumn{2}{c}{East vs West} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
              & (1) SOE & (2) Private & (3) East & (4) West \\
\midrule
Treatment     & 0.089   & 0.213***    & 0.198*** & 0.042    \\
              & (0.067) & (0.058)     & (0.051)  & (0.089)  \\
\midrule
Observations  & 8,423   & 22,977      & 18,640   & 12,760   \\
R-squared     & 0.312   & 0.298       & 0.287    & 0.341    \\
Firm FE       & Yes     & Yes         & Yes      & Yes      \\
Year FE       & Yes     & Yes         & Yes      & Yes      \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes}: Columns (1) and (2) split the sample by ownership type;
      columns (3) and (4) by geographic region. Standard errors clustered at
      firm level. *p<0.1; **p<0.05; ***p<0.01.
\end{tablenotes}
\end{threeparttable}
\end{table}
```

---

### 步骤 4：稳健性检验汇总表

```python
# Python: 汇总多种规格的系数和 SE
robustness_specs = {
    "Baseline":           (0.134, 0.051),
    "Winsorized 5%":      (0.128, 0.054),
    "Balanced panel":     (0.141, 0.058),
    "Drop year 2008":     (0.139, 0.052),
    "Add industry trend": (0.127, 0.053),
    "Logit model":        (0.148, 0.061),
    "Probit model":       (0.143, 0.059),
    "IV (instrument Z)":  (0.198, 0.073),
}

def build_robustness_table(specs_dict):
    rows = []
    for label, (coef, se) in specs_dict.items():
        sig = ""
        t = abs(coef / se)
        if t > 2.576: sig = "***"
        elif t > 1.960: sig = "**"
        elif t > 1.645: sig = "*"
        rows.append({
            "Specification": label,
            "Coef.": f"{coef:.3f}{sig}",
            "SE": f"({se:.3f})",
        })
    return pd.DataFrame(rows)

tbl_robust = build_robustness_table(robustness_specs)
print(tbl_robust.to_string(index=False))
```

**对应 LaTeX 输出**
```latex
\begin{table}[htbp]
\centering
\caption{Robustness Checks}
\label{tab:robustness}
\begin{threeparttable}
\begin{tabular}{lcc}
\toprule
Specification          & Coefficient & SE      \\
\midrule
Baseline               & 0.134**     & (0.051) \\
Winsorized 5\%         & 0.128**     & (0.054) \\
Balanced panel         & 0.141**     & (0.058) \\
Drop year 2008         & 0.139***    & (0.052) \\
Add industry trend     & 0.127**     & (0.053) \\
Logit model            & 0.148**     & (0.061) \\
IV (instrument Z)      & 0.198***    & (0.073) \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{Notes}: Each row reports the coefficient on the Treatment variable
      from a separate regression. All regressions include firm and year FE.
      Standard errors (in parentheses) clustered at firm level.
      *p<0.1; **p<0.05; ***p<0.01.
\end{tablenotes}
\end{threeparttable}
\end{table}
```

---

### 步骤 5：Markdown 表格模板

用于工作论文草稿或 README 快速展示：

```markdown
| Variable    |    N | Mean  |   SD  |  Min  |   Max |
|-------------|------|-------|-------|-------|-------|
| Log Revenue | 31,400 | 8.342 | 1.245 | 3.210 | 13.420 |
| Leverage    | 31,234 | 0.421 | 0.195 | 0.001 | 0.921 |
| ROA         | 30,987 | 0.045 | 0.067 | -0.312 | 0.312 |

*Notes*: All continuous variables winsorized at 1%/99%.

| | (1) OLS | (2) Firm FE | (3) TWFE |
|---|---|---|---|
| Treatment | 0.152*** | 0.134** | 0.178*** |
| | (0.042) | (0.051) | (0.063) |
| N | 3,142 | 3,142 | 3,142 |
| R² | 0.142 | 0.287 | 0.291 |
| Firm FE | No | Yes | Yes |
| Year FE | No | No | Yes |

\*p<0.1; \*\*p<0.05; \*\*\*p<0.01. Standard errors in parentheses.
```

---

## 检验清单

- [ ] 三线表格式：无竖线，无单元格背景色
- [ ] 标准误在括号内（系数下方），不是 t 统计量
- [ ] N（样本量）和 R² 在表格底部单独行
- [ ] 显著性星号说明在底部注释（不在表格内）
- [ ] 固定效应行明确标注（Yes/No）
- [ ] 列标题格式统一（"(1)"、"(2)"...）
- [ ] 多组对比用 `\cmidrule` 分隔，不用竖线
- [ ] 变量名使用人类可读标签（非原始变量名如 "b001000000"）
- [ ] 数值格式统一（系数3位小数，SE 3位小数，N 整数加千分位逗号）
- [ ] LaTeX 文件已用 `latexmk` 或 `pdflatex` 编译通过

---

## 常见错误提醒

1. **括号内放 t 值而非 SE**：经管类期刊标准是括号内放标准误（SE），而非 t 统计量。esttab 默认是 SE，但需确认。
2. **R² 与 Adjusted R²** 混淆：固定效应模型通常报告 Within R²，需在注释中说明。
3. **星号标准不统一**：有的论文用 `* p<0.05 ** p<0.01`，应在投稿前确认目标期刊惯例。通常 AER/QJE 用 `* 0.10 ** 0.05 *** 0.01`。
4. **`threeparttable` 与 `table` 宽度冲突**：`adjustbox{max width=\textwidth}` 可解决表格超出版面问题。
5. **modelsummary 输出变量顺序错误**：用 `coef_map` 参数指定显示顺序和重命名，未列入的变量自动隐藏。
6. **Stata esttab 标准误类型**：需在 `esttab` 命令中用 `vce(cluster firm_id)` 或 `se` 选项明确指定，否则默认 OLS SE。

---

## 输出规范

- LaTeX 表格：`output/tables/table[N]_[name].tex`（如 `table1_descriptive.tex`）
- Excel 版本：`output/tables/table[N]_[name].xlsx`（附录/审稿人）
- 附录表格：`output/tables/tableA[N]_[name].tex`（Online Appendix）
- 主文件中用 `\input{output/tables/table1_descriptive.tex}` 引入
- 表格编号顺序与正文引用顺序一致（`Table 1`, `Table 2`...）
