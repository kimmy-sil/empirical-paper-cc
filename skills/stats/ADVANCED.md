# 描述性统计与统计检验 — 高级内容

本文件按需加载（ADVANCED.md）。包含：
- R面板分解完整代码
- cobalt Love Plot完整配置
- threeparttable完整LaTeX模板
- 相关系数矩阵R格式化函数
- 分组描述R详细实现
- 缺失值missingno可视化（可用时）
- 方差齐性检验（Levene/Bartlett）
- Wilcoxon / Mann-Whitney / 卡方检验

---

## R面板三向分解（完整版）

```r
library(plm)
library(dplyr)

# 完整面板方差分解
panel_decompose_r <- function(pdata, vars) {
  results <- lapply(vars, function(v) {
    x <- pdata[[v]]
    list(
      variable    = v,
      overall_n   = sum(!is.na(x)),
      overall_mean = mean(x, na.rm=TRUE),
      overall_sd  = sd(x, na.rm=TRUE),
      between_sd  = sd(Between(x), na.rm=TRUE),
      within_sd   = sd(Within(x), na.rm=TRUE)
    )
  })
  do.call(rbind, lapply(results, as.data.frame))
}

pdata <- pdata.frame(df, index=c("firm_id","year"))
panel_summary <- panel_decompose_r(pdata, VARS)
print(round(panel_summary[,-1], 4))
```

---

## cobalt Love Plot完整配置

```r
library(cobalt)

# 匹配前
bal_before <- bal.tab(
  treated ~ revenue + assets + leverage + roa + age,
  data = df, thresholds = c(m=0.1)
)

# 若有匹配后样本（如PSM后）
# bal_after <- bal.tab(match_out, thresholds=c(m=0.1))

# Love Plot配置
love.plot(
  bal_before,
  threshold  = 0.1,
  abs        = TRUE,
  shapes     = c("circle filled"),
  colors     = c("#E41A1C"),
  size       = 3,
  title      = "Standardized Mean Differences Before Matching",
  var.order  = "unadjusted",
  sample.names = "Before Matching",
  limits     = c(0, 0.5)
)

# 匹配前后对比（需要match_out对象）
# love.plot(list(Unmatched=bal_before, Matched=bal_after),
#           threshold=0.1, abs=TRUE)
```

---

## 相关系数矩阵R格式化函数（完整版）

```r
library(Hmisc)

format_corr_matrix <- function(df, vars, type="pearson") {
  corr_res <- rcorr(as.matrix(df[,vars]), type=type)
  r_mat <- corr_res$r
  p_mat <- corr_res$P
  n     <- nrow(r_mat)

  out <- matrix("", n, n, dimnames=list(rownames(r_mat), colnames(r_mat)))
  for (i in 1:n) {
    out[i,i] <- "1.000"
    for (j in seq_len(i-1)) {
      p     <- p_mat[i,j]
      stars <- ifelse(p<0.01, "***", ifelse(p<0.05, "**", ifelse(p<0.10,"*","")))
      out[i,j] <- paste0(round(r_mat[i,j],3), stars)
    }
  }
  as.data.frame(out)
}

corr_fmt <- format_corr_matrix(df, VARS)
print(corr_fmt)

# 输出LaTeX（knitr）
library(knitr)
kable(corr_fmt, format="latex", booktabs=TRUE,
      caption="Correlation Matrix") %>%
  kableExtra::kable_styling(latex_options="hold_position")
```

---

## LaTeX threeparttable完整模板

```python
def to_latex_threeparttable(df: pd.DataFrame, caption: str,
                            label: str, note: str="") -> str:
    """
    发表级LaTeX三线表（threeparttable格式）
    用df.to_latex(booktabs=True)生成inner，然后包裹
    """
    inner = df.to_latex(booktabs=True, escape=True)

    note_block = ""
    if note:
        note_block = (
            "  \\begin{tablenotes}\n"
            "    \\small\n"
            f"    \\item \\textit{{Notes}}: {note}\n"
            "  \\end{tablenotes}\n"
        )
    return (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        "  \\begin{threeparttable}\n"
        f"{inner}"
        f"{note_block}"
        "  \\end{threeparttable}\n"
        "\\end{table}\n"
    )

# 使用
latex_code = to_latex_threeparttable(
    tbl.round(3),
    caption="Descriptive Statistics",
    label="tab:descriptive",
    note=r"***, **, and * indicate significance at 1\%, 5\%, and 10\% levels."
)
```

---

## 方差齐性检验

```python
from scipy import stats

# Levene检验（不依赖正态假设，推荐）
groups = [df.loc[df["treated"]==g, "revenue"].dropna() for g in [0,1]]
stat_l, p_l = stats.levene(*groups)
print(f"Levene: F={stat_l:.4f}, p={p_l:.4f}")

# Bartlett检验（假设正态分布）
stat_b, p_b = stats.bartlett(*groups)
print(f"Bartlett: stat={stat_b:.4f}, p={p_b:.4f}")
# p<0.05 → 方差不齐 → t检验用equal_var=False（Welch）
```

---

## 非参数检验

```python
# Mann-Whitney U（适合非正态分布）
stat_mw, p_mw = stats.mannwhitneyu(s1, s0, alternative="two-sided")
print(f"Mann-Whitney U={stat_mw:.1f}, p={p_mw:.4f}")

# Wilcoxon符号秩检验（配对样本）
stat_w, p_w = stats.wilcoxon(df["before"] - df["after"])

# 卡方检验（分类变量独立性）
ct = pd.crosstab(df["industry"], df["treated"])
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)
print(f"χ²={chi2:.4f}, df={dof}, p={p_chi:.4f}")
```

---

## missingno可视化（若包可用）

```python
import importlib.util

if importlib.util.find_spec("missingno"):
    import missingno as msno
    import matplotlib.pyplot as plt

    # 矩阵图（白条=缺失）
    msno.matrix(df, figsize=(12,5))
    plt.savefig("output/figures/missing_matrix.png", dpi=300, bbox_inches="tight")

    # 热力图（缺失模式相关性）
    msno.heatmap(df, figsize=(10,8))
    plt.savefig("output/figures/missing_heatmap_msno.png", dpi=300, bbox_inches="tight")
else:
    print("missingno不可用 → 使用SKILL.md中的seaborn热力图替代")
```
