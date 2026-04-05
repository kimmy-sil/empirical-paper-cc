# 数据清洗 — 高级内容

本文件按需加载（ADVANCED.md）。包含：
- Rubin's rules Python完整实现
- 面板平衡性R完整代码（fixest并行对比）
- 插值操作前后完整诊断代码
- 文本变量编码（Label/One-hot/手动映射）
- Heckman两阶段说明（分析阶段执行）

---

## Rubin's Rules合并MICE结果（Python）

```python
import numpy as np
import pandas as pd

def rubin_pool(coefs_list: list, ses_list: list) -> pd.DataFrame:
    """
    Rubin's rules合并M个插补数据集的回归结果
    coefs_list: list of Series (M个), 每个Series是该数据集的系数
    ses_list:   list of Series (M个), 每个Series是该数据集的标准误
    """
    M = len(coefs_list)
    # pooled coef
    beta_bar = np.mean([c.values for c in coefs_list], axis=0)

    # within-imputation variance
    U_bar = np.mean([se.values**2 for se in ses_list], axis=0)

    # between-imputation variance
    B = np.var([c.values for c in coefs_list], axis=0, ddof=1)

    # total variance
    T = U_bar + (1 + 1/M) * B
    SE_pooled = np.sqrt(T)

    # degrees of freedom (Barnard-Rubin)
    r   = (1 + 1/M) * B / U_bar
    df_barnard = (M - 1) * (1 + 1/r)**2

    t_stats = beta_bar / SE_pooled
    from scipy.stats import t as t_dist
    p_vals = 2 * t_dist.sf(abs(t_stats), df=df_barnard)

    return pd.DataFrame({
        "coef": beta_bar.round(6),
        "se":   SE_pooled.round(6),
        "t":    t_stats.round(4),
        "p":    p_vals.round(4),
    }, index=coefs_list[0].index)

# 使用示例（配合mice_impute函数）
import statsmodels.formula.api as smf
datasets = mice_impute(df, IMP_VARS, M=5)

coefs_list, ses_list = [], []
for df_imp in datasets:
    model = smf.ols("y ~ revenue + assets + leverage + age", data=df_imp).fit()
    coefs_list.append(model.params)
    ses_list.append(model.bse)

pooled = rubin_pool(coefs_list, ses_list)
print(pooled)
```

---

## 面板平衡性R完整代码（fixest并行对比）

```r
library(dplyr)
library(plm)
library(fixest)

# Step 1: 诊断面板结构
T_total <- n_distinct(df$year)
panel_summary <- df %>%
  group_by(firm_id) %>%
  summarise(
    n_obs  = n(),
    t_min  = min(year),
    t_max  = max(year),
    t_gap  = (t_max - t_min + 1) - n_obs
  )

cat("面板概况:\n")
cat("  总个体数:", n_distinct(df$firm_id), "\n")
cat("  平衡个体数:", sum(panel_summary$n_obs == T_total), "\n")
cat("  有缺口个体数:", sum(panel_summary$t_gap > 0), "\n")
cat("  缺失率:", round(mean(panel_summary$t_gap > 0)*100, 1), "%\n")

# Step 2: 构造平衡子样本
balanced_ids <- panel_summary %>%
  filter(n_obs == T_total) %>%
  pull(firm_id)
df_balanced <- df %>% filter(firm_id %in% balanced_ids)

cat("\n主回归（非平衡）N:", nrow(df), "\n")
cat("稳健性（平衡）N:", nrow(df_balanced),
    "(", round(nrow(df_balanced)/nrow(df)*100,1), "%)\n")

# Step 3: 并行对比（系数应接近）
res_unbalanced <- feols(y ~ x1 + x2 | firm_id + year,
                        data=df, cluster=~firm_id)
res_balanced   <- feols(y ~ x1 + x2 | firm_id + year,
                        data=df_balanced, cluster=~firm_id)
etable(res_unbalanced, res_balanced,
       headers=c("非平衡（主回归）","平衡（稳健性）"))
# 若系数接近 → 选择性进入不严重
```

---

## 插值操作完整诊断代码

```python
import pandas as pd
import numpy as np

def interpolate_with_diagnostics(df: pd.DataFrame,
                                  entity_col: str, time_col: str,
                                  var: str, max_gap: int=2) -> pd.DataFrame:
    """
    带完整诊断的面板插值
    max_gap: 最多填补连续缺失期数（建议≤2）
    返回: 含插值列 + 插值标记列 + 诊断信息
    """
    df = df.sort_values([entity_col, time_col]).copy()
    df[f"{var}_orig"]   = df[var]
    df[f"{var}_filled"] = (
        df.groupby(entity_col)[var]
        .transform(lambda x: x.interpolate(method="linear", limit=max_gap,
                                            limit_direction="forward"))
    )
    df[f"{var}_imputed"] = df[f"{var}_orig"].isnull() & df[f"{var}_filled"].notna()

    n_imputed  = df[f"{var}_imputed"].sum()
    n_total    = df[var].notna().sum() + n_imputed
    pct = n_imputed / n_total * 100

    print(f"\n=== 插值诊断: {var} ===")
    print(f"插值观测数: {n_imputed} ({pct:.1f}%)")
    print(f"原始SD:   {df[f'{var}_orig'].std():.4f}")
    print(f"插值后SD: {df[f'{var}_filled'].std():.4f}")
    sd_shrink = (df[f'{var}_orig'].std() - df[f'{var}_filled'].std()) / df[f'{var}_orig'].std() * 100
    print(f"SD缩小: {sd_shrink:.1f}%")

    if pct > 10:
        print(f"⚠️ 严重警告: 插值比例{pct:.1f}%超过10%！")
        print("   建议: 1) 主回归不插值（非平衡面板）")
        print("         2) 稳健性检验对比插值版本")
        print("         3) 或考虑Heckman纠正缺失选择偏误")
    elif sd_shrink > 20:
        print(f"⚠️ 信息失真警告: SD缩小{sd_shrink:.1f}%（>20%），慎用！")

    return df
```

---

## 文本变量编码

```python
from sklearn.preprocessing import LabelEncoder

# Label encoding（有序分类）
le = LabelEncoder()
df["industry_code"] = le.fit_transform(df["industry_name"])

# One-hot encoding
industry_dummies = pd.get_dummies(df["industry_name"], prefix="ind", drop_first=True)
df = pd.concat([df, industry_dummies], axis=1)

# 手动映射（推荐：明确控制编码含义）
region_map = {"北京":1, "上海":2, "广东":3, "浙江":4}
df["region_code"] = df["region"].map(region_map)
```

```r
# R
df$industry_code <- as.integer(factor(df$industry_name))
# One-hot
industry_dummies <- model.matrix(~industry_name - 1, data=df)
```

---

## Heckman两阶段说明（分析阶段执行）

**本skill不执行Heckman**。仅在此说明调用方式：

- **适用场景**：MNAR缺失（如只有特定企业被观测）
- **执行位置**：OLS skill / DID skill 的稳健性检验部分
- **Python实现**：`statsmodels.Probit`手动两阶段（见OLS skill）
  - **不要**使用`from heckman import Heckman`（该包不存在！）
- **R实现**：`sampleSelection::selection()`（推荐，标准误自动纠正）
- **核心要求**：需要至少一个排他性变量（exclusion restriction）
  - 与选择方程相关（Probit中显著）
  - 与结果方程无直接关系（主回归中应不显著）
