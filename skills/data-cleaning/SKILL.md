# 数据清洗 (Data Cleaning)

## 概述

本 skill 提供经管类实证研究中标准化数据清洗流程，涵盖缺失值处理、异常值检测、面板数据平衡化、变量构造、文本编码和数据合并，支持 Python / R / Stata 三种工具。

**适用场景**：
- 原始面板数据清洗与整理
- 多数据源合并与诊断
- 回归前变量工程（交互项、滞后项、差分、对数化）
- 样本筛选与有效样本确认

---

## 前置条件

| 工具 | 依赖库 |
|------|--------|
| Python | `pandas >= 1.5`, `numpy`, `scipy`, `sklearn`, `missingno` |
| R | `dplyr`, `tidyr`, `mice`, `Hmisc`, `naniar` |
| Stata | 版本 ≥ 14，`winsor2`（ssc install） |

数据要求：
- 已明确面板维度（个体 ID + 时间变量）
- 数据字典（codebook）可用，了解每个变量含义与取值范围
- 原始数据已备份

---

## 分析步骤

### 步骤 1：初步探索与缺失值诊断

**Python (pandas)**
```python
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/panel_data.csv")

# 基本信息
print(df.shape)           # 样本量 × 变量数
print(df.dtypes)          # 数据类型
print(df.describe())      # 描述性统计

# 缺失值统计
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    "missing_count": missing,
    "missing_pct": missing_pct
}).sort_values("missing_pct", ascending=False)
print(missing_df[missing_df.missing_count > 0])

# 可视化缺失模式
msno.matrix(df, figsize=(12, 5))
plt.savefig("output/figures/missing_pattern.png", dpi=300)
```

**R (dplyr + naniar)**
```r
library(dplyr)
library(naniar)

df <- read.csv("data/raw/panel_data.csv")

# 缺失概览
miss_var_summary(df)     # 各变量缺失统计
vis_miss(df)             # 可视化缺失

# 按组统计缺失
df %>%
  group_by(year) %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  print()
```

**Stata**
```stata
use "data/raw/panel_data.dta", clear
describe
summarize
misstable summarize    // 缺失值汇总
misstable patterns     // 缺失模式
```

---

### 步骤 2：缺失值处理

#### 策略 A：删除法（listwise deletion）
适用于缺失随机（MCAR）且缺失比例 < 5% 的变量。

```python
# Python: 删除关键变量缺失行
key_vars = ["revenue", "employees", "year", "firm_id"]
df_clean = df.dropna(subset=key_vars)
print(f"删除后样本量: {len(df_clean)} (原始: {len(df)})")
```

```r
# R
df_clean <- df %>% drop_na(revenue, employees)
```

```stata
* Stata
drop if missing(revenue) | missing(employees)
```

#### 策略 B：插值法（适用于时间序列/面板中的短暂缺失）

```python
# Python: 面板内线性插值（先按个体排序再插值）
df = df.sort_values(["firm_id", "year"])
df["revenue_filled"] = (
    df.groupby("firm_id")["revenue"]
    .transform(lambda x: x.interpolate(method="linear", limit=2))
    # limit=2: 最多填补连续2个缺失
)
```

```stata
* Stata: ipolate 对面板数据插值
xtset firm_id year
ipolate revenue year, gen(revenue_filled) by(firm_id)
```

#### 策略 C：多重插补（Multiple Imputation，适用于 MAR 缺失）

```python
# Python: IterativeImputer (MICE)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp_vars = ["revenue", "assets", "leverage", "age"]
imputer = IterativeImputer(max_iter=10, random_state=42)
df[imp_vars] = imputer.fit_transform(df[imp_vars])
```

```r
# R: mice 包
library(mice)
imp_data <- mice(df[, c("revenue","assets","leverage","age")],
                 m = 5,         # 5个插补数据集
                 method = "pmm", # predictive mean matching
                 seed = 42)
df_complete <- complete(imp_data, action = 1)
```

```stata
* Stata: mi impute
mi set wide
mi register imputed revenue assets leverage
mi impute chained (regress) revenue assets leverage = age size, add(5) rseed(42)
```

**注意**：多重插补后需用合并规则（Rubin's rules）合并回归结果，不能简单取均值作回归。

---

### 步骤 3：异常值检测与处理

#### IQR 法（分布无参假设）

```python
# Python: IQR 异常值检测
def detect_outliers_iqr(series, k=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    return (series < lower) | (series > upper)

outlier_flag = detect_outliers_iqr(df["revenue"])
print(f"IQR 异常值数量: {outlier_flag.sum()}")
```

#### Z-score 法（假设正态分布）

```python
from scipy import stats

z_scores = np.abs(stats.zscore(df["revenue"].dropna()))
outlier_flag_z = z_scores > 3
print(f"Z-score 异常值数量: {outlier_flag_z.sum()}")
```

#### Winsorize（缩尾，推荐用于实证研究）

```python
# Python: 上下 1% 缩尾
from scipy.stats import mstats

df["revenue_wins"] = mstats.winsorize(df["revenue"], limits=[0.01, 0.01])
```

```r
# R: DescTools 或 Hmisc
library(DescTools)
df$revenue_wins <- Winsorize(df$revenue, probs = c(0.01, 0.99))
```

```stata
* Stata: winsor2 (ssc install winsor2)
winsor2 revenue, cuts(1 99) replace    // 直接替换原变量
winsor2 revenue, cuts(1 99) suffix(_w) // 生成新变量 revenue_w
```

**经管研究惯例**：通常对连续型财务变量（营收、资产、利润率等）做上下各 1% 缩尾，并在正文说明。

---

### 步骤 4：面板数据平衡化

```python
# Python: 检验平衡性
panel_counts = df.groupby("firm_id")["year"].count()
print(f"非平衡面板: {panel_counts.value_counts()}")

# 构造平衡面板（仅保留每个企业都有 T 期数据的样本）
T = df["year"].nunique()
firms_balanced = panel_counts[panel_counts == T].index
df_balanced = df[df["firm_id"].isin(firms_balanced)]
print(f"平衡面板: {df_balanced['firm_id'].nunique()} 家企业 × {T} 期")
```

```r
# R: 检验与构造平衡面板
library(dplyr)
T_periods <- n_distinct(df$year)
balanced_firms <- df %>%
  count(firm_id) %>%
  filter(n == T_periods) %>%
  pull(firm_id)

df_balanced <- df %>% filter(firm_id %in% balanced_firms)
```

```stata
* Stata
xtset firm_id year
xtdescribe           // 描述面板结构（是否平衡）
// 保留平衡面板
bysort firm_id: gen nobs = _N
keep if nobs == 10   // 假设 T=10
```

**注意**：非平衡面板也可用于双向固定效应，强制平衡会造成样本选择偏误，需在文章中说明选择原因。

---

### 步骤 5：变量构造

#### 交互项

```python
# Python
df["treat_post"] = df["treated"] * df["post"]          # DiD 交互项
df["size_lev"] = df["log_assets"] * df["leverage"]      # 连续×连续
```

```stata
* Stata
gen treat_post = treated * post
gen size_lev = log_assets * leverage
```

#### 滞后项与前向项

```python
# Python: 面板滞后
df = df.sort_values(["firm_id", "year"])
df["revenue_lag1"] = df.groupby("firm_id")["revenue"].shift(1)
df["revenue_lag2"] = df.groupby("firm_id")["revenue"].shift(2)
df["revenue_lead1"] = df.groupby("firm_id")["revenue"].shift(-1)
```

```stata
* Stata (xtset 后可直接用 L. F. 算子)
xtset firm_id year
gen revenue_lag1 = L.revenue
gen revenue_lag2 = L2.revenue
gen revenue_lead1 = F.revenue
```

#### 一阶差分

```python
# Python
df["d_revenue"] = df.groupby("firm_id")["revenue"].diff(1)
```

```stata
* Stata
gen d_revenue = D.revenue    // 需先 xtset
```

#### 对数变换

```python
# Python: 安全对数（处理零值）
df["log_revenue"] = np.log(df["revenue"].clip(lower=1e-6))
# 或者 log(1+x) 变换
df["log1p_revenue"] = np.log1p(df["revenue"])
```

```r
# R
df <- df %>%
  mutate(
    log_revenue = log(pmax(revenue, 1e-6)),
    log1p_revenue = log1p(revenue)
  )
```

---

### 步骤 6：文本变量编码

```python
# Python: 行业/地区等分类变量编码
# Label encoding（有序分类）
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["industry_code"] = le.fit_transform(df["industry_name"])

# One-hot encoding
industry_dummies = pd.get_dummies(df["industry_name"], prefix="ind", drop_first=True)
df = pd.concat([df, industry_dummies], axis=1)

# 手动映射（推荐：明确控制编码含义）
region_map = {"北京": 1, "上海": 2, "广东": 3, "浙江": 4}
df["region_code"] = df["region"].map(region_map)
```

```stata
* Stata: encode 命令
encode industry_name, gen(industry_code)
* 生成虚拟变量
tabulate industry_code, gen(ind_)
```

---

### 步骤 7：数据合并与 _merge 诊断

**Python (pandas)**
```python
# 合并前检查 key 唯一性
assert df_firm["firm_id"].is_unique, "firm 数据 key 不唯一！"
assert df_macro["year"].is_unique, "宏观数据 key 不唯一！"

# 合并
df_merged = pd.merge(
    df_firm,
    df_macro,
    on=["year"],          # 合并键
    how="left",           # 保留全部企业数据
    indicator=True        # 生成 _merge 列
)

# 诊断
print(df_merged["_merge"].value_counts())
# both        → 成功匹配
# left_only   → 企业数据无对应宏观数据（检查年份范围）
# right_only  → 宏观数据无对应企业（通常可以忽略）

# 多键合并
df_merged2 = pd.merge(
    df_left,
    df_right,
    on=["firm_id", "year"],
    how="left",
    indicator=True
)
# 警告：如果 key 不唯一（many-to-many），pandas 会做笛卡尔积！
```

**R (dplyr)**
```r
library(dplyr)

# 检查 key 唯一性
stopifnot(!any(duplicated(df_firm$firm_id)))

df_merged <- left_join(df_firm, df_macro, by = "year")

# 诊断未匹配（dplyr 无 indicator，用 anti_join）
unmatched <- anti_join(df_firm, df_macro, by = "year")
cat("未匹配行数:", nrow(unmatched), "\n")
```

**Stata**
```stata
* 合并前排序
sort firm_id year

* one-to-one merge
merge 1:1 firm_id year using "data/macro_data.dta"
* many-to-one (firm-year to macro-year)
merge m:1 year using "data/macro_data.dta"

* 诊断 _merge 变量（Stata 自动生成）
tab _merge
/*
_merge == 1: 主数据只有，未匹配（检查原因）
_merge == 2: using 数据只有（通常丢弃）
_merge == 3: 两边都有，成功匹配 ✓
*/

* 保留匹配样本并删除标记变量
keep if _merge == 3
drop _merge
```

**合并诊断清单**：
- [ ] 合并前检查 key 是否唯一（避免 many-to-many）
- [ ] 合并后比较样本量变化（不应大幅缩小）
- [ ] `_merge == 1` 的比例是否合理，是否存在系统性缺失
- [ ] 合并后检查关键变量无意外全 NA

---

## 检验清单

- [ ] 样本量与数据字典描述一致
- [ ] 面板维度（个体 × 时间）正确设定
- [ ] 主要变量缺失率 < 20%（超过需说明处理方式）
- [ ] 异常值已做 Winsorize 或删除，并在文章中说明
- [ ] 所有构造变量已核对公式与计算结果
- [ ] 滞后项/差分项的样本量损失已记录
- [ ] 数据合并后 `_merge == 3` 比例 ≥ 90%（否则检查 key）
- [ ] 最终清洗后数据已保存至 `data/processed/`，原始数据未被覆盖

---

## 常见错误提醒

1. **数据类型混淆**：日期列读入为字符串，导致时间排序错误。解决：`pd.to_datetime()` 或 Stata `destring`。
2. **面板滞后跨个体**：未按 `firm_id` 分组直接 `.shift()`，导致不同企业数据错位。解决：始终用 `groupby("firm_id").shift()`。
3. **对数变换零值**：`log(0) = -inf`，需先做 `clip(lower=1e-6)` 或 `log1p()`。
4. **Winsorize 时序错误**：应对截面数据（每年）分别 Winsorize，而非对全样本。但经管研究更多对全样本 Winsorize，需明确说明。
5. **合并键类型不一致**：一个是 `int`，另一个是 `str`，导致无法匹配。解决：合并前统一类型。
6. **多重插补后直接取均值做回归**：错误！需用 Rubin's rules 合并多个插补数据集的回归结果。

---

## 输出规范

- 清洗后数据存放：`data/processed/panel_clean.csv`（或 `.dta`）
- 清洗日志记录：`logs/data_cleaning_log.txt`（包含每步样本量变化）
- 缺失值模式图：`output/figures/missing_pattern.png`
- 数据清洗说明写入论文附录（样本筛选流程表）

```
样本筛选流程示例（附录表）：
初始样本（A股上市公司 2010-2022）          N = 38,500
  - 剔除金融行业                           N = 35,200
  - 剔除 ST/PT 企业                        N = 33,800
  - 剔除主要变量缺失                        N = 31,400
最终样本                                   N = 31,400
```

---

### Section: 缺失机制诊断框架

决策树：
```
缺失值 → 判断缺失机制
├── MCAR（完全随机）→ Little's MCAR检验（R代码: naniar::mcar_test / Python: missingno）
│   └── 检验通过 → 直接用非平衡面板，FE没问题
├── MAR（与可观测变量相关）→ 控制相关变量后用非平衡面板
│   └── 或用MICE多重插补
└── MNAR（与不可观测因素相关）→ Heckman两阶段纠正
```

#### Little's MCAR检验 R代码

```r
# R: naniar包 Little's MCAR检验
library(naniar)

# Little's MCAR检验
# H0: 缺失完全随机（MCAR）
# p > 0.05 → 不能拒绝MCAR → 可直接用非平衡面板
mcar_result <- mcar_test(df)
print(mcar_result)
# 输出: statistic（卡方统计量）, df（自由度）, p.value

# 可视化缺失模式
vis_miss(df)                  # 全数据缺失热图
gg_miss_var(df)               # 各变量缺失比例条形图
gg_miss_upset(df)             # UpSet图（缺失组合模式）

# 按组检验是否MCAR
df %>%
  group_by(year) %>%
  summarise(
    miss_x1 = mean(is.na(x1)),
    miss_x2 = mean(is.na(x2))
  ) %>%
  print()
# 若缺失率随时间系统性变化 → 非MCAR
```

#### Python missingno可视化

```python
# Python: missingno缺失可视化
import missingno as msno
import matplotlib.pyplot as plt

# 矩阵图（白条=缺失，黑条=存在）
msno.matrix(df, figsize=(12, 6), sparkline=True)
plt.savefig("output/figures/missing_matrix.png", dpi=300, bbox_inches='tight')

# 条形图（各变量完整性比例）
msno.bar(df, figsize=(12, 5))
plt.savefig("output/figures/missing_bar.png", dpi=300, bbox_inches='tight')

# 热力图（缺失模式相关性：两变量是否同时缺失）
msno.heatmap(df, figsize=(10, 8))
plt.savefig("output/figures/missing_heatmap.png", dpi=300, bbox_inches='tight')

# 树状图（层级聚类缺失模式）
msno.dendrogram(df, figsize=(10, 5))
plt.savefig("output/figures/missing_dendrogram.png", dpi=300, bbox_inches='tight')

# 辅助检验：缺失是否与观测变量相关（MAR检验思路）
import pandas as pd
import scipy.stats as stats

for col in df.columns:
    if df[col].isnull().sum() > 0:
        miss_flag = df[col].isnull().astype(int)
        for other_col in df.select_dtypes(include='number').columns:
            if other_col != col and df[other_col].notnull().all():
                t_stat, p_val = stats.ttest_ind(
                    df.loc[miss_flag == 1, other_col],
                    df.loc[miss_flag == 0, other_col],
                    nan_policy='omit'
                )
                if p_val < 0.05:
                    print(f"{col}的缺失与{other_col}显著相关(p={p_val:.3f}) → 可能为MAR")
```

---

### Section: Heckman两阶段选择模型

适用场景：因变量存在样本选择偏误（MNAR），如只有特定企业才被观测到/参与某项目。

#### 第一阶段：Probit模型 → 计算逆米尔斯比（IMR）

$$P(S_i = 1 | Z_i) = \Phi(Z_i' \gamma)$$

逆米尔斯比（Inverse Mills Ratio）：
$$\lambda_i = \frac{\phi(Z_i' \hat{\gamma})}{\Phi(Z_i' \hat{\gamma})}$$

- $\phi$：标准正态PDF
- $\Phi$：标准正态CDF
- $Z_i$：选择方程的工具变量（需包含至少一个排他性变量）

#### 第二阶段：将IMR纳入主回归

$$Y_i = X_i' \beta + \rho \sigma \lambda_i + u_i$$

若 $\rho \sigma \lambda_i$ 系数显著 → 存在选择偏误，Heckman纠正必要。

#### Python代码（statsmodels Heckman）

```python
# Python: Heckman两阶段选择模型
import statsmodels.api as sm
from scipy.stats import norm
import pandas as pd
import numpy as np

# ============================================================
# 方法A：手动两阶段（便于理解和调试）
# ============================================================

# 数据准备
# select_vars: 选择方程的解释变量（含排他性变量）
# outcome_vars: 主方程的解释变量

# 第一阶段：Probit（因变量=是否被观测/参与）
probit_model = sm.Probit(
    df['selected'],                         # 1=被观测，0=未被观测
    sm.add_constant(df[select_vars])
)
probit_res = probit_model.fit()
print("第一阶段Probit结果:")
print(probit_res.summary())

# 计算逆米尔斯比（IMR）
xb = probit_res.fittedvalues                # 线性预测值
phi = norm.pdf(xb)                          # 标准正态PDF
Phi = norm.cdf(xb)                          # 标准正态CDF

# 只对被选择样本计算IMR
df['imr'] = np.where(df['selected'] == 1, phi / Phi, np.nan)

# 第二阶段：OLS（仅使用被观测样本，加入IMR）
df_selected = df[df['selected'] == 1].copy()
ols_model = sm.OLS(
    df_selected['y'],
    sm.add_constant(df_selected[outcome_vars + ['imr']])
)
ols_res = ols_model.fit()
print("\n第二阶段OLS结果（含IMR）:")
print(ols_res.summary())

# 解读：若imr系数显著，说明存在样本选择偏误
imr_coef = ols_res.params['imr']
imr_pval = ols_res.pvalues['imr']
print(f"\nIMR系数: {imr_coef:.4f}, p值: {imr_pval:.4f}")
if imr_pval < 0.05:
    print("⚠️ 选择偏误显著，Heckman纠正有必要")
else:
    print("✓ 选择偏误不显著，OLS估计可信")

# ============================================================
# 方法B：使用heckman包（需pip install heckman）
# ============================================================
# pip install heckman
from heckman import Heckman

heck_model = Heckman(
    df['y'],
    df[outcome_vars],
    df[select_vars]
)
heck_res = heck_model.fit(method='twostep')
print(heck_res.summary())
```

#### R代码（sampleSelection包）

```r
# R: Heckman两阶段选择模型
library(sampleSelection)
library(dplyr)

# ============================================================
# 方法A：sampleSelection包（推荐）
# ============================================================

# 构建Heckman选择模型
# 公式：选择方程 | 主方程
# 选择方程需包含至少一个排他性变量（exclusion restriction）

heck_model <- selection(
  selection = selected ~ z1 + z2 + exclusion_var,  # 第一阶段（含排他性变量）
  outcome   = y ~ x1 + x2 + x3,                   # 第二阶段（主方程）
  data      = df,
  method    = "2step"                               # 两步法（或"ml"极大似然）
)
summary(heck_model)

# 提取逆米尔斯比
df$imr_samplesel <- heck_model$imratio

# 检验选择偏误显著性
# 查看summary中 "Inverse Mills Ratio" 的系数和p值

# ============================================================
# 方法B：手动两阶段（便于理解）
# ============================================================
library(sampleSelection)

# 第一阶段：Probit
probit_res <- glm(
  selected ~ z1 + z2 + exclusion_var,
  family = binomial(link = "probit"),
  data   = df
)

# 计算IMR
xb  <- predict(probit_res, type = "link")
phi <- dnorm(xb)
Phi <- pnorm(xb)
df$imr <- phi / Phi

# 第二阶段：OLS（仅选择样本）
df_sel <- df %>% filter(selected == 1)
ols_res2 <- lm(y ~ x1 + x2 + x3 + imr, data = df_sel)
summary(ols_res2)

# 注意：手动两步法标准误需要纠正（使用sampleSelection包自动纠正）
# 建议用sampleSelection::selection()获取正确标准误

# ============================================================
# 排他性约束验证（Exclusion Restriction检验）
# ============================================================
# 排他性变量条件：
# 1. 与选择（selected）相关（Probit中显著）
# 2. 与结果（y）无直接关系（仅通过选择影响y）

# 检验1：第一阶段中排他性变量显著性
cat("排他性变量在第一阶段的显著性:\n")
print(summary(probit_res)$coefficients["exclusion_var", ])

# 检验2：加入排他性变量后主方程系数变化（理论上不变）
ols_with_excl <- lm(y ~ x1 + x2 + x3 + exclusion_var, data = df_sel)
cat("\n排他性变量在主方程中（应不显著）:\n")
print(summary(ols_with_excl)$coefficients["exclusion_var", ])
```

---

### Section: 平衡vs非平衡面板决策树

```
估计方法对面板结构的要求：
├── FE / RE → 非平衡面板可用
├── 事件研究 → 需要事件窗口内连续观测
├── 差分GMM → 接近平衡
└── 系统GMM → 强烈建议平衡面板
稳健性做法：主回归用非平衡面板，稳健性用平衡面板对比
```

#### 诊断与决策代码

```r
# R: 面板平衡性诊断与处理
library(dplyr)
library(plm)

# Step 1: 诊断面板结构
panel_summary <- df %>%
  group_by(entity_id) %>%
  summarise(
    n_obs  = n(),
    t_min  = min(year),
    t_max  = max(year),
    t_span = t_max - t_min + 1,
    t_gap  = t_span - n_obs  # >0 说明有缺口
  )

cat("面板平衡性概况:\n")
cat("  总个体数:", n_distinct(df$entity_id), "\n")
cat("  平衡个体数:", sum(panel_summary$n_obs == max(panel_summary$n_obs)), "\n")
cat("  有缺口个体数:", sum(panel_summary$t_gap > 0), "\n")
cat("  缺失率:", mean(panel_summary$t_gap > 0) * 100, "%\n")

# Step 2: 根据方法要求决策
T_total <- n_distinct(df$year)

# 构造平衡面板（用于稳健性对比）
balanced_ids <- panel_summary %>%
  filter(n_obs == T_total) %>%
  pull(entity_id)
df_balanced <- df %>% filter(entity_id %in% balanced_ids)

cat("\n主回归（非平衡面板）N:", nrow(df), "\n")
cat("稳健性（平衡面板）N:", nrow(df_balanced), "\n")
cat("平衡面板保留比例:", nrow(df_balanced)/nrow(df) * 100, "%\n")

# Step 3: 并行跑非平衡和平衡面板，对比系数
library(fixest)
res_unbalanced <- feols(y ~ x1 + x2 | entity_id + year, data = df,
                        cluster = ~entity_id)
res_balanced   <- feols(y ~ x1 + x2 | entity_id + year, data = df_balanced,
                        cluster = ~entity_id)

etable(res_unbalanced, res_balanced,
       headers = c("非平衡面板（主回归）", "平衡面板（稳健性）"))
# 若两者系数接近 → 选择性进入不严重
```

```python
# Python: 面板平衡性快速诊断
import pandas as pd

def check_panel_balance(df, entity_col='entity_id', time_col='year'):
    """诊断面板平衡性"""
    obs_per_entity = df.groupby(entity_col)[time_col].count()
    T_max = obs_per_entity.max()
    T_min = obs_per_entity.min()

    print(f"面板维度: N={df[entity_col].nunique()}, T_max={T_max}, T_min={T_min}")
    print(f"平衡面板: {T_max == T_min}")
    print(f"个体观测数分布:\n{obs_per_entity.value_counts().sort_index()}")

    # 构造平衡子样本
    balanced_ids = obs_per_entity[obs_per_entity == T_max].index
    df_balanced  = df[df[entity_col].isin(balanced_ids)]
    print(f"\n平衡面板保留: {len(balanced_ids)}/{df[entity_col].nunique()} 个体")
    print(f"观测量: {len(df_balanced)}/{len(df)} ({len(df_balanced)/len(df)*100:.1f}%)")

    return df_balanced

df_balanced = check_panel_balance(df)
```

---

### Section: 插值风险警告

**三重警告（任何插值操作前必须阅读）：**

#### 警告1：信息失真
插值会平滑掉真实的波动。线性插值假设变量在缺失期间匀速变化，但现实中企业财务指标、宏观变量可能存在跳跃、波动或结构性转折。插值后的数据**会低估真实波动**，影响变量的标准差和相关结构。

```python
# 检验：插值前后变量波动性对比
print("插值前后标准差对比:")
print(f"  revenue（原始）: {df['revenue'].std():.4f}")
print(f"  revenue_filled（插值后）: {df['revenue_filled'].std():.4f}")
# 若插值后标准差显著小于原始 → 信息失真严重
```

#### 警告2：低估标准误
填充数据减少样本变异，标准误被人为缩小。插值创造了"伪观测"，这些观测不携带独立信息，但回归程序会将其当做真实观测处理，导致：
- 有效样本量被高估
- 标准误被低估（过度拒绝H0）
- 置信区间过窄

**建议：** 若必须插值，在稳健性检验中使用原始非平衡面板对比结果。

#### 警告3：引入测量误差
- 对MAR（随机缺失）插值 → 可能引入新误差，尤其当预测模型不准确时
- 对MNAR（非随机缺失）插值 → 会**加剧偏误**，因为缺失本身携带信息

```r
# R: 插值操作必须记录并在论文中声明
# 示例声明文字（附录或数据节）：
# "对于连续缺失不超过2期的变量X，采用线性插值填补（共填补XX个观测，
# 占总样本的X.X%）。稳健性检验中使用原始非平衡面板，结果（附录表X）
# 与主回归一致，说明插值处理不影响核心结论。"

# 插值诊断：标注哪些值是插值得到的
library(dplyr)
df <- df %>%
  arrange(entity_id, year) %>%
  group_by(entity_id) %>%
  mutate(
    x_orig  = x,
    x_filled = zoo::na.approx(x, maxgap = 2, na.rm = FALSE),
    x_imputed_flag = (!is.na(x_filled)) & is.na(x_orig)  # TRUE=插值产生
  ) %>%
  ungroup()

cat("插值比例:", mean(df$x_imputed_flag, na.rm = TRUE) * 100, "%\n")
# 插值比例 > 10% → 极度谨慎，考虑改用Heckman纠正
```
