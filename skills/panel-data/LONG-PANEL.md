# 长面板分析（T >> N）

> **加载条件：** T > 20–30 且 N < T，或 Step 0 路由判定为长面板 / 边界面板。
> 长面板的核心风险是**伪回归**：非平稳变量直接回归会产生虚假显著性。

---

## 触发条件

```
是否需要长面板流程？
│
├─ T > 30, N < T → 强长面板特征，必须走本流程
│
├─ 20 < T < 30 → 边界情形
│   ├─ 做单位根检验
│   ├─ 平稳 → 可回到 SKILL.md 短面板流程
│   └─ 非平稳 → 走本流程
│
└─ T ≤ 20 → 短面板，回到 SKILL.md
```

---

## Step 1: 面板单位根检验

**必须在回归前完成。** H₀ 通常为"所有个体均有单位根"。

### 第一代检验（假设截面独立）

```r
library(plm)

# LLC（Levin-Lin-Chu）：假设所有个体有相同的自回归系数
# H0: 所有个体有单位根（ρ=1）
purtest(Y ~ 1, data = df, index = c("entity_id", "time"),
        test = "levinlin", exo = "intercept", lags = "AIC", pmax = 4)

# IPS（Im-Pesaran-Shin）：允许异质自回归系数（更灵活）
purtest(Y ~ 1, data = df, index = c("entity_id", "time"),
        test = "ips", exo = "intercept", lags = "AIC", pmax = 4)

# Fisher-ADF（基于个体 ADF 检验的 p 值组合）
purtest(Y ~ 1, data = df, index = c("entity_id", "time"),
        test = "madwu", exo = "intercept", lags = "AIC", pmax = 4)

# Hadri（H0: 所有个体平稳 — 方向相反！）
purtest(Y ~ 1, data = df, index = c("entity_id", "time"),
        test = "hadri", exo = "intercept")
# Hadri 拒绝 → 存在单位根
```

### 第二代检验（允许截面相关）

```r
# Pesaran CIPS（推荐：允许截面相关 + 异质自回归系数）
# install.packages("xtunitroot") 或手写
library(plm)

# 先检验截面相关
pcdtest(plm(Y ~ X, data=df, index=c("entity_id","time"), model="within"), test="cd")
# 若截面相关显著 → 必须用第二代检验

# CIPS 检验（需 punitroots 包或手写）
# install.packages("punitroots")
library(punitroots)
pescadf(Y ~ 1, data = df, index = c("entity_id", "time"), lags = 2)
```

### 单位根检验决策

```
LLC + IPS + Fisher 三者结果：
│
├─ 全部拒绝 H0 → 变量平稳（I(0)），可直接回归
│
├─ 全部不拒绝 → 变量非平稳（I(1)），做一阶差分后重新检验
│   ├─ 差分后平稳 → 确认 I(1)，进入协整检验
│   └─ 差分后仍非平稳 → I(2)，面板方法难以处理
│
├─ 结果不一致 → 以 IPS 为主（最灵活），辅以 Hadri 交叉验证
│
└─ 截面相关显著 → 以 CIPS 结果为准
```

---

## Step 2: 面板协整检验

**仅当变量为 I(1) 时需要。** 检验变量间是否存在长期均衡关系。

### Pedroni 检验（异质协整）

```r
# install.packages("plm")
library(plm)

# Pedroni（7个统计量，允许异质协整关系）
# 需要手动实现或用 cointReg / urca 辅助
# 以下用 pco 包
# install.packages("pco")
library(pco)
pedroni_test <- pedroni(Y ~ X, data = df, index = c("entity_id", "time"),
                         model = "within", type = "e")
print(pedroni_test)
# 大部分统计量拒绝 H0 → 存在协整关系
```

### Kao 检验（同质协整）

```r
# Kao 检验（假设同质协整系数）
# 更简单但限制更强
library(plm)
kao_test <- purtest(residuals(plm(Y ~ X, data=df,
                    index=c("entity_id","time"), model="within")),
                    test = "levinlin", exo = "none")
print(kao_test)
```

### Westerlund 检验（推荐，允许截面相关）

```r
# install.packages("plm") — Westerlund ECM-based test
# 更现代、更稳健
library(plm)

# Westerlund (2007) 四个统计量
# Ga, Gt: 组均值统计量（允许异质）
# Pa, Pt: 面板统计量（同质 H1）

# 需要 xtwest 或手写；R 中可用以下简化版：
# 基于误差修正模型的思路
ecm_res <- plm(diff(Y) ~ lag(Y, 1) + lag(X, 1) + diff(X),
               data = df, index = c("entity_id", "time"), model = "within")
summary(ecm_res)
# lag(Y,1) 的系数显著为负 → 支持协整（误差修正机制存在）
```

### 协整检验决策

```
协整检验结果：
│
├─ 存在协整 → 估计长期均衡关系
│   ├─ FMOLS（Fully Modified OLS）
│   ├─ DOLS（Dynamic OLS）
│   └─ 面板误差修正模型（PECM）→ Step 3
│
└─ 不存在协整 → 变量无长期关系
    ├─ 对所有变量一阶差分后用短面板流程
    └─ 差分模型只能估计短期效应
```

---

## Step 3: 长期均衡估计

### FMOLS（Fully Modified OLS）

```r
# install.packages("plm")
library(plm)

# FMOLS: 修正序列相关和内生性偏误的协整回归
# 需要 cointReg 包
# install.packages("cointReg")
library(cointReg)

# 逐个体 FMOLS → 取均值（Mean Group 思想）
entities <- unique(df$entity_id)
fmols_coefs <- sapply(entities, function(id) {
  sub <- df[df$entity_id == id, ]
  sub <- sub[order(sub$time), ]
  tryCatch({
    res <- cointRegFM(y = sub$Y, x = sub[, c("X", "control1")], deter = "constant")
    res$theta[, 1]
  }, error = function(e) rep(NA, 2))
})
# 均值组估计
apply(fmols_coefs, 1, mean, na.rm = TRUE)
```

### DOLS（Dynamic OLS）

```r
# DOLS: 加入X的超前和滞后项修正内生性
library(plm)

# Panel DOLS
dols_res <- plm(Y ~ X + control1 +
                lag(X, 1) + lead(X, 1) +     # X 的滞后和超前
                lag(control1, 1) + lead(control1, 1),
                data = df, index = c("entity_id", "time"), model = "within")
summary(dols_res)
# X 的系数即为长期均衡系数
```

### 面板误差修正模型（PECM）

```r
# PECM: 短期动态 + 长期均衡
library(plm)

# Step 1: 估计长期关系（协整回归），获取残差
lr_res <- plm(Y ~ X + control1,
              data = df, index = c("entity_id", "time"), model = "within")
df$ecm_resid <- c(NA, residuals(lr_res))  # 滞后残差

# Step 2: 误差修正模型
pecm <- plm(diff(Y) ~ lag(ecm_resid, 1) +   # 误差修正项（应显著为负）
            diff(X) + diff(control1),         # 短期效应
            data = df, index = c("entity_id", "time"), model = "within")
summary(pecm)
# lag(ecm_resid) 系数 = 调整速度（应在 -1 到 0 之间）
# diff(X) 系数 = 短期效应
# 长期效应来自协整回归
```

---

## Step 4: 偏差校正 LSDV（LSDVC）

当 T 中等（15–30）且模型含滞后因变量，Nickell 偏误不可忽略但 GMM 因 N 小而工具变量不足时，LSDVC 是替代方案。

```r
# install.packages("plm")
library(plm)

# LSDVC: Bruno (2005) 偏差校正
# 初始一致估计量选择：AB = Arellano-Bond, BB = Blundell-Bond, AH = Anderson-Hsiao
lsdvc_res <- plm(Y ~ lag(Y, 1) + X + control1,
                 data = df, index = c("entity_id", "time"), model = "within")

# 手动偏差校正（基于 Nickell bias 公式）
T_bar <- mean(table(df$entity_id))
rho_lsdv <- coef(lsdvc_res)["lag(Y, 1)"]
rho_corrected <- rho_lsdv + (1 + rho_lsdv) / T_bar  # 一阶近似修正
cat(sprintf("LSDV ρ = %.4f → 修正后 ρ ≈ %.4f (T_bar=%.0f)\n",
            rho_lsdv, rho_corrected, T_bar))

# 或使用 xtlsdvc 的 R 移植版
# install.packages("fixest")  # fixest 本身不含 LSDVC，需配合 bootstrap
```

---

## Step 5: 截面相关处理

长面板中截面相关几乎不可避免（共同的全球冲击、区域溢出）。

### Pesaran (2006) CCE 估计量

```r
# CCE: 加入截面均值作为不可观测共同因子的代理
library(plm)

# 计算截面均值
df <- df %>%
  group_by(time) %>%
  mutate(Y_bar = mean(Y, na.rm = TRUE),
         X_bar = mean(X, na.rm = TRUE)) %>%
  ungroup()

# CCE-MG（Mean Group）: 允许异质斜率
cce_results <- lapply(unique(df$entity_id), function(id) {
  sub <- df[df$entity_id == id, ]
  tryCatch({
    res <- lm(Y ~ X + control1 + Y_bar + X_bar, data = sub)
    coef(res)["X"]
  }, error = function(e) NA)
})
cce_mg_beta <- mean(unlist(cce_results), na.rm = TRUE)
cat(sprintf("CCE-MG β_X = %.4f\n", cce_mg_beta))

# CCE-Pooled: 同质斜率
cce_pool <- plm(Y ~ X + control1 + Y_bar + X_bar,
                data = df, index = c("entity_id", "time"), model = "within")
summary(cce_pool)
```

### Driscoll-Kraay 标准误

```r
# 对既有 FE 结果使用 DK SE
library(plm); library(lmtest); library(sandwich)
fe_res <- plm(Y ~ X + control1, data = df,
              index = c("entity_id", "time"), model = "within")
coeftest(fe_res, vcov = vcovSCC(fe_res, type = "HC3", maxlag = floor(T^(1/4))))
```

---

## 检验清单

| 步骤 | 检验 | 通过标准 |
|------|------|---------| 
| Step 1 | 单位根（LLC + IPS + Fisher） | 一致拒绝 → 平稳；不拒绝 → 差分 |
| Step 1 | 截面相关（Pesaran CD） | 显著 → 用第二代检验（CIPS） |
| Step 2 | 协整（Pedroni / Westerlund） | 存在 → FMOLS/DOLS/PECM |
| Step 3 | ECM 调整系数 | 应在 (-1, 0) 之间 |
| Step 5 | 截面相关处理 | CCE 或 DK SE |

---

## 常见错误

> **错误1：长面板直接跑 FE 不检验平稳性**
> 两个 I(1) 变量回归 R² 可高达 0.9 但完全是伪回归。长面板必须先做单位根。

> **错误2：用短面板 GMM 处理长面板**
> GMM 依赖大 N 渐近理论，N 小时工具变量数量不足，Hansen 检验失效。长面板用 FMOLS/DOLS/LSDVC。

> **错误3：忽略截面相关**
> 国家/地区面板几乎一定存在截面相关（全球金融危机、大宗商品价格）。忽略会严重低估标准误。

> **错误4：差分后忘记检验协整**
> 如果变量协整，一阶差分会丢失长期均衡信息，应使用 PECM 同时估计短期和长期效应。

---

## Estimand 声明

**声明模板：**
> "本文使用面板误差修正模型（PECM）估计变量间的长期均衡关系和短期动态调整。面板单位根检验（IPS）确认变量为 I(1)，Westerlund 协整检验确认存在长期均衡关系。长期弹性为 β_LR = [值]，短期弹性为 β_SR = [值]，误差修正速度为 λ = [值]（每期向均衡调整 [值×100]%）。标准误使用 Driscoll-Kraay 修正以处理截面相关。"
