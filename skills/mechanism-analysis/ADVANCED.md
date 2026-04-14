# 机制分析：进阶方法（ADVANCED）

> 按需加载。本文档面向需要**量化间接效应**、处理**机制变量内生性**，
> 或需要与 DID / RDD / IV 主设计保持一致的高级机制分析场景。

---

## 方法选择指引

当出现以下任一情况时，才建议进入本文件：

```text
1. 审稿人明确要求量化“间接效应”或“中介占比”
2. 你怀疑机制变量 M 明显内生
3. 你的主设计本身是 IV / DID / RDD，需要更一致的机制框架
4. 你有多个机制变量，希望区分各条路径
```

**默认原则：** 若 `SKILL.md` 中的两步法已经足够支撑论文主结论，就不要过度升级到高识别负担的方法。

---

## 渐进式路线图

```text
Level 1：只需说明机制存在
  → 回到 SKILL.md 的两步法

Level 2：审稿人要求量化间接效应
  → 因果中介分析（mediation + medsens）

Level 3：机制变量 M 明显内生
  → IV-Mediation 或残差法

Level 4：多个机制并行
  → 多机制框架 + 机制变量平行趋势 / 稳健性
```

---

## 1. 因果中介分析（mediation + medsens）

### 适用场景
- 你希望估计 ACME（平均因果中介效应）和 ADE（平均直接效应）
- 你愿意接受顺序可忽略性（sequential ignorability）假设
- 你能对 `M` 与 `Y` 之间潜在遗漏变量做敏感性分析

### 方法说明

> 因果中介分析比传统三步法更规范，但它并不“自动解决内生性”。
> 它依赖比两步法更强的识别假设，因此必须报告 `medsens()` 敏感性分析。

### R 代码模板

```r
library(mediation)

causal_mediation_r <- function(df, outcome, mediator, treat, controls = NULL) {
  ctrl_str <- if (!is.null(controls) && length(controls) > 0) {
    paste("+", paste(controls, collapse = "+"))
  } else ""

  med_fit <- lm(as.formula(sprintf("%s ~ %s %s", mediator, treat, ctrl_str)), data = df)
  out_fit <- lm(as.formula(sprintf("%s ~ %s + %s %s", outcome, treat, mediator, ctrl_str)), data = df)

  med_res <- mediate(med_fit, out_fit, treat = treat, mediator = mediator,
                     boot = TRUE, sims = 1000)
  summary(med_res)

  sens_res <- medsens(med_res, effect.type = "indirect")
  plot(sens_res)

  list(mediation = med_res, sensitivity = sens_res)
}
```

### 报告模板

```text
本文进一步采用因果中介分析估计间接效应，并通过 medsens() 检验结果对顺序可忽略性违背的敏感程度。
若 ACME 仅在极弱的遗漏变量假设下才成立，则不应将其作为强机制证据。
```

---

## 2. IV-Mediation（Dippel, Ferrara & Heblich 2020）

### 适用场景
- 处理变量 `D` 通过工具变量 `Z` 识别
- 机制变量 `M` 也可能是内生的
- 你希望在 IV 框架下讨论中介路径

### 方法定位

> IV-Mediation 的价值在于：它不是把 `M` 当作外生控制变量塞进 OLS，
> 而是在工具变量框架下讨论因果中介分解。
> 如果你的论文主识别已经是 IV，这通常比传统三步法更一致。

### 实务实现
- **Stata**：`ivmediate`（最成熟的标准实现）
- **R**：可用 `ivreg` + 中介思路做近似工作流，但应明确说明不是原生命令复现

### R 近似模板

```r
library(AER)

iv_mediation_r <- function(df, y, d, m, z, controls = NULL) {
  ctrl_str <- if (!is.null(controls) && length(controls) > 0) {
    paste("+", paste(controls, collapse = "+"))
  } else ""

  # 第一阶段：Z -> D
  fit_d <- ivreg(
    as.formula(sprintf("%s ~ %s %s | %s %s", d, d, ctrl_str, z, ctrl_str)),
    data = df
  )
  df$d_hat <- fitted(fit_d)

  # 中介方程：M ~ D_hat
  fit_m <- lm(as.formula(sprintf("%s ~ d_hat %s", m, ctrl_str)), data = df)

  # 结果方程：Y ~ D_hat + M
  fit_y <- lm(as.formula(sprintf("%s ~ d_hat + %s %s", y, m, ctrl_str)), data = df)

  list(first_stage = summary(fit_d), mediator = summary(fit_m), outcome = summary(fit_y))
}
```

### 报告提醒

```text
若使用 R 中的近似实现，应明确说明这是基于 IV 逻辑的机制分解工作流，
而非对 Stata ivmediate 的完全等价复现。
```

---

## 3. 残差法（适合 DID / RDD / 高维固定效应）

### 核心思路
先将机制变量和结果变量中可由控制变量、固定效应、趋势解释的部分剔除，
再在净化后的残差上研究机制路径。

### 适用场景
- 主设计为 DID / TWFE / 事件研究
- 固定效应很多，直接在 `Y ~ D + M` 中解释困难
- 你希望机制回归与主回归设定保持一致

### R 模板

```r
library(fixest)

residual_mechanism_r <- function(df, outcome, mediator, did_var,
                                 controls = NULL,
                                 unit_id = "unit_id", time = "time") {
  ctrl_str <- if (!is.null(controls) && length(controls) > 0) {
    paste("+", paste(controls, collapse = "+"))
  } else ""

  # 残差化 Y
  f_y <- as.formula(sprintf("%s ~ %s | %s + %s", outcome, ctrl_str, unit_id, time))
  fit_y <- feols(f_y, data = df)
  df$y_resid <- resid(fit_y)

  # 残差化 M
  f_m <- as.formula(sprintf("%s ~ %s | %s + %s", mediator, ctrl_str, unit_id, time))
  fit_m <- feols(f_m, data = df)
  df$m_resid <- resid(fit_m)

  # 在残差上看 D -> M 与 M -> Y 的关系
  res1 <- feols(m_resid ~ did_var, data = df, cluster = as.formula(paste0("~", unit_id)))
  res2 <- feols(y_resid ~ did_var + m_resid, data = df, cluster = as.formula(paste0("~", unit_id)))

  etable(res1, res2,
         headers = c("M residual ~ D", "Y residual ~ D + M residual"),
         title = "机制分析：残差法")

  list(step1 = res1, step2 = res2)
}
```

### 解释提醒
- 残差法改善的是**设定一致性**与高维固定效应处理，不是凭空解决内生性。
- 如果 `m_resid` 仍受遗漏变量影响，它依然不能被机械解释为因果通道。

---

## 4. 机制变量平行趋势检验（DID 场景）

### 目的
在 DID 场景下，如果你声称 `D → M → Y`，那么最好检查机制变量 `M` 在政策前是否也满足平行趋势。

### R 模板

```r
library(fixest)

mediator_pretrend_r <- function(df, mediator, rel_time, treat_var,
                                unit_id = "unit_id", time = "time") {
  f <- as.formula(sprintf("%s ~ i(%s, %s, ref = -1) | %s + %s",
                          mediator, rel_time, treat_var, unit_id, time))
  res <- feols(f, data = df, cluster = as.formula(paste0("~", unit_id)))
  iplot(res, main = "Pre-trend Check for Mediator")
  res
}
```

### 解读规则
- 政策前 lead 系数应整体不显著
- 如果机制变量在政策前已出现趋势差异，则后续机制解释要弱化

---

## 5. 多机制同时分析

### 适用场景
- 你有多个候选机制：`M1`, `M2`, `M3`
- 它们分别对应不同理论通道
- 你不希望把它们一股脑扔进一个三步法回归里互相“抢解释力”

### 推荐策略

```text
不要：
  Y ~ D + M1 + M2 + M3
  → 容易共线、解释混乱、每个 M 都可能内生

推荐：
  逐个机制单独做两步法
  M1: D → M1
  M2: D → M2
  M3: D → M3
  再分别用理论论证 Mi → Y
```

### 汇总表模板

```r
multi_mechanism_table <- function(results_list) {
  # results_list: 每个元素是一条机制的 step2 回归结果
  return(results_list)
}
```

---

## 前沿标注

### 顺序可忽略性不是默认成立

即使 `D` 是外生的，只要 `M` 不是随机分配，因果中介分析就仍然依赖额外假设。
因此，`mediation` 包的结果不能替代研究者对识别条件的说明。

### IV-Mediation 更适合 IV 主设计

如果主论文本身就是通过 IV 识别 `D → Y`，那么机制部分也应尽量沿用同一识别逻辑，
而不是突然退回到传统三步法 OLS。

### 残差法是“与主设计保持一致”的工具

当主回归含高维固定效应、趋势项或复杂控制变量时，残差法特别有用。
但它的价值是设定一致性，不是自动获得因果识别。

---

## 常见升级错误

> **错误 1：主论文已经能用两步法讲清楚，还强行做 ACME 分解**
>
> 过度升级方法会把论文建立在更强、更脆弱的识别假设上。

> **错误 2：把因果中介分析当成“自动修正三步法”**
>
> 它不是自动修正，而是换了一套更强的识别假设；必须做敏感性分析。

> **错误 3：有工具变量就默认可以做 IV-mediation**
>
> 还要看该工具变量是否满足用于机制分解的识别要求，而不是只对总效应有效。

> **错误 4：多机制一起放进结果方程，谁显著就说谁是主机制**
>
> 这通常只是共线性和内生性共同作用下的产物，不是可靠机制比较。

---

## Few-Shot：大数据试验区与就业增长（进阶版）

假设主结论已通过 DID 识别：大数据试验区设立提高了就业增长。

### 机制 1：新注册企业数量（`Newfirm`）
- 推荐：`SKILL.md` 的两步法
- 理由：创业增加带动就业创造，理论链条清晰

### 机制 2：全要素生产率（`TFP`）
- 若只想说明存在：两步法即可
- 若审稿人要求量化传导：可尝试因果中介分析

### 机制 3：信贷可得性（`Bank`）
- 若 `Bank` 是政策前金融环境：用分组回归/交互项法
- 若 `Bank` 随政策变化：它更像中介，不能直接当调节变量

### 推荐写作层次

```text
主文：两步法 + 机制变量平行趋势
附录：三步法补充表
更高阶附录：因果中介 / 残差法 / IV-mediation
```

---

## 输出规范

```text
output/
  mediation_acme_ade.csv
  mediation_sensitivity.png
  iv_mediation_results.csv
  residual_mechanism_results.csv
  mediator_pretrend.png
  multi_mechanism_summary.csv
```
