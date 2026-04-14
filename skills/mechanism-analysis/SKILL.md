---
name: mechanism-analysis
description: "机制分析，默认推荐两步法，兼容三步法、分组回归与调节效应"
---

# 机制分析（Mechanism Analysis）

## 概述

机制分析的目标不是机械地“分解系数”，而是解释：处理变量 `D` 为什么会影响结果变量 `Y`。本 skill 采用**渐进式**设计：先用识别要求最低、最稳妥的方法完成主论证；只有在研究问题和识别条件都允许时，才进入量化间接效应或处理机制变量内生性的进阶路径。

> **方法论立场声明**
>
> 传统中介效应逐步法（尤其是 Step3: `Y ~ D + M`）在经济学因果推断中存在根本性争议。核心问题是：即使 `D` 是外生的，中介变量 `M` 往往不是随机分配的，仍可能受到遗漏变量、反向因果和测量误差影响。因此，Step3 中 `M` 的系数，以及由此构造的“间接效应”“中介百分比”，通常**不具有明确的因果含义**。
>
> 因此，本 skill 的默认推荐顺序是：
> 1. **默认：两步法**（先确认 `D → Y`，再检验 `D → M`，并用理论/文献支撑 `M → Y`）
> 2. **兼容：三步法**（仅用于回应传统审稿习惯，定位为描述性证据）
> 3. **进阶：因果中介 / IV-mediation / 残差法**（见 `ADVANCED.md`）
>
> 简言之：**先识别，再量化；先主论证，再补充审稿人偏好的表格。**

---

## 适用条件

机制分析适用于以下场景：
- 你已经有可信的主效应识别（如 OLS + 强识别、DID、RDD、IV、SCM）
- 你想解释 `D` 影响 `Y` 的具体传导路径
- 你能提出一个具有经济学含义的候选机制变量 `M`
- 你知道 `M` 是中介、调节变量、前定变量，还是可能的坏控制

**不适用或需谨慎的场景：**
- 主效应本身尚未识别清楚
- `M` 明显可能是对撞变量（collider）
- `M` 与 `Y` 同时受共同遗漏变量影响，但你又试图用 OLS 直接解释 `M → Y`
- 研究者只是“看到数据里有个会动的变量”就将其包装成机制

---

## 渐进式路线图

```text
Level 1（默认主路径）
  DAG 判断 M 的角色
  → 两步法
  → 完成主机制论证

Level 2（兼容审稿人）
  两步法
  → 完整三步法 / 三步法变体
  → 明确标注为描述性补充

Level 3（需要更强量化）
  因果中介分析 / IV-mediation / 残差法
  → 见 ADVANCED.md
```

---

## M 的角色 → 方法选择 → 控制策略

| M 的 DAG 角色 | 典型结构 | 推荐方法 | 控制策略 |
|---|---|---|---|
| **中介变量** | `D → M → Y` | 默认用**两步法**；如需量化见 `ADVANCED.md` | 不要在“总效应回归”里把 `M` 当普通控制变量 |
| **混淆因子** | `M → D` 且 `M → Y` | 主回归必须控制 | **好控制**，应纳入 |
| **对撞变量** | `D → M ← Y` | 不应做机制变量 | **绝对不要控制** |
| **前定变量** | `M` 在处理前已确定 | 可做控制变量或异质性分组 | 通常可控制，但它不是中介 |
| **调节变量** | `D × M → Y` | 交互项法 / 分组回归 | 仅在 `M` 不被 `D` 影响时使用 |

---

## 方法对照表

| 方法 | 推荐程度 | 能回答什么 | 主要局限 | 适用期刊/场景 |
|---|---|---|---|---|
| **两步法** | **默认推荐** | `D` 是否影响机制变量 `M` | 不直接量化“中介占比” | 经济学顶刊、中文经管顶刊 |
| **完整三步法** | 兼容型 | 展示加入 `M` 后 `D` 系数如何变化 | **`M` 的系数可能有偏；中介百分比通常无明确因果含义** | 管理学常见，经济学慎用 |
| **三步法变体（Step3 仅 M）** | 补充型 | 规避 `D` 与 `M` 共线性 | 无法判断部分/完全中介，`M` 仍可能内生 | 中文实证论文常见补充 |
| **分组回归** | 条件推荐 | 某类前定环境是否强化/削弱主效应 | 更像异质性，不等于中介 | 调节效应、环境条件 |
| **交互项法** | 条件推荐 | `M` 是否调节 `D → Y` 的强度 | 若 `M` 受 `D` 影响则解释失效 | 调节效应分析 |

---

## 前置：DAG 自检

在开始之前，先回答四个问题：

1. `M` 是否发生在 `D` 之后、`Y` 之前？
2. `M` 会不会同时受到 `Y` 的反向影响？
3. `M` 是否本质上是处理前就确定的前定变量？
4. `M` 是否可能是 `D` 和 `Y` 的共同结果（collider）？

如果第 4 个问题答案是“是”，则应停止把 `M` 当机制变量处理。

---

## 方法 1：两步法（默认推荐）

> **为什么两步法是充分的？**
>
> 两步法的逻辑基础是：
> - 基准回归已经识别了 `D → Y` 的因果效应；
> - 机制回归识别了 `D → M` 的因果效应；
> - `M → Y` 的关系由经济理论、制度背景和既有文献支撑。
>
> 这三部分合在一起，就构成了完整的 `D → M → Y` 机制论证。
> 不需要在同一个 OLS 回归里“验证” `M → Y`，因为 `M` 往往不是随机分配的。

### 适用场景
- 你的核心目标是“说明机制是否存在”，而不是精确分解中介比例
- `M → Y` 的逻辑已经有扎实的理论和既有文献支持
- 你希望主结论尽量避免坏控制问题

### 标准流程

```text
Step 1: 基准回归确认 D → Y
Step 2: 机制回归检验 D → M
Step 3: 结合经济理论与既有文献，论证 M → Y
```

### R 代码模板（适用于面板 / DID）

```r
library(fixest)

mechanism_two_step_r <- function(df, outcome, mediator, did_var,
                                 controls = NULL,
                                 unit_id = "unit_id",
                                 time = "time") {
  ctrl_str <- if (!is.null(controls) && length(controls) > 0) {
    paste("+", paste(controls, collapse = "+"))
  } else ""
  clust <- as.formula(paste0("~", unit_id))

  # Step 1: Y ~ D
  f1 <- as.formula(sprintf("%s ~ %s %s | %s + %s",
                           outcome, did_var, ctrl_str, unit_id, time))
  res_y <- feols(f1, data = df, cluster = clust)

  # Step 2: M ~ D
  f2 <- as.formula(sprintf("%s ~ %s %s | %s + %s",
                           mediator, did_var, ctrl_str, unit_id, time))
  res_m <- feols(f2, data = df, cluster = clust)

  etable(res_y, res_m,
         headers = c("Baseline: Y~D", "Mechanism: M~D"),
         keep = did_var,
         title = "机制分析：两步法")

  list(step1 = res_y, step2 = res_m)
}
```

### 结果解读模板

```text
基准回归显示，处理变量 D 显著影响了结果变量 Y。
机制回归进一步表明，D 对候选机制变量 M 的影响显著。
结合既有理论与文献中关于 M 影响 Y 的充分证据，结果支持“D 通过 M 影响 Y”的机制解释。
```

---

## 方法 2：完整三步法（兼容传统审稿人）

> **⚠️ 方法论缺陷警告（非审稿偏好问题）**
>
> Step3 将 `M` 加入结果方程存在两个根本性问题：
>
> 1. **坏控制变量问题**：若 `M` 受 `D` 影响，则把 `M` 放入回归会阻断间接路径，
>    此时 `D` 的系数不再是总效应；
> 2. **`M` 的内生性问题**：即使 `D` 是外生的，`M` 往往仍不是随机分配的；
>    若存在同时影响 `M` 和 `Y` 的遗漏变量，则 `M` 的系数和“中介占比”都会有偏。
>
> **结论**：完整三步法可以跑，但应在论文中明确定位为**描述性补充证据**，
> 主结论仍应依赖两步法或更强识别方法。

### 标准流程

```text
Step 1: Y ~ D
Step 2: M ~ D
Step 3: Y ~ D + M
```

### R 代码模板

```r
library(fixest)

mechanism_three_step_r <- function(df, outcome, mediator, did_var,
                                   controls = NULL,
                                   unit_id = "unit_id",
                                   time = "time") {
  ctrl_str <- if (!is.null(controls) && length(controls) > 0) {
    paste("+", paste(controls, collapse = "+"))
  } else ""
  clust <- as.formula(paste0("~", unit_id))

  f1 <- as.formula(sprintf("%s ~ %s %s | %s + %s",
                           outcome, did_var, ctrl_str, unit_id, time))
  f2 <- as.formula(sprintf("%s ~ %s %s | %s + %s",
                           mediator, did_var, ctrl_str, unit_id, time))
  f3 <- as.formula(sprintf("%s ~ %s + %s %s | %s + %s",
                           outcome, did_var, mediator, ctrl_str, unit_id, time))

  res_s1 <- feols(f1, data = df, cluster = clust)
  res_s2 <- feols(f2, data = df, cluster = clust)
  res_s3 <- feols(f3, data = df, cluster = clust)

  etable(res_s1, res_s2, res_s3,
         headers = c("Step1: Y~D", "Step2: M~D", "Step3: Y~D+M"),
         keep = c(did_var, mediator),
         title = "机制分析：完整三步法",
         notes = "Step3 仅作描述性补充，不应将系数变化直接解释为因果中介效应。")

  list(step1 = res_s1, step2 = res_s2, step3 = res_s3)
}
```

### 使用口径

```text
若审稿人要求报告传统中介效应逐步法，本文将其作为补充分析提供。
考虑到机制变量 M 可能存在坏控制和内生性问题，三步法结果仅作描述性证据，
主结论仍依赖两步法和理论论证。
```

---

## 方法 3：三步法变体（Step3 只放 M）

> **定位：** 这是一个“缓解共线性”的兼容型变体，不是识别升级。
>
> 三步法变体规避了 `D` 与 `M` 的共线性问题，但 **`M` 的内生性问题依然存在**。
> 如果存在同时影响 `M` 和 `Y` 的遗漏变量，Step3 变体中 `M` 的系数同样不是因果效应。
>
> 同时，因为 Step3 不再放 `D`，该方法**无法判断部分中介 vs 完全中介**。

### R 代码模板

```r
three_step_variant_r <- function(df, outcome, mediator, did_var,
                                 controls = NULL,
                                 unit_id = "unit_id", time = "time") {
  library(fixest)

  ctrl_str <- if (!is.null(controls) && length(controls) > 0) {
    paste("+", paste(controls, collapse = "+"))
  } else ""
  clust <- as.formula(paste0("~", unit_id))

  # Step1: Y ~ D
  res_s1 <- feols(as.formula(sprintf("%s ~ %s %s | %s + %s",
                                     outcome, did_var, ctrl_str, unit_id, time)),
                  data = df, cluster = clust)

  # Step2: M ~ D
  res_s2 <- feols(as.formula(sprintf("%s ~ %s %s | %s + %s",
                                     mediator, did_var, ctrl_str, unit_id, time)),
                  data = df, cluster = clust)

  # Step3 变体: Y ~ M（不放 D）
  res_s3v <- feols(as.formula(sprintf("%s ~ %s %s | %s + %s",
                                      outcome, mediator, ctrl_str, unit_id, time)),
                   data = df, cluster = clust)

  etable(res_s1, res_s2, res_s3v,
         headers = c("Step1: Y~D", "Step2: M~D", "Step3v: Y~M (无D)"),
         keep = c(did_var, mediator),
         title = "机制分析：三步法变体（Step3 仅含 M）",
         notes = "Step3 不放 D 以缓解共线性；代价是无法区分部分/完全中介，且 M 仍可能内生。")
}
```

### 何时可用
- `D` 与 `M` 高度相关，导致 `Y ~ D + M` 中标准误异常大
- 审稿人或期刊习惯要求“三步法表格”
- 你愿意在文中明确承认该方法仅为补充证据

---

## 方法 4：分组回归（更像异质性，不等于中介）

### 适用场景
- 候选变量 `M` 是处理前就确定的环境条件
- 你关心“在高/低 `M` 条件下，处理效应是否不同”
- `M` 更像调节变量，而不是中介变量

### R 代码模板

```r
library(fixest)

heterogeneity_by_group_r <- function(df, outcome, did_var, group_var,
                                     controls = NULL,
                                     unit_id = "unit_id", time = "time") {
  ctrl_str <- if (!is.null(controls) && length(controls) > 0) {
    paste("+", paste(controls, collapse = "+"))
  } else ""
  clust <- as.formula(paste0("~", unit_id))

  df$group_high <- as.integer(df[[group_var]] >= median(df[[group_var]], na.rm = TRUE))

  res_low <- feols(as.formula(sprintf("%s ~ %s %s | %s + %s",
                                      outcome, did_var, ctrl_str, unit_id, time)),
                   data = subset(df, group_high == 0), cluster = clust)

  res_high <- feols(as.formula(sprintf("%s ~ %s %s | %s + %s",
                                       outcome, did_var, ctrl_str, unit_id, time)),
                    data = subset(df, group_high == 1), cluster = clust)

  etable(res_low, res_high,
         headers = c("Low group", "High group"),
         keep = did_var,
         title = "分组回归：高低组异质性")

  list(low = res_low, high = res_high)
}
```

### 解释提醒
- 两组系数“看起来不一样”并不够；最好补做组间差异检验。
- 如果 `M` 本身受 `D` 影响，那么分组回归就不是标准的调节效应设计。

---

## 方法 5：交互项法（仅用于调节变量）

> **关键前提：** 交互项法要求 `M` 是调节变量，而不是中介变量。
> 如果 `M` 受 `D` 影响，则 `D × M` 的解释会变得混乱。

```r
library(fixest)

moderation_interaction_r <- function(df, outcome, did_var, moderator,
                                     controls = NULL,
                                     unit_id = "unit_id", time = "time") {
  ctrl_str <- if (!is.null(controls) && length(controls) > 0) {
    paste("+", paste(controls, collapse = "+"))
  } else ""
  clust <- as.formula(paste0("~", unit_id))

  f <- as.formula(sprintf("%s ~ %s * %s %s | %s + %s",
                          outcome, did_var, moderator, ctrl_str, unit_id, time))
  res <- feols(f, data = df, cluster = clust)
  summary(res)
  res
}
```

---

## 坏控制快速检查

```text
如果你正准备把 M 放进结果方程，请先问：
1. M 是否受 D 影响？
   是 → M 可能是中介，放进去会阻断间接路径
2. M 是否也可能受 Y 或遗漏变量 U 影响？
   是 → M 还可能内生
3. M 是否是 D 和 Y 的共同结果？
   是 → collider，不能控制
```

---

## 常见误区

> **误区 1：Step3 里 D 的系数变小 = 机制成立**
>
> 纠正：不能直接这样解释。把 `M` 放入 Step3 后，`M` 既可能是坏控制，也可能内生；
> `D` 的系数变化不等于“间接效应”。

> **误区 2：两步法只要 `D → M` 显著就够了**
>
> 纠正：还需要经济理论和既有文献支持 `M → Y` 的逻辑成立。
> 否则，`D` 对 `M` 的影响可能只是一个副产品，不一定是真机制。

> **误区 3：分组回归两组系数大小不同 = 机制成立**
>
> 纠正：必须检验组间差异是否统计显著，而不是只看各组是否显著。

> **误区 4：交互项法里的 `M` 可以是中介变量**
>
> 纠正：交互项法要求 `M` 是调节变量，即 `M` 不应被 `D` 影响。

> **误区 5：三步法可以可靠报告“中介百分比”**
>
> 纠正：在传统三步法中，这个比例通常没有稳定的因果含义。
> 若必须量化，应转向 `ADVANCED.md` 中的因果中介或 IV-mediation 框架。

---

## 检验清单

| 检验 | 方法 | 通过标准 |
|------|------|----------|
| DAG 角色识别 | 判断 `M` 是中介/混淆/对撞/调节 | 不把 collider 误当机制 |
| 主效应存在 | 基准回归 `Y ~ D` | `D` 对 `Y` 显著或方向稳定 |
| 机制路径存在 | 机制回归 `M ~ D` | `D` 对 `M` 显著或方向稳定 |
| 理论支撑 | 文献 + 制度背景 | `M → Y` 逻辑清晰 |
| 三步法补充 | `Y ~ D + M` | 仅作描述性，不作强因果解释 |
| 分组差异 | 分组回归 / 交互项 | 最好有组间差异检验 |

---

## 写作模板

### 主文推荐写法（两步法）

```text
本文采用两步法检验作用机制。首先，基准回归表明处理变量 D 对结果变量 Y 具有显著影响。
其次，以候选机制变量 M 为被解释变量的回归显示，D 显著影响 M。
结合关于 M 影响 Y 的理论逻辑与既有文献证据，结果支持 D 通过 M 影响 Y 的机制解释。
```

### 审稿人要求三步法时的写法

```text
作为补充，本文进一步报告传统三步法结果。
需要说明的是，考虑到机制变量 M 可能存在坏控制和内生性问题，
三步法结果仅作为描述性证据，不作为本文机制识别的主要依据。
```

---

## Few-Shot：大数据试验区与就业增长

**研究问题：** 大数据试验区设立是否促进城市就业增长？

- 处理变量 `D`：`Bigdata`（是否设立大数据试验区）
- 结果变量 `Y`：`Labor`（就业增长）
- 候选机制变量：
  - `M1 = Newfirm`：新注册企业数量，代表岗位创造效应
  - `M2 = TFP`：企业全要素生产率，代表市场扩大效应
  - `M3 = Bank`：银行网点数量，代表融资约束缓解或金融环境

### 方法匹配

- **M1：两步法**
  - 检验 `Bigdata → Newfirm` 是否显著；
  - `Newfirm → Labor` 的关系由创业与就业创造文献支撑。

- **M2：完整三步法 / 进阶因果中介**
  - 若审稿人要求展示“传导强度”，可补充三步法；
  - 若需要更严格量化，转到 `ADVANCED.md` 使用因果中介。

- **M3：分组回归 / 交互项法**
  - 若 `Bank` 是处理前金融环境，更适合作为调节变量；
  - 此时它不是中介，而是异质性来源。

### 推荐写作顺序

```text
主论证：两步法
补充表格：三步法或三步法变体
进阶量化：ADVANCED.md 中的因果中介 / IV-mediation / 残差法
```

---

## 输出规范

```text
output/
  mechanism_baseline.csv
  mechanism_step2_mediator.csv
  mechanism_three_step.csv
  mechanism_group_heterogeneity.csv
  mechanism_writeup_notes.md
```
