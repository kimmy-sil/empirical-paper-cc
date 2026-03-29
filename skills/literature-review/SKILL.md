# literature-review — 文献综述与文献管理技能指南

适用场景：文献搜索、筛选、综述撰写、引用格式管理、贡献定位。触发关键词：文献综述、搜文献、找文献、写综述、contribution statement、BibTeX、引用格式。

---

## 1. 文献搜索策略

### 1.1 主要数据库及适用场景

| 数据库 | 适用场景 | 访问方式 |
|--------|---------|---------|
| Google Scholar | 综合搜索，发现引用关系 | scholar.google.com |
| SSRN | 经济学/金融/管理学最新工作论文 | ssrn.com |
| CNKI | 中文期刊（管理学、经济学） | cnki.net（需账号） |
| Web of Science | SCI/SSCI 精确检索，引用分析 | webofscience.com（需机构账号） |
| NBER Working Papers | 经济学前沿工作论文 | nber.org/papers |
| IDEAS/RePEC | 经济学论文自由检索 | ideas.repec.org |
| AEA Journals | American Economic Review 等 | aeaweb.org |

### 1.2 搜索关键词组合技巧

**基本规则：**
- 用英文搜索国际文献，用中文搜索中文文献
- 组合核心概念词（现象词 + 方法词 + 情境词）
- 先宽后窄：从宽泛关键词发现文献，再用精确词过滤

**关键词构建模板：**
```
主题词 AND 方法词 AND 情境词

示例：
"digital economy" AND "difference-in-differences" AND "productivity"
"企业数字化" AND "双重差分" AND "劳动力"
"platform regulation" AND "causal" AND "China"
```

**常用布尔运算：**
- `AND`：两个词都必须出现
- `OR`：任一词出现（扩大范围，适用于同义词）
- `NOT`：排除某词
- `"exact phrase"`：精确短语匹配
- `*`：通配符（`digit*` 匹配 digital/digitize/digitization）

**Google Scholar 进阶技巧：**
- `author:Acemoglu` — 搜索特定作者
- 引用追踪：点击"被引用次数"看后续引用该文的论文
- `site:nber.org digital economy` — 限定网站搜索
- 时间过滤：限定近5年获取最新文献

---

## 2. 文献筛选漏斗

分三轮筛选，逐步缩小范围。

### 第一轮：初筛（标题+摘要，10–15分钟/篇）

筛选标准：
- 研究主题与本文高度相关（核心变量相似）
- 发表在目标期刊层级（SSCI/CSSCI 及以上，或高被引工作论文）
- 时效性：一般以近15年为主，但经典文献不限

**排除标准：**
- 纯理论模型，无实证（除非本文需要理论支撑）
- 与研究问题关联度低（只有边缘关键词匹配）
- 数据/情境与本文研究无可比性

### 第二轮：精读（方法+结论+贡献，30–60分钟/篇）

重点阅读：
1. 研究问题与本文的关联性
2. 识别策略（因果性如何保证？）
3. 主要发现（系数方向、量级）
4. 与本文的异同（哪里可以比较）

记录表格模板：

| 字段 | 内容 |
|------|------|
| 作者+年份 | |
| 研究问题 | |
| 数据/样本 | |
| 识别方法 | |
| 核心发现 | |
| 与本文关系 | 支持/对比/方法借鉴 |
| 引用理由 | |

### 第三轮：纳入决策

决策标准：
- A类（必引）：与本文高度相关，直接支撑或对比本文发现
- B类（选引）：方法或数据有参考价值
- C类（不引）：关联度不足

**最终文献库建议规模：** 正式纳入综述 25–40篇，引用列表 30–50条

---

## 3. 文献综述写作框架

**核心原则：按研究主题/方法/争论点组织，不按时间列举。**

### 3.1 主题组织法（推荐）

将文献按"回答了什么问题"分组，每组形成一段。

**框架示例（三组）：**

**第一组 — [现象/机制的存在性文献]**
> 该组文献研究[X对Y的关系]是否存在。[Author A, Year] 使用[方法]发现……。[Author B, Year] 在[情境]中得到类似结论……。然而，这些研究主要依赖相关性分析，因果识别有待加强。

**第二组 — [因果机制/中间变量文献]**
> 探讨因果路径的文献表明……。其中，[Author C, Year] 首次通过[准自然实验]……。但这些研究集中于[地区/群体]，缺乏对[本文情境]的关注。

**第三组 — [本文方法相关文献]**
> 在识别策略上，本文借鉴[DID/RDD/IV]方法，该方法由[Author, Year]提出并在[领域]获得广泛应用……。

**过渡到贡献定位：**
> 综合来看，现有文献存在以下不足：其一，……；其二，……。本文试图在上述方向上有所推进。

### 3.2 方法对比组织法

适用于：本文方法创新明显，或现有文献在方法上存在争议。

按方法类别分段：OLS相关研究组 → 工具变量研究组 → DID研究组，指出方法局限，引出本文方法。

### 3.3 争论点组织法

适用于：学术界对某结论存在显著分歧。

> "关于[X效应]，学界存在两种对立观点。一方面，[Author A] 等认为……（支撑证据）。另一方面，[Author B] 等发现相反结论（支撑证据）。本文通过……的准自然实验有助于厘清这一争议。"

---

## 4. 引用格式规范

### 4.1 APA 7（国际英文期刊）

**期刊文章：**
```
Athey, S., & Imbens, G. W. (2022). Design-based analysis in difference-in-differences settings with staggered adoption. Journal of Econometrics, 226(1), 62–79. https://doi.org/10.1016/j.jeconom.2020.10.012
```

**工作论文：**
```
Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods. Journal of Econometrics, 225(2), 200–230.
```

**正文引用：**
- 括号式：`(Athey & Imbens, 2022)`
- 叙述式：`Athey and Imbens (2022) show that...`
- 多作者（3+）：首次 `(Callaway et al., 2021)`

### 4.2 GB/T 7714-2015（中文期刊投稿）

**期刊文章：**
```
作者1, 作者2. 文章标题[J]. 期刊名, 年份, 卷(期): 起始页-终止页.
```

示例：
```
张三, 李四. 数字经济与企业生产率：来自中国的证据[J]. 经济研究, 2023, 58(3): 45-62.
```

**顺序编码制**（多数中文期刊）：正文标注 `[1]`，文末按出现顺序列参考文献。

**著者-出版年制**（部分期刊）：正文标注 `（张三，2023）`。

### 4.3 Chicago Author-Date（部分社科期刊）

```
Acemoglu, Daron, and David Autor. 2011. "Skills, Tasks and Technologies: Implications for Employment and Earnings." In Handbook of Labor Economics, vol. 4B, edited by Orley Ashenfelter and David Card, 1043–1171. Elsevier.
```

正文：`(Acemoglu and Autor 2011, 1050)`

---

## 5. BibTeX 管理规范

### 5.1 条目命名规则

`[FirstAuthorLastname][Year][FirstKeyword]`

示例：
```bibtex
@article{callaway2021did,
  author  = {Callaway, Brantly and Sant'Anna, Pedro H. C.},
  title   = {Difference-in-Differences with Multiple Time Periods},
  journal = {Journal of Econometrics},
  year    = {2021},
  volume  = {225},
  number  = {2},
  pages   = {200--230},
  doi     = {10.1016/j.jeconom.2020.12.001}
}

@article{chernozhukov2018dml,
  author  = {Chernozhukov, Victor and Chetverikov, Denis and Demirer, Mert
             and Duflo, Esther and Hansen, Christian and Newey, Whitney
             and Robins, James},
  title   = {Double/Debiased Machine Learning for Treatment and Structural Parameters},
  journal = {The Econometrics Journal},
  year    = {2018},
  volume  = {21},
  number  = {1},
  pages   = {C1--C68},
  doi     = {10.1111/ectj.12097}
}

@article{sunAbraham2021,
  author  = {Sun, Liyang and Abraham, Sarah},
  title   = {Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects},
  journal = {Journal of Econometrics},
  year    = {2021},
  volume  = {225},
  number  = {2},
  pages   = {175--199},
  doi     = {10.1016/j.jeconom.2020.09.006}
}
```

### 5.2 常用 BibTeX 条目类型

| 类型 | 用途 |
|------|------|
| `@article` | 期刊文章 |
| `@book` | 专著 |
| `@incollection` | 书章节 |
| `@techreport` | 工作论文/技术报告 |
| `@unpublished` | 未发表手稿 |
| `@misc` | 其他（数据集、报告等） |

### 5.3 LaTeX 引用设置

```latex
% 导言区
\usepackage[authoryear, round]{natbib}  % APA风格
% 或
\usepackage[numbers, sort]{natbib}       % 数字编号风格

% 正文引用命令
\citep{callaway2021did}    % (Callaway and Sant'Anna, 2021)
\citet{callaway2021did}    % Callaway and Sant'Anna (2021)
\citealt{callaway2021did}  % Callaway and Sant'Anna 2021（无括号）

% 文末参考文献
\bibliographystyle{apalike}   % 或 plainnat, chicago, jpe
\bibliography{references}      % 对应 references.bib 文件
```

---

## 6. 贡献定位（Contribution Statement）

贡献声明是引言最后1–2段，需与具体文献挂钩。

### 标准三类贡献模板

**类型1：文献贡献（最常见）**
> "本文首先贡献于[主题]的文献（[Author A, Year]; [Author B, Year]; [Author C, Year]）。与已有研究不同，本文利用[数据/准自然实验]，首次在[情境]中提供了[X]对[Y]影响的因果证据。"

**类型2：方法贡献**
> "本文在方法上借鉴[DML/Callaway-Sant'Anna/...]框架（[Author, Year]），并将其应用于[领域]，有效处理了[问题，如处理效应异质性]。"

**类型3：政策贡献**
> "本文的发现对[政策领域]具有直接含义。结果表明，[具体政策建议依据]，这为[政府/监管机构]提供了……的实证基础。"

### 贡献写作原则

1. **每条贡献必须对应具体文献**（用括号内引用明确与谁对话）
2. **用"首次"时要有把握**（确认该角度确实无人研究过）
3. **量化贡献**（如"扩充了N个国家的样本"）
4. **避免过度宣称**（"本文彻底解决了…"→改为"有助于厘清…"）
5. **贡献不超过3点**，宁少求精

---

## 7. 文献管理工具推荐

| 工具 | 特点 | 适用场景 |
|------|------|---------|
| Zotero（免费） | 浏览器插件一键抓取，BibTeX导出 | 日常文献管理 |
| Mendeley（免费） | PDF全文搜索，团队共享 | 团队合作 |
| EndNote | 功能全面，期刊模板丰富 | 需机构授权 |
| Obsidian + Zotero | 双链笔记 + 文献管理 | 深度知识管理 |

**Zotero → BibTeX 导出步骤：**
1. 选中文献（Ctrl+A 全选）
2. 右键 → Export Items → BibTeX → 保存为 `references.bib`
3. 置于 LaTeX 项目根目录
