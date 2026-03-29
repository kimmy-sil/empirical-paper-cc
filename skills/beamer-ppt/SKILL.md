# beamer-ppt — Beamer 学术汇报幻灯片技能指南

适用场景：为实证论文制作学术汇报 Beamer 幻灯片。触发关键词：Beamer、幻灯片、学术报告、slides、presentation、seminar slides、PPT。

参考模板：`empirical-pipeline/templates/beamer-slides.tex`

---

## 1. 标准学术汇报结构（15–20 页）

以下为实证经济/管理学论文标准汇报结构，括号内为建议页数：

| # | 页面标题 | 核心内容 | 建议页数 |
|---|---------|---------|---------|
| 1 | Title | 论文标题、作者、单位、会议/日期 | 1 |
| 2 | Outline | 报告结构（4–5 个要点） | 1 |
| 3 | Motivation | 研究问题的重要性、经验现象或政策背景 | 1–2 |
| 4 | Research Question | 精确的研究问题 + 本文做法一句话 | 1 |
| 5 | Contribution | 文献定位，3条贡献 | 1 |
| 6 | Background | 制度背景（外生变异来源） | 1–2 |
| 7 | Data | 数据来源、样本描述、主要变量描述性统计 | 1–2 |
| 8 | Empirical Strategy | 主方程（公式） + 识别假设 | 1–2 |
| 9 | Main Results | 主回归表（精简版） | 1–2 |
| 10 | Event Study | 事件研究图（平行趋势可视化） | 1 |
| 11 | Robustness | 稳健性检验摘要（文字+关键数字） | 1 |
| 12 | Heterogeneity | 异质性分析图或表 | 1–2 |
| 13 | Conclusion | 核心发现（3条）+ 政策含义 | 1 |
| 备用 | Appendix | 供Q&A使用，详见第7节 | 不计入主体 |

**总计：15–20 页主体幻灯片 + N 页备用幻灯片**

---

## 2. Beamer 主题选择建议

### 推荐主题

**metropolis**（最推荐，现代简洁）
```latex
\usetheme{metropolis}
\usepackage{appendixnumberbeamer}
% 特点：无导航条，浅灰配色，标题大字，学术感强
% 安装：texlive-beamer-theme-metropolis 或手动下载
```

**CambridgeUS**（经典，适合正式场合）
```latex
\usetheme{CambridgeUS}
\usecolortheme{beaver}
% 特点：红色调，含导航条，适合技术报告
```

**Madrid**（标准，广泛兼容）
```latex
\usetheme{Madrid}
\usecolortheme{default}
% 特点：底部导航，蓝色调，最常见学术风格
```

### 颜色方案建议

```latex
% 自定义配色（推荐深蓝+白）
\definecolor{mainblue}{RGB}{0,62,116}
\definecolor{accentred}{RGB}{185,20,20}
\setbeamercolor{structure}{fg=mainblue}
\setbeamercolor{alerted text}{fg=accentred}
```

### 字体设置

```latex
% 无衬线字体（幻灯片推荐）
\usefonttheme{professionalfonts}
\usepackage{helvet}  % Helvetica

% 或使用 Fira Sans（与 metropolis 搭配）
\usepackage[sfdefault]{FiraSans}
\usepackage[T1]{fontenc}
```

---

## 3. 完整 Beamer 骨架模板

```latex
\documentclass[aspectratio=169, 12pt]{beamer}
% aspectratio=169 为 16:9 宽屏；43 为传统 4:3

% ============================================================
% 主题与颜色
% ============================================================
\usetheme{metropolis}
\usepackage{appendixnumberbeamer}

% ============================================================
% 中文支持
% ============================================================
\usepackage{xeCJK}
\setCJKmainfont{Noto Sans CJK SC}  % 或 Source Han Sans SC / 微软雅黑
\setCJKsansfont{Noto Sans CJK SC}
\setCJKmonofont{Noto Sans CJK SC}
% 编译命令：xelatex（不能用 pdflatex）

% ============================================================
% 数学与表格
% ============================================================
\usepackage{amsmath, amssymb, bm}
\usepackage{booktabs}       % 三线表
\usepackage{tabularx}       % 自适应列宽
\usepackage{graphicx}
\usepackage{pgfplots}       % 如需内嵌绘图
\pgfplotsset{compat=1.18}

% ============================================================
% 封面信息
% ============================================================
\title{论文标题（中文）\\
       \small Paper Title in English}
\subtitle{论文副标题（如有）}
\author{作者甲\inst{1} \and 作者乙\inst{2}}
\institute{
  \inst{1} 单位一，城市 \and
  \inst{2} 单位二，城市
}
\date{会议名称 \\ \small 2026年X月}

% ============================================================
% 正文
% ============================================================
\begin{document}

\maketitle

% --- 目录 ---
\begin{frame}{Outline}
  \setbeamertemplate{section in toc}[sections numbered]
  \tableofcontents[hideallsubsections]
\end{frame}

% --- 动机 ---
\section{Motivation}
\begin{frame}{研究背景与问题}
  % 内容见第4节
\end{frame}

% ... 各章节 ...

% ============================================================
% 备用幻灯片
% ============================================================
\appendix
\begin{frame}[plain]
  \centering
  {\Huge\bfseries Appendix}
\end{frame}

% 备用幻灯片内容见第7节

\end{document}
```

---

## 4. 公式排版

### 行内公式
```latex
处理效应 $\hat{\beta}$ 在1\%水平显著。
```

### 展示公式（带编号）
```latex
\begin{frame}{实证策略}
  基准回归方程为：
  \begin{equation}
    Y_{it} = \alpha + \beta \, (\text{Treat}_{i} \times \text{Post}_{t})
             + \gamma X_{it} + \mu_i + \lambda_t + \varepsilon_{it}
    \label{eq:main}
  \end{equation}
  其中：
  \begin{itemize}
    \item $Y_{it}$：个体 $i$ 在期 $t$ 的因变量
    \item $\text{Treat}_i \times \text{Post}_t$：核心交乘项
    \item $\mu_i, \lambda_t$：个体与时间固定效应
    \item $\varepsilon_{it}$：聚类在个体层面的标准误
  \end{itemize}
\end{frame}
```

### 对齐公式
```latex
\begin{align}
  \hat{\tau}^{\text{ATT}} &= \mathbb{E}[Y_{it}(1) - Y_{it}(0) \mid D_i = 1] \\
                          &\approx \bar{Y}^{\text{treat,post}} - \bar{Y}^{\text{treat,pre}}
                            - (\bar{Y}^{\text{ctrl,post}} - \bar{Y}^{\text{ctrl,pre}})
\end{align}
```

---

## 5. 三线表在 Beamer 中的排版

```latex
\begin{frame}{主回归结果}
  \begin{table}
    \centering
    \caption{DID 基准回归}
    \small   % 或 \footnotesize，防止表格过大
    \begin{tabular}{lcccc}
      \toprule
                     & \multicolumn{2}{c}{Dep. Var.: $\ln Y$}
                     & \multicolumn{2}{c}{Dep. Var.: $Y/L$} \\
      \cmidrule(lr){2-3} \cmidrule(lr){4-5}
                     & (1)      & (2)      & (3)      & (4) \\
      \midrule
      Treat $\times$ Post
                     & 0.082*** & 0.075*** & 0.031**  & 0.028* \\
                     & (0.021)  & (0.020)  & (0.015)  & (0.016) \\
      Controls       & No       & Yes      & No       & Yes \\
      Entity FE      & Yes      & Yes      & Yes      & Yes \\
      Year FE        & Yes      & Yes      & Yes      & Yes \\
      \midrule
      Observations   & 12,450   & 12,450   & 12,450   & 12,450 \\
      $R^2$          & 0.72     & 0.75     & 0.68     & 0.71 \\
      \bottomrule
    \end{tabular}

    \vspace{2pt}
    {\tiny \textit{注：}括号内为聚类至个体层面的稳健标准误。
    *$p<0.10$, **$p<0.05$, ***$p<0.01$。}
  \end{table}
\end{frame}
```

**Beamer 表格注意事项：**
- `\small` 或 `\footnotesize` 防止溢出
- 合并列用 `\multicolumn`，合并行用 `\multirow`（需 multirow 包）
- 表格注释放在 `\bottomrule` 之后，`{\tiny ...}` 控制字号
- 避免超过6列，否则字体需要进一步缩小

---

## 6. 图表引用与插入

### 插入外部图片（事件研究图等）
```latex
\begin{frame}{事件研究：平行趋势检验}
  \begin{figure}
    \centering
    \includegraphics[width=0.82\textwidth]{figures/event_study.pdf}
    % 推荐 PDF 矢量图；PNG 分辨率需 ≥ 300 dpi
    \caption{\small 事件研究系数图（基期 = $t-1$，95\%置信区间）}
  \end{figure}
  \vspace{-6pt}
  {\footnotesize \textit{注：}竖虚线为政策实施时点。预处理期系数均不显著（联合检验 $p=0.42$）。}
\end{frame}
```

### 并排两图
```latex
\begin{frame}{异质性分析}
  \begin{columns}[T]
    \column{0.48\textwidth}
      \centering
      \includegraphics[width=\textwidth]{figures/het_size.pdf}
      {\small 按企业规模分组}
    \column{0.48\textwidth}
      \centering
      \includegraphics[width=\textwidth]{figures/het_region.pdf}
      {\small 按地区分组}
  \end{columns}
  \vspace{4pt}
  {\footnotesize 竖线为95\%置信区间；实心点显著（$p<0.10$），空心点不显著。}
\end{frame}
```

---

## 7. 备用幻灯片（Appendix Slides for Q&A）

备用幻灯片不计入正式页数（需 `\usepackage{appendixnumberbeamer}` 并在附录前调用 `\appendix`）。

**标准备用幻灯片清单：**

| 页面 | 目的 | 应对场景 |
|------|------|---------|
| 数据详细说明 | 变量定义、数据来源细节 | "样本怎么构建的？" |
| 平行趋势替代检验 | 不同窗口期的事件研究图 | "平行趋势稳健吗？" |
| 安慰剂检验 | 虚假时点/虚假处理组 | "是否为随机结果？" |
| 稳健性表格（完整版） | 全部列的稳健性结果 | "稳健性到底怎样？" |
| 机制分析 | 中介变量回归 | "为什么会有这个效应？" |
| 交错DID细节 | CS/SA方法结果 | "处理时点交错怎么处理？" |
| 外部效度讨论 | 样本代表性分析 | "结论能推广吗？" |

**备用幻灯片代码结构：**
```latex
\appendix

\begin{frame}[plain, noframenumbering]
  \centering
  \vspace{2cm}
  {\Huge\color{mainblue} Appendix}
\end{frame}

\begin{frame}[noframenumbering]{A1. 变量定义}
  % 内容
\end{frame}

\begin{frame}[noframenumbering]{A2. 安慰剂检验}
  % 内容
\end{frame}
```

---

## 8. 中文支持详细说明（xeCJK）

```latex
% 必须使用 XeLaTeX 编译
\usepackage{xeCJK}

% 字体选项（按系统安装情况选一）
% macOS:
\setCJKmainfont{PingFang SC}
\setCJKsansfont{PingFang SC}

% Windows:
\setCJKmainfont{Microsoft YaHei}

% Linux / TeX Live（开源）:
\setCJKmainfont{Noto Sans CJK SC}

% 字号控制
\setCJKmainfont[BoldFont={Noto Sans CJK SC Bold}]{Noto Sans CJK SC}
```

**编译命令：**
```bash
xelatex slides.tex   # 第一次
xelatex slides.tex   # 第二次（确保引用和目录正确）
```

**如需 BibTeX：**
```bash
xelatex slides.tex
bibtex slides
xelatex slides.tex
xelatex slides.tex
```

---

## 9. 汇报技巧

### 时间控制
- 15分钟报告 → 15页主体（每页约1分钟）
- 30分钟报告 → 20页主体 + 充分的Q&A备用页
- 不要在幻灯片上写整段文字，观众不会阅读

### 每页内容原则
- 一页一主题（One page, one message）
- 要点用不超过3–4条简洁短句
- 关键数字高亮（`\alert{0.082***}` 或 `\textbf{}`）

### `\alert{}` 使用
```latex
% 高亮关键发现
处理效应为 \alert{8.2\%}（$p<0.01$），
且在大型企业中效应翻倍（\alert{16.5\%}）。
```

### 进度指示
```latex
% 在每节开始添加目录高亮当前节
\begin{frame}{Outline}
  \tableofcontents[currentsection, hideallsubsections]
\end{frame}
```

---

## 10. 快速检查清单

编译前确认：

- [ ] 编译命令为 `xelatex`（非 `pdflatex`）
- [ ] 中文字体已在系统安装
- [ ] 图片路径正确（相对路径或绝对路径）
- [ ] `\appendix` 在备用幻灯片前
- [ ] `appendixnumberbeamer` 已加载（备用页不计页数）
- [ ] 三线表使用 `booktabs`（`\toprule`, `\midrule`, `\bottomrule`）
- [ ] 表格注释字号为 `\tiny` 或 `\footnotesize`
- [ ] 核心结论用 `\alert{}` 高亮
- [ ] 主体幻灯片 ≤ 20 页
- [ ] 备用幻灯片覆盖常见Q&A问题
