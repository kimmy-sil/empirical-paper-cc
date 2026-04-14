# LaTeX Templates for Empirical Papers

> **Reference file** — read when setting up a new paper or needing LaTeX syntax guidance.

---

## Minimal Compilable Skeleton

```latex
\documentclass[12pt,a4paper]{article}

% === Encoding & Language ===
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
% For Chinese: uncomment below and comment out inputenc/fontenc
% \usepackage[UTF8]{ctex}

% === Math ===
\usepackage{amsmath,amssymb,amsthm}

% === Tables ===
\usepackage{booktabs}       % \toprule, \midrule, \bottomrule (NEVER \hline)
\usepackage{tabularx}       % flexible column widths
\usepackage{threeparttable} % table notes below

% === Figures ===
\usepackage{graphicx}
\usepackage{float}          % [H] placement

% === Citations ===
\usepackage[authoryear,comma]{natbib}
\bibliographystyle{aer}     % AER style; alternatives: chicago, plainnat
% For Chinese journals using numbered citations:
% \usepackage[numbers,sort&compress]{natbib}
% \bibliographystyle{gbt7714-numerical}

% === Layout ===
\usepackage[margin=1in]{geometry}
\usepackage{setspace}
\doublespacing              % most journals require double spacing

% === Cross-references ===
\usepackage[hidelinks]{hyperref}
\usepackage{cleveref}       % \cref{tab:main} → "Table 1"

% === Appendix ===
\usepackage[title]{appendix}

% ============================================================
\title{[PAPER TITLE]}
\author{
  [Author 1]\thanks{Affiliation. Email: author1@university.edu} \and
  [Author 2]\thanks{Affiliation. Email: author2@university.edu}
}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
  \textbf{[TBD]} % Draft with paper-writing skill
\end{abstract}

\noindent\textbf{Keywords:} [keyword 1], [keyword 2], [keyword 3] \\
\textbf{JEL Classification:} [C00], [D00], [L00]

\newpage
\input{sections/introduction}
\input{sections/background}
\input{sections/strategy}
\input{sections/data}
\input{sections/results}
\input{sections/conclusion}

\newpage
\singlespacing
\bibliography{Bibliography_base}

\newpage
\begin{appendices}
\input{sections/appendix}
\end{appendices}

\end{document}
```

---

## Citation Commands (natbib)

| Command | Output | Use when |
|---------|--------|----------|
| `\citet{Smith2020}` | Smith (2020) | Subject of sentence |
| `\citep{Smith2020}` | (Smith, 2020) | Parenthetical |
| `\citet[p.~15]{Smith2020}` | Smith (2020, p. 15) | Specific page |
| `\citep[see][]{Smith2020}` | (see Smith, 2020) | With prefix |
| `\citep{Smith2020,Jones2021}` | (Smith, 2020; Jones, 2021) | Multiple |
| `\citeauthor{Smith2020}` | Smith | Author only (already cited nearby) |
| `\citeyear{Smith2020}` | 2020 | Year only |
| `\citealp{Smith2020}` | Smith, 2020 | No parentheses (for inside parens) |

**Rule**: Every `\cite{}` key must exist in `Bibliography_base.bib`. If uncertain, mark `\textbf{[VERIFY]} \cite{MaybeKey2020}`.

---

## Table Template (booktabs + threeparttable)

```latex
\begin{table}[htbp]
\centering
\begin{threeparttable}
\caption{Effect of X on Y}
\label{tab:main}
\begin{tabular}{lccccc}
\toprule
                & (1)        & (2)        & (3)        & (4)        \\
                & OLS        & + Controls & + FE       & Preferred  \\
\midrule
Treatment       & 0.052***   & 0.048***   & 0.041**    & 0.039**    \\
                & (0.012)    & (0.011)    & (0.015)    & (0.016)    \\
                &            &            &            &            \\
Controls        & No         & Yes        & Yes        & Yes        \\
Fixed Effects   & No         & No         & Yes        & Yes        \\
Clustered SE    & No         & No         & No         & Yes        \\
                &            &            &            &            \\
Observations    & 10{,}000   & 9{,}850    & 9{,}850    & 9{,}850    \\
R-squared       & 0.05       & 0.12       & 0.35       & 0.35       \\
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]
\small
\item \textit{Notes:} Standard errors in parentheses, clustered at [LEVEL]
in column (4). * p<0.10, ** p<0.05, *** p<0.01.
\end{tablenotes}
\end{threeparttable}
\end{table}
```

**Table conventions**:
- Never use `\hline` — always `\toprule`, `\midrule`, `\bottomrule`
- Never use vertical lines (`|` in column spec)
- Use `{,}` for thousands separator in numbers: `10{,}000`
- Align decimal points: consider `siunitx` package with `S` column type
- Notes go inside `threeparttable` environment, use `\begin{tablenotes}`

---

## Equation Conventions

```latex
% Numbered equation (most equations in empirical papers)
\begin{equation}
Y_{it} = \alpha + \beta D_{it} + \mathbf{X}_{it}'\gamma
  + \mu_i + \delta_t + \varepsilon_{it}
\label{eq:main}
\end{equation}

% Reference: Equation~\eqref{eq:main} or \cref{eq:main}
```

### Standard Notation (keep consistent across paper)

| Symbol | Meaning | LaTeX |
|--------|---------|-------|
| $Y_{it}$ | Outcome | `Y_{it}` |
| $D_{it}$ | Treatment indicator | `D_{it}` |
| $\mathbf{X}_{it}$ | Control vector | `\mathbf{X}_{it}` |
| $\mu_i$ | Unit fixed effect | `\mu_i` |
| $\delta_t$ | Time fixed effect | `\delta_t` |
| $\varepsilon_{it}$ | Error term | `\varepsilon_{it}` |
| $\beta$ | Coefficient of interest | `\beta` |
| $\hat{\beta}$ | Estimated coefficient | `\hat{\beta}` |
| $\mathbb{E}[\cdot]$ | Expectation | `\mathbb{E}[\cdot]` |

---

## Figure Inclusion

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{figures/event_study.pdf}
\caption{Event Study: Dynamic Treatment Effects}
\label{fig:event_study}
\floatfoot{\textit{Notes:} Point estimates and 95\% confidence intervals
from Equation~\eqref{eq:event}. The omitted period is $t=-1$.
Vertical dashed line indicates treatment onset.}
\end{figure}
```

**Prefer PDF** for figures (vector graphics, scales perfectly). Use PNG only for raster images (photos, screenshots). Resolution: 300+ DPI for PNG.

---

## Project Directory Structure

```
paper/
├── main.tex                    ← Skeleton above
├── Bibliography_base.bib       ← All citations
├── sections/
│   ├── introduction.tex
│   ├── background.tex
│   ├── strategy.tex
│   ├── data.tex
│   ├── results.tex
│   ├── conclusion.tex
│   └── appendix.tex
├── tables/                     ← Generated by estimation code
│   ├── tab_main.tex
│   ├── tab_robust.tex
│   └── tab_hetero.tex
└── figures/                    ← Generated by estimation code
    ├── event_study.pdf
    └── coef_plot.pdf
```

---

## Chinese-Specific LaTeX Notes

```latex
% Use ctex for Chinese documents
\documentclass[12pt,a4paper]{ctexart}

% Chinese abstract
\renewcommand{\abstractname}{摘\quad 要}

% Chinese keywords
\noindent\textbf{关键词：}数字经济；全要素生产率；双重差分

% Chinese section names are automatic with ctex
% \section{引言} \section{制度背景} etc.

% Citation style for Chinese journals
\bibliographystyle{gbt7714-numerical}  % GB/T 7714 numbered
% or
\bibliographystyle{gbt7714-author-year}  % GB/T 7714 author-year
```