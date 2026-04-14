# Paper Writing — Advanced Templates & Detailed Guidance

> **Load on demand.** The main SKILL.md references this file. Read only the section you need.

---

## Table of Contents (anchor reference for SKILL.md)

- [Introduction Template](#introduction-template)
- [Background / Literature Section](#background-section)
- [Identification Strategy Template](#identification-strategy-template)
- [Data & Variables Template](#data-and-variables-template)
- [Results Template](#results-template)
- [Conclusion Template](#conclusion-template)
- [Abstract Template](#abstract-template)
- [Two-Stage Writing Process](#two-stage-writing-process)
- [Chinese vs English Journal Differences](#chinese-vs-english-journal-differences)
- [Writing Process Checklist](#writing-process-checklist)
- [References & Style Guides](#references-and-style-guides)

---

## Introduction Template

The introduction follows a **6-paragraph funnel** (adapt to 5 or 7 as needed):

### Paragraph 1: Hook — Why This Matters

Open with a concrete fact, policy event, or economic puzzle — NOT "This paper examines."

```latex
\section{Introduction}

% Hook: concrete fact or policy event
[CONCRETE FACT / POLICY EVENT / ECONOMIC PUZZLE that establishes real-world importance].
[ONE SENTENCE expanding scope or magnitude].
Despite [BROAD RELEVANCE], the causal effect of [X] on [Y] remains unclear.
```

### Paragraph 2: Institutional Context / Policy Event

For papers exploiting policy variation, describe the event here. Chinese journals expect more detail (3–5 sentences); English Top-5 expect 1–2 sentences with a reference to a later Background section.

```latex
% Institutional context (especially for DID/RDD exploiting policy shocks)
In [YEAR], [COUNTRY/REGION] implemented [POLICY NAME], which [DESCRIPTION OF CHANGE].
This reform provides a [natural experiment / quasi-experimental setting] because
[WHY IT GENERATES EXOGENOUS VARIATION].
```

### Paragraph 3: Research Question + Method Preview

```latex
% Research question and identification
This paper asks: [RESEARCH QUESTION IN PLAIN LANGUAGE].
To identify the causal effect, we exploit [IDENTIFICATION STRATEGY]
using [DATA SOURCE] covering [TIME PERIOD] and [N] observations.
```

### Paragraph 4: Main Finding + Magnitude

State the headline result with a number. Reviewers look for this.

```latex
% Main finding with magnitude
We find that [X] [CAUSAL VERB per design] [Y] by [POINT ESTIMATE]
([UNIT / PERCENT / SD]), equivalent to [ECONOMIC INTERPRETATION].
This effect is [concentrated among / driven by] [SUBGROUP],
suggesting [MECHANISM].
```

### Paragraph 5: Contribution — Name Specific Papers

Never write "we contribute to the literature." Instead:

```latex
% Contribution — name papers
Our paper relates to three strands of literature.
First, we extend \citet{Author2020} by [SPECIFIC EXTENSION].
Unlike \citet{Other2019}, who [THEIR APPROACH], we [YOUR DIFFERENCE].
Second, we provide evidence on [MECHANISM] that complements
\citet{Third2021}. Finally, our findings inform [POLICY DEBATE]
by showing [SPECIFIC POLICY-RELEVANT RESULT].
```

### Paragraph 6: Roadmap

```latex
% Roadmap
The remainder of the paper is organized as follows.
Section~\ref{sec:background} provides institutional background.
Section~\ref{sec:strategy} describes our empirical strategy.
Section~\ref{sec:data} introduces the data.
Section~\ref{sec:results} presents results.
Section~\ref{sec:conclusion} concludes.
```

---

## Background Section

Two sub-components, order depends on journal convention:

**Institutional Background** (required for policy-shock papers):
- Policy timeline with exact dates
- Mechanism through which policy affects outcome
- Why this policy provides exogenous variation
- Prior studies using the same policy (differentiate your contribution)

**Literature Review** positioning strategy:
- Identify 2–3 literature strands
- For each: cite 3–5 key papers, state what they found, state YOUR difference
- Use \citet{} for textual citations, \citep{} for parenthetical

```latex
\subsection{Related Literature}

Our paper relates to the literature on [STRAND 1]. \citet{A2020} find
[THEIR RESULT] using [THEIR METHOD]. We differ by [YOUR EXTENSION].
More recently, \citet{B2022} show [RESULT], but their analysis
[LIMITATION YOUR PAPER ADDRESSES].
```

---

## Identification Strategy Template

This section must contain: formal model, key assumption, and threats discussion.

```latex
\section{Empirical Strategy}
\label{sec:strategy}

% === ESTIMAND DECLARATION (from SKILL.md protocol) ===

\subsection{Econometric Model}

Our baseline specification is:
\begin{equation}
Y_{it} = \alpha + \beta D_{it} + \mathbf{X}_{it}'\gamma + \mu_i + \delta_t + \varepsilon_{it}
\label{eq:main}
\end{equation}

where $Y_{it}$ is [OUTCOME] for [UNIT] $i$ in [TIME UNIT] $t$,
$D_{it}$ is [TREATMENT INDICATOR], $\mathbf{X}_{it}$ is a vector of
[CONTROLS], $\mu_i$ and $\delta_t$ are [UNIT] and [TIME] fixed effects.
The coefficient of interest is $\beta$, which identifies [ESTIMAND]
under [KEY ASSUMPTION].

\subsection{Identification Assumption}

[STATE ASSUMPTION FORMALLY]. This assumption requires that [PLAIN LANGUAGE].
We provide supporting evidence in [Section X / Appendix Y].

\subsection{Threats to Identification}

[THREAT 1]: [DESCRIPTION]. We address this by [SOLUTION].
[THREAT 2]: [DESCRIPTION]. We show in Table~\ref{tab:robust} that [EVIDENCE].
```

---

## Data and Variables Template

```latex
\section{Data and Variables}
\label{sec:data}

\subsection{Data Sources}

Our primary data come from [SOURCE] (\citealp{DataCitation}).
We supplement with [SECONDARY SOURCE] for [PURPOSE].
The sample covers [TIME PERIOD] at [FREQUENCY] frequency,
yielding [N] unit-time observations across [M] units.

\subsection{Variable Construction}

% Use a table for variable definitions
\begin{table}[htbp]
\centering
\caption{Variable Definitions}
\label{tab:variables}
\begin{tabular}{lll}
\toprule
Variable & Definition & Source \\
\midrule
$Y_{it}$ & [OUTCOME: definition, unit] & [SOURCE] \\
$D_{it}$ & [TREATMENT: definition, coding] & [SOURCE] \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Descriptive Statistics}

Table~\ref{tab:sumstats} reports summary statistics.
[HIGHLIGHT 1-2 key patterns relevant to identification].
```

---

## Results Template

Results sections walk readers through tables **column by column**.

```latex
\section{Results}
\label{sec:results}

\subsection{Main Results}

Table~\ref{tab:main} presents our main results.
Column~(1) shows the baseline specification without controls.
The coefficient on [TREATMENT] is [ESTIMATE] (s.e.~= [SE]),
[statistically significant at the Z\% level / not statistically
significant at conventional levels].

In column~(2), we add [CONTROLS]. The point estimate
[increases/decreases/remains stable] to [ESTIMATE].
Our preferred specification in column~(4) includes [FULL SET]
and yields an estimate of [ESTIMATE] (s.e.~= [SE]).

% Economic magnitude interpretation
To gauge economic significance, [INTERPRETATION IN REAL UNITS].
A one standard deviation increase in [X] is associated with
a [Y]-[unit] [increase/decrease] in [OUTCOME],
or approximately [PERCENT]\% of the sample mean.

\subsection{Robustness Checks}

% Organize by threat addressed
[Table~\ref{tab:robust} / Figure~\ref{fig:robust}] presents
robustness checks. We address [THREAT] by [TEST].
The results are [qualitatively similar / strengthen our baseline].

\subsection{Heterogeneity Analysis}

Table~\ref{tab:hetero} explores heterogeneity by [DIMENSION].
The effect is [concentrated among / larger for] [SUBGROUP],
consistent with [MECHANISM INTERPRETATION].

\subsection{Mechanism Analysis}
% Optional — only if you have credible mechanism tests
To shed light on mechanisms, we examine [CHANNEL VARIABLE].
Table~\ref{tab:mechanism} shows that [TREATMENT] [VERB]
[CHANNEL] by [ESTIMATE], suggesting that [INTERPRETATION].
```

---

## Conclusion Template

```latex
\section{Conclusion}
\label{sec:conclusion}

% Restate question and finding with magnitude
This paper examines [RESEARCH QUESTION]. Using [METHOD] applied to
[DATA], we find that [MAIN FINDING WITH SPECIFIC MAGNITUDE].

% Policy implications
Our findings have implications for [POLICY AREA].
Specifically, [ACTIONABLE IMPLICATION].

% Limitations (brief, honest — 2-3 sentences max)
Several caveats deserve mention. First, [EXTERNAL VALIDITY LIMITATION].
Second, [DATA/METHOD LIMITATION]. Future research could address
[SPECIFIC SUGGESTION].
```

---

## Abstract Template

Write as ONE flowing paragraph (no labeled sub-sections unless journal requires it):

```latex
\begin{abstract}
[ONE SENTENCE: research question].
[ONE SENTENCE: method and data].
[ONE SENTENCE: main finding with specific magnitude].
[ONE SENTENCE: key robustness or mechanism result].
[ONE SENTENCE: implication or contribution].
\end{abstract}

\medskip
\noindent\textbf{Keywords:} [keyword 1], [keyword 2], [keyword 3]

\noindent\textbf{JEL Classification:} [C00], [D00], [L00]
```

---

## Two-Stage Writing Process

For each section, follow this process:

**Stage 1: Outline with key points (internal only)**

```
- Main argument: X causes Y through mechanism Z
  * Key evidence: Table 3, column 4, β = 0.05
  * Cite: Author2020 for comparison, Author2021 for mechanism
  * Threat: selection → addressed in Table 5
- Sub-finding: effect concentrated in subgroup A
  * Evidence: Table 4, interaction term
```

**Stage 2: Convert to flowing prose**

1. Transform bullets into complete sentences with subjects and verbs
2. Add transitions (however, moreover, in contrast, building on this)
3. Integrate citations naturally within sentences
4. Ensure logical flow from one sentence to the next
5. Vary sentence structure — mix short declarative with longer compound
6. Read aloud mentally — if it sounds robotic, rewrite

**Critical**: Stage 1 is scaffolding — NEVER include bullet points in the final .tex output.

---

## Chinese vs English Journal Differences

| Dimension | English (Top 5 / field) | Chinese (经济研究/管理世界/中国工业经济) |
|-----------|------------------------|----------------------------------------|
| **Introduction style** | Tight, 1000–1500 words | Extended, 3000–5000 chars, more lit review inline |
| **Institutional background** | Brief (1–2 paragraphs) or separate section | Extensive (often 1500+ chars with policy quotes) |
| **Literature review** | Woven into introduction | Often separate section OR long intro sub-section |
| **摘要后附加** | Keywords + JEL codes | 关键词 + JEL + 中图分类号 (some journals) |
| **Footnotes** | Sparingly | Frequently (制度细节、数据说明、robustness notes) |
| **结论** | Forward-looking | 政策建议更具体，常需要"对策建议"段 |
| **引用格式** | natbib (author-year) | GB/T 7714 or journal-specific numbered format |
| **文内引用** | \citet{} / \citep{} | 部分期刊用脚注引用 |

### Language-Specific Rules

**English**:
- Never start a sentence with a numeral (write "Fifteen percent..." or restructure)
- Use Oxford comma in lists
- "data" is plural ("the data show...")
- Percent: "5 percent" or "5%" — be consistent, follow journal style

**Chinese (中文)**:
- 避免欧化长句——中文句子宜短（15–25字为佳）
- 引用格式：（作者，年份）或 脚注，遵循目标期刊要求
- 数字规范：万/亿 为单位（不写 10,000），百分比用"%"
- 正文中变量首次出现需加中文解释

---

## Writing Process Checklist

### Before writing
- [ ] Read all context files (Step 0 in SKILL.md)
- [ ] Confirm target journal style from journal-profiles.md
- [ ] Verify all cited tables/figures exist
- [ ] Review Estimand Declaration

### During writing (per section)
- [ ] Stage 1: outline → Stage 2: prose
- [ ] Causal language matches design (see SKILL.md table)
- [ ] Effect sizes include units AND economic interpretation
- [ ] Each paragraph has one main idea
- [ ] Transitions connect paragraphs logically

### After writing (per section)
- [ ] Run humanizer pass (SKILL.md protocol)
- [ ] Grep for banned phrases
- [ ] Verify all \cite{} keys exist in .bib
- [ ] Check TBD/VERIFY/PLACEHOLDER items flagged
- [ ] Section word count within standards

---

## References & Style Guides

For detailed writing principles (clarity, conciseness, accuracy, verb tense, word choice), read:
- `skills/paper-writing/references/writing_principles.md`

For LaTeX document skeleton and package guidance, read:
- `skills/paper-writing/references/latex_templates.md`

External references:
- Cochrane (2005) "Writing Tips for PhD Students"
- Shapiro (2019) "How to Give an Applied Micro Talk"
- Angrist & Pischke (2017) writing style in *Mostly Harmless Econometrics*
- McCloskey (2000) *Economical Writing*
- Thomson (2011) *A Guide for the Young Economist*