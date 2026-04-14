---
name: paper-writing
description: Draft academic paper sections with pipeline-aware context, estimand protocol, causal language calibration, and automatic humanizer pass.
argument-hint: "[section: intro | background | strategy | data | results | conclusion | abstract | full | humanize] [--journal NAME] [--lang cn|en]"
workflow_stage: 5
allowed-tools: Read, Write, Edit, Grep, Glob, Task
tags:
  - LaTeX
  - academic-writing
  - empirical-paper
trigger: Manual — invoke after regression outputs exist in quality_reports/
---

# Paper Writing

Draft paper sections or apply humanizer pass. Each invocation targets **one section** (or `full` for sequential drafting with pauses).

> **This is the user's paper, not Claude's.** Match their voice. Never fabricate results. Only cite confirmed papers.

---

## Step 0: Context Gathering

Before drafting ANY section, read all available context in order:

| # | Path | Purpose |
|---|------|---------|
| 1 | `quality_reports/research_spec_*.md` | Research question, hypotheses, strategy |
| 2 | `.claude/references/domain-profile.md` | Field conventions, terminology |
| 3 | `.claude/references/journal-profiles.md` | Target journal style calibration |
| 4 | `.claude/references/endogeneity-routing.md` | Identification strategy rationale |
| 5 | `paper/sections/` | Existing draft sections (if any) |
| 6 | `paper/tables/` and `paper/figures/` | Generated regression output |
| 7 | `quality_reports/results_summary.md` | Coder's results summary (if exists) |
| 8 | `Bibliography_base.bib` | Available citation keys |
| 9 | `data/raw/data_provenance.md` | Data sources for Data section |

**Journal calibration**: If `--journal` is specified or journal-profiles.md exists, adjust:
- Word count range per section
- Introduction style (tight vs. extended institutional background)
- Citation density and format (natbib vs biblatex)
- 制度背景长度（中文期刊通常更长）

---

## Section Routing

Based on `$ARGUMENTS`:

| Argument | Output file | Dependencies |
|----------|------------|--------------|
| `intro` | `paper/sections/introduction.tex` | Write LAST — needs all results |
| `background` | `paper/sections/background.tex` | Can write early (needs domain-profile) |
| `strategy` | `paper/sections/strategy.tex` | Needs research_spec + endogeneity-routing |
| `data` | `paper/sections/data.tex` | Needs data_provenance + descriptive stats tables |
| `results` | `paper/sections/results.tex` | Needs regression tables in paper/tables/ |
| `conclusion` | `paper/sections/conclusion.tex` | Needs results section complete |
| `abstract` | `paper/sections/abstract.tex` | Write LAST — needs all sections |
| `full` | All sections in sequence | Pause between sections for user feedback |
| `humanize [file]` | Edited file in place | Existing draft |

### Recommended Writing Order

```
background → strategy → data → results → conclusion → intro → abstract
```

Introduction and abstract are written LAST because they must reference final results.

---

## Section Standards

| Section | EN words | CN chars | Key Requirements |
|---------|----------|----------|-----------------|
| Abstract | 100–150 | 300–500 | Question → method → finding with magnitude → implication |
| Introduction | 1000–1500 | 3000–5000 | Hook → gap → question → method → finding → contribution → roadmap |
| Background | 800–1200 | 2500–4000 | Institutional context, policy timeline, literature positioning |
| Strategy | 800–1200 | 2500–4000 | Estimand → formal model → assumptions → threats |
| Data & Variables | 600–1000 | 2000–3000 | Sources, sample construction, variable definitions, descriptive stats |
| Results | 800–1500 | 2500–5000 | Main spec → robustness → heterogeneity → mechanisms |
| Conclusion | 500–700 | 1500–2200 | Restate finding + magnitude → policy → limitations → future |

For detailed templates of each section, read `skills/paper-writing/ADVANCED.md`.

---

## Estimand Declaration Protocol

Before writing the Strategy section, produce this table:

```latex
% === ESTIMAND DECLARATION ===
% Target parameter : ATT / ATE / LATE / ITT / CATE
% Identification   : DID / IV / RDD / FE / Matching / RCT
% Key assumption   : Parallel trends / Exclusion / Continuity
% Estimator        : TWFE / CS-DID / 2SLS / Local poly / IPW-DR
% Std. errors      : Cluster(firm) / HC2 / Wild bootstrap
% Primary equation : Eq. (\ref{eq:main})
```

This table is referenced by the Strategy section AND the methods-referee agent.

---

## Causal Language Calibration

| Design | Allowed verbs | Forbidden |
|--------|--------------|-----------|
| RCT | causes, leads to, increases | — |
| Quasi-experimental (DID/IV/RDD) | affects, leads to, results in | proves, causes (unless very careful) |
| Selection-on-observables | is associated with, predicts, correlates with | causes, leads to, affects |
| Descriptive | co-occurs with, is correlated with | any causal verb |

**Hedging gradient**: Use "suggests" for single-study findings, "indicates" for findings consistent with prior work, "demonstrates" only for RCT/very strong designs.

---

## Incomplete Data Protocol

When data is unavailable, use markers (NEVER fabricate):

| Marker | Meaning | Render as |
|--------|---------|-----------|
| `\textbf{[TBD]}` | Regression result not yet available | Bold placeholder |
| `\textbf{[VERIFY]}` | Citation exists but needs user confirmation | Bold flag |
| `\textbf{[PLACEHOLDER: description]}` | Effect size awaiting final estimate | Bold with description |

---

## Quality Red Lines

### Banned Phrases (auto-check before output)

Grep the draft for these — if found, rewrite:

```
delve, utilize, leverage, nuanced, robust (as adjective for findings),
noteworthy, it is important to note, notably, interestingly,
in conclusion (mid-paper), as shown above, the above analysis,
令人瞩目, 毋庸置疑, 众所周知, 不言而喻
```

### Humanizer Pass (automatic on every draft)

Strip AI patterns across 4 categories before presenting:

| Category | Patterns to remove |
|----------|--------------------|
| **Structural** | Forced narrative arc, artificial "First...Second...Third" progression |
| **Lexical** | See banned phrases above + "furthermore", "moreover" overuse |
| **Rhetorical** | Rule-of-three, negative parallelisms, em-dash overuse (max 2 per page) |
| **Formatting** | Excessive bullet points in prose sections, promotional adjectives |

---

## Quality Self-Check

Before presenting ANY draft section to user:

- [ ] Every displayed equation is numbered (`\label{eq:...}`)
- [ ] All `\cite{}` keys exist in `Bibliography_base.bib`
- [ ] Introduction contribution paragraph names specific papers (not "prior literature")
- [ ] Effect sizes stated with units and economic interpretation
- [ ] No banned phrases (run grep)
- [ ] Notation consistent with Estimand Declaration
- [ ] All referenced tables/figures actually exist in `paper/tables/` or `paper/figures/`
- [ ] Causal language matches identification design
- [ ] TBD/VERIFY/PLACEHOLDER items flagged clearly to user

### Present to User

Flag items needing attention:
- **TBD items**: Where empirical results are needed but not yet available
- **VERIFY items**: Citations that need user confirmation
- **PLACEHOLDER items**: Effect sizes awaiting final estimates