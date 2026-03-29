---
name: empirical-pipeline
description: >
  一站式实证论文生成流水线。上传数据集或指定研究主题后，自动完成7阶段流程：
  数据审计→研究问题生成→识别策略设计→计量分析→论文撰写→审稿修订→终稿输出。
  支持DID/面板FE/事件研究/RDD/IV。适用于管理学、公共管理、数字经济、数据治理、AI政策评估。
  触发：实证论文、DID分析、面板回归、政策评估、一站式论文、研究流水线、empirical paper。
metadata:
  author: kim
  version: '1.0'
  inspired-by: APE, HLER, academic-research-skills
---

# 一站式实证论文生成流水线

## Architecture

7-stage pipeline with 3 human decision gates:

```
S1:DataAudit → S2:QuestionGen → ◆SELECT → S3:PreAnalysis → ◆CONFIRM
→ S4:Analysis → S5:Drafting → S6:Review → ◆APPROVE → S7:Finalize
```

## Entry Points

**A) Has data file** → Start Stage 1
**B) Has topic, no data** → Start Stage 2 (search public data sources first)
**C) Mid-stream** → Ask user's current progress, jump to matching stage

---

## Stage 1: Data Audit

Goal: Understand data structure, quality, and research potential to constrain hypothesis generation.

### Steps

1. Load data → print shape, dtypes, time range, entity count
2. Classify variables: panel IDs, outcome candidates, treatment candidates, controls, instruments
3. Quality check: missing rates, panel balance, outliers, within-variation
4. Generate `output/data_audit_report.md`

Read `references/data-audit-checklist.md` for detailed diagnostics.

### Output

`output/data_audit_report.md` containing:
- Data overview table
- Variable dictionary (name, type, description, missing%, distribution)
- Panel structure diagnosis
- Potential research directions
- Data limitations

---

## Stage 2: Research Question Generation

Goal: Generate 3-5 feasible, dataset-aware research questions with scoring.

### Steps

1. **Dataset-aware constraint**: Each question must use variables that exist with <30% missing, sufficient within-variation, adequate sample size
2. For each candidate, produce structured assessment: see `templates/question-template.md`
3. Quick literature scan (search for existing work, identify gaps)
4. Score each question on 4 dimensions (25pts each = 100 total):
   - Data adequacy
   - Identification credibility
   - Novelty
   - Policy relevance

### ◆ HUMAN GATE: Select Research Question

Present all candidates with scores. Ask user to:
1. Select one question, OR
2. Modify/combine candidates, OR
3. Request regeneration

---

## Stage 3: Pre-Analysis Plan

Goal: Design rigorous causal identification strategy and write complete pre-analysis plan.

### Method Selection

Based on data structure and research question, select from method toolbox.
Read `references/did-methodology.md` for DID details.
Read `references/panel-fe-methods.md` for panel FE details.
Read `references/other-methods.md` for RDD, IV, and other methods.

### DID Checklist (most common)

1. Define treatment & control groups clearly
2. Identify policy implementation date
3. Plan parallel trends test (event study plot)
4. Choose standard error clustering level (match treatment level)
5. Plan placebo tests (fake timing / fake treatment group)
6. If staggered: use CS(2021) or SA(2021), NOT naive TWFE
7. Plan heterogeneity analysis
8. Rule out confounding concurrent policies
9. Plan anticipation test (lead coefficients should be insignificant)
10. Plan sample composition test

### Output

Generate `output/pre_analysis_plan.md` following `templates/pre-analysis-template.md`.

### ◆ HUMAN GATE: Confirm Strategy

Present pre-analysis plan. Ask user to confirm:
1. Is identification strategy sound?
2. Any variable adjustments needed?
3. Additional robustness checks to add?

---

## Stage 4: Econometric Analysis

Goal: Execute all statistical analysis per pre-analysis plan.

### Steps

1. **Data prep**: Clean, construct variables, handle missing values
2. **Descriptive stats**: Table 1 (by treatment/control), mean difference tests
3. **Main regression**: Execute chosen identification strategy
4. **Parallel trends / identification tests**: Event study plot, pre-trend F-test
5. **Robustness**: Placebo, alternative controls, alternative DVs, different clustering, add/drop controls
6. **Heterogeneity**: Sub-sample regressions, interaction terms

Use scripts in `scripts/` as templates. All analysis code saved to `output/replication/`.

### Output

- `output/tables/` — all regression tables in Markdown
- `output/figures/` — event study plot, trend plots, coefficient plots
- `output/analysis_log.md` — complete analysis log

---

## Stage 5: Paper Drafting

Goal: Write complete academic paper based on analysis results.

### Structure

Follow `templates/paper-structure.md` for section-by-section guidance.

Standard sections:
1. Abstract (250-300 words)
2. Introduction (1500-2000 words)
3. Institutional Background (1000-1500 words)
4. Literature Review (1500-2000 words)
5. Data & Variables (1000-1500 words)
6. Empirical Strategy (1000-1500 words)
7. Results (2000-3000 words)
8. Conclusion (800-1000 words)
9. References (≥25)

### Writing Quality Rules

- Academic register: rigorous, objective, professional
- Full paragraph prose only (NO bullet lists in body text)
- Every claim backed by data or citations
- Careful causal language (only causal claims if identification is credible)
- Chinese style: 经管期刊风格，专业但不晦涩
- AVOID: "值得注意的是", "综上所述", "取决于", excessive dashes, exclamation marks

### Output

`output/paper_draft.md` + `output/paper_summary.md` (1-page summary)

---

## Stage 6: Review & Revision

Goal: Simulate peer review and iterate.

### Review Dimensions (0-20 each, 100 total)

| Dimension | What to evaluate |
|-----------|-----------------|
| Identification | Causal credibility, assumption validity, test sufficiency |
| Novelty | Research gap, contribution to literature |
| Policy Relevance | Practical implications for policy/management |
| Execution | Data handling, analysis rigor, result presentation |
| Writing | Logic, completeness, formatting |

### Review Format Checks

- [ ] Paper ≥ 25 pages (incl. tables/figures, excl. references)
- [ ] References ≥ 25
- [ ] Full paragraph prose (no bullet lists in body)
- [ ] Each major section has 3-4 substantive paragraphs
- [ ] All figures/tables numbered, titled, with notes
- [ ] No placeholder data ("XXX", "TBD", "Figure ??")

### Process

Round 1: Generate `output/review_round1.md` with verdict (Accept/Minor/Major/Reject) + detailed comments → Revise → Generate `output/revision_response.md`

Round 2 (if needed): Re-review revised draft → Final revise

Max 2 revision rounds.

### ◆ HUMAN GATE: Approve Final

Present review report + revision plan. Ask user:
1. Agree with revision plan?
2. Additional comments?
3. Satisfied with quality?

---

## Stage 7: Finalization

### Output Package

```
output/
├── paper_final.md
├── tables/
│   ├── table1_descriptive.md
│   ├── table2_main_results.md
│   ├── table3_robustness.md
│   └── table4_heterogeneity.md
├── figures/
│   ├── fig1_trends.png
│   ├── fig2_event_study.png
│   └── fig3_coefficients.png
├── pre_analysis_plan.md
├── data_audit_report.md
├── analysis_log.md
├── review_round1.md
├── revision_response.md
└── replication/
    ├── analysis.py
    └── README.md
```

### Replication Package Requirements

1. Complete analysis code (Python or R)
2. Data dictionary
3. Run instructions (dependencies, execution order)
4. Expected output description
