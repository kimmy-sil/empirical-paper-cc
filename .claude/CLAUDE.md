# Empirical Paper Pipeline (实证论文生成流水线)

一站式实证论文生成工具，覆盖从数据审计到终稿输出的完整研究流程。

## Skills

| Skill | Purpose | Invoke |
|-------|---------|--------|
| `empirical-pipeline` | 全流程编排器（7阶段流水线） | `/empirical-pipeline` 或自动触发 |

## Routing Rules

- 用户上传数据文件（CSV/Excel/Stata）→ 从 Stage 1 (Data Audit) 开始
- 用户给出研究主题 → 从 Stage 2 (Question Generation) 开始，先搜索数据源
- 用户已有回归结果/初稿 → 跳至对应阶段（Stage 5/6）
- 用户要求审稿修订 → 直接进入 Stage 6

## Pipeline Flow

```
Stage 1: Data Audit → Stage 2: Question Generation
  → ◆ Human Gate: Select Research Question
Stage 3: Pre-Analysis Plan
  → ◆ Human Gate: Confirm Identification Strategy
Stage 4: Econometric Analysis → Stage 5: Paper Drafting
Stage 6: Review & Revision (max 2 rounds)
  → ◆ Human Gate: Approve Final Draft
Stage 7: Finalization → Output Package
```

## Key Rules

- 所有假设必须 dataset-aware（基于数据可行性约束）
- DID 必做平行趋势检验 + 安慰剂检验
- 交错 DID 不得盲目使用 TWFE，必须使用 CS(2021)/SA(2021) 等新方法
- 标准误聚类层级必须匹配处理层级
- 正文使用完整段落散文，不使用项目符号列表
- 参考文献 ≥ 25 篇
- 中文写作避免 AI 腔（不用"值得注意的是"、"综上所述"、"取决于"）
- 不过度使用破折号

## Output Structure

所有输出保存在 `output/` 目录：

```
output/
├── data_audit_report.md
├── candidate_questions.md
├── pre_analysis_plan.md
├── analysis_log.md
├── tables/
├── figures/
├── paper_draft.md
├── review_round1.md
├── revision_response.md
├── paper_final.md
└── replication/
    ├── analysis.py (or analysis.R)
    └── README.md
```

## Default Preferences

- **Language**: 中文（除非用户指定英文）
- **Methods**: DID, Panel FE, Event Study, RDD, IV
- **Analysis Tool**: Python (pandas + statsmodels + linearmodels) — 如用户要求可切换到 R (fixest + did)
- **Citation Style**: APA 7 或 GB/T 7714
- **Domain Focus**: 数据治理、AI Agent、数字经济、公共管理、组织行为
