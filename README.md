# Empirical Paper Pipeline (实证论文生成流水线)

一站式实证论文生成 Claude Code Skill，从数据审计到终稿输出的完整 7 阶段研究流水线。

灵感来源：[APE](https://ape.socialcatalystlab.org/) (Social Catalyst Lab)、[HLER](https://arxiv.org/abs/2603.07444) (Chen Zhu)、[academic-research-skills](https://github.com/Imbad0202/academic-research-skills) (Imbad0202)

## 功能

```
Stage 1: 数据审计 → Stage 2: 研究问题生成 → ◆ 用户选择
Stage 3: 识别策略设计 → ◆ 用户确认 → Stage 4: 计量分析
Stage 5: 论文撰写 → Stage 6: 审稿修订 → ◆ 用户批准
Stage 7: 终稿输出（含复现包）
```

- 支持 DID（经典/交错/三重差分）、面板固定效应、事件研究、RDD、IV
- 内置 Python 和 R 双语言代码模板
- Dataset-aware 假设生成（先审计数据再生成假设）
- 3 个人机协作决策关卡
- 自动审稿与修订（最多 2 轮）

## 安装

### 方法 1：作为项目 Skill（推荐）

```bash
cd /your/research/project
mkdir -p .claude/skills
git clone https://github.com/YOUR_USERNAME/empirical-paper-cc.git .claude/skills/empirical-paper-cc
```

然后将 `.claude/CLAUDE.md` 的内容复制到你项目的 `.claude/CLAUDE.md` 中。

### 方法 2：作为全局 Skill

```bash
git clone https://github.com/YOUR_USERNAME/empirical-paper-cc.git ~/.claude/skills/empirical-paper-cc
```

### 方法 3：独立项目

```bash
git clone https://github.com/YOUR_USERNAME/empirical-paper-cc.git
cd empirical-paper-cc
claude
```

## 使用

在 Claude Code 中直接对话即可：

```
# 有数据
"我上传了一份面板数据，帮我做实证分析"

# 有主题
"我想做一篇关于数字经济对企业创新影响的实证论文"

# 中途加入
"审稿人要求加平行趋势检验和安慰剂检验"

# 用 slash command
/empirical-pipeline
```

## 目录结构

```
.claude/
  CLAUDE.md                          # 项目入口配置
empirical-pipeline/
  SKILL.md                           # 核心 Skill 定义
  references/
    data-audit-checklist.md           # 数据审计诊断代码
    data-sources.md                   # 公开数据源索引
    did-methodology.md                # DID 方法论详细指南（Python + R）
    panel-fe-methods.md               # 面板固定效应指南
    other-methods.md                  # RDD, IV, PSM, 合成控制
  templates/
    question-template.md              # 研究问题评估模板
    pre-analysis-template.md          # 预分析计划模板
    paper-structure.md                # 论文结构与写作指南
  scripts/
    did_analysis.py                   # DID 分析模板脚本
output/                               # 输出目录（自动创建）
```

## 适用领域

- 数据治理与数据要素市场
- AI 政策评估
- 数字经济与平台经济
- 公共管理与组织行为
- 劳动经济学
- 其他需要面板数据计量分析的社科领域

## 推荐环境

- Claude Code（Claude Opus 4.6 或 Sonnet 4.6）
- Python 3.9+（pandas, numpy, statsmodels, linearmodels, matplotlib, scipy）
- 可选：R 4.0+（fixest, did, rdrobust, ggplot2）

## License

MIT
