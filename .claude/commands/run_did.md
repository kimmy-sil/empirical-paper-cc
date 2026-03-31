description: 执行DID全流程分析
argument-hint: <数据文件路径>
---
检查：是否存在已确认的Research Design Memo？
├── 存在 → 跳过三问，直接执行
└── 不存在 → 先回答三个问题：
    1. 处理变量为什么是外生的？
    2. 最大的内生性威胁是什么？
    3. 估计的是谁的效应？

对 $ARGUMENTS 执行完整DID分析：
1. 数据审计 + 描述性统计
2. 基准回归（TWFE → 自动Bacon分解 → 如需切换CS/SA）
3. 事件研究图（平行趋势）
4. 必做稳健性：HonestDiD + 安慰剂500次
5. 推荐稳健性：替换变量 + 缩短窗口
6. Estimand声明

所有结果保存到 output/。
