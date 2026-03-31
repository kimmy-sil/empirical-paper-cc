description: 执行RDD全流程分析
argument-hint: <数据文件路径>
---
检查Research Design Memo → 存在则跳过三问。

对 $ARGUMENTS 执行RDD分析：
1. RD Plot（IMSE + MV带宽）
2. 密度检验（rddensity）
3. 协变量平衡检验
4. 主估计（rdrobust，MSE带宽，报告Robust CI）
5. 带宽敏感性（0.5h~1.5h）
6. 多项式阶数敏感性（p=1,2）
7. 安慰剂断点
8. CER带宽置信区间
9. Estimand声明（LATE at cutoff）
