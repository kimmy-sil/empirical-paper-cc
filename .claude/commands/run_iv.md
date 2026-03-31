description: 执行IV/2SLS全流程分析
argument-hint: <数据文件路径>
---
检查Research Design Memo → 存在则跳过三问。

对 $ARGUMENTS 执行IV分析：
1. 第一阶段回归 + F统计量
2. 2SLS主回归
3. 倍增比计算（|β_2SLS/β_OLS|，>5警告）
4. Anderson-Rubin弱IV稳健推断
5. plausexog / Lee bounds
6. Jackknife影响力诊断
7. Complier特征描述
8. Estimand声明（LATE）
