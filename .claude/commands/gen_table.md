description: 生成LaTeX三线表
argument-hint: <回归结果对象或CSV路径>
---
将 $ARGUMENTS 格式化为LaTeX三线表（booktabs + threeparttable）。
自动添加：标准误括号、显著性星号、N和R²、固定效应Yes/No行、注释行。
保存为 .tex 文件到 output/tables/。
论文中用 \input{} 引用。
