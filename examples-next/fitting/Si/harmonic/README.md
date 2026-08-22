# Si 拟合二阶力常数

input/train.extxyz 是由 ALAMODE 数据转换得到的严格 ASE 训练输入。ALAMODE 数据只作为训练来源，
不作为拟合力常数的真值参考。

    uv run python fit.py

fit.py 只执行 FC2 拟合和导出。拟合结果写入 results/；应与旧案例的拟合 FC2、有限差分 FC2 和
生成的声子谱进行比较。
