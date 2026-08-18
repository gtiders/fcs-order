# Si 拟合三阶和四阶力常数

input/train.extxyz 包含 100 个 64 原子训练构型。案例独立拟合 FC2、FC3 和 FC4，使用原有 cutoff、
body-order 和求解器设置。

    uv run python fit.py

fit.py 只负责拟合和逐阶导出；分析和绘图使用单独脚本。生成结果写入 results/，迁移时与旧案例的
稀疏键、张量、拟合残差和声子谱逐项比较。
