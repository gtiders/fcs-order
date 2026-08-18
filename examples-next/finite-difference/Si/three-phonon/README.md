# Si 有限差分三声子力常数

本案例使用 4x4x4 reference supercell、0.01 Angstrom 位移和 FC3 cutoff -5，从 132 个已保存的
thirdorder 兼容力计算中重建 FC3。source/ 是 MLFCS 计算顺序，reference/thirdorder/ 保留 thirdorder
的 vasprun.xml 和输入，便于核对结构与力。

按顺序执行：

    uv run python collect_forces.py
    uv run python fit.py

两个脚本分别负责力收集和 FC3 拟合/导出。生成的结果由上述命令重新生成，并与旧案例及 thirdorder 参考结果比较。
