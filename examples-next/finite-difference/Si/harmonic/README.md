# Si 有限差分二阶力常数

本案例使用 4x4x4 reference supercell 和 0.01 Angstrom 位移，从已保存的 VASP OUTCAR 重建 FC2。
source/ 保留位移结构、VASP 输入和力计算结果；reference/phonopy/ 保留 phonopy 的参考路线。

按顺序执行：

    uv run python collect_forces.py
    uv run python fit.py
    uv run python plot.py --supercell input/supercell.vasp --force-constants results/FORCE_CONSTANTS_2ND --output figures/mlfcs-phonon-band.png

collect_forces.py 只负责验证 sow 顺序并收集力，fit.py 负责有限差分和导出，plot.py 只负责绘图。
结果应与旧案例的 FC2 和 phonopy 参考声子谱进行比较。
