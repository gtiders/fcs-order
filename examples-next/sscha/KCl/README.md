# KCl SSCHA

本案例在 600 K 使用 pypolymlp 势函数，两个实现都采用 100 个快照、50 次迭代和随机种子 42：

- fit_phonopy.py：运行 phonopy 的 MLPSSCHA；
- fit_mlfcs.py：运行 MLFCS 原生 SSCHA；
- analyze.py：合并两套自由能迭代记录；
- plot_bands.py：重新计算并绘制四幅声子谱对比；
- plot_free_energy.py：重新绘制自由能收敛图。

输入和第三方参考数据位于 input/ 与 reference/。所有力常数、自由能记录和图片都从新脚本重新生成。

依次执行：

    uv run --with pypolymlp --with phonopy python fit_phonopy.py
    uv run --with pypolymlp --with phonopy python fit_mlfcs.py
    uv run python analyze.py
    uv run --with phonopy --with seekpath --with matplotlib python plot_bands.py
    uv run --with matplotlib python plot_free_energy.py
    uv run python compare_legacy.py
