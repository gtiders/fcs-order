# Ba8Ga16Ge30 有限差分

本案例使用公开 hiPhive FCP 作为 ASE 计算器，在 2 x 2 x 2 扩包上生成二阶和三阶有限差分力常数。二阶截断为 5.40 Å，三阶截断为 4.35 Å，最大体阶均为二体。

运行：

    python run.py --order 2 3 --overwrite

结果写入 `results/harmonic/` 和 `results/three-phonon/`，保留新生成的力评估、`mlfcs.h5`、phonopy 二阶文本和 ShengBTE 三阶文本。
