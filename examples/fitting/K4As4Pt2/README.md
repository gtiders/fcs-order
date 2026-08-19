# K4As4Pt2 拟合

输入快照来自 `reference/DFTSET_RAND`，已转换为严格 ASE `extxyz` 的副本放在 `input/train.extxyz`。`input/primitive.vasp` 和 `input/reference.vasp` 是拟合使用的原胞与参考超胞。

运行 `uv run python fit.py --body-order-4 3` 拟合二、三、四阶力常数；将 `--body-order-4` 改为 `4` 可生成四体四阶截断。结果写入 `results/three-body/` 或 `results/four-body/`，同时包含原生 HDF5、phonopy 二阶文本、ShengBTE 三阶和四阶文本。

`reference/` 仅保存 ALAMODE 原始输入，便于核对数据来源；它不是新拟合流程的隐式输入。

绘制三体截断拟合结果的声子谱：

    uv run --with phonopy --with seekpath --with matplotlib python plot.py
