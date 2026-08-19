# Ba8Ga16Ge30 拟合

## 静态训练集

`input/training.extxyz` 是公开 hiPhive 示例的 200 个训练快照，`input/primitive.vasp` 和 `input/reference.vasp` 是对应结构。运行：

    python run.py

结果写入 `results/`，包含通用 `mlfcs.h5`、phonopy 二阶文本和 ShengBTE 三阶/四阶文本。模型使用二阶 5.40 Å、三阶和四阶 4.35 Å 截断，三者最大体阶均为二体。

## 温度相关有效 IFC

对 MD 目录中的 `nve.extxyz` 运行：

    python fit_effective_ifcs.py ../../md/Ba8Ga16Ge30/results/T300K

结果写入 `results/T300K/mlfcs/`。持久化文件不包含 phono3py 专用 FC3 HDF5；需要 phono3py 时由输运脚本临时转换。

绘制 T600K 拟合 FC2 的声子谱：

    uv run --with phonopy --with seekpath --with matplotlib python plot.py
