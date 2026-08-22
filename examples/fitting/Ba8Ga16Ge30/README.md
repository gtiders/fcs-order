# Ba8Ga16Ge30 温度相关拟合

本案例只保留公开的 hiPhive FCP 势函数、由它生成的分温度 MD 轨迹，以及 MLFCS 的温度相关有效 IFC 拟合。不再保留原先原子数与 reference 不一致的顶层静态训练集。

## 温度相关有效 IFC

对 MD 目录中的 `nve.extxyz` 运行，例如 300 K：

    uv run python fit_effective_ifcs.py md/results/T300K

结果写入 `results/T300K/mlfcs/`。持久化文件不包含 phono3py 专用 FC3 HDF5；需要 phono3py 时由输运脚本临时转换。

绘制拟合 FC2 的声子谱：

    uv run --with phonopy --with seekpath --with matplotlib python plot.py \
        --force-constants results/T300K/mlfcs/FORCE_CONSTANTS_2ND \
        --output results/T300K/mlfcs/phonon-band.png
