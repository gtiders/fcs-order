# Ba8Ga16Ge30 温度相关分子动力学

本案例使用公开 hiPhive Model-4 FCP，通过 ASE 计算器生成 300、400、500、600 K 的 NVT/NVE 轨迹。每个温度使用 2 x 2 x 2 扩包、10000 步 Langevin 平衡和 5000 步 NVE，NVE 每 50 步保存一帧。

输入 FCP 和 54 原子参考结构位于 `input/`。运行：

    python run.py --temperatures 300 --overwrite

新轨迹和带力的 `nve.extxyz` 写入 `results/T300K/`。有效 IFC 拟合使用：

    python ../../fitting/Ba8Ga16Ge30/fit_effective_ifcs.py results/T300K

拟合结果写入 `examples/fitting/Ba8Ga16Ge30/results/T300K/`。
