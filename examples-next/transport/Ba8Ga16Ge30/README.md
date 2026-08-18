# Ba8Ga16Ge30 热导率

先用 MD 轨迹拟合温度相关有效 IFC，再运行：

    python run_rta.py --temperatures 300

脚本只读取新生成的 `mlfcs.h5`，在内部临时转换为 phono3py 所需的 FC2/FC3 HDF5，使用 3 x 3 x 3 网格进行 RTA；临时文件不会写入案例目录。参考超胞由脚本提供给 phono3py，primitive 由其自动识别。
