# MoS2 单层旋转约束案例

`input/training.extxyz` 是带显式位移和力的训练集，`input/primitive.vasp` 和
`input/reference.vasp` 固定 primitive 与参考超胞的原子标签。案例只拟合二阶力常数，
随后分别输出只施加 ASR 和同时施加 Born-Huang/Huang 条件的结果。

运行 `python run.py`。结果写入 `results/asr/` 和 `results/born-huang-huang/`，生成的缓存
和力常数不作为输入数据提交。

绘制两种二阶约束结果的对比声子谱：

    uv run --with phonopy --with seekpath --with matplotlib python plot.py
