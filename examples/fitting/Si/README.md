# Si 力常数拟合

本组案例包括独立 FC2、联合 FC2-FC4 和冻结 FC2 三条路线。每个案例有独立的 fit.py、analyze.py
和必要的绘图脚本；拟合结果不直接作为其他案例的隐藏依赖。

绘制联合 FC2-FC4 拟合中的 FC2 声子谱：

    uv run --with phonopy --with seekpath --with matplotlib python plot.py
