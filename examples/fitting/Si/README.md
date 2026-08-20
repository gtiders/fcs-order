# Si 力常数拟合

本组案例包括独立 FC2 和联合 FC2-FC4 两条路线。每个案例有独立的 fit.py、analyze.py
和必要的绘图脚本；拟合结果不直接作为其他案例的隐藏依赖。MLFCS 不支持把外部低阶 IFC
冻结到高阶拟合中，所有阶数必须在同一个 Wick 参数空间中联合确定。

绘制联合 FC2-FC4 拟合中的 FC2 声子谱：

    uv run --with phonopy --with seekpath --with matplotlib python plot.py
