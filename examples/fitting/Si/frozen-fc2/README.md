# Si 冻结二阶参数拟合

本案例验证从已有 FC2 冻结参数后继续拟合高阶力常数。基线文件放在 reference/baselines/，并分别测试
拟合 FC2 和有限差分 FC2。有限差分基线与当前训练 reference supercell 不等价时必须明确拒绝。

    uv run python fit.py --baseline harmonic-fit
    uv run python fit.py --baseline finite-difference

第一个命令应完成严格冻结拟合；第二个命令用于验证结构不相容时的拒绝路径。
