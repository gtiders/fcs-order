# MLFCS 示例

本目录按 MLFCS 的工作流功能组织。每个材料案例都有自己的输入、脚本、参考数据和说明，脚本不跨案例复用。

当前迁移中的 Si 案例：

- finite-difference/Si：有限差分 FC2、FC3 和参考数据；
- fitting/Si：FC2、FC3/FC4 联合拟合；
- transport/Si：基于有限差分或拟合力常数的热导率计算。

当前迁移中的 Ba8Ga16Ge30 案例：

- finite-difference/Ba8Ga16Ge30：使用公开 hiPhive FCP 的 ASE 有限差分 FC2、FC3；
- fitting/Ba8Ga16Ge30：静态训练集和温度相关 MD 快照的有效 IFC 拟合；
- md/Ba8Ga16Ge30：生成温度相关 NVT/NVE 快照；
- transport/Ba8Ga16Ge30：将 MLFCS IFC 临时转换为 phono3py 输入并运行 RTA。

生成的力常数、缓存、日志和中间数组默认写入案例的 results/，不作为输入提交。声子谱等用于审查的图片保存在 figures/。
