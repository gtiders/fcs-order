# hiPhive 公开案例基准

本可选基准使用 `materials-modeling/hiphive-examples` 公开的 Si 与 BaGaGe
笼状化合物数据。该仓库约含 114 MB Git LFS 对象，因此不直接纳入本仓库。

## 获取数据

```bash
git clone --depth 1 https://gitlab.com/materials-modeling/hiphive-examples.git
git -C hiphive-examples lfs pull
git -C hiphive-examples rev-parse HEAD
```

本次记录使用提交 `05216055abca04ef9476bb9a5ba5b0f050993b2d`。案例数据由
hiPhive 0.5 生成；运行性能对比使用当前环境中的 hiPhive 1.5 和 TrainStation
1.2，但保持论文案例的数据与模型定义不变。

## 复现

内存密集任务应逐个运行：

```bash
/usr/bin/time -v uv run python reference_tools/benchmark_hiphive_examples.py si-mlfcs
/usr/bin/time -v uv run python reference_tools/benchmark_hiphive_examples.py si-hiphive
uv run python reference_tools/benchmark_hiphive_examples.py si-compare
uv run python reference_tools/benchmark_hiphive_examples.py bagage-wick
```

Si 对比使用全部 20 个 Si250 快照、FC2/FC3 均为 9.65 Å 截断、最小二乘及
平移 ASR。两种实现均得到 150 个轨道、2598 个未约束参数和 77 个独立 ASR
约束，即 2521 个约束后自由度。

## 已记录结果

| 指标 | MLFCS | hiPhive |
|---|---:|---:|
| 力 RMSE（meV/Å） | 3.131365 | 3.131190 |
| 相对力误差 | 0.949040% | 0.948987% |
| 墙钟时间 | 427.79 s | 251.12 s |
| 峰值 RSS | 2,164,816 KiB | 1,926,200 KiB |

对齐后的张量相对 RMS 差为：FC2 `1.31e-5`，FC3 `7.60e-4`。比较前显式
对齐了超胞原子索引和张量轴，不比较序列化文件的排列。

对 200 个 BaGaGe 快照可复现地抽取 8192 个三次特征后，线性/三次特征绝对
相关系数如下：

| 统计量 | Taylor 三次特征 | Wick 三次特征 |
|---|---:|---:|
| 均值 | 0.06409 | 0.06280 |
| RMS | 0.08094 | 0.07867 |
| 95% 分位数 | 0.15781 | 0.15399 |
| 99% 分位数 | 0.21127 | 0.20241 |
| 最大值 | 0.68239 | 0.40341 |

因此 Wick 明显压低了最严重的 FC2/FC4 特征相关，但不会消除有限样本和非高斯
分布造成的全部相关性。这是抽样的原始特征诊断，并不等价于声称约束后物理设计
矩阵的每一个奇异值都会改善。

紧凑周期坐标存储与 Gram 前约束参数化使 MLFCS 峰值内存相对早期的
4,951,592 KiB 实现下降 56%，同时保持力和张量结果不变。当前剩余耗时差主要来自物理
设计特征计算，而不是 Gram 求解。

论文中的 BaGaGe 主模型使用 200 个结构、`[5.4, 4.35, 4.35] Å` 截断、双体
支撑和 10 折 OLS；其摘要记录 6052 个参数、训练 RMSE 48.17 meV/Å、验证
RMSE 69.67 meV/Å。MLFCS 的物理参数化含 25,495 个系数，但新的分块稀疏 ASR 映射会在
构造 Gram 前把它约化为 6052 个拟合坐标，使 Gram 从约 4.84 GiB 降为约 279 MiB。
完整 200 结构运行仍作为可选基准，不进入常规 CI。
