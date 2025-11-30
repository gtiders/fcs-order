# Phono3py Wedge3.py Optimization Plan

## 1. 背景与目标 (Background & Objective)
`wedge3.py` 是 `phono3py` 中用于寻找不可约三原子三元组（irreducible triplets）的核心脚本。随着超胞尺寸的增加，三元组的数量呈立方级增长，当前的纯 Python 实现（虽然有部分 Numba 加速）在处理大规模系统时面临严重的性能瓶颈。

本计划旨在通过算法优化和计算加速，显著降低计算三阶力常数（FC3）所需的预处理时间。

## 2. 瓶颈分析 (Bottleneck Analysis)

通过代码审查和分析，我们确定了以下四个主要的性能瓶颈：

1.  **三元组去重效率低 ($O(N^2)$)**
    *   **现状**: 使用线性扫描 (`_triplet_in_list`) 在已发现的列表中查找三元组。
    *   **问题**: 随着列表 `self.alllist` 变大，查找时间呈平方级增长。
    
2.  **距离计算开销大**
    *   **现状**: 在 Python 的四重/五重循环中手动计算原子图像间的欧几里得距离。
    *   **问题**: Python 解释器在处理这种密集型数值计算时开销巨大，虽然使用了 `track` 显示进度，但单步耗时太长。

3.  **对称性扫描与系数矩阵构建 (Python Loop Overhead)**
    *   **现状**: `_reduce` 方法中核心的 `for iperm` 和 `for isym` 循环是纯 Python 实现。
    *   **问题**: 对于每个候选三元组，都要进行 $6 \times N_{sym}$ 次迭代，且内部包含大量的数组索引和切片操作，Python 循环开销显著。

4.  **变换矩阵构建效率低 (缺少 BLAS 加速)**
    *   **现状**: `_build_transformationarray3` 使用 5 层嵌套循环逐元素计算矩阵乘积。
    *   **问题**: 未利用 CPU 的 SIMD 指令集或 BLAS 库（如 MKL/OpenBLAS）的高效矩阵乘法能力。

---

## 3. 详细优化方案与代码对比 (Detailed Optimization Plan)

### 策略 1: 引入哈希集合实现 O(1) 查找

**目标**: 消除 $O(N^2)$ 的查找瓶颈。

#### 改动前 (Before)
```python
# In Wedge._reduce() loop
triplet[0] = ii
triplet[1] = jj
triplet[2] = kk
if _triplet_in_list(triplet, self.alllist, self.nalllist):
    continue
```

#### 改动后 (After)
```python
# In Wedge.__init__
self.visited_triplets = set()

# In Wedge._reduce() loop
# 直接使用元组在 set 中查找，无需调用 Numba 函数进行线性扫描
if (ii, jj, kk) in self.visited_triplets:
    continue

# 当发现新的等价三元组时 (在内层循环)
self.visited_triplets.add((triplet_sym[0], triplet_sym[1], triplet_sym[2]))
```

---

### 策略 2: JIT 加速距离筛选

**目标**: 消除 Python 解释器在密集数值计算中的开销。

#### 改动前 (Before)
```python
# In Wedge._reduce() - 纯 Python 多层循环
for iaux in range(n2equi):
    # ... 手动计算 car2_0, car2_1 ...
    for jaux in range(n3equi):
        # ... 手动计算 car3_0 ...
        d2 = (car3_0 - car2_0)**2 + ...
        if d2 < d2_min:
            d2_min = d2
```

#### 改动后 (After)
```python
# 新增 JIT 函数
@jit(nopython=True)
def _compute_min_distance(n2equi, n3equi, shift2all, shift3all, lattvec, coordall, jj, kk):
    d2_min = np.inf
    # 预取坐标
    c_jj = coordall[:, jj]
    c_kk = coordall[:, kk]
    
    for iaux in range(n2equi):
        # ... C 级速度的计算 ...
        for jaux in range(n3equi):
             # ... C 级速度的计算 ...
             if d2 < d2_min:
                d2_min = d2
    return d2_min

# In Wedge._reduce()
d2_min = _compute_min_distance(
    n2equi, n3equi, shift2all, shift3all, lattvec, coordall, jj, kk
)
```

---

### 策略 3: JIT 加速对称性扫描核心循环

**目标**: 加速 `coeffi` 矩阵的构建过程。这部分逻辑最复杂，包含排列和对称操作。

#### 改动前 (Before)
```python
# In Wedge._reduce()
coeffi[:, :] = 0.0
nnonzero = 0
for iperm in range(6):
    # ... 设置 triplet_perm ...
    for isym in range(nsym):
        # ... 大量的数组操作和逻辑判断 ...
        # ... 调用 _triplets_are_equal ...
        # ... 填充 self.transformation ...
        # ... 填充 coeffi ...
```

#### 改动后 (After)
```python
# 新增 JIT 函数 (这是一个简化的接口示意，实际需要传入所有相关数组)
@jit(nopython=True)
def _scan_symmetry_and_build_coeffi(
    triplet, permutations, id_equi, ind_cell, ind_species, ngrid, 
    rot, rot2, nonzero, nequi_arr, equilist, allequilist, transformation, coeffi, 
    current_nlist_idx
):
    # 将双重循环逻辑全部移入此函数
    # ...
    return nnonzero

# In Wedge._reduce()
coeffi[:, :] = 0.0
nnonzero = _scan_symmetry_and_build_coeffi(
    triplet, permutations, id_equi, ind_cell, ind_species, ngrid, 
    self.rot, self.rot2, self.nonzero, self.nequi, equilist, 
    self.allequilist, self.transformation, coeffi, self.nlist - 1
)
```

---

### 策略 4: 使用矩阵乘法 (GEMM) 加速变换矩阵

**目标**: 利用 BLAS 库加速张量收缩。

#### 改动前 (Before)
```python
@jit(nopython=True)
def _build_transformationarray3(...):
    for ii in range(nlist):
        for jj in range(ne):
            for kk in range(27):
                for ll in range(nind):
                    s = 0.0
                    for iaux in range(27):
                        s += transformation[...] * transformationaux[...]
                    out_array[...] = s
```

#### 改动后 (After)
```python
@jit(nopython=True)
def _build_transformationarray3_fast(
    transformation, transformationaux, nequi, nindependentbasis, nlist, out_array
):
    for ii in range(nlist):
        ne = nequi[ii]
        nind = nindependentbasis[ii]
        
        # 提取矩阵块
        # (27, nind)
        aux_mat = np.ascontiguousarray(transformationaux[:, :nind, ii])
        
        for jj in range(ne):
            # (27, 27)
            trans_mat = np.ascontiguousarray(transformation[:, :, jj, ii])
            
            # 矩阵乘法 (BLAS 加速)
            # (27, 27) x (27, nind) -> (27, nind)
            res = np.dot(trans_mat, aux_mat)
            
            # 结果写回 4D 数组
            # 注意处理极小值归零
            for kk in range(27):
                for ll in range(nind):
                    val = res[kk, ll]
                    if np.abs(val) < 1e-12:
                        val = 0.0
                    out_array[kk, ll, jj, ii] = val
```

---

## 4. 实施步骤 (Implementation Steps)

1.  **备份**: 复制 `wedge3.py` 为 `wedge3_backup.py`。
2.  **引入 Set**: 修改 `Wedge.__init__` 添加 `self.visited_triplets`，并在 `_reduce` 中使用。
3.  **粘贴 JIT 函数**: 将上述定义的 `_compute_min_distance`, `_scan_symmetry_and_build_coeffi`, `_build_transformationarray3_fast` 函数添加到文件顶部（`class Wedge` 之前）。
4.  **替换逻辑**: 修改 `_reduce` 方法，移除旧的循环块，替换为对新函数的调用。
5.  **测试**: 运行 Phono3py 的标准测试用例或用户的特定算例，验证正确性并评估加速效果。
