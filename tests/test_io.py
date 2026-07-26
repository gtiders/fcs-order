import h5py
import numpy as np
import pytest
from ase.build import bulk

from mlfcs import ForceConstantCalculation, SparseOrderForceConstants


def test_reap_keeps_sparse_clusters_and_hdf5_writes_them(tmp_path):
    calculation = ForceConstantCalculation(
        bulk("Si", "diamond", a=5.43),
        order=3,
        supercell=(2, 2, 2),
        cutoff=-1,
    )
    forces = np.zeros((len(calculation.plan), len(calculation.supercell), 3))
    result = calculation.reap(forces)
    assert result.arrays == {}
    assert result.orders == (3,)

    target = tmp_path / "fc.h5"
    result.write(target, format="hdf5")
    with h5py.File(target) as handle:
        group = handle["force_constants/3"]
        assert group.attrs["representation"] == "sparse-cluster"
        assert group["clusters"].shape[1] == 3
        assert group["tensors"].shape[1:] == (3, 3, 3)


def test_dense_materialization_has_a_memory_budget():
    sparse = SparseOrderForceConstants(
        order=5,
        n_primitive=2,
        n_supercell=64,
        clusters=np.empty((0, 5), dtype=np.int32),
        tensors=np.empty((0,) + (3,) * 5),
    )
    with pytest.raises(MemoryError, match="write HDF5 directly"):
        sparse.to_dense(max_bytes=1_000_000)
