"""Compare regenerated finite-difference archives with the legacy case."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

CASE = Path(__file__).resolve().parent
NEW = CASE / "results"
OLD = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "cases"
    / "K4As4Pt2"
    / "finite-difference"
    / "polymlp"
)


def compare_hdf5(new_path: Path, old_path: Path) -> None:
    with h5py.File(new_path, "r") as new, h5py.File(old_path, "r") as old:
        new_keys: list[str] = []
        old_keys: list[str] = []
        new.visititems(
            lambda key, value: new_keys.append(key) if isinstance(value, h5py.Dataset) else None
        )
        old.visititems(
            lambda key, value: old_keys.append(key) if isinstance(value, h5py.Dataset) else None
        )
        if sorted(new_keys) != sorted(old_keys):
            raise AssertionError(f"HDF5 dataset paths differ: {new_path.name}")
        for key in new_keys:
            np.testing.assert_array_equal(new[key][...], old[key][...], err_msg=key)


def main() -> None:
    for name in ("harmonic", "three-phonon"):
        new = NEW / name
        old = OLD / name
        np.testing.assert_array_equal(
            np.load(new / "forces.npz")["forces"], np.load(old / "forces.npz")["forces"]
        )
        compare_hdf5(new / "mlfcs.h5", old / "mlfcs.h5")
    compare_hdf5(NEW / "harmonic" / "fc2.h5", OLD / "harmonic" / "fc2.h5")
    compare_hdf5(NEW / "phono3py-reference" / "fc2.h5", OLD / "phono3py-reference" / "fc2.h5")
    compare_hdf5(NEW / "phono3py-reference" / "fc3.h5", OLD / "phono3py-reference" / "fc3.h5")
    print(
        "K4As4Pt2 regenerated finite-difference force archives and HDF5 datasets match legacy outputs"
    )


if __name__ == "__main__":
    main()
