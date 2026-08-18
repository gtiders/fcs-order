"""Compare regenerated KCl SSCHA data with the legacy case outputs."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from common import RESULTS

ROOT = Path(__file__).resolve().parents[3]
LEGACY = ROOT / "examples" / "cases" / "KCl" / "sscha" / "output"


def _compare_hdf5(new_path: Path, old_path: Path) -> None:
    with h5py.File(new_path, "r") as new, h5py.File(old_path, "r") as old:
        new_keys = []
        old_keys = []
        new.visititems(
            lambda key, value: new_keys.append(key) if isinstance(value, h5py.Dataset) else None
        )
        old.visititems(
            lambda key, value: old_keys.append(key) if isinstance(value, h5py.Dataset) else None
        )
        new_keys.sort()
        old_keys.sort()
        if new_keys != old_keys:
            raise AssertionError(f"dataset keys differ for {new_path.name}")
        for key in new_keys:
            if not np.array_equal(new[key][...], old[key][...]):
                raise AssertionError(f"dataset differs: {new_path.name}:{key}")


def main() -> None:
    new_fc2 = np.load(RESULTS / "phonopy_sscha_final_fc2.npy")
    old_fc2 = np.load(LEGACY / "phonopy_sscha_final_fc2.npy")
    np.testing.assert_array_equal(new_fc2, old_fc2)

    for key in ("cartesian", "canonical"):
        new = np.load(RESULTS / "mlfcs_sscha_fc2.npz")[key]
        old = np.load(LEGACY / "mlfcs_sscha_fc2.npz")[key]
        np.testing.assert_array_equal(new, old)

    _compare_hdf5(RESULTS / "mlfcs_cartesian.h5", LEGACY / "mlfcs_cartesian.h5")
    _compare_hdf5(RESULTS / "mlfcs_canonical.h5", LEGACY / "mlfcs_canonical.h5")

    new = json.loads((RESULTS / "free_energy_convergence.json").read_text())
    old = json.loads((LEGACY / "free_energy_convergence.json").read_text())
    if new != old:
        raise AssertionError("free-energy history differs")

    print(
        "KCl regenerated SSCHA arrays, HDF5 datasets, and free-energy histories match legacy outputs"
    )


if __name__ == "__main__":
    main()
