"""Compare the regenerated K4As4Pt2 fit with the legacy fit output."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from mlfcs import read_hdf5

CASE = Path(__file__).resolve().parent
NEW = CASE / "results" / "three-body"
OLD = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "cases"
    / "K4As4Pt2"
    / "fitting"
    / "anharmonic"
    / "three-body"
)


def compare_hdf5() -> None:
    with h5py.File(NEW / "mlfcs.h5", "r") as new, h5py.File(OLD / "mlfcs.h5", "r") as old:
        new_keys: list[str] = []
        old_keys: list[str] = []
        new.visititems(
            lambda key, value: new_keys.append(key) if isinstance(value, h5py.Dataset) else None
        )
        old.visititems(
            lambda key, value: old_keys.append(key) if isinstance(value, h5py.Dataset) else None
        )
        if sorted(new_keys) != sorted(old_keys):
            raise AssertionError("native HDF5 dataset paths differ")
        for key in new_keys:
            np.testing.assert_array_equal(new[key][...], old[key][...], err_msg=key)


def main() -> None:
    compare_hdf5()
    for name in ("FORCE_CONSTANTS_2ND", "FORCE_CONSTANTS_3RD", "FORCE_CONSTANTS_4TH"):
        new = read_hdf5(NEW / "mlfcs.h5")
        old = read_hdf5(OLD / "mlfcs.h5")
        order = {"FORCE_CONSTANTS_2ND": 2, "FORCE_CONSTANTS_3RD": 3, "FORCE_CONSTANTS_4TH": 4}[name]
        np.testing.assert_array_equal(
            new.sparse[order].tensors, old.sparse[order].tensors, err_msg=name
        )
    print("K4As4Pt2 regenerated fit HDF5 and IFC tensors match the legacy three-body result")


if __name__ == "__main__":
    main()
