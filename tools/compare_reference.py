"""Compare a new compact IFC tensor with a captured legacy sparse map."""

from __future__ import annotations

import argparse

import h5py
import numpy as np


def read_force_constants(handle: h5py.File, order: int) -> np.ndarray:
    node = handle[f"force_constants/{order}"]
    if isinstance(node, h5py.Dataset):
        return node[:]
    shape = tuple(int(value) for value in node.attrs["dense_shape"])
    result = np.zeros(shape, dtype=float)
    counts = np.zeros(shape[:order], dtype=np.int16)
    for cluster, tensor in zip(node["clusters"], node["tensors"], strict=True):
        key = tuple(int(atom) for atom in cluster)
        result[key] += tensor
        counts[key] += 1
    nonzero = counts > 0
    result[nonzero] /= counts[nonzero].reshape((-1,) + (1,) * order)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("new_hdf5")
    parser.add_argument("legacy_npz")
    parser.add_argument("--order", type=int, choices=(3, 4), required=True)
    args = parser.parse_args()

    with h5py.File(args.new_hdf5) as handle:
        new = read_force_constants(handle, args.order)
    with np.load(args.legacy_npz) as legacy:
        keys = legacy["keys"]
        reference = legacy["values"]

    # Legacy keys store Cartesian directions first, followed by atom indices.
    directions = keys[:, : args.order]
    atoms = keys[:, args.order :]
    actual = np.asarray(
        [
            new[tuple(atom_indices) + tuple(direction_indices)]
            for atom_indices, direction_indices in zip(atoms, directions, strict=True)
        ]
    )
    error = actual - reference
    scale = np.maximum(np.abs(reference), 1e-12)
    print(f"components: {len(error)}")
    print(f"max_abs: {np.max(np.abs(error)):.12e}")
    print(f"rms: {np.sqrt(np.mean(error**2)):.12e}")
    print(f"max_relative: {np.max(np.abs(error) / scale):.12e}")
    print(f"new_max_abs: {np.max(np.abs(actual)):.12e}")
    print(f"legacy_max_abs: {np.max(np.abs(reference)):.12e}")


if __name__ == "__main__":
    main()
