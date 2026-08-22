"""Compare two ShengBTE FC3 files as canonical sparse physical blocks."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from ase.io import read

CASE = Path(__file__).resolve().parent
PRIMITIVE = read(CASE.parents[1] / "input" / "reference.vasp")
PERIODIC_REPEATS = np.array((2, 2, 2), dtype=int)


def _integer_translation(vector: np.ndarray) -> tuple[int, int, int]:
    fractional = np.asarray(vector) @ np.linalg.inv(PRIMITIVE.cell.array)
    rounded = np.rint(fractional).astype(int)
    if not np.allclose(fractional, rounded, atol=2e-6, rtol=0.0):
        raise ValueError(f"ShengBTE offset is not an integer primitive translation: {vector}")
    return tuple(int(value) for value in rounded)


def _parse(path: Path) -> tuple[int, dict[tuple, np.ndarray]]:
    lines = path.read_text().splitlines()
    cursor = 0
    while not lines[cursor].strip():
        cursor += 1
    count = int(lines[cursor].strip())
    cursor += 1
    blocks = {}
    for _ in range(count):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        cursor += 1  # block serial number
        offset_1 = _integer_translation(np.fromstring(lines[cursor], sep=" "))
        cursor += 1
        offset_2 = _integer_translation(np.fromstring(lines[cursor], sep=" "))
        cursor += 1
        sites = tuple(int(value) - 1 for value in lines[cursor].split())
        cursor += 1
        tensor = np.zeros((3, 3, 3), dtype=float)
        for _ in range(27):
            fields = lines[cursor].split()
            tensor[int(fields[0]) - 1, int(fields[1]) - 1, int(fields[2]) - 1] = float(fields[3])
            cursor += 1
        offsets = (np.zeros(3, dtype=int), np.asarray(offset_1), np.asarray(offset_2))
        best = None
        for permutation in itertools.permutations(range(3)):
            anchor = offsets[permutation[0]]
            relative = tuple(
                tuple(((offsets[index] - anchor) % PERIODIC_REPEATS).tolist())
                for index in permutation[1:]
            )
            key = (
                tuple(sites[index] for index in permutation),
                relative,
            )
            candidate = tensor.transpose(permutation)
            if best is None or key < best[0]:
                best = (key, candidate)
        key, canonical_tensor = best
        del best
        if key in blocks and not np.allclose(blocks[key], canonical_tensor, atol=2e-8, rtol=0.0):
            raise ValueError(f"inconsistent duplicate physical FC3 key in {path}: {key}")
        blocks[key] = canonical_tensor
    return count, blocks


def main() -> None:
    hiphive_count, hiphive = _parse(CASE / "hiphive" / "FORCE_CONSTANTS_3RD")
    mlfcs_count, mlfcs = _parse(CASE / "mlfcs" / "FORCE_CONSTANTS_3RD")
    common = set(hiphive) & set(mlfcs)
    only_hiphive = set(hiphive) - set(mlfcs)
    only_mlfcs = set(mlfcs) - set(hiphive)
    differences = [hiphive[key] - mlfcs[key] for key in common]
    stacked = (
        np.concatenate([value.ravel() for value in differences]) if differences else np.zeros(0)
    )
    reference = np.concatenate([mlfcs[key].ravel() for key in common]) if common else np.zeros(0)
    summary = {
        "hiphive_block_count": hiphive_count,
        "mlfcs_block_count": mlfcs_count,
        "hiphive_canonical_key_count": len(hiphive),
        "mlfcs_canonical_key_count": len(mlfcs),
        "common_key_count": len(common),
        "only_hiphive_key_count": len(only_hiphive),
        "only_mlfcs_key_count": len(only_mlfcs),
        "common_tensor_maximum_difference": float(np.max(np.abs(stacked), initial=0.0)),
        "common_tensor_relative_difference": float(
            np.linalg.norm(stacked) / max(np.linalg.norm(reference), np.finfo(float).tiny)
        ),
    }
    output = CASE / "hiphive" / "shengbte-fc3-comparison.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
