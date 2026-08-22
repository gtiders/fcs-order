"""Compare MLFCS and phonopy finite-difference FC2 for the same KCl potential."""

from __future__ import annotations

from mlfcs import realize_force_constants
import json

import numpy as np
from common import FIGURES, RESULTS, ase_from_phonopy, bands, harmonic_phonopy, map_reference_to_phonopy, mlfcs_calculator, mlfcs_working_cells
from matplotlib import pyplot as plt

from mlfcs import ForceConstantCalculation, build_supercell
from mlfcs.force_constants.dense import expand_compact_fc2

DISPLACEMENT = 0.01


def _set_path_ticks(axis, segments, distances):
    ticks = {}
    for (start, end), distance in zip(segments, distances, strict=True):
        for location, label in ((float(distance[0]), start), (float(distance[-1]), end)):
            label = r"$\Gamma$" if label == "GAMMA" else label
            if location in ticks and label not in ticks[location].split("|"):
                ticks[location] += f"|{label}"
            else:
                ticks.setdefault(location, label)
    for location in ticks:
        axis.axvline(location, color="#d7e0e0", linewidth=0.6, zorder=0)
    axis.set_xticks(list(ticks), list(ticks.values()))


def _mlfcs_fc2(primitive, reference, calculator, *, target=None):
    result = ForceConstantCalculation(
        primitive,
        order=2,
        reference=reference,
        cutoff=None,
        displacement=DISPLACEMENT,
        verbose=True,
    ).run(calculator)
    realized = result if target is None else realize_force_constants(result, target, primitive=primitive)
    compact = realized.materialize(2, max_bytes=None)
    return result, expand_compact_fc2(compact, realized.relation.reference)


def _phonopy_fc2(phonon, calculator):
    phonon.force_constants = None
    phonon.generate_displacements(distance=DISPLACEMENT, is_plusminus=True)
    forces = []
    for displaced in phonon.supercells_with_displacements:
        atoms = ase_from_phonopy(displaced)
        atoms.calc = calculator
        forces.append(atoms.get_forces())
    phonon.forces = np.asarray(forces)
    phonon.produce_force_constants(
        calculate_full_force_constants=True,
        fc_calculator="symfc",
    )
    return np.asarray(phonon.force_constants)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    phonon, primitive, reference = mlfcs_working_cells()
    calculator = mlfcs_calculator()
    mlfcs_result, mlfcs_fc2 = _mlfcs_fc2(primitive, reference, calculator)
    large_reference = build_supercell(primitive, (4, 4, 4))
    large_result, large_mlfcs_fc2 = _mlfcs_fc2(
        primitive,
        large_reference,
        calculator,
        target=reference,
    )
    mlfcs_fc2 = map_reference_to_phonopy(mlfcs_fc2, phonon)
    large_mlfcs_fc2 = map_reference_to_phonopy(large_mlfcs_fc2, phonon)
    phonopy_fc2 = _phonopy_fc2(phonon, calculator)
    difference = mlfcs_fc2 - phonopy_fc2
    large_difference = large_mlfcs_fc2 - phonopy_fc2
    metrics = {
        "displacement_A": DISPLACEMENT,
        "mlfcs_force_evaluations": 8,
        "phonopy_force_evaluations": len(phonon.supercells_with_displacements),
        "fc2_max_abs_eV_per_A2": float(np.max(np.abs(difference))),
        "fc2_relative_frobenius": float(
            np.linalg.norm(difference) / np.linalg.norm(phonopy_fc2)
        ),
        "large_reference_atoms": len(large_reference),
        "large_reference_cutoff_A": large_result.metadata["cutoff_angstrom"],
        "large_fc2_max_abs_eV_per_A2": float(np.max(np.abs(large_difference))),
        "large_fc2_relative_frobenius": float(
            np.linalg.norm(large_difference) / np.linalg.norm(phonopy_fc2)
        ),
    }
    (RESULTS / "finite_difference_comparison.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="ascii"
    )
    np.save(RESULTS / "mlfcs_finite_difference_fc2.npy", mlfcs_fc2)
    np.save(RESULTS / "phonopy_finite_difference_fc2.npy", phonopy_fc2)
    np.save(RESULTS / "mlfcs_large_finite_difference_fc2.npy", large_mlfcs_fc2)

    distance_m, frequencies_m, segments = bands(phonon, mlfcs_fc2)
    distance_l, frequencies_l, _ = bands(phonon, large_mlfcs_fc2)
    distance_p, frequencies_p, _ = bands(phonon, phonopy_fc2)
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for distance, frequencies, color, label in (
        (distance_p, frequencies_p, "#4f8785", "phonopy"),
        (distance_l, frequencies_l, "#786fa6", "MLFCS 4×4×4 → 2×2×2"),
        (distance_m, frequencies_m, "#d49368", "MLFCS"),
    ):
        first = True
        for segment, values in zip(distance, frequencies, strict=True):
            for branch in values.T:
                axis.plot(
                    segment,
                    branch,
                    color=color,
                    linewidth=1.5,
                    label=label if first else None,
                )
                first = False
    axis.axhline(0, color="#777777", linewidth=0.8, linestyle=":")
    _set_path_ticks(axis, segments, distance_m)
    axis.set_xlabel("Wave vector")
    axis.set_ylabel("Frequency (THz)")
    axis.set_title("KCl finite-difference FC2 from the same potential")
    axis.legend()
    figure.savefig(FIGURES / "finite_difference_mlfcs_vs_phonopy.png", dpi=220)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
