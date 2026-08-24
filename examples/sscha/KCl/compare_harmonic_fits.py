"""Fit the same random KCl data with MLFCS and hiPhive and compare FC2."""

from __future__ import annotations

from mlfcs import write_force_constants
import json

import numpy as np
from common import FIGURES, RESULTS, SEED, ase_from_phonopy, bands, harmonic_phonopy, map_reference_to_phonopy, mlfcs_calculator, mlfcs_working_cells
from hiphive import ClusterSpace, ForceConstantPotential, StructureContainer
from matplotlib import pyplot as plt
from trainstation import Optimizer

from mlfcs.force_constants.dense import expand_compact_fc2
from mlfcs.structure.relation import StructureRelation
from mlfcs.interactions.enumerate import resolve_primitive_cutoff
from mlfcs.fitting import ForceConstantFitter

SNAPSHOTS = 100
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


def _training_structures(reference, calculator):
    rng = np.random.default_rng(SEED)
    displacements = rng.normal(
        scale=DISPLACEMENT, size=(SNAPSHOTS, len(reference), 3)
    )
    displacements -= displacements.mean(axis=1, keepdims=True)
    structures = []
    for displacement in displacements:
        atoms = reference.copy()
        atoms.positions += displacement
        atoms.calc = calculator
        forces = atoms.get_forces()
        atoms.calc = None
        atoms.new_array("displacements", displacement)
        atoms.new_array("forces", forces)
        structures.append(atoms)
    return structures


def _fit_mlfcs(primitive, reference, structures):
    fitter = ForceConstantFitter(
        primitive,
        reference,
        orders=(2,),
        cutoffs={2: None},
    )
    result = fitter.fit(structures, validation_split=0.0, acoustic_sum_rule=True)
    compact = result.force_constants.materialize(2, max_bytes=None)
    relation = StructureRelation.from_atoms(primitive, reference)
    return result, expand_compact_fc2(compact, relation.reference)


def _fit_hiphive(primitive, reference, structures, cutoff):
    cluster_space = ClusterSpace(primitive, [cutoff])
    container = StructureContainer(cluster_space)
    for atoms in structures:
        prepared = reference.copy()
        prepared.new_array("displacements", atoms.arrays["displacements"].copy())
        prepared.new_array("forces", atoms.arrays["forces"].copy())
        container.add_structure(prepared)
    optimizer = Optimizer(
        container.get_fit_data(),
        fit_method="least-squares",
        train_size=1.0,
        standardize=False,
    )
    optimizer.train()
    potential = ForceConstantPotential(cluster_space, optimizer.parameters)
    force_constants = potential.get_force_constants(reference)
    return optimizer, force_constants.get_fc_array(order=2)


def _force_rmse(structures, fc2):
    residuals = []
    for atoms in structures:
        prediction = -np.einsum("ijab,jb->ia", fc2, atoms.arrays["displacements"])
        residuals.append(prediction - atoms.arrays["forces"])
    return float(np.sqrt(np.mean(np.square(residuals))))


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    phonon, primitive, reference = mlfcs_working_cells()
    cutoff = resolve_primitive_cutoff(primitive, None, reference=reference)
    structures = _training_structures(reference, mlfcs_calculator())
    mlfcs_result, mlfcs_fc2 = _fit_mlfcs(primitive, reference, structures)
    hiphive_optimizer, hiphive_fc2 = _fit_hiphive(
        primitive, reference, structures, cutoff
    )

    difference = mlfcs_fc2 - hiphive_fc2
    scale = max(float(np.linalg.norm(hiphive_fc2)), np.finfo(float).tiny)
    metrics = {
        "cutoff_A": cutoff,
        "snapshots": SNAPSHOTS,
        "displacement_A": DISPLACEMENT,
        "mlfcs_force_rmse_eV_per_A": _force_rmse(structures, mlfcs_fc2),
        "hiphive_force_rmse_eV_per_A": _force_rmse(structures, hiphive_fc2),
        "fc2_max_abs_eV_per_A2": float(np.max(np.abs(difference))),
        "fc2_relative_frobenius": float(np.linalg.norm(difference) / scale),
        "mlfcs_fit_rmse_eV_per_A": float(
            mlfcs_result.training_force_rmse
        ),
        "hiphive_fit_rmse_eV_per_A": float(hiphive_optimizer.rmse_train),
    }
    (RESULTS / "harmonic_fit_comparison.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="ascii"
    )
    np.save(RESULTS / "mlfcs_random_fit_fc2.npy", mlfcs_fc2)
    np.save(RESULTS / "hiphive_random_fit_fc2.npy", hiphive_fc2)
    write_force_constants(mlfcs_result.force_constants, 
        RESULTS / "mlfcs_harmonic.h5", format="hdf5"
    )
    write_force_constants(mlfcs_result.force_constants, 
        RESULTS / "FORCE_CONSTANTS_MLFCS_HARMONIC", format="phonopy", order=2
    )

    distance_m, frequencies_m, labels = bands(
        phonon, map_reference_to_phonopy(mlfcs_fc2, phonon)
    )
    distance_h, frequencies_h, _ = bands(
        phonon, map_reference_to_phonopy(hiphive_fc2, phonon)
    )
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for distance, frequencies, color, name in (
        (distance_h, frequencies_h, "#4f8785", "hiPhive"),
        (distance_m, frequencies_m, "#d49368", "MLFCS"),
    ):
        first_branch = True
        for segment, values in zip(distance, frequencies, strict=True):
            for branch in values.T:
                axis.plot(
                    segment,
                    branch,
                    color=color,
                    linewidth=1.5,
                    label=name if first_branch else None,
                )
                first_branch = False
    axis.axhline(0, color="#777777", linewidth=0.8, linestyle=":")
    _set_path_ticks(axis, labels, distance_m)
    axis.set_ylabel("Frequency (THz)")
    axis.set_xlabel("Wave vector")
    axis.set_title("KCl random-displacement harmonic fits at identical cutoff")
    axis.legend()
    figure.savefig(FIGURES / "harmonic_fit_mlfcs_vs_hiphive.png", dpi=220)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
