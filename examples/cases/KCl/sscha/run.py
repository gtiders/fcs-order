"""Compare harmonic and SSCHA KCl bands from phonopy and native MLFCS."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seekpath
from ase import Atoms
from matplotlib.lines import Line2D
from phonopy import load
from phonopy.file_IO import write_FORCE_CONSTANTS
from phonopy.interface.mlp import PhonopyMLP
from phonopy.sscha.core import MLPSSCHA
from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

from mlfcs.structure.geometry import StructureRelation
from mlfcs.ifc.model import ForceConstants, SparseOrderForceConstants
from mlfcs.anharmonic.sscha import SSCHA

CASE = Path(__file__).resolve().parent
DATA = CASE / "data"
HARMONIC = DATA / "phonopy_fc222_JPCM2022.yaml.xz"
SELF_CONSISTENT = DATA / "phonopy_sscha_fc_JPCM2022.yaml.xz"
POTENTIAL = DATA / "polymlp.yaml"
OUTPUT = CASE / "output"
TEMPERATURE = 600.0


def conventional_cell() -> Atoms:
    return Atoms(
        symbols=["K"] * 4 + ["Cl"] * 4,
        scaled_positions=[
            [0, 0, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0],
            [0.5, 0.5, 0.5],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0, 0, 0.5],
        ],
        cell=[6.292, 6.292, 6.292],
        pbc=True,
    )


def _ase_from_phonopy(cell) -> Atoms:
    """Build the ASE working cell without independently guessing its layout."""
    return Atoms(
        numbers=np.asarray(cell.numbers, dtype=int),
        cell=np.asarray(cell.cell, dtype=float),
        scaled_positions=np.asarray(cell.scaled_positions, dtype=float),
        pbc=True,
    )


def _map_reference_to_phonopy(full: np.ndarray, phonon) -> np.ndarray:
    reference = _ase_from_phonopy(phonon.unitcell).repeat((2, 2, 2))
    target = phonon.supercell
    if len(reference) != len(target):
        raise ValueError("MLFCS and phonopy supercells have different sizes")
    permutation = []
    for position, number in zip(target.positions, target.numbers, strict=True):
        candidates = np.flatnonzero(reference.numbers == number)
        distances = np.linalg.norm(reference.positions[candidates] - position, axis=1)
        index = int(candidates[np.argmin(distances)])
        if float(np.min(distances)) > 1e-8:
            raise ValueError("cannot map MLFCS KCl supercell onto phonopy order")
        permutation.append(index)
    permutation = np.asarray(permutation, dtype=int)
    return np.asarray(full)[np.ix_(permutation, permutation)]


def _paths(phonon):
    primitive = phonon.primitive
    structure = (
        np.asarray(primitive.cell),
        np.asarray(primitive.scaled_positions),
        np.asarray(primitive.numbers),
    )
    path_data = seekpath.get_path(structure, recipe="hpkot")
    labels = path_data["point_coords"]
    paths = []
    connections = []
    for index, (start, end) in enumerate(path_data["path"]):
        paths.append(np.linspace(labels[start], labels[end], 101))
        connections.append(
            index + 1 < len(path_data["path"]) and end == path_data["path"][index + 1][0]
        )
    return paths, path_data["path"], connections


def _bands(phonon, force_constants):
    phonon = copy.deepcopy(phonon)
    phonon.force_constants = force_constants
    paths, labels, connections = _paths(phonon)
    phonon.run_band_structure(paths, path_connections=connections)
    return phonon.band_structure.distances, phonon.band_structure.frequencies, labels


def _mlfcs_result(values: np.ndarray, reference: Atoms, primitive: Atoms) -> ForceConstants:
    relation = StructureRelation.from_atoms(primitive, reference)
    index = relation.index
    clusters = []
    sites = []
    translations = []
    tensors = []
    for site in range(index.n_primitive):
        anchor = index.representative(site)
        for atom in range(len(reference)):
            sites.append((site, int(index.primitive[atom])))
            clusters.append((anchor, atom))
            translations.append(
                index.canonical_translation(index.translations[atom] - index.translations[anchor])
            )
            tensors.append(values[anchor, atom])
    sparse = SparseOrderForceConstants(
        2,
        index.n_primitive,
        len(reference),
        np.asarray(clusters),
        np.asarray(tensors),
        np.asarray(sites),
        np.asarray(translations)[:, None, :],
    )
    return ForceConstants(
        {},
        reference,
        metadata={"method": "sscha", "temperature": TEMPERATURE},
        sparse={2: sparse},
        relation=relation,
    )


def _plot(bands, output: Path) -> None:
    colors = {
        "phonopy harmonic": "#5b8f8a",
        "phonopy SSCHA": "#5b8f8a",
        "MLFCS Cartesian": "#c27b70",
        "MLFCS canonical": "#c27b70",
    }
    panels = (
        ("phonopy: harmonic vs SSCHA", ("phonopy harmonic", "phonopy SSCHA")),
        ("MLFCS: harmonic vs SSCHA", ("MLFCS Cartesian", "MLFCS canonical")),
        ("Harmonic: phonopy vs MLFCS", ("phonopy harmonic", "MLFCS Cartesian")),
        ("SSCHA: phonopy vs MLFCS", ("phonopy SSCHA", "MLFCS canonical")),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), sharey=True, constrained_layout=True)
    for axis, (title, selected) in zip(axes.flat, panels, strict=True):
        ticks = {}
        for name in selected:
            distances, frequencies, labels = bands[name]
            linestyle = "--" if "SSCHA" in name or "canonical" in name else "-"
            for distance, values in zip(distances, frequencies, strict=True):
                for branch in np.asarray(values).T:
                    axis.plot(
                        distance,
                        branch,
                        color=colors[name],
                        linewidth=1.0,
                        linestyle=linestyle,
                    )
            if not ticks:
                for (start, end), distance in zip(labels, distances, strict=True):
                    ticks.setdefault(float(distance[0]), start)
                    ticks[float(distance[-1])] = end
        for location in ticks:
            axis.axvline(location, color="#d7e0e0", linewidth=0.6, zorder=0)
        axis.axhline(0.0, color="#718080", linewidth=0.6, linestyle=":")
        axis.set_xticks(list(ticks), list(ticks.values()))
        axis.set_title(title, fontsize=11)
        axis.grid(axis="y", color="#edf1f1", linewidth=0.5)
        legend = []
        for name in selected:
            linestyle = "--" if "SSCHA" in name or "canonical" in name else "-"
            legend.append(
                Line2D([0], [0], color=colors[name], linewidth=1.6, linestyle=linestyle, label=name)
            )
        axis.legend(handles=legend, frameon=False, fontsize=8, loc="best")
    axes[0, 0].set_ylabel("Frequency (THz)")
    axes[1, 0].set_ylabel("Frequency (THz)")
    axes[1, 0].set_xlabel("Wave vector")
    axes[1, 1].set_xlabel("Wave vector")
    figure.suptitle("KCl harmonic and SSCHA phonon-band comparisons", fontsize=15)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_free_energy(data: dict[str, dict[str, list[float]]], output: Path) -> None:
    colors = {"phonopy": "#5b8f8a", "MLFCS": "#c27b70"}
    figure, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for name, values in data.items():
        x = np.asarray(values["iteration"])
        y = np.asarray(values["free_energy_eV_per_atom"])
        error = np.asarray(values["error_eV_per_atom"])
        axis.plot(x, y, color=colors[name], linewidth=1.4, marker="o", markersize=3.0, label=name)
        finite = np.isfinite(error)
        if np.any(finite):
            axis.fill_between(
                x[finite],
                y[finite] - error[finite],
                y[finite] + error[finite],
                color=colors[name],
                alpha=0.14,
                linewidth=0,
            )
    axis.set_xlabel("Canonical iteration")
    axis.set_ylabel("Free energy (eV/atom)")
    axis.set_title("KCl SSCHA free-energy convergence")
    axis.grid(color="#edf1f1", linewidth=0.5)
    axis.legend(frameon=False)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, default=OUTPUT / "kcl_sscha_bands.png")
    parser.add_argument(
        "--plot-existing",
        action="store_true",
        help="replot stored FC2 results without rerunning SSCHA",
    )
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    harmonic = load(HARMONIC)
    working_cell = _ase_from_phonopy(harmonic.unitcell)
    if args.plot_existing:
        phonopy_final = np.load(OUTPUT / "phonopy_sscha_final_fc2.npy")
        stored = np.load(OUTPUT / "mlfcs_sscha_fc2.npz")
        initial = stored["cartesian"]
        canonical = stored["canonical"]
        bands = {
            "phonopy harmonic": _bands(harmonic, harmonic.force_constants),
            "phonopy SSCHA": _bands(harmonic, phonopy_final),
            "MLFCS Cartesian": _bands(harmonic, _map_reference_to_phonopy(initial, harmonic)),
            "MLFCS canonical": _bands(harmonic, _map_reference_to_phonopy(canonical, harmonic)),
        }
        _plot(bands, args.output)
        print(f"wrote {args.output} from existing FC2 results")
        return
    reference = working_cell.repeat((2, 2, 2))
    calculator = PolymlpASECalculator(pot=POTENTIAL)
    phonopy_mlp = PhonopyMLP().load(POTENTIAL)
    phonopy_sscha = MLPSSCHA(
        harmonic,
        phonopy_mlp,
        temperature=TEMPERATURE,
        number_of_snapshots=args.snapshots,
        max_iterations=args.iterations,
        random_seed=42,
    )
    phonopy_history = {"iteration": [], "free_energy_eV_per_atom": [], "error_eV_per_atom": []}
    for iteration in phonopy_sscha:
        phonopy_sscha.calculate_free_energy()
        free_energy = float(phonopy_sscha.free_energy) / 2.0
        phonopy_history["iteration"].append(iteration)
        phonopy_history["free_energy_eV_per_atom"].append(free_energy)
        error = getattr(phonopy_sscha, "free_energy_error", float("nan"))
        error = float(error) / 2.0
        phonopy_history["error_eV_per_atom"].append(error)
        print(
            f"FREE_ENERGY software=phonopy iteration={iteration} eV_per_atom={free_energy:.12e} error_eV_per_atom={error:.12e}",
            flush=True,
        )
    phonopy_final = np.asarray(phonopy_sscha.force_constants)
    np.save(OUTPUT / "phonopy_sscha_final_fc2.npy", phonopy_final)
    write_FORCE_CONSTANTS(
        phonopy_final,
        filename=OUTPUT / "FORCE_CONSTANTS_PHONOPY_SSCHA",
    )
    sscha = SSCHA(
        working_cell,
        reference=reference,
        temperature=TEMPERATURE,
        snapshots=args.snapshots,
        max_iterations=args.iterations,
        random_seed=42,
        imaginary_modes="absolute",
    )
    sscha.run(calculator, calculate_free_energy=True)
    initial = sscha.history[0].force_constants
    canonical = sscha.history[-1].force_constants
    mlfcs_history = {"iteration": [], "free_energy_eV_per_atom": [], "error_eV_per_atom": []}
    for result in sscha.history:
        if result.free_energy is not None:
            free_energy = float(result.free_energy) / 8.0
            error = float(result.free_energy_error) / 8.0
            mlfcs_history["iteration"].append(result.index)
            mlfcs_history["free_energy_eV_per_atom"].append(free_energy)
            mlfcs_history["error_eV_per_atom"].append(error)
            print(
                f"FREE_ENERGY software=MLFCS iteration={result.index} eV_per_atom={free_energy:.12e} error_eV_per_atom={error:.12e}",
                flush=True,
            )
    free_energy = {"phonopy": phonopy_history, "MLFCS": mlfcs_history}
    (OUTPUT / "free_energy_convergence.json").write_text(
        json.dumps(free_energy, indent=2) + "\n", encoding="ascii"
    )
    _plot_free_energy(free_energy, OUTPUT / "free_energy_convergence.png")
    primitive = working_cell
    _mlfcs_result(initial, reference, primitive).write(OUTPUT / "mlfcs_cartesian.h5", format="hdf5")
    _mlfcs_result(canonical, reference, primitive).write(
        OUTPUT / "mlfcs_canonical.h5", format="hdf5"
    )
    _mlfcs_result(initial, reference, primitive).write(
        OUTPUT / "FORCE_CONSTANTS_MLFCS_CARTESIAN", format="phonopy", order=2
    )
    _mlfcs_result(canonical, reference, primitive).write(
        OUTPUT / "FORCE_CONSTANTS_MLFCS_CANONICAL", format="phonopy", order=2
    )
    np.savez_compressed(
        OUTPUT / "mlfcs_sscha_fc2.npz",
        cartesian=initial,
        canonical=canonical,
    )
    bands = {
        "phonopy harmonic": _bands(harmonic, harmonic.force_constants),
        "phonopy SSCHA": _bands(harmonic, phonopy_final),
        "MLFCS Cartesian": _bands(harmonic, _map_reference_to_phonopy(initial, harmonic)),
        "MLFCS canonical": _bands(harmonic, _map_reference_to_phonopy(canonical, harmonic)),
    }
    _plot(bands, args.output)
    (OUTPUT / "metadata.json").write_text(
        json.dumps(
            {
                "temperature_K": TEMPERATURE,
                "snapshots": args.snapshots,
                "iterations": args.iterations,
                "random_seed": 42,
                "phonopy_harmonic": HARMONIC.name,
                "phonopy_self_consistent": "generated by phonopy MLPSSCHA",
                "published_sscha_reference": SELF_CONSISTENT.name,
                "free_energy": {
                    "normalization": "eV/atom",
                    "phonopy_atoms_per_cell": 2,
                    "mlfcs_atoms_per_cell": 8,
                },
                "working_cell": "phonopy.unitcell (shared with MLFCS)",
                "supercell_atoms": len(reference),
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
