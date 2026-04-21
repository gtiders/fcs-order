"""Output writers for HIFINIT force constants."""

from __future__ import annotations

import os
from collections import namedtuple
from itertools import product as iter_product

import numpy as np
from ase import Atoms
from hiphive import ForceConstantPotential
from hiphive.core.structures import Supercell
from hiphive.force_constants import SortedForceConstants


def write_shengBTE_fc4(
    filename: str,
    fcs: SortedForceConstants,
    prim: Atoms,
    symprec: float = 1e-5,
    cutoff: float = np.inf,
    fc_tol: float = 1e-8,
    optim_write: bool = True,
) -> None:
    """Write fourth-order force constants in ShengBTE format."""
    _ShengEntry4 = namedtuple(
        "Entry",
        [
            "site_0",
            "site_1",
            "site_2",
            "site_3",
            "pos_1",
            "pos_2",
            "pos_3",
            "fc",
            "offset_1",
            "offset_2",
            "offset_3",
        ],
    )

    def _fancy_to_raw4(sheng):
        raw_sheng = []
        for entry in sheng:
            raw_entry = list(entry[:8])
            raw_entry[0] += 1
            raw_entry[1] += 1
            raw_entry[2] += 1
            raw_entry[3] += 1
            raw_sheng.append(raw_entry)
        return raw_sheng

    def _write_raw_sheng4(raw_sheng, out_path):
        with open(out_path, "w") as f:
            f.write("{}\n\n".format(len(raw_sheng)))

            for index, fc4_row in enumerate(raw_sheng, start=1):
                i, j, k, l, cell_pos2, cell_pos3, cell_pos4, fc4_ijkl = fc4_row  # noqa: E741

                f.write("{:5d}\n".format(index))
                f.write((3 * "{:14.10f} " + "\n").format(*cell_pos2))
                f.write((3 * "{:14.10f} " + "\n").format(*cell_pos3))
                f.write((3 * "{:14.10f} " + "\n").format(*cell_pos4))
                f.write((4 * "{:5d}" + "\n").format(i, j, k, l))
                for w, x, y, z in iter_product(range(3), repeat=4):
                    f.write((4 * " {:}").format(w + 1, x + 1, y + 1, z + 1))
                    f.write("    {:14.10f}\n".format(fc4_ijkl[w, x, y, z]))
                f.write("\n")

    if optim_write:

        def _fcs_to_sheng4(fcs_obj, prim_obj, symprec_val, cutoff_val, fc_tol_val):
            supercell = Supercell(fcs_obj.supercell, prim_obj, symprec_val)
            assert all(fcs_obj.supercell.pbc) and all(prim_obj.pbc)

            n_atoms = len(supercell)
            d = fcs_obj.supercell.get_all_distances(mic=False, vector=True)
            d_mic = fcs_obj.supercell.get_all_distances(mic=True, vector=True)
            m = np.eye(n_atoms, dtype=bool)
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    m[i, j] = (
                        np.allclose(d[i, j], d_mic[i, j], atol=symprec_val, rtol=0)
                        and np.linalg.norm(d[i, j]) < cutoff_val
                    )
                    m[j, i] = m[i, j]

            data = {}
            seen_sites = set()

            for a0 in supercell:
                if a0.site in seen_sites:
                    continue
                seen_sites.add(a0.site)

                for a1 in supercell:
                    if not m[a0.index, a1.index]:
                        continue
                    for a2 in supercell:
                        if not (m[a0.index, a2.index] and m[a1.index, a2.index]):
                            continue
                        for a3 in supercell:
                            if not (
                                m[a0.index, a3.index]
                                and m[a1.index, a3.index]
                                and m[a2.index, a3.index]
                            ):
                                continue

                            ijkl = (a0.index, a1.index, a2.index, a3.index)
                            fc = fcs_obj[ijkl]
                            if np.max(np.abs(fc)) < fc_tol_val:
                                continue

                            offset_1 = np.subtract(a1.offset, a0.offset)
                            offset_2 = np.subtract(a2.offset, a0.offset)
                            offset_3 = np.subtract(a3.offset, a0.offset)
                            sites = (a0.site, a1.site, a2.site, a3.site)
                            key = (
                                sites
                                + tuple(offset_1)
                                + tuple(offset_2)
                                + tuple(offset_3)
                            )

                            if key in data:
                                assert np.allclose(data[key], fc, atol=fc_tol_val)
                            else:
                                data[key] = fc

            sheng = []
            for k, fc in data.items():
                offset_1 = k[4:7]
                pos_1 = np.dot(offset_1, prim_obj.cell)
                offset_2 = k[7:10]
                pos_2 = np.dot(offset_2, prim_obj.cell)
                offset_3 = k[10:13]
                pos_3 = np.dot(offset_3, prim_obj.cell)
                sheng.append(
                    _ShengEntry4(
                        *k[:4], pos_1, pos_2, pos_3, fc, offset_1, offset_2, offset_3
                    )
                )
            return sheng
    else:

        def _fcs_to_sheng4(fcs_obj, prim_obj, symprec_val, cutoff_val, fc_tol_val):
            supercell = Supercell(fcs_obj.supercell, prim_obj, symprec_val)
            assert all(fcs_obj.supercell.pbc) and all(prim_obj.pbc)

            n_atoms = len(supercell)
            d = fcs_obj.supercell.get_all_distances(mic=False, vector=True)
            d_mic = fcs_obj.supercell.get_all_distances(mic=True, vector=True)
            m = np.eye(n_atoms, dtype=bool)
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    m[i, j] = (
                        np.allclose(d[i, j], d_mic[i, j], atol=symprec_val, rtol=0)
                        and np.linalg.norm(d[i, j]) < cutoff_val
                    )
                    m[j, i] = m[i, j]

            data = {}
            for a0 in supercell:
                for a1 in supercell:
                    if not m[a0.index, a1.index]:
                        continue
                    for a2 in supercell:
                        if not (m[a0.index, a2.index] and m[a1.index, a2.index]):
                            continue
                        for a3 in supercell:
                            if not (
                                m[a0.index, a3.index]
                                and m[a1.index, a3.index]
                                and m[a2.index, a3.index]
                            ):
                                continue
                            offset_1 = np.subtract(a1.offset, a0.offset)
                            offset_2 = np.subtract(a2.offset, a0.offset)
                            offset_3 = np.subtract(a3.offset, a0.offset)
                            sites = (a0.site, a1.site, a2.site, a3.site)
                            key = (
                                sites
                                + tuple(offset_1)
                                + tuple(offset_2)
                                + tuple(offset_3)
                            )
                            ijkl = (a0.index, a1.index, a2.index, a3.index)
                            fc = fcs_obj[ijkl]
                            if key in data:
                                assert np.allclose(data[key], fc, atol=fc_tol_val)
                            else:
                                data[key] = fc

            sheng = []
            for k, fc in data.items():
                if np.max(np.abs(fc)) < fc_tol_val:
                    continue
                offset_1 = k[4:7]
                pos_1 = np.dot(offset_1, prim_obj.cell)
                offset_2 = k[7:10]
                pos_2 = np.dot(offset_2, prim_obj.cell)
                offset_3 = k[10:13]
                pos_3 = np.dot(offset_3, prim_obj.cell)
                sheng.append(
                    _ShengEntry4(
                        *k[:4], pos_1, pos_2, pos_3, fc, offset_1, offset_2, offset_3
                    )
                )
            return sheng

    sheng = _fcs_to_sheng4(fcs, prim, symprec, cutoff, fc_tol)
    raw_sheng = _fancy_to_raw4(sheng)
    _write_raw_sheng4(raw_sheng, filename)


def save_force_constants(
    prim: Atoms,
    fcp: ForceConstantPotential,
    fcs_final: SortedForceConstants,
    max_order: int,
    out_dir: str = "./hifinit_results",
    verbose: bool = True,
) -> None:
    """Save force constants to files in multiple formats."""
    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print("\n" + "=" * 60)
        print("Saving Force Constants")
        print("=" * 60)

    fcp.write(os.path.join(out_dir, "potential.fcp"))
    if verbose:
        print(f"ForceConstantPotential: {os.path.join(out_dir, 'potential.fcp')}")

    fcs_final.write_to_phonopy(
        os.path.join(out_dir, "FORCE_CONSTANTS_2ND"), format="text"
    )
    fcs_final.write_to_phonopy(os.path.join(out_dir, "fc2.hdf5"), format="hdf5")
    if verbose:
        print(
            f"2nd order FCs: {os.path.join(out_dir, 'fc2.hdf5')} and "
            f"{os.path.join(out_dir, 'FORCE_CONSTANTS_2ND')}"
        )

    if max_order >= 3:
        fcs_final.write_to_shengBTE(os.path.join(out_dir, "FORCE_CONSTANTS_3RD"), prim)
        fcs_final.write_to_phono3py(os.path.join(out_dir, "fc3.hdf5"))
        if verbose:
            print(
                f"3rd order FCs: {os.path.join(out_dir, 'fc3.hdf5')} and "
                f"{os.path.join(out_dir, 'FORCE_CONSTANTS_3RD')}"
            )

        if max_order >= 4:
            write_shengBTE_fc4(
                os.path.join(out_dir, "FORCE_CONSTANTS_4TH"), fcs_final, prim
            )
            if verbose:
                print(f"4th order FCs: {os.path.join(out_dir, 'FORCE_CONSTANTS_4TH')}")

    if verbose:
        print("\n" + "=" * 60)
        print("All force constants saved successfully!")
        print("=" * 60)
