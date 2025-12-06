#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Output writers for force constants and other data."""

import io
import itertools

import numpy as np


def write_ifcs3(
    phifull: np.ndarray,
    poscar: dict,
    sposcar: dict,
    dmin: np.ndarray,
    nequi: np.ndarray,
    shifts: np.ndarray,
    frange: float,
    filename: str,
) -> None:
    """
    Write out the full anharmonic interatomic force constant matrix,
    taking the force cutoff into account.

    Args:
        phifull: Full IFCS tensor.
        poscar: Primitive cell dictionary.
        sposcar: Supercell dictionary.
        dmin: Minimum distance matrix.
        nequi: Number of equivalent images matrix.
        shifts: Cell shifts matrix.
        frange: Force range cutoff.
        filename: Output filename.
    """
    natoms = len(poscar["types"])
    ntot = len(sposcar["types"])

    shifts27 = list(itertools.product(range(-1, 2), range(-1, 2), range(-1, 2)))
    frange2 = frange * frange

    nblocks = 0
    f = io.StringIO()
    for ii, jj in itertools.product(range(natoms), range(ntot)):
        if dmin[ii, jj] >= frange:
            continue
        jatom = jj % natoms
        shiftsij = [shifts27[i] for i in shifts[ii, jj, : nequi[ii, jj]]]
        for kk in range(ntot):
            if dmin[ii, kk] >= frange:
                continue
            katom = kk % natoms
            shiftsik = [shifts27[i] for i in shifts[ii, kk, : nequi[ii, kk]]]
            d2min = np.inf
            best2 = None
            best3 = None
            for shift2 in shiftsij:
                carj = np.dot(sposcar["lattvec"], shift2 + sposcar["positions"][:, jj])
                for shift3 in shiftsik:
                    cark = np.dot(
                        sposcar["lattvec"], shift3 + sposcar["positions"][:, kk]
                    )
                    d2 = ((carj - cark) ** 2).sum()
                    if d2 < d2min:
                        best2 = shift2
                        best3 = shift3
                        d2min = d2
            if d2min >= frange2:
                continue
            nblocks += 1
            Rj = np.dot(
                sposcar["lattvec"],
                best2 + sposcar["positions"][:, jj] - sposcar["positions"][:, jatom],
            )
            Rk = np.dot(
                sposcar["lattvec"],
                best3 + sposcar["positions"][:, kk] - sposcar["positions"][:, katom],
            )
            f.write("\n")
            f.write("{:>5}\n".format(nblocks))
            f.write(
                "{0[0]:>15.10e} {0[1]:>15.10e} {0[2]:>15.10e}\n".format(list(10.0 * Rj))
            )
            f.write(
                "{0[0]:>15.10e} {0[1]:>15.10e} {0[2]:>15.10e}\n".format(list(10.0 * Rk))
            )
            f.write("{:>6d} {:>6d} {:>6d}\n".format(ii + 1, jatom + 1, katom + 1))
            for ll, mm, nn in itertools.product(range(3), range(3), range(3)):
                f.write(
                    "{:>2d} {:>2d} {:>2d} {:>20.10e}\n".format(
                        ll + 1, mm + 1, nn + 1, phifull[ll, mm, nn, ii, jj, kk]
                    )
                )

    with open(filename, "w") as ffinal:
        ffinal.write("{:>5}\n".format(nblocks))
        ffinal.write(f.getvalue())
    f.close()


def write_ifcs4(
    phifull: np.ndarray,
    poscar: dict,
    sposcar: dict,
    dmin: np.ndarray,
    nequi: np.ndarray,
    shifts: np.ndarray,
    frange: float,
    filename: str,
) -> None:
    """
    Write out the full fourth-order interatomic force constant matrix,
    taking the force cutoff into account.

    Args:
        phifull: Full IFCS tensor.
        poscar: Primitive cell dictionary.
        sposcar: Supercell dictionary.
        dmin: Minimum distance matrix.
        nequi: Number of equivalent images matrix.
        shifts: Cell shifts matrix.
        frange: Force range cutoff.
        filename: Output filename.
    """
    natoms = len(poscar["types"])
    ntot = len(sposcar["types"])

    shifts27 = list(itertools.product(range(-1, 2), range(-1, 2), range(-1, 2)))
    frange2 = frange * frange

    nblocks = 0
    f = io.StringIO()
    for ii, jj in itertools.product(range(natoms), range(ntot)):
        if dmin[ii, jj] >= frange:
            continue
        jatom = jj % natoms
        shiftsij = [shifts27[i] for i in shifts[ii, jj, : nequi[ii, jj]]]
        for kk in range(ntot):
            if dmin[ii, kk] >= frange:
                continue
            katom = kk % natoms
            shiftsik = [shifts27[i] for i in shifts[ii, kk, : nequi[ii, kk]]]
            for ll in range(ntot):
                if dmin[ii, ll] >= frange:
                    continue
                latom = ll % natoms
                shiftsil = [shifts27[i] for i in shifts[ii, ll, : nequi[ii, ll]]]

                d2min = np.inf
                best2 = None
                best3 = None
                best4 = None

                for shift2 in shiftsij:
                    carj = np.dot(
                        sposcar["lattvec"], shift2 + sposcar["positions"][:, jj]
                    )
                    for shift3 in shiftsik:
                        cark = np.dot(
                            sposcar["lattvec"], shift3 + sposcar["positions"][:, kk]
                        )
                        for shift4 in shiftsil:
                            carl = np.dot(
                                sposcar["lattvec"], shift4 + sposcar["positions"][:, ll]
                            )
                            d2_1 = ((carj - cark) ** 2).sum()
                            d2_2 = ((carj - carl) ** 2).sum()
                            d2_3 = ((cark - carl) ** 2).sum()
                            d2 = max(d2_1, d2_2, d2_3)
                            if d2 < d2min:
                                best2 = shift2
                                best3 = shift3
                                best4 = shift4
                                d2min = d2
                if d2min >= frange2:
                    continue
                nblocks += 1
                Rj = np.dot(
                    sposcar["lattvec"],
                    best2
                    + sposcar["positions"][:, jj]
                    - sposcar["positions"][:, jatom],
                )
                Rk = np.dot(
                    sposcar["lattvec"],
                    best3
                    + sposcar["positions"][:, kk]
                    - sposcar["positions"][:, katom],
                )
                Rl = np.dot(
                    sposcar["lattvec"],
                    best4
                    + sposcar["positions"][:, ll]
                    - sposcar["positions"][:, latom],
                )
                f.write("\n")
                f.write("{:>5}\n".format(nblocks))
                f.write(
                    "{0[0]:>15.10e} {0[1]:>15.10e} {0[2]:>15.10e}\n".format(
                        list(10.0 * Rj)
                    )
                )
                f.write(
                    "{0[0]:>15.10e} {0[1]:>15.10e} {0[2]:>15.10e}\n".format(
                        list(10.0 * Rk)
                    )
                )
                f.write(
                    "{0[0]:>15.10e} {0[1]:>15.10e} {0[2]:>15.10e}\n".format(
                        list(10.0 * Rl)
                    )
                )
                f.write(
                    "{:>6d} {:>6d} {:>6d} {:>6d}\n".format(
                        ii + 1, jatom + 1, katom + 1, latom + 1
                    )
                )
                for mm, nn, oo, pp in itertools.product(
                    range(3), range(3), range(3), range(3)
                ):
                    f.write(
                        "{:>2d} {:>2d} {:>2d} {:>2d} {:>20.10f}\n".format(
                            mm + 1,
                            nn + 1,
                            oo + 1,
                            pp + 1,
                            phifull[mm, nn, oo, pp, ii, jj, kk, ll],
                        )
                    )

    with open(filename, "w") as ffinal:
        ffinal.write("{:>5}\n".format(nblocks))
        ffinal.write(f.getvalue())
    f.close()
