#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import sys
import copy
import itertools

import numpy as np
import scipy.linalg
import scipy.spatial
import scipy.spatial.distance

H = 1e-3  # Magnitude of the finite displacements, in nm.
SYMPREC = 1e-5  # Tolerance for symmetry search


def gen_SPOSCAR(poscar, na, nb, nc):
    """
    Create a dictionary similar to the first argument but describing a
    supercell.
    """
    nruter = dict()
    nruter["na"] = na
    nruter["nb"] = nb
    nruter["nc"] = nc
    nruter["lattvec"] = np.array(poscar["lattvec"])
    nruter["lattvec"][:, 0] *= na
    nruter["lattvec"][:, 1] *= nb
    nruter["lattvec"][:, 2] *= nc
    nruter["elements"] = copy.copy(poscar["elements"])
    nruter["numbers"] = na * nb * nc * poscar["numbers"]
    nruter["positions"] = np.empty((3, poscar["positions"].shape[1] * na * nb * nc))
    pos = 0
    for pos, (k, j, i, iat) in enumerate(
        itertools.product(
            range(nc), range(nb), range(na), range(poscar["positions"].shape[1])
        )
    ):
        nruter["positions"][:, pos] = (poscar["positions"][:, iat] + [i, j, k]) / [
            na,
            nb,
            nc,
        ]
    nruter["types"] = []
    for i in range(na * nb * nc):
        nruter["types"].extend(poscar["types"])
    return nruter


def calc_dists(sposcar):
    """
    Return the distances between atoms in the supercells, their
    degeneracies and the associated supercell vectors.
    """
    ntot = sposcar["positions"].shape[1]
    posi = np.dot(sposcar["lattvec"], sposcar["positions"])
    d2s = np.empty((27, ntot, ntot))
    for j, (ja, jb, jc) in enumerate(
        itertools.product(range(-1, 2), range(-1, 2), range(-1, 2))
    ):
        posj = np.dot(sposcar["lattvec"], (sposcar["positions"].T + [ja, jb, jc]).T)
        d2s[j, :, :] = scipy.spatial.distance.cdist(posi.T, posj.T, "sqeuclidean")
    d2min = d2s.min(axis=0)
    dmin = np.sqrt(d2min)
    degenerate = np.abs(d2s - d2min) < 1e-4
    nequi = degenerate.sum(axis=0, dtype=np.intc)
    maxequi = nequi.max()
    shifts = np.empty((ntot, ntot, maxequi))
    sorting = np.argsort(np.logical_not(degenerate), axis=0)
    shifts = np.transpose(sorting[:maxequi, :, :], (1, 2, 0)).astype(np.intc)
    return (dmin, nequi, shifts)


def calc_frange(poscar, sposcar, n, dmin):
    """
    Return the maximum distance between n-th neighbors in the structure.
    """
    natoms = len(poscar["types"])
    tonth = []
    warned = False
    for i in range(natoms):
        ds = dmin[i, :].tolist()
        ds.sort()
        u = []
        for j in ds:
            for k in u:
                if np.allclose(k, j):
                    break
            else:
                u.append(j)
        try:
            tonth.append(0.5 * (u[n] + u[n + 1]))
        except IndexError:
            if not warned:
                sys.stderr.write(
                    "Warning: supercell too small to find n-th neighbours\n"
                )
                warned = True
            tonth.append(1.1 * max(u))
    return max(tonth)


def move_two_atoms(poscar, iat, icoord, ih, jat, jcoord, jh):
    """
    Return a copy of poscar with atom iat displaced by ih nm along
    its icoord-th Cartesian coordinate and atom jat displaced by
    jh nm along its jcoord-th Cartesian coordinate.
    """
    nruter = copy.deepcopy(poscar)
    disp = np.zeros(3)
    disp[icoord] = ih
    nruter["positions"][:, iat] += scipy.linalg.solve(nruter["lattvec"], disp)
    disp[:] = 0.0
    disp[jcoord] = jh
    nruter["positions"][:, jat] += scipy.linalg.solve(nruter["lattvec"], disp)
    return nruter


def write_ifcs(phifull, poscar, sposcar, dmin, nequi, shifts, frange, filename):
    """
    Write out the full anharmonic interatomic force constant matrix,
    taking the force cutoff into account.
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
    ffinal = open(filename, "w")
    ffinal.write("{:>5}\n".format(nblocks))
    ffinal.write(f.getvalue())
    f.close()
    ffinal.close()
