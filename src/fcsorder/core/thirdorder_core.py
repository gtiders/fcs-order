import numpy as np
import scipy as sp
import sparse
from rich.progress import track
import typer

from fcsorder.core.domain.symmetry import SymmetryOperations
from fcsorder.core.domain.wedge3 import Wedge
from fcsorder.core.domain.gaussian import gaussian


from fcsorder.io.io_abstraction import read_structure
from fcsorder.core.domain.common import (
    SYMPREC,
    _parse_cutoff,
    _validate_cutoff,
    calc_dists,
    calc_frange,
    gen_SPOSCAR,
)

from numba import jit


@jit(nopython=True)
def _build_ijv_thirdorder(
    vind1,
    vind2,
    naccumindependent,
    natoms,
    ntot,
    vtrans,
):
    nnz = 0
    for ii in range(natoms):
        for jj in range(ntot):
            tribasisindex = 0
            for ll in range(3):
                for mm in range(3):
                    for nn in range(3):
                        for kk in range(ntot):
                            ix = vind1[ii, jj, kk]
                            if ix != -1:
                                start = naccumindependent[ix]
                                end = naccumindependent[ix + 1]
                                nnz += end - start
                        tribasisindex += 1

    i = np.empty(nnz, dtype=np.intp)
    j = np.empty(nnz, dtype=np.intp)
    v = np.empty(nnz, dtype=np.float64)

    p = 0
    colindex = 0
    for ii in range(natoms):
        for jj in range(ntot):
            tribasisindex = 0
            for ll in range(3):
                for mm in range(3):
                    for nn in range(3):
                        for kk in range(ntot):
                            ix = vind1[ii, jj, kk]
                            if ix != -1:
                                start = naccumindependent[ix]
                                end = naccumindependent[ix + 1]
                                row = vind2[ii, jj, kk]
                                for ss in range(start, end):
                                    tt = ss - start
                                    i[p] = ss
                                    j[p] = colindex
                                    v[p] = vtrans[tribasisindex, tt, row, ix]
                                    p += 1
                        tribasisindex += 1
                        colindex += 1

    return i, j, v


def prepare_calculation3(na, nb, nc, cutoff, poscar_path: str = "POSCAR"):
    _validate_cutoff(na, nb, nc)
    nneigh, frange = _parse_cutoff(cutoff)
    typer.echo("Reading structure")
    poscar = read_structure(poscar_path, in_format="auto")
    typer.echo("Analyzing the symmetries")
    symops = SymmetryOperations(
        poscar["lattvec"], poscar["types"], poscar["positions"].T, SYMPREC
    )
    typer.echo(f"Symmetry group {symops.symbol} detected")
    typer.echo(f"{symops.translations.shape[0]} symmetry operations")
    typer.echo("Creating the supercell")
    sposcar = gen_SPOSCAR(poscar, na, nb, nc)
    typer.echo("Computing all distances in the supercell")
    dmin, nequi, shifts = calc_dists(sposcar)
    if nneigh is not None:
        frange = calc_frange(poscar, sposcar, nneigh, dmin)
        typer.echo(f"Automatic cutoff: {frange} nm")
    else:
        typer.echo(f"User-defined cutoff: {frange} nm")
    typer.echo("Looking for an irreducible set of third-order IFCs")

    return poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh


def reconstruct_ifcs(phipart, wedge, list4, poscar, sposcar):
    """
    Recover the full anharmonic IFC set from the irreducible set of
    force constants and the information contained in a wedge object.
    """
    nlist = wedge.nlist
    natoms = len(poscar["types"])
    ntot = len(sposcar["types"])

    typer.echo("using sparse method with dok sparse matrix !")
    vnruter = sparse.zeros((3, 3, 3, natoms, ntot, ntot), format="dok")

    naccumindependent = np.insert(
        np.cumsum(wedge.nindependentbasis[:nlist], dtype=np.intc),
        0,
        np.zeros(1, dtype=np.intc),
    )
    ntotalindependent = naccumindependent[-1]
    vphipart = phipart
    nlist4 = len(list4)
    for ii in track(range(nlist4), description="Processing list4"):
        e0, e1, e2, e3 = list4[ii]
        vnruter[e2, e3, :, e0, e1, :] = vphipart[:, ii, :]
    philist = []
    for ii in track(range(nlist), description="Building philist"):
        for jj in range(wedge.nindependentbasis[ii]):
            ll = wedge.independentbasis[jj, ii] // 9
            mm = (wedge.independentbasis[jj, ii] % 9) // 3
            nn = wedge.independentbasis[jj, ii] % 3
            philist.append(
                vnruter[
                    ll,
                    mm,
                    nn,
                    wedge.llist[0, ii],
                    wedge.llist[1, ii],
                    wedge.llist[2, ii],
                ]
            )
    aphilist = np.array(philist, dtype=np.double)
    vind1 = -np.ones((natoms, ntot, ntot), dtype=np.intc)
    vind2 = -np.ones((natoms, ntot, ntot), dtype=np.intc)
    vequilist = wedge.allequilist
    for ii in range(nlist):
        for jj in range(wedge.nequi[ii]):
            vind1[
                vequilist[0, jj, ii],
                vequilist[1, jj, ii],
                vequilist[2, jj, ii],
            ] = ii
            vind2[
                vequilist[0, jj, ii],
                vequilist[1, jj, ii],
                vequilist[2, jj, ii],
            ] = jj

    vtrans = wedge.transformationarray

    nrows = ntotalindependent
    ncols = natoms * ntot * 27

    typer.echo("Storing the coefficients in a sparse matrix")
    i, j, v = _build_ijv_thirdorder(
        vind1,
        vind2,
        naccumindependent,
        natoms,
        ntot,
        vtrans,
    )
    aa = sp.sparse.coo_matrix((v, (i, j)), (nrows, ncols)).tocsr()
    D = sp.sparse.spdiags(aphilist, [0], aphilist.size, aphilist.size, format="csr")
    bbs = D.dot(aa)
    ones = np.ones_like(aphilist)
    multiplier = -sp.sparse.linalg.lsqr(bbs, ones)[0]
    compensation = D.dot(bbs.dot(multiplier))

    aphilist += compensation

    nnz = 0
    EPSVAL = 1e-10
    for ii in range(nlist):
        nind = wedge.nindependentbasis[ii]
        ne = wedge.nequi[ii]
        if nind == 0 or ne == 0:
            continue
        offset = naccumindependent[ii]
        phi = aphilist[offset : offset + nind]
        T = wedge.transformationarray[:, :nind, :ne, ii]
        out = np.tensordot(T, phi, axes=([1], [0]))
        for jj in range(ne):
            block = out[:, jj].reshape(3, 3, 3)
            for ll in range(3):
                for mm in range(3):
                    for nn in range(3):
                        if block[ll, mm, nn] != 0.0 and abs(block[ll, mm, nn]) > EPSVAL:
                            nnz += 1
    coords = np.empty((6, nnz), dtype=np.intp)
    data = np.empty(nnz, dtype=np.double)
    p = 0
    for ii in range(nlist):
        nind = wedge.nindependentbasis[ii]
        ne = wedge.nequi[ii]
        if nind == 0 or ne == 0:
            continue
        offset = naccumindependent[ii]
        phi = aphilist[offset : offset + nind]
        T = wedge.transformationarray[:, :nind, :ne, ii]
        out = np.tensordot(T, phi, axes=([1], [0]))
        for jj in range(ne):
            e0 = vequilist[0, jj, ii]
            e1 = vequilist[1, jj, ii]
            e2 = vequilist[2, jj, ii]
            block = out[:, jj].reshape(3, 3, 3)
            for ll in range(3):
                for mm in range(3):
                    for nn in range(3):
                        val = block[ll, mm, nn]
                        if val != 0.0 and abs(val) > EPSVAL:
                            coords[0, p] = ll
                            coords[1, p] = mm
                            coords[2, p] = nn
                            coords[3, p] = e0
                            coords[4, p] = e1
                            coords[5, p] = e2
                            data[p] = val
                            p += 1
    vnruter = sparse.COO(coords, data, shape=(3, 3, 3, natoms, ntot, ntot))

    return vnruter
