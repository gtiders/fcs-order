import numpy as np
import scipy as sp
import sparse
import typer
from rich.progress import track

from fcsorder.core.domain.symmetry import SymmetryOperations
from fcsorder.core.domain.wedge4 import Wedge
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
from numba.typed import List


@jit(nopython=True)
def _build_ifc4_coords_vals_list(
    nlist,
    nequi,
    nindependentbasis,
    vequilist,
    naccumindependent,
    transformationarray,
    aphilist,
):
    """Build coordinate/value lists for 4th-order IFCs in nopython mode.

    Returns 8 index lists (ll, mm, nn, aa1, e0, e1, e2, e3) and one value list.
    """

    ifc4_rows0 = List()
    ifc4_rows1 = List()
    ifc4_rows2 = List()
    ifc4_rows3 = List()
    ifc4_cols0 = List()
    ifc4_cols1 = List()
    ifc4_cols2 = List()
    ifc4_cols3 = List()
    ifc4_vals = List()

    for ii in range(nlist):
        ne = nequi[ii]
        nind = nindependentbasis[ii]

        for jj in range(ne):
            e0 = vequilist[0, jj, ii]
            e1 = vequilist[1, jj, ii]
            e2 = vequilist[2, jj, ii]
            e3 = vequilist[3, jj, ii]
            for ll in range(3):
                for mm in range(3):
                    for nn in range(3):
                        for aa1 in range(3):
                            tribasisindex = ((ll * 3 + mm) * 3 + nn) * 3 + aa1
                            for ix in range(nind):
                                val = (
                                    transformationarray[tribasisindex, ix, jj, ii]
                                    * aphilist[naccumindependent[ii] + ix]
                                )
                                if val == 0.0:
                                    continue
                                ifc4_rows0.append(ll)
                                ifc4_rows1.append(mm)
                                ifc4_rows2.append(nn)
                                ifc4_rows3.append(aa1)
                                ifc4_cols0.append(e0)
                                ifc4_cols1.append(e1)
                                ifc4_cols2.append(e2)
                                ifc4_cols3.append(e3)
                                ifc4_vals.append(val)

    return (
        ifc4_rows0,
        ifc4_rows1,
        ifc4_rows2,
        ifc4_rows3,
        ifc4_cols0,
        ifc4_cols1,
        ifc4_cols2,
        ifc4_cols3,
        ifc4_vals,
    )


@jit(nopython=True)
def _build_ijv_fourthorder(
    vind1,
    vind2,
    naccumindependent,
    natoms,
    ntot,
    nlist,
    vtrans,
):
    i = List()
    j = List()
    v = List()
    colindex = 0

    for ii in range(natoms):
        for jj in range(ntot):
            tribasisindex = 0
            for ll in range(3):
                for mm in range(3):
                    for nn in range(3):
                        for aa1 in range(3):
                            for kk in range(ntot):
                                for bb in range(ntot):
                                    for ix in range(nlist):
                                        if vind1[ii, jj, kk, bb] == ix:
                                            for ss in range(
                                                naccumindependent[ix],
                                                naccumindependent[ix + 1],
                                            ):
                                                tt = ss - naccumindependent[ix]
                                                i.append(ss)
                                                j.append(colindex)
                                                v.append(
                                                    vtrans[
                                                        tribasisindex,
                                                        tt,
                                                        vind2[ii, jj, kk, bb],
                                                        ix,
                                                    ]
                                                )
                            tribasisindex += 1
                            colindex += 1

    return i, j, v


def prepare_calculation4(na, nb, nc, cutoff, poscar_path: str = "POSCAR"):
    """
    Validate the input parameters and prepare the calculation.
    """
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
    typer.echo("Looking for an irreducible set of fourth-order IFCs")

    return poscar, sposcar, symops, dmin, nequi, shifts, frange, nneigh


def reconstruct_ifcs(phipart, wedge, list4, poscar, sposcar):
    """
    Recover the full fourth-order IFC set from the irreducible set of
    force constants and the information contained in a wedge object.
    """
    nlist = wedge.nlist
    natoms = len(poscar["types"])
    ntot = len(sposcar["types"])

    typer.echo("using sparse method with dok sparse matrix !")
    vnruter = sparse.zeros((3, 3, 3, 3, natoms, ntot, ntot, ntot), format="dok")

    naccumindependent = np.insert(
        np.cumsum(wedge.nindependentbasis[:nlist], dtype=np.intc),
        0,
        np.zeros(1, dtype=np.intc),
    )
    ntotalindependent = naccumindependent[-1]
    vphipart = phipart

    nlist6 = len(list4)
    for ii in track(range(nlist6), description="Processing list6"):
        e0, e1, e2, e3, e4, e5 = list4[ii]
        vnruter[e3, e4, e5, :, e0, e1, e2, :] = vphipart[:, ii, :]
    vnruter=vnruter.to_coo()

    philist = []
    for ii in track(range(nlist), description="Building philist"):
        for jj in range(wedge.nindependentbasis[ii]):
            kk = wedge.independentbasis[jj, ii] // 27
            ll = wedge.independentbasis[jj, ii] % 27 // 9
            mm = wedge.independentbasis[jj, ii] % 9 // 3
            nn = wedge.independentbasis[jj, ii] % 3
            philist.append(
                vnruter[
                    kk,
                    ll,
                    mm,
                    nn,
                    wedge.llist[0, ii],
                    wedge.llist[1, ii],
                    wedge.llist[2, ii],
                    wedge.llist[3, ii],
                ]
            )
    aphilist = np.array(philist, dtype=np.double)

    vind1 = -np.ones((natoms, ntot, ntot, ntot), dtype=np.intc)
    vind2 = -np.ones((natoms, ntot, ntot, ntot), dtype=np.intc)
    vequilist = wedge.allequilist
    for ii in range(nlist):
        for jj in range(wedge.nequi[ii]):
            vind1[
                vequilist[0, jj, ii],
                vequilist[1, jj, ii],
                vequilist[2, jj, ii],
                vequilist[3, jj, ii],
            ] = ii
            vind2[
                vequilist[0, jj, ii],
                vequilist[1, jj, ii],
                vequilist[2, jj, ii],
                vequilist[3, jj, ii],
            ] = jj

    vtrans = wedge.transformationarray

    nrows = ntotalindependent
    ncols = natoms * ntot * 81

    typer.echo("Storing the coefficients in a sparse matrix")
    i_list, j_list, v_list = _build_ijv_fourthorder(
        vind1,
        vind2,
        naccumindependent,
        natoms,
        ntot,
        nlist,
        vtrans,
    )
    i = np.array(i_list, dtype=np.intp)
    j = np.array(j_list, dtype=np.intp)
    v = np.array(v_list, dtype=np.float64)
    aaa = sp.sparse.coo_matrix((v, (i, j)), (nrows, ncols)).tocsr()
    D = sp.sparse.spdiags(aphilist, [0], aphilist.size, aphilist.size, format="csr")
    bbs = D.dot(aaa)
    ones = np.ones_like(aphilist)
    multiplier = -sp.sparse.linalg.lsqr(bbs, ones)[0]
    compensation = D.dot(bbs.dot(multiplier))
    aphilist += compensation

    typer.echo("Build the final, full set of 4th-order IFCs.")
    (
        ifc4_rows0_list,
        ifc4_rows1_list,
        ifc4_rows2_list,
        ifc4_rows3_list,
        ifc4_cols0_list,
        ifc4_cols1_list,
        ifc4_cols2_list,
        ifc4_cols3_list,
        ifc4_vals_list,
    ) = _build_ifc4_coords_vals_list(
        nlist,
        wedge.nequi,
        wedge.nindependentbasis,
        vequilist,
        naccumindependent,
        wedge.transformationarray,
        aphilist,
    )

    (
        ifc4_rows0,
        ifc4_rows1,
        ifc4_rows2,
        ifc4_rows3,
        ifc4_cols0,
        ifc4_cols1,
        ifc4_cols2,
        ifc4_cols3,
    ) = (
        np.array(ifc4_rows0_list, dtype=np.intp),
        np.array(ifc4_rows1_list, dtype=np.intp),
        np.array(ifc4_rows2_list, dtype=np.intp),
        np.array(ifc4_rows3_list, dtype=np.intp),
        np.array(ifc4_cols0_list, dtype=np.intp),
        np.array(ifc4_cols1_list, dtype=np.intp),
        np.array(ifc4_cols2_list, dtype=np.intp),
        np.array(ifc4_cols3_list, dtype=np.intp),
    )
    ifc4_vals = np.array(ifc4_vals_list, dtype=np.double)

    ifc4_coords = np.array(
        [
            ifc4_rows0,
            ifc4_rows1,
            ifc4_rows2,
            ifc4_rows3,
            ifc4_cols0,
            ifc4_cols1,
            ifc4_cols2,
            ifc4_cols3,
        ],
        dtype=np.intp,
    )
    vnruter = sparse.COO(
        ifc4_coords,
        ifc4_vals,
        shape=(3, 3, 3, 3, natoms, ntot, ntot, ntot),
        has_duplicates=True,
    )

    return vnruter
