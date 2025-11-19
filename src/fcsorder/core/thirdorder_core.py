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
from numba.typed import List


@jit(nopython=True)
def _build_ijv_thirdorder(
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
                        for kk in range(ntot):
                            for ix in range(nlist):
                                if vind1[ii, jj, kk] == ix:
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
                                                vind2[ii, jj, kk],
                                                ix,
                                            ]
                                        )
                        tribasisindex += 1
                        colindex += 1

    return i, j, v


@jit(nopython=True)
def _build_ifc3_coords_vals_list(
    nlist,
    nequi,
    nindependentbasis,
    vequilist,
    naccumindependent,
    transformationarray,
    aphilist,
):
    rows = List()
    cols0 = List()
    cols1 = List()
    cols2 = List()
    cols3 = List()
    cols4 = List()
    vals = List()

    for ii in range(nlist):
        ne = nequi[ii]
        nind = nindependentbasis[ii]

        for jj in range(ne):
            e0 = vequilist[0, jj, ii]
            e1 = vequilist[1, jj, ii]
            e2 = vequilist[2, jj, ii]
            for ll in range(3):
                for mm in range(3):
                    for nn in range(3):
                        tribasisindex = (ll * 3 + mm) * 3 + nn
                        for ix in range(nind):
                            val = (
                                transformationarray[tribasisindex, ix, jj, ii]
                                * aphilist[naccumindependent[ii] + ix]
                            )
                            if val == 0.0:
                                continue
                            rows.append(ll)
                            cols0.append(mm)
                            cols1.append(nn)
                            cols2.append(e0)
                            cols3.append(e1)
                            cols4.append(e2)
                            vals.append(val)

    return rows, cols0, cols1, cols2, cols3, cols4, vals


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
    vnruter = vnruter.to_coo()

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
    i_list, j_list, v_list = _build_ijv_thirdorder(
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
    aa = sp.sparse.coo_matrix((v, (i, j)), (nrows, ncols)).tocsr()
    D = sp.sparse.spdiags(aphilist, [0], aphilist.size, aphilist.size, format="csr")
    bbs = D.dot(aa)
    ones = np.ones_like(aphilist)
    multiplier = -sp.sparse.linalg.lsqr(bbs, ones)[0]
    compensation = D.dot(bbs.dot(multiplier))

    aphilist += compensation

    typer.echo("Build the final, full set of anharmonic IFCs.")
    (
        ifc3_rows_list,
        ifc3_cols0_list,
        ifc3_cols1_list,
        ifc3_cols2_list,
        ifc3_cols3_list,
        ifc3_cols4_list,
        ifc3_vals_list,
    ) = _build_ifc3_coords_vals_list(
        nlist,
        wedge.nequi,
        wedge.nindependentbasis,
        vequilist,
        naccumindependent,
        wedge.transformationarray,
        aphilist,
    )

    (
        ifc3_rows,
        ifc3_cols0,
        ifc3_cols1,
        ifc3_cols2,
        ifc3_cols3,
        ifc3_cols4,
    ) = (
        np.array(ifc3_rows_list, dtype=np.intp),
        np.array(ifc3_cols0_list, dtype=np.intp),
        np.array(ifc3_cols1_list, dtype=np.intp),
        np.array(ifc3_cols2_list, dtype=np.intp),
        np.array(ifc3_cols3_list, dtype=np.intp),
        np.array(ifc3_cols4_list, dtype=np.intp),
    )
    ifc3_vals = np.array(ifc3_vals_list, dtype=np.double)

    ifc3_coords = np.array(
        [ifc3_rows, ifc3_cols0, ifc3_cols1, ifc3_cols2, ifc3_cols3, ifc3_cols4],
        dtype=np.intp,
    )
    vnruter = sparse.COO(
        ifc3_coords, ifc3_vals, shape=(3, 3, 3, natoms, ntot, ntot), has_duplicates=True
    )
    return vnruter
