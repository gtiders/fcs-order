import numpy as np
import scipy as sp
import sparse
from rich.progress import track
import typer

from .symmetry import SymmetryOperations
from .gaussian import gaussian
from .wedge3 import Wedge


def reconstruct_ifcs(phipart, wedge, list4, poscar, sposcar, is_sparse):
    """
    Recover the full anharmonic IFC set from the irreducible set of
    force constants and the information contained in a wedge object.
    """
    nlist = wedge.nlist
    natoms = len(poscar["types"])
    ntot = len(sposcar["types"])
    if is_sparse:
        typer.echo("using sparse method with dok sparse matrix !")
        vnruter = sparse.zeros((3, 3, 3, natoms, ntot, ntot), format="dok")
    else:
        vnruter = np.zeros((3, 3, 3, natoms, ntot, ntot))
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

    typer.echo("- Storing the coefficients in a sparse matrix")
    i = []
    j = []
    v = []
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
                                        naccumindependent[ix], naccumindependent[ix + 1]
                                    ):
                                        tt = ss - naccumindependent[ix]
                                        i.append(ss)
                                        j.append(colindex)
                                        v.append(
                                            vtrans[
                                                tribasisindex, tt, vind2[ii, jj, kk], ix
                                            ]
                                        )
                        tribasisindex += 1
                        colindex += 1
    aa = sp.sparse.coo_matrix((v, (i, j)), (nrows, ncols)).tocsr()
    D = sp.sparse.spdiags(aphilist, [0], aphilist.size, aphilist.size, format="csr")
    bbs = D.dot(aa)
    ones = np.ones_like(aphilist)
    multiplier = -sp.sparse.linalg.lsqr(bbs, ones)[0]
    compensation = D.dot(bbs.dot(multiplier))

    aphilist += compensation

    if is_sparse:
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
                            if (
                                block[ll, mm, nn] != 0.0
                                and abs(block[ll, mm, nn]) > EPSVAL
                            ):
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
    else:
        vnruter = np.zeros((3, 3, 3, natoms, ntot, ntot), dtype=np.double)

    for ii in track(range(nlist), description="Building final IFCs"):
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
            if not is_sparse:
                for ll in range(3):
                    for mm in range(3):
                        for nn in range(3):
                            val = block[ll, mm, nn]
                            if val != 0.0:
                                vnruter[ll, mm, nn, e0, e1, e2] += val
    return vnruter
