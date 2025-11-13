import numpy as np
import scipy as sp
import sparse
import typer
from rich.progress import track
from math import fabs

from numba import jit

# Permutations of 4 elements listed in the same order as in the old Fortran code.
permutations = np.array(
    [
        [0, 1, 2, 3],
        [0, 2, 1, 3],
        [0, 1, 3, 2],
        [0, 3, 1, 2],
        [0, 3, 2, 1],
        [0, 2, 3, 1],
        [1, 0, 2, 3],
        [1, 0, 3, 2],
        [1, 2, 0, 3],
        [1, 2, 3, 0],
        [1, 3, 0, 2],
        [1, 3, 2, 0],
        [2, 0, 1, 3],
        [2, 0, 3, 1],
        [2, 1, 0, 3],
        [2, 1, 3, 0],
        [2, 3, 0, 1],
        [2, 3, 1, 0],
        [3, 0, 1, 2],
        [3, 0, 2, 1],
        [3, 1, 0, 2],
        [3, 1, 2, 0],
        [3, 2, 0, 1],
        [3, 2, 1, 0],
    ],
    dtype=np.intc,
)

from fcsorder.core.domain.symmetry import SymmetryOperations
from fcsorder.core.domain.gaussian import gaussian


@jit(nopython=True)
def _ind2id(icell, ispecies, ngrid, nspecies):
    """
    Merge a set of cell+atom indices into a single index into a supercell.
    icell: shape (3,)
    ngrid: shape (3,)
    """
    return (
        icell[0] + (icell[1] + icell[2] * ngrid[1]) * ngrid[0]
    ) * nspecies + ispecies


@jit(nopython=True)
def _quartet_in_list(quartet, llist, nlist):
    """
    Return True if quartet is found in llist[:,:nlist]. The first dimension of list must have a length of 4.
    quartet: shape (4,)
    llist: shape (4, N)
    """
    for i in range(nlist):
        if (
            quartet[0] == llist[0, i]
            and quartet[1] == llist[1, i]
            and quartet[2] == llist[2, i]
            and quartet[3] == llist[3, i]
        ):
            return True
    return False


@jit(nopython=True)
def _quartets_are_equal(quartet1, quartet2):
    """
    Return True if two quartets are equal and False otherwise.
    """
    for i in range(4):
        if quartet1[i] != quartet2[i]:
            return False
    return True


@jit(nopython=True)
def _id2ind(ngrid, nspecies):
    """
    Create a map from supercell indices to cell+atom indices.
    Returns (icell, ispecies) as numpy arrays with shapes (3,ntot) and (ntot,)
    """
    ntot = ngrid[0] * ngrid[1] * ngrid[2] * nspecies
    icell = np.empty((3, ntot), dtype=np.intc)
    ispecies = np.empty(ntot, dtype=np.intc)
    for ii in range(ntot):
        tmp, ispecies[ii] = divmod(ii, nspecies)
        tmp, icell[0, ii] = divmod(tmp, ngrid[0])
        icell[2, ii], icell[1, ii] = divmod(tmp, ngrid[1])
    return icell, ispecies


@jit(nopython=True)
def _build_transformationarray(
    v_transformation,
    v_transformationaux,
    v_nequi,
    v_nindependentbasis,
    nlist,
    out_array,
):
    """
    JIT kernel to compute:
        out_array[kk, ll, jj, ii] = sum_{iaux=0..80} v_transformation[kk, iaux, jj, ii]
                                              * v_transformationaux[iaux, ll, ii]
    Only columns ll < v_nindependentbasis[ii] are filled; others remain as initialized.
    Applies small-value thresholding (1e-15) inline.
    Shapes:
      v_transformation: (81,81,nequi_i,nlist)
      v_transformationaux: (81,81,nlist)
      out_array: (81,81,nequi_i,nlist)
    """
    for ii in range(nlist):
        ne = v_nequi[ii]
        nind = v_nindependentbasis[ii]
        for jj in range(ne):
            for kk in range(81):
                for ll in range(nind):
                    s = 0.0
                    for iaux in range(81):
                        s += (
                            v_transformation[kk, iaux, jj, ii]
                            * v_transformationaux[iaux, ll, ii]
                        )
                    if s != 0.0 and abs(s) < 1e-15:
                        s = 0.0
                    out_array[kk, ll, jj, ii] = s


class Wedge:
    """
    Objects of this class allow the user to extract irreducible sets
    of force constants and to reconstruct the full Fourth-order IFC
    matrix from them.
    """

    # Public attributes to match Cython class interface
    symops: object
    poscar: dict
    sposcar: dict
    allocsize: int
    allallocsize: int
    nalllist: int
    nlist: int
    nequi: np.ndarray
    llist: np.ndarray
    allequilist: np.ndarray
    nindependentbasis: np.ndarray
    independentbasis: np.ndarray
    transformationarray: np.ndarray

    # Internal buffers
    alllist: np.ndarray
    transformation: np.ndarray
    transformationaux: np.ndarray

    # Extra fields akin to memoryviews
    nequis: np.ndarray
    shifts: np.ndarray
    dmin: np.ndarray
    frange: float

    def __init__(self, poscar, sposcar, symops, dmin, nequis, shifts, frange):
        self.poscar = poscar
        self.sposcar = sposcar
        self.symops = symops
        self.dmin = dmin
        self.nequis = nequis
        self.shifts = shifts
        self.frange = frange

        self.allocsize = 0
        self.allallocsize = 0
        self._expandlist()
        self._expandalllist()

        self._reduce()

    def _expandlist(self):
        """
        Expand nequi, allequilist, transformationarray, transformation,
        transformationaux, nindependentbasis, independentbasis, and llist.
        """
        if self.allocsize == 0:
            self.allocsize = 32
            self.nequi = np.empty(self.allocsize, dtype=np.intc)
            self.allequilist = np.empty(
                (4, 24 * self.symops.nsyms, self.allocsize), dtype=np.intc
            )
            self.transformationarray = np.empty(
                (81, 81, 24 * self.symops.nsyms, self.allocsize), dtype=np.double
            )
            self.transformation = np.empty(
                (81, 81, 24 * self.symops.nsyms, self.allocsize), dtype=np.double
            )
            self.transformationaux = np.empty((81, 81, self.allocsize), dtype=np.double)
            self.nindependentbasis = np.empty(self.allocsize, dtype=np.intc)
            self.independentbasis = np.empty((81, self.allocsize), dtype=np.intc)
            self.llist = np.empty((4, self.allocsize), dtype=np.intc)
        else:
            self.allocsize <<= 1
            self.nequi = np.concatenate((self.nequi, self.nequi), axis=-1)
            self.allequilist = np.concatenate(
                (self.allequilist, self.allequilist), axis=-1
            )
            self.transformation = np.concatenate(
                (self.transformation, self.transformation), axis=-1
            )
            self.transformationarray = np.concatenate(
                (self.transformationarray, self.transformationarray), axis=-1
            )
            self.transformationaux = np.concatenate(
                (self.transformationaux, self.transformationaux), axis=-1
            )
            self.nindependentbasis = np.concatenate(
                (self.nindependentbasis, self.nindependentbasis), axis=-1
            )
            self.independentbasis = np.concatenate(
                (self.independentbasis, self.independentbasis), axis=-1
            )
            self.llist = np.concatenate((self.llist, self.llist), axis=-1)

    def _expandalllist(self):
        """Expand alllist to accommodate more elements."""
        if self.allallocsize == 0:
            self.allallocsize = 512
            self.alllist = np.empty((4, self.allallocsize), dtype=np.intc)
        else:
            self.allallocsize <<= 1
            self.alllist = np.concatenate((self.alllist, self.alllist), axis=-1)

    def _reduce(self):
        frange2 = self.frange * self.frange

        ngrid1 = self.sposcar["na"]
        ngrid2 = self.sposcar["nb"]
        ngrid3 = self.sposcar["nc"]
        ngrid = np.array([ngrid1, ngrid2, ngrid3], dtype=np.intc)
        nsym = self.symops.nsyms
        natoms = len(self.poscar["types"])
        ntot = len(self.sposcar["types"])

        lattvec = self.sposcar["lattvec"]
        coordall = np.dot(lattvec, self.sposcar["positions"])
        orth = np.transpose(self.symops.crotations, (1, 2, 0))

        self.nlist = 0
        self.nalllist = 0
        v_nequi = self.nequi
        v_allequilist = self.allequilist
        v_transformation = self.transformation
        v_transformationarray = self.transformationarray
        v_transformationaux = self.transformationaux
        v_nindependentbasis = self.nindependentbasis
        v_independentbasis = self.independentbasis
        v_llist = self.llist
        v_alllist = self.alllist

        # 27 neighbor shifts in [-1,0,1]^3
        iaux = 0
        shifts27 = np.empty((27, 3), dtype=np.intc)
        for ii in range(-1, 2):
            for jj in range(-1, 2):
                for kk in range(-1, 2):
                    shifts27[iaux, 0] = ii
                    shifts27[iaux, 1] = jj
                    shifts27[iaux, 2] = kk
                    iaux += 1

        basis = np.empty(4, dtype=np.intc)
        quartet = np.empty(4, dtype=np.intc)
        quartet_perm = np.empty(4, dtype=np.intc)
        quartet_sym = np.empty(4, dtype=np.intc)
        shift2all = np.empty((3, 27), dtype=np.intc)
        shift3all = np.empty((3, 27), dtype=np.intc)
        shift4all = np.empty((3, 27), dtype=np.intc)
        equilist = np.empty((4, nsym * 24), dtype=np.intc)
        coeffi = np.empty((24 * nsym * 81, 81), dtype=np.double)
        id_equi = self.symops.map_supercell(self.sposcar)
        ind_cell, ind_species = _id2ind(ngrid, natoms)

        # Rotation matrices for fourth derivatives and related quantities.
        rot = np.empty((24, nsym, 81, 81), dtype=np.double)
        for iperm in range(24):
            for isym in range(nsym):
                for ibasisprime in range(3):
                    for jbasisprime in range(3):
                        for kbasisprime in range(3):
                            for lbasisprime in range(3):
                                indexijklprime = (
                                    (ibasisprime * 3 + jbasisprime) * 3 + kbasisprime
                                ) * 3 + lbasisprime
                                for ibasis in range(3):
                                    basis[0] = ibasis
                                    for jbasis in range(3):
                                        basis[1] = jbasis
                                        for kbasis in range(3):
                                            basis[2] = kbasis
                                            for lbasis in range(3):
                                                basis[3] = lbasis
                                                indexijkl = (
                                                    ibasis * 27
                                                    + jbasis * 9
                                                    + kbasis * 3
                                                    + lbasis
                                                )
                                                ibasispermut = basis[
                                                    permutations[iperm, 0]
                                                ]
                                                jbasispermut = basis[
                                                    permutations[iperm, 1]
                                                ]
                                                kbasispermut = basis[
                                                    permutations[iperm, 2]
                                                ]
                                                lbasispermut = basis[
                                                    permutations[iperm, 3]
                                                ]
                                                rot[
                                                    iperm,
                                                    isym,
                                                    indexijklprime,
                                                    indexijkl,
                                                ] = (
                                                    orth[
                                                        ibasisprime, ibasispermut, isym
                                                    ]
                                                    * orth[
                                                        jbasisprime, jbasispermut, isym
                                                    ]
                                                    * orth[
                                                        kbasisprime, kbasispermut, isym
                                                    ]
                                                    * orth[
                                                        lbasisprime, lbasispermut, isym
                                                    ]
                                                )
        rot2 = rot.copy()
        nonzero = np.zeros((24, nsym, 81), dtype=np.intc)
        for iperm in range(24):
            for isym in range(nsym):
                for indexijklprime in range(81):
                    rot2[iperm, isym, indexijklprime, indexijklprime] -= 1.0
                    for indexijkl in range(81):
                        if fabs(rot2[iperm, isym, indexijklprime, indexijkl]) > 1e-12:
                            nonzero[iperm, isym, indexijklprime] = 1
                        else:
                            rot2[iperm, isym, indexijklprime, indexijkl] = 0.0

        # Scan all atom quartets (ii,jj,kk,mm) in the supercell.
        for ii in range(natoms):
            for jj in range(ntot):
                dist = self.dmin[ii, jj]
                if dist >= self.frange:
                    continue
                n2equi = self.nequis[ii, jj]
                for kk in range(n2equi):
                    shift2all[:, kk] = shifts27[self.shifts[ii, jj, kk], :]
                for kk in range(ntot):
                    dist = self.dmin[ii, kk]
                    if dist >= self.frange:
                        continue
                    n3equi = self.nequis[ii, kk]
                    for ll in range(n3equi):
                        shift3all[:, ll] = shifts27[self.shifts[ii, kk, ll], :]
                    for mm in range(ntot):
                        dist = self.dmin[ii, mm]
                        if dist >= self.frange:
                            continue
                        n4equi = self.nequis[ii, mm]
                        for nn in range(n4equi):
                            shift4all[:, nn] = shifts27[self.shifts[ii, mm, nn], :]
                        d2_min1 = np.inf
                        d2_min2 = np.inf
                        d2_min3 = np.inf
                        for iaux in range(n2equi):
                            # car2 for jj image
                            car2_0 = (
                                shift2all[0, iaux] * lattvec[0, 0]
                                + shift2all[1, iaux] * lattvec[0, 1]
                                + shift2all[2, iaux] * lattvec[0, 2]
                                + coordall[0, jj]
                            )
                            car2_1 = (
                                shift2all[0, iaux] * lattvec[1, 0]
                                + shift2all[1, iaux] * lattvec[1, 1]
                                + shift2all[2, iaux] * lattvec[1, 2]
                                + coordall[1, jj]
                            )
                            car2_2 = (
                                shift2all[0, iaux] * lattvec[2, 0]
                                + shift2all[1, iaux] * lattvec[2, 1]
                                + shift2all[2, iaux] * lattvec[2, 2]
                                + coordall[2, jj]
                            )
                            for jaux in range(n3equi):
                                # car3 for kk image
                                car3_0 = (
                                    shift3all[0, jaux] * lattvec[0, 0]
                                    + shift3all[1, jaux] * lattvec[0, 1]
                                    + shift3all[2, jaux] * lattvec[0, 2]
                                    + coordall[0, kk]
                                )
                                car3_1 = (
                                    shift3all[0, jaux] * lattvec[1, 0]
                                    + shift3all[1, jaux] * lattvec[1, 1]
                                    + shift3all[2, jaux] * lattvec[1, 2]
                                    + coordall[1, kk]
                                )
                                car3_2 = (
                                    shift3all[0, jaux] * lattvec[2, 0]
                                    + shift3all[1, jaux] * lattvec[2, 1]
                                    + shift3all[2, jaux] * lattvec[2, 2]
                                    + coordall[2, kk]
                                )
                                for kaux in range(n4equi):
                                    # car4 for mm image
                                    car4_0 = (
                                        shift4all[0, kaux] * lattvec[0, 0]
                                        + shift4all[1, kaux] * lattvec[0, 1]
                                        + shift4all[2, kaux] * lattvec[0, 2]
                                        + coordall[0, mm]
                                    )
                                    car4_1 = (
                                        shift4all[0, kaux] * lattvec[1, 0]
                                        + shift4all[1, kaux] * lattvec[1, 1]
                                        + shift4all[2, kaux] * lattvec[1, 2]
                                        + coordall[1, mm]
                                    )
                                    car4_2 = (
                                        shift4all[0, kaux] * lattvec[2, 0]
                                        + shift4all[1, kaux] * lattvec[2, 1]
                                        + shift4all[2, kaux] * lattvec[2, 2]
                                        + coordall[2, mm]
                                    )
                                # distances
                                d2_31 = (
                                    (car3_0 - car2_0) ** 2
                                    + (car3_1 - car2_1) ** 2
                                    + (car3_2 - car2_2) ** 2
                                )
                                if d2_31 < d2_min1:
                                    d2_min1 = d2_31
                                d2_42 = (
                                    (car4_0 - car2_0) ** 2
                                    + (car4_1 - car2_1) ** 2
                                    + (car4_2 - car2_2) ** 2
                                )
                                if d2_42 < d2_min2:
                                    d2_min2 = d2_42
                                d2_43 = (
                                    (car4_0 - car3_0) ** 2
                                    + (car4_1 - car3_1) ** 2
                                    + (car4_2 - car3_2) ** 2
                                )
                                if d2_43 < d2_min3:
                                    d2_min3 = d2_43
                        if d2_min1 >= frange2:
                            continue
                        if d2_min2 >= frange2:
                            continue
                        quartet[0] = ii
                        quartet[1] = jj
                        quartet[2] = kk
                        quartet[3] = mm
                        if _quartet_in_list(quartet, v_alllist, self.nalllist):
                            continue
                        self.nlist += 1
                        if self.nlist == self.allocsize:
                            self._expandlist()
                            v_nequi = self.nequi
                            v_allequilist = self.allequilist
                            v_transformation = self.transformation
                            v_transformationarray = self.transformationarray
                            v_transformationaux = self.transformationaux
                            v_nindependentbasis = self.nindependentbasis
                            v_independentbasis = self.independentbasis
                            v_llist = self.llist
                        v_llist[0, self.nlist - 1] = ii
                        v_llist[1, self.nlist - 1] = jj
                        v_llist[2, self.nlist - 1] = kk
                        v_llist[3, self.nlist - 1] = mm
                        v_nequi[self.nlist - 1] = 0
                        coeffi[:, :] = 0.0
                        nnonzero = 0
                        # 24 permutations
                        for iperm in range(24):
                            quartet_perm[0] = quartet[permutations[iperm, 0]]
                            quartet_perm[1] = quartet[permutations[iperm, 1]]
                            quartet_perm[2] = quartet[permutations[iperm, 2]]
                            quartet_perm[3] = quartet[permutations[iperm, 3]]
                            # apply all sym ops
                            for isym in range(nsym):
                                quartet_sym[0] = id_equi[isym, quartet_perm[0]]
                                quartet_sym[1] = id_equi[isym, quartet_perm[1]]
                                quartet_sym[2] = id_equi[isym, quartet_perm[2]]
                                quartet_sym[3] = id_equi[isym, quartet_perm[3]]
                                vec1 = ind_cell[
                                    :, id_equi[isym, quartet_perm[0]]
                                ].copy()
                                vec2 = ind_cell[
                                    :, id_equi[isym, quartet_perm[1]]
                                ].copy()
                                vec3 = ind_cell[
                                    :, id_equi[isym, quartet_perm[2]]
                                ].copy()
                                vec4 = ind_cell[
                                    :, id_equi[isym, quartet_perm[3]]
                                ].copy()
                                # ensure atom 0 in first cell
                                if not (vec1[0] == 0 and vec1[1] == 0 and vec1[2] == 0):
                                    vec4 = (vec4 - vec1) % ngrid
                                    vec3 = (vec3 - vec1) % ngrid
                                    vec2 = (vec2 - vec1) % ngrid
                                    vec1[:] = 0
                                    ispecies1 = ind_species[
                                        id_equi[isym, quartet_perm[0]]
                                    ]
                                    ispecies2 = ind_species[
                                        id_equi[isym, quartet_perm[1]]
                                    ]
                                    ispecies3 = ind_species[
                                        id_equi[isym, quartet_perm[2]]
                                    ]
                                    ispecies4 = ind_species[
                                        id_equi[isym, quartet_perm[3]]
                                    ]
                                    quartet_sym[0] = _ind2id(
                                        vec1, ispecies1, ngrid, natoms
                                    )
                                    quartet_sym[1] = _ind2id(
                                        vec2, ispecies2, ngrid, natoms
                                    )
                                    quartet_sym[2] = _ind2id(
                                        vec3, ispecies3, ngrid, natoms
                                    )
                                    quartet_sym[3] = _ind2id(
                                        vec4, ispecies4, ngrid, natoms
                                    )
                                # new image? add
                                if (iperm == 0 and isym == 0) or not (
                                    _quartets_are_equal(quartet_sym, quartet)
                                    or _quartet_in_list(
                                        quartet_sym, equilist, v_nequi[self.nlist - 1]
                                    )
                                ):
                                    v_nequi[self.nlist - 1] += 1
                                    for ll in range(4):
                                        equilist[ll, v_nequi[self.nlist - 1] - 1] = (
                                            quartet_sym[ll]
                                        )
                                        v_allequilist[
                                            ll,
                                            v_nequi[self.nlist - 1] - 1,
                                            self.nlist - 1,
                                        ] = quartet_sym[ll]
                                    self.nalllist += 1
                                    if self.nalllist == self.allallocsize:
                                        self._expandalllist()
                                        v_alllist = self.alllist
                                    for ll in range(4):
                                        v_alllist[ll, self.nalllist - 1] = quartet_sym[
                                            ll
                                        ]
                                    for iaux in range(81):
                                        for jaux in range(81):
                                            v_transformation[
                                                iaux,
                                                jaux,
                                                v_nequi[self.nlist - 1] - 1,
                                                self.nlist - 1,
                                            ] = rot[iperm, isym, iaux, jaux]
                                # identity? add row to coeffi
                                if _quartets_are_equal(quartet_sym, quartet):
                                    for indexijklprime in range(81):
                                        if nonzero[iperm, isym, indexijklprime]:
                                            for ll in range(81):
                                                coeffi[nnonzero, ll] = rot2[
                                                    iperm, isym, indexijklprime, ll
                                                ]
                                            nnonzero += 1
                        coeffi_reduced = np.zeros(
                            (max(nnonzero, 81), 81), dtype=np.double
                        )
                        for iaux in range(nnonzero):
                            for jaux in range(81):
                                coeffi_reduced[iaux, jaux] = coeffi[iaux, jaux]
                        # Gaussian independent set
                        b, independent = gaussian(coeffi_reduced)
                        for iaux in range(81):
                            for jaux in range(81):
                                v_transformationaux[iaux, jaux, self.nlist - 1] = b[
                                    iaux, jaux
                                ]
                        v_nindependentbasis[self.nlist - 1] = independent.shape[0]
                        for ll in range(independent.shape[0]):
                            v_independentbasis[ll, self.nlist - 1] = independent[ll]
        v_transformationarray[:, :, :, :] = 0.0
        _build_transformationarray(
            v_transformation,
            v_transformationaux,
            v_nequi,
            v_nindependentbasis,
            self.nlist,
            v_transformationarray,
        )

    def build_list4(self):
        """
        Build a list of 6-uples from the results of the reduction.
        """
        list6 = []
        for ii in range(self.nlist):
            for jj in range(self.nindependentbasis[ii]):
                kk = self.independentbasis[jj, ii] // 27
                ll = (self.independentbasis[jj, ii] % 27) // 9
                mm = (self.independentbasis[jj, ii] % 9) // 3
                nn = self.independentbasis[jj, ii] % 3
                list6.append(
                    (
                        kk,
                        self.llist[0, ii],
                        ll,
                        self.llist[1, ii],
                        mm,
                        self.llist[2, ii],
                        nn,
                        self.llist[3, ii],
                    )
                )
        nruter = []
        for i in list6:
            fournumbers = (i[1], i[3], i[5], i[0], i[2], i[4])
            if fournumbers not in nruter:
                nruter.append(fournumbers)
        return nruter
