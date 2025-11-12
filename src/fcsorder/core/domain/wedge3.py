import numpy as np
import scipy as sp
import sparse
from rich.progress import track
import typer
from math import fabs

from numba import jit

# Keep the same permutations constant (order matters)
permutations = np.array(
    [[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1], [1, 2, 0], [2, 0, 1]], dtype=np.intc
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
def _triplet_in_list(triplet, llist, nlist):
    """
    Return True if triplet is found in llist[:,:nlist]. The first dimension of list must have a length of 3.
    triplet: shape (3,)
    llist: shape (3, N)
    """
    for i in range(nlist):
        if (
            triplet[0] == llist[0, i]
            and triplet[1] == llist[1, i]
            and triplet[2] == llist[2, i]
        ):
            return True
    return False


@jit(nopython=True)
def _triplets_are_equal(triplet1, triplet2):
    """
    Return True if two triplets are equal and False otherwise.
    triplet1, triplet2: shape (3,)
    """
    for i in range(3):
        if triplet1[i] != triplet2[i]:
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
def _build_transformationarray3(
    v_transformation,
    v_transformationaux,
    v_nequi,
    v_nindependentbasis,
    nlist,
    out_array,
):
    """
    Compute for 3rd-order case (27 basis size):
      out_array[kk, ll, jj, ii] = sum_{iaux=0..26} v_transformation[kk, iaux, jj, ii]
                                             * v_transformationaux[iaux, ll, ii]
    Only ll < v_nindependentbasis[ii] columns are written. Values with |s| < 1e-12 are zeroed.
    """
    for ii in range(nlist):
        ne = v_nequi[ii]
        nind = v_nindependentbasis[ii]
        for jj in range(ne):
            for kk in range(27):
                for ll in range(nind):
                    s = 0.0
                    for iaux in range(27):
                        s += (
                            v_transformation[kk, iaux, jj, ii]
                            * v_transformationaux[iaux, ll, ii]
                        )
                    if s != 0.0 and abs(s) < 1e-12:
                        s = 0.0
                    out_array[kk, ll, jj, ii] = s


class Wedge:
    """
    Objects of this class allow the user to extract irreducible sets
    of force constants and to reconstruct the full third-order IFC
    matrix from them.
    """

    # Public attributes kept to match the Cython class interface
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

    # Private buffers (same names as Cython code)
    alllist: np.ndarray
    transformation: np.ndarray
    transformationaux: np.ndarray

    # More fields mirroring the Cython memoryviews (stored as np.ndarray)
    nequis: np.ndarray
    shifts: np.ndarray
    dmin: np.ndarray
    frange: float

    def __init__(self, poscar, sposcar, symops, dmin, nequis, shifts, frange):
        """
        Build the object by computing all the relevant information about
        irreducible IFCs.
        """
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
        transformationaux, nindependentbasis, independentbasis,
        and llist to accommodate more elements.
        """
        if self.allocsize == 0:
            self.allocsize = 16
            self.nequi = np.empty(self.allocsize, dtype=np.intc)
            self.allequilist = np.empty(
                (3, 6 * self.symops.nsyms, self.allocsize), dtype=np.intc
            )
            self.transformationarray = np.empty(
                (27, 27, 6 * self.symops.nsyms, self.allocsize), dtype=np.double
            )
            self.transformation = np.empty(
                (27, 27, 6 * self.symops.nsyms, self.allocsize), dtype=np.double
            )
            self.transformationaux = np.empty((27, 27, self.allocsize), dtype=np.double)
            self.nindependentbasis = np.empty(self.allocsize, dtype=np.intc)
            self.independentbasis = np.empty((27, self.allocsize), dtype=np.intc)
            self.llist = np.empty((3, self.allocsize), dtype=np.intc)
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
        """
        Expand alllist to accommodate more elements.
        """
        if self.allallocsize == 0:
            self.allallocsize = 512
            self.alllist = np.empty((3, self.allallocsize), dtype=np.intc)
        else:
            self.allallocsize <<= 1
            self.alllist = np.concatenate((self.alllist, self.alllist), axis=-1)

    def _reduce(self):
        """
        Method that performs most of the actual work.
        Ported directly from the Cython implementation; names and logic preserved.
        """
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

        # build 27 neighbor shifts in [-1,0,1]^3
        iaux = 0
        shifts27 = np.empty((27, 3), dtype=np.intc)
        for ii in range(-1, 2):
            for jj in range(-1, 2):
                for kk in range(-1, 2):
                    shifts27[iaux, 0] = ii
                    shifts27[iaux, 1] = jj
                    shifts27[iaux, 2] = kk
                    iaux += 1

        basis = np.empty(3, dtype=np.intc)
        triplet = np.empty(3, dtype=np.intc)
        triplet_perm = np.empty(3, dtype=np.intc)
        triplet_sym = np.empty(3, dtype=np.intc)
        shift2all = np.empty((3, 27), dtype=np.intc)
        shift3all = np.empty((3, 27), dtype=np.intc)
        equilist = np.empty((3, nsym * 6), dtype=np.intc)
        coeffi = np.empty((6 * nsym * 27, 27), dtype=np.double)
        id_equi = self.symops.map_supercell(self.sposcar)
        ind_cell, ind_species = _id2ind(ngrid, natoms)

        # Rotation matrices for third derivatives and related quantities.
        rot = np.empty((6, nsym, 27, 27), dtype=np.double)
        for iperm in range(6):
            for isym in range(nsym):
                for ibasisprime in range(3):
                    for jbasisprime in range(3):
                        for kbasisprime in range(3):
                            indexijkprime = (
                                ibasisprime * 3 + jbasisprime
                            ) * 3 + kbasisprime
                            for ibasis in range(3):
                                basis[0] = ibasis
                                for jbasis in range(3):
                                    basis[1] = jbasis
                                    for kbasis in range(3):
                                        basis[2] = kbasis
                                        indexijk = ibasis * 9 + jbasis * 3 + kbasis
                                        ibasispermut = basis[permutations[iperm, 0]]
                                        jbasispermut = basis[permutations[iperm, 1]]
                                        kbasispermut = basis[permutations[iperm, 2]]
                                        rot[iperm, isym, indexijkprime, indexijk] = (
                                            orth[ibasisprime, ibasispermut, isym]
                                            * orth[jbasisprime, jbasispermut, isym]
                                            * orth[kbasisprime, kbasispermut, isym]
                                        )
        rot2 = rot.copy()
        nonzero = np.zeros((6, nsym, 27), dtype=np.intc)
        for iperm in range(6):
            for isym in range(nsym):
                for indexijkprime in range(27):
                    rot2[iperm, isym, indexijkprime, indexijkprime] -= 1.0
                    for indexijk in range(27):
                        if fabs(rot2[iperm, isym, indexijkprime, indexijk]) > 1e-12:
                            nonzero[iperm, isym, indexijkprime] = 1
                        else:
                            rot2[iperm, isym, indexijkprime, indexijk] = 0.0

        # Scan all atom triplets (ii, jj, kk) in the supercell.
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
                    d2_min = np.inf
                    for iaux in range(n2equi):
                        # car2: cartesian coordinate for jj image
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
                            # car3: cartesian coordinate for kk image
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
                        d2 = (
                            (car3_0 - car2_0) ** 2
                            + (car3_1 - car2_1) ** 2
                            + (car3_2 - car2_2) ** 2
                        )
                        if d2 < d2_min:
                            d2_min = d2
                    if d2_min >= frange2:
                        continue
                    # This point is only reached if there is a choice of periodic images of
                    # ii, jj and kk such that all pairs ii-jj, ii-kk and jj-kk are within
                    # the specified interaction range.
                    triplet[0] = ii
                    triplet[1] = jj
                    triplet[2] = kk
                    if _triplet_in_list(triplet, v_alllist, self.nalllist):
                        continue
                    # This point is only reached if the triplet is not equivalent to any
                    # of the triplets already considered, including permutations and symmetries.
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
                    v_nequi[self.nlist - 1] = 0
                    coeffi[:, :] = 0.0
                    nnonzero = 0
                    # Scan the six possible permutations of triplet (ii, jj, kk).
                    for iperm in range(6):
                        triplet_perm[0] = triplet[permutations[iperm, 0]]
                        triplet_perm[1] = triplet[permutations[iperm, 1]]
                        triplet_perm[2] = triplet[permutations[iperm, 2]]
                        # Explore the effect of all symmetry operations on each of the permuted triplets.
                        for isym in range(nsym):
                            triplet_sym[0] = id_equi[isym, triplet_perm[0]]
                            triplet_sym[1] = id_equi[isym, triplet_perm[1]]
                            triplet_sym[2] = id_equi[isym, triplet_perm[2]]
                            vec1 = ind_cell[:, id_equi[isym, triplet_perm[0]]].copy()
                            vec2 = ind_cell[:, id_equi[isym, triplet_perm[1]]].copy()
                            vec3 = ind_cell[:, id_equi[isym, triplet_perm[2]]].copy()
                            # Choose a displaced version of triplet_sym chosen so that atom 0 is in first unit cell.
                            if not (vec1[0] == 0 and vec1[1] == 0 and vec1[2] == 0):
                                vec3 = (vec3 - vec1) % ngrid
                                vec2 = (vec2 - vec1) % ngrid
                                vec1[:] = 0
                                ispecies1 = ind_species[id_equi[isym, triplet_perm[0]]]
                                ispecies2 = ind_species[id_equi[isym, triplet_perm[1]]]
                                ispecies3 = ind_species[id_equi[isym, triplet_perm[2]]]
                                triplet_sym[0] = _ind2id(vec1, ispecies1, ngrid, natoms)
                                triplet_sym[1] = _ind2id(vec2, ispecies2, ngrid, natoms)
                                triplet_sym[2] = _ind2id(vec3, ispecies3, ngrid, natoms)
                            # If permutation+symmetry yields unseen image, add it and fill transformation.
                            if (iperm == 0 and isym == 0) or not (
                                _triplets_are_equal(triplet_sym, triplet)
                                or _triplet_in_list(
                                    triplet_sym, equilist, v_nequi[self.nlist - 1]
                                )
                            ):
                                v_nequi[self.nlist - 1] += 1
                                for ll in range(3):
                                    equilist[ll, v_nequi[self.nlist - 1] - 1] = (
                                        triplet_sym[ll]
                                    )
                                    v_allequilist[
                                        ll, v_nequi[self.nlist - 1] - 1, self.nlist - 1
                                    ] = triplet_sym[ll]
                                self.nalllist += 1
                                if self.nalllist == self.allallocsize:
                                    self._expandalllist()
                                    v_alllist = self.alllist
                                for ll in range(3):
                                    v_alllist[ll, self.nalllist - 1] = triplet_sym[ll]
                                for iaux in range(27):
                                    for jaux in range(27):
                                        v_transformation[
                                            iaux,
                                            jaux,
                                            v_nequi[self.nlist - 1] - 1,
                                            self.nlist - 1,
                                        ] = rot[iperm, isym, iaux, jaux]
                            # If the permutation+symmetry amounts to identity, add a row to the coefficient matrix.
                            if _triplets_are_equal(triplet_sym, triplet):
                                for indexijkprime in range(27):
                                    if nonzero[iperm, isym, indexijkprime]:
                                        for ll in range(27):
                                            coeffi[nnonzero, ll] = rot2[
                                                iperm, isym, indexijkprime, ll
                                            ]
                                        nnonzero += 1
                    coeffi_reduced = np.zeros((max(nnonzero, 27), 27), dtype=np.double)
                    for iaux in range(nnonzero):
                        for jaux in range(27):
                            coeffi_reduced[iaux, jaux] = coeffi[iaux, jaux]
                    # Obtain a set of independent IFCs for this triplet equivalence class.
                    b, independent = gaussian(coeffi_reduced)
                    for iaux in range(27):
                        for jaux in range(27):
                            v_transformationaux[iaux, jaux, self.nlist - 1] = b[
                                iaux, jaux
                            ]
                    v_nindependentbasis[self.nlist - 1] = independent.shape[0]
                    for ll in range(independent.shape[0]):
                        v_independentbasis[ll, self.nlist - 1] = independent[ll]
        v_transformationarray[:, :, :, :] = 0.0
        _build_transformationarray3(
            v_transformation,
            v_transformationaux,
            v_nequi,
            v_nindependentbasis,
            self.nlist,
            v_transformationarray,
        )

    def build_list4(self):
        """
        Build a list of 4-uples from the results of the reduction.
        """
        list6 = []
        for ii in range(self.nlist):
            for jj in range(self.nindependentbasis[ii]):
                ll = self.independentbasis[jj, ii] // 9
                mm = (self.independentbasis[jj, ii] % 9) // 3
                nn = self.independentbasis[jj, ii] % 3
                list6.append(
                    (
                        ll,
                        self.llist[0, ii],
                        mm,
                        self.llist[1, ii],
                        nn,
                        self.llist[2, ii],
                    )
                )
        nruter = []
        for i in list6:
            fournumbers = (i[1], i[3], i[0], i[2])
            if fournumbers not in nruter:
                nruter.append(fournumbers)
        return nruter
