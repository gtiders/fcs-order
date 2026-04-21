from numba import njit, typed, types
import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import itertools
from mlfcs.utils.symmetry import SymmetryOperations
permutations = np.array([
    [0, 1, 2],
    [1, 0, 2],
    [2, 1, 0],
    [0, 2, 1],
    [1, 2, 0],
    [2, 0, 1]], dtype=np.int8)


class IFCMap:
    """
    Efficient map for storing 6-index IFC values using Numba typed.Dict.
    
    Keys are packed into a single int64 for fast lookup:
    key = (dir0 << 62) | (dir1 << 60) | (dir2 << 58) | 
          (atom0 << 42) | (atom1 << 28) | atom2
    """
    
    def __init__(self):
        self._data = typed.Dict.empty(types.int64, types.float64)
    
    @staticmethod
    def _make_key(dir0, dir1, dir2, atom0, atom1, atom2):
        """
        Pack 3 direction indices (2 bits each) and 3 atom indices (14 bits each) into int64.
        
        Direction indices: range 0-2 (only need 2 bits)
        Atom indices: range 0-16383 (14 bits, supports supercells up to 16384 atoms)
        """
        return (np.int64(dir0) << 62) | (np.int64(dir1) << 60) | \
               (np.int64(dir2) << 58) | \
               (np.int64(atom0) << 42) | (np.int64(atom1) << 28) | \
               np.int64(atom2)
    
    @staticmethod
    def _unpack_key(k):
        dir0 = (k >> 62) & 0x3
        dir1 = (k >> 60) & 0x3
        dir2 = (k >> 58) & 0x3
        atom0 = (k >> 42) & 0x3FFF
        atom1 = (k >> 28) & 0x3FFF
        atom2 = k & 0x3FFF
        return (dir0, dir1, dir2, atom0, atom1, atom2)
    
    def set_item(self, dir0, dir1, dir2, atom0, atom1, atom2, val):
        self._data[self._make_key(dir0, dir1, dir2, atom0, atom1, atom2)] = val
    
    def get_item(self, dir0, dir1, dir2, atom0, atom1, atom2):
        k = self._make_key(dir0, dir1, dir2, atom0, atom1, atom2)
        if k in self._data:
            return self._data[k]
        return 0.0
    
    def add_item(self, dir0, dir1, dir2, atom0, atom1, atom2, val):
        k = self._make_key(dir0, dir1, dir2, atom0, atom1, atom2)
        if k in self._data:
            self._data[k] += val
        else:
            self._data[k] = val
    
    def clear(self):
        self._data.clear()
    
    def size(self):
        return len(self._data)
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, key):
        return self.get_item(*key)
    
    def __setitem__(self, key, value):
        self.set_item(*key, value)
    
    def get(self, key, default=0.0):
        k = self._make_key(*key)
        if k in self._data:
            return self._data[k]
        return default
    
    def __contains__(self, key):
        return self._make_key(*key) in self._data
    
    def items(self):
        result = []
        for k, v in self._data.items():
            result.append((self._unpack_key(k), v))
        return result


@njit
def _ind2id(icell, ispecies, ngrid, nspecies):
    """
    Merge a set of cell+atom indices into a single index into a supercell.
    """
    return (icell[0] + (icell[1] + icell[2] * ngrid[1]) * ngrid[0]) * nspecies + ispecies


@njit
def _triplet_in_list(triplet, llist, nlist):
    """
    Return True if triplet is found in llist[:,:nlist]. The first dimension
    of list must have a length of 3.
    """
    for i in range(nlist):
        if (triplet[0] == llist[0, i] and
            triplet[1] == llist[1, i] and triplet[2] == llist[2, i]):
            return True
    return False


@njit
def _triplets_are_equal(triplet1, triplet2):
    """
    Return True if two triplets are equal and False otherwise.
    """
    for i in range(3):
        if triplet1[i] != triplet2[i]:
            return False
    return True


@njit
def _id2ind(ngrid, nspecies):
    """
    Create a map from supercell indices to cell+atom indices.
    """
    ntot = ngrid[0] * ngrid[1] * ngrid[2] * nspecies
    icell = np.empty((3, ntot), dtype=np.int64)
    ispecies = np.empty(ntot, dtype=np.int64)
    
    for ii in range(ntot):
        tmp, ispecies[ii] = divmod(ii, nspecies)
        tmp, icell[0, ii] = divmod(tmp, ngrid[0])
        icell[2, ii], icell[1, ii] = divmod(tmp, ngrid[1])
    
    return (icell, ispecies)


@njit
def _build_sparse_coefs(natoms, ntot, vind1, vind2, naccumindependent, vtrans):
    """
    Build sparse matrix coefficients with exact preallocation for third-order.
    
    Returns (i, j, v) arrays for COO sparse matrix.
    """
    total_elements = 0
    for ii in range(natoms):
        for jj in range(ntot):
            for kk in range(ntot):
                ix = vind1[ii, jj, kk]
                if ix >= 0:
                    total_elements += 27 * (naccumindependent[ix + 1] - naccumindependent[ix])
    
    i_arr = np.empty(total_elements, dtype=np.int64)
    j_arr = np.empty(total_elements, dtype=np.int64)
    v_arr = np.empty(total_elements, dtype=np.float64)
    
    idx = 0
    for ii in range(natoms):
        for jj in range(ntot):
            base_colindex = ii * (ntot * 27) + jj * 27
            for kk in range(ntot):
                ix = vind1[ii, jj, kk]
                if ix < 0:
                    continue
                start_ss = naccumindependent[ix]
                num_ss = naccumindependent[ix + 1] - start_ss
                v2 = vind2[ii, jj, kk]
                
                for tribasisindex in range(27):
                    colindex = base_colindex + tribasisindex
                    for tt in range(num_ss):
                        i_arr[idx] = start_ss + tt
                        j_arr[idx] = colindex
                        v_arr[idx] = vtrans[tribasisindex, tt, v2, ix]
                        idx += 1
    
    return i_arr, j_arr, v_arr


def reconstruct_ifcs(phipart, wedge, list4, poscar, sposcar):
    """
    Recover the full anharmonic IFC set from the irreducible set of
    force constants and the information contained in a wedge object.
    """
    nlist = wedge.nlist
    natoms = len(poscar["types"])
    ntot = len(sposcar["types"])
    vnruter = IFCMap()
    
    naccumindependent = np.insert(np.cumsum(
        wedge.nindependentbasis[:nlist], dtype=np.int64), 0,
        np.zeros(1, dtype=np.int64))
    ntotalindependent = naccumindependent[-1]
    vphipart = phipart
    nlist4 = len(list4)
    for ii in range(nlist4):
        e0, e1, e2, e3 = list4[ii]
        for k in range(3):
            for l in range(ntot):
                val = vphipart[k, ii, l]
                if val != 0.0:
                    vnruter.set_item(e2, e3, k, e0, e1, l, val)
    
    philist = []
    for ii in range(nlist):
        for jj in range(wedge.nindependentbasis[ii]):
            ll = wedge.independentbasis[jj, ii] // 9
            mm = (wedge.independentbasis[jj, ii] % 9) // 3
            nn = wedge.independentbasis[jj, ii] % 3
            val = vnruter.get_item(ll, mm, nn,
                          wedge.llist[0, ii],
                          wedge.llist[1, ii],
                          wedge.llist[2, ii])
            philist.append(val)
    aphilist = np.array(philist, dtype=np.float64)
    vind1 = -np.ones((natoms, ntot, ntot), dtype=np.int64)
    vind2 = -np.ones((natoms, ntot, ntot), dtype=np.int64)
    vequilist = wedge.allequilist
    for ii in range(nlist):
        for jj in range(wedge.nequi[ii]):
            vind1[vequilist[0, jj, ii],
                  vequilist[1, jj, ii],
                  vequilist[2, jj, ii]] = ii
            vind2[vequilist[0, jj, ii],
                  vequilist[1, jj, ii],
                  vequilist[2, jj, ii]] = jj
    
    vtrans = wedge.transformationarray
    
    nrows = ntotalindependent
    ncols = natoms * ntot * 27
    
    print("- Storing the coefficients in a sparse matrix")
    i_arr, j_arr, v_arr = _build_sparse_coefs(
        natoms, ntot, vind1, vind2, naccumindependent, vtrans)
    print("- \t Density: {0:.2g}%".format(100. * len(i_arr) / float(nrows * ncols)))
    aa = sp.sparse.coo_matrix((v_arr, (i_arr, j_arr)), (nrows, ncols)).tocsr()
    D = sp.sparse.spdiags(aphilist, [0, ], aphilist.size, aphilist.size,
                           format="csr")
    bbs = D.dot(aa)
    ones = np.ones_like(aphilist)
    multiplier = -sp.sparse.linalg.lsqr(bbs, ones)[0]
    compensation = D.dot(bbs.dot(multiplier))
    
    aphilist += compensation
    
    vnruter.clear()
    for ii in range(nlist):
        for jj in range(wedge.nequi[ii]):
            for ll in range(3):
                for mm in range(3):
                    for nn in range(3):
                        tribasisindex = (ll * 3 + mm) * 3 + nn
                        for ix in range(wedge.nindependentbasis[ii]):
                            val = wedge.transformationarray[
                                        tribasisindex, ix, jj, ii] * aphilist[
                                            naccumindependent[ii] + ix]
                            if val != 0.0:
                                vnruter.add_item(ll, mm, nn,
                                         vequilist[0, jj, ii],
                                         vequilist[1, jj, ii],
                                         vequilist[2, jj, ii], val)
    return vnruter


class Wedge:
    """
    Objects of this class allow the user to extract irreducible sets
    of force constants and to reconstruct the full third-order IFC
    matrix from them.
    """
    
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
        self.nalllist = 0
        self.nlist = 0
        
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
            self.nequi = np.empty(self.allocsize, dtype=np.int64)
            self.allequilist = np.empty((3, 6 * self.symops.nsyms, self.allocsize), dtype=np.int64)
            self.transformationarray = np.empty((27, 27, 6 * self.symops.nsyms, self.allocsize), dtype=np.float64)
            self.transformation = np.empty((27, 27, 6 * self.symops.nsyms, self.allocsize), dtype=np.float64)
            self.transformationaux = np.empty((27, 27, self.allocsize), dtype=np.float64)
            self.nindependentbasis = np.empty(self.allocsize, dtype=np.int64)
            self.independentbasis = np.empty((27, self.allocsize), dtype=np.int64)
            self.llist = np.empty((3, self.allocsize), dtype=np.int64)
        else:
            self.allocsize <<= 1
            self.nequi = np.concatenate((self.nequi, self.nequi), axis=-1)
            self.allequilist = np.concatenate((self.allequilist, self.allequilist), axis=-1)
            self.transformation = np.concatenate((self.transformation, self.transformation), axis=-1)
            self.transformationarray = np.concatenate((self.transformationarray, self.transformationarray), axis=-1)
            self.transformationaux = np.concatenate((self.transformationaux, self.transformationaux), axis=-1)
            self.nindependentbasis = np.concatenate((self.nindependentbasis, self.nindependentbasis), axis=-1)
            self.independentbasis = np.concatenate((self.independentbasis, self.independentbasis), axis=-1)
            self.llist = np.concatenate((self.llist, self.llist), axis=-1)
    
    def _expandalllist(self):
        """
        Expand alllist to accommodate more elements.
        """
        if self.allallocsize == 0:
            self.allallocsize = 512
            self.alllist = np.empty((3, self.allallocsize), dtype=np.int64)
        else:
            self.allallocsize <<= 1
            self.alllist = np.concatenate((self.alllist, self.alllist), axis=-1)
    
    def _reduce(self):
        """
        Method that performs most of the actual work.
        """
        frange2 = self.frange * self.frange
        
        ngrid1 = self.sposcar["na"]
        ngrid2 = self.sposcar["nb"]
        ngrid3 = self.sposcar["nc"]
        ngrid = np.array([ngrid1, ngrid2, ngrid3], dtype=np.int64)
        nsym = self.symops.nsyms
        natoms = len(self.poscar["types"])
        ntot = len(self.sposcar["types"])
        vec1 = np.empty(3, dtype=np.int64)
        vec2 = np.empty(3, dtype=np.int64)
        vec3 = np.empty(3, dtype=np.int64)
        
        lattvec = self.sposcar["lattvec"]
        coordall = np.dot(lattvec, self.sposcar["positions"])
        orth = np.transpose(self.symops.crotations, (1, 2, 0))
        car2 = np.empty(3, dtype=np.float64)
        car3 = np.empty(3, dtype=np.float64)
        
        summ = 0
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
        
        shifts27 = np.array(list(itertools.product([-1, 0, 1], repeat=3)), dtype=np.int64)
        
        basis = np.empty(3, dtype=np.int64)
        triplet = np.empty(3, dtype=np.int64)
        triplet_perm = np.empty(3, dtype=np.int64)
        triplet_sym = np.empty(3, dtype=np.int64)
        shift2all = np.empty((3, 27), dtype=np.int64)
        shift3all = np.empty((3, 27), dtype=np.int64)
        equilist = np.empty((3, nsym * 6), dtype=np.int64)
        coeffi = np.empty((6 * nsym * 27, 27), dtype=np.float64)
        id_equi = self.symops.map_supercell(self.sposcar)
        ind_cell, ind_species = _id2ind(ngrid, natoms)
        
        rot = np.empty((6, nsym, 27, 27), dtype=np.float64)
        for iperm in range(6):
            for isym in range(nsym):
                for ibasisprime in range(3):
                    for jbasisprime in range(3):
                        for kbasisprime in range(3):
                            indexijkprime = (ibasisprime * 3 + jbasisprime) * 3 + kbasisprime
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
                                            orth[ibasisprime, ibasispermut, isym] *
                                            orth[jbasisprime, jbasispermut, isym] *
                                            orth[kbasisprime, kbasispermut, isym])
        rot2 = rot.copy()
        nonzero = np.zeros((6, nsym, 27), dtype=np.int64)
        for iperm in range(6):
            for isym in range(nsym):
                for indexijkprime in range(27):
                    rot2[iperm, isym, indexijkprime, indexijkprime] -= 1.
                    for indexijk in range(27):
                        if abs(rot2[iperm, isym, indexijkprime, indexijk]) > 1e-12:
                            nonzero[iperm, isym, indexijkprime] = 1
                        else:
                            rot2[iperm, isym, indexijkprime, indexijk] = 0.
        
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
                        for ll in range(3):
                            car2[ll] = (shift2all[0, iaux] * lattvec[ll, 0] +
                                        shift2all[1, iaux] * lattvec[ll, 1] +
                                        shift2all[2, iaux] * lattvec[ll, 2] +
                                        coordall[ll, jj])
                        for jaux in range(n3equi):
                            for ll in range(3):
                                car3[ll] = (shift3all[0, jaux] * lattvec[ll, 0] +
                                            shift3all[1, jaux] * lattvec[ll, 1] +
                                            shift3all[2, jaux] * lattvec[ll, 2] +
                                            coordall[ll, kk])
                            d2_min = min(d2_min,
                                         (car3[0] - car2[0]) ** 2 +
                                         (car3[1] - car2[1]) ** 2 +
                                         (car3[2] - car2[2]) ** 2)
                    if d2_min >= frange2:
                        continue
                    summ += 1
                    triplet[0] = ii
                    triplet[1] = jj
                    triplet[2] = kk
                    if _triplet_in_list(triplet, v_alllist, self.nalllist):
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
                    v_nequi[self.nlist - 1] = 0
                    coeffi[:, :] = 0.
                    nnonzero = 0
                    for iperm in range(6):
                        triplet_perm[0] = triplet[permutations[iperm, 0]]
                        triplet_perm[1] = triplet[permutations[iperm, 1]]
                        triplet_perm[2] = triplet[permutations[iperm, 2]]
                        for isym in range(nsym):
                            triplet_sym[0] = id_equi[isym, triplet_perm[0]]
                            triplet_sym[1] = id_equi[isym, triplet_perm[1]]
                            triplet_sym[2] = id_equi[isym, triplet_perm[2]]
                            for ll in range(3):
                                vec1[ll] = ind_cell[ll, id_equi[isym, triplet_perm[0]]]
                                vec2[ll] = ind_cell[ll, id_equi[isym, triplet_perm[1]]]
                                vec3[ll] = ind_cell[ll, id_equi[isym, triplet_perm[2]]]
                            if not (vec1[0] == 0 and vec1[1] == 0 and vec1[2] == 0):
                                for ll in range(3):
                                    vec3[ll] = (vec3[ll] - vec1[ll]) % ngrid[ll]
                                    vec2[ll] = (vec2[ll] - vec1[ll]) % ngrid[ll]
                                    vec1[ll] = 0
                                ispecies1 = ind_species[id_equi[isym, triplet_perm[0]]]
                                ispecies2 = ind_species[id_equi[isym, triplet_perm[1]]]
                                ispecies3 = ind_species[id_equi[isym, triplet_perm[2]]]
                                triplet_sym[0] = _ind2id(vec1, ispecies1, ngrid, natoms)
                                triplet_sym[1] = _ind2id(vec2, ispecies2, ngrid, natoms)
                                triplet_sym[2] = _ind2id(vec3, ispecies3, ngrid, natoms)
                            if (iperm == 0 and isym == 0) or not (
                                    _triplets_are_equal(triplet_sym, triplet) or
                                    _triplet_in_list(triplet_sym, equilist, v_nequi[self.nlist - 1])):
                                v_nequi[self.nlist - 1] += 1
                                for ll in range(3):
                                    equilist[ll, v_nequi[self.nlist - 1] - 1] = triplet_sym[ll]
                                    v_allequilist[ll, v_nequi[self.nlist - 1] - 1, self.nlist - 1] = triplet_sym[ll]
                                self.nalllist += 1
                                if self.nalllist == self.allallocsize:
                                    self._expandalllist()
                                    v_alllist = self.alllist
                                for ll in range(3):
                                    v_alllist[ll, self.nalllist - 1] = triplet_sym[ll]
                                for iaux in range(27):
                                    for jaux in range(27):
                                        v_transformation[iaux, jaux, v_nequi[self.nlist - 1] - 1, self.nlist - 1] = rot[iperm, isym, iaux, jaux]
                            if _triplets_are_equal(triplet_sym, triplet):
                                for indexijkprime in range(27):
                                    if nonzero[iperm, isym, indexijkprime]:
                                        for ll in range(27):
                                            coeffi[nnonzero, ll] = rot2[iperm, isym, indexijkprime, ll]
                                        nnonzero += 1
                    coeffi_reduced = np.zeros((max(nnonzero, 27), 27), dtype=np.float64)
                    if nnonzero > 0:
                        coeffi_reduced[:nnonzero, :] = coeffi[:nnonzero, :]
                    b, independent = gaussian(coeffi_reduced)
                    for iaux in range(27):
                        for jaux in range(27):
                            v_transformationaux[iaux, jaux, self.nlist - 1] = b[iaux, jaux]
                    v_nindependentbasis[self.nlist - 1] = independent.shape[0]
                    for ll in range(independent.shape[0]):
                        v_independentbasis[ll, self.nlist - 1] = independent[ll]
        v_transformationarray[:, :, :, :] = 0.
        for ii in range(self.nlist):
            n_indep = v_nindependentbasis[ii]
            for jj in range(v_nequi[ii]):
                trans_sub = v_transformation[:, :, jj, ii]
                aux_sub = v_transformationaux[:, :n_indep, ii]
                v_transformationarray[:, :n_indep, jj, ii] = trans_sub @ aux_sub
                
                mask = np.abs(v_transformationarray[:, :n_indep, jj, ii]) < 1e-12
                v_transformationarray[:, :n_indep, jj, ii][mask] = 0.
    
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
                list6.append((ll, self.llist[0, ii],
                              mm, self.llist[1, ii],
                              nn, self.llist[2, ii]))
        nruter = []
        seen = set()
        for i in list6:
            fournumbers = (i[1], i[3], i[0], i[2])
            if fournumbers not in seen:
                seen.add(fournumbers)
                nruter.append(fournumbers)
        return nruter


EPS = 1e-10

@njit
def gaussian(a):
    """
    Specialized version of Gaussian elimination.
    """
    row = a.shape[0]
    col = a.shape[1]
    
    dependent = np.empty(col, dtype=np.int64)
    independent = np.empty(col, dtype=np.int64)
    b = np.zeros((col, col), dtype=np.float64)
    
    irow = 0
    ndependent = 0
    nindependent = 0
    for k in range(min(row, col)):
        for i in range(row):
            if abs(a[i, k]) < EPS:
                a[i, k] = 0.
        for i in range(irow + 1, row):
            if abs(a[i, k]) - abs(a[irow, k]) > EPS:
                for j in range(k, col):
                    tmp = a[irow, j]
                    a[irow, j] = a[i, j]
                    a[i, j] = tmp
        if abs(a[irow, k]) > EPS:
            dependent[ndependent] = k
            ndependent += 1
            for j in range(col - 1, k, -1):
                a[irow, j] /= a[irow, k]
            a[irow, k] = 1.
            for i in range(row):
                if i == irow:
                    continue
                for j in range(col - 1, k, -1):
                    a[i, j] -= a[i, k] * a[irow, j] / a[irow, k]
                a[i, k] = 0.
            if irow < row - 1:
                irow += 1
        else:
            independent[nindependent] = k
            nindependent += 1
    for j in range(nindependent):
        for i in range(ndependent):
            b[dependent[i], j] = -a[i, independent[j]]
        b[independent[j], j] = 1.
    return (b, independent[:nindependent])
