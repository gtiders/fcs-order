from numba import njit, typed, types
import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import itertools
from mlfcs.utils.symmetry import SymmetryOperations
permutations = np.array([
    [0,1,2,3], [0,2,1,3], [0,1,3,2], [0,3,1,2],
    [0,3,2,1], [0,2,3,1], [1,0,2,3], [1,0,3,2],
    [1,2,0,3], [1,2,3,0], [1,3,0,2], [1,3,2,0],
    [2,0,1,3], [2,0,3,1], [2,1,0,3], [2,1,3,0],
    [2,3,0,1], [2,3,1,0], [3,0,1,2], [3,0,2,1],
    [3,1,0,2], [3,1,2,0], [3,2,0,1], [3,2,1,0]], dtype=np.int8)


class IFCMap:
    """
    Efficient map for storing 8-index IFC values using Numba typed.Dict.
    
    Keys are packed into a single int64 for fast lookup:
    key = (a << 56) | (b << 48) | (c << 40) | (d << 32) | 
          (e << 24) | (f << 16) | (g << 8) | h
    """
    
    def __init__(self):
        self._data = typed.Dict.empty(types.int64, types.float64)
    
    @staticmethod
    def _make_key(dir0, dir1, dir2, dir3, atom0, atom1, atom2, atom3):
        """
        Pack 4 direction indices (2 bits each) and 4 atom indices (14 bits each) into int64.
        
        Direction indices: range 0-2 (only need 2 bits)
        Atom indices: range 0-16383 (14 bits, supports supercells up to 16384 atoms)
        """
        return (np.int64(dir0) << 62) | (np.int64(dir1) << 60) | \
               (np.int64(dir2) << 58) | (np.int64(dir3) << 56) | \
               (np.int64(atom0) << 42) | (np.int64(atom1) << 28) | \
               (np.int64(atom2) << 14) | np.int64(atom3)
    
    @staticmethod
    def _unpack_key(k):
        dir0 = (k >> 62) & 0x3
        dir1 = (k >> 60) & 0x3
        dir2 = (k >> 58) & 0x3
        dir3 = (k >> 56) & 0x3
        atom0 = (k >> 42) & 0x3FFF
        atom1 = (k >> 28) & 0x3FFF
        atom2 = (k >> 14) & 0x3FFF
        atom3 = k & 0x3FFF
        return (dir0, dir1, dir2, dir3, atom0, atom1, atom2, atom3)
    
    def set_item(self, dir0, dir1, dir2, dir3, atom0, atom1, atom2, atom3, val):
        self._data[self._make_key(dir0, dir1, dir2, dir3, atom0, atom1, atom2, atom3)] = val
    
    def get_item(self, dir0, dir1, dir2, dir3, atom0, atom1, atom2, atom3):
        k = self._make_key(dir0, dir1, dir2, dir3, atom0, atom1, atom2, atom3)
        if k in self._data:
            return self._data[k]
        return 0.0
    
    def add_item(self, dir0, dir1, dir2, dir3, atom0, atom1, atom2, atom3, val):
        k = self._make_key(dir0, dir1, dir2, dir3, atom0, atom1, atom2, atom3)
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
def _quartet_in_list(quartet, llist, nlist):
    """
    Return True if quartet is found in llist[:,:nlist]. The first dimension
    of list must have a length of 4.
    """
    for i in range(nlist):
        if (quartet[0] == llist[0, i] and quartet[1] == llist[1, i] and
            quartet[2] == llist[2, i] and quartet[3] == llist[3, i]):
            return True
    return False

@njit
def _quartets_are_equal(quartet1, quartet2):
    """
    Return True if two quartets are equal and False otherwise.
    """
    for i in range(4):
        if quartet1[i] != quartet2[i]:
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
    Build sparse matrix coefficients with exact preallocation.
    
    Returns (i, j, v) arrays for COO sparse matrix.
    """
    total_elements = 0
    for ii in range(natoms):
        for jj in range(ntot):
            for kk in range(ntot):
                for bb in range(ntot):
                    ix = vind1[ii, jj, kk, bb]
                    if ix >= 0:
                        total_elements += 81 * (naccumindependent[ix + 1] - naccumindependent[ix])
    
    i_arr = np.empty(total_elements, dtype=np.int64)
    j_arr = np.empty(total_elements, dtype=np.int64)
    v_arr = np.empty(total_elements, dtype=np.float64)
    
    idx = 0
    for ii in range(natoms):
        for jj in range(ntot):
            base_colindex = ii * (ntot * 81) + jj * 81
            for kk in range(ntot):
                for bb in range(ntot):
                    ix = vind1[ii, jj, kk, bb]
                    if ix < 0:
                        continue
                    start_ss = naccumindependent[ix]
                    num_ss = naccumindependent[ix + 1] - start_ss
                    v2 = vind2[ii, jj, kk, bb]
                    
                    for tribasisindex in range(81):
                        colindex = base_colindex + tribasisindex
                        for tt in range(num_ss):
                            i_arr[idx] = start_ss + tt
                            j_arr[idx] = colindex
                            v_arr[idx] = vtrans[tribasisindex, tt, v2, ix]
                            idx += 1
    
    return i_arr, j_arr, v_arr


def reconstruct_ifcs(phipart,wedge,list4,poscar,sposcar):
    """
    Recover the full fourth-order IFC set from the irreducible set of
    force constants and the information contained in a wedge object.
    """

    nlist=wedge.nlist
    natoms=len(poscar["types"])
    ntot=len(sposcar["types"])
    vnruter=IFCMap()
    
    naccumindependent=np.insert(np.cumsum(
        wedge.nindependentbasis[:nlist],dtype=np.int64),0,
        np.zeros(1,dtype=np.int64))
    ntotalindependent=naccumindependent[-1]
    vphipart=phipart
    nlist6=len(list4)
    for ii in range(nlist6):
        e0,e1,e2,e3,e4,e5=list4[ii]
        for k in range(3):
            for l in range(ntot):
                val = vphipart[k, ii, l]
                if val != 0.0:
                    vnruter.set_item(e3, e4, e5, k, e0, e1, e2, l, val)
    philist=[]
    for ii in range(nlist):
        for jj in range(wedge.nindependentbasis[ii]):
            kk=wedge.independentbasis[jj,ii]//27
            ll=wedge.independentbasis[jj,ii]%27//9
            mm=wedge.independentbasis[jj,ii]%9//3
            nn=wedge.independentbasis[jj,ii]%3
            val = vnruter.get_item(kk,ll,mm,nn,
                          wedge.llist[0,ii],
                          wedge.llist[1,ii],
                          wedge.llist[2,ii],
                          wedge.llist[3,ii])
            philist.append(val)
    aphilist=np.array(philist,dtype=np.double)
    vind1=-np.ones((natoms,ntot,ntot,ntot),dtype=np.int64)
    vind2=-np.ones((natoms,ntot,ntot,ntot),dtype=np.int64)
    vequilist=wedge.allequilist
    for ii in range(nlist):
        for jj in range(wedge.nequi[ii]):
            vind1[vequilist[0,jj,ii],vequilist[1,jj,ii],vequilist[2,jj,ii],vequilist[3,jj,ii]]=ii
            vind2[vequilist[0,jj,ii],vequilist[1,jj,ii],vequilist[2,jj,ii],vequilist[3,jj,ii]]=jj 

    vtrans=wedge.transformationarray

    nrows=ntotalindependent
    ncols=natoms*ntot*81
    print("- Storing the coefficients in a sparse matrix")
    i_arr, j_arr, v_arr = _build_sparse_coefs(
        natoms, ntot, vind1, vind2, naccumindependent, vtrans)
    print("- \t Density: {0:.2g}%".format(100.*len(i_arr)/float(nrows*ncols)))
    aaa=sp.sparse.coo_matrix((v_arr,(i_arr,j_arr)),(nrows,ncols)).tocsr()
    D=sp.sparse.spdiags(aphilist,[0,],aphilist.size,aphilist.size,
                           format="csr")
    bbs=D.dot(aaa)
    ones=np.ones_like(aphilist)
    multiplier=-sp.sparse.linalg.lsqr(bbs,ones)[0]
    compensation=D.dot(bbs.dot(multiplier))

    aphilist+=compensation

    vnruter.clear()
    for ii in range(nlist):
        for jj in range(wedge.nequi[ii]):
            for ll in range(3):
                for mm in range(3):
                    for nn in range(3):
                        for aa1 in range(3):
                            tribasisindex=((ll*3+mm)*3+nn)*3+aa1
                            for ix in range(wedge.nindependentbasis[ii]):
                                val = wedge.transformationarray[
                                        tribasisindex,ix,jj,ii]*aphilist[
                                            naccumindependent[ii]+ix]
                                if val != 0.0:
                                    vnruter.add_item(ll,mm,nn,aa1,
                                            vequilist[0,jj,ii],
                                            vequilist[1,jj,ii],
                                            vequilist[2,jj,ii],
                                            vequilist[3,jj,ii], val)

    return vnruter


class Wedge:
    """
    Objects of this class allow the user to extract irreducible sets
    of force constants and to reconstruct the full Fourth-order IFC
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
            self.allocsize = 32
            self.nequi = np.empty(self.allocsize, dtype=np.int64)
            self.allequilist = np.empty((4, 24 * self.symops.nsyms, self.allocsize), dtype=np.int64)
            self.transformationarray = np.empty((81, 81, 24 * self.symops.nsyms, self.allocsize), dtype=np.float64)
            self.transformation = np.empty((81, 81, 24 * self.symops.nsyms, self.allocsize), dtype=np.float64)
            self.transformationaux = np.empty((81, 81, self.allocsize), dtype=np.float64)
            self.nindependentbasis = np.empty(self.allocsize, dtype=np.int64)
            self.independentbasis = np.empty((81, self.allocsize), dtype=np.int64)
            self.llist = np.empty((4, self.allocsize), dtype=np.int64)
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
            self.alllist = np.empty((4, self.allallocsize), dtype=np.int64)
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
        vec4 = np.empty(3, dtype=np.int64)
        
        lattvec = self.sposcar["lattvec"]
        coordall = np.dot(lattvec, self.sposcar["positions"])
        orth = np.transpose(self.symops.crotations, (1, 2, 0))
        car2 = np.empty(3, dtype=np.float64)
        car3 = np.empty(3, dtype=np.float64)
        car4 = np.empty(3, dtype=np.float64)
        
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
        
        basis = np.empty(4, dtype=np.int64)
        quartet = np.empty(4, dtype=np.int64)
        quartet_perm = np.empty(4, dtype=np.int64)
        quartet_sym = np.empty(4, dtype=np.int64)
        shift2all = np.empty((3, 27), dtype=np.int64)
        shift3all = np.empty((3, 27), dtype=np.int64)
        shift4all = np.empty((3, 27), dtype=np.int64)
        equilist = np.empty((4, nsym * 24), dtype=np.int64)
        coeffi = np.empty((24 * nsym * 81, 81), dtype=np.float64)
        id_equi = self.symops.map_supercell(self.sposcar)
        ind_cell, ind_species = _id2ind(ngrid, natoms)
        
        rot = np.empty((24, nsym, 81, 81), dtype=np.float64)
        for iperm in range(24):
            for isym in range(nsym):
                for ibasisprime in range(3):
                    for jbasisprime in range(3):
                        for kbasisprime in range(3):
                            for lbasisprime in range(3):
                                indexijklprime = ((ibasisprime * 3 + jbasisprime) * 3 + kbasisprime) * 3 + lbasisprime
                                for ibasis in range(3):
                                    basis[0] = ibasis
                                    for jbasis in range(3):
                                        basis[1] = jbasis
                                        for kbasis in range(3):
                                            basis[2] = kbasis
                                            for lbasis in range(3):
                                                basis[3] = lbasis
                                                indexijkl = ibasis * 27 + jbasis * 9 + kbasis * 3 + lbasis
                                                ibasispermut = basis[permutations[iperm, 0]]
                                                jbasispermut = basis[permutations[iperm, 1]]
                                                kbasispermut = basis[permutations[iperm, 2]]
                                                lbasispermut = basis[permutations[iperm, 3]]
                                                rot[iperm, isym, indexijklprime, indexijkl] = (
                                                    orth[ibasisprime, ibasispermut, isym] *
                                                    orth[jbasisprime, jbasispermut, isym] *
                                                    orth[kbasisprime, kbasispermut, isym] *
                                                    orth[lbasisprime, lbasispermut, isym])
        rot2 = rot.copy()
        nonzero = np.zeros((24, nsym, 81), dtype=np.int64)
        for iperm in range(24):
            for isym in range(nsym):
                for indexijklprime in range(81):
                    rot2[iperm, isym, indexijklprime, indexijklprime] -= 1.
                    for indexijkl in range(81):
                        if abs(rot2[iperm, isym, indexijklprime, indexijkl]) > 1e-12:
                            nonzero[iperm, isym, indexijklprime] = 1
                        else:
                            rot2[iperm, isym, indexijklprime, indexijkl] = 0.
        
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
                                for kaux in range(n4equi):
                                    for ll in range(3):
                                        car4[ll] = (shift4all[0, kaux] * lattvec[ll, 0] +
                                                    shift4all[1, kaux] * lattvec[ll, 1] +
                                                    shift4all[2, kaux] * lattvec[ll, 2] +
                                                    coordall[ll, mm])
                                d2_min3 = min(d2_min3,
                                              (car4[0] - car3[0]) ** 2 +
                                              (car4[1] - car3[1]) ** 2 +
                                              (car4[2] - car3[2]) ** 2)
                            if d2_min3 > frange2:
                                continue
                            d2_min2 = min(d2_min2,
                                          (car4[0] - car2[0]) ** 2 +
                                          (car4[1] - car2[1]) ** 2 +
                                          (car4[2] - car2[2]) ** 2)
                            d2_min1 = min(d2_min1,
                                          (car3[0] - car2[0]) ** 2 +
                                          (car3[1] - car2[1]) ** 2 +
                                          (car3[2] - car2[2]) ** 2)
                        if d2_min1 >= frange2:
                            continue
                        if d2_min2 >= frange2:
                            continue
                        summ += 1
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
                        coeffi[:, :] = 0.
                        nnonzero = 0
                        for iperm in range(24):
                            quartet_perm[0] = quartet[permutations[iperm, 0]]
                            quartet_perm[1] = quartet[permutations[iperm, 1]]
                            quartet_perm[2] = quartet[permutations[iperm, 2]]
                            quartet_perm[3] = quartet[permutations[iperm, 3]]
                            for isym in range(nsym):
                                quartet_sym[0] = id_equi[isym, quartet_perm[0]]
                                quartet_sym[1] = id_equi[isym, quartet_perm[1]]
                                quartet_sym[2] = id_equi[isym, quartet_perm[2]]
                                quartet_sym[3] = id_equi[isym, quartet_perm[3]]
                                for ll in range(3):
                                    vec1[ll] = ind_cell[ll, id_equi[isym, quartet_perm[0]]]
                                    vec2[ll] = ind_cell[ll, id_equi[isym, quartet_perm[1]]]
                                    vec3[ll] = ind_cell[ll, id_equi[isym, quartet_perm[2]]]
                                    vec4[ll] = ind_cell[ll, id_equi[isym, quartet_perm[3]]]
                                if not (vec1[0] == 0 and vec1[1] == 0 and vec1[2] == 0):
                                    for ll in range(3):
                                        vec4[ll] = (vec4[ll] - vec1[ll]) % ngrid[ll]
                                        vec3[ll] = (vec3[ll] - vec1[ll]) % ngrid[ll]
                                        vec2[ll] = (vec2[ll] - vec1[ll]) % ngrid[ll]
                                        vec1[ll] = 0
                                    ispecies1 = ind_species[id_equi[isym, quartet_perm[0]]]
                                    ispecies2 = ind_species[id_equi[isym, quartet_perm[1]]]
                                    ispecies3 = ind_species[id_equi[isym, quartet_perm[2]]]
                                    ispecies4 = ind_species[id_equi[isym, quartet_perm[3]]]
                                    quartet_sym[0] = _ind2id(vec1, ispecies1, ngrid, natoms)
                                    quartet_sym[1] = _ind2id(vec2, ispecies2, ngrid, natoms)
                                    quartet_sym[2] = _ind2id(vec3, ispecies3, ngrid, natoms)
                                    quartet_sym[3] = _ind2id(vec4, ispecies4, ngrid, natoms)
                                if (iperm == 0 and isym == 0) or not (
                                        _quartets_are_equal(quartet_sym, quartet) or
                                        _quartet_in_list(quartet_sym, equilist, v_nequi[self.nlist - 1])):
                                    v_nequi[self.nlist - 1] += 1
                                    for ll in range(4):
                                        equilist[ll, v_nequi[self.nlist - 1] - 1] = quartet_sym[ll]
                                        v_allequilist[ll, v_nequi[self.nlist - 1] - 1, self.nlist - 1] = quartet_sym[ll]
                                    self.nalllist += 1
                                    if self.nalllist == self.allallocsize:
                                        self._expandalllist()
                                        v_alllist = self.alllist
                                    for ll in range(4):
                                        v_alllist[ll, self.nalllist - 1] = quartet_sym[ll]
                                    for iaux in range(81):
                                        for jaux in range(81):
                                            v_transformation[iaux, jaux, v_nequi[self.nlist - 1] - 1, self.nlist - 1] = rot[iperm, isym, iaux, jaux]
                                if _quartets_are_equal(quartet_sym, quartet):
                                    for indexijklprime in range(81):
                                        if nonzero[iperm, isym, indexijklprime]:
                                            for ll in range(81):
                                                coeffi[nnonzero, ll] = rot2[iperm, isym, indexijklprime, ll]
                                            nnonzero += 1
                        coeffi_reduced = np.zeros((max(nnonzero, 81), 81), dtype=np.float64)
                        if nnonzero > 0:
                            coeffi_reduced[:nnonzero, :] = coeffi[:nnonzero, :]
                        b, independent = gaussian(coeffi_reduced)
                        for iaux in range(81):
                            for jaux in range(81):
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
                
                mask = np.abs(v_transformationarray[:, :n_indep, jj, ii]) < 1e-15
                v_transformationarray[:, :n_indep, jj, ii][mask] = 0.
    
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
                list6.append((kk, self.llist[0, ii], ll, self.llist[1, ii],
                              mm, self.llist[2, ii], nn, self.llist[3, ii]))
        nruter = []
        seen = set()
        for i in list6:
            fournumbers = (i[1], i[3], i[5], i[0], i[2], i[4])
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
