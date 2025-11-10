from libc.math cimport fabs

import numpy as np
import scipy as sp
import sparse
from rich.progress import track
import typer

cimport cython
cimport numpy as np
np.import_array()

# Permutations of 3 elements listed in the same order as in the old
# Fortran code.
cdef int[:,:] permutations=np.array([
    [0,1,2],
    [1,0,2],
    [2,1,0],
    [0,2,1],
    [1,2,0],
    [2,0,1]],dtype=np.intc)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int _ind2id(int[:] icell,int ispecies,int[:] ngrid,int nspecies):
    """
    Merge a set of cell+atom indices into a single index into a supercell.
    """
    return (icell[0]+(icell[1]+icell[2]*ngrid[1])*ngrid[0])*nspecies+ispecies


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint _triplet_in_list(int[:] triplet,int[:,:] llist,int nlist):
    """
    Return True if triplet is found in llist[:,:nlist]. The first dimension
    of list must have a length of 3.
    """
    # This works fine for the nlist ranges we have to deal with, but
    # using std::vector and std::push_heap would be a better general
    # solution.
    cdef int i

    for i in range(nlist):
        if (triplet[0]==llist[0,i] and
            triplet[1]==llist[1,i] and triplet[2]==llist[2,i]):
            return True
    return False


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint _triplets_are_equal(int[:] triplet1,int[:] triplet2):
    """
    Return True if two triplets are equal and False otherwise.
    """
    cdef int i

    for i in range(3):
        if triplet1[i]!=triplet2[i]:
            return False
    return True


@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple _id2ind(int[:] ngrid,int nspecies):
    """
    Create a map from supercell indices to cell+atom indices.
    """
    cdef int ii,ntot,tmp
    cdef int[:,:] icell
    cdef int[:] ispecies

    cdef np.ndarray np_icell,np_ispecies

    ntot=ngrid[0]*ngrid[1]*ngrid[2]*nspecies
    np_icell=np.empty((3,ntot),dtype=np.intc)
    np_ispecies=np.empty(ntot,dtype=np.intc)
    icell=np_icell
    ispecies=np_ispecies
    for ii in range(ntot):
        tmp,ispecies[ii]=divmod(ii,nspecies)
        tmp,icell[0,ii]=divmod(tmp,ngrid[0])
        icell[2,ii],icell[1,ii]=divmod(tmp,ngrid[1])
    return (np_icell,np_ispecies)

from .symmetry import SymmetryOperations


@cython.boundscheck(False)
def reconstruct_ifcs(phipart,wedge,list4,poscar,sposcar,is_sparse):
    """
    Recover the full anharmonic IFC set from the irreducible set of
    force constants and the information contained in a wedge object.
    """
    cdef int ii,jj,ll,mm,nn,kk,ss,tt,ix,e0,e1,e2,e3
    cdef int nlist,nlist4,natoms,ntot
    cdef int ntotalindependent,tribasisindex,colindex,nrows,ncols
    cdef int[:] naccumindependent
    cdef int[:,:,:] vind1
    cdef int[:,:,:] vind2
    cdef int[:,:,:] vequilist
    cdef double[:] aphilist
    cdef double[:,:] vaa
    cdef double[:,:,:] vphipart
    
    nlist=wedge.nlist
    natoms=len(poscar["types"])
    ntot=len(sposcar["types"])
    if is_sparse:
        typer.print("using sparse method with dok sparse matrix !")
        vnruter=sparse.zeros((3, 3, 3, natoms, ntot, ntot), format="dok")
    else:
        vnruter = np.zeros((3, 3, 3, natoms, ntot, ntot))
    naccumindependent=np.insert(np.cumsum(
        wedge.nindependentbasis[:nlist],dtype=np.intc),0,
        np.zeros(1,dtype=np.intc))
    ntotalindependent=naccumindependent[-1]
    vphipart=phipart
    nlist4=len(list4)
    for ii in track(range(nlist4), description="Processing list4"):
        e0,e1,e2,e3=list4[ii]
        vnruter[e2,e3,:,e0,e1,:]=vphipart[:,ii,:]
    philist=[]
    for ii in track(range(nlist), description="Building philist"):
        for jj in range(wedge.nindependentbasis[ii]):
            ll=wedge.independentbasis[jj,ii]//9
            mm=(wedge.independentbasis[jj,ii]%9)//3
            nn=wedge.independentbasis[jj,ii]%3
            philist.append(vnruter[ll,mm,nn,
                                  wedge.llist[0,ii],
                                  wedge.llist[1,ii],
                                  wedge.llist[2,ii]])
    aphilist=np.array(philist,dtype=np.double)
    vind1=-np.ones((natoms,ntot,ntot),dtype=np.intc)
    vind2=-np.ones((natoms,ntot,ntot),dtype=np.intc)
    vequilist=wedge.allequilist
    for ii in range(nlist):
        for jj in range(wedge.nequi[ii]):
            vind1[vequilist[0,jj,ii],
                  vequilist[1,jj,ii],
                  vequilist[2,jj,ii]]=ii
            vind2[vequilist[0,jj,ii],
                  vequilist[1,jj,ii],
                  vequilist[2,jj,ii]]=jj

    vtrans=wedge.transformationarray

    nrows=ntotalindependent
    ncols=natoms*ntot*27

    typer.print("- Storing the coefficients in a sparse matrix")
    i=[]
    j=[]
    v=[]
    colindex=0
    for ii in range(natoms):
        for jj in range(ntot):
            tribasisindex=0
            for ll in range(3):
                for mm in range(3):
                    for nn in range(3):
                        for kk in range(ntot):
                            for ix in range(nlist):
                                if vind1[ii,jj,kk]==ix:
                                    for ss in range(naccumindependent[ix],
                                                        naccumindependent[ix+1]):
                                        tt=ss-naccumindependent[ix]
                                        i.append(ss)
                                        j.append(colindex)
                                        v.append(vtrans[tribasisindex,tt,
                                                        vind2[ii,jj,kk],ix])
                        tribasisindex+=1
                        colindex+=1
    typer.print("- \t Density: {0:.2g}%".format(100.*len(i)/float(nrows*ncols)))
    aa=sp.sparse.coo_matrix((v,(i,j)),(nrows,ncols)).tocsr()
    D=sp.sparse.spdiags(aphilist,[0,],aphilist.size,aphilist.size,
                           format="csr")
    bbs=D.dot(aa)
    ones=np.ones_like(aphilist)
    multiplier=-sp.sparse.linalg.lsqr(bbs,ones)[0]
    compensation=D.dot(bbs.dot(multiplier))

    aphilist+=compensation

    if is_sparse:
        # Two-pass COO construction: count -> allocate -> fill -> build once
        nnz = 0
        EPSVAL = 1e-15
        for ii in range(nlist):
            nind = wedge.nindependentbasis[ii]
            ne = wedge.nequi[ii]
            if nind==0 or ne==0:
                continue
            offset = naccumindependent[ii]
            phi = aphilist[offset:offset+nind]
            T = wedge.transformationarray[:, :nind, :ne, ii]  # (27, nind, ne)
            out = np.tensordot(T, phi, axes=([1],[0]))        # (27, ne)
            for jj in range(ne):
                block = out[:, jj].reshape(3,3,3)
                # count non-zeros in this 3x3x3 block
                for ll in range(3):
                    for mm in range(3):
                        for nn in range(3):
                            if block[ll,mm,nn]!=0.0 and abs(block[ll,mm,nn])>EPSVAL:
                                nnz += 1
        # allocate and fill
        coords = np.empty((6, nnz), dtype=np.intp)
        data = np.empty(nnz, dtype=np.double)
        p = 0
        for ii in range(nlist):
            nind = wedge.nindependentbasis[ii]
            ne = wedge.nequi[ii]
            if nind==0 or ne==0:
                continue
            offset = naccumindependent[ii]
            phi = aphilist[offset:offset+nind]
            T = wedge.transformationarray[:, :nind, :ne, ii]
            out = np.tensordot(T, phi, axes=([1],[0]))
            for jj in range(ne):
                e0 = vequilist[0,jj,ii]
                e1 = vequilist[1,jj,ii]
                e2 = vequilist[2,jj,ii]
                block = out[:, jj].reshape(3,3,3)
                for ll in range(3):
                    for mm in range(3):
                        for nn in range(3):
                            val = block[ll,mm,nn]
                            if val!=0.0 and abs(val)>EPSVAL:
                                coords[0,p] = ll
                                coords[1,p] = mm
                                coords[2,p] = nn
                                coords[3,p] = e0
                                coords[4,p] = e1
                                coords[5,p] = e2
                                data[p] = val
                                p += 1
        vnruter = sparse.COO(coords, data, shape=(3,3,3,natoms,ntot,ntot))
    else:
        vnruter=np.zeros((3,3,3,natoms,ntot,ntot),dtype=np.double)

    # Vectorized rebuild of final (third-order) IFCs.
    # Rationale: For fixed (ii, jj), original code computes for each tribasisindex in [0..26]
    #   sum_{ix=0..nind-1} transformationarray[tribasisindex, ix, jj, ii] * aphilist[offset+ix].
    # This is a matrix-vector product (27 x nind) @ (nind,), so we can replace the nested
    # loops with a single BLAS-backed dot/tensordot without changing results.
    for ii in track(range(nlist), description="Building final IFCs"):
        nind = wedge.nindependentbasis[ii]
        ne = wedge.nequi[ii]
        if nind==0 or ne==0:
            continue
        offset = naccumindependent[ii]
        phi = aphilist[offset:offset+nind]
        # T has shape (27, nind, ne); each column along axis=2 corresponds to one jj equivalence.
        T = wedge.transformationarray[:, :nind, :ne, ii]
        # out has shape (27, ne); each column is the 27-length result for a given jj.
        out = np.tensordot(T, phi, axes=([1],[0]))
        for jj in range(ne):
            e0 = vequilist[0,jj,ii]
            e1 = vequilist[1,jj,ii]
            e2 = vequilist[2,jj,ii]
            block = out[:, jj].reshape(3,3,3)
            # Dense path writes directly; sparse has been constructed above via COO.
            if not is_sparse:
                for ll in range(3):
                    for mm in range(3):
                        for nn in range(3):
                            val = block[ll,mm,nn]
                            if val!=0.0:
                                vnruter[ll,mm,nn,e0,e1,e2] += val
    return vnruter


cdef class Wedge:
    """
    Objects of this class allow the user to extract irreducible sets
    of force constants and to reconstruct the full third-order IFC
    matrix from them.
    """
    cdef readonly object symops
    cdef readonly dict poscar,sposcar
    cdef int allocsize,allallocsize,nalllist
    cdef readonly int nlist
    cdef readonly np.ndarray nequi,llist,allequilist
    cdef readonly np.ndarray nindependentbasis,independentbasis
    cdef readonly np.ndarray transformationarray
    cdef np.ndarray alllist,transformation,transformationaux

    cdef int[:,:] nequis
    cdef int[:,:,:] shifts
    cdef double[:,:] dmin
    cdef readonly double frange

    def __cinit__(self,poscar,sposcar,symops,dmin,nequis,shifts,frange):
        """
        Build the object by computing all the relevant information about
        irreducible IFCs.
        """
        self.poscar=poscar
        self.sposcar=sposcar
        self.symops=symops
        self.dmin=dmin
        self.nequis=nequis
        self.shifts=shifts
        self.frange=frange

        self.allocsize=0
        self.allallocsize=0
        self._expandlist()
        self._expandalllist()

        self._reduce()

    cdef _expandlist(self):
        """
        Expand nequi, allequilist, transformationarray, transformation,
        transformationaux, nindependentbasis, independentbasis,
        and llist to accommodate more elements.
        """
        if self.allocsize==0:
            self.allocsize=16
            self.nequi=np.empty(self.allocsize,dtype=np.intc)
            self.allequilist=np.empty((3,6*self.symops.nsyms,
                                       self.allocsize),dtype=np.intc)
            self.transformationarray=np.empty((27,27,6*self.symops.nsyms,
                                               self.allocsize),dtype=np.double)
            self.transformation=np.empty((27,27,6*self.symops.nsyms,
                                               self.allocsize),dtype=np.double)
            self.transformationaux=np.empty((27,27,self.allocsize),
                                            dtype=np.double)
            self.nindependentbasis=np.empty(self.allocsize,dtype=np.intc)
            self.independentbasis=np.empty((27,self.allocsize),dtype=np.intc)
            self.llist=np.empty((3,self.allocsize),dtype=np.intc)
        else:
            self.allocsize<<=1
            self.nequi=np.concatenate((self.nequi,self.nequi),axis=-1)
            self.allequilist=np.concatenate((self.allequilist,self.allequilist),axis=-1)
            self.transformation=np.concatenate((self.transformation,self.transformation),
                                               axis=-1)
            self.transformationarray=np.concatenate((self.transformationarray,
                                                     self.transformationarray),axis=-1)
            self.transformationaux=np.concatenate((self.transformationaux,
                                                   self.transformationaux),axis=-1)
            self.nindependentbasis=np.concatenate((self.nindependentbasis,self.nindependentbasis),
                                                  axis=-1)
            self.independentbasis=np.concatenate((self.independentbasis,self.independentbasis),
                                                 axis=-1)
            self.llist=np.concatenate((self.llist,self.llist),axis=-1)

    cdef _expandalllist(self):
        """
        Expand alllist  to accommodate more elements.
        """
        if self.allallocsize==0:
            self.allallocsize=512
            self.alllist=np.empty((3,self.allallocsize),dtype=np.intc)
        else:
            self.allallocsize<<=1
            self.alllist=np.concatenate((self.alllist,self.alllist),axis=-1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _reduce(self):
        """
        C-level method that performs most of the actual work.
        """
        cdef int ngrid1,ngrid2,ngrid3,nsym,natoms,ntot,summ,nnonzero
        cdef int ii,jj,kk,ll,iaux,jaux
        cdef int ibasis,jbasis,kbasis,ibasisprime,jbasisprime,kbasisprime
        cdef int iperm,isym,indexijk,indexijkprime
        cdef int[:] ngrid,ind_species,vec1,vec2,vec3,independent
        cdef int[:] v_nequi,v_nindependentbasis
        cdef int[:] basis,triplet,triplet_perm,triplet_sym
        cdef int[:,:] v_llist,v_alllist,v_independentbasis
        cdef int[:,:] shifts27,shift2all,shift3all
        cdef int[:,:] equilist,id_equi,ind_cell
        cdef int[:,:,:] nonzero
        cdef int[:,:,:] v_allequilist
        cdef double dist,frange2
        cdef double[:] car2,car3
        cdef double[:,:] lattvec,coordall,b,coeffi,coeffi_reduced
        cdef double[:,:,:] orth
        cdef double[:,:,:] v_transformationaux
        cdef double[:,:,:,:] rot,rot2
        cdef double[:,:,:,:] v_transformationarray,v_transformation

        # Preliminary work: memory allocation and initialization.
        frange2=self.frange*self.frange

        ngrid1=self.sposcar["na"]
        ngrid2=self.sposcar["nb"]
        ngrid3=self.sposcar["nc"]
        ngrid=np.array([ngrid1,ngrid2,ngrid3],dtype=np.intc)
        nsym=self.symops.nsyms
        natoms=len(self.poscar["types"])
        ntot=len(self.sposcar["types"])
        vec1=np.empty(3,dtype=np.intc)
        vec2=np.empty(3,dtype=np.intc)
        vec3=np.empty(3,dtype=np.intc)

        lattvec=self.sposcar["lattvec"]
        coordall=np.dot(lattvec,self.sposcar["positions"])
        orth=np.transpose(self.symops.crotations,(1,2,0))
        car2=np.empty(3,dtype=np.double)
        car3=np.empty(3,dtype=np.double)

        summ=0
        self.nlist=0
        self.nalllist=0
        v_nequi=self.nequi
        v_allequilist=self.allequilist
        v_transformation=self.transformation
        v_transformationarray=self.transformationarray
        v_transformationaux=self.transformationaux
        v_nindependentbasis=self.nindependentbasis
        v_independentbasis=self.independentbasis
        v_llist=self.llist
        v_alllist=self.alllist

        iaux=0
        shifts27=np.empty((27,3),dtype=np.intc)
        for ii in range(-1,2):
            for jj in range(-1,2):
                for kk in range(-1,2):
                    shifts27[iaux,0]=ii
                    shifts27[iaux,1]=jj
                    shifts27[iaux,2]=kk
                    iaux+=1

        basis=np.empty(3,dtype=np.intc)
        triplet=np.empty(3,dtype=np.intc)
        triplet_perm=np.empty(3,dtype=np.intc)
        triplet_sym=np.empty(3,dtype=np.intc)
        shift2all=np.empty((3,27),dtype=np.intc)
        shift3all=np.empty((3,27),dtype=np.intc)
        equilist=np.empty((3,nsym*6),dtype=np.intc)
        coeffi=np.empty((6*nsym*27,27),dtype=np.double)
        id_equi=self.symops.map_supercell(self.sposcar)
        ind_cell,ind_species=_id2ind(ngrid,natoms)

        # Rotation matrices for third derivatives and related quantities.
        rot=np.empty((6,nsym,27,27),dtype=np.double)
        for iperm in range(6):
            for isym in range(nsym):
                for ibasisprime in range(3):
                    for jbasisprime in range(3):
                        for kbasisprime in range(3):
                            indexijkprime=(ibasisprime*3+jbasisprime)*3+kbasisprime
                            for ibasis in range(3):
                                basis[0]=ibasis
                                for jbasis in range(3):
                                    basis[1]=jbasis
                                    for kbasis in range(3):
                                        basis[2]=kbasis
                                        indexijk=ibasis*9+jbasis*3+kbasis
                                        ibasispermut=basis[permutations[iperm,0]]
                                        jbasispermut=basis[permutations[iperm,1]]
                                        kbasispermut=basis[permutations[iperm,2]]
                                        rot[iperm,isym,indexijkprime,indexijk]=(
                                            orth[ibasisprime,ibasispermut,isym]*
                                            orth[jbasisprime,jbasispermut,isym]*
                                            orth[kbasisprime,kbasispermut,isym])
        rot2=rot.copy()
        nonzero=np.zeros((6,nsym,27),dtype=np.intc)
        for iperm in range(6):
            for isym in range(nsym):
                for indexijkprime in range(27):
                    rot2[iperm,isym,indexijkprime,indexijkprime]-=1.
                    for indexijk in range(27):
                        if fabs(rot2[iperm,isym,indexijkprime,indexijk])>1e-12:
                            nonzero[iperm,isym,indexijkprime]=1
                        else:
                            rot2[iperm,isym,indexijkprime,indexijk]=0.

        # Scan all atom triplets (ii,jj,kk) in the supercell.
        for ii in range(natoms):
            for jj in range(ntot):
                dist=self.dmin[ii,jj]
                if dist>=self.frange:
                    continue
                n2equi=self.nequis[ii,jj]
                for kk in range(n2equi):
                    shift2all[:,kk]=shifts27[self.shifts[ii,jj,kk],:]
                for kk in range(ntot):
                    dist=self.dmin[ii,kk]
                    if dist>=self.frange:
                        continue
                    n3equi=self.nequis[ii,kk]
                    for ll in range(n3equi):
                        shift3all[:,ll]=shifts27[self.shifts[ii,kk,ll],:]
                    d2_min=np.inf
                    for iaux in range(n2equi):
                        for ll in range(3):
                            car2[ll]=(shift2all[0,iaux]*lattvec[ll,0]+
                                      shift2all[1,iaux]*lattvec[ll,1]+
                                      shift2all[2,iaux]*lattvec[ll,2]+
                                      coordall[ll,jj])
                        for jaux in range(n3equi):
                            for ll in range(3):
                                car3[ll]=(shift3all[0,jaux]*lattvec[ll,0]+
                                          shift3all[1,jaux]*lattvec[ll,1]+
                                          shift3all[2,jaux]*lattvec[ll,2]+
                                          coordall[ll,kk])
                        d2_min=min(d2_min,
                                   (car3[0]-car2[0])**2+
                                   (car3[1]-car2[1])**2+
                                   (car3[2]-car2[2])**2)
                    if d2_min>=frange2:
                        continue
                    # This point is only reached if there is a choice of periodic images of
                    # ii, jj and kk such that all pairs ii-jj, ii-kk and jj-kk are within
                    # the specified interaction range.
                    summ+=1
                    triplet[0]=ii
                    triplet[1]=jj
                    triplet[2]=kk
                    if _triplet_in_list(triplet,v_alllist,self.nalllist):
                        continue
                    # This point is only reached if the triplet is not
                    # equivalent to any of the triplets already considered,
                    # including permutations and symmetries.
                    self.nlist+=1
                    if self.nlist==self.allocsize:
                        self._expandlist()
                        v_nequi=self.nequi
                        v_allequilist=self.allequilist
                        v_transformation=self.transformation
                        v_transformationarray=self.transformationarray
                        v_transformationaux=self.transformationaux
                        v_nindependentbasis=self.nindependentbasis
                        v_independentbasis=self.independentbasis
                        v_llist=self.llist
                    v_llist[0,self.nlist-1]=ii
                    v_llist[1,self.nlist-1]=jj
                    v_llist[2,self.nlist-1]=kk
                    v_nequi[self.nlist-1]=0
                    coeffi[:,:]=0.
                    nnonzero=0
                    # Scan the six possible permutations of triplet (ii,jj,kk).
                    for iperm in range(6):
                        triplet_perm[0]=triplet[permutations[iperm,0]]
                        triplet_perm[1]=triplet[permutations[iperm,1]]
                        triplet_perm[2]=triplet[permutations[iperm,2]]
                        # Explore the effect of all symmetry operations on each of
                        # the permuted triplets.
                        for isym in range(nsym):
                            triplet_sym[0]=id_equi[isym,triplet_perm[0]]
                            triplet_sym[1]=id_equi[isym,triplet_perm[1]]
                            triplet_sym[2]=id_equi[isym,triplet_perm[2]]
                            for ll in range(3):
                                vec1[ll]=ind_cell[ll,id_equi[isym,triplet_perm[0]]]
                                vec2[ll]=ind_cell[ll,id_equi[isym,triplet_perm[1]]]
                                vec3[ll]=ind_cell[ll,id_equi[isym,triplet_perm[2]]]
                            # Choose a displaced version of triplet_sym chosen so that
                            # atom 0 is always in the first unit cell.
                            if not vec1[0]==vec1[1]==vec1[2]==0:
                                for ll in range(3):
                                    vec3[ll]=(vec3[ll]-vec1[ll])%ngrid[ll]
                                    vec2[ll]=(vec2[ll]-vec1[ll])%ngrid[ll]
                                    vec1[ll]=0
                                ispecies1=ind_species[id_equi[isym,triplet_perm[0]]]
                                ispecies2=ind_species[id_equi[isym,triplet_perm[1]]]
                                ispecies3=ind_species[id_equi[isym,triplet_perm[2]]]
                                triplet_sym[0]=_ind2id(vec1,ispecies1,ngrid,natoms)
                                triplet_sym[1]=_ind2id(vec2,ispecies2,ngrid,natoms)
                                triplet_sym[2]=_ind2id(vec3,ispecies3,ngrid,natoms)
                            # If the permutation+symmetry operation changes the triplet into
                            # an as-yet-unseen image, add it to the list of equivalent triplets
                            # and fill the transformation array accordingly.
                            if (iperm==0 and isym==0) or not (
                                    _triplets_are_equal(triplet_sym,triplet) or
                                    _triplet_in_list(triplet_sym,equilist,v_nequi[self.nlist-1])):
                                v_nequi[self.nlist-1]+=1
                                for ll in range(3):
                                    equilist[ll,v_nequi[self.nlist-1]-1]=triplet_sym[ll]
                                    v_allequilist[ll,v_nequi[self.nlist-1]-1,
                                                  self.nlist-1]=triplet_sym[ll]
                                self.nalllist+=1
                                if self.nalllist==self.allallocsize:
                                    self._expandalllist()
                                    v_alllist=self.alllist
                                for ll in range(3):
                                    v_alllist[ll,self.nalllist-1]=triplet_sym[ll]
                                for iaux in range(27):
                                    for jaux in range(27):
                                        v_transformation[iaux,jaux,v_nequi[self.nlist-1]-1,
                                                         self.nlist-1]=rot[iperm,isym,iaux,jaux]
                            # If the permutation+symmetry operation amounts to the identity,
                            # add a row to the coefficient matrix.
                            if _triplets_are_equal(triplet_sym,triplet):
                                for indexijkprime in range(27):
                                    if nonzero[iperm,isym,indexijkprime]:
                                        for ll in range(27):
                                            coeffi[nnonzero,ll]=rot2[iperm,isym,indexijkprime,ll]
                                        nnonzero+=1
                    coeffi_reduced=np.zeros((max(nnonzero,27),27),dtype=np.double)
                    for iaux in range(nnonzero):
                        for jaux in range(27):
                            coeffi_reduced[iaux,jaux]=coeffi[iaux,jaux]
                    # Obtain a set of independent IFCs for this triplet equivalence class.
                    b,independent=gaussian(coeffi_reduced)
                    for iaux in range(27):
                        for jaux in range(27):
                            v_transformationaux[iaux,jaux,self.nlist-1]=b[iaux,jaux]
                    v_nindependentbasis[self.nlist-1]=independent.shape[0]
                    for ll in range(independent.shape[0]):
                        v_independentbasis[ll,self.nlist-1]=independent[ll]
        v_transformationarray[:,:,:,:]=0.
        for ii in range(self.nlist):
            for jj in range(v_nequi[ii]):
                for kk in range(27):
                    for ll in range(v_nindependentbasis[ii]):
                        for iaux in range(27):
                            v_transformationarray[kk,ll,jj,ii]+=(
                                v_transformation[kk,iaux,jj,ii]*
                                v_transformationaux[iaux,ll,ii])
                for kk in range(27):
                    for ll in range(27):
                        if fabs(v_transformationarray[kk,ll,jj,ii])<1e-12:
                            v_transformationarray[kk,ll,jj,ii]=0.

    def build_list4(self):
        """
        Build a list of 4-uples from the results of the reduction.
        """
        cdef int ii,jj,ll,mm,nn
        cdef list list6,nruter

        list6=[]
        for ii in range(self.nlist):
            for jj in range(self.nindependentbasis[ii]):
                ll=self.independentbasis[jj,ii]//9
                mm=(self.independentbasis[jj,ii]%9)//3
                nn=self.independentbasis[jj,ii]%3
                list6.append((ll,self.llist[0,ii],
                        mm,self.llist[1,ii],
                        nn,self.llist[2,ii]))
        nruter=[]
        for i in list6:
            fournumbers=(i[1],i[3],i[0],i[2])
            if fournumbers not in nruter:
                nruter.append(fournumbers)
        return nruter


DEF EPS=1e-10
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef tuple gaussian(double[:,:] a):
    """
    Specialized version of Gaussian elimination.
    """
    cdef int i,j,k,irow
    cdef int row,col,ndependent,nindependent
    cdef double tmp
    cdef int[:] dependent,independent

    row=a.shape[0]
    col=a.shape[1]

    dependent=np.empty(col,dtype=np.intc)
    independent=np.empty(col,dtype=np.intc)
    b=np.zeros((col,col),dtype=np.double)

    irow=0
    ndependent=0
    nindependent=0
    for k in range(min(row,col)):
        for i in range(row):
            if fabs(a[i,k])<EPS:
                a[i,k]=0.
        for i in range(irow+1,row):
            if fabs(a[i,k])-fabs(a[irow,k])>EPS:
                for j in range(k,col):
                    tmp=a[irow,j]
                    a[irow,j]=a[i,j]
                    a[i,j]=tmp
        if fabs(a[irow,k])>EPS:
            dependent[ndependent]=k
            ndependent+=1
            for j in range(col-1,k,-1):
                a[irow,j]/=a[irow,k]
            a[irow,k]=1.
            for i in range(row):
                if i==irow:
                    continue
                for j in range(col-1,k,-1):
                    a[i,j]-=a[i,k]*a[irow,j]/a[irow,k]
                a[i,k]=0.
            if irow<row-1:
                irow+=1
        else:
            independent[nindependent]=k
            nindependent+=1
    for j in range(nindependent):
        for i in range(ndependent):
            b[dependent[i],j]=-a[i,independent[j]]
        b[independent[j],j]=1.
    return (b,independent[:nindependent])
