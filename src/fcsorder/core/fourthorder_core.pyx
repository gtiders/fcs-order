from libc.math cimport fabs

import numpy as np
import scipy as sp
import time
import sparse
import typer
from rich.progress import track

cimport cython
cimport numpy as np
np.import_array()


# Permutations of 4 elements listed in the same order as in the old
# Fortran code.
cdef int[:,:] permutations=np.array([
    [0,1,2,3],
    [0,2,1,3],
    [0,1,3,2],
    [0,3,1,2],
    [0,3,2,1],
    [0,2,3,1],
    [1,0,2,3],
    [1,0,3,2],
    [1,2,0,3],
    [1,2,3,0],
    [1,3,0,2],
    [1,3,2,0],
    [2,0,1,3],
    [2,0,3,1],
    [2,1,0,3],
    [2,1,3,0],
    [2,3,0,1],
    [2,3,1,0],
    [3,0,1,2],
    [3,0,2,1],
    [3,1,0,2],
    [3,1,2,0],
    [3,2,0,1],
    [3,2,1,0]],dtype=np.intc)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int _ind2id(int[:] icell,int ispecies,int[:] ngrid,int nspecies):
    """
    Merge a set of cell+atom indices into a single index into a supercell.
    """
    return (icell[0]+(icell[1]+icell[2]*ngrid[1])*ngrid[0])*nspecies+ispecies


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint _quartet_in_list(int[:] quartet,int[:,:] llist,int nlist):
    """
    Return True if quartet is found in llist[:,:nlist]. The first dimension
    of list must have a length of 4.
    """
    # This works fine for the nlist ranges we have to deal with, but
    # using std::vector and std::push_heap would be a better general
    # solution.
    cdef int i

    for i in range(nlist):
        if (quartet[0]==llist[0,i] and quartet[1]==llist[1,i] and
            quartet[2]==llist[2,i] and quartet[3]==llist[3,i]):
            return True
    return False


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint _quartets_are_equal(int[:] quartet1,int[:] quartet2):
    """
    Return True if two quartets are equal and False otherwise.
    """
    cdef int i

    for i in range(4):
        if quartet1[i]!=quartet2[i]:
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
    Recover the full fourth-order IFC set from the irreducible set of
    force constants and the information contained in a wedge object.
    """
    cdef int ii,jj,ll,mm,nn,kk,aa1,ss,tt,ix,e0,e1,e2,e3,e4,e5
    cdef int nlist,nlist6,natoms,ntot
    cdef int ntotalindependent,tribasisindex,colindex,nrows,ncols
    cdef int[:] naccumindependent
    cdef int[:,:,:,:] vind1
    cdef int[:,:,:,:] vind2
    cdef int[:,:,:] vequilist
    cdef double[:] aphilist
    cdef double[:,:] vaa
    cdef double[:,:,:] vphipart

    nlist=wedge.nlist
    natoms=len(poscar["types"])
    ntot=len(sposcar["types"])
    if is_sparse:
        typer.print("using sparse method with dok sparse matrix !")
        vnruter=sparse.zeros((3,3,3,3,natoms,ntot,ntot,ntot), format="dok")
    else:
        vnruter=np.zeros((3,3,3,3,natoms,ntot,ntot,ntot),dtype=np.double)
    naccumindependent=np.insert(np.cumsum(
        wedge.nindependentbasis[:nlist],dtype=np.intc),0,
        np.zeros(1,dtype=np.intc))
    ntotalindependent=naccumindependent[-1]
    vphipart=phipart
    nlist6=len(list4)
    for ii in track(range(nlist6), description="Processing list6"):
        e0,e1,e2,e3,e4,e5=list4[ii]
        vnruter[e3,e4,e5,:,e0,e1,e2,:]=vphipart[:,ii,:]
    philist=[]
    for ii in track(range(nlist), description="Building philist"):
        for jj in range(wedge.nindependentbasis[ii]):
            kk=wedge.independentbasis[jj,ii]//27
            ll=wedge.independentbasis[jj,ii]%27//9
            mm=wedge.independentbasis[jj,ii]%9//3
            nn=wedge.independentbasis[jj,ii]%3
            philist.append(vnruter[kk,ll,mm,nn,
                                        wedge.llist[0,ii],
                                        wedge.llist[1,ii],
                                        wedge.llist[2,ii],
                                  wedge.llist[3,ii]])
    aphilist=np.array(philist,dtype=np.double)
    vind1=-np.ones((natoms,ntot,ntot,ntot),dtype=np.intc)
    vind2=-np.ones((natoms,ntot,ntot,ntot),dtype=np.intc)
    vequilist=wedge.allequilist
    for ii in range(nlist):
        for jj in range(wedge.nequi[ii]):
            vind1[vequilist[0,jj,ii],vequilist[1,jj,ii],vequilist[2,jj,ii],vequilist[3,jj,ii]]=ii
            vind2[vequilist[0,jj,ii],vequilist[1,jj,ii],vequilist[2,jj,ii],vequilist[3,jj,ii]]=jj 

    vtrans=wedge.transformationarray

    nrows=ntotalindependent
    ncols=natoms*ntot*81
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
                        for aa1 in range(3):
                            for kk in range(ntot):
                                for bb in range(ntot):
                                    for ix in range(nlist):
                                        if vind1[ii,jj,kk,bb]==ix:
                                            for ss in range(naccumindependent[ix],
                                                            naccumindependent[ix+1]):
                                                tt=ss-naccumindependent[ix]
                                                i.append(ss)
                                                j.append(colindex)
                                                v.append(vtrans[tribasisindex,tt,
                                                        vind2[ii,jj,kk,bb],ix])
                            tribasisindex+=1
                            colindex+=1
    typer.print("- \t Density: {0:.2g}%".format(100.*len(i)/float(nrows*ncols)))
    aaa=sp.sparse.coo_matrix((v,(i,j)),(nrows,ncols)).tocsr()
    D=sp.sparse.spdiags(aphilist,[0,],aphilist.size,aphilist.size,
                           format="csr")
    bbs=D.dot(aaa)
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
            T = wedge.transformationarray[:, :nind, :ne, ii]  # (81, nind, ne)
            out = np.tensordot(T, phi, axes=([1],[0]))        # (81, ne)
            for jj in range(ne):
                block = out[:, jj].reshape(3,3,3,3)
                for ll in range(3):
                    for mm in range(3):
                        for nn in range(3):
                            for aa1 in range(3):
                                val = block[ll,mm,nn,aa1]
                                if val!=0.0 and abs(val)>EPSVAL:
                                    nnz += 1
        coords = np.empty((8, nnz), dtype=np.intp)
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
                e3 = vequilist[3,jj,ii]
                block = out[:, jj].reshape(3,3,3,3)
                for ll in range(3):
                    for mm in range(3):
                        for nn in range(3):
                            for aa1 in range(3):
                                val = block[ll,mm,nn,aa1]
                                if val!=0.0 and abs(val)>EPSVAL:
                                    coords[0,p] = ll
                                    coords[1,p] = mm
                                    coords[2,p] = nn
                                    coords[3,p] = aa1
                                    coords[4,p] = e0
                                    coords[5,p] = e1
                                    coords[6,p] = e2
                                    coords[7,p] = e3
                                    data[p] = val
                                    p += 1
        vnruter = sparse.COO(coords, data, shape=(3,3,3,3,natoms,ntot,ntot,ntot))
    else:
        vnruter=np.zeros((3,3,3,3,natoms,ntot,ntot,ntot),dtype=np.double)

    # Vectorized rebuild of final IFCs.
    # Rationale: For fixed (ii, jj), original code computes for each tribasisindex in [0..80]
    #   sum_{ix=0..nind-1} transformationarray[tribasisindex, ix, jj, ii] * aphilist[offset+ix].
    # This is exactly a matrix-vector product (81 x nind) @ (nind,), so we can replace the
    # innermost loops with a single BLAS-backed dot/tensordot without changing results.
    for ii in track(range(nlist), description="Building final IFCs"):
        nind = wedge.nindependentbasis[ii]
        ne = wedge.nequi[ii]
        if nind==0 or ne==0:
            continue
        offset = naccumindependent[ii]
        phi = aphilist[offset:offset+nind]
        # T has shape (81, nind, ne); each column along axis=2 corresponds to one jj equivalence.
        T = wedge.transformationarray[:, :nind, :ne, ii]
        # out has shape (81, ne); each column is the 81-length result for a given jj.
        out = np.tensordot(T, phi, axes=([1],[0]))
        for jj in range(ne):
            e0 = vequilist[0,jj,ii]
            e1 = vequilist[1,jj,ii]
            e2 = vequilist[2,jj,ii]
            e3 = vequilist[3,jj,ii]
            block = out[:, jj].reshape(3,3,3,3)
            # Dense path writes directly; sparse has been constructed above via COO.
            if not is_sparse:
                for ll in range(3):
                    for mm in range(3):
                        for nn in range(3):
                            for aa1 in range(3):
                                val = block[ll,mm,nn,aa1]
                                if val!=0.0:
                                    vnruter[ll,mm,nn,aa1,e0,e1,e2,e3] += val

    return vnruter



cdef class Wedge:
    """
    Objects of this class allow the user to extract irreducible sets
    of force constants and to reconstruct the full Fourth-order IFC
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
            self.allocsize=32
            self.nequi=np.empty(self.allocsize,dtype=np.intc)
            self.allequilist=np.empty((4,24*self.symops.nsyms,
                                       self.allocsize),dtype=np.intc)
            self.transformationarray=np.empty((81,81,24*self.symops.nsyms,
                                               self.allocsize),dtype=np.double)
            self.transformation=np.empty((81,81,24*self.symops.nsyms,
                                               self.allocsize),dtype=np.double)
            self.transformationaux=np.empty((81,81,self.allocsize),
                                            dtype=np.double)
            self.nindependentbasis=np.empty(self.allocsize,dtype=np.intc)
            self.independentbasis=np.empty((81,self.allocsize),dtype=np.intc)
            self.llist=np.empty((4,self.allocsize),dtype=np.intc)
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
            self.alllist=np.empty((4,self.allallocsize),dtype=np.intc)
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
        cdef int ii,jj,kk,nn,mm,ll,iaux,jaux,kaux
        cdef int ibasis,jbasis,kbasis,lbasis,ibasisprime,jbasisprime,kbasisprime,lbasisprime
        cdef int iperm,isym,indexijkl,indexijklprime
        cdef int[:] ngrid,ind_species,vec1,vec2,vec3,vec4,independent
        cdef int[:] v_nequi,v_nindependentbasis
        cdef int[:] basis,quartet,quartet_perm,quartet_sym
        cdef int[:,:] v_llist,v_alllist,v_independentbasis
        cdef int[:,:] shifts27,shift2all,shift3all,shift4all
        cdef int[:,:] equilist,id_equi,ind_cell
        cdef int[:,:,:] nonzero
        cdef int[:,:,:] v_allequilist
        cdef double dist,dist1,frange2
        cdef double[:] car2,car3,car4
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
        vec4=np.empty(3,dtype=np.intc)

        lattvec=self.sposcar["lattvec"]
        coordall=np.dot(lattvec,self.sposcar["positions"])
        orth=np.transpose(self.symops.crotations,(1,2,0))
        car2=np.empty(3,dtype=np.double)
        car3=np.empty(3,dtype=np.double)
        car4=np.empty(3,dtype=np.double)

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

        basis=np.empty(4,dtype=np.intc)
        quartet=np.empty(4,dtype=np.intc)
        quartet_perm=np.empty(4,dtype=np.intc)
        quartet_sym=np.empty(4,dtype=np.intc)
        shift2all=np.empty((3,27),dtype=np.intc)
        shift3all=np.empty((3,27),dtype=np.intc)
        shift4all=np.empty((3,27),dtype=np.intc)
        equilist=np.empty((4,nsym*24),dtype=np.intc)
        coeffi=np.empty((24*nsym*81,81),dtype=np.double)
        id_equi=self.symops.map_supercell(self.sposcar)
        ind_cell,ind_species=_id2ind(ngrid,natoms)

        # Rotation matrices for fourth derivatives and related quantities.
        rot=np.empty((24,nsym,81,81),dtype=np.double)
        for iperm in range(24):
            for isym in range(nsym):
                for ibasisprime in range(3):
                    for jbasisprime in range(3):
                        for kbasisprime in range(3):
                            for lbasisprime in range(3):
                                indexijklprime=((ibasisprime*3+jbasisprime)*3+kbasisprime)*3+lbasisprime
                                for ibasis in range(3):
                                    basis[0]=ibasis
                                    for jbasis in range(3):
                                        basis[1]=jbasis
                                        for kbasis in range(3):
                                            basis[2]=kbasis
                                            for lbasis in range(3):
                                                basis[3]=lbasis
                                                indexijkl=ibasis*27+jbasis*9+kbasis*3+lbasis
                                                ibasispermut=basis[permutations[iperm,0]]
                                                jbasispermut=basis[permutations[iperm,1]]
                                                kbasispermut=basis[permutations[iperm,2]]
                                                lbasispermut=basis[permutations[iperm,3]]
                                                rot[iperm,isym,indexijklprime,indexijkl]=(
                                                orth[ibasisprime,ibasispermut,isym]*
                                                orth[jbasisprime,jbasispermut,isym]*
                                                orth[kbasisprime,kbasispermut,isym]*
                                                orth[lbasisprime,lbasispermut,isym])
        rot2=rot.copy()
        nonzero=np.zeros((24,nsym,81),dtype=np.intc)
        for iperm in range(24):
            for isym in range(nsym):
                for indexijklprime in range(81):
                    rot2[iperm,isym,indexijklprime,indexijklprime]-=1.
                    for indexijkl in range(81):
                        if fabs(rot2[iperm,isym,indexijklprime,indexijkl])>1e-12:
                            nonzero[iperm,isym,indexijklprime]=1
                        else:
                            rot2[iperm,isym,indexijklprime,indexijkl]=0.

        # Scan all atom quartets (ii,jj,kk,mm) in the supercell.
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
                    for mm in range(ntot):
                        dist=self.dmin[ii,mm]
                        if dist>=self.frange:
                           continue
                        n4equi=self.nequis[ii,mm]
                        for nn in range(n4equi):
                            shift4all[:,nn]=shifts27[self.shifts[ii,mm,nn],:]
                        d2_min1=np.inf
                        d2_min2=np.inf
                        d2_min3=np.inf
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
                                for kaux in range(n4equi):
                                    for ll in range(3):
                                        car4[ll]=(shift4all[0,kaux]*lattvec[ll,0]+
                                          shift4all[1,kaux]*lattvec[ll,1]+
                                          shift4all[2,kaux]*lattvec[ll,2]+
                                          coordall[ll,mm])
                     #           d2_min1=min(d2_min1,
                      #             (car3[0]-car2[0])**2+
                       #            (car3[1]-car2[1])**2+
                        #           (car3[2]-car2[2])**2)
                                d2_min3=min(d2_min3,
                                   (car4[0]-car3[0])**2+
                                   (car4[1]-car3[1])**2+
                                   (car4[2]-car3[2])**2)
                            if d2_min3>frange2:
                               continue
                            d2_min2=min(d2_min2,
                               (car4[0]-car2[0])**2+
                               (car4[1]-car2[1])**2+
                               (car4[2]-car2[2])**2)
                            d2_min1=min(d2_min1,
                               (car3[0]-car2[0])**2+
                               (car3[1]-car2[1])**2+
                               (car3[2]-car2[2])**2)
                        if d2_min1>=frange2:
                           continue
                        if d2_min2>=frange2:
                           continue
                    #    if d2_min3>=frange2:
                    #       continue
                    # This point is only reached if there is a choice of periodic images of
                    # ii, jj and kk such that all pairs ii-jj, ii-kk, ii-mm, and jj-kk, jj-mm,mm-kk are within
                    # the specified interaction range.
                        summ+=1
                        quartet[0]=ii
                        quartet[1]=jj
                        quartet[2]=kk
                        quartet[3]=mm
                        if _quartet_in_list(quartet,v_alllist,self.nalllist):
                           continue
                    # This point is only reached if the quartet is not
                    # equivalent to any of the quartets already considered,
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
                        v_llist[3,self.nlist-1]=mm
                        v_nequi[self.nlist-1]=0
                        coeffi[:,:]=0.
                        nnonzero=0
                        # Scan the 24 possible permutations of quartet (ii,jj,kk,mm).
                        for iperm in range(24):
                            quartet_perm[0]=quartet[permutations[iperm,0]]
                            quartet_perm[1]=quartet[permutations[iperm,1]]
                            quartet_perm[2]=quartet[permutations[iperm,2]]
                            quartet_perm[3]=quartet[permutations[iperm,3]]
                            # Explore the effect of all symmetry operations on each of
                            # the permuted quartets.
                            for isym in range(nsym):
                                quartet_sym[0]=id_equi[isym,quartet_perm[0]]
                                quartet_sym[1]=id_equi[isym,quartet_perm[1]]
                                quartet_sym[2]=id_equi[isym,quartet_perm[2]]
                                quartet_sym[3]=id_equi[isym,quartet_perm[3]]
                                for ll in range(3):
                                    vec1[ll]=ind_cell[ll,id_equi[isym,quartet_perm[0]]]
                                    vec2[ll]=ind_cell[ll,id_equi[isym,quartet_perm[1]]]
                                    vec3[ll]=ind_cell[ll,id_equi[isym,quartet_perm[2]]]
                                    vec4[ll]=ind_cell[ll,id_equi[isym,quartet_perm[3]]]
                                # Choose a displaced version of quartet_sym chosen so that
                                # atom 0 is always in the first unit cell.
                                if not vec1[0]==vec1[1]==vec1[2]==0:
                                    for ll in range(3):
                                        vec4[ll]=(vec4[ll]-vec1[ll])%ngrid[ll]
                                        vec3[ll]=(vec3[ll]-vec1[ll])%ngrid[ll]
                                        vec2[ll]=(vec2[ll]-vec1[ll])%ngrid[ll]
                                        vec1[ll]=0
                                    ispecies1=ind_species[id_equi[isym,quartet_perm[0]]]
                                    ispecies2=ind_species[id_equi[isym,quartet_perm[1]]]
                                    ispecies3=ind_species[id_equi[isym,quartet_perm[2]]]
                                    ispecies4=ind_species[id_equi[isym,quartet_perm[3]]]
                                    quartet_sym[0]=_ind2id(vec1,ispecies1,ngrid,natoms)
                                    quartet_sym[1]=_ind2id(vec2,ispecies2,ngrid,natoms)
                                    quartet_sym[2]=_ind2id(vec3,ispecies3,ngrid,natoms)
                                    quartet_sym[3]=_ind2id(vec4,ispecies4,ngrid,natoms)
                                # If the permutation+symmetry operation changes the quartet into
                                # an as-yet-unseen image, add it to the list of equivalent quartets
                                # and fill the transformation array accordingly.
                                if (iperm==0 and isym==0) or not (
                                       _quartets_are_equal(quartet_sym,quartet) or
                                    _quartet_in_list(quartet_sym,equilist,v_nequi[self.nlist-1])):
                                   v_nequi[self.nlist-1]+=1
                                   for ll in range(4):
                                       equilist[ll,v_nequi[self.nlist-1]-1]=quartet_sym[ll]
                                       v_allequilist[ll,v_nequi[self.nlist-1]-1,
                                                  self.nlist-1]=quartet_sym[ll]
                                   self.nalllist+=1
                                   if self.nalllist==self.allallocsize:
                                      self._expandalllist()
                                      v_alllist=self.alllist
                                   for ll in range(4):
                                       v_alllist[ll,self.nalllist-1]=quartet_sym[ll]
                                   for iaux in range(81):
                                       for jaux in range(81):
                                           v_transformation[iaux,jaux,v_nequi[self.nlist-1]-1,
                                                         self.nlist-1]=rot[iperm,isym,iaux,jaux]
                                # If the permutation+symmetry operation amounts to the identity,
                                # add a row to the coefficient matrix.
                                if _quartets_are_equal(quartet_sym,quartet):
                                    for indexijklprime in range(81):
                                        if nonzero[iperm,isym,indexijklprime]:
                                           for ll in range(81):
                                               coeffi[nnonzero,ll]=rot2[iperm,isym,indexijklprime,ll]
                                           nnonzero+=1
                        coeffi_reduced=np.zeros((max(nnonzero,81),81),dtype=np.double)
                        for iaux in range(nnonzero):
                            for jaux in range(81):
                                coeffi_reduced[iaux,jaux]=coeffi[iaux,jaux]
                        # Obtain a set of independent IFCs for this quartet equivalence class.
                        b,independent=gaussian(coeffi_reduced)
                        for iaux in range(81):
                            for jaux in range(81):
                                v_transformationaux[iaux,jaux,self.nlist-1]=b[iaux,jaux]
                        v_nindependentbasis[self.nlist-1]=independent.shape[0]
                        for ll in range(independent.shape[0]):
                            v_independentbasis[ll,self.nlist-1]=independent[ll]
        v_transformationarray[:,:,:,:]=0.
        for ii in range(self.nlist):
            for jj in range(v_nequi[ii]):
                for kk in range(81):
                    for ll in range(v_nindependentbasis[ii]):
                        for iaux in range(81):
                            v_transformationarray[kk,ll,jj,ii]+=(
                                v_transformation[kk,iaux,jj,ii]*
                                v_transformationaux[iaux,ll,ii])
                for kk in range(81):
                    for ll in range(81):
                        if fabs(v_transformationarray[kk,ll,jj,ii])<1e-15:
                            v_transformationarray[kk,ll,jj,ii]=0.

    def build_list4(self):
        """
        Build a list of 6-uples from the results of the reduction.
        """
        cdef int ii,jj,kk,ll,mm,nn
        cdef list list6,nruter

        list6=[]
        for ii in range(self.nlist):
            for jj in range(self.nindependentbasis[ii]):
                kk=self.independentbasis[jj,ii]//27
                ll=(self.independentbasis[jj,ii]%27)//9
                mm=(self.independentbasis[jj,ii]%9)//3
                nn=self.independentbasis[jj,ii]%3
                list6.append((kk, self.llist[0,ii], ll,self.llist[1,ii],
                        mm, self.llist[2,ii],
                        nn, self.llist[3,ii]))
        nruter=[]
        for i in list6:
            fournumbers=(i[1],i[3],i[5],i[0],i[2],i[4])
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
