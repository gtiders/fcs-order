"""
Third-order force constants calculation using Python spglib.

This module replaces the Cython implementation with pure Python,
using Python spglib instead of C spglib for symmetry operations.
"""

import numpy as np
import scipy.linalg as sp
import spglib


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


def _triplets_are_equal(triplet1, triplet2):
    """
    Return True if two triplets are equal and False otherwise.
    """
    for i in range(3):
        if triplet1[i] != triplet2[i]:
            return False
    return True


def _id2ind(ngrid, nspecies):
    """
    Create a map from supercell indices to cell+atom indices.
    """
    ntot = ngrid[0] * ngrid[1] * ngrid[2] * nspecies
    np_icell = np.empty((3, ntot), dtype=np.intc)
    np_ispecies = np.empty(ntot, dtype=np.intc)
    
    for ii in range(ntot):
        tmp, np_ispecies[ii] = divmod(ii, nspecies)
        tmp, np_icell[0, ii] = divmod(tmp, ngrid[0])
        np_icell[2, ii], np_icell[1, ii] = divmod(tmp, ngrid[1])
    
    return (np_icell, np_ispecies)


class SymmetryOperations:
    """
    Object that contains all the interesting information about the
    crystal symmetry group of a set of atoms.
    
    This is the Python version that uses Python spglib instead of C spglib.
    """
    
    def __init__(self, lattvec, types, positions, symprec=1e-5):
        """
        Initialize symmetry operations using Python spglib.
        
        Parameters:
        -----------
        lattvec : array_like, shape (3, 3)
            Lattice vectors in Angstrom
        types : array_like, shape (natoms,)
            Atomic types (integers)
        positions : array_like, shape (natoms, 3)
            Atomic positions in fractional coordinates
        symprec : float, optional
            Symmetry precision (default: 1e-5)
        """
        self.__lattvec = np.array(lattvec, dtype=np.double)
        self.__types = np.array(types, dtype=np.intc)
        self.__positions = np.array(positions, dtype=np.double)
        self.natoms = self.__positions.shape[0]
        self.symprec = symprec
        
        # Validate input shapes
        if self.__positions.shape[0] != self.natoms or self.__positions.shape[1] != 3:
            raise ValueError("positions must be a natoms x 3 array")
        if not (self.__lattvec.shape[0] == self.__lattvec.shape[1] == 3):
            raise ValueError("lattice vectors must form a 3 x 3 matrix")
        
        # Get symmetry information using Python spglib
        self.__get_symmetry_dataset()
    
    def __get_symmetry_dataset(self):
        """
        Get symmetry dataset using Python spglib.
        
        This replaces the C spglib implementation with Python spglib.
        """
        # Prepare cell data for spglib
        cell = (self.__lattvec, self.__positions, self.__types)
        
        # Get symmetry dataset using Python spglib
        dataset = spglib.get_symmetry_dataset(cell, symprec=self.symprec)
        
        if dataset is None:
            raise RuntimeError("Failed to get symmetry dataset from spglib")
        
        # Extract symmetry information
        self.symbol = dataset['international'].strip()
        self.__shift = np.array(dataset['origin_shift'], dtype=np.double)
        self.__transform = np.array(dataset['transformation_matrix'], dtype=np.double)
        self.nsyms = len(dataset['rotations'])
        
        # Store rotations and translations
        self.__rotations = np.array(dataset['rotations'], dtype=np.double)
        self.__translations = np.array(dataset['translations'], dtype=np.double)
        
        # Convert to Cartesian coordinates
        self.__crotations = np.empty_like(self.__rotations)
        self.__ctranslations = np.empty_like(self.__translations)
        
        for i in range(self.nsyms):
            # Convert rotation to Cartesian
            tmp2d = np.dot(self.__lattvec,
                          np.dot(self.__rotations[i, :, :],
                                sp.linalg.inv(self.__lattvec)))
            self.__crotations[i, :, :] = tmp2d
            
            # Convert translation to Cartesian
            tmp1d = np.dot(self.__lattvec, self.__translations[i, :])
            self.__ctranslations[i, :] = tmp1d
    
    @property
    def lattice_vectors(self):
        """Get lattice vectors."""
        return np.asarray(self.__lattvec)
    
    @property
    def types(self):
        """Get atomic types."""
        return np.asarray(self.__types)
    
    @property
    def positions(self):
        """Get atomic positions."""
        return np.asarray(self.__positions)
    
    @property
    def origin_shift(self):
        """Get origin shift."""
        return np.asarray(self.__shift)
    
    @property
    def transformation_matrix(self):
        """Get transformation matrix."""
        return np.asarray(self.__transform)
    
    @property
    def rotations(self):
        """Get symmetry rotations in fractional coordinates."""
        return np.asarray(self.__rotations)
    
    @property
    def translations(self):
        """Get symmetry translations in fractional coordinates."""
        return np.asarray(self.__translations)
    
    @property
    def crotations(self):
        """Get symmetry rotations in Cartesian coordinates."""
        return np.asarray(self.__crotations)
    
    @property
    def ctranslations(self):
        """Get symmetry translations in Cartesian coordinates."""
        return np.asarray(self.__ctranslations)
    
    def __apply_all(self, r_in):
        """
        Apply all symmetry operations to a vector and return the results.
        
        Parameters:
        -----------
        r_in : array_like, shape (3,)
            Input vector
            
        Returns:
        --------
        array, shape (3, nsyms)
            Results of applying all symmetry operations
        """
        r_out = np.zeros((3, self.nsyms), dtype=np.double)
        
        for ii in range(self.nsyms):
            for jj in range(3):
                for kk in range(3):
                    r_out[jj, ii] += self.__crotations[ii, jj, kk] * r_in[kk]
                r_out[jj, ii] += self.__ctranslations[ii, jj]
        
        return r_out
    
    def map_supercell(self, sposcar):
        """
        Each symmetry operation defines an atomic permutation in a supercell. 
        This method returns an array with those permutations. 
        The supercell must be compatible with the unit cell used to create the object.
        
        Parameters:
        -----------
        sposcar : dict
            Supercell information containing positions, lattice vectors, and dimensions
            
        Returns:
        --------
        array, shape (nsyms, ntot)
            Permutation array for each symmetry operation
        """
        positions = sposcar["positions"]
        lattvec = sposcar["lattvec"]
        ngrid = np.array([sposcar["na"], sposcar["nb"], sposcar["nc"]],
                          dtype=np.intc)
        
        ntot = positions.shape[1]
        natoms = ntot // (ngrid[0] * ngrid[1] * ngrid[2])
        
        # Calculate motif positions
        motif = np.empty((3, natoms), dtype=np.double)
        for i in range(natoms):
            for ii in range(3):
                motif[ii, i] = (self.__positions[i, 0] * self.__lattvec[ii, 0] +
                               self.__positions[i, 1] * self.__lattvec[ii, 1] +
                               self.__positions[i, 2] * self.__lattvec[ii, 2])
        
        nruter = np.empty((self.nsyms, ntot), dtype=np.intc)
        
        # Map supercell atoms
        for isym in range(self.nsyms):
            for i in range(ntot):
                # Get cell indices and species
                icell = np.empty(3, dtype=np.intc)
                tmp = i
                ispecies, tmp = divmod(tmp, natoms)
                icell[2], tmp = divmod(tmp, ngrid[0] * ngrid[1])
                icell[1], icell[0] = divmod(tmp, ngrid[0])
                
                # Get position in Cartesian coordinates
                car = np.empty(3, dtype=np.double)
                for ii in range(3):
                    car[ii] = motif[ii, ispecies]
                    for jj in range(3):
                        car[ii] += lattvec[ii, jj] * icell[jj] / ngrid[jj]
                
                # Apply symmetry operation
                tmp = self.__apply_all(car)
                
                # Find equivalent position
                for ii in range(ntot):
                    # Get cell indices and species for candidate
                    jcell = np.empty(3, dtype=np.intc)
                    jtmp = ii
                    jspecies, jtmp = divmod(jtmp, natoms)
                    jcell[2], jtmp = divmod(jtmp, ngrid[0] * ngrid[1])
                    jcell[1], jcell[0] = divmod(jtmp, ngrid[0])
                    
                    # Get candidate position
                    car_sym = np.empty(3, dtype=np.double)
                    for jj in range(3):
                        car_sym[jj] = motif[jj, jspecies]
                        for kk in range(3):
                            car_sym[jj] += lattvec[jj, kk] * jcell[kk] / ngrid[kk]
                    
                    # Check if positions match
                    diff = 0.0
                    for jj in range(3):
                        diff += (tmp[jj, isym] - car_sym[jj]) ** 2
                    
                    if diff < 1e-10:
                        nruter[isym, i] = ii
                        break
        
        return nruter


# Additional functions for third-order calculations would go here
# These would be ported from the remaining parts of the original Cython file

def gaussian(a):
    """
    Specialized version of Gaussian elimination.
    
    Parameters:
    -----------
    a : array_like, shape (n_rows, n_cols)
        Input matrix
        
    Returns:
    --------
    tuple : (b, independent)
        b : array, shape (n_cols, n_independent)
            Transformation matrix
        independent : array, shape (n_independent,)
            Indices of independent columns
    """
    EPS = 1e-10
    
    row = a.shape[0]
    col = a.shape[1]
    
    dependent = np.empty(col, dtype=np.intc)
    independent = np.empty(col, dtype=np.intc)
    b = np.zeros((col, col), dtype=np.double)
    
    irow = 0
    ndependent = 0
    nindependent = 0
    
    for k in range(min(row, col)):
        # Zero out small elements
        for i in range(row):
            if abs(a[i, k]) < EPS:
                a[i, k] = 0.
        
        # Pivoting
        for i in range(irow + 1, row):
            if abs(a[i, k]) - abs(a[irow, k]) > EPS:
                for j in range(k, col):
                    a[irow, j], a[i, j] = a[i, j], a[irow, j]
        
        # Process column
        if abs(a[irow, k]) > EPS:
            dependent[ndependent] = k
            ndependent += 1
            
            # Normalize row
            for j in range(col - 1, k, -1):
                a[irow, j] /= a[irow, k]
            a[irow, k] = 1.
            
            # Eliminate other rows
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
    
    # Build transformation matrix
    for j in range(nindependent):
        for i in range(ndependent):
            b[dependent[i], j] = -a[i, independent[j]]
        b[independent[j], j] = 1.
    
    return (b, independent[:nindependent])