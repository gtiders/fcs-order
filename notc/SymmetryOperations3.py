import sys
import numpy as np
import scipy as sp
import scipy.linalg
import spglib

class SymmetryOperations:
    """
    Object that contains all the interesting information about the
    crystal symmetry group of a set of atoms.
    Python implementation using numpy, scipy and spglib.
    """
    
    def __init__(self, lattvec, types, positions, symprec=1e-5):
        """
        Initialize the symmetry operations object.
        
        Parameters:
        lattvec: 3x3 array of lattice vectors
        types: array of atomic types
        positions: natoms x 3 array of atomic positions
        symprec: symmetry precision
        """
        self.__lattvec = np.array(lattvec, dtype=np.float64)
        self.__types = np.array(types, dtype=np.int32)
        self.__positions = np.array(positions, dtype=np.float64)
        self.natoms = self.__positions.shape[0]
        self.symprec = symprec
        
        # Validate inputs
        if self.__positions.shape[0] != self.natoms or self.__positions.shape[1] != 3:
            raise ValueError("positions must be a natoms x 3 array")
        if not (self.__lattvec.shape[0] == self.__lattvec.shape[1] == 3):
            raise ValueError("lattice vectors must form a 3 x 3 matrix")
        
        # Get symmetry information using spglib
        self._get_symmetry_dataset()
    
    @property
    def lattice_vectors(self):
        """Get lattice vectors as numpy array."""
        return self.__lattvec.copy()
    
    @property
    def types(self):
        """Get atomic types as numpy array."""
        return self.__types.copy()
    
    @property
    def positions(self):
        """Get atomic positions as numpy array."""
        return self.__positions.copy()
    
    @property
    def origin_shift(self):
        """Get origin shift."""
        return self.__shift.copy()
    
    @property
    def transformation_matrix(self):
        """Get transformation matrix."""
        return self.__transform.copy()
    
    @property
    def rotations(self):
        """Get rotation matrices."""
        return self.__rotations.copy()
    
    @property
    def translations(self):
        """Get translation vectors."""
        return self.__translations.copy()
    
    @property
    def crotations(self):
        """Get Cartesian rotation matrices."""
        return self.__crotations.copy()
    
    @property
    def ctranslations(self):
        """Get Cartesian translation vectors."""
        return self.__ctranslations.copy()
    
    def _get_symmetry_dataset(self):
        """
        Get symmetry dataset using spglib and extract relevant information.
        """
        # Convert to spglib format
        lattice = self.__lattvec.T  # spglib expects lattice vectors as columns
        positions = self.__positions
        numbers = self.__types
        
        # Get symmetry dataset
        dataset = spglib.get_symmetry_dataset((lattice, positions, numbers), symprec=self.symprec)
        
        if dataset is None:
            raise ValueError("Failed to get symmetry dataset")
        
        # Extract relevant information
        self.symbol = dataset['international']
        self.__shift = np.array(dataset['origin_shift'], dtype=np.float64)
        self.__transform = np.array(dataset['transformation_matrix'], dtype=np.float64)
        self.nsyms = len(dataset['rotations'])
        
        # Store rotations and translations
        self.__rotations = np.array(dataset['rotations'], dtype=np.float64)
        self.__translations = np.array(dataset['translations'], dtype=np.float64)
        
        # Calculate Cartesian rotations and translations
        self._calculate_cartesian_operations()
    
    def _calculate_cartesian_operations(self):
        """
        Calculate Cartesian rotation matrices and translation vectors.
        """
        self.__crotations = np.empty_like(self.__rotations)
        self.__ctranslations = np.empty_like(self.__translations)
        
        for i in range(self.nsyms):
            # Cartesian rotation: R_cart = L * R * L^-1
            self.__crotations[i] = np.dot(
                self.__lattvec,
                np.dot(
                    self.__rotations[i],
                    np.linalg.inv(self.__lattvec)
                )
            )
            
            # Cartesian translation: t_cart = L * t
            self.__ctranslations[i] = np.dot(self.__lattvec, self.__translations[i])
    
    def _apply_all(self, r_in):
        """
        Apply all symmetry operations to a vector and return the results.
        
        Parameters:
        r_in: input vector (3-element array)
        
        Returns:
        Array of transformed vectors (3 x nsyms)
        """
        r_out = np.zeros((3, self.nsyms), dtype=np.float64)
        
        for i in range(self.nsyms):
            # Apply rotation and translation: r_out = R * r_in + t
            r_out[:, i] = np.dot(self.__crotations[i], r_in) + self.__ctranslations[i]
        
        return r_out
    
    def map_supercell(self, sposcar):
        """
        Each symmetry operation defines an atomic permutation in a supercell.
        This method returns an array with those permutations.
        
        Parameters:
        sposcar: dictionary containing supercell information
        
        Returns:
        Array of permutations (nsyms x ntot)
        """
        positions = sposcar["positions"]
        lattvec = sposcar["lattvec"]
        ngrid = np.array([sposcar["na"], sposcar["nb"], sposcar["nc"]], dtype=np.int32)
        
        ntot = positions.shape[1]
        natoms = ntot // (ngrid[0] * ngrid[1] * ngrid[2])
        
        # Calculate motif positions in Cartesian coordinates
        motif = np.empty((3, natoms), dtype=np.float64)
        for i in range(natoms):
            motif[:, i] = np.dot(
                self.__lattvec,
                self.__positions[i]
            )
        
        nruter = np.empty((self.nsyms, ntot), dtype=np.int32)
        
        # Precompute LU factorization for lattice vectors
        factorization = sp.linalg.lu_factor(self.__lattvec)
        
        for i in range(ntot):
            # Convert supercell position to Cartesian coordinates
            car = np.zeros(3, dtype=np.float64)
            for j in range(3):
                car[j] = np.dot(
                    positions[:, i],
                    lattvec[j, :]
                )
            
            # Apply all symmetry operations
            car_sym = self._apply_all(car)
            
            for isym in range(self.nsyms):
                found = False
                for ii in range(natoms):
                    # Calculate difference vector
                    tmp = car_sym[:, isym] - motif[:, ii]
                    
                    # Convert to fractional coordinates
                    tmp_frac = sp.linalg.lu_solve(factorization, tmp)
                    
                    # Round to nearest integer
                    vec = np.round(tmp_frac).astype(np.int32)
                    
                    # Calculate rounding error
                    diff = np.sum(np.abs(vec - tmp_frac))
                    
                    # Apply periodic boundary conditions
                    vec = vec % ngrid
                    
                    if diff < 1e-4:
                        # Found equivalent atom
                        nruter[isym, i] = self._ind2id(vec, ii, ngrid, natoms)
                        found = True
                        break
                
                if not found:
                    raise RuntimeError(
                        f"Error: equivalent atom not found for isym={isym}, atom={i}"
                    )
        
        return nruter
    