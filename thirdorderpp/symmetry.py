"""
Symmetry operations module for crystal structures.

This module provides a pure Python class for handling crystallographic symmetry
operations using the spglib package.
"""

import sys
from math import fabs
import numpy as np
import scipy as sp
import scipy.linalg
import spglib


class SymmetryOperations:
    """
    Crystal symmetry operations handler.
    
    This class encapsulates all symmetry information for a crystal structure,
    including space group, symmetry operations, and related transformations.
    Uses Python spglib package directly for symmetry analysis.
    
    Attributes:
        symbol (str): International space group symbol
        natoms (int): Number of atoms in the unit cell
        nsyms (int): Number of symmetry operations
        symprec (float): Symmetry precision tolerance
        lattice_vectors: Lattice vectors of the unit cell (3x3 array)
        positions: Fractional atomic positions (natoms x 3 array)
        types: Atomic type indices (natoms array)
        rotations: Rotation matrices in fractional coordinates (nsyms x 3 x 3)
        translations: Translation vectors in fractional coordinates (nsyms x 3)
        crotations: Rotation matrices in Cartesian coordinates (nsyms x 3 x 3)
        ctranslations: Translation vectors in Cartesian coordinates (nsyms x 3)
        origin_shift: Origin shift for standardization (3 array)
        transformation_matrix: Transformation matrix for standardization (3x3 array)
    
    Example:
        >>> lattvec = np.eye(3) * 5.0  # 5 Å cubic cell
        >>> types = np.array([1, 1], dtype=int)
        >>> positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        >>> symops = SymmetryOperations(lattvec, types, positions)
        >>> print(f"Space group: {symops.symbol}")
        >>> print(f"Number of operations: {symops.nsyms}")
    """
    
    def __init__(self, lattvec, types, positions, symprec=1e-5):
        """
        Initialize symmetry operations for a crystal structure.
        
        Parameters:
            lattvec : array_like, shape (3, 3)
                Lattice vectors as row vectors
            types : array_like, shape (natoms,)
                Atomic type indices (integers)
            positions : array_like, shape (natoms, 3)
                Fractional atomic positions
            symprec : float, optional
                Symmetry precision tolerance (default: 1e-5)
        
        Raises:
            ValueError: If input arrays have incorrect shapes or spglib fails
        """
        # Convert and validate input arrays
        self._lattvec = np.array(lattvec, dtype=np.double)
        self._types = np.array(types, dtype=np.intc)
        self._positions = np.array(positions, dtype=np.double)
        self.symprec = symprec
        
        # Validate shapes
        if self._lattvec.shape != (3, 3):
            raise ValueError("lattice vectors must form a 3 x 3 matrix")
        
        self.natoms = self._positions.shape[0]
        if self._positions.shape != (self.natoms, 3):
            raise ValueError("positions must be a natoms x 3 array")
        
        if len(self._types) != self.natoms:
            raise ValueError("types length must match number of atoms")
        
        # Get symmetry information from spglib
        self._get_symmetry_dataset()
    
    def _get_symmetry_dataset(self):
        """
        Retrieve symmetry dataset from spglib and compute derived quantities.
        
        This method:
        1. Calls spglib to get crystallographic symmetry information
        2. Extracts space group symbol and symmetry operations
        3. Computes Cartesian representations of symmetry operations
        
        Raises:
            ValueError: If spglib fails to find symmetry
        """
        # Prepare cell for spglib: (lattice, positions, numbers)
        cell = (self._lattvec, self._positions, self._types)
        
        # Call spglib
        dataset = spglib.get_symmetry_dataset(cell, symprec=self.symprec)
        
        if dataset is None:
            raise ValueError("spglib failed to find symmetry for the given structure")
        
        # Extract symmetry information
        self.symbol = dataset['international'].strip()
        self._shift = np.array(dataset['origin_shift'], dtype=np.double)
        self._transform = np.array(dataset['transformation_matrix'], dtype=np.double)
        self.nsyms = len(dataset['rotations'])
        self._rotations = np.array(dataset['rotations'], dtype=np.double)
        self._translations = np.array(dataset['translations'], dtype=np.double)
        
        # Compute Cartesian representations
        self._crotations = np.empty_like(self._rotations)
        self._ctranslations = np.empty_like(self._translations)
        
        lat_inv = sp.linalg.inv(self._lattvec)
        
        for i in range(self.nsyms):
            # Convert rotation from fractional to Cartesian
            # R_cart = L · R_frac · L^(-1)
            self._crotations[i] = np.dot(
                self._lattvec,
                np.dot(self._rotations[i], lat_inv)
            )
            
            # Convert translation from fractional to Cartesian
            # t_cart = L · t_frac
            self._ctranslations[i] = np.dot(self._lattvec, self._translations[i])
    
    @property
    def lattice_vectors(self):
        """Lattice vectors as a 3x3 numpy array."""
        return self._lattvec.copy()
    
    @property
    def types(self):
        """Atomic type indices."""
        return self._types.copy()
    
    @property
    def positions(self):
        """Fractional atomic positions."""
        return self._positions.copy()
    
    @property
    def origin_shift(self):
        """Origin shift for standardization."""
        return self._shift.copy()
    
    @property
    def transformation_matrix(self):
        """Transformation matrix for standardization."""
        return self._transform.copy()
    
    @property
    def rotations(self):
        """Rotation matrices in fractional coordinates."""
        return self._rotations.copy()
    
    @property
    def translations(self):
        """Translation vectors in fractional coordinates."""
        return self._translations.copy()
    
    @property
    def crotations(self):
        """Rotation matrices in Cartesian coordinates."""
        return self._crotations.copy()
    
    @property
    def ctranslations(self):
        """Translation vectors in Cartesian coordinates."""
        return self._ctranslations.copy()
    
    def apply_all(self, r_in):
        """
        Apply all symmetry operations to a vector.
        
        Parameters:
            r_in : array_like, shape (3,)
                Input vector in Cartesian coordinates
        
        Returns:
            ndarray, shape (3, nsyms)
                Transformed vectors for all symmetry operations
        """
        r_in = np.asarray(r_in, dtype=np.double)
        if r_in.shape != (3,):
            raise ValueError("Input vector must have shape (3,)")
        
        r_out = np.zeros((3, self.nsyms), dtype=np.double)
        
        for isym in range(self.nsyms):
            # Apply rotation and translation: r' = R·r + t
            r_out[:, isym] = np.dot(self._crotations[isym], r_in) + self._ctranslations[isym]
        
        return r_out
    
    def map_supercell(self, sposcar):
        """
        Map symmetry operations to supercell atomic permutations.
        
        Each symmetry operation defines a permutation of atoms in the supercell.
        This method computes these permutations for a supercell that is compatible
        with the unit cell.
        
        Parameters:
            sposcar : dict
                Supercell structure with keys:
                - 'positions': atomic positions (3 x ntot array, fractional coords)
                - 'lattvec': lattice vectors (3 x 3 array)
                - 'na', 'nb', 'nc': supercell dimensions (integers)
        
        Returns:
            ndarray, shape (nsyms, ntot), dtype=int
                Permutation array where result[isym, iatom] gives the index of the atom
                that iatom maps to under symmetry operation isym
        
        Raises:
            SystemExit: If equivalent atom cannot be found (indicates incompatible supercell)
        """
        # Extract supercell information
        positions = sposcar["positions"]  # 3 x ntot
        lattvec = sposcar["lattvec"]      # 3 x 3
        na, nb, nc = sposcar["na"], sposcar["nb"], sposcar["nc"]
        ngrid = np.array([na, nb, nc], dtype=int)
        
        ntot = positions.shape[1]
        natoms = ntot // (na * nb * nc)
        
        if natoms != self.natoms:
            raise ValueError(
                f"Supercell has {natoms} atoms per unit cell, "
                f"but SymmetryOperations was initialized with {self.natoms} atoms"
            )
        
        # Compute unit cell motif in Cartesian coordinates
        motif = np.zeros((3, natoms), dtype=np.double)
        for i in range(natoms):
            # Convert fractional to Cartesian: r_cart = L · r_frac
            motif[:, i] = np.dot(self._lattvec, self._positions[i])
        
        # Initialize output
        permutations = np.empty((self.nsyms, ntot), dtype=np.intc)
        
        # Precompute LU factorization for fractional coordinate conversion
        factorization = sp.linalg.lu_factor(self._lattvec)
       
        # For each atom in supercell
        for iatom in range(ntot):
            # Convert to Cartesian coordinates
            car = np.dot(lattvec, positions[:, iatom])
            
            # Apply all symmetry operations
            car_sym = self.apply_all(car)
            
            # For each symmetry operation
            for isym in range(self.nsyms):
                found = False
                
                # Try to match with each atom type in unit cell
                for itype in range(natoms):
                    # Compute difference vector
                    diff_cart = car_sym[:, isym] - motif[:, itype]
                    
                    # Convert to fractional coordinates to find cell translation
                    diff_frac = sp.linalg.lu_solve(factorization, diff_cart)
                    
                    # Round to nearest integer (cell translation)
                    translation = np.round(diff_frac).astype(int)
                    
                    # Check if this is a valid lattice translation
                    error = np.sum(np.abs(translation - diff_frac))
                    
                    if error < 1e-4:
                        # Apply periodic boundary conditions to get cell indices
                        cell_indices = translation % ngrid
                        
                        # Compute supercell index
                        # index = ix + (iy + iz*ngrid[1])*ngrid[0])*natoms + itype
                        supercell_index = (
                            (cell_indices[0] + 
                             (cell_indices[1] + cell_indices[2] * ngrid[1]) * ngrid[0]
                            ) * natoms + itype
                        )
                        
                        permutations[isym, iatom] = supercell_index
                        found = True
                        break
                
                if not found:
                    sys.exit(
                        f"Error: equivalent atom not found for "
                        f"symmetry operation {isym}, atom {iatom}\n"
                        f"This indicates the supercell is incompatible with the unit cell."
                    )
        
        return permutations
    
    def __repr__(self):
        """String representation of the SymmetryOperations object."""
        return (
            f"SymmetryOperations(symbol='{self.symbol}', "
            f"natoms={self.natoms}, nsyms={self.nsyms}, "
            f"symprec={self.symprec})"
        )
    
    def __str__(self):
        """Human-readable string representation."""
        return (
            f"Symmetry Operations\n"
            f"  Space group: {self.symbol}\n"
            f"  Number of atoms: {self.natoms}\n"
            f"  Number of symmetry operations: {self.nsyms}\n"
            f"  Symmetry precision: {self.symprec}"
        )
