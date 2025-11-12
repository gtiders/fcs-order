import sys

import numpy as np
import scipy as sp
import spglib


class SymmetryOperations:
    """
    Object that contains all the interesting information about the
    crystal symmetry group of a set of atoms.
    """

    @property
    def lattice_vectors(self):
        return np.asarray(self.__lattvec)

    @property
    def types(self):
        return np.asarray(self.__types)

    @property
    def positions(self):
        return np.asarray(self.__positions)

    @property
    def origin_shift(self):
        return np.asarray(self.__shift)

    @property
    def transformation_matrix(self):
        return np.asarray(self.__transform)

    @property
    def rotations(self):
        return np.asarray(self.__rotations)

    @property
    def translations(self):
        return np.asarray(self.__translations)

    @property
    def crotations(self):
        return np.asarray(self.__crotations)

    @property
    def ctranslations(self):
        return np.asarray(self.__ctranslations)

    def __spg_get_dataset(self):
        lattice = np.asarray(self.__lattvec).T
        positions = np.asarray(self.__positions)
        types = np.asarray(self.__types)

        dataset = spglib.get_symmetry_dataset(
            (lattice, positions, types), symprec=self.symprec
        )
        if dataset is None:
            raise MemoryError()

        self.symbol = dataset.international.strip()
        self.__shift = np.array(dataset.origin_shift, dtype=np.double)
        self.__transform = np.array(dataset.transformation_matrix, dtype=np.double)
        self.nsyms = len(dataset.rotations)
        self.__rotations = np.array(dataset.rotations, dtype=np.double)
        self.__translations = np.array(dataset.translations, dtype=np.double)

        self.__crotations = np.empty_like(self.__rotations)
        self.__ctranslations = np.empty_like(self.__translations)
        inv_latt = sp.linalg.inv(self.__lattvec)
        for i in range(self.nsyms):
            tmp2d = np.dot(self.__lattvec, np.dot(self.__rotations[i, :, :], inv_latt))
            self.__crotations[i, :, :] = tmp2d
            tmp1d = np.dot(self.__lattvec, self.__translations[i, :])
            self.__ctranslations[i, :] = tmp1d

    def __init__(self, lattvec, types, positions, symprec=1e-5):
        self.__lattvec = np.array(lattvec, dtype=np.double)
        self.__types = np.array(types, dtype=np.intc)
        self.__positions = np.array(positions, dtype=np.double)
        self.natoms = self.positions.shape[0]
        self.symprec = symprec
        if self.__positions.shape[0] != self.natoms or self.__positions.shape[1] != 3:
            raise ValueError("positions must be a natoms x 3 array")
        if not (self.__lattvec.shape[0] == self.__lattvec.shape[1] == 3):
            raise ValueError("lattice vectors must form a 3 x 3 matrix")
        self.__spg_get_dataset()

    def __apply_all(self, r_in):
        r_in = np.asarray(r_in, dtype=np.double)
        r_out = np.zeros((3, self.nsyms), dtype=np.double)
        for ii in range(self.nsyms):
            r_out[:, ii] = (
                np.dot(self.__crotations[ii, :, :], r_in) + self.__ctranslations[ii, :]
            )
        return r_out

    def map_supercell(self, sposcar):
        positions = sposcar["positions"]
        lattvec = sposcar["lattvec"]
        ngrid = np.array([sposcar["na"], sposcar["nb"], sposcar["nc"]], dtype=np.intc)
        ntot = positions.shape[1]
        natoms = ntot // (ngrid[0] * ngrid[1] * ngrid[2])
        motif = np.empty((3, natoms), dtype=np.double)
        for i in range(natoms):
            for ii in range(3):
                motif[ii, i] = (
                    self.__positions[i, 0] * self.__lattvec[ii, 0]
                    + self.__positions[i, 1] * self.__lattvec[ii, 1]
                    + self.__positions[i, 2] * self.__lattvec[ii, 2]
                )
        nruter = np.empty((self.nsyms, ntot), dtype=np.intc)
        car = np.empty(3, dtype=np.double)
        tmp = np.empty(3, dtype=np.double)
        vec = np.empty(3, dtype=np.intc)
        factorization = sp.linalg.lu_factor(self.__lattvec)
        for i in range(ntot):
            for ii in range(3):
                car[ii] = (
                    positions[0, i] * lattvec[ii, 0]
                    + positions[1, i] * lattvec[ii, 1]
                    + positions[2, i] * lattvec[ii, 2]
                )
            car_sym = self.__apply_all(car)
            for isym in range(self.nsyms):
                found = False
                for ii in range(natoms):
                    for ll in range(3):
                        tmp[ll] = car_sym[ll, isym] - motif[ll, ii]
                    tmp = sp.linalg.lu_solve(factorization, tmp)
                    for ll in range(3):
                        vec[ll] = int(round(tmp[ll]))
                    diff = (
                        abs(vec[0] - tmp[0])
                        + abs(vec[1] - tmp[1])
                        + abs(vec[2] - tmp[2])
                    )
                    for ll in range(3):
                        vec[ll] = vec[ll] % ngrid[ll]
                    if diff < 1e-4:
                        nruter[isym, i] = (
                            vec[0] + (vec[1] + vec[2] * ngrid[1]) * ngrid[0]
                        ) * natoms + ii
                        found = True
                        break
                if not found:
                    sys.exit(
                        "Error: equivalent atom not found for isym={}, atom={}".format(
                            isym, i
                        )
                    )
        return nruter
