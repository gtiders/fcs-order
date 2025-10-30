cdef extern from "include/spglib.h":
  ctypedef struct SpglibDataset:
    int spacegroup_number
    int hall_number
    char international_symbol[11]
    char hall_symbol[17]
    double transformation_matrix[4][4]
    double origin_shift[4]
    int n_operations
    int (*rotations)[4][4]
    double (*translations)[4]
    int n_atoms
    int *wyckoffs
    int *equivalent_atoms
  SpglibDataset *spg_get_dataset(double lattice[3][3],
                                 double position[][3],
                                 int types[],
                                 int num_atom,
                                 double symprec)
  void spg_free_dataset(SpglibDataset *dataset)
