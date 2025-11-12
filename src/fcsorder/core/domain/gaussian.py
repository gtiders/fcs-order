import numpy as np

from numba import jit

EPS = 1e-10


def gaussian(a: np.ndarray):
    """
    Specialized version of Gaussian elimination.
    Matches the behavior of the previous Cython cdef gaussian(double[:,:] a):
    - Modifies `a` in place.
    - Returns (b, independent_indices)
      where b has shape (col, col) and dtype np.double,
      independent_indices is a 1-D array of dtype np.intc.
    """
    # Ensure float64 and contiguous array before passing to Numba core
    a = np.ascontiguousarray(a, dtype=np.double)
    return _gaussian_core(a)


@jit(nopython=True)
def _gaussian_core(a):
    # Core implementation expecting a 2D np.double array

    row, col = a.shape

    dependent = np.empty(col, dtype=np.intc)
    independent = np.empty(col, dtype=np.intc)
    b = np.zeros((col, col), dtype=np.double)

    irow = 0
    ndependent = 0
    nindependent = 0

    for k in range(min(row, col)):
        # Zero-out near-zero entries in column k (exactly like the Cython code)
        colk = a[:, k]
        colk[np.abs(colk) < EPS] = 0.0

        # Row swapping to move a larger pivot into position irow, emulating original loop
        if irow < row:
            for i in range(irow + 1, row):
                if abs(a[i, k]) - abs(a[irow, k]) > EPS:
                    tmp = a[irow, k:col].copy()
                    a[irow, k:col] = a[i, k:col]
                    a[i, k:col] = tmp

        if irow < row and abs(a[irow, k]) > EPS:
            # This column is dependent
            dependent[ndependent] = k
            ndependent += 1

            # Normalize pivot row entries j in (k+1 .. col-1)
            piv = a[irow, k]
            if col - 1 > k:
                a[irow, k + 1 : col] /= piv
            a[irow, k] = 1.0

            # Eliminate other rows in column k
            for i in range(row):
                if i == irow:
                    continue
                if col - 1 > k:
                    a[i, k + 1 : col] -= a[i, k] * a[irow, k + 1 : col] / a[irow, k]
                a[i, k] = 0.0

            if irow < row - 1:
                irow += 1
        else:
            # This column is independent
            independent[nindependent] = k
            nindependent += 1

    # Build b from reduced matrix
    for j in range(nindependent):
        for i in range(ndependent):
            b[dependent[i], j] = -a[i, independent[j]]
        b[independent[j], j] = 1.0

    return b, independent[:nindependent]
