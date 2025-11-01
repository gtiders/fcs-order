"""
Sparse 6D tensor implementation using scipy sparse matrices.
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple
import itertools


class SparseTensor6D:
    """
    A sparse 6D tensor implementation using dictionary storage with scipy support.
    Supports high-dimensional sparse arrays and slice operations.
    """

    def __init__(
        self, shape: Tuple[int, int, int, int, int, int], threshold: float = 1e-4
    ):
        """
        Initialize sparse 6D tensor.

        Args:
            shape: Shape of the 6D tensor (dim1, dim2, dim3, dim4, dim5, dim6)
            threshold: Values smaller than this are treated as zero
        """
        self.shape = shape
        self.threshold = threshold
        self.ndim = 6

        # Use dictionary to store non-zero elements for efficient access
        self._data = {}  # {(i1,i2,i3,i4,i5,i6): value}

        # For efficient slicing and operations, also maintain scipy COO format
        self._coo_matrix = None
        self._coo_needs_update = True

    def _flatten_index(self, indices: Tuple[int, int, int, int, int, int]) -> int:
        """Convert 6D indices to flattened 1D index."""
        i1, i2, i3, i4, i5, i6 = indices
        return (
            (((i1 * self.shape[1] + i2) * self.shape[2] + i3) * self.shape[3] + i4)
            * self.shape[4]
            + i5
        ) * self.shape[5] + i6

    def _unflatten_index(self, flat_index: int) -> Tuple[int, int, int, int, int, int]:
        """Convert flattened 1D index back to 6D indices."""
        i6 = flat_index % self.shape[5]
        flat_index //= self.shape[5]
        i5 = flat_index % self.shape[4]
        flat_index //= self.shape[4]
        i4 = flat_index % self.shape[3]
        flat_index //= self.shape[3]
        i3 = flat_index % self.shape[2]
        flat_index //= self.shape[2]
        i2 = flat_index % self.shape[1]
        i1 = flat_index // self.shape[1]
        return (i1, i2, i3, i4, i5, i6)

    def __getitem__(self, indices):
        """Get value at specified indices."""
        if isinstance(indices, tuple) and len(indices) == 6:
            # Check if any element is a slice
            has_slice = any(isinstance(idx, slice) for idx in indices)
            if has_slice:
                return self._get_slice(indices)
            else:
                # Single element access
                key = tuple(int(i) for i in indices)
                # Check bounds
                for i, (idx, dim) in enumerate(zip(key, self.shape)):
                    if not (0 <= idx < dim):
                        raise IndexError(
                            f"Index {idx} out of bounds for axis {i} with size {dim}"
                        )
                return self._data.get(key, 0.0)
        else:
            raise IndexError("SparseTensor6D requires 6 indices")

    def __setitem__(self, indices, value):
        """Set value at specified indices."""
        if isinstance(indices, tuple) and len(indices) == 6:
            # Check if any element is a slice
            has_slice = any(isinstance(idx, slice) for idx in indices)
            if has_slice:
                self._set_slice(indices, value)
            else:
                # Single element assignment
                key = tuple(int(i) for i in indices)
                # Check bounds
                for i, (idx, dim) in enumerate(zip(key, self.shape)):
                    if not (0 <= idx < dim):
                        raise IndexError(
                            f"Index {idx} out of bounds for axis {i} with size {dim}"
                        )

                if abs(value) > self.threshold:
                    self._data[key] = float(value)
                else:
                    # Remove from storage if value is below threshold
                    if key in self._data:
                        del self._data[key]

                # Mark COO matrix as needing update
                self._coo_needs_update = True
        else:
            raise IndexError("SparseTensor6D requires 6 indices")

    def _get_slice(self, key):
        """Handle slicing operations."""
        # Parse the slice specification
        slice_dims, fixed_dims, slice_shapes = self._parse_slice_key(key)

        # Create result array
        result_shape = slice_shapes
        result = np.zeros(result_shape, dtype=np.float64)

        # Fill in values from sparse storage
        for stored_key, value in self._data.items():
            if self._key_matches_slice(stored_key, fixed_dims):
                result_idx = tuple(stored_key[i] for i in slice_dims)
                result[result_idx] = value

        return result

    def _set_slice(self, key, values):
        """Handle slice assignment operations."""
        # Parse the slice specification
        slice_dims, fixed_dims, slice_shapes = self._parse_slice_key(key)

        # Handle scalar assignment (e.g., tensor[:,:,:,:,:,:] = 0.)
        if np.isscalar(values):
            scalar_value = float(values)
            # Assign scalar to all positions in the slice
            slice_ranges = [range(slice_shapes[i]) for i in range(len(slice_shapes))]

            for idx_combo in itertools.product(*slice_ranges):
                # Build complete 6D index
                full_key = [0] * 6
                for i, dim in enumerate(slice_dims):
                    full_key[dim] = idx_combo[i]
                for dim, val in fixed_dims:
                    full_key[dim] = val

                # Apply threshold filtering
                flat_key = tuple(full_key)
                if abs(scalar_value) > self.threshold:
                    self._data[flat_key] = scalar_value
                else:
                    self._data.pop(flat_key, None)
        else:
            # Handle array assignment
            # Convert values to numpy array
            values = np.asarray(values, dtype=np.float64)

            # Validate shape compatibility
            if values.shape != tuple(slice_shapes):
                raise ValueError(
                    f"Shape mismatch: expected {tuple(slice_shapes)}, got {values.shape}"
                )

            # Assign all values, applying threshold filtering
            slice_ranges = [range(slice_shapes[i]) for i in range(len(slice_shapes))]

            for idx_combo in itertools.product(*slice_ranges):
                # Build complete 6D index
                full_key = [0] * 6
                for i, dim in enumerate(slice_dims):
                    full_key[dim] = idx_combo[i]
                for dim, val in fixed_dims:
                    full_key[dim] = val

                # Get the value
                value = values[idx_combo]

                # Apply threshold filtering
                flat_key = tuple(full_key)
                if abs(value) > self.threshold:
                    self._data[flat_key] = float(value)
                else:
                    self._data.pop(flat_key, None)

        # Mark COO matrix as needing update
        self._coo_needs_update = True

    def _parse_slice_key(self, key):
        """Parse slice key into slice dimensions, fixed dimensions, and shapes."""
        if not isinstance(key, tuple):
            key = (key,)

        slice_dims = []
        fixed_dims = []
        slice_shapes = []

        for i, k in enumerate(key):
            if isinstance(k, slice) or k == slice(None):
                slice_dims.append(i)
                # Handle slice ranges
                start = k.start if k.start is not None else 0
                stop = k.stop if k.stop is not None else self.shape[i]
                step = k.step if k.step is not None else 1
                slice_shapes.append(len(range(start, stop, step)))
            elif isinstance(k, (int, np.integer)):
                fixed_dims.append((i, int(k)))
            else:
                raise TypeError(f"Invalid key component {k} of type {type(k)}")

        return slice_dims, fixed_dims, slice_shapes

    def _key_matches_slice(self, flat_key, fixed_dims):
        """Check if a key matches the fixed dimensions of a slice."""
        for dim, val in fixed_dims:
            if flat_key[dim] != val:
                return False
        return True

    def to_dense(self) -> np.ndarray:
        """Convert to dense numpy array."""
        dense = np.zeros(self.shape, dtype=np.float64)
        for key, value in self._data.items():
            dense[key] = value
        return dense

    def to_coo_matrix(self) -> sp.coo_matrix:
        """Convert to scipy COO matrix (flattened)."""
        self._update_coo_matrix()
        return self._coo_matrix

    def _update_coo_matrix(self):
        """Update the internal COO matrix representation."""
        if not self._coo_needs_update:
            return

        if not self._data:
            # Create empty COO matrix
            total_size = np.prod(self.shape)
            self._coo_matrix = sp.coo_matrix(
                ([], ([], [])), shape=(total_size, 1), dtype=np.float64
            )
        else:
            # Convert dictionary to COO format
            flat_indices = []
            values = []

            for key, value in self._data.items():
                flat_idx = self._flatten_index(key)
                flat_indices.append(flat_idx)
                values.append(value)

            total_size = np.prod(self.shape)
            self._coo_matrix = sp.coo_matrix(
                (values, (flat_indices, [0] * len(values))),
                shape=(total_size, 1),
                dtype=np.float64,
            )

        self._coo_needs_update = False

    def fill(self, value: float):
        """Fill entire tensor with a value (respects threshold)."""
        if abs(value) <= self.threshold:
            self.clear()
            return

        self._data.clear()
        # Iterate through all indices and set value
        for indices in itertools.product(*[range(dim) for dim in self.shape]):
            self._data[indices] = float(value)

        self._coo_needs_update = True

    def clear(self):
        """Clear all data."""
        self._data.clear()
        self._coo_matrix = None
        self._coo_needs_update = True

    def get_sparsity(self) -> float:
        """Calculate sparsity ratio."""
        total_elements = np.prod(self.shape)
        non_zero_elements = len(self._data)
        return 1.0 - (non_zero_elements / total_elements)

    def nonzero(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return arrays of indices for non-zero elements."""
        if not self._data:
            return (
                np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=int),
                np.array([], dtype=int),
            )

        indices = list(self._data.keys())
        i1, i2, i3, i4, i5, i6 = zip(*indices)
        return (
            np.array(i1, dtype=int),
            np.array(i2, dtype=int),
            np.array(i3, dtype=int),
            np.array(i4, dtype=int),
            np.array(i5, dtype=int),
            np.array(i6, dtype=int),
        )

    def values(self) -> np.ndarray:
        """Return array of non-zero values."""
        return np.array(list(self._data.values()), dtype=np.float64)

    def __repr__(self):
        return f"SparseTensor6D(shape={self.shape}, non_zero={len(self._data)}, sparsity={self.get_sparsity():.2%})"
