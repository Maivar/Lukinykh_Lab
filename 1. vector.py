import numpy as np
from scipy import sparse
from typing import Sequence

#  Function to create a random dense column vector with the specified dimension
def get_vector(dim: int) -> numpy.ndarray:
    """ Create random column vector with dimension dim.

    Args:
        dim (int): vector dimension.

    Returns:
        numpy.ndarray: column vector.
    """ 
    return np.random.rand(dim, 1)

#  Function to create a random sparse column vector with the specified dimension
def get_sparse_vector(dim: int) -> sparse.coo_matrimatrix:
    """ Create random sparse column vector with dimension dim.

    Args:
        dim (int): vector dimension.

    Returns:
        sparse.coo_matrimatrix: sparse column vector.
    """ 
    #  Generate random data for approximately half of the vector's elements
    data = np.random.rand(dim // 2)
    #  Randomly choose indices for non-zero elements
    row_indices = np.random.choice(dim, size=dim // 2, replace=False)
    #  Column indices are all zeros because this is a column vector
    col_indices = np.zeros(dim // 2, dtype=int)
    #  Create and return the sparse matrix in COO format
    return sparse.coo_matrix((data, (row_indices, col_indices)), shape=(dim, 1))

#  Function to perform element-wise addition of two vectors
def add(matrix: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
    """ Vector addition. 

    Args:
        x (numpy.ndarray): 1st vector.
        y (numpy.ndarray): 2nd vector.

    Returns:
        numpy.ndarray: vector sum.
    """ 
    #  Check if the vectors have the same shape
    if x.shape != y.shape:
        raise ValueError("Vectors must have the same shape.")
    #  Return the element-wise sum of the vectors
    return x + y

#  Function to multiply a vector by a scalar
def scalar_multiplication(matrix: numpy.ndarray, a: float) -> numpy.ndarray:
    """ Vector multiplication by scalar.

    Args:
        x (numpy.ndarray): vector.
        a (float): scalar.

    Returns:
        numpy.ndarray: multiplied vector.
    """ 
    #  Multiply each element of the vector by the scalar and return the result
    return a * x

#  Function to compute the linear combination of a sequence of vectors and coefficients
def linear_combination(vectors: Sequence[numpy.ndarray], coeffs: Sequence[float]) -> numpy.ndarray:
    """ Linear combination of vectors.

    Args:
        vectors (Sequence[numpy.ndarray]): list of vectors of len N.
        coeffs (Sequence[float]): list of coefficients of len N.

    Returns:
        numpy.ndarray: linear combination of vectors.
    """ 
    #  Ensure the number of vectors matches the number of coefficients
    if len(vectors) != len(coeffs):
        raise ValueError("Number of vectors and coefficients must match.")
    #  Compute and return the weighted sum of the vectors
    return sum(coeff * vector for coeff, vector in zip(coeffs, vectors))
