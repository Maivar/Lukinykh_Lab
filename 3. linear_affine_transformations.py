import numpy as np
from scipy.linalg import lu, qr, svd
from scipy.linalg import eig as compute_eigen
from scipy.linalg import det as compute_determinant
import matplotlib.pyplot as plt

#  Task 1: Linear Mappings (LU decomposition, Eigenvalues, SVD)

def lu_decomposition(matrix):
    """ 
    Perform LU decomposition on a given matrix.

    Args:
        matrix (numpy.ndarray): The input matrix to decompose.

    Returns:
        tuple: (Permutation Matrix, Lower Matrix, Upper Matrix)
    """ 
    p, l, u = lu(matrix)
    return p, l, u

def eigen(matrix):
    """ 
    Compute eigenvalues and eigenvectors of the matrix.

    Args:
        matrix (numpy.ndarray): The input matrix.

    Returns:
        tuple: (Eigenvalues, Eigenvectors)
    """ 
    values, vectors = compute_eigen(matrix)
    return values, vectors

def svd_decomposition(matrix):
    """ 
    Perform Singular Value Decomposition (SVD) on a matrix.

    Args:
        matrix (numpy.ndarray): The input matrix.

    Returns:
        tuple: (U, S, Vt)
    """ 
    U, S, Vt = svd(matrix)
    return U, S, Vt

#  Task 2: Affine Transformations

def affine_transform(matrix, alpha_deg, scale, shear, translate):
    """ 
    Apply affine transformation on the matrix.

    Args:
        matrix (numpy.ndarray): The input matrix to transform.
        alpha_deg (float): The angle of rotation in degrees.
        scale (tuple): Scaling factors for x and y axes.
        shear (tuple): Shearing factors for x and y axes.
        translate (tuple): Translation factors for x and y axes.

    Returns:
        numpy.ndarray: The transformed matrix.
    """ 
    #  Convert angle to radians
    alpha_rad = np.deg2rad(alpha_deg)

    #  Define transformation matrices
    rotation_matrix = np.array([[np.cos(alpha_rad), -np.sin(alpha_rad)],
                                 [np.sin(alpha_rad), np.cos(alpha_rad)]])
    
    scale_matrix = np.diag(scale)
    
    shear_matrix = np.array([[1, shear[0]],
                             [shear[1], 1]])
    
    #  Translation is applied after matrix multiplication (in homogeneous coordinates)
    translation_matrix = np.array(translate).reshape(-1, 1)

    #  Apply transformations: Rotation -> Scaling -> Shearing
    transform_matrix = rotation_matrix @ scale_matrix @ shear_matrix
    
    #  Apply the transformation on the input matrix
    result = transform_matrix @ matrix + translation_matrix
    return result

#  Task 3: Visualization of the transformation

def plot_transformation(original, transformed):
    """ 
    Visualize the original and transformed matrices.

    Args:
        original (numpy.ndarray): The original matrix.
        transformed (numpy.ndarray): The transformed matrix.
    """ 
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original, cmap='viridis')
    plt.subplot(1, 2, 2)
    plt.title("Transformed")
    plt.imshow(transformed, cmap='viridis')
    plt.show()

#  Demonstrate the entire process

#  Example 1: LU Decomposition
matrix_lu = np.array([[4, 3], [6, 3]])
p, l, u = lu_decomposition(matrix_lu)
print("LU Decomposition")
print("Permutation Matrix (P):
", p)
print("Lower Triangular Matrix (L):
", l)
print("Upper Triangular Matrix (U):
", u)

#  Example 2: Eigenvalues and Eigenvectors
matrix_eigen = np.array([[4, 2], [1, 3]])
values, vectors = eigen(matrix_eigen)
print("\nEigenvalues and Eigenvectors")
print("Eigenvalues:\n", values)
print("Eigenvectors:\n", vectors)

#  Example 3: SVD Decomposition
matrix_svd = np.array([[1, 2], [3, 4]])
U, S, Vt = svd_decomposition(matrix_svd)
print("\nSingular Value Decomposition (SVD)")
print("U:\n", U)
print("S (singular values):\n", S)
print("Vt (transpose of V):\n", Vt)

#  Example 4: Affine Transformation (Scaling, Rotation, Shear, and Translation)
matrix_affine = np.array([[1, 2], [3, 4]])

#  Perform affine transformation with 45-degree rotation, scaling by 2, shear by 0.1, and translation by (1, 1)
transformed_affine = affine_transform(matrix_affine, 45, (2, 2), (0.1, 0.1), (1, 1))
print("\nAffine Transformation Result:")
print(transformed_affine)

#  Visualize the original vs transformed matrix
plot_transformation(matrix_affine, transformed_affine)
