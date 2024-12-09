import numpy as np
from scipy.linalg import lu, svd, qr
from scipy.linalg import det, eig
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

def lu_decomposition(matrix: numpy.ndarray):
    """ Perform LU decomposition.""" 
    P, L, U = lu(x)
    return P, L, U

def qr_decomposition(matrix: numpy.ndarray):
    """ Perform QR decomposition.""" 
    Q, R = qr(x)
    return Q, R

def determinant(matrix: numpy.ndarray):
    """ Calculate the determinant.""" 
    return det(x)

def eigen(matrix: numpy.ndarray):
    """ Compute eigenvalues and eigenvectors.""" 
    values, vectors = eig(x)
    return values, vectors

def singular_value_decomposition(matrix: numpy.ndarray):
    """ Perform Singular Value Decomposition (SVD).""" 
    U, S, VT = svd(x)
    return U, S, VT

def negative_matrix(matrix: numpy.ndarray):
    """ Negate each element in the matrix.""" 
    return -x

def reverse_matrix(matrix: numpy.ndarray):
    """ Reverse the order of elements in the matrix.""" 
    return np.flip(x)

def affine_transform(
    matrix: numpy.ndarray, alpha_deg: float, scale: tuple, shear: tuple, translate: tuple
):
    """ Perform affine transformation.""" 
    alpha_rad = np.radians(alpha_deg)
    rotation_matrix = np.array([
        [np.cos(alpha_rad), -np.sin(alpha_rad)],
        [np.sin(alpha_rad), np.cos(alpha_rad)]
    ])
    scaling_matrix = np.array([
        [scale[0], shear[0]],
        [shear[1], scale[1]]
    ])
    transformation_matrix = np.dot(rotation_matrix, scaling_matrix)
    transformed = np.dot(transformation_matrix, x) + np.array(translate).reshape(-1, 1)
    return transformed

def low_rank_approximation(A, rank):
    """ Compute low-rank approximation of a matrix.""" 
    U, S, VT = singular_value_decomposition(A)
    S_reduced = np.diag(S[:rank])
    A_approx = U[:, :rank] @ S_reduced @ VT[:rank, :]
    return A_approx

def matrix_factorization(X):
    """ Perform Non-Negative Matrix Factorization.""" 
    model = NMF(n_components=2, init='random', random_state=42)
    W = model.fit_transform(np.nan_to_num(X))
    H = model.components_
    X_approx = np.dot(W, H)
    return X_approx

def main():
    """ Main function to demonstrate tasks.""" 
    A = np.array([[1, 2], [3, 4], [5, 6]])

    #  LU Decomposition
    P, L, U = lu_decomposition(A)
    print("LU Decomposition:")
    print("P:", P)
    print("L:", L)
    print("U:", U)

    #  QR Decomposition
    Q, R = qr_decomposition(A)
    print("QR Decomposition:")
    print("Q:", Q)
    print("R:", R)

    #  Determinant
    det_A = determinant(A[:2, :2])  #  Must be square
    print("Determinant:", det_A)

    #  Eigenvalues and Eigenvectors
    values, vectors = eigen(A[:2, :2])
    print("Eigenvalues:", values)
    print("Eigenvectors:", vectors)

    #  SVD
    U, S, VT = singular_value_decomposition(A)
    print("Singular Value Decomposition:")
    print("U:", U)
    print("S:", S)
    print("V^T:", VT)

    #  Low-Rank Approximation
    rank = 1
    A_approx = low_rank_approximation(A, rank)
    print("Low-Rank Approximation:")
    print(A_approx)

    #  Negation
    neg_A = negative_matrix(A)
    print("Negative Matrimatrix:", neg_A)

    #  Reverse
    rev_A = reverse_matrix(A)
    print("Reversed Matrimatrix:", rev_A)

    #  Affine Transformation
    transformed = affine_transform(A.T, 45, (1, 1), (0, 0), (1, 1))
    print("Affine Transformed Matrimatrix:", transformed)

    #  Matrix Factorization
    X = np.array([[5, 3, np.nan], [4, np.nan, 2], [1, 1, 1]])
    X_approx = matrix_factorization(X)
    print("Matrix Factorization:", X_approx)

    #  Visualize Low-Rank Approximation Difference
    plt.imshow(A - A_approx, cmap='hot', interpolation='nearest')
    plt.title("Difference between Original and Low-Rank Approximation")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
