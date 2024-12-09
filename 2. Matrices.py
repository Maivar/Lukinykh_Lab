
import numpy as np
from scipy.linalg import solve, svd
from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#  Task 1: Matrix Operations and Linear Algebra with NumPy

#  1.1 Matrix Creation
A = np.array([[3, 2, 1], [4, 5, 6], [7, 8, 9]])  #  3x3 Matrix A
b = np.array([1, 2, 3])  #  3x1 Vector b
print("Matrix A:\n", A)
print("Vector b:", b)

#  1.2 Matrix Addition and Subtraction
C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  #  Identity matrix
A_plus_C = A + C
A_minus_C = A - C
print("\nMatrix A + C:\n", A_plus_C)
print("Matrix A - C:\n", A_minus_C)

#  1.3 Matrix Multiplication
A_T = A.T  #  Transpose of A
A_times_A_T = np.dot(A, A_T)
print("\nMatrix A * A^T:\n", A_times_A_T)

#  1.4 Matrix Inversion
try:
    A_inv = np.linalg.inv(A)  #  Inverse of A
    print("\nInverse of A:\n", A_inv)
except np.linalg.LinAlgError:
    print("\nMatrix A is singular and cannot be inverted.")

#  1.5 Solving a System of Linear Equations
#  We want to solve Ax = b, where A is a 3x3 matrix and b is a 3x1 vector
x = np.dot(A_inv, b)  #  Using A_inv to solve for x
print("\nSolution to Ax = b using A_inv:\n", x)

#  Task 2: Solving Linear Systems with SciPy

#  2.1 Using scipy.linalg.solve
x_scipy = solve(A, b)  #  Directly solve the system using SciPy
print("\nSolution to Ax = b using scipy.linalg.solve:\n", x_scipy)

#  Task 3: Matrix Factorization and Regularization

#  3.1 Regularization in Linear Regression

#  Generate synthetic regression data (100 samples, 2 features)
X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
print("\nFirst 5 samples of X:\n", X[:5])
print("First 5 target values of y:\n", y[:5])

#  3.2 Ridge Regression (L2 Regularization)
ridge_reg = Ridge(alpha=1.0)  #  L2 regularization
ridge_reg.fit(X, y)  #  Fit the model to the data
print("\nRidge Regression Coefficients:", ridge_reg.coef_)

#  3.3 Lasso Regression (L1 Regularization)
lasso_reg = Lasso(alpha=0.1)  #  L1 regularization
lasso_reg.fit(X, y)  #  Fit the model to the data
print("Lasso Regression Coefficients:", lasso_reg.coef_)

#  Task 4: Matrix Factorization (SVD)

#  4.1 Singular Value Decomposition (SVD)
matrix = np.random.rand(3, 3)  #  Create a random 3x3 matrix
U, S, Vt = svd(matrix)  #  Perform SVD
print("\nSVD - U Matrimatrix:\n", U)
print("SVD - Singular Values:\n", S)
print("SVD - Vt Matrimatrix:\n", Vt)

#  4.2 Matrix Reconstruction from SVD
reconstructed_matrix = np.dot(U, np.dot(np.diag(S), Vt))  #  Reconstruct the matrix
print("\nReconstructed Matrix from SVD:\n", reconstructed_matrix)

#  Task 5: Handling Incorrect or Missing Data

#  5.1 Handling Missing Data by Imputation
A_with_nan = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])  #  Matrix with NaN values
print("\nMatrix with Missing Data (NaN):\n", A_with_nan)

#  Replace NaN values with the column mean
col_means = np.nanmean(A_with_nan, axis=0)  #  Compute column means ignoring NaN values
inds = np.where(np.isnan(A_with_nan))  #  Find indices of NaNs
A_with_nan[inds] = np.take(col_means, inds[1])  #  Replace NaNs with the column means
print("\nMatrix after Imputation:\n", A_with_nan)

#  Task 6: Regularization in Machine Learning Models

#  6.1 Train-Test Split for Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  6.2 Ridge Regression Model (L2 Regularization)
ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)  #  Fit the model to the training data

#  Predict using the trained model
y_pred_ridge = ridge.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)  #  Evaluate performance using MSE
print("\nRidge Regression Mean Squared Error:", mse_ridge)

#  6.3 Lasso Regression Model (L1 Regularization)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)  #  Fit the model to the training data

#  Predict using the trained model
y_pred_lasso = lasso.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)  #  Evaluate performance using MSE
print("Lasso Regression Mean Squared Error:", mse_lasso)
