
#  Lab: Exploring Regularization Techniques in Machine Learning

#  Import necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.metrics import mean_squared_error, accuracy_score

#  Step 1: Preprocessing Function
def preprocess(X: numpy.ndarray, y: numpy.ndarray):
    """ Preprocess the data by scaling and splitting into train and test sets.""" 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2)

#  Step 2: Load and preprocess data
#  Load regression data (diabetes dataset)
diabetes = load_diabetes()
X_reg, y_reg = diabetes.data, diabetes.target
X_train_reg, X_test_reg, y_train_reg, y_test_reg = preprocess(X_reg, y_reg)

#  Load classification data (breast cancer dataset)
cancer = load_breast_cancer()
X_clf, y_clf = cancer.data, cancer.target
X_train_clf, X_test_clf, y_train_clf, y_test_clf = preprocess(X_clf, y_clf)

#  Step 3: Linear Regression (Baseline)
lin_reg = LinearRegression()
lin_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = lin_reg.predict(X_test_reg)
mse_lin_reg = mean_squared_error(y_test_reg, y_pred_reg)
print(f"Linear Regression MSE: {mse_lin_reg:.4f}")

#  Step 4: Ridge Regression with Hyperparameter Tuning
ridge = Ridge()
param_grid_ridge = {"alpha": [0.1, 1, 10, 100]}
ridge_cv = GridSearchCV(ridge, param_grid_ridge, scoring="neg_mean_squared_error", cv=5)
ridge_cv.fit(X_train_reg, y_train_reg)
y_pred_ridge = ridge_cv.best_estimator_.predict(X_test_reg)
mse_ridge = mean_squared_error(y_test_reg, y_pred_ridge)
print(f"Ridge Regression MSE: {mse_ridge:.4f}, Best Alpha: {ridge_cv.best_params_['alpha']}")

#  Step 5: Lasso Regression with Hyperparameter Tuning
lasso = Lasso()
param_grid_lasso = {"alpha": [0.01, 0.1, 1, 10]}
lasso_cv = GridSearchCV(lasso, param_grid_lasso, scoring="neg_mean_squared_error", cv=5)
lasso_cv.fit(X_train_reg, y_train_reg)
y_pred_lasso = lasso_cv.best_estimator_.predict(X_test_reg)
mse_lasso = mean_squared_error(y_test_reg, y_pred_lasso)
print(f"Lasso Regression MSE: {mse_lasso:.4f}, Best Alpha: {lasso_cv.best_params_['alpha']}")

#  Step 6: Logistic Regression (Baseline)
log_reg = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)
log_reg.fit(X_train_clf, y_train_clf)
y_pred_clf = log_reg.predict(X_test_clf)
accuracy_log_reg = accuracy_score(y_test_clf, y_pred_clf)
print(f"Logistic Regression Accuracy (No Regularization): {accuracy_log_reg:.4f}")

#  Step 7: Logistic Regression with L2 Regularization
log_reg_l2 = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)
param_grid_l2 = {"C": [0.1, 1, 10, 100]}
log_reg_l2_cv = GridSearchCV(log_reg_l2, param_grid_l2, scoring="accuracy", cv=5)
log_reg_l2_cv.fit(X_train_clf, y_train_clf)
y_pred_l2 = log_reg_l2_cv.best_estimator_.predict(X_test_clf)
accuracy_l2 = accuracy_score(y_test_clf, y_pred_l2)
print(f"Logistic Regression L2 Accuracy: {accuracy_l2:.4f}, Best C: {log_reg_l2_cv.best_params_['C']}")

#  Step 8: Logistic Regression with L1 Regularization
log_reg_l1 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
param_grid_l1 = {"C": [0.1, 1, 10, 100]}
log_reg_l1_cv = GridSearchCV(log_reg_l1, param_grid_l1, scoring="accuracy", cv=5)
log_reg_l1_cv.fit(X_train_clf, y_train_clf)
y_pred_l1 = log_reg_l1_cv.best_estimator_.predict(X_test_clf)
accuracy_l1 = accuracy_score(y_test_clf, y_pred_l1)
print(f"Logistic Regression L1 Accuracy: {accuracy_l1:.4f}, Best C: {log_reg_l1_cv.best_params_['C']}")
