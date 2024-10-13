import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
import os
# Add the parent directory (ML101) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from regression.linear_regression import LinearRegression
from database_injection.Database_Injection import Database_Injection

"""
Principal Component Analysis (PCA) is a dimensionality reduction technique that is used to reduce the number of features in a dataset.
PCA is used to reduce the number of features in a dataset by projecting the data onto a lower-dimensional subspace.
https://costinchitic.co/notes/principal_component_analysis
"""

class PCA:

    def __init__(self, no_components, ticker, start, end):
        self.no_components = no_components
        self.ticker = ticker
        self.start_date = start
        self.end_date = end
        self.db_instance = Database_Injection(ticker, start, end)
        self.linear_regression_instance = LinearRegression(ticker, start, end)

    def prepare_data(self, df):
        # Prepare data for linear regression
        # Here, we use the 'date' as feature and 'close_price' as the target variable.
        df['date_ordinal'] = df['date'].map(datetime.date.toordinal) # convert date to numerical input

        # Feature matrix (X) will now include multiple features
        X = df[['date_ordinal', 'open_price', 'volume']].values
        y = df['close_price'].values  # target variable (closing price). Need to add more.
        return X, y
    
    def standardize(self, X):
        # Standardize the data
        Z = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        return Z
    
    def compute_covariance_matrix(self, Z):
        # Compute the covariance matrix
        mean = np.mean(Z, axis=0)
        cov_matrix = (Z - mean).T @ (Z - mean) / (Z.shape[0] - 1)

    def is_upper_triangular(self, matrix):
        # Check if a matrix is upper triangular
        for i in range(0, len(matrix)):
            for j in range(0, i):
                if matrix[i, j] != 0:
                    return False
        return True
    
    def QR_Decomp_Householder(self, A):
        # Source: https://kwokanthony.medium.com/detailed-explanation-with-example-on-qr-decomposition-by-householder-transformation-5e964d7f7656
        # "Q" in QR decomp represents an orthogonal matrix => Q^T * Q = I
        # "R" in QR decomp represents an upper triangular matrix

        rows, cols = A.shape
        Q = np.eye(rows)
        R = np.copy(A)

        for i in range(rows):
            H = np.eye(rows)
            a = R[i:, 1]
            norm_a = np.linalg.norm(a)

            if a[0] < 0.0:
                norm_a = -norm_a

            w = a / (a[0] + norm_a)
            w[0] = 1.0
            h = np.eye(len(a)) # H reflection
            h -= 2.0/(w.T @ w) * (w[:, None] @ w[None, :]) # same as np.outer(w, w)

            # First understand and the write code.


    def compute_eigenvalues_eigenvectors(self, cov_matrix):
        # cov_matrix must be square, symmetric
        # Source: https://jamesmccaffrey.wordpress.com/2023/12/22/eigenvalues-and-eigenvectors-from-scratch-using-python/
        
        n = len(cov_matrix)
        X = np.copy(cov_matrix) # Eigen Vectors
        pq = np.eye(n) 
        max_ct = 10000 # max number of iterations

        
