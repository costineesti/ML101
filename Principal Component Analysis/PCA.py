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
        return cov_matrix

    def is_upper_triangular(self, matrix, tol):
        # Check if a matrix is upper triangular
        for i in range(0, len(matrix)):
            for j in range(0, i):
                if np.abs(matrix[i, j]) > tol:
                    return False
        return True
    
    def QR_Decomp_Householder(self, A):
        # Source: https://kwokanthony.medium.com/detailed-explanation-with-example-on-qr-decomposition-by-householder-transformation-5e964d7f7656
        # "Q" in QR decomp represents an orthogonal matrix => Q^T * Q = I
        # "R" in QR decomp represents an upper triangular matrix

        rows, cols = A.shape
        Q = np.eye(rows)
        R = np.copy(A)
        if cols == rows: 
            end = cols-1
        else: 
            end = cols

        for i in range(0, end):
            H = np.eye(rows)
            a = R[i:, i]
            norm_a = np.linalg.norm(a)

            if a[0] < 0.0:
                norm_a = -norm_a

            w = a / (a[0] + norm_a)
            w[0] = 1.0
            h = np.eye(len(a)) # H reflection
            h -= 2.0/(w.T @ w) * (w[:, None] @ w[None, :]) # same as np.outer(w, w)

            H[i:, i:] = h # Update the H matrix
            R = H @ R
            Q = Q @ H

        return Q, R


    def compute_eigenvalues_eigenvectors(self, cov_matrix):
        # cov_matrix must be square, symmetric
        # Source: https://jamesmccaffrey.wordpress.com/2023/12/22/eigenvalues-and-eigenvectors-from-scratch-using-python/
        
        n = len(cov_matrix)
        X = np.copy(cov_matrix) # Eigen Vectors
        pq = np.eye(n) 
        max_ct = 10000 # max number of iterations

        ct = 0
        while ct < max_ct:
            Q, R = self.QR_Decomp_Householder(X)
            pq = pq @ Q
            X = R @ Q
            ct += 1

            if self.is_upper_triangular(X, 1.0e-8):
                break

        if ct == max_ct:
            print("Failed to converge")

        # Eigenvalues are diagonal elements of X
        eigenvalues = np.diag(X)
        # Eigenvectors are columns of pq
        eigenvectors = np.copy(pq)

        return (eigenvalues, eigenvectors)
    
    def explained_variance_ratio(self, eigenvalues):
        # Compute the explained variance ratio
        return eigenvalues / np.sum(eigenvalues)
    
    def sort_eigenvalues_eigenvectors(self, eigenvalues, eigenvectors):
        # Sort them in descending order
        idx = np.argsort(eigenvalues)[::-1] # ::-1 reverses the array [1,2,3] -> [3,2,1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx] # Another method: eigenvectors = np.array(eigenvectors[idx]).T
        return eigenvalues, eigenvectors

    def plot_eigenvalues(self, eigenvalues):
        # Plot the eigenvalues
        plt.bar(range(len(eigenvalues)), eigenvalues)
        plt.xlabel('Principal Components')
        plt.ylabel('Percentage of explained variances')
        plt.title('Principal Components Analysis')
        plt.show()

    def select_principal_components(self, eigenvectors):
        # Select the top no_components eigenvectors
        principal_components = eigenvectors[:, :self.no_components]
        return principal_components

    def project_data(self, Z, principal_components):
        # Project the standardized data onto the principal components
        projected_data = Z @ principal_components
        return projected_data
    
    def _run(self):
        ticker_stock_data = self.db_instance.fetch_ticker_data(ticker) # Get the stock info in DataFrame format
        X, y = self.prepare_data(ticker_stock_data) # Prepare the data for PCA
        Z = self.standardize(X) # Standardize the data
        # Compute the covariance matrix in order to substract the eigenvalues and eigenvectors
        cov_matrix = self.compute_covariance_matrix(Z)
        eigenvalues, eigenvectors = self.compute_eigenvalues_eigenvectors(cov_matrix)
        explained_variance_ratio = self.explained_variance_ratio(eigenvalues)
        # I can see that the first axis holds ~79% of the information, while the second axis holds ~19.3% of the information. That leaves 3rd with 1.7%.
        # In conclusion, I can reduce the number of features from 3 to 2, while still retaining ~98% of the information. Case of AAPL
        trust_percentage = np.sum(explained_variance_ratio[:no_components]) * 100
        print(f"Trust percentage: {trust_percentage}%")
        self.plot_eigenvalues(eigenvalues)

        sorted_eigvalues, sorted_eigvectors = self.sort_eigenvalues_eigenvectors(eigenvalues, eigenvectors)
        principal_components = self.select_principal_components(sorted_eigvectors)
        # Now I can project the data onto the principal components
        projected_data = self.project_data(Z, principal_components)

if __name__ == "__main__":

    ticker = 'NVDA'
    end = datetime.date.today() # last index
    start = datetime.date(2015, 1, 1) # 01/01/2015
    no_components = 2 # Number of principal components to keep
    myPCA = PCA(no_components, '/Users/costinchitic/Documents/Github/ML101/database_injection/long_stock_symbol_list.txt', start, end)
    myPCA._run()