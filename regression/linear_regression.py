import matplotlib.pyplot as plt
import datetime
import numpy as np
import sys
import os
# Add the parent directory (ML101) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database_injection.Database_Injection import Database_Injection

class LinearRegression:

    def __init__(self, ticker, start, end):
        self.ticker = ticker
        self.start_date = start
        self.end_date = end
        self.db_instance = Database_Injection(ticker, start, end)
    
    def compute_weighted_linear_regression(X, y, res):
        C = np.diag(res**2)
        w_wlr = np.linalg.inv(X.T @ np.linalg.inv(C) @ X) @ (X.T @ np.linalg.inv(C) @ y)

        return w_wlr
    
    def compute_weights_and_residuals(X, y):
        # compute weights
        w = np.linalg.inv(X.T @ X) @ X.T @ y
        # compute residuals
        res = y - X @ w

        return w, res

    def compute_prediction(X, w):
        return X @ w

if __name__ == "__main__":

    ### FOR DEBUGGING PURPOSES
    ticker = 'AAPL'
    end = datetime.date(2024, 10, 6) # last index
    start = datetime.date(2015, 1, 1) # 01/01/2015
    linreg = LinearRegression('/Users/costinchitic/Documents/Github/ML101/database_injection/long_stock_symbol_list.txt', start, end)