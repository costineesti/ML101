import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
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
    
    def compute_weighted_linear_regression(self, X, y, res):
        C = np.diag(res**2)
        w_wlr = np.linalg.inv(X.T @ np.linalg.inv(C) @ X) @ (X.T @ np.linalg.inv(C) @ y)
        return w_wlr
    
    def compute_weights_and_residuals(self, X, y):
        # compute weights
        w = np.linalg.inv(X.T @ X) @ X.T @ y # coefficients of y = ax + b. So we have a and b now! 
        # compute residuals
        res = (y - X @ w)**2
        return w, res

    def compute_prediction(self, X, w):
        return X @ w
    
    def prepare_data(self, df):
        # Prepare data for linear regression
        # Here, we use the 'date' as feature and 'close_price' as the target variable.
        df['date_ordinal'] = df['date'].map(datetime.date.toordinal) # convert date to numerical input

        X = np.vstack([df['date_ordinal'].values, np.ones(len(df))]).T  # feature matrix
        y = df['close_price'].values  # target variable (closing price). Need to add more.
        return X, y
    
    def plot_prediction(self, simple_prediction, weighted_prediction, stas_prediction, target, df, confidence = 0.95):
        df['date'] = pd.to_datetime(df['date']) # Ensuring the 'date' column is in datetime format
        residuals = target - simple_prediction
        z = 1.96  # For 95% confidence level

        upper = simple_prediction + z*np.std(residuals)
        lower = simple_prediction - z*np.std(residuals)

        plt.figure(figsize=(10,6))
        plt.plot(df['date'], df['close_price'], label = f'{ticker} Closing Price')
        plt.plot(df['date'], simple_prediction, label = "Predicted Prices", linestyle='--')
        plt.plot(df['date'], weighted_prediction, label = "Weighted Predicted Prices", linestyle='--')
        plt.plot(df['date'], stas_prediction, label = "STAS Predicted Prices", linestyle='--')

        # Plot bounds:
        plt.plot(df['date'], lower, label = "LOWER BOUND", linestyle='--')
        plt.plot(df['date'], upper, label = "UPPER BOUND", linestyle='--')

        plt.fill_between(df['date'], lower, upper, color='orange', alpha=0.3, label='95% Confidence Interval')

        plt.title(f'{ticker} Closing prices from {self.start_date} to {self.end_date}')
        plt.xlabel('Date') 
        plt.ylabel('Closing Price ($USD)')
        plt.grid(True)
        plt.legend()

        plt.show()

    def STAS_LR(self, df):
        from sklearn.linear_model import LinearRegression
        stas_lr = LinearRegression()
        df['date_ordinal'] = df['date'].map(datetime.date.toordinal) # convert date to numerical input

        stas_lr.fit(df[['date_ordinal']], df[['close_price']])  # Train the model
        stas_prediction = stas_lr.predict(df[['date_ordinal']])  # Make predictions
        r2_score_stas = stas_lr.score(df[['date_ordinal']], df[['close_price']])

        return stas_prediction, r2_score_stas  # Return predictions for plotting
    
    def _run(self):
        
        ticker_stock_data = self.db_instance.fetch_ticker_data(ticker) # Get the stock info in DataFrame format
        feature_matrix, target = self.prepare_data(ticker_stock_data) # Compute y and x from y = ax+b
        coeff_matrix, residuals = self.compute_weights_and_residuals(feature_matrix, target) # Compute a,b and e
        coeff_matrix_weighted = self.compute_weighted_linear_regression(feature_matrix, target, residuals) # Advanced method that takes e in consideration and gives a and b

        simple_prediction = self.compute_prediction(feature_matrix, coeff_matrix)
        weighted_linear_prediction = self.compute_prediction(feature_matrix, coeff_matrix_weighted)
        stas_linearRegression_prediction, R2_score_STAS = self.STAS_LR(ticker_stock_data)
        self.plot_prediction(simple_prediction, weighted_linear_prediction, stas_linearRegression_prediction, target, ticker_stock_data)

        SSR = np.sum(residuals)
        SST = np.sum((target - np.mean(target))**2)
        R2 = 1 - (SSR/SST)

        print(f'Error of my custom implementation: {R2}')
        print(f'Error of STAS implementation: {R2_score_STAS}')


if __name__ == "__main__":

    ### FOR DEBUGGING PURPOSES
    ticker = 'AAPL'
    end = datetime.date(2024, 10, 6) # last index
    start = datetime.date(2015, 1, 1) # 01/01/2015
    linreg = LinearRegression('/Users/costinchitic/Documents/Github/ML101/database_injection/long_stock_symbol_list.txt', start, end)
    linreg._run()