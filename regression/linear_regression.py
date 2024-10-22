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

    def __init__(self, ticker, ticker_code, start, end):
        self.ticker = ticker
        self.ticker_code = ticker_code
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

        # Feature matrix (X) will now include multiple features
        X = np.vstack([df['date_ordinal'].values, np.ones(len(df))]).T  # feature matrix
        # X = df[['date_ordinal', 'open_price', 'volume']].values
        y = df['close_price'].values  # target variable (closing price). Need to add more.
        return X, y
    
    def plot_prediction(self, simple_prediction, stas_prediction, target, df, confidence=0.95, plot_bounds = False):
        df['date'] = pd.to_datetime(df['date'])  # Ensure the 'date' column is in datetime format
        residuals = target - simple_prediction
        z = 1.96  # For 95% confidence level

        # Adjust the prediction level to match the actual closing prices
        actual_mean = df['close_price'].mean()
        predicted_mean = simple_prediction.mean()
        adjustment = actual_mean - predicted_mean
        simple_prediction += adjustment  # Shift the predicted data

        upper = simple_prediction + z * np.std(residuals)
        lower = simple_prediction - z * np.std(residuals)

        plt.figure(figsize=(10, 6))
        plt.plot(df['date'], df['close_price'], label=f'{self.ticker_code} Closing Price')  # Use self.ticker
        plt.plot(df['date'], simple_prediction, label="Predicted Prices", linestyle='--')

        # Only plot STAS prediction if it is not None
        if stas_prediction is not None:
            plt.plot(df['date'], stas_prediction, label="STAS Predicted Prices", linestyle='--')

        # Plot bounds:
        if plot_bounds:
            plt.plot(df['date'], lower, label="LOWER BOUND", linestyle='--')
            plt.plot(df['date'], upper, label="UPPER BOUND", linestyle='--')

            plt.fill_between(df['date'], lower, upper, color='orange', alpha=0.3, label='95% Confidence Interval')

        plt.title(f'{self.ticker_code} Closing prices from {self.start_date} to {self.end_date}')  # Use self.start_date, self.end_date
        plt.xlabel('Date')
        plt.ylabel('Closing Price ($USD)')
        plt.grid(True)
        plt.legend()

        plt.show()

    def plot_quarterly_analysis(self, quarter_predictions, full_data):
        plt.figure(figsize=(10,6))
        # Plot the actual stock prices for the last year
        for keys in full_data.keys():
            TimeSeries = full_data[keys]
            plt.plot(TimeSeries['date'], TimeSeries['close_price'], color='blue')

        # Plot the linear regression for each quarter
        for i, (quarter, prediction, year_quarter) in enumerate(quarter_predictions):
            plt.plot(quarter['date'], prediction, label=year_quarter, linestyle='--')
            plt.fill_between(quarter['date'], 0, TimeSeries['close_price'].max()+50, prediction, alpha=0.3)

        # Add vertical lines to separate quarters
        for i in range(1, len(quarter_predictions)):
            quarter_start = quarter_predictions[i][0]['date'].min()  # First date of the next quarter
            plt.axvline(x=quarter_start, color='black', linestyle=':')

        plt.title(f'{ticker} - Quarterly Linear Regression for Last Year')
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
    
    def _run_full_data(self):
        # Run algorithm for all data.
        ticker_stock_data = self.db_instance.fetch_ticker_data(ticker) # Get the stock info in DataFrame format
        feature_matrix, target = self.prepare_data(ticker_stock_data) # Compute y and x from y = ax+b
        coeff_matrix, residuals = self.compute_weights_and_residuals(feature_matrix, target) # Compute a,b and e

        simple_prediction = self.compute_prediction(feature_matrix, coeff_matrix)
        stas_linearRegression_prediction, R2_score_STAS = self.STAS_LR(ticker_stock_data)
        self.plot_prediction(simple_prediction, stas_linearRegression_prediction, target, ticker_stock_data)

        SSR = np.sum(residuals)
        SST = np.sum((target - np.mean(target))**2)
        R2 = 1 - (SSR/SST)

        print(f'Error of my custom implementation: {R2}')
        print(f'Error of STAS implementation: {R2_score_STAS}')

    def filter_last_year_quarterly(self, df):
        most_recent_date = df['date'].max() # most recent date
        last_year = most_recent_date - pd.DateOffset(years=1) # substract 1 year
        df_last_year = df[df['date'] >= last_year] # only data from last year.

        # Add a 'quarter' column and a 'year' column
        quarters = df_last_year['date'].dt.quarter
        years = df_last_year['date'].dt.year

        df_last_year['year_quarter'] = years.astype(str) + ' Q' + quarters.astype(str) # E.g. 2023 Q4, ..., 2024 Q4. I saw this bug and fixed it like this.

        #Prepare data for further processing.
        quarters_data = {}
        for year_quarter in df_last_year['year_quarter']: # 2023 Q4 -> data for that quarter.
            quarters_data[year_quarter] = df_last_year[df_last_year['year_quarter'] == year_quarter]

        return quarters_data

    def _run_quarterly_analysis_for_last_year(self):
        # Run algorithm for last year and present on quarters (Q1, Q2, Q3, Q4)
        ticker_stock_data = self.db_instance.fetch_ticker_data(ticker) # Get the stock info in DataFrame format
        quarters = self.filter_last_year_quarterly(ticker_stock_data)

        quarter_predictions = [] 
        for year_quarter, quarter_data in quarters.items(): #Return the dictionary's key-value pairs in tuples (a,b)
            if len(quarter_data) > 0:
                # Prepare data
                X, y = self.prepare_data(quarter_data)
                coeff_matrix, _ = self.compute_weights_and_residuals(X, y)
                predictions = self.compute_prediction(X, coeff_matrix)
                quarter_predictions.append((quarter_data, predictions, year_quarter))

        self.plot_quarterly_analysis(quarter_predictions, quarters)


    def _run_projected_data_from_PCA(self, projected_data, y, ticker_stock_data):
        coeff_matrix, residuals = self.compute_weights_and_residuals(projected_data, y) # Compute a,b and e
        simple_prediction = self.compute_prediction(projected_data, coeff_matrix)
        self.plot_prediction(simple_prediction, None, y, ticker_stock_data)

        SSR = np.sum((y - simple_prediction) ** 2)
        SST = np.sum((y - np.mean(y)) ** 2)
        R2 = 1 - (SSR / SST)
        print(f'R-squared: {R2}')

    
    def _run(self):
        
        self._run_full_data()
        self._run_quarterly_analysis_for_last_year()


if __name__ == "__main__":

    ### FOR DEBUGGING PURPOSES
    ticker = 'NVDA'
    end = datetime.date.today() # last index
    start = datetime.date(2015, 1, 1) # 01/01/2015
    linreg = LinearRegression('/Users/costinchitic/Documents/Github/ML101/database_injection/long_stock_symbol_list.txt', ticker, start, end)
    linreg._run()