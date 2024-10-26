import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import MySQLdb
import pandas as pd
import os

class Database_Injection:
    
    def __init__(self, stocks_file, start_date, end_date):
        self.tickers_list = []
        with open(stocks_file) as tickers:
            for ticker in tickers:
                self.tickers_list.append(ticker.replace('\n',''))
        self.start_date = start_date
        self.end_date = end_date

    def create_connection(self):
        try:
            print("Connecting ...")
            connection = MySQLdb.connect(
                host = "localhost",
                user = "root",
                password = os.getenv("DB_PASSWORD"),
                database = "stocksDB"
            )
            print("Connected to stocksDB!")
            return connection
        except MySQLdb.Error as e:
            print(f"Error occurred while connecting to the DB: '{e}'")
            return None
        
    def get_latest_date_for_ticker(self, connection, ticker):
        cursor = connection.cursor()
        query = """
        SELECT MAX(date) FROM stocks WHERE ticker = %s;
        """
        cursor.execute(query, (ticker,))
        result = cursor.fetchone()
        return result[0]  # This returns the latest date for the ticker or None if no data exists

    def insert_stock_data(self, connection, ticker, stock_data):
        cursor = connection.cursor()
        for row in stock_data.itertuples():
            insert_query = """
            INSERT INTO stocks (ticker, date, open_price, high_price, low_price, close_price, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (ticker, row.Index, row.Open, row.High, row.Low, row.Close, row.Volume))
        connection.commit()

    def update_database(self, delete_existing_data=False):
        connection = self.create_connection()
        if connection:
            for ticker in self.tickers_list:
                print(f"Fetching data for {ticker}")
                
                # Delete existing data for the ticker if needed
                if delete_existing_data:
                    self.delete_database(connection, ticker)

                # Check the latest date for the ticker in the database
                latest_date = self.get_latest_date_for_ticker(connection, ticker)
                
                if latest_date:
                    # Fetch new data starting from the next day after the latest date
                    start_date = (latest_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                    stock_data = yf.download(ticker, start=start_date)
                else:
                    # If no data exists, fetch from a default date (or adjust as needed)
                    stock_data = yf.download(ticker, start=self.start_date)

                if not stock_data.empty:
                    self.insert_stock_data(connection, ticker, stock_data)
                else:
                    print(f"No new data for {ticker}")

            connection.close()
        else:
            print("Failed to connect to the database.")
            
            
    """
    TO DO: Plot any stock from the list by parsing the ticker.
    """
    def plot_data(self, ticker):
        connection = self.create_connection()

        if connection:
            cursor = connection.cursor()
            query = """
            SELECT date, close_price FROM stocks
            WHERE ticker = %s AND date BETWEEN %s AND %s
            ORDER BY date;
            """
            cursor.execute(query, (ticker, self.start_date, self.end_date))
            rows = cursor.fetchall()

            # Check if data was fetched:
            if not rows:
                print(f"No data was found for {ticker}")
                return
            
            # Now I convert th query results into a Pandas DataFrame for easier plotting:
            df = pd.DataFrame(rows, columns=['date', 'close_price'])
            df['date'] = pd.to_datetime(df['date']) # Ensuring the 'date' column is in datetime format

            plt.figure(figsize=(10,6))
            plt.plot(df['date'], df['close_price'], label = f'{ticker} Closing Price')

            plt.title(f'{ticker} Closing prices from {self.start_date} to {self.end_date}')
            plt.xlabel('Date') 
            plt.ylabel('Closing Price ($USD)')
            plt.grid(True)
            plt.legend()

            plt.show()

            connection.close()
        else:
            print(f"Failed to connect to DB in order to plot {ticker}")

    """
    Function to fetch any ticker's data. All columns.
    """
    def fetch_ticker_data(self, ticker):
        connection = self.create_connection()

        if connection:
            cursor = connection.cursor()
            query = """
            SELECT date, open_price, high_price, low_price, close_price, volume FROM stocks
            WHERE ticker = %s
            ORDER BY date;
            """
            cursor.execute(query, (ticker,))
            rows = cursor.fetchall()
            if not rows:
                print(f"No data was found for {ticker}")
                return
            # Now I convert the query results into a Pandas DataFrame for returnal:
            df = pd.DataFrame(rows, columns=['date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume'])
            df['date'] = pd.to_datetime(df['date']) # Ensuring the 'date' column is in datetime format

            connection.close()
            return df
        
    """
    Function to delete the existing database. I had to use it in order to fetch earlier data.
    """
    def delete_database(self, connection, ticker):
        cursor = connection.cursor()
        query = """
        DELETE FROM stocks WHERE ticker = %s;
        """
        cursor.execute(query, (ticker,))
        connection.commit()

if __name__ == "__main__":

    """
    Declare the start and end date and the list of tickers.
    """
    stocks_file = 'long_stock_symbol_list.txt'
    end = datetime.date.today() # today
    start = datetime.date(1990, 1, 1) # 01/01/2015
    db_injector = Database_Injection(stocks_file, start, end)
    db_injector.update_database(delete_existing_data=True)
    db_injector.plot_data('AAPL')