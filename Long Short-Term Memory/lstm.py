import matplotlib.pyplot as plt
import datetime
import numpy as np
import sys
import os
# Add the parent directory (ML101) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from regression.linear_regression import LinearRegression

"""
Implementation of Long Short-Term Memory from scratch.
Source: https://costinchitic.co/notes/LSTM
"""
class LSTM:

    def __init__(self, list_of_tickers, ticker_code, start, end, hidden_dim=64, output_dim=1, learning_rate=0.01,  xavier_init='normal'):
        self.ticker = ticker
        self.ticker_code = ticker_code
        self.start_date = start
        self.end_date = end
        self.eta = learning_rate
        self.linear_regression_instance = LinearRegression(list_of_tickers, ticker_code, start, end)
        self.ticker_stock_data = self.linear_regression_instance.db_instance.fetch_ticker_data(self.ticker_code)

        # Xavier (Glorot) Initialization for weights. Link of publication: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        self.X, self.y = self.prepare_data(self.ticker_stock_data)
        input_dim = self.X.shape[1] # Number of features - date, open_price, volume
        normal_xavier_init = np.sqrt(2 / (input_dim + hidden_dim))
        uniform_xavier_init = np.sqrt(6 / (input_dim + hidden_dim))

        # Initialize weights
        if xavier_init == 'normal':
            # Forget Gate Weights
            self.Wf = np.random.normal(0, normal_xavier_init, (hidden_dim, input_dim))
            self.Uf = np.random.normal(0, normal_xavier_init, (hidden_dim, hidden_dim))
            self.bf = np.ones((1, hidden_dim)) # Initialize the forget gate bias to 1 to improve learning stability in the beginning.
            
            # Input Gate Weights
            self.Wi = np.random.normal(0, normal_xavier_init, (hidden_dim, input_dim))
            self.Ui = np.random.normal(0, normal_xavier_init, (hidden_dim, hidden_dim))
            self.bi = np.zeros((1, hidden_dim))
            
            # Output Gate Weights
            self.Wo = np.random.normal(0, normal_xavier_init, (hidden_dim, input_dim))
            self.Uo = np.random.normal(0, normal_xavier_init, (hidden_dim, hidden_dim))
            self.bo = np.zeros((1, hidden_dim))
            
            # Cell State Weights
            self.Wc = np.random.normal(0, normal_xavier_init, (hidden_dim, input_dim))
            self.Uc = np.random.normal(0, normal_xavier_init, (hidden_dim, hidden_dim))
            self.bc = np.zeros((1, hidden_dim))
            
            # Final Gate Weights
            self.Wv = np.random.normal(0, normal_xavier_init, (output_dim, hidden_dim))
            self.bv = np.zeros((1, output_dim))
        elif xavier_init == 'uniform':
            self.Wf = np.random.uniform(-uniform_xavier_init, uniform_xavier_init, (hidden_dim, input_dim))
            self.Uf = np.random.uniform(-uniform_xavier_init, uniform_xavier_init, (hidden_dim, hidden_dim))
            self.bf = np.ones((1, hidden_dim))
            
            self.Wi = np.random.uniform(-uniform_xavier_init, uniform_xavier_init, (hidden_dim, input_dim))
            self.Ui = np.random.uniform(-uniform_xavier_init, uniform_xavier_init, (hidden_dim, hidden_dim))
            self.bi = np.zeros((1, hidden_dim))
            
            self.Wo = np.random.uniform(-uniform_xavier_init, uniform_xavier_init, (hidden_dim, input_dim))
            self.Uo = np.random.uniform(-uniform_xavier_init, uniform_xavier_init, (hidden_dim, hidden_dim))
            self.bo = np.zeros((1, hidden_dim))
            
            self.Wc = np.random.uniform(-uniform_xavier_init, uniform_xavier_init, (hidden_dim, input_dim))
            self.Uc = np.random.uniform(-uniform_xavier_init, uniform_xavier_init, (hidden_dim, hidden_dim))
            self.bc = np.zeros((1, hidden_dim))
            
            self.Wv = np.random.uniform(-uniform_xavier_init, uniform_xavier_init, (output_dim, hidden_dim))
            self.bv = np.zeros((1, output_dim))

        # Diffs (derivative of loss function w.r.t all parameters)
        self.dWf = np.zeros(self.Wf.shape)
        self.dUf = np.zeros(self.Uf.shape)
        self.dbf = np.zeros(self.bf.shape)
        self.dWi = np.zeros(self.Wi.shape)
        self.dUi = np.zeros(self.Ui.shape)
        self.dbi = np.zeros(self.bi.shape)
        self.dWo = np.zeros(self.Wo.shape)
        self.dUo = np.zeros(self.Uo.shape)
        self.dbo = np.zeros(self.bo.shape)
        self.dWc = np.zeros(self.Wc.shape)
        self.dUc = np.zeros(self.Uc.shape)
        self.dbc = np.zeros(self.bc.shape)

    def prepare_data(self, df):
        df['date_ordinal'] = df['date'].map(datetime.date.toordinal) # convert date to numerical input
        df['date_ordinal'] -= df['date_ordinal'].min() # Normalization which improves numerical stability
        # Feature matrix (X) will now include multiple features
        X = df[['date_ordinal', 'open_price', 'volume']].values
        y = df['close_price'].values # target variable (closing price). Need to add more.

        return X, y
    
    def prepare_sets(self, X, y, split=0.75):
        split_index = int(split * len(X))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        return X_train, X_test, y_train, y_test

    @staticmethod
    def sigmoid(x, derivative=False):
        if derivative:
            return x * (1 - x) # When the derivative functions are called later, they will be called on variables that have already had sigmoid applied to them.
        
        return 1 / (1 + np.exp(-x)) # Maybe look into ReLU

    @staticmethod
    def tanh(x, derivative=False):
        if derivative:
            return 1 - x**2 # When the derivative functions are called later, they will be called on variables that have already had tanh applied to them.
        
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)) # Maybe look into ReLU
    
    @staticmethod
    def softsign(x):
        return x / (1 + np.abs(x))
    
    
    """
    Forget gate: how much of the previous cell state to keep. 1=keep, 0=forget
    Try an use this: https://cnvrg.io/pytorch-lstm/
    ht-1 = prior hidden state
    Xt = Current input
    Wf = Forget gate weights for the prior hidden state
    bf = Forget gate bias
    Uf = Forget gate weights for the current input
    """
    def forget_gate(self, x, h, bf):
        # Use the link above.
        sigmoid_argument = bf + self.Wf @ x + self.Uf @ h
        return self.sigmoid(sigmoid_argument)
    
    def input_gate(self, x, h, bi):
        sigmoid_argument = bi + self.Wi @ x + self.Ui @ h
        return self.sigmoid(sigmoid_argument)
    
    def forward_propagation(self):
        # https://medium.com/@CallMeTwitch/building-a-neural-network-zoo-from-scratch-the-long-short-term-memory-network-1cec5cf31b7
        # Just follow this implementation because it actually makes sense. At least I have a better understanding of what's going on.
        pass
    
    

if __name__ == "__main__":

### FOR DEBUGGING PURPOSES
    ticker = 'AAPL'
    end = datetime.date.today() # last index
    start = datetime.date(2020, 1, 1) # 01/01/2015
    lstm = LSTM('/Users/costinchitic/Documents/Github/ML101/database_injection/long_stock_symbol_list.txt', 'AAPL', start, end)
