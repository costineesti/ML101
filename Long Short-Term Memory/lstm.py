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
            self.bf = np.ones((1, hidden_dim)) # Initialize the forget gate bias to 1 to improve learning stability in the beginning.
            
            # Input Gate Weights
            self.Wi = np.random.normal(0, normal_xavier_init, (hidden_dim, input_dim))
            self.bi = np.zeros((1, hidden_dim))
            
            # Output Gate Weights
            self.Wo = np.random.normal(0, normal_xavier_init, (hidden_dim, input_dim))
            self.bo = np.zeros((1, hidden_dim))
            
            # Cell State Weights
            self.Wc = np.random.normal(0, normal_xavier_init, (hidden_dim, input_dim))
            self.bc = np.zeros((1, hidden_dim))
            
            # Final Gate Weights
            self.Wv = np.random.normal(0, normal_xavier_init, (output_dim, hidden_dim))
            self.bv = np.zeros((1, output_dim))
        elif xavier_init == 'uniform':
            self.Wf = np.random.uniform(-uniform_xavier_init, uniform_xavier_init, (hidden_dim, input_dim))
            self.bf = np.ones((1, hidden_dim))
            
            self.Wi = np.random.uniform(-uniform_xavier_init, uniform_xavier_init, (hidden_dim, input_dim))
            self.bi = np.zeros((1, hidden_dim))
            
            self.Wo = np.random.uniform(-uniform_xavier_init, uniform_xavier_init, (hidden_dim, input_dim))
            self.bo = np.zeros((1, hidden_dim))
            
            self.Wc = np.random.uniform(-uniform_xavier_init, uniform_xavier_init, (hidden_dim, input_dim))
            self.bc = np.zeros((1, hidden_dim))
            
            self.Wv = np.random.uniform(-uniform_xavier_init, uniform_xavier_init, (output_dim, hidden_dim))
            self.bv = np.zeros((1, output_dim))

        # Diffs (derivative of loss function w.r.t all parameters)
        self.dWf = np.zeros(self.Wf.shape)
        self.dbf = np.zeros(self.bf.shape)
        self.dWi = np.zeros(self.Wi.shape)
        self.dbi = np.zeros(self.bi.shape)
        self.dWo = np.zeros(self.Wo.shape)
        self.dbo = np.zeros(self.bo.shape)
        self.dWc = np.zeros(self.Wc.shape)
        self.dbc = np.zeros(self.bc.shape)
        self.dWv = np.zeros(self.Wv.shape)
        self.dbv = np.zeros(self.bv.shape)

        # Hidden state and cell state
        self.h = np.zeros((hidden_dim, 1))
        self.cell_state = np.zeros((hidden_dim, 1))
        # Declare outputs list
        self.outputs = []

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
    Forward Propagation: Go through the input data and calculate the output for each time step.
    """
    
    def forward_propagation(self):
        # https://medium.com/@CallMeTwitch/building-a-neural-network-zoo-from-scratch-the-long-short-term-memory-network-1cec5cf31b7
        # Just follow this implementation because it actually makes sense. At least I have a better understanding of what's going on.
        outputs = []

        for i in range(len(self.X)):
            concatenated_argument = np.concatenate((self.X[i], self.h), axis=1)
            forget_gate = self.sigmoid(self.bf + np.dot(self.Wf, concatenated_argument))
            input_gate = self.sigmoid(self.bi + np.dot(self.Wi, concatenated_argument))
            candidate_gate = self.tanh(self.bc + np.dot(self.Wc, concatenated_argument))
            output_gate = self.sigmoid(self.bo + np.dot(self.Wo, concatenated_argument))
            
            self.cell_state = forget_gate * self.cell_state + input_gate * candidate_gate
            self.h = output_gate * self.tanh(self.cell_state)

            output = self.Wv * self.h + self.bv
            outputs.append(output)

        return self.outputs
    
    """
    Backward Propagation: Calculate the gradients of the loss function w.r.t all parameters. Update weights and biases.
    """
    def back_propagation(self):
        # reset gradients
        self.dWf.fill(0)
        self.dbf.fill(0)
        self.dWi.fill(0)
        self.dbi.fill(0)
        self.dWo.fill(0)
        self.dbo.fill(0)
        self.dWc.fill(0)
        self.dbc.fill(0)
        self.dWv.fill(0)
        self.dbv.fill(0)

        d_h = np.zeros_like(self.h)
        d_C = np.zeros_like(self.cell_state)

        loss = 0

        # Iterate backwards through times steps to calculate gradients
        
    

if __name__ == "__main__":

### FOR DEBUGGING PURPOSES
    ticker = 'AAPL'
    end = datetime.date.today() # last index
    start = datetime.date(2020, 1, 1) # 01/01/2015
    lstm = LSTM('/Users/costinchitic/Documents/Github/ML101/database_injection/long_stock_symbol_list.txt', 'AAPL', start, end)
