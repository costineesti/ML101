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

    def __init__(self, list_of_tickers, ticker_code, start, end, hidden_dim=64, output_dim=1, learning_rate=0.002,  xavier_init='normal'):
        self.ticker = ticker
        self.ticker_code = ticker_code
        self.start_date = start
        self.end_date = end
        self.eta = learning_rate
        self.hidden_dim = hidden_dim
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
            self.Wf = np.random.normal(0, normal_xavier_init, (hidden_dim, hidden_dim + input_dim))
            self.bf = np.zeros((hidden_dim, 1))
            
            # Input Gate Weights
            self.Wi = np.random.normal(0, normal_xavier_init, (hidden_dim, hidden_dim + input_dim))
            self.bi = np.zeros((hidden_dim, 1))
            
            # Output Gate Weights
            self.Wo = np.random.normal(0, normal_xavier_init, (hidden_dim, hidden_dim + input_dim))
            self.bo = np.zeros((hidden_dim, 1))
            
            # Cell State Weights
            self.Wc = np.random.normal(0, normal_xavier_init, (hidden_dim, hidden_dim + input_dim))
            self.bc = np.zeros((hidden_dim, 1))
            
            # Final Gate Weights
            self.Wv = np.random.normal(0, normal_xavier_init, (output_dim, hidden_dim))
            self.bv = np.zeros((output_dim, 1))
        elif xavier_init == 'uniform':
            self.Wf = np.random.uniform(-uniform_xavier_init, uniform_xavier_init, (hidden_dim, hidden_dim + input_dim))
            self.bf = np.zeros((hidden_dim, 1))
            
            self.Wi = np.random.uniform(-uniform_xavier_init, uniform_xavier_init, (hidden_dim, hidden_dim + input_dim))
            self.bi = np.zeros((hidden_dim, 1))
            
            self.Wo = np.random.uniform(-uniform_xavier_init, uniform_xavier_init, (hidden_dim, hidden_dim + input_dim))
            self.bo = np.zeros((hidden_dim, 1))
            
            self.Wc = np.random.uniform(-uniform_xavier_init, uniform_xavier_init, (hidden_dim, hidden_dim + input_dim))
            self.bc = np.zeros((hidden_dim, 1))
            
            self.Wv = np.random.uniform(-uniform_xavier_init, uniform_xavier_init, (output_dim, hidden_dim))
            self.bv = np.zeros((output_dim, 1))

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

        self.outputs = []
        self.gate_dictionary = {
            'forget_gate' : {},
            'input_gate' : {},
            'output_gate' : {},
            'candidate_gate' : {},

            'cell_state' : {-1:np.zeros((hidden_dim, 1))},
            'hidden_state' : {-1:np.zeros((hidden_dim, 1))}
        }

    def prepare_data(self, df):
        df['date_ordinal'] = df['date'].map(datetime.date.toordinal) # convert date to numerical input
        # Normalize the data
        df['date_ordinal'] = (df['date_ordinal'] - df['date_ordinal'].min()) / (df['date_ordinal'].max() - df['date_ordinal'].min())
        df['open_price'] = (df['open_price'] - df['open_price'].min()) / (df['open_price'].max() - df['open_price'].min())
        df['volume'] = (df['volume'] - df['volume'].min()) / (df['volume'].max() - df['volume'].min())
        df['close_price'] = (df['close_price'] - df['close_price'].min()) / (df['close_price'].max() - df['close_price'].min())
        # Feature matrix (X) will now include multiple features
        X = df[['date_ordinal', 'open_price', 'volume']].values
        y = df['close_price'].values

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
    
    def reset_gradients(self):
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

    def reset_forwardprop(self, reset=False):
        self.outputs = []
        if not reset:
            self.gate_dictionary = {
                'forget_gate' : {},
                'input_gate' : {},
                'output_gate' : {},
                'candidate_gate' : {},

                'cell_state' : {-1:np.zeros((self.hidden_dim, 1))},
                'hidden_state' : {-1:np.zeros((self.hidden_dim, 1))}
            }
        else: # Fix for the first time step of the test data.
            self.gate_dictionary = {
                'forget_gate' : {},
                'input_gate' : {},
                'output_gate' : {},
                'candidate_gate' : {},

                'cell_state' : {-1:self.gate_dictionary['cell_state'][-1]},
                'hidden_state' : {-1:self.gate_dictionary['hidden_state'][-1]}
            }


    """
    Forward Propagation: Go through the input data and calculate the output for each time step.
    """
    
    def forward_propagation(self, X, reset=False):
        # https://medium.com/@CallMeTwitch/building-a-neural-network-zoo-from-scratch-the-long-short-term-memory-network-1cec5cf31b7
        # Just follow this implementation because it actually makes sense. At least I have a better understanding of what's going on.
        self.reset_forwardprop(reset)

        for t in range(len(X)):
            concatenated_input = np.concatenate((self.gate_dictionary['hidden_state'][t-1],
                                                  X[t].reshape(-1,1)),
                                                  axis=0)
            var = X[t].reshape(-1,1)
            self.gate_dictionary['forget_gate'][t] = self.sigmoid(np.dot(self.Wf, concatenated_input) + self.bf)
            self.gate_dictionary['input_gate'][t] = self.sigmoid(np.dot(self.Wi, concatenated_input) + self.bi)
            self.gate_dictionary['output_gate'][t] = self.sigmoid(np.dot(self.Wo, concatenated_input) + self.bo)
            self.gate_dictionary['candidate_gate'][t] = self.tanh(np.dot(self.Wc, concatenated_input) + self.bc)

            self.gate_dictionary['cell_state'][t] = self.gate_dictionary['forget_gate'][t] * self.gate_dictionary['cell_state'][t-1] + self.gate_dictionary['input_gate'][t] * self.gate_dictionary['candidate_gate'][t]
            self.gate_dictionary['hidden_state'][t] = self.gate_dictionary['output_gate'][t] * self.tanh(self.gate_dictionary['cell_state'][t])

            output = np.dot(self.Wv, self.gate_dictionary['hidden_state'][t]) + self.bv
            self.outputs.append(output)
            # print(f"Output: {output}")

        return self.outputs
    
    
    """
    Backward Propagation: Calculate the gradients of the loss function w.r.t all parameters. Update weights and biases.
    """
    def back_propagation(self, y):
        # reset gradients
        self.reset_gradients()

        d_h_top = np.zeros((self.Wf.shape[0], 1))
        d_C_top = np.zeros((self.Wc.shape[0], 1))
        loss = 0
        # Iterate backwards through times steps to calculate gradients
        for t in reversed(range(len(self.outputs))):
            y_hat = self.outputs[t]
            y_true = y[t]
            loss += 0.5 * np.sum((y_hat - y_true)**2)

            # derivative of the loss function w.r.t the output
            d_y = y_hat - y_true # dL/dz

            # Gradients for output weights and bias
            self.dWv += np.dot(d_y, self.gate_dictionary['hidden_state'][t].T)
            self.dbv += d_y

            # Gradient for the hidden state
            d_h = np.dot(self.Wv.T, d_y) + d_h_top

            # Gradient for the output gate
            d_o = d_h * self.tanh(self.gate_dictionary['cell_state'][t])
            d_o = self.sigmoid(self.gate_dictionary['output_gate'][t], derivative=True) * d_o
            self.dWo += np.dot(d_o,
                               np.concatenate((self.X[t].reshape(-1,1), # 3x1
                                              self.gate_dictionary['hidden_state'][t]), # 64x1
                                              axis=0).T)
            self.dbo += d_o

            # Gradient for the cell state
            d_C = d_h * self.gate_dictionary['output_gate'][t] * self.tanh(self.gate_dictionary['cell_state'][t], derivative=True) + d_C_top

            # Gradient for the forget gate
            d_f = d_C * self.gate_dictionary['cell_state'][t-1]
            d_f = self.sigmoid(self.gate_dictionary['forget_gate'][t], derivative=True) * d_f
            self.dWf += np.dot(d_f,
                               np.concatenate((self.X[t].reshape(-1,1),
                                              self.gate_dictionary['hidden_state'][t]),
                                              axis=0).T)
            self.dbf += d_f

            # Gradient for the input gate
            d_i = d_C * self.gate_dictionary['candidate_gate'][t]
            d_i = self.sigmoid(self.gate_dictionary['input_gate'][t], derivative=True) * d_i
            self.dWi += np.dot(d_i,
                               np.concatenate((self.X[t].reshape(-1,1),
                                              self.gate_dictionary['hidden_state'][t]),
                                              axis=0).T)
            self.dbi += d_i

            # Gradient for the candidate gate
            d_c = d_C * self.gate_dictionary['input_gate'][t]
            d_c = self.tanh(self.gate_dictionary['candidate_gate'][t], derivative=True) * d_c
            self.dWc += np.dot(d_c,
                               np.concatenate((self.X[t].reshape(-1,1),
                                              self.gate_dictionary['hidden_state'][t]),
                                              axis=0).T)
            self.dbc += d_c

            # Propagate the gradients back in time
            d_h_top = d_h * self.gate_dictionary['forget_gate'][t]
            d_C_top = d_C * self.gate_dictionary['input_gate'][t]

        # Update weights and biases
        self.Wf -= self.eta * self.dWf
        self.bf -= self.eta * self.dbf
        self.Wi -= self.eta * self.dWi
        self.bi -= self.eta * self.dbi
        self.Wo -= self.eta * self.dWo
        self.bo -= self.eta * self.dbo
        self.Wc -= self.eta * self.dWc
        self.bc -= self.eta * self.dbc
        self.Wv -= self.eta * self.dWv
        self.bv -= self.eta * self.dbv

        return loss
    

    def plot_results(self, y_train, y_test, predictions):
        # Combine the training and test data so we can plot them together
        actual = np.concatenate((y_train, y_test))

        # Prepare the time axis
        time_steps = np.arange(len(actual) + len(predictions))
        
        # Plot the entire dataset: training and test
        plt.figure(figsize=(12, 6))
        plt.plot(time_steps[:len(y_train)], y_train, color='blue', label='Historical Training Data')
        plt.plot(time_steps[len(y_train):len(y_train) + len(y_test)], y_test, color='green', label='Historical Test Data')
        
        # Overlay the predictions
        # Note: Adjust the starting index according to how your predictions align with the test data
        prediction_steps = time_steps[len(y_train):len(y_train) + len(predictions)]
        plt.plot(prediction_steps, predictions, color='red', label='Predicted Prices', linestyle='--')
        
        plt.title(f'Stock Price Prediction - LSTM Model for {self.ticker_code}')
        plt.xlabel(f'{self.ticker_code} FROM {self.start_date} TO {self.end_date}')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.grid(True)
        plt.show()


    def _runLSTM(self, epochs=150):
        losses = []
        for epoch in range(epochs):
            self.forward_propagation()
            loss = self.back_propagation()
            losses.append(loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch} - Loss: {loss}")

        # Plot the loss function
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Training Loss')
        plt.title('LSTM Training Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()


    def train_and_test_LSTM(self, epochs = 100):
        losses = []
        X_train, X_test, y_train, y_test = self.prepare_sets(self.X, self.y)
        for epoch in range(epochs):
            training_set = self.forward_propagation(X_train)
            loss = self.back_propagation(y_train)
            losses.append(loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch} - Loss: {loss}")

        # Plot the loss function
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Training Loss')
        plt.title('LSTM Training Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Predict on the test data
        predictions = self.forward_propagation(X_test, reset=True)

        # Since predictions might be wrapped in extra dimensions, squeeze eliminates axes of length 1.
        predictions = np.squeeze(predictions)  # This will change shape from (55, 1, 1) to (55,)
        self.plot_results(y_train, y_test, predictions)

    

if __name__ == "__main__":

### FOR DEBUGGING PURPOSES
    ticker = 'AMZN'
    end = datetime.date.today() # last index
    start = datetime.date(2024, 1, 1) # 01/01/2015
    lstm = LSTM('/Users/costinchitic/Documents/Github/ML101/database_injection/long_stock_symbol_list.txt', 'AAPL', start, end, xavier_init='uniform')
    lstm.train_and_test_LSTM(epochs=100)