import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
import sys
import os
# Add the parent directory (ML101) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database_injection.Database_Injection import Database_Injection
from regression.linear_regression import LinearRegression
import helper.helper as helper

"""
Implementation of Long Short-Term Memory from scratch.
Source: https://costinchitic.co/notes/LSTM
"""
class LSTM:

    def __init__(self, ticker, ticker_code, start, end):
        self.ticker = ticker
        self.ticker_code = ticker_code
        self.start_date = start
        self.end_date = end
        self.db_instance = Database_Injection(ticker, start, end)
        self.linear_regression_instance = LinearRegression(ticker, ticker_code, start, end)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x)) # Maybe look into ReLU

    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)) # Maybe look into ReLU
    
    """
    Forget gate: how much of the previous cell state to keep. 1=keep, 0=forget
    Try an use this: https://cnvrg.io/pytorch-lstm/
    """
    def forget_gate(self, x, h, bf):
        # Use the link above.