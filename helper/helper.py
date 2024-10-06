"""
helper functions like mean computing, standard deviation, etc.
"""

import numpy as np
import datetime

"""
args: 
df - DataFrame that comes from the database,
columns - columns we want to normalize.

Goal:
This scales the values to a fixed range, usually [0,1].
"""
def min_max_normalize(df, columns):
    normalized_df = df.copy()
    for column in columns:
        col_data = df[column].values
        col_min = np.min(col_data)
        col_max = np.max(col_data)
        normalized_df[column] = (col_data - col_min) / (col_max - col_min)
    
    return normalized_df

"""
args:
df - DataFrame that comes from the database,
columns - columns we want to normalize.

Goal:
To compute Z = (X - mean)/(standard deviation). It brings all values to mean 0 and standard deviation of 1.
"""
def z_score_normalization(df, columns):
    normalized_df = df.copy()
    for column in columns:
        col_data = df[column].values
        col_mean = np.mean(col_data)
        col_std = np.std(col_data)
        normalized_df[column] = (col_data - col_mean) / col_std

    return normalized_df

"""
args:
df - DataFrame that comes from the database,
date_column - column of dates.

Goal:
Normalizing the dates since they are in a large numerical range (e.g. 2015-2024).
"""
def dates_normalization(df, date_column):
    df[date_column] = df[date_column].map(np.datetime64).map(datetime.toordinal)
    return min_max_normalize(df, [date_column])

