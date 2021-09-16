"""
    This files creates a classification model based on K-nearest
    neighbors.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing

""" Reading the dataset """
df = pd.read_csv('teleCust1000t.csv')
print(df.describe())
print(df['custcat'].value_counts())

""" In order to use Scikit-learn library, a numpy array is needed """
x = df [['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values
x[0:5]
y = df['custcat'].values
y[0:5]