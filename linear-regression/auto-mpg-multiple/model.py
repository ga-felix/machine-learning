"""
    Linear regression model that tries to predict miles per galon
    based on car's traits.
    Dataset source: https://archive.ics.uci.edu/ml/datasets/Auto+MPG
"""

import pandas as pd
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

""" Read dataframe from csv 'auto_mpg' """
df = pd.read_csv('auto-mpg.csv', delimiter=';', sep='.')
df = df[df['horsepower'] != '?']
print(df.describe())

""" Split dataset in train and test sets """
cdf = df[['cylinders', 'weight', 'horsepower', 'displacement', 'mpg']]
mask = np.random.rand(len(cdf)) < 0.8 
train = cdf[mask]
test = cdf[~mask]

""" Model creation """
train_x = np.asanyarray(train[['cylinders', 'weight', 'horsepower', 'displacement']])
train_y = np.asanyarray(train[['mpg']])
model = linear_model.LinearRegression()
model.fit(train_x, train_y)

""" Model validation """
test_x = np.asanyarray(test[['cylinders', 'weight', 'horsepower', 'displacement']])
test_y = np.asanyarray(test[['mpg']])
predict = model.predict(test_x)
print('Variance score: %.2f' % model.score(test_x, test_y))
print('Residual sum of squares: %.2f' % np.mean((predict - test_y) ** 2))

""" Conclusion: this model has 73 % out of sample accuracy in predicting
miles per galon considering horsepower, weight, displacement and cylinders """