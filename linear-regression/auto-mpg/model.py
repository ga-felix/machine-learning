"""
    Linear regression model that tries to predict miles per galon
    based on car's weight.
    Dataset source: https://archive.ics.uci.edu/ml/datasets/Auto+MPG
"""

import pandas as pd
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

""" Loads dataset and filters missing row """

df = pd.read_csv('auto-mpg.csv', delimiter=';', sep='.')
df = df[df['horsepower'] != '?']
cdf = df[['horsepower', 'mpg', 'cylinders', 'weight']]

""" Plots horsepower / mpg histogram """

plt.scatter(cdf['weight'], cdf['mpg'])
plt.xlabel('Weight')
plt.ylabel('Miles per Galon (MPG)')
plt.show()

""" Splits dataset """

mask = np.random.rand(len(cdf)) < 0.8
train = cdf[mask]
test = cdf[~mask]

""" Linear regression model creation """

model = linear_model.LinearRegression()
weight = np.asanyarray(train[['weight']], dtype='float32')
mpg = np.asanyarray(train[['mpg']], dtype='float32')
model.fit(weight, mpg)
r_sqr = model.score(weight, mpg)
print(r_sqr)

""" Plot line """

plt.scatter(train['weight'], train['mpg'])
plt.plot(weight, (weight * model.coef_[0][0]) + model.intercept_[0], '-r', scalex = True, scaley = True)
plt.xlabel('Weight')
plt.ylabel('Miles per Galon (MPG)')
plt.show()

""" 
    Conclusion: the model predicts with 70 % out of sample
    accuracy mpg values based on car's weight
"""