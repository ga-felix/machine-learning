"""
    This file aims to load a dataset, split the dataset into training
    and testing, create a precise polynomial model.
    Process overview:
    Load dataset -> summarize it -> create train and test datasets ->
    create model -> verify model's accuracy.
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Loads the dataset into RAM
df = pd.read_csv("FuelConsumption.csv")
cdf = df[['ENGINESIZE', 'CO2EMISSIONS']]

# Displays dataset information
print(cdf.head())

# Summarizes the dataset
print(cdf.describe())

# Data visualization
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS)
plt.xlabel('Engine size')
plt.ylabel('CO2 emissions')
plt.show()

# Creating mask
mask = np.random.rand(len(cdf)) < 0.8
train = cdf[mask]
test = cdf[~mask]

# Training a polynomial model of quadratic degree
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)

# Coefficients
print('Coefficient:', clf.coef_, 'Intercept:', clf.intercept_)

# Curve plot
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS)
adjustment = np.arange(0, 10, 0.1)
curve_plot = clf.intercept_ + clf.coef_[0][1] * adjustment + clf.coef_[0][2] * np.power(adjustment, 2)
plt.plot(adjustment, curve_plot, '-r')
plt.xlabel('Engine size')
plt.ylabel('CO2 emissions')
plt.show()

# Evaluation
test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)
print('Residual sum of squares (MSE): %.2f' % np.mean((test_y_ - test_y) ** 2))
print('R2-score: %.2f' % r2_score(test_y,test_y_))