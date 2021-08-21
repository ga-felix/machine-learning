"""
    This file aims to load a dataset, split the dataset into training
    and testing, create a precise linear regression model.
    Process overview:
    Load dataset -> summarize it -> create train and test datasets ->
    create model -> verify model's accuracy.
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model

# Loads the dataset into RAM
df = pd.read_csv("FuelConsumption.csv")

# Displays dataset information
print(df.head())

# Summarizes the dataset
print(df.describe())

# Gets just few columns
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head(5))

# Generates histograms of different columns
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

# Analysing fuel consumption linearity
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color="blue")
plt.xlabel("Litres / km")
plt.ylabel("CO2 Emission g / km")
plt.show()

# Analysing engine size linearity
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color="blue")
plt.xlabel("Engine size")
plt.ylabel("CO2 Emission g / km")
plt.show()

# Analysing cylinders number linearity
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color="blue")
plt.xlabel("Cylinder number")
plt.ylabel("CO2 Emission g / km")
plt.show()

# Splitting the dataset into training set and testing set
mask = np.random.rand(len(df)) < 0.8
training_set = cdf[mask]
testing_set = cdf[~mask]

# Train data distribution
plt.scatter(training_set.ENGINESIZE, training_set.CO2EMISSIONS, color="red")
plt.xlabel("Engine size")
plt.ylabel("CO2 Emission g / km")
plt.show()

# Creating the model
regression = linear_model.LinearRegression()
training_set_x = np.asanyarray(training_set[["ENGINESIZE"]])
training_set_y = np.asanyarray(training_set[["CO2EMISSIONS"]])
regression.fit(training_set_x, training_set_y)
print(regression.coef_)
print(regression.intercept_)