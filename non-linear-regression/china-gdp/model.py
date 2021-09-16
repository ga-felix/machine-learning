"""
    This file aims to load a dataset, split the dataset into training
    and testing, create a precise non linear regression model.
    Process overview:
    Load dataset -> summarize it -> create train and test datasets ->
    create model -> verify model's accuracy.
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn.metrics import r2_score

# Loads the dataset into RAM
df = pd.read_csv('china_gdp.csv')

# Describes
df.describe()

# Figure
plt.scatter(df['Year'], df['Value'])
plt.xlabel('Year')
plt.ylabel('GDP')
plt.show()
