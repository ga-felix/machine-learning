"""
    This files creates a classification model based on K-nearest
    neighbors.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

""" Reading the dataset """
df = pd.read_csv('teleCust1000t.csv')
print(df.describe())
print(df['custcat'].value_counts())

""" In order to use Scikit-learn library, a numpy array is needed """
x = df [['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values
y = df['custcat'].values

""" Data standardization """
x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))

""" Splitting training and testing sets """
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

""" Training """
k = 4
neigh = KNeighborsClassifier(n_neighbors = k).fit(x_train, y_train)

""" Predicting """
yhat = neigh.predict(x_test)

""" Evaluation """
print("Train set accuracy: ", metrics.accuracy_score(y_train, neigh.predict(x_train)))
print("Out of sample accuracy: ", metrics.accuracy_score(y_test, yhat))