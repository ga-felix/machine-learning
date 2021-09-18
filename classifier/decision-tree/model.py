"""
    This files creates a classification model based on decision
    tree that aims to pick the best drug to a pacient.
"""

import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

""" Reading the dataset """
df = pd.read_csv('drug200.csv')
print(df.describe())
print(df.shape)

""" Splitting dataset """
x = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = df[['Drug']].values

""" Sklearn does not handle categorical variables. Then,
these variables must be mapped to numerical values. """
la_sex = preprocessing.LabelEncoder()
la_sex.fit(['F', 'M'])
x[:,1] = la_sex.transform(x[:,1])

la_bp = preprocessing.LabelEncoder()
la_bp.fit(['LOW', 'NORMAL', 'HIGH'])
x[:,2] = la_bp.transform(x[:,2])

la_cholesterol = preprocessing.LabelEncoder()
la_cholesterol.fit(['NORMAL', 'HIGH'])
x[:,3] = la_cholesterol.transform(x[:,3])

""" Splitting the dataset into training and testing sets """
x_trainset, x_testset, y_trainset, y_testset = train_test_split(x, y, test_size=0.3, random_state=3)

""" Training """
drugTree = DecisionTreeClassifier(criterion='entropy', max_depth = 4)
drugTree.fit(x_trainset,y_trainset)

""" Prediction """
predTree = drugTree.predict(x_testset)

""" Evaluation """
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))