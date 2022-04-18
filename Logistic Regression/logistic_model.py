# load libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# load data
data = pd.read_csv("G:/rauf/STEPBYSTEP/Tutorial/Repeat Knowledge/ML/Logistic Regression/logistic_regression_data.csv")
print(data)
x = data.iloc[:,:-1].values
y = data.iloc[:, -1].values
print(x, y)
# create model
model = LogisticRegression()

#model training
model.fit(x, y)

# model predicting
result_posetive = model.predict([[85]])
print(result_posetive) # model correctly prediced because i entered age 85 which likely buy insurance 

result_negative = model.predict([[18]]) # model has to predict likely negative because our client is young
print(result_negative)
