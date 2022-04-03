# load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# load data
data = pd.read_csv("G:/rauf/STEPBYSTEP/Tutorial/Repeat Knowledge/ML/Linear Regression/house data.csv")
x = data.iloc[:, :-1].values
y = data.iloc[:,-1].values
print(x, y)
# Load model
model = LinearRegression()
model.fit(x, y)
result = model.predict([[3600]])
print(result)