# load libraries
import numpy as np
import pandas as pd

# load dataset
data = pd.read_csv("G:/rauf/STEPBYSTEP/Tutorial/Repeat Knowledge/ML/Logistic Regression/logistic_regression_data.csv")
print(data)

# sigmoid fucntion 
def sigmoid(z):
    return 1/(1+np.exp(-z))

