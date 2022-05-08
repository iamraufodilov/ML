# load required libraries 
import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt


# load the dataset
data = pd.read_csv("G:/rauf/STEPBYSTEP/Tutorial/Repeat Knowledge/ML/PCA/bodyPerformance.csv")
print(data.head(5)) # as you can see our data has 12 features or in other word it has 12D dimension

# we do not need trget collumn and categorical collumns
data.drop(columns = ['gender', 'class'], inplace = True)
print(data.head(5))


# next our task is to challenge data
def data_normalization(data):
    for col in data.columns:
        data[col] = (data[col]-data[col].mean())/data[col].std()

    return data

data_normalization(data)
print(data.head(5))


# we have to calculate covrieance
def caculate_covariance(data):
    for col in data.columns:
        data[col] = data[col] - data[col].mean()

    return np.dot(data.T, data)/(len(data)-1)

cov_calculated = caculate_covariance(data)
print('>>>>>>>>>>>>>>>>>>>>',cov_calculated)


# calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = eig(cov_calculated)

# her is brief NOTE:
# higher eigenvalue means that corresponding eignenvector is more important
# lets visualize eigenvalues to know which eigenvectros are important
plt.bar(["e"+str(i+1) for i in eigenvalues], eigenvalues)
plt.xlabel("Eigenvalues")
plt.show() # from the plot it can be seen that our first 5 eigenvalues are huge and corresponding first 5 eigenvectors 
# are holds important value to the data it means we can throw away last five data as it is not much importsnt

# now we can clearly see our 10 dimension data can be shrinked to 5 dimension
# because some other features does not hold huge impact to the model performance 

useful_PCA = eigenvectors[:, :5]
principle_data = np.dot(data.values, useful_PCA)
print("Here our 5 dimension dataset: \n",principle_data[:5,:])
print("Shape of the data", np.shape(principle_data))

# nice we manage to change the data dimension through PCA method
# thank you
# rauf odilov