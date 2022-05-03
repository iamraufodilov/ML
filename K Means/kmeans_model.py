# load libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# load dataset
data = pd.read_csv("G:/rauf/STEPBYSTEP/Tutorial/Repeat Knowledge/ML/K Means/country_datta.csv")
print(data)

# prepare datset
X = data.iloc[:, 1:3].values
print(X)

# plot the data to see its data position
def visualize_data(data):
    plt.scatter(data[:,0], data[:,-1])
    plt.show() # if we see the result we can easily see our dataset is three different type of cluster. 

#visualize_data(X)


# now our task is to group them into clusters
# but first we need number of k How to choose K?
# we will use elbov method
def elbow_method(data_point):
    SSE = []
    number_k = list(range(1,7))
    for i in number_k:
        k_model = KMeans(i)
        k_model.fit(data_point)
        error = k_model.inertia_
        SSE.append(error)
    plt.plot(number_k, SSE)
    plt.show()

# elbow_method(X) # as we see from the graph after the k (number of clusters reach to 3) then error wont change SUPER


# so now we know how many number of cluster we need
number_cluster = 3
model = KMeans(n_clusters = number_cluster, random_state=0)
model.fit(X)

predicted_classes = model.fit_predict(X)
print("Here our predicted classes: ", predicted_classes) # from result we can see that we had three group of data

data['cluster'] = predicted_classes
print(data)

# visualize the result
plt.scatter(data['latitude'], data['longitude'], c=data['cluster'])
plt.show() # yeah boy threr result is out with desired looking I mean three type of cluster with different color

# rauf odilov
