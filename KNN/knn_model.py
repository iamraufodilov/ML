# load libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
data = pd.read_csv("G:/rauf/STEPBYSTEP/Tutorial/Repeat Knowledge/ML/KNN/iris.csv")
print(data.head(5))

X = data.iloc[:,0:4].values
y = data.iloc[:, -1].values

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# load the model
model_knn = KNeighborsClassifier()
model_knn.fit(X_train,y_train)
y_predicted = model_knn.predict(X_test)
accuracy = accuracy_score(y_predicted, y_test)
print("Here we go there is our accuracy score: {}%".format(int(accuracy*100)))

# Yes we are trained the model with 96% accuracy