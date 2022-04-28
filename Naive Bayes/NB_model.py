# import libraries
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load dataset
data = pd.read_csv("G:/rauf/STEPBYSTEP/Tutorial/Repeat Knowledge/ML/Naive Bayes/iris.csv")
print(data.head(5))

X = data.iloc[:, 0:4].values
y = data.iloc[:, -1].values
print(X[:5], y[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# load model amd train it

model = GaussianNB()
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)

acuracy = accuracy_score(y_predicted, y_test)
print("Here is our models accuracy: {}%".format(int(acuracy*100)))

# random datat identification
random_flower = [[4.6, 3.1, 1.5, 0.2]]
result = model.predict(random_flower)
print(result)