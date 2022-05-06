# load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# create data
X, y = make_classification()
print(X.shape, y.shape)
print("Feature", X[0,:], "Label", y[0])

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# load the model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)

accuracy = accuracy_score(y_predicted, y_test)
cm = confusion_matrix(y_predicted, y_test)
print("Here our model's accuracy: {}%".format(int(accuracy*100)))
print("Here is confusion: ", cm)

# lets do some fun by predicteding random data from dataset
# for that lets chose first data from dataset
random_X, random_y = X[0,:], y[0]
random_X_2D = np.reshape(random_X, (1, 20))
random_predicted = model.predict(random_X_2D)
if random_predicted==random_y:
    print("Wow you trained very strong model")
else:
    print("You bitch, study more")

# ok
# you see I do not have to study more 
# model trained 
# rauf odilov