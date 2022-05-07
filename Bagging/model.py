# load the library
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


# create dataset
X, y = make_classification()
print(X.shape, y.shape)

def visualize_data(x, y):
    # lets visualiza the data
    d_list = list(np.mean(i) for i in x)
    plt.scatter(d_list, y)
    plt.xlabel("Mean of 20 fetures")
    plt.ylabel("label")
    plt.title("Two class of data")
    plt.show()

#visualize_data(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# load model 
model = BaggingClassifier()
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)

# model evaluation
accuracy = accuracy_score(y_predicted, y_test)
print("Here is accuracy of our model: {}%".format(int(accuracy*100))) # good bitch 100% accuracy

def make_random_predict(feature, label):
    rand_n = random.choice(range(100))
    random_feature = feature[rand_n]
    random_feature_2D = np.reshape(random_feature, (1,20))
    y_random_predicted = model.predict(random_feature_2D)
    if y_random_predicted == label[rand_n]:
        print("Congratulations!")
    else:
        print("Sorry! Something wrong")

make_random_predict(X, y) # yes yes model wrking correctly

# time for next algorithm
# rauf odilov