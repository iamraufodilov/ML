# load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# load data
data = pd.read_csv("G:/rauf/STEPBYSTEP/Tutorial/Repeat Knowledge/ML/AdaBoosting & Gradient Boosting/fruit.csv")
print(data.head(5))
# we have to change categorical variables to numeric such as color
dummies = pd.get_dummies(data.color)
data = pd.concat([data, dummies], axis=1)
print(data)

# separate data to feature and label
y = data.pop('target')
y = pd.array(y)
X = data.drop(columns='color')
X = X.iloc[:,:].values
print(X, y)

# visualize dataset
mean_X = list(np.mean(a) for a in X)
plt.scatter(mean_X, y)
plt.xlabel("X data")
plt.ylabel("y data")
plt.title("Our data visualization")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)


# load model
model = AdaBoostClassifier()
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)


# evaluation
accuracy = accuracy_score(y_predicted, y_test)
cm = confusion_matrix(y_predicted, y_test)
print("Here our model's accuracy: {}%".format(int(accuracy*100)))
print("Here is confusion matrix also", cm)

# here we go we will trained our model with 100% accuracy
# fuck off