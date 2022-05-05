# load libraires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# load the dataset
data = pd.read_csv("G:/rauf/STEPBYSTEP/Tutorial/Repeat Knowledge/ML/Random Forest/titanic.csv")
print(data.head(5))

# here we load dataset
# but before we have to do some job to clean dataset to the model
# in case we do not need some features and we have to fill missing datasets also
# in the final we will split data to label and feature and also test and train datasets

# first we have to drop some unuseful feature collumns

data.drop(['Name', 'Siblings/Spouses Aboard', 'Parents/Children Aboard'], axis=1, inplace=True)
print(data.head(5)) # ok we have only 4 features which is comfortable to our model

# next challenge is to change categorical data into numeric data
dummies = pd.get_dummies(data.Sex)
merged_data = pd.concat([data, dummies], axis='columns')

data = merged_data.drop('Sex', axis='columns')
print(data.head(5))


# lets split dataset
X, y = data.iloc[:,1:6].values, data.iloc[:, 0].values
print(X[:5], y[:5])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)


# model loading
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)
accuracy = accuracy_score(y_predicted, y_test)
print("Here is accuracy: {}% from our simpler Decision Tree model".format(int(accuracy*100)))

# here our model trained with 76% accuracy which is not good

# Now we will see how Random forest model which powerful ensemble model performs our task

RF_model = RandomForestClassifier()
RF_model.fit(X_train, y_train)
RF_y_predicted = RF_model.predict(X_test)
RF_accuracy = accuracy_score(RF_y_predicted, y_test)
print("Here is our accuracy: {}% from powerful Random Forest model".format(int(RF_accuracy*100)))
# here our model performs slightly better

# rauf odilov