# load libraries
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load dataset
data = pd.read_csv("G:/rauf/STEPBYSTEP/Tutorial/Repeat Knowledge/ML/SVM/svm_data.csv")
X = data.iloc[:,0:2].values
y = data.iloc[:,-1].values
print(data)
print(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

# load and train the model
model = SVC(kernel='rbf', random_state=1)
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_pred, y_test)
print("Here our accurcy score ladies and gentelmens: {}%".format(int(accuracy*100)))
