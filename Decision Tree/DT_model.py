# load libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# load datset
data = pd.read_csv("G:/rauf/STEPBYSTEP/Tutorial/Repeat Knowledge/ML/Decision Tree/decision_tree_animal_detection.csv")
print(data.head(), data.shape)

def prepare_data(data):
    dummies = pd.get_dummies(data.color)
    merged_data = pd.concat([dummies, data], axis='columns')
    ready_data = merged_data.drop("color", axis='columns')
    print(ready_data)
    return ready_data
def split_data(data):
    # split dataset
    X = data.iloc[:, 0:5].values
    y = data.iloc[:, -1].values
    print(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test

def model_train(X_train, y_train):
    model = DecisionTreeClassifier(criterion='gini',
                                   random_state = 100)
    model.fit(X_train, y_train)
    print("Our model succesfully trained")
    return model

def test_model(X_test, model):
    y_predicted = model.predict(X_test)
    print("Our predicted value is: ", y_predicted)
    return y_predicted

def get_accuracy(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    print("Our accuracy is: ", accuracy)
    return accuracy

def predict_random(random_data, trained_model):
    result = trained_model.predict(random_data)
    if result=='crocodile':
        print("our program preidcting you entered details of \"Crocodile\"")
    elif result =='elephant':
        print("our program preidcting you entered details of \"Elephant\"")
    else:
        print("our program preidcting you entered details of \"Monkey\"")

random_animal_crocodile = [[0, 0, 1, 60, 400]] # I know this is crocodile
random_animal_elephant = [[0, 1, 0, 285, 972]] # I know this is elephant

def main():
    ready_data = prepare_data(data)
    X_train, X_test, y_train, y_test = split_data(ready_data)
    trained_model = model_train(X_train, y_train)
    y_predicted = test_model(X_test, trained_model)
    accuracy = get_accuracy(y_test, y_predicted)
    print("Our model succesfully trained and here is our accuracy score: ", accuracy)
    print("----> Here we will do some experiment")
    predict_random(random_animal_crocodile, trained_model) # model has to predict Crocodile
    predict_random(random_animal_elephant, trained_model) # I know it has to predict Elephant



if __name__=="__main__":
    main()



# Very good Our custom model trained with accuracy 100% 
# And predicted random data correctly
# by Rauf Odilov