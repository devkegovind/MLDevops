import pandas as pd
import numpy as np
import os

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import argsparse


def get_data():
    URL = "D:\Downloads\wine+quality\winequality-red.csv"

    try:
        df = pd.read_csv(URL, sep=";")
        return df
    except Exception as e:
        raise e

def evaluate(y_true, y_pred):
    '''mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)*100'''

    accuracy = accuracy_score(y_true, y_pred)*100
    return accuracy

    #return mae, mse, rmse, r2


def main():
    df = get_data()
    train,test = train_test_split(df)

    #Train Test Split with the raw data
    X_train = train.drop('quality', axis = 1)
    X_test = test.drop('quality', axis = 1)

    y_train = train[['quality']]
    y_test = test[['quality']]

    #Model Training
    '''lr = ElasticNet()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)'''

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)


    #Evaluate the Model
    #mae, mse, rmse, r2 = evaluate(y_test, pred)
    accuracy = evaluate(y_test, pred)

    print(df)
    print()
    print(train)
    print()
    print(test)
    print()
    print(X_train)
    print()
    print(X_test)
    print()
    print(y_train)
    print()
    print(y_test)
    print()
    print(f"Accuracy = {accuracy}") #Classifiation
    #print(f"MAE = {mae}, MSE = {mse}, RMSE = {rmse}, R2 = {r2}")---Regression


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        raise e 

