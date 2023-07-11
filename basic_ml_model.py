import pandas as pd
import numpy as np
import os

import mlflow
import mlflow.sklearn
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet

from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def get_data():
    URL = "D:\Downloads\wine+quality\winequality-red.csv"

    try:
        df = pd.read_csv(URL, sep=";")
        return df
    except Exception as e:
        raise e


def evaluate(y_true, y_pred, pred_prob):
    '''mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)*100'''
    
    #return mae, mse, rmse, r2


    accuracy = accuracy_score(y_true, y_pred)*100
    roc_auc_score1 = roc_auc_score(y_true, pred_prob, multi_class = 'ovr')

    return accuracy, roc_auc_score1

    


def main(n_estimators, max_depth):
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



    rf = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth)
    rf.fit(X_train, y_train)

    pred = rf.predict(X_test)

    pred_prob = rf.predict_proba(X_test)


    #Evaluate the Model
    #mae, mse, rmse, r2 = evaluate(y_test, pred)
    accuracy, roc_auc_score1 = evaluate(y_test, pred, pred_prob)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('roc_auc_score', roc_auc_score1)

    #MLFlow Model Logging
    mlflow.sklearn.log_model(rf, "RandomForestModels")

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
    print(f"Accuracy = {accuracy}, ROC_AUC_Score = {roc_auc_score1}") #Classifiation

    #print(f"MAE = {mae}, MSE = {mse}, RMSE = {rmse}, R2 = {r2}")---Regression

    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--n_estimators", "-n", default = 50, type = int)
    args.add_argument("--max_depth", "-m", default = 5, type= int)
    parse_args = args.parse_args()
    try:
        main(n_estimators = parse_args.n_estimators, max_depth = parse_args.max_depth)
    except Exception as e:
        raise e 

