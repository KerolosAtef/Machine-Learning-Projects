import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv("ChurnData.csv")
    # print(df.columns)
    # print(df.head(5))
    x = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',
            'callcard', 'wireless', 'longmon', 'tollmon', 'equipmon', 'cardmon',
            'wiremon', 'longten', 'tollten', 'cardten', 'voice', 'pager',
            'internet', 'callwait', 'confer', 'ebill', 'loglong', 'logtoll',
            'lninc', 'custcat', ]].values.astype(float)
    y = df['churn'].values.astype(int)
    # mean normalization with unit standard diviation
    x = preprocessing.StandardScaler().fit(x).transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    lr_model = LogisticRegression(C=2)  # C is the inverse of lmbda in regularization
    lr_model.fit(x_train, y_train)
    y_hat = lr_model.predict(x_test)
    train_acc = lr_model.score(x_train, y_train)
    test_acc = lr_model.score(x_test, y_test)
    print("train accuracy", train_acc)
    print("test accuracy", test_acc)
    cnf_matrix = confusion_matrix(y_test, y_hat, labels=[1, 0])
    print(cnf_matrix)
    print(classification_report(y_test,y_hat))
    # Another way to calculate accuracy
    # print("train accuracy", accuracy_score(y_train, lr_model.predict(x_train)))
    # print("test accuracy", accuracy_score(y_test, y_hat))
