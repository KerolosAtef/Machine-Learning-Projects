import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
# accuracy
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

def data_preprocessing(df):
    df = pd.read_csv("loan_train.csv")
    # print(df.head())
    df['due_date'] = pd.to_datetime(df['due_date'])
    df['effective_date'] = pd.to_datetime(df['effective_date'])
    # print(df.head())
    # print(df['loan_status'].value_counts())
    # Convert date to day of the week
    df['dayofweek'] = df['effective_date'].dt.dayofweek
    # print(df.groupby(['dayofweek'])['loan_status'].value_counts(normalize=True))
    # We see that people who get the loan at the end of the week dont pay it off,
    # so lets use Feature binarization to set a threshold values less then day 4
    df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x > 3) else 0)
    # 86 % of female pay there loans while only 73 % of males pay there loan
    # print(df.groupby(['Gender'])['loan_status'].value_counts(normalize=True))
    # Lets convert male to 0 and female to 1:
    df['Gender'].replace(to_replace=['male', 'female'], value=[0, 1], inplace=True)
    df['loan_status'].replace(to_replace=['PAIDOFF', 'COLLECTION'], value=[0, 1], inplace=True)
    # let's see the effect of education
    df.groupby(['education'])['loan_status'].value_counts(normalize=True)
    # Education one hot encoding
    feature = df[['Principal', 'terms', 'age', 'Gender', 'weekend']]
    feature = pd.concat([feature, pd.get_dummies(df['education'])], axis=1)
    feature.drop(['Master or Above'], axis=1, inplace=True)
    x = feature.values
    # Normalization
    x = preprocessing.StandardScaler().fit(x).transform(x)
    y = df['loan_status'].values
    return df, x, y


if __name__ == '__main__':
    train_data_frame=pd.read_csv("loan_train.csv")
    test_data_frame =pd.read_csv("loan_test.csv")
    df_train, X_train, Y_train = data_preprocessing(train_data_frame)
    df_test, x_test, y_test = data_preprocessing(test_data_frame)
    x_train, x_validation, y_train, y_validation = train_test_split(X_train, Y_train, test_size=0.2, shuffle=True)
    # KNN
    # validation_acc =[]
    # for k in range(1,20) :
    #     knn_model =KNeighborsClassifier(n_neighbors=k)
    #     knn_model.fit(x_train,y_train)
    #     validation_acc.append(knn_model.score(x_validation, y_validation))
    # best_validation_accuracy=max(validation_acc)
    # best_k =validation_acc.index(best_validation_accuracy)+1
    # print("Best k is ",best_k)
    # print("Best Validation accuracy is ",best_validation_accuracy)
    # knn_model = KNeighborsClassifier(n_neighbors=best_k)
    # test_acc= knn_model.score(x_test,y_test)
    # print("Test accuracy",test_acc)

    # Decision tree
    # decision_tree_model= DecisionTreeClassifier(criterion='entropy',max_depth=4)
    # decision_tree_model.fit(x_train,y_train)
    # train_acc=decision_tree_model.score(x_train,y_train)
    # validation_acc=decision_tree_model.score(x_validation, y_validation)
    # print("Train accuracy", train_acc)
    # print("Validation accuracy", validation_acc)

    # SVM
    # svm_model = svm.SVC(C=0.1, kernel='rbf')
    # svm_model.fit(x_train, y_train)
    # train_acc = svm_model.score(x_train, y_train)
    # validation_acc = svm_model.score(x_validation, y_validation)
    # print("Train accuracy", train_acc)
    # print("Validation accuracy", validation_acc)

    # Logistic regression model
    logistic_reg_model = LogisticRegression(C=0.1)
    logistic_reg_model.fit(x_train,y_train)
    train_acc=logistic_reg_model.score(x_train,y_train)
    validation_acc=logistic_reg_model.score(x_validation,y_validation)
    test_acc=logistic_reg_model.score(x_test,y_test)
    print("Train accuracy", train_acc)
    print("Validation accuracy", validation_acc)
    print("test accuracy", test_acc)
    logistic_reg_prediction=logistic_reg_model.predict(x_test)
    print(f1_score(y_test,logistic_reg_prediction,average='weighted'))
    print(log_loss(y_test,logistic_reg_prediction))


