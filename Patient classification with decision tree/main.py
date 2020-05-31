import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv("drug200.csv")
    # print(df.columns)
    x=df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
    y=df['Drug'].values
    x[:, 1] =preprocessing.LabelEncoder().fit(['F','M']).transform(x[:,1])
    x[:, 2] = preprocessing.LabelEncoder().fit(['LOW', 'HIGH','NORMAL']).transform(x[:, 2])
    x[:, 3] = preprocessing.LabelEncoder().fit(['HIGH', 'NORMAL']).transform(x[:, 3])
    x_train,x_test,y_train,y_test =train_test_split(x,y,random_state=3,shuffle=True,test_size=0.2)
    drug_tree=DecisionTreeClassifier(criterion='entropy',max_depth=4)
    drug_tree.fit(x_train,y_train)
    train_acc =drug_tree.score(x_train,y_train)
    test_acc=drug_tree.score(x_test,y_test)
    print("train accuracy",train_acc)
    print("test accuracy",test_acc)
    # Another way to calculate accuracy
    y_hat = drug_tree.predict(x_test)
    print("train accuracy", metrics.accuracy_score(y_train, drug_tree.predict(x_train)))
    print("test accuracy",metrics.accuracy_score(y_test,y_hat))

