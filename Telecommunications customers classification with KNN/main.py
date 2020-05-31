import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    data_frame = pd.read_csv("teleCust1000t.csv")
    # print(data_frame.columns)
    x = data_frame[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
                    'employ', 'retire', 'gender', 'reside']].values
    y = data_frame['custcat'].values
    # Normalization is better before KNN
    # StandardScaler gives us zero mean with unit standard diviation
    x = StandardScaler().fit(X=x).transform(x.astype(float))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
    knn = KNeighborsClassifier(n_neighbors=16).fit(x_train, y_train)
    y_hat = knn.predict(x_test)
    print("train accuracy", accuracy_score(y_train, knn.predict(x_train)))
    print("test accuracy", accuracy_score(y_test, y_hat))
    train_acc = []
    test_acc = []
    for k in range(1, 20):
        knn = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
        y_hat = knn.predict(x_test)
        train_acc.append(accuracy_score(y_train, knn.predict(x_train)))
        test_acc.append(accuracy_score(y_test, y_hat))

    print("train accuracy", train_acc)
    print("test accuracy", test_acc)

    # plot the graph
    plt.xlabel("number of k ")
    plt.ylabel("test accuracy")
    plt.plot(range(1, 20), test_acc)
    plt.show()
