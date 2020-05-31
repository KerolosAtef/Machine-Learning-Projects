import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    df = pd.read_csv("Cust_Segmentation.csv")
    # print(df.columns)
    df = df.drop(['Customer Id', 'Address'], axis=1)
    # print(df.columns)
    x = df[['Age', 'Edu', 'Years Employed', 'Income', 'Card Debt', 'Other Debt',
            'Defaulted', 'DebtIncomeRatio']].values
    # To remove any nan number in the dataset
    x = np.nan_to_num(x)
    # Normalize dataset
    x = StandardScaler().fit(x).transform(x)
    # Elbow method to choose the best k
    # inertia=[]
    # for n_clusters in range(1,20) :
    #     k_means_model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20)
    #     k_means_model.fit(x)
    #     inertia.append(k_means_model.inertia_)
    #     # print(k_means_model.inertia_)
    #     # print(k_means_model.score(x))
    #     # labels = k_means_model.labels_
    #     # print(labels)
    # print(inertia)
    # plt.plot(range(1,20),inertia)
    # plt.xlabel("number of clusters")
    # plt.ylabel("total distance")
    # plt.show()
    # best k is 7
    k_means_model = KMeans(n_clusters=7, init='k-means++', n_init=20)
    # n_init, default=10
    # Number of time the k-means algorithm will be run with different centroid seeds.
    # The final results will be the best output of n_init consecutive runs in terms of inertia.
    # init{‘k-means++’, ‘random’, ndarray, callable}, default=’k-means++’}
    # Method for initialization:
    # ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence
    k_means_model.fit(x)
    df["cluster class"] = k_means_model.labels_
    print(df.head())
    print(df["cluster class"].value_counts())
    print(df.groupby("cluster class").mean())
