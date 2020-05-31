import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    df = pd.read_csv("cars_clus.csv")
    print(df.columns)
    df[['sales', 'resale', 'type', 'price', 'engine_s',
        'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
        'mpg', 'lnsales']] = df[['sales', 'resale', 'type', 'price', 'engine_s',
                                 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
                                 'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    df = df.reset_index(drop=True)
    x = df[['engine_s', 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']].values
    # Now we can normalize x. MinMaxScaler transforms features by scaling each feature to a given range.
    # It is by default (0, 1). That is, this estimator scales and translates each feature individually
    # such that it is between zero and one
    x = MinMaxScaler().fit_transform(x)  # Normalized x
    dist_matrix = distance_matrix(x, x)
    # print(dist_matrix)
    agglom = AgglomerativeClustering(n_clusters=6, linkage='complete')
    agglom.fit(x)
    # print(agglom.labels_)
    df['cluster_'] = agglom.labels_
    print(df.head())
    print(df.groupby(['cluster_','type'])['cluster_'].count())
    agg_cars = df.groupby(['cluster_', 'type'])['horsepow', 'engine_s', 'mpg', 'price'].mean()
    # print(agg_cars)
