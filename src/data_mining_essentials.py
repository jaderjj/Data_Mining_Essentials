"""
Author: Riley Jefferson, rileyjefferson65@gmail.com
Git: https://github.com/jaderjj/Data_Mining_Essentials
"""


import random
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import numpy as np


def create_blobs(n_samples=1000):
    """
    Parameters
    ----------
    n_samples : TYPE, optional
        DESCRIPTION. The default is 1000.

    Returns
    -------
    X : Synthetic dataset with n_samples each having a restricted random
    amount of features (between 3 and 30)
    y : Cluster labels for X.

    """
    # Initialize a restricted random amount of features.
    n_features = random.randint(3, 30)
    X, y =  make_blobs(n_samples=n_samples, n_features=n_features)
    print(f'There are {n_samples} samples in the dataset, \
each having {n_features} features')
    return X, y


def dimensional_reduce_blobs(X):
    """
    Parameters
    ----------
    X : dtype; array -  Synthetic dataset with n_samples each having a
    restricted random amount of features (between 3 and 30)

    Returns
    -------
    X_pca: dtype; array - Dimensionnaly reduced X dataset (2 dimensions)
    X_tsne: dtype; array - Dimensionally reduced X dataset (2 dimensions)
    """
    # Use Principal Component Analysis (PCA) to dimensionally reduce data
    pca = PCA(n_components=2, svd_solver='full')
    X_pca = pca.fit_transform(X)
    # View how much variance was maintaned in the reduced PCA dataset
    print(f'PCA dimensionally reduced our \
dataset from {X.shape[1]} dimensions to 2. As a result it kept \
{(round(sum(pca.explained_variance_ratio_),2))*100}% of the original variance.')
    # Use TSNE to dimensionally reduce data
    X_tsne = TSNE(n_components=2).fit_transform(X)
    return X_pca, X_tsne


def kmeans_elbow(X_reduced):
    """
    Parameters
    ----------
    X_reduced: dtype; array, either X_pca or X_tsne from dimensional_reduce_blobs()

    Returns
    -------
    None.
    
    First need to find the optimal number of clusters with the
    unsupervised kmeans algorithm by using the Elbow Method on our dataset
    created in create_data()
    
    The Elbow method is a technique and the idea is to run k-means clustering 
    for a range of clusters k (let’s say from 1 to 15) and for each value, 
    we are calculating the sum of squared distances from each point to its
    assigned center (distortions). Plotting these distortions vs. 
    the number of clusters will show an inflection (elbow) point, this is typically
    the ideal number of clusters to use.
    """
    assert (X_reduced.shape[1] == 2)
    # Run kmeans for a range of clusters to calculate distortions.
    distortions = []
    nclusters = range(1, 10)
    for k in nclusters:
        KM = KMeans(n_clusters=k)
        KM.fit(X_reduced)
        distortions.append(KM.inertia_)
    # plot the distortions
    plt.figure(figsize=(9,5))
    plt.plot(nclusters, distortions, 'rx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.title('Kmeans Elbow Method')
    plt.show()


def plot_clustering(X_reduced, y, title, time):
    df = pd.DataFrame(X_reduced, columns=["X", "Y"])
    df["cluster"] = y
    plt.figure(figsize=(9, 5))
    ax = sns.scatterplot(x="X", y="Y", 
                         hue=df.cluster.tolist(),
                         data=df)
    plt.title('%s - algorithm time: %s s' % (title, round(time,2)))
    plt.show()


def clustering(X_reduced):
    """
    Parameters
    ----------
    X_reduced: dtype; array, either X_pca or X_tsne from dimensional_reduce_blobs()

    Returns
    -------
    None.

    """
    # run k-means elbow methods and get nclusters at inflection point
    kmeans_elbow(X_reduced)
    try:
        ncluster_prompt = int(input('Enter nclusters at inflection point \
from elbow method: '))
    except ValueError:
        print('Not a valid integer input. Try again')
    if ncluster_prompt > 10:
        raise Exception('Must specify less than 10 clusters.')
    kmeans_t1 = time.process_time()
    kmeans_blobs = KMeans(n_clusters=ncluster_prompt,
                          init='k-means++',
                          n_init=1000).fit(X_reduced)
    kmeans_t2 = time.process_time()
    # plot kmeans clusters with the algorithm run time
    plot_clustering(X_reduced, kmeans_blobs.labels_, "K-means", kmeans_t2 - kmeans_t1)
    # run DBSCAN clustering and plot with run time
    dbscan_time_1 = time.process_time()
    dbscan = DBSCAN(eps=4).fit(X_reduced)
    dbscan_time_2 = time.process_time()
    plot_clustering(X_reduced, dbscan.labels_, "DBSCAN", dbscan_time_2 - dbscan_time_1)

  
def kmeans_silhoutte(X_reduced):
    """
    Another intrinsic metric to evaluate the quality of a clustering is 
    silhouette analysis, which can also be applied to clustering algorithms 
    such as k-means. Silhouette analysis can be used as a graphical tool to 
    plot a measure of how tightly grouped the samples in the clusters are. To
    calculate the silhouette coefficient three steps are required:
        1. Calculate cluster cohesion, i.e. the average distance between a sample
        and all other points in its cluster, a(i)
        2. Calculate cluster separation, i.e. the average distance between a sample
        and all samples in the neartest cluster, b(i)
        3. Calculate the silhouetter, s(i) = (b(i) - a(i)) / max{b(i)-a(i)}

    Parameters:
    ----------
    X_reduced: dtype; array, either X_pca or X_tsne from dimensional_reduce_blobs()
    """
    from sklearn.metrics import silhouette_samples
    from matplotlib import cm
    km = KMeans(n_clusters=3, init='k-means++',
                n_init=1000)
    y_km = km.fit_predict(X_reduced)
    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0] 
    silhouette_vals = silhouette_samples(X_reduced, y_km, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0 
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c] 
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(i / n_clusters) 
        plt.barh(range(y_ax_lower, y_ax_upper),
                 c_silhouette_vals, height=1.0, 
                 edgecolor='none', color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals) 
    silhouette_avg = np.mean(silhouette_vals) 
    plt.axvline(silhouette_avg, color="red",
                linestyle="--") 
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.show()

    


