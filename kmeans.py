import pickle

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from kneed import KneeLocator

class Kmeans:
    def __init__(self):
        pass

    def run(self, data, n):
        print(data.isnull().any())
        print((data == np.inf).any())
        print((data == -np.inf).any())
        self.labelClustering(data, n)

    def save_model(self, model, filename):
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

    # def load_model(self, filename):
    #     with open(filename, 'rb') as f:
    #         model = pickle.load(f)
    #     return model

    def labelClustering(self, data, c):
        # buat list untuk menyimpan nilai inersia
        wcss = []
        db_scores = []
        ch_scores = []

        # lakukan iterasi dari 2 sampai 10 cluster
        for i in range(2, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(data)
            wcss.append(kmeans.inertia_)

            labels = kmeans.labels_
            db_score = davies_bouldin_score(data, labels)
            db_scores.append(db_score)
            ch_score = calinski_harabasz_score(data, labels)
            ch_scores.append(ch_score)

        # mencari titik elbow secara otomatis
        kl = KneeLocator(range(2, 11), wcss, curve="convex", direction="decreasing")

        # jumlah cluster optimal
        n = kl.elbow
        print('optimal elbow ', n)

        # Menampilkan scatter plot untuk Elbow Method
        plt.figure(figsize=(6, 6))
        plt.plot(range(2, 11), wcss, marker='o')
        plt.vlines(kl.elbow, min(wcss), max(wcss), colors='r', linestyles='dashed')
        plt.title(f'Elbow Method showing the optimal k')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()

        kmeans = KMeans(n_clusters=c, n_init=10)
        kmeans.fit(data)
        self.save_model(kmeans, './models/model.pkl')
        labels = kmeans.labels_

        print('K-means Centroid:', kmeans.cluster_centers_)

        score = silhouette_score(data, labels)
        print('K-means Centroid:', kmeans.cluster_centers_)
        print('K-means Silhouette score:', score)

        # Kalkulasi Calinski-Harabasz score
        ch_score = calinski_harabasz_score(data, labels)
        print("K-means Calinski-Harabasz score: ", ch_score)

        #kalkulasi davis boulder scrore
        db_score = davies_bouldin_score(data, labels)
        print("K-means Davies-Bouldin score: ", db_score)

        data['Cluster'] = labels
        tsne = TSNE(n_components=2)
        data_tsne = tsne.fit_transform(data.drop('Cluster', axis=1))
        data_2d = pd.DataFrame(data_tsne, columns=['Component 1', 'Component 2'])
        data_2d['Cluster'] = labels
        plt.figure(figsize=(6, 6))

        for cluster in data_2d['Cluster'].unique():
            cluster_data = data_2d[data_2d['Cluster'] == cluster]
            plt.scatter(cluster_data['Component 1'], cluster_data['Component 2'], label=f'Cluster {cluster}')

        plt.legend()
        plt.show()

        # Menampilkan scatter plot untuk Davies-Bouldin score
        plt.figure(figsize=(6, 6))
        plt.plot(range(2, 11), db_scores, marker='o')
        plt.title(f'Davies-Bouldin score: {db_score}')
        plt.xlabel('Number of clusters')
        plt.ylabel('DB score')
        plt.show()

        # Menampilkan scatter plot untuk Calinski-Harabasz score
        plt.figure(figsize=(6, 6))
        plt.plot(range(2, 11), ch_scores, marker='o')
        plt.title(f'Calinski-Harabasz score {ch_score}')
        plt.xlabel('Number of clusters')
        plt.ylabel('CH score')
        plt.show()
