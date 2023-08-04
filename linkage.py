from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


class Linkages:
    def __init__(self):
        pass

    def run(self, data):
        # Menggunakan linkage untuk melakukan Hierarchical Clustering
        linked = linkage(data, 'ward')

        plt.figure(figsize=(10, 7))
        # Plot dendrogram
        dendrogram(linked,
                   orientation='top',
                   distance_sort='descending',
                   show_leaf_counts=True)
        plt.show()