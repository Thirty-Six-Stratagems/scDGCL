import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
class Cluster:
    def __init__(self, data, data_label, k):

        self.data = data
        self.cell = data.shape[0]
        self.gene = data.shape[1]
        self.label = data_label
        self.k = k

        self.random_state = 42
        self.w = 0.9

        self.sc = []
        self.fitness = []
        self.feature = []
        self.subset_index_list = []

    def get_feature_subset(self, squirrel):
        subset = []
        for i in range(squirrel.size):
            theta = squirrel[i]
            alpha = np.cos(theta) * np.cos(theta)
            beta = np.sin(theta) * np.sin(theta)
            if alpha < beta:
                subset.append(i)
        feature_subset = self.data[:, subset]
        return feature_subset, subset

    def fitness_without_label(self, subset, label, feature):
        f = (metrics.silhouette_score(subset, label, metric='euclidean') + 1) / 2
        if feature < 700: return -100, 0
        return self.w * f + (1 - self.w) * (1 - (feature / self.gene)), f

    def fitness_kmeans(self, population):
        self.sc.clear()
        self.fitness.clear()
        self.feature.clear()
        self.subset_index_list.clear()

        for unity in population:
            feature_subset, subset_index = self.get_feature_subset(unity)

            self.subset_index_list.append(subset_index)
            kmeans = KMeans(n_clusters=self.k, random_state=self.random_state, n_init=20).fit(feature_subset)  # Kmeans

            self.feature.append(feature_subset.shape[1])
            fit, f = self.fitness_without_label(feature_subset, kmeans.labels_, feature_subset.shape[1])
            self.fitness.append(fit)
            self.sc.append(f)
        return self.fitness, self.sc, self.feature, self.subset_index_list
