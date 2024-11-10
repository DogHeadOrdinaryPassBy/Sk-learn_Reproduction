import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, n_iter=100):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.centroids = None

    def fit(self, X):
        n_samples, n_features = X.shape
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.n_iter):
            clusters = self._assign_clusters(X)
            new_centroids = self._compute_centroids(X, clusters)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, X):
        return self._assign_clusters(X)

    def _assign_clusters(self, X):
        distances = np.array([[np.linalg.norm(x - centroid) for centroid in self.centroids] for x in X])
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X, clusters):
        centroids = []
        for i in range(self.n_clusters):
            cluster_points = X[clusters == i]
            centroids.append(np.mean(cluster_points, axis=0))
        return np.array(centroids)
