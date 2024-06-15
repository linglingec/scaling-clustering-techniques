from collections import defaultdict
import numpy as np
from LSH import *

class KMeans:

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        self.centers = X[:self.k]
        self.labels = np.zeros(X.shape[0])
        self.convergence_history = []
        self.iterations = 0

        while True:
            prev_labels = self.labels.copy()
            self.labels = self._assign_clusters(X)
            self.centers = self._compute_centers(X)

            if np.all(prev_labels == self.labels):
                break

            self.convergence_history.append(np.mean(np.linalg.norm(X - self.centers[self.labels], axis=1)))
            self.iterations += 1

        self.distance_computations = self.iterations * X.shape[0] * self.k

    def fit_predict(self, X):
        self.fit(X)
        return self.labels
    
    def _calculate_distances(self, X, centers):
        X_sq = np.sum(X ** 2, axis=1).reshape(-1, 1)
        centers_sq = np.sum(self.centers ** 2, axis=1).reshape(1, -1)
        squared_distances = X_sq + centers_sq - 2 * np.dot(X, self.centers.T)
        distances = np.sqrt(squared_distances)
        return distances
    
    def _assign_clusters(self, X):
        distances = self._calculate_distances(X, self.centers)
        return np.argmin(distances, axis=1)

    def _compute_centers(self, X):
        new_centers = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            points_in_cluster = X[self.labels == i]
            if points_in_cluster.shape[0] > 0:
                new_centers[i] = points_in_cluster.mean(axis=0)
        return new_centers

class KMeansLSH(KMeans):

    def __init__(self, k, num_hashes=4, hash_size=4, combination_method='AND'):
        super().__init__(k)
        self.num_hashes = num_hashes
        self.hash_size = hash_size
        self.combination_method = combination_method
        self.lsh = None
    
    def fit(self, X):
        self.centers = X[:self.k]
        self.labels = np.zeros(X.shape[0], dtype=int)
        self.convergence_history = []
        self.iterations = 0
        self.lsh = LSH(self.num_hashes, self.hash_size, X.shape[1])

        self.point_hashes = self.lsh.combined_hash(X, method=self.combination_method)
        
        while True:
            prev_labels = self.labels.copy()
            self.labels = self._assign_clusters(X).astype(int)
            self.centers = self._compute_centers(X)

            if np.all(prev_labels == self.labels):
                break

            self.convergence_history.append(np.mean(np.linalg.norm(X - self.centers[self.labels], axis=1)))
            self.iterations += 1

        self.distance_computations = self.iterations * X.shape[0] * self.k
    
    def _assign_clusters(self, X):
        center_hashes = self.lsh.combined_hash(self.centers, method=self.combination_method)
        hash_buckets = defaultdict(list)

        for idx, point_hash in enumerate(self.point_hashes.T):
            bucket_key = tuple(point_hash)
            hash_buckets[bucket_key].append(idx)

        labels = np.zeros(X.shape[0], dtype=int)
        for center_idx, center_hash in enumerate(center_hashes.T):
            bucket_key = tuple(center_hash)
            if bucket_key in hash_buckets:
                for point_idx in hash_buckets[bucket_key]:
                    labels[point_idx] = center_idx
        
        remaining_points = np.setdiff1d(np.arange(X.shape[0]), np.array([idx for indices in hash_buckets.values() for idx in indices]))
        distances = self._calculate_distances(X[remaining_points], self.centers)
        labels[remaining_points] = np.argmin(distances, axis=1)
        
        return labels