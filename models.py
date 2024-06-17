from collections import defaultdict
import numpy as np
from LSH import *

import numpy as np


class KMeans:
    """
    A simple implementation of the K-means clustering algorithm.

    Attributes:
        k (int): Number of clusters.
        centers (np.ndarray): Array of cluster centers.
        labels (np.ndarray): Array of labels indicating the cluster assignment of each point.
        convergence_history (list): List storing the average distance to centers at each iteration, showing convergence.
        iterations (int): Total number of iterations run until convergence.
        distance_computations (int): Total number of distance calculations performed.
    """

    def __init__(self, k):
        """
        Initializes the KMeans instance with the specified number of clusters.

        Args:
            k (int): Number of clusters.
        """
        self.k = k

    def fit(self, X):
        """
        Computes K-means clustering.

        Args:
            X (np.ndarray): Data points to cluster in a 2D numpy array format.
        """
        # Initialize cluster centers as the first 'k' elements of the dataset.
        self.centers = X[: self.k]

        self.labels = np.zeros(X.shape[0])
        self.convergence_history = []
        self.iterations = 0

        while True:
            # Record the current labels to check for convergence later.
            prev_labels = self.labels.copy()

            self.labels = self._assign_clusters(X)
            self.centers = self._compute_centers(X)

            # Check for convergence: if labels haven't changed, stop.
            if np.all(prev_labels == self.labels):
                break

            # Track convergence
            self.convergence_history.append(
                np.mean(np.linalg.norm(X - self.centers[self.labels], axis=1))
            )
            self.iterations += 1

        # Calculate total distance computations
        self.distance_computations = self.iterations * X.shape[0] * self.k

    def fit_predict(self, X):
        """
        Fits the K-means model to the data and returns the cluster labels.

        Args:
            X (np.ndarray): Data points to cluster.

        Returns:
            np.ndarray: Cluster labels for each point in the dataset.
        """
        self.fit(X)
        return self.labels

    def _calculate_distances(self, X, centers):
        """
        Calculate the Euclidean distance between each point and each cluster center.

        Args:
            X (np.ndarray): Data points.
            centers (np.ndarray): Cluster centers.

        Returns:
            np.ndarray: Distances from each point to each cluster center.
        """
        X_sq = np.sum(X**2, axis=1).reshape(-1, 1)
        centers_sq = np.sum(self.centers**2, axis=1).reshape(1, -1)
        squared_distances = X_sq + centers_sq - 2 * np.dot(X, self.centers.T)
        distances = np.sqrt(squared_distances)
        return distances

    def _assign_clusters(self, X):
        """
        Assigns each data point to the nearest cluster.

        Args:
            X (np.ndarray): Data points.

        Returns:
            np.ndarray: New cluster assignments.
        """
        distances = self._calculate_distances(X, self.centers)
        return np.argmin(distances, axis=1)

    def _compute_centers(self, X):
        """
        Computes the new centers as the mean of points assigned to each cluster.

        Args:
            X (np.ndarray): Data points.

        Returns:
            np.ndarray: New cluster centers.
        """
        new_centers = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            points_in_cluster = X[self.labels == i]
            if points_in_cluster.shape[0] > 0:
                new_centers[i] = points_in_cluster.mean(axis=0)
        return new_centers


import numpy as np
from collections import defaultdict


class KMeansLSH(KMeans):
    """
    KMeansLSH extends KMeans by utilizing LSH to enhance the efficiency and performance of
    clustering in high-dimensional data spaces.

    Attributes:
        k (int): Number of clusters.
        num_hashes (int): Number of hash functions for LSH.
        hash_size (int): Number of dimensions in each hash output.
        combination_method (str):  Specifies how hash values are combined in LSH ('AND' or 'OR').
        lsh (LSH): An instance of the LSH class that handles the hashing of data points.
        point_hashes (np.ndarray): Hash values of the data points processed through LSH.
    """

    def __init__(self, k, num_hashes=4, hash_size=4, combination_method="AND"):
        """
        Initializes the KMeansLSH object with specified parameters for clustering and hashing.

        Args:
            k (int): Number of desired clusters.
            num_hashes (int): Number of hash functions for LSH.
            hash_size (int): Number of dimensions in each hash output.
            combination_method (str): Specifies how hash values are combined in LSH ('AND' or 'OR').
        """
        super().__init__(k)
        self.num_hashes = num_hashes
        self.hash_size = hash_size
        self.combination_method = combination_method
        self.lsh = None

    def fit(self, X):
        """
        Fits the KMeansLSH model by performing clustering with LSH approach.

        Args:
            X (np.ndarray): The data points to be clustered, where each row corresponds to a data point.
        """
        # Initialize cluster centers and labels.
        self.centers = X[: self.k]
        self.labels = np.zeros(X.shape[0], dtype=int)
        self.convergence_history = []
        self.iterations = 0

        # Create LSH object for hashing data points.
        self.lsh = LSH(self.num_hashes, self.hash_size, X.shape[1])
        self.point_hashes = self.lsh.combined_hash(X, method=self.combination_method)

        while True:
            # Store previous labels for convergence check.
            prev_labels = self.labels.copy()

            self.labels = self._assign_clusters(X).astype(int)
            self.centers = self._compute_centers(X)

            # Check for convergence: if labels haven't changed, stop.
            if np.all(prev_labels == self.labels):
                break

            # Track convergence
            self.convergence_history.append(
                np.mean(np.linalg.norm(X - self.centers[self.labels], axis=1))
            )
            self.iterations += 1

        # Calculate total distance computations
        self.distance_computations = self.iterations * X.shape[0] * self.k

    def _assign_clusters(self, X):
        """
        Assigns clusters based on hash bucketing and distance to cluster centers.

        Args:
            X (np.ndarray): The data points to be clustered.

        Returns:
            np.ndarray: Updated labels for each data point in the dataset.
        """
        center_hashes = self.lsh.combined_hash(
            self.centers, method=self.combination_method
        )
        hash_buckets = defaultdict(list)

        # Bucket data points
        for idx, point_hash in enumerate(self.point_hashes.T):
            bucket_key = tuple(point_hash)
            hash_buckets[bucket_key].append(idx)

        labels = np.zeros(X.shape[0], dtype=int)

        # Assign points in the same hash bucket to the nearest center
        for center_idx, center_hash in enumerate(center_hashes.T):
            bucket_key = tuple(center_hash)
            if bucket_key in hash_buckets:
                for point_idx in hash_buckets[bucket_key]:
                    labels[point_idx] = center_idx

        # Handle points that do not fall into any bucket.
        remaining_points = np.setdiff1d(
            np.arange(X.shape[0]),
            np.array([idx for indices in hash_buckets.values() for idx in indices]),
        )
        distances = self._calculate_distances(X[remaining_points], self.centers)
        labels[remaining_points] = np.argmin(distances, axis=1)

        return labels
