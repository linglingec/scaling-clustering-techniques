import numpy as np
from sklearn.metrics import pairwise_distances


class Coreset:
    """
    A class for constructing a coreset from a given data set.

    Attributes:
        X (np.ndarray): The data points of the dataset, where each row is a data point.
        m (int): The desired size of the coreset.
        mu (np.ndarray): The mean of the dataset, computed across each dimension.
        q (np.ndarray): Probabilities associated with each data point in the dataset, used for sampling the coreset.

    Methods:
        get_q(): Calculates the sampling probabilities for each data point in the dataset.
        sample_coreset(): Samples `m` points from the dataset to form the coreset based on computed probabilities.
    """

    def __init__(self, X, m):
        """
        Initializes the Coreset object with a dataset and the size of the coreset.

        Args:
            X (np.ndarray): The data points of the dataset, where each row is a data point.
            m (int):  The desired size of the coreset.
        """
        self.X = X
        self.m = m
        self.mu = np.mean(X, axis=0)
        self.q = self.get_q()

    def get_q(self):
        """
        Computes the sampling probabilities for each data point in the dataset based on their squared distances to the mean.

        Returns:
            np.ndarray: An array of sampling probabilities for each data point.
        """
        n = len(self.X)
        dists = pairwise_distances(self.X, [self.mu]) ** 2
        dist_sum = np.sum(dists)
        q = (0.5 / n) + (0.5 * dists / dist_sum)
        return q.flatten()

    def sample_coreset(self):
        """
        Samples `m` points from the dataset using the probabilities computed in `get_q`.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The sampled coreset.
                - np.ndarray: The weights for each point in the coreset, computed as the inverse of their selection probability scaled by the number of points in the coreset.
        """
        # Normalize probabilities so they sum to 1.
        probs = self.q / np.sum(self.q)
        indices = np.random.choice(len(self.X), size=self.m, replace=False, p=probs)
        # Calculate weights for the sampled points.
        weights = 1 / (self.m * self.q[indices])
        coreset = self.X[indices]

        return coreset, weights
