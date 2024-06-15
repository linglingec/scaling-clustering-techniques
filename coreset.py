import numpy as np
from sklearn.metrics import pairwise_distances

class Coreset:

    def __init__(self, X, m):
        self.X = X
        self.m = m
        self.mu = np.mean(X, axis=0)
        self.q = self.get_q()

    def get_q(self):
        n = len(self.X)
        dists = pairwise_distances(self.X, [self.mu]) ** 2
        dist_sum = np.sum(dists)
        q = (0.5 / n) + (0.5 * dists / dist_sum)
        return q.flatten()

    def sample_coreset(self):
        probs = self.q / np.sum(self.q)
        indices = np.random.choice(len(self.X), size=self.m, replace=False, p=probs)
        weights = 1 / (self.m * self.q[indices])
        coreset = self.X[indices]
        return coreset, weights