import numpy as np

class LSH:

    def __init__(self, num_hashes, hash_size, input_dim):
        self.num_hashes = num_hashes
        self.hash_size = hash_size
        self.input_dim = input_dim
        self.planes = [np.random.randn(input_dim, hash_size) for _ in range(num_hashes)]
    
    def hash(self, X):
        hash_values = np.array([self._hash_single(X, plane) for plane in self.planes])
        return hash_values
    
    def _hash_single(self, X, plane):
        projections = np.dot(X, plane)
        return (projections > 0).astype(int)
    
    def combine_hashes(self, hash_values, method='AND'):
        if method == 'AND':
            combined = np.all(hash_values, axis=0).astype(int)
        elif method == 'OR':
            combined = np.any(hash_values, axis=0).astype(int)
        return combined

    def combined_hash(self, X, method='AND'):
        hash_values = self.hash(X)
        return self.combine_hashes(hash_values, method)