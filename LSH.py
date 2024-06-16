import numpy as np


class LSH:
    """
    Implements Locality-Sensitive Hashing (LSH) for dimensionality reduction
      and efficient similarity search in high-dimensional spaces.

    Attributes:
        num_hashes (int): The number of hash functions used, which corresponds to the number of random hyperplanes generated.
        hash_size (int): The number of bits in each hash, corresponding to the dimensionality of the output hash space.
        input_dim (int): The dimensionality of the input data.
        planes (list): A list of arrays where each array represents a hyperplane used to generate hash values.
    """

    def __init__(self, num_hashes, hash_size, input_dim):
        """
        Initializes an instance of the LSH class with specified parameters.

        Args:
            num_hashes (int): Number of hash functions (hyperplanes) to create.
            hash_size (int): The number of output dimensions for each hash function.
            input_dim (int): The dimensionality of the input data.
        """
        self.num_hashes = num_hashes
        self.hash_size = hash_size
        self.input_dim = input_dim

        # Initialize random hyperplanes. Each plane corresponds to a hash function.
        self.planes = [np.random.randn(input_dim, hash_size) for _ in range(num_hashes)]

    def hash(self, X):
        """
        Computes the hash values for a given input matrix X using all the defined hyperplanes.

        Args:
            X (np.ndarray): The data points to hash, where each row represents a data point.

        Returns:
            np.ndarray: A matrix of hash values where each row corresponds to the hashed output of a data point.
        """
        # Apply each hash function to X.
        hash_values = np.array([self._hash_single(X, plane) for plane in self.planes])
        return hash_values

    def _hash_single(self, X, plane):
        """
        Computes a single hash for the input data X using a specified hyperplane.

        Args:
            X (np.ndarray): The data points to hash.
            plane (np.ndarray): A hyperplane used as the hash function.

        Returns:
            np.ndarray: A binary vector where each element is 0 or 1 based on the side of the plane the data point lies.
        """
        projections = np.dot(X, plane)
        return (projections > 0).astype(int)

    def combine_hashes(self, hash_values, method="AND"):
        """
        Combines multiple hash values into a single hash code per data point using either the AND or OR method.

        Args:
            hash_values (np.ndarray): Hash values to combine.
            method (str): The method to use for combining hash values ('AND' or 'OR').

        Returns:
            np.ndarray: The combined hash value for each data point.
        """
        if method == "AND":
            combined = np.all(hash_values, axis=0).astype(int)
        elif method == "OR":
            combined = np.any(hash_values, axis=0).astype(int)
        return combined

    def combined_hash(self, X, method="AND"):
        """
        Computes and combines hash values for the input data X using a specified combination method.

        Args:
            X (np.ndarray): The data points to hash.
            method (str): Specifies the method to combine the hash values ('AND' or 'OR').

        Returns:
            np.ndarray: The combined hash value for each data point using the specified method.
        """
        hash_values = self.hash(X)
        return self.combine_hashes(hash_values, method)
