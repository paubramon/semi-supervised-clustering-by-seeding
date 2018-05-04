"""
Seeded KMeans:
"""

# Author : Pau Bramon Mora <paubramonmora@gmail.com>

import numpy as np

class SeededKMeans:
    """
    Seeded KMeans:
    Semi-supervised clustering algorithm described in the paper Sugato Basu, Arindam Banerjee, and R. Mooney. Semi-
    supervised clustering by seeding. In Proceedings of 19th International Conference on Machine Learning (ICML-2002)

    Parameters
    -----------------------

    n_clusters : number of clusters to look for
    max_iter: maximum number of iterations
    seed_datapoints = (Sx,Sy): tupple defining the labelled datapoints to build the semi-supervised clustering
        algorithm.
        Sx: NxM numpy matrix containing the M attributes of the N seed datapoints.
        Sy: N numpy array with labels of the N datapoints to train the semisupervised method.
    append_seeds: if True, the seeds will be added to the dataset in the fit method. If the seeds are already in the X
        input, this variable must be set to False.

    """

    def __init__(self, seed_datapoints=([], []), n_clusters=10, max_iter=100, append_seeds=False):
        self.seed_datapoints = seed_datapoints
        self.K = n_clusters
        self.max_iter = max_iter
        self.append_seeds = append_seeds

    def _check_seed_datapoints(self, seed_datapoints):
        """Checks the input format of the seed datapoints"""
        if type(seed_datapoints) != tuple or len(seed_datapoints) == 2:
            raise ValueError('''seed_datapoints variable has an incorrect format. 
            It should be a tuple containing the labelled datapoints to build the semi-supervised clustering algorithm. 
            Sx: NxM matrix containing the M attributes of the N seed datapoints. 
            Sy: N labels of the N datapoints to train the semisupervised method.''')
        if type(seed_datapoints[0]) != np.ndarray:
            raise ValueError(
                "Sx seed data has not the correct format. This matrix should be definied as a numpy matrix")
        if type(seed_datapoints[1]) != np.ndarray:
            raise ValueError(
                "Sy seed data has not the correct format. This variable should be definied as a numpy array")
        if len(seed_datapoints[1].shape) != 1:
            raise ValueError("Sy dimension is not correct. This variable should be a simple 1 dimensional array")
        self.Sx = seed_datapoints[0]
        self.Sy = seed_datapoints[1]

    def _initialize_random_centroids(self):
        """initializes centeroids with random values"""
        random_seeds = np.random.permutation(self.X.shape[0])[:self.K]
        self.centroids = self.X[random_seeds, :]

    def _check_datainput(self, X):
        """Checks the input format of the datapoints and if it at least larger than the number of clusters"""
        if type(X) != np.ndarray:
            raise ValueError("Input data should be a numpy array")
        if X.shape[0] < self.K:
            raise ValueError("Not enough data. There should be at least K examples")
        self.X = X

    def _initialize_centroids(self):
        # Get seed instances grouped by label
        index = 0
        self.seed_dict = {}
        temp_Sy = np.array(self.Sy)
        for i in np.sort(np.unique(self.Sy)):
            self.seed_dict[index] = self.Sx[np.where(self.Sy == i)[0], :]
            temp_Sy[np.where(self.Sy == i)[0]] = index
            index += 1

        # initialize centers with random values (just in case not enough seeds where defined)
        self._initialize_random_centroids()
        for i in range(index):
            self.centroids[i, :] = np.mean(self.seed_dict[i], axis=0)

    def _calculateEuclideanDistance(self, vector, matrix):
        # This method calculates the euclidean distance between a vector and all elements of the matrix, returning a
        # vector of distances.
        d = vector - matrix
        distances = d ** 2
        return np.sqrt(np.sum(distances, 1))

    def _assign_clusters(self):
        # This method assigns the closest centroid to each instance
        temp_distance_matrix = np.zeros((self.X.shape[0], self.K))
        for i in range(self.K):
            temp_distance_matrix[:, i] = self._calculateEuclideanDistance(self.centroids[i, :], self.X)
        self.cluster_assignments = np.argmin(temp_distance_matrix, axis=1)

    def _compute_centroids(self):
        # This method recalculates the centroids centers.
        for i in range(self.K):
            self.centroids[i, :] = np.mean(self.X[np.where(self.cluster_assignments == i)[0], :], axis=0)

    def _fit(self):
        # This is the clustering loop
        finished = False
        iteration = 0
        while (not finished):
            self._assign_clusters()
            self._compute_centroids()
            iteration += 1

            # Check convergence or max iterations
            if iteration >= self.max_iter:
                finished = True
        self._assign_clusters()

    def fit(self, X):
        # Previous checks
        self._check_seed_datapoints(self.seed_datapoints)
        self._check_datainput(X)

        # Starts fitting
        print("Seeded KMeans fitting")
        self._initialize_centroids()
        if self.append_seeds:
            self.X = np.vstack((self.X, self.Sx))

        # run normal kmeans
        self._fit()
