"""
Seeded KMeans:
"""

# Author : Pau Bramon Mora <paubramonmora@gmail.com>

import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min


class SeededKMeans(object):
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
    append_seeds: if True, the seeds will be added to the dataset in the fit method. If it is False, the seeds will not
    be used after the initialization

    """

    def __init__(self, seed_datapoints=([], []), n_clusters=10, max_iter=100, append_seeds=True, tolerance=1e-5,
                 verbose=False):
        self.seed_datapoints = seed_datapoints
        self.K = n_clusters
        self.max_iter = max_iter
        self.append_seeds = append_seeds
        self.tolerance = tolerance
        self.verbose = verbose

    def _check_seed_datapoints(self, seed_datapoints):
        """Checks the input format of the seed datapoints"""
        if type(seed_datapoints) != tuple or len(seed_datapoints) != 2:
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
        # if len(np.unique(seed_datapoints[1])) > self.K:
        #    raise ValueError("More clusters than the specified number n_clusters have been detected in the seed input")

        self.Sx = seed_datapoints[0]
        self.Sy = seed_datapoints[1]

    def _initialize_random_centroids(self):
        """initializes centeroids with random values"""
        random_seeds = np.random.permutation(self.X.shape[0])[:self.K]
        if random_seeds.size >= self.K:
            # In the normal process
            self.centroids = self.X[random_seeds, :]
        else:
            # In order to run extreme experiments with all seeds and no input datapoints
            self.centroids = np.random.rand(self.K, self.Sx.shape[1])

    def _check_data_input(self, X):
        """Checks the input format of the datapoints and if it at least larger than the number of clusters"""
        if type(X) != np.ndarray:
            raise ValueError("Input data should be a numpy array")
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

    def _calculate_euclidean_distance(self, vector, matrix):
        """Calculates the euclidean distance between vectors within two matrices (or a vector and a matrix)
            Note that the vector variable can be either a vector or a matrix of the same dimensions as the other
            variable.
        """
        # This method calculates the euclidean distance between a vector and all elements of the matrix, returning a
        # vector of distances.
        d = vector - matrix
        distances = d ** 2
        return np.sqrt(np.sum(distances, 1))

    def _assign_clusters(self):
        # This method assigns the closest centroid to each instance

        # Old method, computationally inefficient
        # temp_distance_matrix = np.zeros((self.X.shape[0], self.K))
        # for i in range(self.K):
        #     temp_distance_matrix[:, i] = self._calculate_euclidean_distance(self.centroids[i, :], self.X)
        # self.cluster_assignments = np.argmin(temp_distance_matrix, axis=1)

        # New method, computationally efficient
        self.cluster_assignments, distances = pairwise_distances_argmin_min(self.X, self.centroids)

    def _compute_centroids(self):
        # This method recalculates the centroids centers.
        for i in range(self.K):
            if i in self.cluster_assignments:
                self.centroids[i, :] = np.mean(self.X[np.where(self.cluster_assignments == i)[0], :], axis=0)

    def _print_log(self, start=False, iteration=0, total=100):
        if self.verbose:
            if start:
                print("Start Seeded k-Means:")
            else:
                print("Seeded k-Means Iteration: %i/%i" % (iteration, total))

    def _fitting_loop(self):
        # This is the clustering loop
        finished = False
        iteration = 0
        while not finished:
            self._print_log(start=False, iteration=iteration, total=self.max_iter)
            self._assign_clusters()
            old_centroids = np.array(self.centroids)
            self._compute_centroids()
            iteration += 1

            # Check convergence or max iterations
            max_variation_centroids = max(self._calculate_euclidean_distance(old_centroids, self.centroids))
            if iteration >= self.max_iter or max_variation_centroids <= self.tolerance:
                finished = True
        self._assign_clusters()

    def _fit(self):
        # Starts fitting
        self._print_log(start=True)
        self._initialize_centroids()
        if self.append_seeds:
            if self.X.size == 0:
                # In order to run extreme experiments with all seeds and no input datapoints
                self.X = self.Sx
            else:
                self.X = np.vstack((self.X, self.Sx))

        # run normal kmeans
        self._fitting_loop()

    def fit(self, X):
        # Previous checks
        self._check_seed_datapoints(self.seed_datapoints)
        self._check_data_input(X)

        # Fitting procedure
        self._fit()

        return self.cluster_assignments

    def predict(self, X):
        self._check_data_input(X)
        self._assign_clusters()

        return self.cluster_assignments


class ConstrainedKMeans(SeededKMeans):
    """
    Constrained KMeans:
    Semi-supervised clustering algorithm described in the paper Sugato Basu, Arindam Banerjee, and R. Mooney. Semi-
    supervised clustering by seeding with constrains. In Proceedings of 19th International Conference on Machine Learning (ICML-2002)

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

    def __init__(self, seed_datapoints=([], []), n_clusters=10, max_iter=100, tolerance=1e-5, verbose=False):
        super(ConstrainedKMeans, self).__init__(seed_datapoints, n_clusters=n_clusters, max_iter=max_iter,
                                                append_seeds=True, tolerance=tolerance, verbose=verbose)

    def _assign_clusters(self):
        # This method assigns the closest centroid to each instance

        # Old method, computationally inefficient
        # temp_distance_matrix = np.zeros((self.X.shape[0], self.K))
        # for i in range(self.K):
        # for i in range(self.K):
        #     temp_distance_matrix[:, i] = self._calculate_euclidean_distance(self.centroids[i, :], self.X)
        # self.cluster_assignments = np.argmin(temp_distance_matrix, axis=1)

        # New method, computationally efficient
        self.cluster_assignments, distances = pairwise_distances_argmin_min(self.X, self.centroids)

        # Reassign the seed instances.
        self.cluster_assignments[self.seeds_indexes] = self.seeds_initial_assignment

    def _print_log(self, start=False, iteration=0, total=100):
        if self.verbose:
            if start:
                print("Start Constrained k-Means:")
            else:
                print("Constrained k-Means Iteration: %i/%i" % (iteration, total))

    def _fit(self):
        # Starts fitting
        self._print_log(start=True)
        self._initialize_centroids()
        if self.X.size == 0:
            # Just for the extreme case with all seeds and no data.
            self.X = self.Sx
            self.seeds_initial_assignment = self.Sy
            self.seeds_indexes = range(self.X.shape[0])
        else:
            self.X = np.vstack((self.X, self.Sx))
            self.seeds_initial_assignment = self.Sy
            self.seeds_indexes = range(self.X.shape[0] - self.Sx.shape[0], self.X.shape[0])

        # run normal kmeans
        self._fitting_loop()

    def predict(self, X):
        self._check_data_input(X)
        self.cluster_assignments, distances = pairwise_distances_argmin_min(self.X, self.centroids)

        return self.cluster_assignments
