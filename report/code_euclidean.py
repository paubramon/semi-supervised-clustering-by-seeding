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

	temp_distance_matrix = np.zeros((self.X.shape[0], self.K))
	for i in range(self.K):
	    temp_distance_matrix[:, i] = self._calculate_euclidean_distance(self.centroids[i, :], self.X)
	self.cluster_assignments = np.argmin(temp_distance_matrix, axis=1)

