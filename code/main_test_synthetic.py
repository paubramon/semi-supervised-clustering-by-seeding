import numpy as np
from sklearn.cluster import KMeans
from seeded_KMeans import SeededKMeans, ConstrainedKMeans
from datasets.samples_generator import make_blobs
from prepare_iris_data import prepareIrisData
import matplotlib.pyplot as plt

# Parameters
num_clusters = 3
max_iterations = 300
tolerance = 0.00001
num_seeds = 10

test_X, test_y = make_blobs(n_samples=500, n_features=2, centers=num_clusters, cluster_std=2.5, center_box=(-10, 10),
                            shuffle=True, random_state=10)

# Plot ground truth
fig = plt.figure()
plt.scatter(test_X[:, 0], test_X[:, 1], c=test_y)

# KMeans
kmeans = KMeans(n_clusters=num_clusters, max_iter=max_iterations, tol=tolerance)
predicted_labels_kmeans = kmeans.fit_predict(test_X)

# Seeded KMeans
seeded_kmeans = SeededKMeans(seed_datapoints=(test_X[:num_seeds, :], test_y[:num_seeds]), append_seeds=False,
                             max_iter=max_iterations, n_clusters=num_clusters, tolerance=tolerance, verbose=True)
predicted_labels_seeded = seeded_kmeans.fit(test_X)

# Constrained KMeans
constrained_kmeans = ConstrainedKMeans(seed_datapoints=(test_X[:num_seeds, :], test_y[:num_seeds]), append_seeds=False,
                                       max_iter=max_iterations, n_clusters=num_clusters, tolerance=tolerance,
                                       verbose=True)
predicted_labels_const = constrained_kmeans.fit(test_X)

fig = plt.figure()
plt.scatter(test_X[:, 0], test_X[:, 1], c=predicted_labels_kmeans)
plt.title("Results with k-Means")

fig = plt.figure()
plt.scatter(test_X[:, 0], test_X[:, 1], c=predicted_labels_seeded)
plt.title("Results with Seeded k-Means")

fig = plt.figure()
plt.scatter(test_X[:, 0], test_X[:, 1], c=predicted_labels_const)
plt.title("Results with Constrained k-Means")
plt.show()
