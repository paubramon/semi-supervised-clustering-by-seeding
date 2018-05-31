import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mutual_info_score
from seeded_KMeans import SeededKMeans, ConstrainedKMeans
from prepare_iris_data import prepareIrisData
import matplotlib.pyplot as plt

# Parameters
num_clusters = 3
max_iterations = 300
tolerance = 0.00001
num_seeds = 10

test_X, test_y = prepareIrisData()

# Plot ground truth
fig = plt.figure()
plt.subplot(221)
plt.scatter(test_X[:, 0], test_X[:, 1], c=test_y)
plt.subplot(222)
plt.scatter(test_X[:, 1], test_X[:, 2], c=test_y)
plt.subplot(223)
plt.scatter(test_X[:, 2], test_X[:, 3], c=test_y)
plt.subplot(224)
plt.scatter(test_X[:, 1], test_X[:, 3], c=test_y)

# KMeans
kmeans = KMeans(n_clusters=num_clusters, max_iter=max_iterations, tol=tolerance)
predicted_labels_kmeans = kmeans.fit_predict(test_X)

# Seeded KMeans
seeded_kmeans = SeededKMeans(seed_datapoints=(test_X[:num_seeds, :], test_y[:num_seeds]), append_seeds=False,
                             max_iter=max_iterations, n_clusters=num_clusters, tolerance=tolerance)
predicted_labels_seeded = seeded_kmeans.fit(test_X)

# Constrained KMeans
constrained_kmeans = ConstrainedKMeans(seed_datapoints=(test_X[:num_seeds, :], test_y[:num_seeds]), append_seeds=False,
                                       max_iter=max_iterations, n_clusters=num_clusters, tolerance=tolerance)
predicted_labels_const = constrained_kmeans.fit(test_X)

# Calculate accuracies
MI_kmeans = mutual_info_score(labels_true=test_y,labels_pred=predicted_labels_kmeans)
MI_seeded = mutual_info_score(labels_true=test_y,labels_pred=predicted_labels_seeded)
MI_const = mutual_info_score(labels_true=test_y,labels_pred=predicted_labels_const)

# Kmeans
fig = plt.figure()
plt.subplot(221)
plt.scatter(test_X[:, 0], test_X[:, 1], c=predicted_labels_kmeans)
plt.subplot(222)
plt.scatter(test_X[:, 1], test_X[:, 2], c=predicted_labels_kmeans)
plt.subplot(223)
plt.scatter(test_X[:, 2], test_X[:, 3], c=predicted_labels_kmeans)
plt.subplot(224)
plt.scatter(test_X[:, 1], test_X[:, 3], c=predicted_labels_kmeans)
plt.suptitle("Results with k-means MI = " + str(MI_kmeans))

# Seeded Kmeans
fig = plt.figure()
plt.subplot(221)
plt.scatter(test_X[:, 0], test_X[:, 1], c=predicted_labels_seeded)
plt.subplot(222)
plt.scatter(test_X[:, 1], test_X[:, 2], c=predicted_labels_seeded)
plt.subplot(223)
plt.scatter(test_X[:, 2], test_X[:, 3], c=predicted_labels_seeded)
plt.subplot(224)
plt.scatter(test_X[:, 1], test_X[:, 3], c=predicted_labels_seeded)
plt.suptitle("Results with Seeded k-means MI = " + str(MI_seeded))

# Constrained Kmeans
fig = plt.figure()
plt.subplot(221)
plt.scatter(test_X[:, 0], test_X[:, 1], c=predicted_labels_const)
plt.subplot(222)
plt.scatter(test_X[:, 1], test_X[:, 2], c=predicted_labels_const)
plt.subplot(223)
plt.scatter(test_X[:, 2], test_X[:, 3], c=predicted_labels_const)
plt.subplot(224)
plt.scatter(test_X[:, 1], test_X[:, 3], c=predicted_labels_const)
plt.suptitle("Results with Constrained k-means MI = " + str(MI_const))
