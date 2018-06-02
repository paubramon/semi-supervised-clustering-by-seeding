import matplotlib.pyplot as plt
from samples_generator import make_blobs
from sklearn.cluster import KMeans
from semi_supervised_KMeans import SeededKMeans, ConstrainedKMeans
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd

# Parameters
num_clusters = 20
max_iterations = 100
tolerance = 0.0001
num_seeds = 10

all_results_kmeans = None
all_results_seededkmeans = None
all_results_constrainedkmeans = None

filename = 'synthetic_'
clusters = [2, 5, 10, 20]
row_names = ['cluster_std_' + str(f) for f in clusters]
column_names = ['fold_' + str(i + 1) for i in range(10)] + ['mean']

for f in clusters:
    num_clusters = f
    data, labels = make_blobs(n_samples=5000, n_features=40, centers=num_clusters, cluster_std=20, center_box=(-10, 10),
                              shuffle=True, random_state=10)

    seed_percentage = 0.2
    adj_mi_kmeans = []
    adj_mi_seededkmeans = []
    adj_mi_constrainedkmeans = []
    kf = KFold(n_splits=10)
    kf.get_n_splits(data)
    iteration = 1
    for train_index, test_index in kf.split(data):
        print("Fold %i/10" % (iteration))
        iteration += 1
        train_X = data[train_index, :]
        train_y = labels[train_index]
        test_X = data[test_index, :]
        test_y = labels[test_index]

        train_X_semi, seeds_X, train_y_semi, seeds_y = train_test_split(train_X, train_y, test_size=seed_percentage)

        # Ensure all arrays are numpys
        train_X_semi = np.array(train_X_semi)
        seeds_X = np.array(seeds_X)
        seeds_y = np.array(seeds_y)

        # KMeans
        kmeans = KMeans(n_clusters=num_clusters, max_iter=max_iterations, tol=tolerance, init='random', n_jobs=-1)
        kmeans.fit_predict(train_X)
        predicted_labels_kmeans = kmeans.predict(test_X)

        # Seeded KMeans
        seeded_kmeans = SeededKMeans(seed_datapoints=(seeds_X, seeds_y), append_seeds=True,
                                     max_iter=max_iterations, n_clusters=num_clusters, tolerance=tolerance,
                                     verbose=True)
        seeded_kmeans.fit(train_X_semi)
        predicted_labels_seeded = seeded_kmeans.predict(test_X)

        # Constrained KMeans
        constrained_kmeans = ConstrainedKMeans(seed_datapoints=(seeds_X, seeds_y),
                                               max_iter=max_iterations, n_clusters=num_clusters, tolerance=tolerance,
                                               verbose=True)
        constrained_kmeans.fit(train_X_semi)
        predicted_labels_const = constrained_kmeans.predict(test_X)

        adj_mi_kmeans.append(adjusted_mutual_info_score(labels_true=test_y, labels_pred=predicted_labels_kmeans))
        adj_mi_seededkmeans.append(adjusted_mutual_info_score(labels_true=test_y, labels_pred=predicted_labels_seeded))
        adj_mi_constrainedkmeans.append(
            adjusted_mutual_info_score(labels_true=test_y, labels_pred=predicted_labels_const))

    if all_results_kmeans is None:
        all_results_kmeans = adj_mi_kmeans
        all_results_seededkmeans = adj_mi_seededkmeans
        all_results_constrainedkmeans = adj_mi_constrainedkmeans
    else:
        all_results_kmeans = np.vstack((all_results_kmeans, adj_mi_kmeans))
        all_results_seededkmeans = np.vstack((all_results_seededkmeans, adj_mi_seededkmeans))
        all_results_constrainedkmeans = np.vstack((all_results_constrainedkmeans, adj_mi_constrainedkmeans))

mean_results_kmeans = np.mean(all_results_kmeans, axis=1)
all_results_kmeans = np.column_stack((all_results_kmeans, mean_results_kmeans))
final_results_kmeans = pd.DataFrame(all_results_kmeans, columns=column_names, index=row_names)
final_results_kmeans.to_csv(filename + 'kmeans')

mean_results_seededkmeans = np.mean(all_results_seededkmeans, axis=1)
all_results_seededkmeans = np.column_stack((all_results_seededkmeans, mean_results_seededkmeans))
final_results_seededkmeans = pd.DataFrame(all_results_seededkmeans, columns=column_names, index=row_names)
final_results_seededkmeans.to_csv(filename + 'seededkmeans')

mean_results_constrainedkmeans = np.mean(all_results_constrainedkmeans, axis=1)
all_results_constrainedkmeans = np.column_stack((all_results_constrainedkmeans, mean_results_constrainedkmeans))
final_results_constrainedkmeans = pd.DataFrame(all_results_constrainedkmeans, columns=column_names, index=row_names)
final_results_constrainedkmeans.to_csv(filename + 'constrainedkmeans')

plt.figure()
plt.plot(clusters, final_results_kmeans['mean'], label='kmeans')
plt.plot(clusters, final_results_seededkmeans['mean'], label='seeded-kmeans')
plt.plot(clusters, final_results_constrainedkmeans['mean'], label='constrained-kmeans')
plt.legend()
plt.grid()
plt.xlabel('Number of clusters')