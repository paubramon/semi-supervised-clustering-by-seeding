import numpy as np
from sklearn.datasets import fetch_20newsgroups_vectorized, fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split, KFold
from seeded_KMeans import SeededKMeans, ConstrainedKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import argparse

import pickle


def run_algorithm(algorithm, seed_percentage):
    # Parameters
    num_clusters = 20
    max_iterations = 100
    tolerance = 0.00005
    whole_dataset = False  # True or False

    # Get data and vectorize
    if whole_dataset:
        dataset_20newsgroups = fetch_20newsgroups_vectorized()
        data = dataset_20newsgroups.data.toarray()
        labels = dataset_20newsgroups.target
    else:
        newsgroups_train = fetch_20newsgroups(subset='train')
        cats = list(newsgroups_train.target_names)
        number_of_categories = 5;
        newsgroups_train = fetch_20newsgroups(subset='train', categories=cats[:number_of_categories])
        vectorizer = TfidfVectorizer()
        data = vectorizer.fit_transform(newsgroups_train.data).toarray()
        labels = newsgroups_train.target

    # data,labels = prepareIrisData()

    MI_score = []
    kf = KFold(n_splits=10)
    kf.get_n_splits(data)
    iteration = 1
    for train_index, test_index in kf.split(data):
        print("Fold %i/10" % (iteration))
        train_X = data[train_index, :]
        train_y = labels[train_index]
        test_X = data[test_index, :]
        test_y = labels[test_index]

        if seed_percentage == 1.0:
            train_X = np.array([])
            seeds_X = train_X
            train_y = []
            seeds_y = train_y
        else:
            train_X, seeds_X, train_y, seeds_y = train_test_split(train_X, train_y, test_size=seed_percentage)

        # KMeans
        if algorithm == "kmeans":
            kmeans = KMeans(n_clusters=num_clusters, max_iter=max_iterations, tol=tolerance)
            predicted_labels = kmeans.fit_predict(train_X)
        elif algorithm == "seeded_kmeans":
            # Seeded KMeans
            kmeans = SeededKMeans(seed_datapoints=(seeds_X, seeds_y), append_seeds=True,
                                  max_iter=max_iterations, n_clusters=num_clusters, tolerance=tolerance, verbose=True)
            predicted_labels = kmeans.fit(train_X)
        elif algorithm == "constrained_kmeans":
            # Constrained KMeans
            kmeans = ConstrainedKMeans(seed_datapoints=(seeds_X, seeds_y), append_seeds=True,
                                       max_iter=max_iterations, n_clusters=num_clusters, tolerance=tolerance,
                                       verbose=True)
            predicted_labels = kmeans.fit(train_X)

        predicted_labels = kmeans.predict(test_X)

        # Calculate MI
        MI_score.append(mutual_info_score(labels_true=test_y, labels_pred=predicted_labels))

        iteration += 1

    # Save history file
    return np.mean(MI_score)


def launcher():
    n_times = 10
    algorithms = ["kmeans", "seeded_kmeans", "constrained_kmeans"]
    seed_percentage = list(np.linspace(0, 1, 11))

    for algo in algorithms:
        for i in range(n_times):
            results = []
            for seed in seed_percentage:
                print("Running seed %f with algorithm %s" % (seed, algorithm))
                results.append(run_algorithm(algorithm=algo, seed_percentage=seed))

    print("Lets go!")

    seed_percentage = list(np.linspace(0, 1, 11))
    column_names = ['run_' + str(i) for i in range(n_times)]

    config_name = "20_newsgroups"
    results = []
    np_all_results = None
    for seed in seed_percentage:
        print("Running seed %f with algorithm %s" % (seed, algorithm))
        results.append(run_algorithm(algorithm=algo, seed_percentage=seed))
    if np_all_results is not None:
        np_all_results = np.column_stack((np_all_results, results))
    else:
        np_all_results = np_all_results

    mean_results = np.mean(np_all_results, axis=0)
    np_all_results = np.column_stack((np_all_results, mean_results))

    final_results = pd.Dataframe(np_all_results, columns=)

    with open(config_name + "/" + algorithm + ".p", 'wb') as file_pi:
        pickle.dump(results, file_pi)

def run_single_test(algo):
    print("Lets go!")
    seed_percentage = list(np.linspace(0, 1, 11))

    config_name = "20_newsgroups"
    results = []
    for seed in seed_percentage:
        print("Running seed %f with algorithm %s" % (seed, algorithm))
        results.append(run_algorithm(algorithm=algo, seed_percentage=seed))

    with open(config_name + "/" + algorithm + ".p", 'wb') as file_pi:
        pickle.dump(results, file_pi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='kmeans', help='Algorithm to run')
    args = parser.parse_args()
    algorithm = args.config
    possible_algorithms = ["kmeans", "seeded_kmeans", "constrained_kmeans"]
    if algorithm in possible_algorithms:
        run_algorithm(algorithm)
    elif algorithm is 'run_all':
        launcher()
    else:
        raise ValueError("Incorrect algorithm configuration!!!")
