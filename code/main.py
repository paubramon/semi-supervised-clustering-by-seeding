import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score, accuracy_score, adjusted_mutual_info_score
from sklearn.model_selection import train_test_split, KFold
from semi_supervised_KMeans import SeededKMeans, ConstrainedKMeans
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer
from prepare_iris_data import prepareIrisData
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import pandas as pd
import argparse
import pickle
import time


def run_algorithm(algorithm, data, labels, seed_percentage, num_clusters, max_iterations=100, tolerance=0.00005):
    MI_score = []
    adjusted_MI_score = []
    accuracy = []
    performance_time = []
    kf = KFold(n_splits=10)
    kf.get_n_splits(data)
    iteration = 1
    for train_index, test_index in kf.split(data):
        print("Fold %i/10" % (iteration))
        train_X = data[train_index, :]
        train_y = labels[train_index]
        test_X = data[test_index, :]
        test_y = labels[test_index]

        init_time = time.time()

        if algorithm != 'kmeans':
            if seed_percentage == 1.0:
                train_X = np.array([])
                seeds_X = data[train_index, :]
                train_y = np.array([])
                seeds_y = labels[train_index]
            else:
                train_X, seeds_X, train_y, seeds_y = train_test_split(train_X, train_y, test_size=seed_percentage)

                # Ensure all arrays are numpys
                train_X = np.array(train_X)
                seeds_X = np.array(seeds_X)
                seeds_y = np.array(seeds_y)

        kmeans = None
        # KMeans
        if algorithm == "kmeans":
            kmeans = KMeans(n_clusters=num_clusters, max_iter=max_iterations, tol=tolerance, verbose=False, init='random',n_jobs=-1)
            predicted_labels = kmeans.fit_predict(train_X)
        elif algorithm == "seeded_kmeans":
            # Seeded KMeans
            kmeans = SeededKMeans(seed_datapoints=(seeds_X, seeds_y), append_seeds=True,
                                  max_iter=max_iterations, n_clusters=num_clusters, tolerance=tolerance, verbose=True)
            predicted_labels = kmeans.fit(train_X)
        elif algorithm == "constrained_kmeans":
            # Constrained KMeans
            kmeans = ConstrainedKMeans(seed_datapoints=(seeds_X, seeds_y), max_iter=max_iterations,
                                       n_clusters=num_clusters, tolerance=tolerance, verbose=True)
            predicted_labels = kmeans.fit(train_X)

        end_time = time.time()

        predicted_labels = kmeans.predict(test_X)

        # Calculate MI
        MI_score.append(mutual_info_score(labels_true=test_y, labels_pred=predicted_labels))
        adjusted_MI_score.append(adjusted_mutual_info_score(labels_true=test_y, labels_pred=predicted_labels))
        accuracy.append(accuracy_score(test_y, predicted_labels))
        performance_time.append(end_time - init_time)

        iteration += 1

    return MI_score, adjusted_MI_score, accuracy, performance_time


def launcher(algorithm, data, labels, num_clusters, file_root='test'):
    #algorithms = ["kmeans", "seeded_kmeans", "constrained_kmeans"]
    algorithms = ["kmeans"]
    seed_percentage = list(np.linspace(0, 1, 11))
    column_names = ['fold_' + str(i + 1) for i in range(10)] + ['mean']
    index_names = ['seed_' + str(i) for i in seed_percentage]

    for algo in algorithms:
        filename = file_root + '_' + algo
        np_all_results = None
        np_adj_all_results = None
        np_acc_all_results = None
        np_times = None

        for seed in seed_percentage:
            print("Running seed %f with algorithm %s" % (seed, algorithm))
            mi_scores, ad_mi_scores, acc_scores, perf_times = run_algorithm(algo, data, labels, seed, num_clusters)

            if np_all_results is not None:
                np_all_results = np.vstack((np_all_results, mi_scores))
                np_adj_all_results = np.vstack((np_adj_all_results, ad_mi_scores))
                np_acc_all_results = np.vstack((np_acc_all_results, acc_scores))
                np_times = np.vstack((np_times, perf_times))
            else:
                np_all_results = mi_scores
                np_adj_all_results = ad_mi_scores
                np_acc_all_results = acc_scores
                np_times = perf_times

        mean_results = np.mean(np_all_results, axis=1)
        np_all_results = np.column_stack((np_all_results, mean_results))

        mean_adj_results = np.mean(np_adj_all_results, axis=1)
        np_adj_all_results = np.column_stack((np_adj_all_results, mean_adj_results))

        mean_acc_results = np.mean(np_acc_all_results, axis=1)
        np_acc_all_results = np.column_stack((np_acc_all_results, mean_acc_results))

        mean_times = np.mean(np_times, axis=1)
        np_times = np.column_stack((np_times, mean_times))

        final_results = pd.DataFrame(np_all_results, columns=column_names, index=index_names)
        final_results.to_csv(filename + '_mi')

        final_adj_results = pd.DataFrame(np_adj_all_results, columns=column_names, index=index_names)
        final_adj_results.to_csv(filename + '_adj')

        final_acc_results = pd.DataFrame(np_acc_all_results, columns=column_names, index=index_names)
        final_acc_results.to_csv(filename + '_acc')

        final_times = pd.DataFrame(np_times, columns=column_names, index=index_names)
        final_times.to_csv(filename + '_time')


def run_single_test(algorithm, data, labels, num_clusters, file_root='test'):
    print("Lets go!")
    seed_percentage = list(np.linspace(0, 1, 11))
    seed_percentage = [1.0]

    config_name = file_root + '_' + algorithm
    results = []
    for seed in seed_percentage:
        print("Running seed %f with algorithm %s" % (seed, algorithm))
        _, temp_res, _, _ = run_algorithm(algorithm, data, labels, seed, num_clusters)
        results.append(temp_res)

    with open(config_name + "/" + algorithm + ".p", 'wb') as file_pi:
        pickle.dump(results, file_pi)


def main(algorithm, dataset):
    # Get dataset
    data = None
    labels = None
    num_clusters = None
    file_root = 'test'
    if dataset == '20newsgroup':
        num_clusters = 5
        file_root = '20newsgroup'
        newsgroups_train = fetch_20newsgroups(subset='train')
        cats = list(newsgroups_train.target_names)
        number_of_categories = num_clusters
        newsgroups_train = fetch_20newsgroups(subset='train', categories=cats[:number_of_categories])
        vectorizer = TfidfVectorizer()
        data = vectorizer.fit_transform(newsgroups_train.data).toarray()
        labels = newsgroups_train.target
    elif dataset == 'diff3newsgroup':
        num_clusters = 3
        file_root = 'diff3newsgroup'
        cats = ['alt.atheism', 'rec.sport.baseball', 'sci.space']
        newsgroups_data = fetch_20newsgroups(subset='all', categories=cats)
        labels = newsgroups_data.target
        hasher = HashingVectorizer(n_features=21631,
                                   stop_words='english', alternate_sign=False,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer(), Normalizer(copy=False))
        data = vectorizer.fit_transform(newsgroups_data.data).toarray()

    elif dataset == 'same3newsgroup':
        num_clusters = 3
        file_root = 'same3newsgroup'
        cats = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.windows.x']
        newsgroups_data = fetch_20newsgroups(subset='all', categories=cats)
        labels = newsgroups_data.target
        hasher = HashingVectorizer(n_features=21631,
                                   stop_words='english', alternate_sign=False,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer(), Normalizer(copy=False))
        data = vectorizer.fit_transform(newsgroups_data.data).toarray()
    elif dataset == 'iris':
        num_clusters = 3
        file_root = 'iris'
        data, labels = prepareIrisData()

    possible_algorithms = ["kmeans", "seeded_kmeans", "constrained_kmeans"]
    if algorithm in possible_algorithms:
        run_single_test(algorithm, data, labels, num_clusters, file_root=file_root)
    elif algorithm == 'run_all':
        launcher(algorithm, data, labels, num_clusters, file_root=file_root)
    else:
        raise ValueError("Incorrect algorithm configuration!!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='kmeans', help='Algorithm to run')
    parser.add_argument('--dataset', default='20newsgroup', help='Dataset to use')
    args = parser.parse_args()
    main(args.config, args.dataset)
