import pandas as pd
import matplotlib.pyplot as plt

root_name = 'diff3newsgroup_'
metric = '_adj'

k_means_results = pd.read_csv(root_name + 'kmeans'+metric)
seeded_k_means_results = pd.read_csv(root_name + 'seeded_kmeans'+metric)
constrained_k_means_results = pd.read_csv(root_name + 'constrained_kmeans'+metric)
plt.figure()
plt.plot(k_means_results['mean'], label = 'kmeans')
plt.plot(seeded_k_means_results['mean'], label = 'seeded-kmeans')
plt.plot(constrained_k_means_results['mean'], label = 'constrained-kmeans')
plt.legend()
plt.grid()