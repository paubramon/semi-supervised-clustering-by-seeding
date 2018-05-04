import numpy as np
from seeded_KMeans import SeededKMeans
from datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

test_X, test_y = make_blobs(n_samples=500, n_features=2, centers=3, cluster_std=1.0, center_box=(-10, 10), shuffle=True)
