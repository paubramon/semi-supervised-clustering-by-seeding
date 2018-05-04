"""
.. module:: samples_generator

samples_generator
*************

:Description: samples_generator

    

:Authors: bejar
    

:Version: 

:Created on: 21/01/2015 9:02 

"""

__author__ = 'bejar'

# from rpy2.robjects.packages import importr
import numpy as np
import numbers
from sklearn.utils import check_random_state, check_array


def make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0,
               center_box=(-10.0, 10.0), shuffle=True, random_state=None):
    """Generate isotropic Gaussian blobs for clustering.

    7/10/2015
    A fixed and more flexible version of the scikit-learn function

    Parameters
    ----------
    n_samples : int, or sequence of integers, optional (default=100)
        The total number of points equally divided among clusters.
        or a sequence of the number of examples of each cluster

    n_features : int, optional (default=2)
        The number of features for each sample.

    centers : int or array of shape [n_centers, n_features], optional
        (default=3)
        The number of centers to generate, or the fixed center locations.

    cluster_std: float or sequence of floats, optional (default=1.0)
        The standard deviation of the clusters.
        now works for the list of floats

    center_box: pair of floats (min, max), optional (default=(-10.0, 10.0))
        The bounding box for each cluster center when centers are
        generated at random.

    shuffle : boolean, optional (default=True)
        Shuffle the samples.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.

    y : array of shape [n_samples]
        The integer labels for cluster membership of each sample.

    Examples
    --------
    >>> from sklearn.datasets.samples_generator import make_blobs
    >>> X, y = make_blobs(n_samples=10, centers=3, n_features=2,
    ...                   random_state=0)
    >>> print(X.shape)
    (10, 2)
    >>> y
    array([0, 0, 1, 0, 2, 2, 2, 1, 1, 0])
    """
    generator = check_random_state(random_state)

    if isinstance(centers, numbers.Integral):
        centers = generator.uniform(center_box[0], center_box[1],
                                    size=(centers, n_features))
    else:
        centers = check_array(centers)
        n_features = centers.shape[1]

    X = []
    y = []

    n_centers = centers.shape[0]
    if not isinstance(n_samples, list):
        n_samples_per_center = [int(n_samples // n_centers)] * n_centers
        for i in range(n_samples % n_centers):
            n_samples_per_center[i] += 1
    else:
        if len(n_samples) != n_centers:
            raise NameError('List of number of examples per center doer not match number of centers')
        n_samples_per_center = n_samples
        n_samples = sum(n_samples)

    if not isinstance(cluster_std, list):
        std_list = [cluster_std] * centers.shape[0]
    else:
        if len(cluster_std) != n_centers:
            raise NameError('List of number of examples per center doer not match number of centers')
        std_list = cluster_std

    for i, (n, st) in enumerate(zip(n_samples_per_center, std_list)):
        X.append(centers[i] + generator.normal(scale=st,
                                               size=(n, n_features)))
        y += [i] * n

    X = np.concatenate(X)
    y = np.array(y)

    if shuffle:
        indices = np.arange(n_samples)
        generator.shuffle(indices)
        X = X[indices]
        y = y[indices]

    return X, y


# def cluster_generator(n_clusters=3, sepval=0.5, numNonNoisy=5, numNoisy=0, numOutlier=0,
#                       clustszind=2, clustSizeEq=100, rangeN=[100, 150], rotateind=True):
#     """
#     Generates clusters using the R package CusterGeneration
#     See the documentation of that package for the meaning of the parameters
#
#     You must have an R installation with the clusterGeneration package
#
#     :param n_clusters:
#     :param sepval:
#     :return:
#     """
#     clusterG = importr('clusterGeneration')
#
#     params = {'numClust': n_clusters,
#           'sepVal': sepval,
#           'numNonNoisy': numNonNoisy,
#           'numNoisy': numNoisy,
#           'numOutlier': numOutlier,
#           'numReplicate': 1,
#           'clustszind': clustszind,
#           'clustSizeEq': clustSizeEq,
#           'rangeN': rangeN,
#           'rotateind': rotateind,
#           'outputDatFlag': False,
#           'outputLogFlag': False,
#           'outputEmpirical': False,
#           'outputInfo': False
#          }
#
#     x = clusterG.genRandomClust(**params)
#     # nm = np.array(x[2][0].colnames)
#     # nm = np.concatenate((nm, ['class']))
#     m = np.matrix(x[2][0])
#     v = np.array(x[3][0])
#     v.resize((len(x[3][0])))
#     #m = np.concatenate((m, v), axis=1)
#     return m, v
