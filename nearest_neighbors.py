from sklearn.metrics.pairwise import distance
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from distances import euclidean_distance, cosine_distance
import numpy as np


class _MY_OWN_NEIGHBOURS():
    __slots__ = ('k', 'metric', 'matrix', 'test_block_size')

    def __init__(self, n_neighbors, metric, test_block_size):
        self.k = n_neighbors
        self.metric = euclidean_distance if metric == 'euclidean' else cosine_distance
        self.test_block_size = test_block_size

    def fit(self, X):
        self.matrix = X
        return self

    def kneighbors(self, X, return_distance):
        dists = self.metric(X, self.matrix)
        indexes = np.argpartition(dists, self.k)[:, :self.k]

        # numpy magic
        k_dists = np.take_along_axis(dists, indexes, axis=1)
        k_dists_indexes = np.argsort(k_dists)
        k_indexes = np.take_along_axis(indexes, k_dists_indexes, axis=1)

        return (
            np.take_along_axis(k_dists, k_dists_indexes, axis=1),
            k_indexes
        ) if return_distance else k_indexes


def KNN_type_checker(func):
    def inner(self, k: int, strategy: str, metric: str, weights: bool, test_block_size: int):
        if not isinstance(k, int):
            raise TypeError('k must be int')
        if strategy not in ['my_own', 'brute', 'kd_tree', 'ball_tree']:
            raise TypeError('not valid name of strategy')
        if metric not in ['euclidean', 'cosine']:
            raise TypeError('not valid name of metric')
        if not isinstance(weights, bool):
            raise TypeError('weights must be a boolean')
        if not isinstance(test_block_size, int) and test_block_size is not None:
            raise TypeError('test_block_size must be integer or None')
        if strategy in ['kd_tree', 'ball_tree'] and metric == 'cosine':
            raise TypeError('Cosine can not be used with such strategy')
        func(self, k=k, strategy=strategy, metric=metric, weights=weights, test_block_size=test_block_size)

    return inner


class KNNClassifier():
    __slots__ = ('k', 'strategy', 'metric', 'weights', 'test_block_size', 'target')

    @KNN_type_checker
    def __init__(self, k: int, strategy: str, metric: str, weights: bool, test_block_size: int):
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size

        if strategy in ['brute', 'kd_tree', 'ball_tree']:
            self.strategy = NearestNeighbors(n_neighbors=k, metric=metric, algorithm=strategy)
        else:
            self.strategy = _MY_OWN_NEIGHBOURS(n_neighbors=k, metric=metric, test_block_size=test_block_size)

    def fit(self, X: np.ndarray, y):
        self.target = y
        self.strategy.fit(X)

        return self

    def find_kneighbors(self, X, return_distance):

        def _get_parts(self, Y):
            if self.test_block_size is None:
                yield Y
            else:
                for block_index in range(int(Y.shape[0] / self.test_block_size) + 1):
                    yield Y[self.test_block_size * block_index: self.test_block_size * (block_index + 1)]

        answ = []
        answ_dist = []

        for part in _get_parts(self, X):
            if len(part) == 0:
                break

            if return_distance:
                temp_dist, temp_answ = self.strategy.kneighbors(part, return_distance=return_distance)
                answ_dist.append(temp_dist)
                answ.append(temp_answ)
            else:
                answ.append(self.strategy.kneighbors(part, return_distance=return_distance))

        if return_distance:
            return np.vstack(answ_dist), np.vstack(answ)
        else:
            return np.vstack(answ)

    def predict(self, X: np.ndarray, *, _k=None, _dists=None, _indexes=None):

        def _find_best_match(weights, indexes):
            buckets = defaultdict(lambda: 0)
            for i, ind in enumerate(indexes):
                buckets[self.target[ind]] += weights[i]
            return max(buckets, key=buckets.get)

        if _k is not None or _dists is not None or _indexes is not None:
            if _k is None or _dists is None or _indexes is None:
                raise TypeError("all 3 aprams k, dists, indx must be None or not None")
            else:
                answ = []
                for _dist, _indx in zip(_dists, _indexes):
                    answ.append(_find_best_match(_dist, _indx))
                return answ

        if self.weights:
            dists, indexes = self.find_kneighbors(X, True)
            answ = []
            for dist, indx in zip(dists, indexes):
                answ.append(_find_best_match(1 / (np.sort(dist) + 1E-5), indx))
            return answ
        else:
            indexes = self.find_kneighbors(X, False)
            answ = []
            for indx in indexes:
                answ.append(_find_best_match(np.ones(indx.shape), indx))
            return answ
