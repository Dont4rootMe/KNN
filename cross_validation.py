from collections import defaultdict
import numpy as np
from nearest_neighbors import KNNClassifier


def kfold(n: int, n_folds: int):
    k = [n // n_folds + 1 if i < n % n_folds else n // n_folds for i in range(n_folds)]
    answ = []
    for i in range(len(k)):
        test_bool = np.hstack([np.full((r), False) if ind != i else np.full((r), True) for ind, r in enumerate(k)])
        test = np.where(test_bool)[0]
        train = np.where(np.logical_not(test_bool))[0]
        answ.append((train, test))
    return answ


def CV_type_checker(func):
    def inner(X, y, k_list, score, cv, **kwargs):
        if 'k' in kwargs.keys():
            raise TypeError('k передается через лист k_list')
        if not isinstance(k_list, list):
            raise TypeError('k_list должен быть листом количества соседей')
        if not isinstance(score, str):
            raise TypeError('score должна быть строкой')
        if score not in ['accuracy']:
            raise TypeError('score указана не верно')
        answ = func(X, y, k_list, score, cv, **kwargs)
        return answ
    return inner


@CV_type_checker
def knn_cross_val_score(X, y, k_list, score, cv=None, **kwargs):

    def _accuracy(pred, ground):
        return np.sum(pred == ground) / len(pred)

    if cv is None:
        cv = kfold(len(X), int(5))

    k_scoring = defaultdict(lambda: [])
    k_max = sorted(k_list, reverse=True)[0]
    knn = KNNClassifier(**kwargs, k=k_max)

    for train_idx, test_idx in cv:
        model = knn.fit(X[train_idx], y[train_idx]).fit(X[train_idx], y[train_idx])
        dist, indexes = model.find_kneighbors(X[test_idx], True)

        for k in sorted(k_list):
            y_pred = model.predict(X[test_idx], _k=k, _dists=dist[:, :k], _indexes=indexes[:, :k])
            if score == 'accuracy':
                k_scoring[k].append(_accuracy(y_pred, y[test_idx]))

    return dict(k_scoring)
