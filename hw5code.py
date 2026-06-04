import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов нужно брать среднее двух соседних при сортировке значений признака
    * Поведение функции в случае константного признака может быть любым
    * При одинаковых приростах критерия Джини для нескольких порогов нужно выбирать сплит, у которого значение порога минимально
    * Достаточно поддерживать только бинарную классификацию.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов, len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно разделить на две различные подвыборки или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds, len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    feature_vector = np.asarray(feature_vector, dtype=float)
    target_vector = np.asarray(target_vector)

    order = np.argsort(feature_vector, kind='mergesort')
    x_sorted = feature_vector[order]
    y_sorted = target_vector[order].astype(float)
    n = x_sorted.shape[0]

    candidate_thresholds = (x_sorted[:-1] + x_sorted[1:]) / 2.
    valid = x_sorted[:-1] != x_sorted[1:]
    if not np.any(valid):
        return np.array([]), np.array([]), None, None

    cum_ones = np.cumsum(y_sorted)[:-1]
    i_left = np.arange(1, n)
    i_right = n - i_left

    total_ones = y_sorted.sum()

    p1_left = cum_ones / i_left
    p1_right = (total_ones - cum_ones) / i_right
    H_left = 1.0 - p1_left  ** 2 - (1.0 - p1_left)  ** 2
    H_right = 1.0 - p1_right ** 2 - (1.0 - p1_right) ** 2
    Q = -(i_left / n) * H_left - (i_right / n) * H_right

    thresholds = candidate_thresholds[valid]
    ginis = Q[valid]

    best_idx = np.argmax(ginis)
    threshold_best = float(thresholds[best_idx])
    gini_best = float(ginis[best_idx])

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    """
    Простое классификационное дерево, поддерживающее:
    * real / categorical признаки
    * binary цели (метки могут быть числами или строками)
    * ограничения max_depth, min_samples_split, min_samples_leaf (как в sklearn по смыслу)

    ВНИМАНИЕ: в методе _fit_node ниже могут быть намеренно оставлены некоторые ошибки.
    Их нужно исправить в рамках задания.
    """
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        if len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    current_click = clicks.get(key, 0)
                    ratio[key] = current_click / current_count
                sorted_categories = [x[0] for x in sorted(ratio.items(),
                                                          key=lambda x: x[1])]
                categories_map = dict(zip(sorted_categories,
                                          range(len(sorted_categories))))
                feature_vector = np.array(
                    [categories_map[x] for x in sub_X[:, feature]]
                )
            else:
                raise ValueError

            if len(np.unique(feature_vector)) < 2:
                continue

            thresholds, ginis, _, _ = find_best_split(feature_vector, sub_y)
            if len(thresholds) == 0:
                continue

            sorted_vals = np.sort(feature_vector)
            left_sizes = np.searchsorted(sorted_vals, thresholds, side='left')
            right_sizes = len(feature_vector) - left_sizes

            ok = ((left_sizes >= self._min_samples_leaf) &
                  (right_sizes >= self._min_samples_leaf))
            if not np.any(ok):
                continue

            ginis_ok = ginis[ok]
            thresholds_ok = thresholds[ok]
            best_local = int(np.argmax(ginis_ok))
            threshold = float(thresholds_ok[best_local])
            gini = float(ginis_ok[best_local])

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [x[0] for x in categories_map.items()
                                      if x[1] < threshold]
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split],
                       node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split],
                       node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        feature_type = self._feature_types[feature]

        if feature_type == "real":
            to_left = x[feature] < node["threshold"]
        elif feature_type == "categorical":
            to_left = x[feature] in node["categories_split"]
        else:
            raise ValueError

        if to_left:
            return self._predict_node(x, node["left_child"])
        else:
            return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
