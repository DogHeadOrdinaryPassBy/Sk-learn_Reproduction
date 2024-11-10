import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if n_samples <= 1 or depth >= self.max_depth:
            return np.mean(y)

        best_feature, best_threshold = self._best_split(X, y)
        left_idx = X[:, best_feature] < best_threshold
        right_idx = X[:, best_feature] >= best_threshold
        left_subtree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y):
        best_feature, best_threshold = 0, 0
        return best_feature, best_threshold

    def _predict(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feature, threshold, left, right = tree
        if x[feature] < threshold:
            return self._predict(x, left)
        else:
            return self._predict(x, right)
