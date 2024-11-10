import numpy as np

class RandomizedSearchCV:
    def __init__(self, estimator, param_distributions, n_iter=10, cv=5):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = -float("inf")

    def fit(self, X, y):
        for _ in range(self.n_iter):
            params = {key: np.random.choice(values) for key, values in self.param_distributions.items()}
            self.estimator.set_params(**params)
            scores = []
            for fold in range(self.cv):
                X_train, X_val, y_train, y_val = self._split_data(X, y, fold)
                self.estimator.fit(X_train, y_train)
                score = self.estimator.score(X_val, y_val)
                scores.append(score)
            avg_score = np.mean(scores)
            if avg_score > self.best_score_:
                self.best_score_ = avg_score
                self.best_params_ = params

    def _split_data(self, X, y, fold):
        n_samples = len(X)
        fold_size = n_samples // self.cv
        start = fold * fold_size
        end = start + fold_size
        X_val, y_val = X[start:end], y[start:end]
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)
        return X_train, X_val, y_train, y_val
