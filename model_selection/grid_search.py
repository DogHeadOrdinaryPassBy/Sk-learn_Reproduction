from itertools import product

class GridSearchCV:
    def __init__(self, model, param_grid, cv=5):
        self.model = model
        self.param_grid = param_grid
        self.cv = cv

    def fit(self, X, y):
        best_score = -float('inf')
        best_params = None
        for params in product(*self.param_grid.values()):
            param_dict = dict(zip(self.param_grid.keys(), params))
            self.model.set_params(**param_dict)
            score = cross_validate(self.model, X, y, cv=self.cv)
            if score > best_score:
                best_score = score
                best_params = param_dict
        self.best_params_ = best_params
        self.best_score_ = best_score
