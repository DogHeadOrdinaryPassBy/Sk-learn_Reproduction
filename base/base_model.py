class BaseModel:
    def fit(self, X, y):
        raise NotImplementedError("fit method not implemented")

    def predict(self, X):
        raise NotImplementedError("predict method not implemented")

    def score(self, X, y):
        predictions = self.predict(X)
        return (predictions == y).mean()
