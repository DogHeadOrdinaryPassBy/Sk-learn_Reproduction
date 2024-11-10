import numpy as np

def cross_validate(model, X, y, cv=5):
    fold_size = len(X) // cv
    scores = []
    for i in range(cv):
        start = i * fold_size
        end = (i + 1) * fold_size
        X_val, y_val = X[start:end], y[start:end]
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)
    return np.mean(scores)
