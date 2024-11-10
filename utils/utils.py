import numpy as np

def load_csv(file_path, delimiter=','):
    """Loads a CSV file into a NumPy array."""
    return np.genfromtxt(file_path, delimiter=delimiter)

def train_test_split(X, y, test_size=0.2, random_state=None):
    """Splits the dataset into training and testing sets."""
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(len(X) * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
