import numpy as np
from scipy.spatial.distance import euclidean, mahalanobis

class KNearestNeighbors:
    def __init__(self, n_neighbors : int = 5, metric : str = 'euclidean') -> None:
        self.n : int = n_neighbors
        assert metric in ['euclidean', 'mahalanobis']
        self.metric : int = 0 if metric == 'euclidean' else 1
        self.X_train : np.ndarray = None
        self.y_train : np.ndarray = None
        self.cov : np.ndarray = None
    

    def fit(self, X_train : np.ndarray, y_train : np.ndarray) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.cov = np.cov(X_train.T)
    

    def pred(self, X_test : np.ndarray) -> np.ndarray:
        distances : np.ndarray = np.zeros((X_test.shape[0], self.X_train.shape[0]))
        if self.metric == 0:
            for i in range(X_test.shape[0]):
                for j in range(self.X_train.shape[0]):
                        distances[i][j] = euclidean(self.X_train[j], X_test[i])
        else:
            for i in range(X_test.shape[0]):
                for j in range(self.X_train.shape[0]):
                        distances[i][j] = mahalanobis(self.X_train[j], X_test[i], self.cov)

        indexes : np.ndarray = np.argpartition(distances, self.n)
        neighbors = self.y_train[indexes[:, :(self.n)]].reshape(X_test.shape[0], self.n)
        result = np.zeros((neighbors.shape[0], 1))
        for i in range(neighbors.shape[0]):
            result[i] = np.argmax(np.bincount(neighbors[i].astype(int)))
        return result

