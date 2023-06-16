import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean, mahalanobis

class KMeans:
    def __init__(self, n_clusters : int, distance : str = 'euclidian') -> None:
        self.n_clusters = n_clusters
        try:
            assert distance in ['euclidian', 'mahalanobis']
            self.distance_criterion = euclidean if distance == 'euclidian' else mahalanobis
        except AssertionError:
            print('Distância precisa ser ou euclidiana, ou de mahalanobis.')
        self.scaler : sklearn.preprocessing._data.StandardScaler = MinMaxScaler()
        self.data : np.ndarray = None


    def _normalize_data(self, data) -> np.ndarray:
        return self.scaler.fit_transform(data)
    

    def _find_partition(self) -> np.ndarray:
        distance_matrix : np.ndarray = np.zeros((self.data.shape[0], self.n_clusters))
        for i in range(self.data.shape[0]):
            for j in range(self.n_clusters):
                distance_matrix[i][j] = self.distance_criterion(self.data[i].reshape(1 ,-1)[0], self.centroids[j].reshape(1, -1)[0])
        return np.argmin(distance_matrix, axis=1)


    def fit(self, data : np.ndarray, n_epochs : int = 100) -> None:
        self.data = self._normalize_data(data)
        self.centroids : np.ndarray = np.random.rand(self.n_clusters, self.data.shape[1])
        for i in range(n_epochs):
            partitions = self._find_partition()
            partitions_indexes = np.array([np.where(partitions == i)[0] for i in range(self.n_clusters)])
            for j in range(self.n_clusters):
                mean = np.mean(self.data[partitions_indexes[j]], axis=0)
                if True not in np.isnan(mean):
                    self.centroids[j] = mean
    

    def predict(self) -> np.ndarray:
        data = self.scaler.inverse_transform(self.data)
        indexes = self._find_partition()
        return np.array([data[np.where(indexes == i)[0]] for i in range(self.n_clusters)])


    def get_centroids(self) -> np.ndarray:
        return self.scaler.inverse_transform(self.centroids)
