import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean, mahalanobis
from sklearn.metrics import davies_bouldin_score

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
        self.distance = distance


    def _normalize_data(self, data) -> np.ndarray:
        return self.scaler.fit_transform(data)
    

    def _find_partition(self) -> np.ndarray:
        distance_matrix : np.ndarray = np.zeros((self.data.shape[0], self.n_clusters))
        iv = np.linalg.inv(np.cov(self.data.T))
        for i in range(self.data.shape[0]):
            for j in range(self.n_clusters):
                if self.distance == 'euclidian':
                    distance_matrix[i][j] = self.distance_criterion(self.data[i].reshape(1 ,-1)[0], self.centroids[j].reshape(1, -1)[0])
                else:
                    distance_matrix[i][j] = self.distance_criterion(self.data[i].reshape(1 ,-1)[0], self.centroids[j].reshape(1, -1)[0], iv)
        return np.argmin(distance_matrix, axis=1)


    def fit(self, data : np.ndarray, n_epochs : int = 20):
        self.data = self._normalize_data(data)
        self.centroids : np.ndarray = np.random.rand(self.n_clusters, self.data.shape[1])
        for i in range(n_epochs):
            partitions = self._find_partition()
            partitions_indexes = [np.where(partitions == i)[0] for i in range(self.n_clusters)]
            for j in range(self.n_clusters):
                if len(self.data[partitions_indexes[j]]) > 0:
                    mean = np.mean(self.data[partitions_indexes[j]], axis=0)
                    self.centroids[j] = mean
        return self
    

    def predict_points(self) -> np.ndarray:
        data = self.scaler.inverse_transform(self.data)
        indexes = self._find_partition()
        return [data[np.where(indexes == i)[0]] for i in range(self.n_clusters)]


    def predict_indexes(self) -> np.ndarray:
        return self._find_partition()
    

    def get_db_index(self) -> float:
        return davies_bouldin_score(self.data, self.predict_indexes())


    def get_nclusters(self) -> int:
        return self.n_clusters
