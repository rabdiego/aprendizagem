import numpy as np
from sklearn.preprocessing import StandardScaler

class PCA:
    def __init__(self, n_features : int) -> None:
        self.n_features = n_features
        self.scaler = StandardScaler()
        self.explained_variance = None


    def fit_transform(self, data : np.ndarray) -> np.ndarray:
        data = self.scaler.fit_transform(data)

        covariance_matrix = np.cov(data.T)

        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        P = eigenvectors[:, :self.n_features]
        self.explained_variance = np.sum(eigenvalues[:self.n_features])

        return data@P


    def get_explained_variance(self) -> float:
        return self.explained_variance