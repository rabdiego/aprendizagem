import numpy as np

class GaussianDiscriminantAnalysis:
    def __init__(self) -> None:
        self.probabilities : np.ndarray = None
        self.means : np.ndarray = None
        self.covariance : np.ndarray = None
        self.classes : np.ndarray = None
    
    
    def fit(self, X_train : np.ndarray, y_train : np.ndarray) -> None:
        self.probabilities = list()
        self.means = list()
        self.covariance = list()
        self.classes = np.unique(y_train, axis=0)
        
        for unique_class in self.classes:
            indexes = np.argwhere((y_train == unique_class).all(1))
            X_train_class = X_train[indexes]
            y_train_class = y_train[indexes]
            self.probabilities.append(y_train_class.shape[0]/y_train.shape[0])
            self.means.append(np.mean(X_train_class, axis=0))
            self.covariance.append(np.cov(X_train_class))
        
        self.probabilities = np.array(self.probabilities)
        self.means = np.array(self.means)
        self.covariance = np.array(self.covariance)