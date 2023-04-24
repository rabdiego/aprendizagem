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
        self.classes = np.unique(y_train, axis=0)
        
        self.covariance = np.cov(X_train.T)
        for unique_class in self.classes:
            indexes = np.argwhere((y_train == unique_class).all(1))
            X_train_class = X_train[indexes]
            y_train_class = y_train[indexes]
            self.probabilities.append(y_train_class.shape[0]/y_train.shape[0])
            self.means.append(np.mean(X_train_class, axis=0))
            
        
        self.probabilities = np.array(self.probabilities)
        self.means = np.array(self.means)
    
    
    def pred(self, X_test : np.ndarray) -> np.ndarray:
        predicted = list()
        for i in self.classes:
            a = int(i[0])
            log = np.log(self.probabilities[a])
            norm = np.linalg.norm(self.covariance[a])
            lognorm = (0.5)*np.log(norm)
            diff = (X_test - self.means[a])
            solve = np.linalg.solve(self.covariance, diff.T)
            r = diff @ solve
            r *= 0.5
            print((log - lognorm - r).shape)
            predicted.append(np.argmax(log - lognorm - r, axis=1))
        predicted = np.array(predicted)
        return predicted
