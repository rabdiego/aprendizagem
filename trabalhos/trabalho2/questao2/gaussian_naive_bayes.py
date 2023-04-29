import numpy as np

class GaussianNaiveBayes:
    def __init__(self) -> None:
        self.probabilities : np.ndarray = None
        self.means : np.ndarray = None
        self.variances : np.ndarray = None
        self.classes : np.ndarray = None
    
    def fit(self, X_train : np.ndarray, y_train : np.ndarray) -> None:
        self.probabilities = list()
        self.means = list()
        self.variances = list()
        self.classes = np.unique(y_train, axis=0)
        
        for unique_class in self.classes:
            indexes = np.argwhere((y_train == unique_class).all(1))
            X_train_class = X_train[indexes]
            y_train_class = y_train[indexes]
            self.probabilities.append(y_train_class.shape[0]/y_train.shape[0])
            self.means.append(np.mean(X_train_class, axis=0))
            self.variances.append(np.var(X_train_class, axis=0))
        
        self.probabilities = np.array(self.probabilities)
        self.means = np.array(self.means)
        self.variances = np.array(self.variances)
    
    
    def pred(self, X_test : np.ndarray) -> np.ndarray:
        predicted = np.zeros((self.classes.shape[0], X_test.shape[0]))
        for i in range(len(self.classes)):
            predicted[i] = np.log(self.probabilities[i]) - (0.5)*(np.log(2*np.pi*self.variances[i])).sum(axis=1) - (0.5)*(np.power((X_test - self.means[i]), 2)/self.variances[i]).sum(axis=1)
        return np.argmax(predicted, axis=0)
    
