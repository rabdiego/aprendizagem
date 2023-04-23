import numpy as np

def gaussian_probability(X_test, mean, variance):
    return np.exp(-(((np.power((X_test - mean), 2))/(2*variance))))/np.sqrt(2*np.pi*variance)

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
        return_array = list()
        for i in range(X_test.shape[0]):
            Px_y = gaussian_probability(X_test[i], self.means, self.variances)
            Px_y = np.array(Px_y)
            Px_y = np.prod(Px_y[:, 0], axis=1)
            Py_x = Px_y * self.probabilities
            return_array.append(np.argmax(Py_x))
        return_array = np.array(return_array)
        return return_array.reshape(-1, 1)
    
