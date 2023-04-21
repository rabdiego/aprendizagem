import numpy as np

def sigmoid(z):
    return np.power((1 + np.exp(-z)), -1)

class LogisticRegression:
    def __init__(self) -> None:
        self.w : np.ndarray = None
        self.log : list = list()
        
        
    def pred(self, X_test : np.ndarray) -> np.ndarray:
        y_pred : np.ndarray = X_test@self.w
        return sigmoid(y_pred) - 10**(-15)

    
    def get_bce_loss(self, X_test : np.ndarray, y_test : np.ndarray) -> np.ndarray:
        y_pred : np.ndarray = self.pred(X_test)
        bce = -(((y_test*np.log2(y_pred) + (1 - y_test)*np.log2(1 - y_pred)).sum(axis=0))/(X_test.shape[0]))
        return bce
    
    
    def fit(self, X_train : np.ndarray, y_train : np.ndarray, lr : float = 0.01, num_epochs : int = 100) -> None:
        self.w = np.ones((X_train.shape[1], 1))
        for i in range(num_epochs):
            for j in range(X_train.shape[0]):
                y_pred = sigmoid(self.w.T@X_train[j])
                
                error = y_train[j] - y_pred
                
                self.w += (lr*(error@(X_train[j].reshape(1, -1)))).reshape(-1, 1)
                
                self.log.append(self.get_bce_loss(X_train, y_train).item())

    
    def get_params(self) -> np.ndarray:
        return self.w
    
    
    def get_log(self) -> list:
        return self.log

