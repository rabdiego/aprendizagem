import numpy as np

def sigmoid(z):
    return np.power((1 + np.exp(-z)), -1)

class SoftmaxRegression:
    def __init__(self) -> None:
        self.w : np.ndarray = None
        self.log : list = list()
        
        
    def pred(self, X_test : np.ndarray) -> np.ndarray:
        r = X_test@self.w.T
        return (np.exp(r)/np.exp(r).sum())
    
    
    def get_mcce_loss(self, X_test : np.ndarray, y_test : np.ndarray) -> np.ndarray:
        y_pred : np.ndarray = self.pred(X_test)
        mcce = -(np.sum(np.sum(y_test*np.log2(y_pred), axis=1)))/(X_test.shape[0])
        return mcce
    
    
    def fit(self, X_train : np.ndarray, y_train : np.ndarray, lr : float = 0.01, num_epochs : int = 100) -> None:
        classes = np.unique(y_train, axis=0)
        self.w = np.zeros((len(classes), X_train.shape[1]))
        self.log = list()
        for i in range(num_epochs):
            for j in range(X_train.shape[0]):
                y_pred = self.pred(np.array([X_train[j]]))
                
                error = y_train[j] - y_pred[0]
                self.w += (lr*(error.reshape(-1, 1))@(X_train[j].reshape(1, -1)))
                
                self.log.append(self.get_mcce_loss(X_train, y_train).item())
        

    
    def get_params(self) -> np.ndarray:
        return self.w
    
    
    def get_log(self) -> list:
        return self.log

