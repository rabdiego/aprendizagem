import numpy as np

class LinearRegression:
    def __init__(self) -> None:
        self.w : np.ndarray = None
        self.log : list = list()
        
        
    def pred(self, X_test : np.ndarray) -> np.ndarray:
        y_pred : np.ndarray = X_test@self.w
        return y_pred

    
    def get_mse(self, X_test : np.ndarray, y_test : np.ndarray) -> np.ndarray:
        y_pred : np.ndarray = self.pred(X_test)
        mse = ((((y_test - y_pred)**2).sum(axis=0))/(X_test.shape[0]))
        return mse
    
    
    def fit_ols(self, X_train : np.ndarray, y_train : np.ndarray) -> None:
        self.w = np.linalg.solve(X_train.T@X_train, X_train.T@y_train)


    def fit_gd(self, X_train : np.ndarray, y_train : np.ndarray, lr : float = 0.1, num_epochs : int = 100) -> None:
        self.w = np.ones((X_train.shape[1], 1))
        for i in range(num_epochs):
            y_pred = self.w.T@X_train.T
            y_pred = y_pred.reshape(-1, 1)
            error = y_train - y_pred
    
            self.w += (lr/X_train.shape[0])*(((error.T@X_train).sum(axis=0)).reshape(-1, 1))
            
            self.log.append(self.get_mse(X_train, y_train).item())
            

    
    def fit_sgd(self, X_train : np.ndarray, y_train : np.ndarray, lr : float = 0.01, num_epochs : int = 100) -> None:
        self.w = np.ones((X_train.shape[1], 1))
        for i in range(num_epochs):
            for j in range(X_train.shape[0]):
                y_pred = self.w.T@X_train[j]
                
                error = y_train[j] - y_pred
                
                self.w += (lr*(error@(X_train[j].reshape(1, -1)))).reshape(-1, 1)
                
                self.log.append(self.get_mse(X_train, y_train).item())

    
    def get_params(self) -> np.ndarray:
        return self.w
    
    
    def get_log(self) -> list:
        return self.log

