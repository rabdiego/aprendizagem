import numpy as np

class PolynomialRegression:
    def __init__(self) -> None:
        self.w : np.ndarray = None
        self.degree : int = None
    
    
    def convert_degree(self, data : np.ndarray, degree : int) -> np.ndarray:
        original_data : np.ndarray = data
        
        for i in range(2, degree + 1):
            data = np.hstack((data, np.power(original_data, i)[:, 1:]))
        
        return data
    
    
    def pred(self, X_test : np.ndarray) -> np.ndarray:
        X_test = self.convert_degree(X_test, self.degree)
        y_pred : np.ndarray = X_test@self.w
        return y_pred

    
    def get_rmse(self, X_test : np.ndarray, y_test : np.ndarray) -> np.ndarray:
        X_test = self.convert_degree(X_test, self.degree)
        y_pred : np.ndarray = X_test@self.w
        mse = ((((y_test - y_pred)**2).sum(axis=0))/(X_test.shape[0]))
        return mse**(0.5)
    
    
    def fit(self, X_train : np.ndarray, y_train : np.ndarray, degree : int, regularization_term : float = 0.0) -> None:
        self.degree = degree
        X_train = self.convert_degree(X_train, degree)
        self.w = np.linalg.solve(X_train.T@X_train + regularization_term*np.eye(len(X_train.T@X_train)), X_train.T@y_train)


    def get_params(self):
        return self.w

