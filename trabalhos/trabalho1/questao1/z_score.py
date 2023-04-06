import numpy as np

class ZScore:
    def __init__(self) -> None:
        self.mean : np.ndarray = None
        self.sd : np.ndarray = None
    
    
    def fit(self, X : np.ndarray) -> None:
        self.mean = np.mean(X, axis=0)
        self.sd = np.std(X, axis=0, ddof=1)
    
    
    def fit_transform(self, X : np.ndarray) -> np.ndarray:
        self.mean = np.mean(X, axis=0)
        self.sd = np.std(X, axis=0, ddof=1)
        return (X - self.mean)/self.sd
    
    
    def transform(self, X : np.ndarray) -> np.ndarray:
        return (X - self.mean)/self.sd


    def untransform(self, X : np.ndarray) -> np.ndarray:
        return self.sd*X + self.mean

