import numpy as np

class MinMax:
    def __init__(self) -> None:
        self.min : np.ndarray = None
        self.max : np.ndarray = None
    
    
    def fit_transform(self, data : np.ndarray) -> np.ndarray:
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        
        return (data - self.min)/(self.max - self.min)
    
    
    def transform(self, data : np.ndarray) -> np.ndarray:
        return (data - self.min)/(self.max - self.min)
     
    
    def untransform(self, data : np.ndarray) -> np.ndarray:
        return data*(self.max - self.min) + self.min

