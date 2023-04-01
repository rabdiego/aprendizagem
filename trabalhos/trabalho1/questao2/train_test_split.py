import numpy as np
import random

def train_test_split(X : np.ndarray, y : np.ndarray, test_size : float = 0.2) -> tuple:
    if len(X) != len(y):
        return -1
    else:
        n : int = int(len(X) * test_size)
        rng : list = range(len(X))
        
        test_index : list = random.sample(rng, n)
        train_index : list = [i for i in rng if i not in test_index]
        
        X_train = np.array([X[i] for i in train_index])
        X_test = np.array([X[i] for i in test_index])
        y_train = np.array([y[i] for i in train_index])
        y_test = np.array([y[i] for i in test_index])
        
        return X_train, X_test, y_train, y_test

