import numpy as np
from numpy.linalg import pinv, inv

class LinearRegression:
    def __init__(self):
        self.theta = None

    #Taken from slide 10, w = (X^T * X)^-1 * X^T * y
    def fit(self, X, y):
        self.theta = inv(X.T @ X) @ X.T @ y

    #Taken from slide 3, g(x) = xw + b
    def predict(self, X):
        return X @ self.theta

    #Taken from slide 5, MSE = 1/n * sum(y_i - y_i^)^2
    #RSME = sqrt(MSE)
    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    #Taken from slide 6, SMAPE = 1/N * sum((y_i - y_i^)/(y_i + y_i^))
    @staticmethod
    def smape(y_true, y_pred):
        return 100 / len(y_true) * np.sum(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))