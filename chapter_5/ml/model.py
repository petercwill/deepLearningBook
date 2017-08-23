import abc
import numpy as np

class Model(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def fit(self, X, y):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass


class LSQ(Model):

    def __init__(self, X, theta=None):
        n_rows, n_cols = X.shape
        self.num_features = n_cols
        self.theta = theta

    @property
    def theta(self):
        return self.__theta

    @theta.setter
    def theta(self, theta):
        if (theta is not None) and len(theta) != self.num_features:
            raise ValueError("length of parameters must match \\"
                "number of input cols")
        elif theta is not None:
            self.__theta = theta.reshape(theta.size, 1)
        else:
            self.__theta = np.zeros((self.num_features,1))


    def predict(self, X):
        return np.matmul(X, self.theta)

    def fit(self, X, y):
        pass
