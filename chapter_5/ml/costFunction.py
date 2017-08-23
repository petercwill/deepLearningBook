import abc
import numpy as np

class CostFunction(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def eval(self, y_pred, y, theta, **kwargs):
        pass


class MSE(CostFunction):

    @classmethod
    def eval(self, y_pred, y, theta, **kwargs):
        return (1/y_pred.size)*np.linalg.norm(y_pred - y)**2


class MSEL1(CostFunction):

    def __init__(self, lambda_reg=.5):
        self.lambda_reg = lambda_reg

    def eval(self, y_pred, y, theta, **kwargs):
        return (1/y_pred.size)*np.linalg.norm(y_pred - y)**2 + self.lambda_reg * np.linalg.norm(theta, ord=1)


class MSEL2(CostFunction):

    def __init__(self, lambda_reg=.5):
        self.lambda_reg = lambda_reg

    def eval(self, y_pred, y, theta, **kwargs):
        reg_lambda = kwargs.get("l",.5)
        return (1/y_pred.size)*np.linalg.norm(y_pred - y)**2 + self.lambda_reg*np.linalg.norm(theta, ord=2)


class LogLiklihood(CostFunction):

    def __init__(self, lambda_reg=.5):
        self.lambda_reg = lambda_reg

    def eval(self, y_pred, y, theta, **kwargs):
        reg_lambda = kwargs.get("l",.5)
        return (1/y_pred.size)*np.linalg.norm(y_pred - y)**2 + self.lambda_reg*np.linalg.norm(theta, ord=2)
