import scipy.optimize
import numpy as np
from ml import model
from ml import costFunction
import abc

class Optimizer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def run():
        pass


class GD(Optimizer):
    '''
    vanilla gradient descent class with armijo backtracking line
    search.

    Given a cost function, model, and data set, will return the
    minimizing parameters of the model for that data set.
    '''

    def __init__(self, costFunction, model, X, y):
        '''
        costFunction: a CostFunction class instance from costFunction.py
        model: model class instance from Model.py
        X: dataset cofactors
        y: dataset labels
        '''

        self.costFunction = costFunction
        self.model = model
        self.X = X
        self.y = y

    def costFunctionCaller(self, theta):
        self.model.theta = theta
        return self.costFunction.eval(
            self.model.predict(self.X),
            self.y,
            theta
            )

    def run(
        self,
        theta0=None,
        grad_tol=10**-4,
        diff_tol=10**-10,
        max_steps=10**4,
        alpha0=1,
        c=.5,
        tau=.5
    ):

        '''
        Runs gradient descent procedure.
        theta0: initial values for model parameters
        grad_tol: stopping condition on the L2 norm of the
            approximate gradient of the cost function.
        diff_tol: stopping condition on the difference between
            succesive model parameter vectors.
        max_steps: number of gradient descent iterations to run.
        alpha: stepsize for gradient descent.
        c: control parameter for backtracking line search.
            scales expected descrease in cost per gradient
            descent step.
        tau: control parameter for backtracking line search.
            fraction by which step size is reduced if expected
            decrease in cost is not observed.

        Returns:
        theta: minimizer for cost function
        grad_norm: norm of the cost function gradient at termination
        step_count: number of steps taken
        theta_diff_norm: norm of difference between succesive
            theta vectors visted before termination.
        '''

        step_count = 0
        if(theta0):
            theta = theta0
        else:
            theta = self.model.theta

        grad_norm = np.inf
        theta_diff_norm = np.inf
        while (
            grad_norm > grad_tol and
            step_count < max_steps and
            theta_diff_norm > diff_tol
        ):

            old_cost = new_cost = self.costFunctionCaller(theta)


            grad = self.approx_grad(theta)
            grad_norm = np.linalg.norm(grad)

            p = (-grad / grad_norm).reshape(grad.size,1)
            t = c*grad_norm

            alpha = alpha0

            while (old_cost - new_cost) < (alpha*t):
                new_theta = theta + alpha*p
                new_cost = self.costFunctionCaller(new_theta)
                alpha *= tau

            step_count += 1
            theta_diff_norm = np.linalg.norm(theta - new_theta)
            theta = new_theta

        print(grad_norm, step_count, theta_diff_norm)
        return(theta, grad_norm, step_count, theta_diff_norm)

    def approx_grad(self, theta, epsilon=10**-10):
        '''
        Finite Difference approximation of the gradient of the
        cost function using forward euler
        '''
        theta = theta.reshape(theta.size)
        return(scipy.optimize.approx_fprime(
            theta,
            self.costFunctionCaller,
            epsilon))



class SGD(GD):

    def run(
        self,
        n_batches=20,
        use_weighted=True,
        theta0=None,
        grad_tol=10**-4,
        diff_tol=10**-5,
        max_steps=10**4,
        alpha0=1,
        c=.5,
        tau=.5
    ):

        '''
        Runs mini-batched stochastic gradient descent procedure.
        n_batches: number of batches to run SGD with
        theta0: initial values for model parameters
        grad_tol: stopping condition on the L2 norm of the
            approximate gradient of the cost function.
        diff_tol: stopping condition on the difference between
            succesive model parameter vectors.
        max_steps: number of gradient descent iterations to run.
        alpha: stepsize for gradient descent.
        c: control parameter for backtracking line search.
            scales expected descrease in cost per gradient
            descent step.
        tau: control parameter for backtracking line search.
            fraction by which step size is reduced if expected
            decrease in cost is not observed.

        Returns:
        theta: minimizer for cost function
        grad_norm: norm of the cost function gradient at termination
        step_count: number of steps taken
        theta_diff_norm: norm of difference between succesive
            theta vectors visted before termination.
        '''

        step_count = 0
        if(theta0):
            theta = theta0
        else:
            theta = self.model.theta

        grad_norm = np.inf
        theta_diff_norm = np.inf
        ncols = self.X.shape[1]
        data = np.hstack((self.X, self.y))
        weighted_coffs = theta
        while (
            grad_norm > grad_tol and
            step_count < max_steps and
            theta_diff_norm > diff_tol
        ):


            np.random.shuffle(data)
            minibatches = [
                np.hsplit(b, [ncols])
                for b in np.array_split(data, n_batches)
            ]

            for (mini_X, mini_Y) in minibatches:
                self.X = mini_X
                self.y = mini_Y

                old_cost = new_cost = self.costFunctionCaller(theta)


                grad = self.approx_grad(theta)
                grad_norm = np.linalg.norm(grad)

                p = (-grad / grad_norm).reshape(grad.size,1)
                t = c*grad_norm

                alpha = alpha0

                while (old_cost - new_cost) < (alpha*t):
                    new_theta = theta + alpha*p
                    new_cost = self.costFunctionCaller(new_theta)
                    alpha *= tau

                step_count += 1
                theta_diff_norm = np.linalg.norm(theta - new_theta)
                theta = new_theta

                weight_count = 0
                if step_count > .2*max_steps:
                    weighted_coffs = (weighted_coffs*weight_count + theta) / (weight_count+1)
                    weight_count+=1

        if use_weighted:
            return(weighted_coffs, grad_norm, step_count, theta_diff_norm)
        else:
            return(theta, grad_norm, step_count, theta_diff_norm)
