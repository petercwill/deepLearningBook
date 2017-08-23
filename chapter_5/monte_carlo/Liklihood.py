from __future__ import division
import fake_data
import math
import numpy as np
import abc
import random as rn



class Likelihood(object):

    def __init__(self, model, y):
        self.model = model
        self.y = y

    def target(theta, sigma=1):
        nrows, ncols = self.model.X.shape
        y_pred = self.model.predict()
        resids = self.y - y_pred
        llh = -nrows * math.log(sigma) - (nrows / 2) * math.log(2 * math.pi) - (np.norm(resids)**2 / (2*sigma**2))
        prior = prior(theta)

        return llh*prior


    def prior(theta):
        '''
        prior distribution for model parameters.
        Represent uninformative prior, with sigma
        restricted to positive values
        '''
        return 1


class Simulation(metaclass=abc.ABCMeta):


    class MetropolisHastings(Simulation):


        def __init__(self, likelihood, proposal):
            self.likelihood = likelihood
            self.proposal = proposal


        def run(self, theta0=None, maxSteps=10**7, batch_size = 10**5):

            if(theta0):
                theta = theta0
            else:
                theta = self.model.theta

            q1 = l.target(theta)
            accept_count = 0

            for i in maxSteps:
                theta_prime = self.proposal.sample(theta)
                q2 = l.target(theta_prime)
                alpha = max(1, q2 / q1)

                if rn.uniform(0, 1) <= accept_prop:
                    accept_count += 1
                    theta = theta_prime
                    q1 = q2









    class HastingsWithinGibbs(Simulation):



class ProposalDistribution(metaclass=abc.ABCMeta):

    def __init__(self, theta, r):
        self.theta = theta
        self.r = r

    @property
    def theta(self):
        return self.__theta

    @theta.setter
    def theta(self, theta):
        if self.theta.size != self.r.size:
            raise ValueError("length of parameters must equal length of stepsizes")
        else:
            self.__theta = theta.reshape(theta.size)

    @property
    def r(self):
        return self.__r

    @r.setter
    def r(self, r):
        if self.r.size != self.theta.size:
            raise ValueError("length of parameters must equal length of stepsizes")
        else:
            self.__r = r.reshape(r.size)

    def sample(self):



class IsotropicGauss(ProposalDistribution):

    def sample(self, theta):
        cov = np.diag(self.r)
        self.theta = theta
        return np.random.multivariate_normal(self.theta, cov=cov)
