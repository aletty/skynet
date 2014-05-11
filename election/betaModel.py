import mynumpy as np
import scipy as sp
from scipy.stats import beta as Beta
from model import Model
from tools import *

class BetaModel(Model):
    def __init__(self,i,b,w,bins=10):
        super(BetaModel,self).__init__(i,b,w,bins)
        assert self.num_candidates == 2

    #####################################################
    # Use beta distributions for two dimensional models #
    #####################################################
    def pInitial(self):
        """
        Returns a distribution over initial states (z)
        The distribution is discretized into the number of bins specified by self.bins
        """
        cdf = Beta.cdf(self.quantiles,self.i[0],self.i[1])
        return np.diff(cdf)
    
    def pTransition(self,z,x):
        """
        Returns a distribution over transitions to z given the current state (z) and observation (x)
        The distribution is discretized into the number of bins specified by self.bins
        """
        alpha = self.w[0]*z + sum([x[i]*w for i, w in enumerate(self.w[1:])])
        cdf = Beta.cdf(self.quantiles, alpha[0], alpha[1])
        return np.diff(cdf)

    def pEmission(self,z,x):
        """
        Returns a number proportional to the probability of x given z
        """
        # TODO: each poll should be chosen according to a multinomial distribution rather than a dirichlet distribution
        res = 1
        for i in xrange(self.num_polls):
            alpha = self.b[i]*z
            print x
            res *= Beta.pdf(x[i,0], alpha[0], alpha[1])
        return res

if __name__ == "__main__":
    # import matplotlib.pyplot as plt

    # Generate a test case
    # Define size of network
    T = 100 # number of timebinss
    M = 3   # number of polls

    # Randomly model parameters based on priors
    I = 2*np.ones(2)
    B = 10*np.random.randn(M,2)
    B = np.exp(B)
    W = np.concatenate((.5*np.random.randn(1) + 7,np.random.randn(M)))
    W = np.exp(W)

    model = BetaModel(i=I,b=B,w=W)

    ## Generate Test Data
    Z, X = model.generate(T)
    M = model.pState(X,0)
    print np.sum(M,1)
    # plt.plot(range(T),Z)
    # plt.show()