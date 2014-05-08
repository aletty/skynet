import numpy as np
import scipy as sp
from scipy.stats import beta as beta
from model import Model

class BetaModel(Model):
    def __init__(self,i,b,w,bins=10):
        super(BetaModel,self).__init__(i,b,w,bins)
        assert self.num_candidates == 2
        assert self.num_polls == 2

    #####################################################
    # Use beta distributions for two dimensional models #
    #####################################################
    def pInitial(self):
        """
        Returns a distribution over initial states (z)
        The distribution is discretized into the number of bins specified by self.bins
        """
        cdf = beta(self.i[0],self.i[1]).cdf(self.quantiles)
        return np.diff(cdf)
    
    def pTransition(self,z,x):
        """
        Returns a distribution over transitions to z given the current state (z) and observation (x)
        The distribution is discretized into the number of bins specified by self.bins
        """
        alpha = self.w[0]*z + sum([x[i]*w for i, w in enumerate(self.w[1:])])
        cdf = beta(alpha[0], alpha[1]).cdf(self.quantiles)
        return np.diff(cdf)

    def pEmission(self,z,x):
        """
        Returns a number proportional to the probability of x given z
        """
        # TODO: each poll should be chosen according to a multinomial distribution rather than a dirichlet distribution
        alpha = self.b[i]*z
        return beta_pdf(alpha[0], alpha[1], x[0])

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Genrate a test case
    # Define size of network
    T = 100 # number of timebinss
    M = 2   # number of polls

    # Randomly model parameters based on priors
    I = 2*np.ones(2)
    B = 10*np.random.randn(M,2)
    B = np.exp(B)
    W = np.concatenate((.5*np.random.randn(1) + 7,np.random.randn(M)))
    W = np.exp(W)

    model = BetaModel(i=I,b=B,w=W)

    print model.states
    ## Generate Test Data
    # Z, X = model.generate(T)
    # plt.plot(range(T),Z)
    # plt.show()