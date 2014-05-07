import numpy as np
import scipy as sp
from scipy.stats import beta as beta
from model import Model

class BetaModel(Model):
    def __init__(self,i,b,w,bins=10):
        super(BetaModel,self).__init__(i,b,w,bins)
        assert self.num_candidates == 2

    #####################################################
    # Use beta distributions for two dimensional models #
    #####################################################
    def pInitial(self):
        cdf = beta(self.i[0],self.i[1]).cdf(self.quantiles)
        return np.diff(cdf)
    
    def pTransition(self,z,x):
        alpha = self.w[0]*z + sum([x[i]*self.w for i, w in enumerate(self.w[1:])])
        cdf = beta(alpha[0], alpha[1]).cdf(self.quantiles)
        return np.diff(cdf)

    def pEmission(self,z):
        # TODO: each poll should be chosen according to a multinomial distribution rather than a dirichlet distribution
        x = []

        for i in range(self.num_polls):
            alpha = self.b[i]*z
            cdf = beta(alpha[0], alpha[1]).cdf(self.quantiles)
            x.append(np.diff(cdf)/self.num_polls)

        x = np.array(x)
        return x

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Genrate a test case
    # Define size of network
    T = 100 # number of timebinss
    M = 5   # number of polls

    # Randomly model parameters based on priors
    I = 2*np.ones(2)
    B = 10*np.random.randn(M,2)
    B = np.exp(B)
    W = np.concatenate((.5*np.random.randn(1) + 7,np.random.randn(M)))
    W = np.exp(W)

    model = BetaModel(i=I,b=B,w=W)
    
    ## Generate Test Data
    # Z, X = model.generate(T)
    # plt.plot(range(T),Z)
    # plt.show()