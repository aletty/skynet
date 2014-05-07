from __future__ import division
import numpy as np
import scipy as sp

from tools import *

class Model(object):
    def __init__(self, i, b, w, bins=10):
        # set the model parameters
        self.i = i
        self.b = b
        self.w = w

        # extract additional information
        self.num_polls = b.shape[0]
        self.num_candidates = b.shape[1]

        # set computational parameters
        self.bins = bins
        self.quantiles = np.linspace(0,1,bins+1)

    ####################
    # Generative Model #
    ####################

    def sampleInitial(self):
        return np.random.dirichlet(self.i)
    
    def sampleTransition(self,z,x):
        alpha = self.w[0]*z + sum([x[i]*w for i, w in enumerate(self.w[1:])])
        return np.random.dirichlet(alpha)        
    
    def sampleEmision(self,z):
        x = []

        for i in range(self.num_polls):
            alpha = self.b[i]*z
            x.append(np.random.dirichlet(alpha))        

        x = np.array(x)
        return x

    def generate(self,T):
    	assert T >= 1
        try:
        	Z = [self.sampleInitial()]
        	X = [self.sampleEmision(Z[0])]

        	for t in xrange(1,T):
        		Z.append(self.sampleTransition(Z[t-1], X[t-1]))
        		X.append(self.sampleEmision(Z[t]))

        	return np.array(Z), np.array(X)
        except ZeroDivisionError:
            return self.generate(T)

    ###################################
    # Given probability distributions #
    ###################################

    def pInitial(self):
        return dirichlet(self.i)
    
    def pTransition(self,z,x):
        alpha = self.w[0]*z + sum([x[i]*self.w for i, w in enumerate(self.w[1:])])
        return dirichlet(alpha, self.bins)

    def pEmission(self,z):
        # TODO: each poll should be chosen according to a multinomial distribution rather than a dirichlet distribution
        x = []

        for i in range(self.num_polls):
            alpha = self.b[i]*z
            x.append(dirichlet(alpha,self.bins)/self.num_polls)

        x = np.array(x)
        return x

    #####################################
    # Derived Probability Distributions #
    #####################################

    def logLiklihood(self, X):
        pass

    def pState(self,X,t):
        pass

    def pStatePair(self,X,t):
        pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Define size of network
    T = 100 # number of timebinss
    N = 3   # number of candidates
    M = 5   # number of polls

    # Randomly model parameters based on priors
    I = 2*np.ones(N)
    B = 10*np.random.randn(M,N)
    B = np.exp(B)
    W = np.concatenate((.5*np.random.randn(1) + 7,np.random.randn(M)))
    W = np.exp(W)

    model = Model(i=I,b=B,w=W)
    Z, X = model.generate(T)

    plt.plot(range(T),Z)
    plt.show()