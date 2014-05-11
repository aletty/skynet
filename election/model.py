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
        self.states = np.linspace(0,(bins-1)/bins,bins) + np.diff(self.quantiles)/2

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

    def pInitial(self,params=None):
        i = params['i'] if params else self.i
        return dirichlet(i)
    
    def pTransition(self,z,x,params=None):
        w = params['w'] if params else self.w
        alpha = w[0]*z + sum([x[i]*_w for i, _w in enumerate(w[1:])])
        return dirichlet(alpha, self.bins)

    def pEmission(self,z,x,params=None):
        b = params['b'] if params else self.b
        # TODO: each poll should be chosen according to a multinomial distribution rather than a dirichlet distribution
        # FIXME: implement this
        pass

    #####################################
    # Derived Probability Distributions #
    #####################################

    def logLiklihood(self, X, new_params):

        # likelihood of initial state
        l = np.log(self.pInitial())
        p = self.pState(X,0)
        initial = sum(p*l)

        # likelihood of transitions
        transition = 0
        for x in X[:-1]:
            for j in xrange(self.bins):
                l = np.log(self.pTransition(j,X[t-1]))
                p = self.pStatePair(j,X,t)
                transition += sum(p*l)

        # likelihood of observations
        emission = 0
        for t, x in enumerate(X):
            l = np.array([np.log(self.pEmission(z,X[t]))])
            p = self.pState(X,t)
            emission += sum(p*l)

        return initial + transition + emission

    def pState(self,X,t):

        def M(t):
            return np.matrix([self.pTransition(z,X[t-1]) for z in self.states]).T
            
        # forwards algorithm
        def alpha(t):
            if t==0:
                # TODO return stopping value
                return self.pInitial()*np.array([self.pEmission(z,X[0]) for z in self.states])
            evidence = np.array([self.pEmission(z,x[t]) for z in self.states])
            trans = np.array(M(t).T*alpha(t-1))
            return  evidence*trans 
        
        # backwards algorithm
        def beta(t):
            if t==(len(X)-1):
                return np.ones(self.bins)
            evidence = np.array([self.pEmission(z,X[t]) for z in self.states])
            return np.array(M(t)*np.matrix(evidence*beta(t+1)).T)

        res = alpha(t)*beta(t)
        return res/sum(res)

    def pStatePair(self,z,X,t):

        def M(t):
            # TODO implement this
            pass

        # forwards algorithm
        def alpha(t):
            if t==0:
                return # TODO return stopping value
            return # TODO return the appropriate value

        # backwards algorithm
        def beta(t):
            if t==(len(X)-1):
                return np.ones(self.bins)
            return # TODO return the apprpriate value

        res = alpha(t)*beta(t)
        return res/sum(res)

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