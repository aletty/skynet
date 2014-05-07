from model import Model
import numpy as np

def logLiklihood(data, model):
    pass

def pStates(X, t):
    """P(Z_t, Z_{t-1} | X)"""

def pState(X, t):
    """P(Z_t | X)"""

if __name__ == "__main__":
    # Generate Data
    T, N, M = 100, 3, 5 # network size: time, candidates, polls
    I = 2*np.ones(N)    # model parameters
    B = np.exp(10*np.random.randn(M,N))
    W = np.exp(np.concatenate((.5*np.random.randn(1) + 7,np.random.randn(M))))

    m = Model(i=I, b=B, w=W)
    _, X = m.generate(T)

