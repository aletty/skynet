import numpy as np
import scipy as sp

def sample_surveys(O,B):
    # FIXME
    S = zeros(B.shape)
    for i in S.shape[1]
        S = np.random.dirichlet(np.exp(B[:,i]))*O
        return S/np.sum(S)

def sample_opinion(O,S,W):
    return np.random.dirichlet(W[0]*O + np.dot(S,W[1:]))

if __name__ == "__main__":
    T = 100 # number of timesteps
    N = 3   # number of candidates
    M = 5   # number of polls

    # Choose biases and weights
    B = np.random.randn(M,N)
    W = np.random.dirichlet(np.ones(M+1))

    # Generate initial public opinion
    O = []
    O.append(np.random.dirichlet(np.ones(N)))

    # Initialize the survey list
    S = []

    for t in range(T):
        S.append(sample_surveys(O[-1],B))
        O.append(sample_opinion(O[-1],S[-1],W))

    print O