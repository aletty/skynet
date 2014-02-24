import numpy as np
import scipy as sp

def sample_surveys(O,B):
    S = []

    for i in range(B.shape[0]):
        S.append(np.random.dirichlet(np.exp(B[i,:]))*O)
        S[i] = S[i]/np.sum(S[i])

    return S

def sample_opinion(O,S,W):
    avg = W[0]*O + sum([S[i]*w for i, w in enumerate(W[1:])])
    return np.random.dirichlet(avg)

if __name__ == "__main__":
    T = 100 # number of timesteps
    N = 3   # number of candidates
    M = 5   # number of polls

    # Choose biases and weights
    B = 100*np.random.randn(M,N)
    W = np.random.dirichlet(np.ones(M+1))

    # Generate initial public opinion
    O = []
    O.append(np.random.dirichlet(np.ones(N)))

    # Initialize the survey list
    S = []

    for t in range(T):
        S.append(sample_surveys(O[t],B))
        O.append(sample_opinion(O[t],S[t],W))

    print O