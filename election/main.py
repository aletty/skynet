import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def sample_surveys(O,B):
    S = []

    for i in range(B.shape[0]):
        S.append(np.random.dirichlet(B[i,:]*O))

    return S

def sample_opinion(O,S,W):
    avg = W[0]*O + sum([S[i]*w for i, w in enumerate(W[1:])])
    return np.random.dirichlet(avg)

if __name__ == "__main__":
    # Define size of network
    T = 100 # number of timesteps
    N = 3   # number of candidates
    M = 5   # number of polls

    # Randomly select biases and weights based on priors
    B = 10*np.random.randn(M,N)
    B = np.exp(B)
    W = np.concatenate((.5*np.random.randn(1) + 7,np.random.randn(M)))
    W = np.exp(W)

    # Generate initial public opinion
    O = []
    O.append(np.random.dirichlet(2*np.ones(N)))

    # Initialize the survey list
    S = []

    for t in range(T):
        S.append(sample_surveys(O[t],B))
        O.append(sample_opinion(O[t],S[t],W))

    O = np.array(O)
    print O.shape
    print T


    plt.plot(range(T+1), O)
    plt.show()