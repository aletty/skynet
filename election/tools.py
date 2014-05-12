from __future__ import division
import numpy as np
import scipy as sp
from scipy.special import beta
from operator import mul
from functools import wraps

def generate(n,total):
    if n == 1:
        return [[total]]

    res = []
    for i in xrange(total+1):
        temp = [[i] + a for a in generate(n-1,total-i)]
        res.extend(temp)

    return res

def generate_states(n,bins):
    return map(lambda x: [v/bins for v in x], generate(n,bins))

def beta_pdf(a,b,x):
    return (x**a)*((1-x)**b)/sp.special.beta(a,b)

def dirichlet(alpha,bins):
    """
    return a dirichlet distribution using parameters 'alpha' and discretized into 'bins' bins
    """
    B = sp.special.gamma(sum(alpha))/mul(sp.special.gamma(alpha))

    # TODO: implement this
    pass

def sanitize(v):
    return np.array(map(lambda x: 1e-300 if x==0 else x,v))

if __name__ == "__main__":
    print generate(3,3)