from __future__ import division
import numpy as np

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

def mul(list):
    return reduce(lambda x, y: x*y, list)

def dirichlet(alpha,bins):
    """
    return a dirichlet distribution using parameters 'alpha' and discretized into 'bins' bins
    """
    B = sp.special.gamma(sum(alpha))/mul(sp.special.gamma(alpha))

    # TODO: implement this
    pass

if __name__ == "__main__":
    print generate(3,3)