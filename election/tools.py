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

if __name__ == "__main__":
    print len(generate_states(5,10))
    # print generate_states(5,5)
    # print map(lambda x: [v/(5-1) for v in x], generate_states(3,5))