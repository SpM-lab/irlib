from __future__ import print_function

import numpy
import irlib
import scipy.integrate as integrate
import matplotlib.pyplot as plt

from mpmath import *

plt.rc('text', usetex=True)

N = 1000
xvec = numpy.linspace(-1, -1+0.01, N)

## Construct basis
idx = 0
markers = ['o', 's', 'x', '+', 'v']
ls = ['-', '--', ':']
colors = ['r', 'b', 'g', 'k']

for Lambda in [1000.0]:
    #print("Loading basis functions...")
    b = irlib.loadtxt("np10/basis_f-mp-Lambda"+str(Lambda)+".txt")

    print("dim = ", b.dim())

    edges = numpy.array([b.section_edge(s) for s in range(b.num_sections()+1)])
    for s in range(1,b.num_sections()):
        print(edges[s])

    order = 3
    dim = b.dim()

    plt.figure(1)
    for l in [dim-1]:
        plt.plot(xvec+1, numpy.abs(numpy.array([b.ulx_derivative(l,x,order) for x in xvec])), marker='', linestyle='-', color=colors[idx])

    idx += 1

plt.figure(1)
plt.xlabel('$x$')
plt.ylabel('$u_l(x)$')
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig('ulx_deriv.pdf')
