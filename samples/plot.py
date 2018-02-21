from __future__ import print_function

import numpy
import irlib
import scipy.integrate as integrate
import matplotlib.pyplot as plt

from mpmath import *

plt.rc('text', usetex=True)

N = 1000
xvec = numpy.linspace(-1, 1, N)

## Construct basis
idx = 0
markers = ['o', 's', 'x', '+', 'v']
ls = ['-', '--', ':']
colors = ['r', 'b', 'g', 'k']
for Lambda in [1000.0]:
    #print("Loading basis functions...")
    b = irlib.loadtxt("basis_b-mp-Lambda"+str(Lambda)+".txt")

    plt.figure(1)
    print("dim = ", b.dim())

    for l in [0, 2, 4]:
        plt.plot(xvec, numpy.array([b.ulx(l,x) for x in xvec]), marker='', linestyle='-', color=colors[idx])

    plt.figure(2)
    for l in [0, 2, 4]:
        plt.plot(xvec, numpy.array([b.vly(l,x) for x in xvec]), marker='', linestyle='-', color=colors[idx], label='l='+str(l))

    idx += 1

plt.figure(1)
plt.xlabel('$x$')
plt.ylabel('$u_l(x)$')
#plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig('ulx.pdf')

plt.figure(2)
plt.xlabel('$y$')
plt.ylabel('$v_l^\mathrm{B}(y)$')
#plt.yscale("log")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('vly.pdf')
