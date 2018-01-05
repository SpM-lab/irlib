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
#for Lambda in [10.0, 100.0, 1000.0, 10000.0, 100000.0]:
for Lambda in [10.0, 100.0]:
    print("Loading basis functions...")
    b = irlib.loadtxt("basis_f-mp-Lambda"+str(Lambda)+".txt")
    print("Done!")

    plt.figure(1)
    print("dim = ", b.dim())

    x = 0.999
    mp.dps = 50

    tmp = b.ulx(b.dim()-1,x)
    tmp2 = b.ulx_mp(b.dim()-1,mpf(x))
    print(tmp, tmp2, tmp-tmp2)

    tmp = b.vly(b.dim()-1,x)
    tmp2 = b.vly_mp(b.dim()-1,mpf(x))
    print(tmp, tmp2, tmp-tmp2)

    for l in [0, b.dim()-1]:
        plt.plot(xvec, numpy.array([numpy.abs(b.ulx(l,x)) for x in xvec]), marker='', linestyle='-', color=colors[idx])

    plt.figure(2)
    for l in [0, b.dim()-1]:
        plt.plot(xvec, numpy.array([numpy.abs(b.vly(l,x)) for x in xvec]), marker='', linestyle='-', color=colors[idx])

    idx += 1

plt.figure(1)
plt.xlabel('$x$')
plt.ylabel('$u_l(x)$')
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig('ulx.pdf')

plt.figure(2)
plt.xlabel('$y$')
plt.ylabel('$v_l(y)$')
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig('vly.pdf')
