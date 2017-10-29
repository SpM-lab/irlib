from __future__ import print_function

import numpy
import irlib
import scipy.integrate as integrate
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)

N = 1000
xvec = numpy.linspace(-1, 1, N)

max_dim = 100
cutoff = 1e-8

## Construct basis
idx = 0
markers = ['o', 's', 'x', '+']
for Lambda in [10000]:
    print("Computing basis functions... It may take some time")
    b = irlib.basis_f_dp(Lambda, max_dim, cutoff)
    print("Done!")

    plt.figure(1)
    for l in [0, b.dim()-1]:
        plt.plot(xvec, numpy.array([numpy.abs(b.ulx(l,x)) for x in xvec]))

    plt.figure(2)
    for l in [0, b.dim()-1]:
        plt.plot(xvec, numpy.array([numpy.abs(b.vly(l,x)) for x in xvec]))

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
