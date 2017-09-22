from __future__ import print_function

import numpy
import irlib.basis
import scipy.integrate as integrate
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)

N = 1000
xvec = numpy.linspace(-1, 1, N)

## Construct basis
idx = 0
markers = ['o', 's', 'x', '+']
for Lambda in [1,10,100]:
    max_dim = 20
    b = irlib.basis.basis_f(Lambda, max_dim)

    plt.figure(1)
    for l in range(1,2):
        plt.plot(xvec, numpy.array([b.ulx(l,x) for x in xvec]))

    plt.figure(2)
    for l in range(1,2):
        plt.plot(xvec, numpy.array([b.vly(l,x) for x in xvec]))

plt.figure(1)
plt.xlabel('$x$')
plt.ylabel('$u_l(x)$')
plt.legend()
plt.tight_layout()
plt.savefig('ulx.pdf')

plt.figure(2)
plt.xlabel('$y$')
plt.ylabel('$v_l(y)$')
plt.legend()
plt.tight_layout()
plt.savefig('vly.pdf')
