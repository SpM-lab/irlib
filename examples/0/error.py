from __future__ import print_function

import numpy
import irlib
import scipy.integrate as integrate
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)

N = 1000
xvec = numpy.linspace(-1, 1, N)

max_dim = 20
cutoff = 1e-8

markers = ['o', 'x']

## Construct basis
idx = 0
markers = ['o', 's', 'x', '+']
for Lambda in [100, 10000]:
    print("Computing basis functions... It may take some time")
    b_dp = irlib.basis_f_dp(Lambda, max_dim, cutoff)
    b = irlib.basis_f(Lambda, max_dim, cutoff)
    print("Done!")

    plt.figure(1)
    for l in [0, b.dim()-1]:
        plt.plot(xvec, numpy.array([numpy.abs(b.ulx(l,x)-b_dp.ulx(l,x)) for x in xvec]))

    plt.figure(2)
    for l in [0, b.dim()-1]:
        plt.plot(xvec, numpy.array([numpy.abs(b.vly(l,x)-b_dp.vly(l,x)) for x in xvec]))

    n_vec = numpy.array([2**i for i in range(30)])
    Tnl = b.compute_Tnl(n_vec)
    Tnl_dp = b_dp.compute_Tnl(n_vec)
    ip = 0
    for l in [0, b.dim()-1]:
        plt.figure(3)
        plt.plot(n_vec, numpy.abs(Tnl[:,l]), linestyle='-', marker=markers[ip])

        plt.figure(4)
        plt.plot(n_vec, numpy.abs((Tnl[:,l]-Tnl_dp[:,l])/Tnl[:,l]), linestyle='-', marker=markers[ip])

        ip += 1

plt.figure(1)
plt.xlabel('$x$')
plt.ylabel('Error in $u_l(x)$')
plt.legend()
plt.yscale("log")
plt.tight_layout()
plt.savefig('error_ulx.pdf')

plt.figure(2)
plt.xlabel('$y$')
plt.ylabel('Error in $v_l(y)$')
plt.legend()
plt.yscale("log")
plt.tight_layout()
plt.savefig('error_vly.pdf')

plt.figure(3)
plt.xlabel('$n$')
plt.ylabel('$T_{nl}$')
plt.legend()
plt.yscale("log")
plt.tight_layout()
plt.savefig('Tnl.pdf')

plt.figure(4)
plt.xlabel('$n$')
plt.ylabel('Relative error in $T_{nl}$')
plt.legend()
plt.yscale("log")
plt.tight_layout()
plt.savefig('error_Tnl.pdf')
