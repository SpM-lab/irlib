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
for Lambda in [10000.0]:
    b10 = irlib.loadtxt("np10/basis_f-mp-Lambda"+str(Lambda)+".txt")
    b8 = irlib.loadtxt("np8/basis_f-mp-Lambda"+str(Lambda)+".txt")

    print("dim = ", b10.dim(), b8.dim())

    plt.figure(4)
    #nvec = numpy.arange(100)
    nvec = numpy.array([0,10,10**2,400, 600, 800, 10**3, 2000, 4000, 6000, 10**4,10**5,10**6])
    Tnl10 = b10.compute_Tnl(nvec)
    Tnl8 = b8.compute_Tnl(nvec)
    for l in [b10.dim()-1]:
        plt.plot(nvec, numpy.abs(Tnl10[:,l]), marker='x', linestyle='-', color='r', label='l='+str(l))
        plt.plot(nvec, numpy.abs(Tnl8[:,l]), marker='', linestyle='--', color='b', label='l='+str(l))
        plt.plot(nvec, numpy.abs(Tnl8[:,l]-Tnl10[:,l]), marker='x', linestyle='-', color='r', label='abs error')
        plt.plot(nvec, numpy.abs(Tnl8[:,l]-Tnl10[:,l])/numpy.abs(Tnl10[:,l]), marker='o', linestyle='-', color='r', label='relative error')
        #plt.plot(nvec, Tnl[:,l].imag, marker='x', linestyle='-', color='r', label='l='+str(l))
        #plt.plot(nvec, Tnl20[:,l].imag, marker='', linestyle='--', color='b', label='l='+str(l))

    idx += 1

plt.figure(4)
plt.xlabel('$n$')
plt.ylabel('$T_{nl}$')
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.savefig('Tnl.pdf')
