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
    b_np8 = irlib.loadtxt("np8/basis_f-mp-Lambda"+str(Lambda)+".txt")
    b_np10 = irlib.loadtxt("np10/basis_f-mp-Lambda"+str(Lambda)+".txt")

    print("dim = ", b_np8.dim(), b_np10.dim())

    plt.figure(4)
    #nvec = numpy.arange(100)
    nvec = numpy.array([0,10,10**2,400, 600, 800, 10**3, 2000, 4000, 6000, 10**4,10**5,10**6])
    #nvec = numpy.array([10,100])
    Tnl_np8 = b_np8.compute_Tnl(nvec)
    Tnl_np10 = b_np10.compute_Tnl(nvec)
    for l in [b_np8.dim()-1]:
        plt.plot(nvec, numpy.abs(Tnl_np8[:,l]), marker='x', linestyle='-', color='r', label='l='+str(l))
        plt.plot(nvec, numpy.abs(Tnl_np10[:,l]), marker='', linestyle='--', color='b', label='l='+str(l))
        plt.plot(nvec, numpy.abs(Tnl_np10[:,l]-Tnl_np8[:,l]), marker='x', linestyle='-', color='r', label='abs error')
        plt.plot(nvec, numpy.abs(Tnl_np10[:,l]-Tnl_np8[:,l])/numpy.abs(Tnl_np8[:,l]), marker='o', linestyle='-', color='r', label='relative error')
        print(Tnl_np10[:,-1])
        print(Tnl_np8[:,-1])
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
