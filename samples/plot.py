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
    #print("Loading basis functions...")
    b = irlib.loadtxt("np8/basis_f-mp-Lambda"+str(Lambda)+".txt")

    print("dim = ", b.dim())
    print("ns = ", b.num_sections_ulx())

    plt.figure(1)
    for l in [b.dim()-1]:
        plt.plot(xvec, numpy.array([b.ulx(l,x) for x in xvec]), marker='', linestyle='-', color='r')

    plt.figure(2)
    for l in [0, 4, 6]:
        plt.plot(xvec, numpy.array([b.vly(l,x) for x in xvec])/b.vly(l,1), marker='', linestyle='-', color=colors[idx], label='l='+str(l))

    plt.figure(3)
    plt.plot([b.sl(l)/b.sl(0) for l in range(b.dim())], marker='+', linestyle='-', color=colors[idx])

    plt.figure(4)
    #nvec = numpy.array([0, 1, 10, 20,30,40,50,60,70,80,90,10**2, 10**3, 10**4, 10**5, 10**6, 10**7, 10**8])
    nvec = numpy.arange(100)
    Tnl = b.compute_Tnl(nvec)
    for l in [b.dim()-1]:
        #plt.plot(nvec, numpy.abs(Tnl[:,l]), marker='x', linestyle='-', color='r', label='l='+str(l))
        print(Tnl[:,l])
        Tnl_new = numpy.array([b.compute_Tnl_safe(int(n), l) for n in nvec])
        print(Tnl_new)
        plt.plot(nvec, Tnl_new.imag, marker='x', linestyle='-', color='r', label='l='+str(l))
        plt.plot(nvec, Tnl[:,l].imag, marker='o', linestyle='-', color='r', label='l='+str(l))

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
plt.ylabel('$v_l(y)$')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('vly.pdf')

plt.figure(3)
plt.xlabel('$l$')
plt.ylabel('$s_l/s_0$')
plt.legend()
plt.yscale("log")
plt.tight_layout()
plt.savefig('sl.pdf')

plt.figure(4)
plt.xlabel('$n$')
plt.ylabel('$T_{nl}$')
plt.legend()
plt.xscale("log")
plt.ylim([-0.2,0.2])
#plt.yscale("log")
plt.tight_layout()
plt.savefig('Tnl.pdf')
