from __future__ import print_function

import numpy
import irlib
import scipy.integrate as integrate
import matplotlib.pyplot as plt

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
    b = irlib.loadtxt("np10/basis_f-mp-Lambda"+str(Lambda)+".txt")

    plt.figure(1)
    ns = b.num_sections_ulx()
    n_local_poly = b.num_local_poly_ulx()
    for l in [b.dim()-1]:
        coeff = numpy.zeros((ns, n_local_poly), dtype=float)
        for s in range(ns):
            for p in range(n_local_poly):
                coeff[s,p] = b.coeff_ulx(l,s,p)

        plt.plot(numpy.abs(coeff.reshape((ns*n_local_poly))), marker='+', linestyle='-', color='r')

    plt.figure(2)
    ns = b.num_sections_vly()
    n_local_poly = b.num_local_poly_vly()
    for l in [b.dim()-1]:
        coeff = numpy.zeros((ns, n_local_poly), dtype=float)
        for s in range(ns):
            for p in range(n_local_poly):
                coeff[s,p] = b.coeff_vly(l,s,p)

        plt.plot(numpy.abs(coeff.reshape((ns*n_local_poly))), marker='+', linestyle='-', color='r')

    idx += 1

plt.figure(1)
plt.xlabel('index')
plt.ylabel('Coeff u')
plt.legend()
plt.tight_layout()
plt.yscale("log")
plt.savefig('coeff_ulx.pdf')

plt.figure(2)
plt.xlabel('index')
plt.ylabel('Coeff v')
plt.legend()
plt.tight_layout()
plt.yscale("log")
plt.savefig('coeff_vly.pdf')
