from __future__ import print_function

import numpy
import scipy.integrate as integrate
import irlib.basis
import matplotlib.pyplot as plt

beta = 100.0

## There are delta peaks at 5 and -5. Particle-hole symmetric case.
gap = 4.0
GE = lambda x,E : -numpy.exp(-(x+1)*0.5*beta*E)/(1+numpy.exp(-beta*E))
GiwE = lambda n,E : 1/(1J*(2*n+1)*numpy.pi/beta - E)

# Compute imaginary-time GF for a given x (tau = (x+1)*0.5*beta)
Gx = lambda x : GE(x,0.5*gap) + GE(x,-0.5*gap)

# Compute GF for a given index of positive Matsubara freq.
Giw = lambda n :  GiwE(n, 0.5*gap) + GiwE(n, -0.5*gap)

# tau points on which G(tau) will be reconstructed
N = 200
xs = numpy.linspace(-1, 1, N)
Gtau_exact = Gx(xs)

# Number of Matsubara freq for which G(iomega_n) will be reconstructed
niw = 100

ni_reconstruct = 15

## Construct basis
idx = 0
markers = ['o', 's', 'x', '+']
for Lambda in [0.0, 100.0, 1000.0, 10000.0]:
    max_dim = 50
    b = irlib.basis.basis_f(Lambda, max_dim)

    ## Compute expansion coefficients of GF in terms of the ir basis by means of numerical integration over x
    Gl = numpy.zeros((b.dim(),), dtype=float)
    for l in range(b.dim()):
        Gl[l] = (beta/numpy.sqrt(2.0))*integrate.quad(lambda x: Gx(x)*b.value(x,l), -1.0, 1.0, epsabs=1e-6, limit=400)[0]

    plt.figure(1)
    plt.semilogy(numpy.abs(Gl), marker=markers[idx], linestyle='', label='Lambda='+str(Lambda))

    # Reconstruct G(tau) from the first ni_reconstruct data points
    Gtau_ir = numpy.zeros((N,), dtype=float)
    for i in range(N):
        x = xs[i]
        Gtau_ir[i] = numpy.dot(Gl[:ni_reconstruct], b.values(x)[:ni_reconstruct])*numpy.sqrt(2.0)/beta

    plt.figure(2)
    plt.plot((xs+1)*0.5*beta, Gtau_ir, marker=markers[idx], linestyle='', label='Lambda='+str(Lambda))

    plt.figure(3)
    plt.semilogy((xs+1)*0.5*beta, numpy.abs(Gtau_ir-Gtau_exact), marker=markers[idx], linestyle='-', label='Lambda='+str(Lambda))

    # Reconstruct G(iomega_n) for positive Matsubara freq. from the first ni_reconstruct data points
    Tnl = b.compute_Tnl(numpy.arange(niw))

    plt.figure(4)
    plt.plot(numpy.dot(Tnl[:,:ni_reconstruct], Gl[:ni_reconstruct]).imag, marker=markers[idx], linestyle='', label='Lambda='+str(Lambda))

    idx += 1

plt.figure(1)
plt.ylim([1e-5,100])
plt.xlabel('i')
plt.ylabel('Gl')
plt.legend()
plt.tight_layout()
plt.savefig('Gl.pdf')

plt.figure(2)
plt.xlim([ 0, 20])
plt.ylim([-1, 0])
plt.plot((xs+1)*0.5*beta, Gx(xs), marker='', linestyle='-', label='Exact')
plt.xlabel('tau')
plt.ylabel('G(tau)')
plt.legend()
plt.tight_layout()
plt.savefig('Gtau.pdf')

plt.figure(3)
plt.xlim([ 0, 20])
plt.xlabel('tau')
plt.ylabel('Delta G(tau)')
plt.legend()
plt.tight_layout()
plt.savefig('Gtau-diff.pdf')

plt.figure(4)
plt.xlabel('n')
plt.plot(Giw(numpy.arange(niw)).imag, marker='', linestyle='-', label='Exact')
plt.ylabel('G(iomega n)')
plt.legend()
plt.tight_layout()
plt.savefig('Giw.pdf')
