from __future__ import print_function

import numpy
import irlib

# Compute basis functions very accurately

max_dim = 1000
cutoff = 1e-8
r_tol = 1e-8
verbose = True
prec = 64
n_local_poly = 10
n_gl_node = 24

## Construct basis
statis = 'F'
for Lambda in [10.0]:
    print("Computing basis functions... It may take some time")
    if statis == 'F':
        b = irlib.compute_basis(irlib.FERMIONIC, Lambda, max_dim, cutoff, "mp", r_tol, prec, n_local_poly, n_gl_node, verbose)
        irlib.savetxt("basis_f-mp-Lambda"+str(Lambda)+".txt", b)
    else:
        b = irlib.compute_basis(irlib.BOSONIC, Lambda, max_dim, cutoff, "mp", r_tol, prec, n_local_poly, n_gl_node, verbose)
        irlib.savetxt("basis_b-mp-Lambda"+str(Lambda)+".txt", b)
    print("Done!")
