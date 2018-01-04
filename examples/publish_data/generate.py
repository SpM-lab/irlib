from __future__ import print_function

import numpy
import irlib

# Compute basis functions very accurately

max_dim = 1000
cutoff = 1e-8
a_tol = 1e-6
verbose = True
prec = 64

## Construct basis
for Lambda in [100.0, 1000.0, 10000.0]:
    print("Computing basis functions... It may take some time")
    b = irlib.compute_basis(irlib.FERMIONIC, Lambda, max_dim, cutoff, "mp", a_tol, prec, verbose)
    print("Done!")

    irlib.savetxt("basis_f-mp-Lambda"+str(Lambda)+".txt", b)
