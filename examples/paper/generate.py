from __future__ import print_function

import numpy
import irlib

# This is an example of next generation interface (experimental!)

max_dim = 1000
cutoff = 1e-30
a_tol = 1e-5
verbose = True
prec = 64

## Construct basis
for Lambda in [10.0, 100.0, 1000.0, 10000.0, 100000.0]:
    print("Computing basis functions... It may take some time")
    b = irlib.compute_basis(irlib.FERMIONIC, Lambda, max_dim, cutoff, "mp", a_tol, prec, verbose)
    print("Done!")

    irlib.savetxt("basis_f-mp-Lambda"+str(Lambda)+".txt", b)
