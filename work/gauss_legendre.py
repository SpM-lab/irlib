import sys
from mpmath import *
from mpmath.calculus.quadrature import GaussLegendre

dps = 50

mp.dps = dps
prec = int(dps * 3.33333)
mp.pretty = False

print("""
#include <vector>
#include <utility>

namespace {

template<typename T>
std::vector<std::pair<T,T>>
gauss_legendre_nodes(int degree) {
""")

#Note: mpmath gives wrong results for degree==1! 
for degree in range(2,7):
    g = GaussLegendre(mp)
    gl = g.get_nodes(-1, 1, degree=degree, prec=prec)



    N = 3*2**(degree-1)

    print("""
    if (degree == %d) {
        std::vector<std::pair<T,T>> nodes(%d);
"""%(degree,N))

    for i in range(len(gl)):
        print(
"        nodes[%d] = std::make_pair<T>(T(\"%s\"), T(\"%s\"));"%(i, gl[i][0], gl[i][1])
);

    print("""
        return nodes;
    }
""")

print("""
    throw std::runtime_error("Invalid degree passed to gauss_legendre_nodes");
}

}
""")
