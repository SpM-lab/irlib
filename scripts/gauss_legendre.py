import sys
from mpmath import *
from mpmath.calculus.quadrature import GaussLegendre

dps = 300

mp.dps = dps
prec = int(dps * 3.33333)
mp.pretty = False

print("""
#pragma once

#include <vector>
#include <string>
#include <utility>

namespace irlib {
namespace detail {

template<typename S>
inline S stoscalar(const std::string& s);

template<>
inline double stoscalar<double>(const std::string& s) {
   return std::stod(s);
}

template<>
inline long double stoscalar<long double>(const std::string& s) {
   return std::stold(s);
}

template<>
inline mpfr::mpreal stoscalar<mpfr::mpreal>(const std::string& s) {
   return mpfr::mpreal(s);
}

template<typename T>
std::vector<std::pair<T,T>>
gauss_legendre_nodes(int num_nodes) {
""")

#Note: mpmath gives wrong results for degree==1! 
for degree in range(2,8):
    g = GaussLegendre(mp)
    gl = g.get_nodes(-1, 1, degree=degree, prec=prec)



    N = 3*2**(degree-1)

    print("""
    if (num_nodes == %d) {
        std::vector<std::pair<T,T>> nodes(%d);
"""%(N,N))

    for i in range(len(gl)):
        print(
"        nodes[%d] = std::make_pair<T>(stoscalar<T>(\"%s\"), stoscalar<T>(\"%s\"));"%(i, gl[i][0], gl[i][1])
);

    print("""
        return nodes;
    }
""")

print("""
    throw std::runtime_error("Invalid num_nodes passed to gauss_legendre_nodes");
}

}
}
""")
